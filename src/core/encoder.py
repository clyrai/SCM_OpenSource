"""
MeaningEncoder: Extract semantic concepts from text using LLM.

Phase 6 adds HybridEncoder: a triage encoder that uses cheap heuristic
extraction by default and escalates to a full LLM call only on turns whose
salience proxy passes a configurable threshold. This cuts cloud-LLM cost
~3-5x on conversational data without sacrificing quality on durable facts.

Phase 7 adds provider-embedding backends. The default sentence-transformer
all-MiniLM-L6-v2 is small and fast but mediocre at bridging question-form
queries to declarative-form facts (e.g., "Where do I work?" → "I'm at Atlas
Labs"). Provider backends let SCM use higher-quality embedding models from
Ollama (free, local) or any OpenAI-compatible embedding API (paid).

Selection is via the SCM_EMBEDDING_BACKEND environment variable:
    sentence_transformers  — default, local, MiniLM-L6-v2 (384-dim).
    ollama                 — local Ollama server, default model nomic-embed-text (768-dim).
    openai_compat          — any OpenAI-compatible /v1/embeddings endpoint.
    hash                   — deterministic offline fallback.
"""
import json
import os
from typing import List, Dict, Any, Optional
import re
import math
from collections import Counter

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import requests as _requests
except Exception:
    _requests = None

from .models import Concept, Relation, PredicateType, ConceptType, ImportanceVector
from .config import EMBEDDING_MODEL, EMBEDDING_DIM


class HashEmbeddingModel:
    """
    Offline-safe embedding fallback.

    Produces deterministic sparse vectors from token hashes.
    """

    def __init__(self, dim: int):
        self.dim = max(8, dim)

    def encode(self, text: str):
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        counts = Counter(tokens)
        vector = [0.0] * self.dim

        if not counts:
            vector[0] = 1.0
            return vector

        for token, count in counts.items():
            idx = hash(token) % self.dim
            vector[idx] += float(count)

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector


class OllamaEmbeddingModel:
    """
    Embedding backend that calls a local Ollama server.

    Default model is `nomic-embed-text` (768-dim), which gives a meaningful
    quality jump over sentence-transformers/all-MiniLM-L6-v2 (384-dim) on
    question→fact bridging tasks — but only when called with retrieval
    task prefixes. Models like nomic-embed-text and bge are instruction-
    tuned and produce dramatically different vectors when given a task
    hint vs raw text.

    The model name → prefix mapping is in `MODEL_PREFIXES` below. A model
    not in the table is encoded with no prefix, which is the right
    default for prefix-less models like text-embedding-3.

    Network calls are local (default http://localhost:11434) so latency is
    typically <50ms per encode. The class falls back to a hash embedding
    silently if Ollama is unreachable, so a missing daemon doesn't crash
    the whole pipeline.
    """

    # model name → (query_prefix, document_prefix). Keys are matched by
    # substring, lowercased. None entries mean "use raw text."
    MODEL_PREFIXES = {
        "nomic-embed-text": ("search_query: ", "search_document: "),
        "bge-large": (
            "Represent this sentence for searching relevant passages: ",
            "",
        ),
        "bge-base": (
            "Represent this sentence for searching relevant passages: ",
            "",
        ),
        "mxbai-embed-large": (
            "Represent this sentence for searching relevant passages: ",
            "",
        ),
    }

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        fallback_dim: int = 768,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._fallback = HashEmbeddingModel(fallback_dim)
        self._dim: Optional[int] = None
        self._failed_count = 0

    def _prefix(self, mode: str) -> str:
        """Return the appropriate task prefix for this model and mode.

        mode: 'query' (text is a search query) or 'document' (text is to
        be stored / retrieved against). Falls back to no prefix if the
        model isn't in the prefix table.
        """
        m = self.model.lower()
        for key, (q_pref, d_pref) in self.MODEL_PREFIXES.items():
            if key in m:
                return q_pref if mode == "query" else d_pref
        return ""

    def encode(self, text: str, mode: str = "document") -> List[float]:
        """Encode text. Pass mode='query' for retrieval queries.

        The default mode='document' is correct for ingest-time storage,
        which is the common case (every Concept gets its embedding via
        encode() during creation).
        """
        if _requests is None:
            return self._fallback.encode(text)
        if not text:
            return self._fallback.encode("")
        prefixed = f"{self._prefix(mode)}{text}"
        try:
            r = _requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": prefixed},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            # Ollama may return either {"embeddings": [[...]]} or {"embedding": [...]}
            embeddings = data.get("embeddings")
            if embeddings and isinstance(embeddings, list) and embeddings:
                vec = embeddings[0]
            else:
                vec = data.get("embedding")
            if not vec:
                return self._fallback.encode(text)
            if self._dim is None:
                self._dim = len(vec)
            return list(vec)
        except Exception:
            self._failed_count += 1
            if self._failed_count == 1:
                # Only print once to avoid log spam.
                print(
                    f"[OllamaEmbeddingModel] Falling back to hash; "
                    f"Ollama at {self.base_url} unreachable or model {self.model!r} unavailable"
                )
            return self._fallback.encode(text)


class OpenAICompatibleEmbeddingModel:
    """
    Embedding backend for any OpenAI-compatible /v1/embeddings endpoint.

    Works with OpenAI itself, Voyage, Together, Anyscale, and other
    providers that mirror the OpenAI embedding response shape:
        POST /v1/embeddings  {"model": "...", "input": "..."}
        → {"data": [{"embedding": [...]}], ...}

    API key + base URL come from environment variables so credentials never
    appear in code.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        fallback_dim: int = 1536,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._fallback = HashEmbeddingModel(fallback_dim)
        self._dim: Optional[int] = None
        self._failed_count = 0

    def encode(self, text: str) -> List[float]:
        if _requests is None or not self.api_key:
            return self._fallback.encode(text)
        if not text:
            return self._fallback.encode("")
        try:
            r = _requests.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": text},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            entries = data.get("data") or []
            if not entries or "embedding" not in entries[0]:
                return self._fallback.encode(text)
            vec = entries[0]["embedding"]
            if self._dim is None:
                self._dim = len(vec)
            return list(vec)
        except Exception:
            self._failed_count += 1
            if self._failed_count == 1:
                print(
                    f"[OpenAICompatibleEmbeddingModel] Falling back to hash; "
                    f"endpoint {self.base_url}/embeddings unreachable or rejected"
                )
            return self._fallback.encode(text)


# ─── Singleton caches ──────────────────────────────────────────────────────
#
# Embedding models are HEAVY (sentence-transformers MiniLM is ~80-100 MB
# resident; nomic-embed-text-class clients hold a requests.Session). In a
# multi-user MCP / API server, every per-user ChatEngine used to construct
# its own MeaningEncoder which loaded its own model copy. With 10 users
# that's 1 GB of duplicated weights for no gain.
#
# These caches make every backend a process-wide singleton: built on first
# use, shared across all ChatEngine instances thereafter.

import threading as _threading

_singleton_lock = _threading.Lock()
_sentence_transformer_singleton = None
_ollama_singletons: Dict[str, Any] = {}
_openai_singletons: Dict[str, Any] = {}


def _autodetect_backend() -> str:
    """Pick a sensible default backend if the user hasn't specified one.

    Order of preference:
      1. Ollama with a known embedding model already pulled (best quality
         that works out of the box on most dev machines)
      2. sentence-transformers MiniLM (always works, mediocre quality)

    The auto-detect is bounded to a 0.5s probe so missing Ollama doesn't
    add startup latency.
    """
    if _requests is None:
        return "sentence_transformers"
    base = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
    try:
        r = _requests.get(f"{base.rstrip('/')}/api/tags", timeout=0.5)
        if r.status_code == 200:
            models = [m.get("name", "") for m in (r.json().get("models") or [])]
            for preferred in ("nomic-embed-text", "mxbai-embed-large", "bge-large"):
                if any(preferred in name for name in models):
                    return "ollama"
    except Exception:
        pass
    return "sentence_transformers"


def _build_embedding_model(
    explicit_backend: Optional[str] = None,
    explicit_model: Optional[str] = None,
):
    """Factory: build the configured embedding backend.

    Selection precedence: explicit args > env vars > auto-detected default.
    """
    backend = (
        explicit_backend
        or os.environ.get("SCM_EMBEDDING_BACKEND")
        or _autodetect_backend()
    ).lower()

    if backend == "hash":
        return HashEmbeddingModel(EMBEDDING_DIM)

    if backend == "ollama":
        model = (
            explicit_model
            or os.environ.get("SCM_EMBEDDING_MODEL")
            or "nomic-embed-text"
        )
        url = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
        # Singleton per (model, base_url) — share HTTP client + dim cache.
        cache_key = f"{model}@{url}"
        with _singleton_lock:
            inst = _ollama_singletons.get(cache_key)
            if inst is None:
                inst = OllamaEmbeddingModel(model=model, base_url=url)
                _ollama_singletons[cache_key] = inst
        return inst

    if backend in ("openai", "openai_compat", "openai-compatible"):
        model = (
            explicit_model
            or os.environ.get("SCM_EMBEDDING_MODEL")
            or "text-embedding-3-small"
        )
        api_key = (
            os.environ.get("SCM_EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        base_url = (
            os.environ.get("SCM_EMBEDDING_BASE_URL")
            or "https://api.openai.com/v1"
        )
        # Singleton per (model, base_url, api_key) — share HTTP session.
        # api_key is part of the key so different tenants don't collide.
        cache_key = f"{model}@{base_url}#{(api_key or '')[-8:]}"
        with _singleton_lock:
            inst = _openai_singletons.get(cache_key)
            if inst is None:
                inst = OpenAICompatibleEmbeddingModel(
                    model=model, api_key=api_key, base_url=base_url
                )
                _openai_singletons[cache_key] = inst
        return inst

    # Default: sentence_transformers (with hash fallback handled by the
    # MeaningEncoder caller via _maybe_load_sentence_transformer).
    return None  # signals "use sentence-transformer path"


class MeaningEncoder:
    """
    Converts raw text into semantic concept graph.
    Uses LLM to extract structured information from text.
    """

    def __init__(self, llm=None, embedding_backend: Optional[str] = None,
                 embedding_model_name: Optional[str] = None):
        self.llm = llm
        # If a provider backend is requested, use it directly. Otherwise the
        # legacy path (HashEmbeddingModel + lazy sentence-transformer
        # upgrade) preserves backward compatibility.
        provider = _build_embedding_model(
            explicit_backend=embedding_backend,
            explicit_model=embedding_model_name,
        )
        if provider is not None:
            self.embedding_model = provider
            self._tried_sentence_transformer = True  # skip the lazy load
            self._using_provider_embedding = True
        else:
            self.embedding_model = HashEmbeddingModel(EMBEDDING_DIM)
            self._tried_sentence_transformer = False
            self._using_provider_embedding = False

    def _maybe_load_sentence_transformer(self):
        """
        Try once to load local sentence-transformer weights.
        Falls back to hash embeddings if unavailable.

        v0.7.3: the loaded SentenceTransformer is a process-wide
        singleton so multi-user MCP / API servers don't load N copies of
        the same ~80 MB model. Net savings on a 10-user box: ~700 MB.
        """
        if self._tried_sentence_transformer:
            return

        self._tried_sentence_transformer = True
        if SentenceTransformer is None:
            return

        global _sentence_transformer_singleton
        with _singleton_lock:
            if _sentence_transformer_singleton is None:
                try:
                    _sentence_transformer_singleton = SentenceTransformer(
                        EMBEDDING_MODEL,
                        local_files_only=True,
                    )
                except Exception as e:
                    print(
                        f"[MeaningEncoder] Using hash embeddings "
                        f"(local model unavailable): {e}"
                    )
                    return
            self.embedding_model = _sentence_transformer_singleton

    def extract(self, text: str) -> List[Concept]:
        """
        Extract concepts and relations from text.

        Returns:
            List of Concept objects extracted from text
        """
        if self.llm:
            return self._extract_with_llm(text)
        return self._extract_heuristic(text)

    def _extract_with_llm(self, text: str) -> List[Concept]:
        """
        Use LLM to extract structured concepts from text.
        """
        try:
            llm_data = self.llm.extract_concepts(text)
            concepts = []

            for item in llm_data:
                imp = ImportanceVector(
                    novelty=item.get('novelty', 0.5),
                    emotional=item.get('emotional', 0.0),
                    task_relevance=item.get('task_relevance', 0.5),
                    repetition=0.5
                )
                concept = Concept(
                    type=ConceptType(item.get('type', 'fact')),
                    description=item.get('description', ''),
                    importance=imp,
                    embedding=self._get_embedding(item.get('description', ''))
                )
                concepts.append(concept)

            return concepts if concepts else self._extract_heuristic(text)
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return self._extract_heuristic(text)

    def _extract_heuristic(self, text: str) -> List[Concept]:
        """
        Heuristic extraction for when LLM is not available.
        Simple pattern-based approach.
        """
        concepts = []

        # Extract name patterns (e.g., "My name is X", "I am X")
        name_match = re.search(r"(?:my name is|i am|i'm|I'm)\s+([A-Z][a-z]+)", text, re.IGNORECASE)
        if name_match:
            concepts.append(Concept(
                type=ConceptType.PERSON,
                description=f"Person: {name_match.group(1)}",
                importance=ImportanceVector(novelty=0.9, emotional=0.5)
            ))

        # Extract preferences (e.g., "I prefer X", "I like Y")
        pref_match = re.search(r"(?:prefer|like|hate|dislike|enjoy)\s+([^.!?]+)", text, re.IGNORECASE)
        if pref_match:
            concepts.append(Concept(
                type=ConceptType.PREFERENCE,
                description=f"Preference: {pref_match.group(1).strip()}",
                importance=ImportanceVector(task_relevance=0.8)
            ))

        # Extract facts (named entities, dates, numbers)
        fact_patterns = [
            (r"(\d+)\s+(?:years?\s+old|yo)", "Age fact"),
            (r"(?:work on|working on|building|developing)\s+([A-Z][a-zA-Z]+)", "Work interest"),
        ]

        for pattern, desc in fact_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                concepts.append(Concept(
                    type=ConceptType.FACT,
                    description=f"{desc}: {match.group(1)}",
                    importance=ImportanceVector(task_relevance=0.7)
                ))

        # If no concepts found, create a general concept
        if not concepts:
            concepts.append(Concept(
                type=ConceptType.FACT,
                description=text,
                importance=ImportanceVector()
            ))

        return concepts

    def extract_with_relations(self, text: str) -> tuple[List[Concept], List[Relation]]:
        """
        Extract concepts and relations from text.

        Returns:
            Tuple of (concepts, relations)
        """
        # Get base concepts
        concepts = self.extract(text)

        # Infer relations between concepts
        relations = self._infer_relations(concepts, text)

        return concepts, relations

    def _infer_relations(self, concepts: List[Concept], text: str) -> List[Relation]:
        """
        Infer relations between extracted concepts.
        """
        relations = []

        for i, concept in enumerate(concepts):
            # Person -> Preference relation
            if concept.type == ConceptType.PERSON and i + 1 < len(concepts):
                next_concept = concepts[i + 1]
                if next_concept.type == ConceptType.PREFERENCE:
                    relations.append(Relation(
                        subject_id=concept.id,
                        predicate=PredicateType.HAS_PROPERTY,
                        object_id=next_concept.id,
                        strength=0.9
                    ))

        return relations

    def _get_embedding(self, text: str, mode: str = "document") -> List[float]:
        """
        Get semantic embedding for text.

        mode: 'document' (default, correct for stored concepts) or 'query'
        (correct for retrieval queries — instruction-tuned models like
        nomic-embed-text and bge expect a task-specific prefix).
        """
        self._maybe_load_sentence_transformer()
        # Provider backends accept the mode arg; sentence-transformer +
        # hash backends don't have a mode concept and ignore it gracefully.
        # SentenceTransformer raises ValueError for unknown kwargs, plain
        # callables raise TypeError — catch both.
        try:
            embedding = self.embedding_model.encode(text, mode=mode)
        except (TypeError, ValueError):
            embedding = self.embedding_model.encode(text)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.
        """
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)

        # Cosine similarity
        from numpy import dot
        from numpy.linalg import norm
        similarity = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
        return float(similarity)


# ─── Phase 6: HybridEncoder ─────────────────────────────────────────────────


# All keyword lists below are loaded from linguistic_resources at instance
# init time, NOT hardcoded here. See HybridEncoder.__init__.
class HybridEncoder:
    """
    Salience-gated triage encoder.

    Strategy:
      1. Run cheap heuristic extraction (regex patterns on entities).
      2. Compute a salience proxy from the input text alone (length, named
         entities, high-signal keywords, content density).
      3. If proxy >= escalate_threshold AND an LLM-backed encoder is available,
         call the LLM and merge its concepts into the result.
      4. Otherwise return the heuristic concepts.

    This preserves zero-API-cost on filler turns and only spends LLM tokens on
    turns that are likely to carry durable facts.

    Cost reduction: ~3-5x on typical chat data (most turns are filler).
    """

    def __init__(
        self,
        llm=None,
        escalate_threshold: float = 0.45,
        always_escalate_min_words: int = 25,
        force_llm_only: bool = False,
    ):
        self.llm = llm
        self.escalate_threshold = max(0.0, min(1.0, escalate_threshold))
        self.always_escalate_min_words = max(1, always_escalate_min_words)
        self.force_llm_only = bool(force_llm_only)
        # Stats accessible to callers for cost accounting.
        self.stats = {
            "calls": 0,
            "llm_escalations": 0,
            "heuristic_only": 0,
            "concepts_emitted": 0,
        }
        self._heuristic = MeaningEncoder(llm=None)
        self._llm_encoder = MeaningEncoder(llm=llm) if llm is not None else None
        # Share embedding cache
        self.embedding_model = self._heuristic.embedding_model
        # Load filler / signal keywords from linguistic_resources (NOT hardcoded).
        # Lazy-imported to avoid circular import at module load.
        from . import linguistic_resources as lr_mod
        kw = lr_mod.get_hybrid_encoder_keywords()
        self._filler_tokens = {t.lower() for t in kw.get("filler_tokens", [])}
        self._filler_phrases = tuple(kw.get("filler_phrases", []))
        self._high_signal_keywords = tuple(kw.get("high_signal_keywords", []))

    def compute_salience_proxy(self, text: str) -> float:
        """
        Cheap text-only salience score in [0, 1].

        Higher = more likely to carry a durable fact. Used to gate the LLM call.
        """
        t = (text or "").strip()
        if not t:
            return 0.0
        lower = t.lower()
        words = re.findall(r"[a-zA-Z']+", t)
        n_words = len(words)

        # Filler-only turn → nearly zero
        if n_words <= 3:
            if all(w.lower() in self._filler_tokens for w in words):
                return 0.0
        for phrase in self._filler_phrases:
            if lower.startswith(phrase) and n_words <= 6:
                return 0.05

        # Components
        length_score = min(1.0, n_words / float(self.always_escalate_min_words))
        proper_nouns = len([w for w in words if w[:1].isupper() and len(w) > 2 and not w.isupper()])
        proper_score = min(1.0, proper_nouns / 3.0)
        digit_score = 1.0 if re.search(r"\d", t) else 0.0
        keyword_score = 0.0
        for kw in self._high_signal_keywords:
            if kw in lower:
                keyword_score = 1.0
                break

        salience = (
            0.30 * length_score
            + 0.25 * proper_score
            + 0.20 * digit_score
            + 0.25 * keyword_score
        )
        return max(0.0, min(1.0, salience))

    def extract(self, text: str) -> List[Concept]:
        """Hybrid extraction: heuristic first, escalate to LLM if salience is high."""
        self.stats["calls"] += 1
        proxy = self.compute_salience_proxy(text)
        use_llm = self.force_llm_only or (
            self._llm_encoder is not None and proxy >= self.escalate_threshold
        )

        if use_llm:
            try:
                concepts = self._llm_encoder.extract(text)
                if concepts:
                    self.stats["llm_escalations"] += 1
                    self.stats["concepts_emitted"] += len(concepts)
                    return concepts
            except Exception:
                pass

        concepts = self._heuristic.extract(text)
        self.stats["heuristic_only"] += 1
        self.stats["concepts_emitted"] += len(concepts)
        return concepts

    def extract_with_relations(self, text: str):
        concepts = self.extract(text)
        # Reuse heuristic relation inference; LLM relation inference is rarely
        # better than the heuristic for our predicate set.
        relations = self._heuristic._infer_relations(concepts, text)
        return concepts, relations

    def _get_embedding(self, text: str, mode: str = "document") -> List[float]:
        return self._heuristic._get_embedding(text, mode=mode)

    def compute_similarity(self, text1: str, text2: str) -> float:
        return self._heuristic.compute_similarity(text1, text2)

    def get_stats(self) -> Dict[str, Any]:
        d = dict(self.stats)
        total = max(1, d["calls"])
        d["llm_escalation_rate"] = round(d["llm_escalations"] / total, 4)
        d["heuristic_rate"] = round(d["heuristic_only"] / total, 4)
        return d
