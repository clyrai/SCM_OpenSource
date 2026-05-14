"""
LLM Integration for SleepAI
Routes to Ollama (local) or any OpenAI-compatible cloud API (DeepSeek, OpenAI, ...)
based on the LLM_PROVIDER environment variable.
"""
import os
import re
from typing import List, Dict, Any, Optional
import json

import ollama


class LLMExtractor:
    """
    LLM-based concept and relation extraction.

    Backends:
      - LLM_PROVIDER=ollama   (default) -> local Ollama via the `ollama` SDK
      - LLM_PROVIDER=deepseek           -> DeepSeek API via OpenAI-compatible HTTP
      - LLM_PROVIDER=openai             -> OpenAI API via OpenAI-compatible HTTP
    """

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        timeout: int = None,
        provider: str = None,
    ):
        from ..core.config import LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT, LLM_PROVIDER
        self.model = model or LLM_MODEL
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.timeout = timeout if timeout is not None else LLM_TIMEOUT
        self.provider = (provider or LLM_PROVIDER or "ollama").lower()

        # OpenAI-compatible providers: cache the client lazily so import-time
        # failure on missing keys doesn't break unrelated codepaths.
        self._openai_client = None
        if self.provider in {"deepseek", "openai"}:
            self._init_openai_compat()

    def _init_openai_compat(self) -> None:
        """Configure OpenAI-compatible client for DeepSeek or OpenAI."""
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError(
                f"openai package is required for provider={self.provider!r}; "
                f"run `venv/bin/pip install openai`. Original error: {exc}"
            )

        if self.provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            default_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        else:  # openai
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        if not api_key:
            raise RuntimeError(
                f"{self.provider.upper()}_API_KEY is not set; cannot use provider={self.provider!r}."
            )

        # If the user kept LLM_MODEL=llama3.2:latest while switching providers,
        # transparently fall back to the provider-default model.
        if self.model == "llama3.2:latest" or not self.model:
            self.model = default_model

        self._openai_client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout)

    def _chat(self, prompt: str, num_predict: int = 256) -> str:
        """Dispatch to the configured backend."""
        if self.provider in {"deepseek", "openai"}:
            return self._chat_openai_compat(prompt, num_predict)
        return self._chat_ollama(prompt, num_predict)

    def _chat_ollama(self, prompt: str, num_predict: int) -> str:
        messages = [{'role': 'user', 'content': prompt}]
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                'temperature': self.temperature,
                'timeout': self.timeout,
                'num_predict': num_predict,
                'think': False,
            },
        )
        return response.message.content.strip()

    def _chat_openai_compat(self, prompt: str, num_predict: int) -> str:
        completion = self._openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=num_predict,
        )
        choice = completion.choices[0]
        content = choice.message.content if choice.message else ""
        return (content or "").strip()

    def extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract structured concepts from text using LLM.

        Cloud providers (deepseek/openai) get a JSON-mode prompt for reliable parsing;
        Ollama keeps the legacy free-form prompt + heuristic parser.
        """
        if self.provider in {"deepseek", "openai"}:
            return self._extract_concepts_json(text)
        return self._extract_concepts_freeform(text)

    def _extract_concepts_json(self, text: str) -> List[Dict[str, Any]]:
        """Structured-output extraction for OpenAI-compatible APIs."""
        prompt = (
            "Extract structured memory concepts from the user message.\n"
            "Return a JSON object of the form {\"concepts\": [...]}.\n"
            "Each concept has fields: type, description, novelty, emotional, task_relevance.\n"
            "  - type   ∈ {person, preference, fact, event, location, object, abstract}\n"
            "  - description: short factual phrase (≤80 chars), self-contained, no quotes.\n"
            "  - novelty: 0.0–1.0 (1.0 = totally new info).\n"
            "  - emotional: −1.0–1.0 (negative = unpleasant, positive = pleasant, 0 = neutral).\n"
            "  - task_relevance: 0.0–1.0 (1.0 = highly task-relevant).\n"
            "Skip filler. Skip 'None'. Output up to 6 concepts.\n\n"
            f"User message: \"\"\"{text}\"\"\"\n\nRespond with JSON only."
        )
        try:
            completion = self._openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = (completion.choices[0].message.content or "").strip()
            if not raw:
                return []
            data = json.loads(raw)
            items = data.get("concepts") if isinstance(data, dict) else None
            if not items and isinstance(data, list):
                items = data
            if not items:
                return []
            concepts: List[Dict[str, Any]] = []
            for item in items[:6]:
                if not isinstance(item, dict):
                    continue
                ctype = str(item.get("type", "fact")).lower()
                if "name" in ctype or "person" in ctype:
                    ctype = "person"
                elif "prefer" in ctype or "hobby" in ctype:
                    ctype = "preference"
                elif "place" in ctype or "location" in ctype:
                    ctype = "location"
                elif ctype not in {"fact", "event", "object", "abstract"}:
                    ctype = "fact"
                desc = str(item.get("description") or "").strip()
                if not desc or desc.lower() == "none":
                    continue
                novelty = float(item.get("novelty", 0.5) or 0.5)
                if novelty > 1.0:
                    novelty = novelty / 10.0
                concepts.append({
                    "type": ctype,
                    "description": desc[:200],
                    "novelty": max(0.0, min(1.0, novelty)),
                    "emotional": max(-1.0, min(1.0, float(item.get("emotional", 0.0) or 0.0))),
                    "task_relevance": max(0.0, min(1.0, float(item.get("task_relevance", 0.5) or 0.5))),
                })
            return concepts
        except Exception as exc:
            print(f"[LLMExtractor] JSON extraction failed ({self.provider}): {exc}")
            return []

    def _extract_concepts_freeform(self, text: str) -> List[Dict[str, Any]]:
        """Legacy Ollama free-form prompt + heuristic parser."""
        prompt = f"""You are a memory extraction system.

RULES (apply to the input text below — do not extract content from these rules themselves):
1. Extract a brief list of: person names, preferences mentioned, locations, and facts.
2. Skip any entity that is being NEGATED in the input. If the input says "not X" / "no longer X" / "used to X, now Y" / "actually Y, not X", extract ONLY the new positive value Y. Do not extract X.
3. Only extract entities that the input explicitly mentions as currently true for the speaker.

INPUT TEXT TO EXTRACT FROM (everything between the <<< and >>> markers):
<<<
{text}
>>>

Now produce the brief list. Only include items literally present in the input text above. Do not include any items that only appeared in the rules section."""

        try:
            raw = self._chat(prompt, num_predict=256)
            print(f"  [LLM raw response]: {raw[:200] if raw else 'EMPTY'}...")

            if not raw or len(raw) < 5:
                return []

            concepts = self._parse_simple_extraction(raw, text)
            return concepts

        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return []

    def _parse_simple_extraction(self, llm_response: str, original_text: str) -> List[Dict[str, Any]]:
        """Parse LLM's text response into structured concepts"""
        concepts = []
        import re

        # Clean markdown code blocks
        raw = llm_response
        if '```' in raw:
            parts = raw.split('```')
            for part in parts:
                if part.strip().startswith('[') or part.strip().startswith('{'):
                    raw = part.strip()
                    break

        # Try to parse as JSON
        try:
            import json
            data = json.loads(raw)
            if isinstance(data, list):
                for item in data:
                    concept_type = str(item.get('type', 'fact')).lower()
                    if 'name' in concept_type or 'person' in concept_type:
                        concept_type = 'person'
                    elif 'hobby' in concept_type or 'prefer' in concept_type:
                        concept_type = 'preference'
                    elif 'location' in concept_type or 'place' in concept_type:
                        concept_type = 'location'

                    novelty = item.get('novelty', 0.5)
                    if isinstance(novelty, (int, float)) and novelty > 1:
                        novelty = novelty / 10

                    concepts.append({
                        'type': concept_type,
                        'description': str(item.get('description', '')),
                        'novelty': novelty,
                        'emotional': item.get('emotional', 0.0),
                        'task_relevance': item.get('task_relevance', 0.5)
                    })
                return concepts[:5]
        except:
            pass

        # Parse structured text format from llama3.2
        lines = raw.split('\n')

        # Track which section we're in
        section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect section headers
            lower = line.lower()
            if 'person' in lower and ('name' in lower or ':' in line):
                section = 'person'
                continue
            elif 'prefer' in lower or ('mentioned' in lower and 'name' not in lower):
                section = 'preference'
                continue
            elif 'location' in lower:
                section = 'location'
                continue
            elif 'fact' in lower:
                section = 'fact'
                continue
            elif line.endswith(':') and len(line) < 40:
                # Generic section header
                if 'none' in lower:
                    section = None
                continue

            # Skip "None" entries
            if line.lower() == 'none' or line.lower() == 'none (only a statement' or line.lower().startswith('none '):
                continue

            # Extract bullet points
            if ('+' in line or '-' in line or '*' in line or '•' in line):
                # Remove bullet markers
                clean = line
                for marker in ['+', '-', '*', '•']:
                    if clean.startswith(marker):
                        clean = clean[1:].strip()
                        break

                # Clean up descriptions
                clean = clean.strip('.,;:')
                if len(clean) < 2 or clean.lower() == 'none':
                    continue

                if section == 'person':
                    # Extract name (capitalized word)
                    name_match = re.search(r'([A-Z][a-z]+)', clean)
                    if name_match:
                        name = name_match.group(1)
                        if name.lower() not in ['none', 'null', 'empty', 'the', 'and']:
                            concepts.append({
                                'type': 'person',
                                'description': f'Person: {name}',
                                'novelty': 0.7,
                                'emotional': 0.3,
                                'task_relevance': 0.5
                            })

                elif section == 'preference':
                    # Remove "Dislike for" / "Love for" / "Enjoyment of" prefixes
                    clean = re.sub(r'^(Dislike for|Love for|Enjoyment of|Likes?|Enjoys?)\s+', '', clean, flags=re.IGNORECASE)
                    clean = re.sub(r'^(the|a|an)\s+', '', clean, flags=re.IGNORECASE)
                    if len(clean) > 2 and len(clean) < 80:
                        emot = 0.5
                        if 'dislike' in line.lower() or 'hate' in line.lower():
                            emot = -0.3
                        concepts.append({
                            'type': 'preference',
                            'description': clean,
                            'novelty': 0.4,
                            'emotional': emot,
                            'task_relevance': 0.4
                        })

                elif section == 'location':
                    loc_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', clean)
                    if loc_match:
                        loc = loc_match.group(1)
                        if loc.lower() not in ['none', 'null', 'empty']:
                            concepts.append({
                                'type': 'location',
                                'description': f'Location: {loc}',
                                'novelty': 0.5,
                                'emotional': 0.0,
                                'task_relevance': 0.4
                            })

        # Fallback if nothing found
        if not concepts:
            words = original_text.split()
            if len(words) > 3:
                concepts.append({
                    'type': 'fact',
                    'description': original_text[:80],
                    'novelty': 0.5,
                    'emotional': 0.0,
                    'task_relevance': 0.4
                })

        return concepts[:5]

    def extract_with_relations(self, text: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract concepts AND relations from text."""
        concepts_data = self.extract_concepts(text)
        concepts = []
        relations = []

        concept_id_map = {}
        for i, c in enumerate(concepts_data):
            cid = f"concept_{i}"
            concept_id_map[c['description'][:20]] = cid
            concepts.append({
                'id': cid,
                'type': c['type'],
                'description': c['description'],
                'novelty': c['novelty'],
                'emotional': c['emotional'],
                'task_relevance': c['task_relevance']
            })

        # Infer simple relations
        for i in range(len(concepts) - 1):
            relations.append({
                'subject_id': concept_id_map.get(concepts[i]['description'][:20], f'concept_{i}'),
                'predicate': 'related_to',
                'object_id': concept_id_map.get(concepts[i+1]['description'][:20], f'concept_{i+1}')
            })

        return concepts, relations

    def generate_dream_narrative(self, concept_descriptions: List[str], emotional_tone: str = "neutral") -> str:
        """Generate a dream narrative from concepts (for REM phase)."""
        concepts_str = ", ".join(concept_descriptions[:10])

        prompt = f"""Generate a brief, abstract dream narrative connecting: {concepts_str}

Keep it 2-3 sentences, dreamlike."""

        try:
            raw = self._chat(prompt, num_predict=100)
            return raw if raw and len(raw) > 5 else ""
        except Exception as e:
            print(f"Dream generation failed: {e}")
            return ""

    def summarize_memory_conflicts(self, concept_pairs: List[tuple]) -> List[Dict[str, str]]:
        """Given conflicting concept pairs, suggest resolutions."""
        if not concept_pairs:
            return []

        conflicts_str = ", ".join([f"{a} vs {b}" for a, b in concept_pairs[:5]])

        prompt = f"""Analyze these concepts and suggest how they might coexist: {conflicts_str}"""

        try:
            raw = self._chat(prompt, num_predict=256)
            if raw and len(raw) > 10:
                return [{'concept1': str(a), 'concept2': str(b), 'resolution': raw[:100]} for a, b in concept_pairs[:3]]
            return []
        except Exception as e:
            print(f"Conflict resolution failed: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """Check if LLM is available and responsive"""
        import time
        try:
            start = time.time()
            self._chat("OK", num_predict=10)
            latency = time.time() - start
            return {
                'available': True,
                'provider': self.provider,
                'model': self.model,
                'latency_ms': round(latency * 1000, 1),
                'status': 'healthy',
            }
        except Exception as e:
            return {
                'available': False,
                'provider': self.provider,
                'model': self.model,
                'error': str(e),
                'status': 'unhealthy',
            }