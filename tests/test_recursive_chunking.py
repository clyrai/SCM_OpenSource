"""
Test: Recursive Semantic Chunking vs Flat Extraction

Compares current SCM flat concept extraction against Zhong-inspired
recursive chunking + per-chunk extraction.

Run:  python tests/test_recursive_chunking.py
Requires: Ollama running with llama3.2:latest pulled
"""
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMExtractor


# ─── Test inputs ─────────────────────────────────────────────────────────────

SIMPLE_INPUT = "My name is Saish and I live in Bangalore."

MULTI_TOPIC_INPUT = """Hey, so a few things happened today. First, I got promoted to senior engineer 
at work — they're moving me to the platform team starting next month. I'm pretty excited about it 
but also nervous since I'll be leading a team of five for the first time. Also, my doctor called 
with my blood test results — everything looks normal except cholesterol is slightly elevated, so 
she wants me to cut down on red meat and start exercising more. On a different note, I finally 
booked tickets for the Japan trip in October — flying into Tokyo on the 5th, spending three days 
in Kyoto, and coming back on the 15th. Mara is coming with me. Oh and I almost forgot — the 
neighborhood committee meeting is this Saturday at 4pm, they want to discuss the new park proposal."""

LONG_NARRATIVE_INPUT = """Let me tell you about my week. Monday was hectic — I had back-to-back meetings 
from 9am to 5pm, including a difficult conversation with the VP about our Q3 roadmap. She wants us to 
ship the auth feature by September, which I think is aggressive but doable if we cut scope on the admin 
dashboard. Tuesday was better — I spent the morning coding the OAuth integration and finally got the 
PKCE flow working after debugging it for three days. In the afternoon I mentored Jun, our new junior 
engineer, on how to write proper PR descriptions. Wednesday I worked from home and caught up on emails. 
I also had a long call with my mom — she's worried about my brother's job situation since his startup 
laid off 30% of staff. Thursday was the team offsite at the park. We did a retrospective on the last 
sprint and played cricket afterward. I scored 42 runs which I'm proud of. Friday I left early to pick 
up Mara from the airport — she was visiting her parents in Delhi. We ordered Thai food and watched a 
documentary about deep sea creatures. Saturday morning I went for a 10k run along the lake, did it in 
52 minutes which is a personal best. In the afternoon I worked on my side project — an open-source 
memory system for AI agents inspired by how the brain consolidates memories during sleep."""


# ─── Chunking prompt ─────────────────────────────────────────────────────────

CHUNK_PROMPT_TEMPLATE = """Split the following text into at most {K} semantically coherent segments.
Each segment should cover a distinct topic or theme.

Return a JSON array of strings, where each string is one segment of the original text.
Preserve the original wording exactly — do not paraphrase or summarize.
The segments should concatenate back to the full original text.

Text:
\"\"\"{text}\"\"\"

Respond with JSON only. Example format: ["segment 1 text", "segment 2 text", "segment 3 text"]"""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def chunk_text_heuristic(text: str, K: int = 4) -> list[str]:
    """Heuristic chunking: split by sentences, group into K segments."""
    import re
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= 1:
        # Try splitting by commas or semicolons for single-sentence inputs
        sentences = re.split(r'[,;]\s+', text.strip())

    if len(sentences) <= 1:
        return [text]

    # Group sentences into K roughly equal segments
    n = len(sentences)
    seg_size = max(1, n // K)
    segments = []
    for i in range(0, n, seg_size):
        chunk = ' '.join(sentences[i:i+seg_size])
        if chunk.strip():
            segments.append(chunk.strip())

    return segments[:K]


def chunk_text(llm: LLMExtractor, text: str, K: int = 4) -> list[str]:
    """Chunk text into K semantically coherent segments."""
    prompt = CHUNK_PROMPT_TEMPLATE.format(K=K, text=text)
    try:
        raw = llm._chat(prompt, num_predict=1024)
    except Exception:
        return chunk_text_heuristic(text, K)

    # Try to parse JSON
    try:
        clean = raw
        if '```' in clean:
            parts = clean.split('```')
            for part in parts:
                stripped = part.strip()
                if stripped.startswith('[') or stripped.startswith('{'):
                    clean = stripped
                    break

        segments = json.loads(clean)
        if isinstance(segments, list) and all(isinstance(s, str) for s in segments):
            result = [s.strip() for s in segments if s.strip()]
            if len(result) > 1:
                return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to parse bullet-point list from LLM
    lines = [l.strip().lstrip('*-•').strip() for l in raw.split('\n') if l.strip()]
    # Filter to lines that look like text segments (not headers/labels)
    candidates = [l for l in lines if len(l) > 20 and not l.endswith(':')]
    if len(candidates) >= 2:
        return candidates[:K]

    # Fallback: heuristic chunking
    return chunk_text_heuristic(text, K)


def flat_extract(llm: LLMExtractor, text: str) -> list[dict]:
    """Current SCM behavior: flat extraction from full text."""
    return llm.extract_concepts(text)


def hierarchical_extract(llm: LLMExtractor, text: str, K: int = 4) -> list[dict]:
    """Proposed: chunk first, then extract from each chunk."""
    segments = chunk_text(llm, text, K)
    all_concepts = []
    seen_descriptions = set()

    for i, segment in enumerate(segments):
        concepts = llm.extract_concepts(segment)
        for c in concepts:
            # Deduplicate by description prefix
            desc_key = c['description'][:40].lower()
            if desc_key not in seen_descriptions:
                seen_descriptions.add(desc_key)
                c['source_segment'] = i
                all_concepts.append(c)

    return all_concepts, segments


def compute_entropy(text: str) -> float:
    """Simple character-level entropy estimate (Shannon-style)."""
    from collections import Counter
    import math
    if not text:
        return 0.0
    counts = Counter(text.lower())
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return entropy


# ─── Main test ───────────────────────────────────────────────────────────────

def run_test(name: str, text: str, K: int = 4, model: str = "deepseek-chat", provider: str = "deepseek"):
    """Run flat vs hierarchical extraction on a single input."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    print(f"Input length: {len(text)} chars, {len(text.split())} words")
    print(f"K (branching factor): {K}")
    print(f"Provider: {provider} | Model: {model}")
    print()

    llm = LLMExtractor(model=model, provider=provider)

    # ── Flat extraction (current) ──
    t0 = time.time()
    flat_concepts = flat_extract(llm, text)
    flat_time = time.time() - t0

    print(f"── FLAT EXTRACTION (current SCM) ──")
    print(f"  Concepts extracted: {len(flat_concepts)}")
    print(f"  Time: {flat_time:.2f}s")
    for c in flat_concepts:
        print(f"  [{c['type']:10s}] {c['description'][:70]}")
    print()

    # ── Hierarchical extraction (proposed) ──
    t0 = time.time()
    hier_concepts, segments = hierarchical_extract(llm, text, K)
    hier_time = time.time() - t0

    print(f"── HIERARCHICAL EXTRACTION (Zhong-inspired) ──")
    print(f"  Segments created: {len(segments)}")
    print(f"  Concepts extracted: {len(hier_concepts)}")
    print(f"  Time: {hier_time:.2f}s")
    for i, seg in enumerate(segments):
        print(f"  Segment {i}: {seg[:60]}...")
    print()
    for c in hier_concepts:
        seg_idx = c.get('source_segment', '?')
        print(f"  [seg {seg_idx}] [{c['type']:10s}] {c['description'][:70]}")
    print()

    # ── Comparison ──
    flat_types = set(c['type'] for c in flat_concepts)
    hier_types = set(c['type'] for c in hier_concepts)

    flat_descs = set(c['description'][:40].lower() for c in flat_concepts)
    hier_descs = set(c['description'][:40].lower() for c in hier_concepts)

    unique_to_flat = flat_descs - hier_descs
    unique_to_hier = hier_descs - flat_descs
    overlap = flat_descs & hier_descs

    entropy = compute_entropy(text)

    print(f"── COMPARISON ──")
    print(f"  Flat concepts:     {len(flat_concepts)}")
    print(f"  Hierarchical:      {len(hier_concepts)}")
    print(f"  Overlap:           {len(overlap)}")
    print(f"  Unique to flat:    {len(unique_to_flat)}")
    print(f"  Unique to hier:    {len(unique_to_hier)}")
    print(f"  Flat types:        {flat_types}")
    print(f"  Hier types:        {hier_types}")
    print(f"  Text entropy:      {entropy:.3f} bits/char")
    print(f"  Flat time:         {flat_time:.2f}s")
    print(f"  Hier time:         {hier_time:.2f}s")
    print(f"  Time overhead:     {hier_time - flat_time:+.2f}s ({(hier_time/flat_time - 1)*100:+.0f}%)")

    if unique_to_hier:
        print(f"\n  CONCEPTS ONLY HIERARCHICAL FOUND:")
        for d in unique_to_hier:
            print(f"    + {d}")

    if unique_to_flat:
        print(f"\n  CONCEPTS ONLY FLAT FOUND:")
        for d in unique_to_flat:
            print(f"    - {d}")

    return {
        "name": name,
        "flat_count": len(flat_concepts),
        "hier_count": len(hier_concepts),
        "segments": len(segments),
        "overlap": len(overlap),
        "unique_to_flat": len(unique_to_flat),
        "unique_to_hier": len(unique_to_hier),
        "flat_time": flat_time,
        "hier_time": hier_time,
        "entropy": entropy,
    }


def main():
    MODEL = "deepseek-chat"
    PROVIDER = "deepseek"
    print("Recursive Semantic Chunking vs Flat Extraction")
    print("=" * 70)
    print("This test compares current SCM flat extraction against")
    print("Zhong-inspired recursive chunking + per-chunk extraction.")
    print(f"Provider: {PROVIDER} | Model: {MODEL}")
    print()

    results = []

    # Test 1: Simple input (should be ~same)
    results.append(run_test("Simple Input", SIMPLE_INPUT, K=2, model=MODEL, provider=PROVIDER))

    # Test 2: Multi-topic input (hierarchical should win)
    results.append(run_test("Multi-Topic Input", MULTI_TOPIC_INPUT, K=4, model=MODEL, provider=PROVIDER))

    # Test 3: Long narrative (hierarchical should find more)
    results.append(run_test("Long Narrative", LONG_NARRATIVE_INPUT, K=4, model=MODEL, provider=PROVIDER))

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY — {PROVIDER}/{MODEL}")
    print(f"{'='*70}")
    print(f"{'Test':<25} {'Flat':>5} {'Hier':>5} {'Segs':>5} {'Overlap':>8} {'+Hier':>6} {'+Flat':>6} {'Entropy':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<25} {r['flat_count']:>5} {r['hier_count']:>5} {r['segments']:>5} {r['overlap']:>8} {r['unique_to_hier']:>6} {r['unique_to_flat']:>6} {r['entropy']:>8.3f}")

    print()
    print("If hierarchical consistently finds MORE unique concepts,")
    print("the chunking approach is validated for implementation.")


if __name__ == "__main__":
    main()
