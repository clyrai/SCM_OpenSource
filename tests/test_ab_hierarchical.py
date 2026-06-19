"""
A/B test: Hierarchical vs Flat extraction using real SCM MeaningEncoder.
Uses DeepSeek cloud LLM.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.encoder import MeaningEncoder
from src.llm import LLMExtractor

MULTI_TOPIC = """Hey, so a few things happened today. First, I got promoted to senior engineer 
at work — they're moving me to the platform team starting next month. I'm pretty excited about it 
but also nervous since I'll be leading a team of five for the first time. Also, my doctor called 
with my blood test results — everything looks normal except cholesterol is slightly elevated, so 
she wants me to cut down on red meat and start exercising more. On a different note, I finally 
booked tickets for the Japan trip in October — flying into Tokyo on the 5th, spending three days 
in Kyoto, and coming back on the 15th. Mara is coming with me. Oh and I almost forgot — the 
neighborhood committee meeting is this Saturday at 4pm, they want to discuss the new park proposal."""

LONG_NARRATIVE = """Let me tell you about my week. Monday was hectic — I had back-to-back meetings 
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


def test_mode(name, text, hierarchical):
    os.environ["HIERARCHICAL_EXTRACTION"] = "true" if hierarchical else "false"
    # Reload config
    import importlib
    import src.core.config as cfg
    importlib.reload(cfg)

    llm = LLMExtractor(provider="deepseek", model="deepseek-chat")
    encoder = MeaningEncoder(llm=llm)

    t0 = time.time()
    concepts = encoder.extract(text)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Mode: {'HIERARCHICAL' if hierarchical else 'FLAT'}")
    print(f"Concepts: {len(concepts)}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Types: {set(c.type for c in concepts)}")
    for c in concepts:
        print(f"  [{str(c.type):10s}] {c.description[:70]}")
    return concepts, elapsed


def main():
    print("A/B Test: Hierarchical vs Flat (DeepSeek)")
    print("=" * 60)

    # Multi-topic
    flat_c, flat_t = test_mode("MULTI-TOPIC INPUT", MULTI_TOPIC, hierarchical=False)
    hier_c, hier_t = test_mode("MULTI-TOPIC INPUT", MULTI_TOPIC, hierarchical=True)

    print(f"\n{'='*60}")
    print(f"MULTI-TOPIC COMPARISON")
    print(f"{'='*60}")
    print(f"Flat:         {len(flat_c)} concepts in {flat_t:.2f}s")
    print(f"Hierarchical: {len(hier_c)} concepts in {hier_t:.2f}s")
    print(f"Improvement:  {len(hier_c)/max(len(flat_c),1):.1f}x more concepts")
    print(f"Time cost:    +{hier_t - flat_t:.2f}s ({(hier_t/flat_t - 1)*100:+.0f}%)")

    # Long narrative
    flat_c2, flat_t2 = test_mode("LONG NARRATIVE", LONG_NARRATIVE, hierarchical=False)
    hier_c2, hier_t2 = test_mode("LONG NARRATIVE", LONG_NARRATIVE, hierarchical=True)

    print(f"\n{'='*60}")
    print(f"LONG NARRATIVE COMPARISON")
    print(f"{'='*60}")
    print(f"Flat:         {len(flat_c2)} concepts in {flat_t2:.2f}s")
    print(f"Hierarchical: {len(hier_c2)} concepts in {hier_t2:.2f}s")
    print(f"Improvement:  {len(hier_c2)/max(len(flat_c2),1):.1f}x more concepts")
    print(f"Time cost:    +{hier_t2 - flat_t2:.2f}s ({(hier_t2/flat_t2 - 1)*100:+.0f}%)")


if __name__ == "__main__":
    main()
