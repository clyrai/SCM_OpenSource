Subject: Re: Related Prior Work on Memory Consolidation in LLMs

Hi Weishun,

Thanks for reaching out — I enjoyed reading both papers.

The Semantic Chunking paper is particularly relevant to our work. The finding that optimal branching factor K* varies with content complexity (K* ∈ [2,6]) is interesting — especially that K*=4 recovers Shannon's classic estimate. The ~80% redundancy in natural language suggests that memory systems storing raw conversational content accumulate substantial predictable material, which is exactly the problem SCM's forgetting module addresses.

I've cited the Semantic Chunking paper in our second manuscript, which evaluates SCM's memory lifecycle under stress testing. SCM implements a 7-item bounded working memory (Miller's 7±2), and we show that sleep-stage consolidation produces measurable disambiguation gains over awake-only baselines — awake-only controls remain at 0.0 disambiguation recall while sleep-enabled SCM reaches 0.9052. Your entropy findings are consistent with our observation that most stored content is noise and selective retention is necessary.

Our first paper is published if you'd like to reference it for context on the system architecture.

Would be happy to discuss potential connections further — especially whether the K-ary tree model could inform how we structure concept extraction at different scales.

Best,
Saish
