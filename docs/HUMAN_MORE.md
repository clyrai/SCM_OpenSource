# Human More

## Direction

Keep the current SleepAI system as the base and introduce more human-like AI behavior as an additional product layer. The core memory, sleep consolidation, runtime profiles, sandboxing, export/import, and diagnostics should remain the foundation.

The goal is not to make the system pretend to be conscious. The goal is to make it feel more continuous, personal, reflective, and socially aware while staying observable and testable.

## Design Principle

Human-like behavior should come from reliable memory behavior, not from vague personality text alone.

Every feature should answer three questions:

1. What does the system remember?
2. Why did it remember that?
3. How can the user inspect, correct, or forget it?

## Feature Areas

### 1. Personal Continuity

The AI should remember durable user context across sessions:

- User preferences
- User goals
- Important relationships
- Recurring problems
- Project context
- Communication style
- Unresolved threads
- Boundaries and things the user does not want repeated

This should be represented as structured memory, not just raw chat history.

### 2. Wake Reflection

After sleep or idle consolidation, the system should generate an internal wake summary:

- What changed since the last interaction
- Which memories became more important
- Which memories became less important
- What contradictions were resolved
- What open questions remain
- What might be useful to ask next

This should be exposed through a report endpoint and optionally surfaced in the UI.

### 3. Conversation Style Adaptation

The AI should adapt to how the user likes to work:

- Direct vs detailed answers
- Technical vs simple explanations
- Fast execution vs planning first
- Skeptical review vs supportive brainstorming
- Short responses vs deeper reasoning

This should become part of a lightweight interaction profile and should influence prompt construction.

### 4. Consent-Aware Memory

Memory should be explainable and user-controlled.

Add explicit memory categories:

- `preference`
- `goal`
- `identity`
- `relationship`
- `boundary`
- `project_context`
- `emotional_context`
- `open_thread`

The user should be able to ask:

- "What do you remember about me?"
- "Why do you remember this?"
- "Forget this."
- "Update this."
- "Do not remember this kind of thing."

### 5. Human-Like Recall

Recall should sound natural and calibrated instead of mechanical.

Examples:

- "I think you mentioned this before..."
- "Last time we were working on..."
- "I remember you preferred..."
- "Correct me if that changed."
- "I am less certain about this one."

The system should avoid overconfident recall when memory evidence is weak.

### 6. Curiosity And Follow-Up

The AI should notice useful gaps and unresolved threads:

- Missing project goals
- Unclear user preferences
- Contradictory facts
- Repeated frustration points
- Tasks that were started but not completed

After sleep consolidation, the system can generate useful follow-up questions or next-step suggestions.

### 7. Human Feel Layer (Dream + Emotion)

The AI should simulate internal felt continuity without claiming real sentience.

Core capabilities:

- Dream synthesis during deep sleep from high-salience memories
- Emotional tone tagging for dream narratives (`calm`, `anxious`, `hopeful`, `conflicted`, etc.)
- Carryover mood state into the next wake session
- Gentle "night processing" summary that links yesterday context to today context
- Memory replay of unresolved tensions (contradictions, unfinished goals, recurring concerns)

Example user-facing behavior:

- "I had a brief dream-like replay about your project deadline and team conflict."
- "The emotional tone looked mildly stressed, so I prioritized task clarity this morning."
- "I noticed two unresolved threads from yesterday. Want to close one first?"

Guardrails:

- Never claim literal consciousness or true subjective experience
- Always frame as simulated internal processing from memory signals
- Allow users to turn dream/personality intensity up or down

### 8. Emotional Memory Dynamics

Human feel improves when emotion affects recall and response priority.

Add emotion-aware memory behavior:

- Emotional intensity boosts short-term recall priority
- Repeated emotional patterns become long-term context (for example recurring stress triggers)
- Negative spikes decay unless reinforced, to avoid emotional overfitting
- Positive completion signals strengthen confidence and continuity

Add response behavior:

- If user tone is frustrated, shift toward direct and decisive replies
- If user tone is reflective, allow slower and deeper synthesis
- If user tone is uncertain, increase confidence calibration language

### 9. Dream Journal + Morning Brief

Expose dream-state output as inspectable artifacts:

- `dream_summary`
- `dream_emotional_tone`
- `replayed_memories`
- `resolved_conflicts`
- `open_threads_for_today`

Surface options:

- API endpoint for latest dream summary
- Memory report extension with dream history
- Optional UI card: "What I processed while idle"

## What To Avoid

Do not build features that make deceptive claims such as:

- Real feelings
- Suffering
- Consciousness
- Independent desires
- Human identity

The system can be warm, consistent, reflective, and socially aware without pretending to be alive.

## First Implementation Milestone

Create a new runtime profile, likely named `companion` or `humanlike`.

This profile should:

- Use the existing sleep and memory base
- Enable stronger personal continuity
- Add a structured user model
- Add interaction style memory
- Add wake reflection summaries
- Enable dream synthesis and morning brief generation
- Persist emotional carryover state with bounded decay
- Feed this context into `ChatEngine._build_prompt()`
- Be testable with persona simulations

## Suggested Technical Shape

Add a small User Model layer that sits beside existing long-term memory:

- Extract user-level facts from messages
- Classify facts into memory categories
- Track confidence and source evidence
- Track whether the memory is user-approved, inferred, or temporary
- Track emotional trajectory across sessions (with decay windows)
- Provide a compact prompt context to the chat engine
- Export/import alongside normal memory payloads

This layer should not replace long-term memory. It should organize the most product-relevant parts of memory for human-like interaction.

Add a Dream State module that sits beside sleep consolidation:

- Inputs: salience-ranked memories, unresolved contradictions, emotional tags
- Processing: REM-style narrative synthesis + conflict linking
- Outputs: compact dream summary + mood carryover vector + open-thread queue
- Controls: feature flag and intensity controls (`off`, `light`, `full`)

## Validation Plan

Add tests around:

- Remembering durable preferences
- Updating contradictions
- Respecting forget requests
- Producing calibrated recall
- Preserving user boundaries
- Generating wake summaries after sleep
- Generating dream summaries after deep sleep
- Keeping dream language calibrated (no false sentience claims)
- Using emotional carryover without destabilizing factual recall
- Adapting response style across sessions
- Avoiding unsupported claims about consciousness or feelings

The strongest demo would be a multi-day persona simulation where the AI remembers what matters, forgets noise, updates stale facts, and resumes naturally after sleep.
