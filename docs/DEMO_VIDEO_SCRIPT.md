# Demo Video Script — 90 seconds

The script you record. Optimized to land the wake-summary moment in under 90 seconds — the single most-shareable framing of SCM.

---

## Production parameters

- **Length:** 80-95 seconds. Strict. Anything over 100s hurts shareability.
- **Format:** 1080p screen-record + voice-over. No fancy editing required.
- **Tone:** matter-of-fact, not salesy. The product speaks for itself if you let the wake-summary moment land.
- **Mic:** any decent USB mic. Avoid laptop built-in.
- **Recording tools:** OBS / QuickTime / Loom. Output as MP4 (H.264 + AAC).

---

## The script

### [0:00 – 0:08] Hook + the problem

🎙 **VO**: "Every AI agent today has a fundamental flaw. The moment you stop talking to it, it stops thinking."

📺 **Screen**: dark text on plain background:
> *"The agent that wakes up tomorrow is exactly the agent that went to sleep last night."*

### [0:08 – 0:18] The category framing

🎙 **VO**: "That's not how memory works in any system that does it well — including yours. Sleep is when memory consolidates. You wake up with a better version of yesterday's understanding, not the same one."

📺 **Screen**: split-screen — left side: "your brain" with arrows showing day → sleep → consolidated next-day memory. Right side: "today's AI memory" with arrows showing day → flat-line → same memory tomorrow. Labels keep it simple.

### [0:18 – 0:28] The thesis

🎙 **VO**: "Meet SCM — the first agent memory layer with both a wake phase and a sleep phase. It learns about you the way humans learn — including while you're not paying attention."

📺 **Screen**: zoom into the SCM logo (or just the text "SCM — Memory that works like yours").

### [0:28 – 0:55] The actual demo (CRITICAL — this is the moment)

🎙 **VO** (cut to terminal recording): "Here's what that looks like. I'll talk to my agent for a few turns, then close it for the day."

📺 **Screen**: terminal showing:

```
You> Hi, I'm Alex. I'm a backend engineer in Lisbon.
Bot> Nice to meet you, Alex!

You> I run every Tuesday morning along the river.
Bot> Got it.

You> Tuesday again — did 6km this time.
Bot> Sounds like Tuesday is your running day.

You> /quit
```

🎙 **VO**: "Now the agent's been idle. SCM noticed. It quietly fired a sleep cycle. Schemas formed. Patterns abstracted. Watch what happens when I come back."

📺 **Screen**: clear screen, blink, restart terminal:

```
$ python chatbot.py

💤 While you were away:
While you were away I noticed Tuesday-morning runs along the river have
become a weekly pattern. I also have you logged as a backend engineer in
Lisbon.

You> Morning!
Bot> Good morning, Alex! Hope today's run was good.
```

🎙 **VO** (slowly, let the moment land): "The agent thought about me while I wasn't there. **It's the only one that does.**"

📺 **Screen**: pause on the wake summary for 2 full seconds. This is the screenshot people will share.

### [0:55 – 1:15] What it means

🎙 **VO**: "SCM is open-source. MIT-licensed. Works with any LLM — OpenAI, Claude, DeepSeek, Ollama, anything. Drops into LangChain, Claude Desktop, ChatGPT custom GPTs, or your own agent. The whole architecture is in a 35-page paper. The brutal-test harness that found four real bugs is on GitHub."

📺 **Screen**: rapid montage:
- Logo + URL: `scm.run` (or wherever the demo is hosted)
- `pip install scm-memory` in a terminal, INSTANT install completion
- Claude Desktop config snippet showing SCM MCP server
- LangChain code snippet
- GitHub repo URL: `github.com/Saish15/sleepai`

### [1:15 – 1:30] Close

🎙 **VO**: "Try it in your browser. No install, no signup, no credit card. The link's below. Memory that actually works like yours."

📺 **Screen**: large URL displayed prominently:
> **scm.run**
>
> *Open-source · Works with any LLM · Privacy-first*

End on a clean shot of the URL for at least 2 seconds. People screenshot the end frame.

---

## Recording checklist

Before you hit record:

- [ ] Terminal font ≥ 16pt (people will watch on phones)
- [ ] Terminal background = high contrast (dark on light or light on dark)
- [ ] Browser zoomed in ≥ 125% for any web shots
- [ ] Notifications disabled (no Slack pings, no calendar popups)
- [ ] Mic levels checked — say "test 1 2 3" and play it back
- [ ] No mouse jitter — slow, deliberate movements only
- [ ] Pre-script the terminal commands in a `.sh` file so you don't fat-finger live

## Editing notes

- Cut all "ums" and "uhhs" — be ruthless.
- Music: subtle ambient loop at -20dB, NOT overpowering the voice. Royalty-free from epidemicsound.com or similar.
- No sound effects. No swooshes. No transitions other than hard cuts.
- Color grading: leave it neutral. This is a tech demo, not a Tesla launch.

## Where to publish

| Platform | Aspect ratio | Length cap | Notes |
|---|---|---|---|
| YouTube (long-form) | 16:9 | 90s perfect | Embed in README |
| Twitter / X | 1:1 or 16:9 | 140s max | Auto-plays in feed; this is the highest-impact placement for the launch |
| HackerNews submission | n/a | n/a | Embed YouTube link in the Show HN comment |
| GitHub README | 16:9 | n/a | Use the YouTube embed |
| LinkedIn | 16:9 | 10 min cap | Cross-post for the engineering / dev-tools audience |

**Priority:** Twitter > YouTube > HN > GitHub > LinkedIn. The Twitter post is what gets the launch noticed; everything else is secondary.

---

## Alternative scripts (if you hate this one)

### "Bug-fix" framing (technical audience)

Lead with a bug we caught: *"This is what 4 hours of brutal testing looks like."* Show the 5,561× latency speedup chart. Show 16/16 brutal scenarios passing. End on `pip install scm-memory`.

### "Two-phase memory" framing (researcher audience)

Lead with the wake/sleep table from the README. Show the architecture diagram from the paper. Cut to terminal showing the wake summary. End on the arXiv paper preview.

### "Just the wake summary" framing (consumer audience)

No setup. Just: terminal → user types one message → agent responds → user closes the window → time-skip → user reopens → wake summary appears. 30 seconds total. Best for TikTok / Instagram Reels.

---

## What success looks like

A 90-second video. Posted to Twitter with the URL `scm.run` and one tagline. Someone shares it. 24 hours later: the launch is real.

A 10-minute video would not be shared. A 30-second video might be. A 90-second video that lands the wake-summary moment is the right length.
