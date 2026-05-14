# scm-memory — SCM JS/TS SDK

The SCM memory layer, in JavaScript. Works in Node 18+, Bun, Deno, modern
browsers, Cloudflare Workers, Vercel Edge.

```bash
npm install scm-memory
```

## Quickstart

```js
import { SCM } from "scm-memory";

const scm = new SCM({
  baseUrl: "http://localhost:8000/v1",
  userId: "alice@example.com",
});

// Store a memory
await scm.addMemory("My name is Alice and I run on Tuesdays.");

// Retrieve later
const { memories, wake_summary_pending } =
  await scm.searchMemory("what's my routine?");

console.log(memories);
// [{ id, description, type, confidence, created_at }]

if (wake_summary_pending) {
  // SCM detected idle activity, ran a sleep cycle while you weren't
  // looking, and is surfacing the report on this call.
  console.log("While you were away:", wake_summary_pending.narrative);
}
```

## API surface

```ts
addMemory(text: string, metadata?: object): Promise<AddMemoryResponse>
searchMemory(query: string, limit?: number): Promise<SearchMemoryResponse>
consolidate(mode?: "deep" | "micro"): Promise<ConsolidateResponse>
wakeSummary(sinceHours?: number): Promise<WakeSummaryResponse>
forget(memoryId: string): Promise<ForgetResponse>
listTools(format?: "openai"|"anthropic"|"gemini"|"openapi"|"all"): Promise<...>
health(): Promise<...>
```

Most callers use only `addMemory` + `searchMemory`. SCM auto-fires sleep
cycles when the user has been idle past the configured threshold (default
5 minutes), so `consolidate` is a manual override you rarely need.

## Wiring SCM into an OpenAI agent

```js
import OpenAI from "openai";
import { SCM } from "scm-memory";

const openai = new OpenAI();
const scm = new SCM({ userId: "alice" });

// Pull tool defs in OpenAI function-calling format
const { tools } = await scm.listTools("openai");

const resp = await openai.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "I'm allergic to peanuts." }],
  tools,
});

// When the model returns a tool_call, route it to SCM:
for (const tc of resp.choices[0].message.tool_calls || []) {
  const args = JSON.parse(tc.function.arguments);
  if (tc.function.name === "add_memory") {
    await scm.addMemory(args.text, args.metadata);
  } else if (tc.function.name === "search_memory") {
    await scm.searchMemory(args.query, args.limit);
  }
  // ...etc.
}
```

The same pattern works with Anthropic Claude (`scm.listTools("anthropic")`)
and Google Gemini (`scm.listTools("gemini")`).

## License

MIT.
