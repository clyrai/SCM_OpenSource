/**
 * SCM JavaScript SDK
 *
 * Minimal client for the SCM /v1 REST API. Idle detection is automatic —
 * SCM tracks per-user activity and fires sleep cycles in the background.
 * The agent only needs to call addMemory() and searchMemory(); a cached
 * wake summary is automatically surfaced in the next response after idle.
 *
 * Usage (Node 18+ / browsers / Edge runtime):
 *
 *     import { SCM } from "scm-memory";
 *     const scm = new SCM({ baseUrl: "http://localhost:8000/v1", userId: "alice" });
 *
 *     await scm.addMemory("My name is Alice and I run on Tuesdays.");
 *     const { memories, memoryContext, wakeSummaryPending } =
 *         await scm.searchMemory("what's my routine?");
 *
 *     if (wakeSummaryPending) {
 *         console.log("Wake summary:", wakeSummaryPending.narrative);
 *     }
 */

const DEFAULT_BASE_URL = "http://localhost:8000/v1";
const DEFAULT_TIMEOUT_MS = 30_000;

export class SCM {
  /**
   * @param {object} options
   * @param {string} [options.baseUrl]   SCM server URL (default: http://localhost:8000/v1)
   * @param {string} [options.userId]    Stable per-user identifier (default: "default")
   * @param {string} [options.apiKey]    Optional bearer token
   * @param {number} [options.timeoutMs] Request timeout (default: 30000)
   */
  constructor({
    baseUrl = DEFAULT_BASE_URL,
    userId = "default",
    apiKey = null,
    timeoutMs = DEFAULT_TIMEOUT_MS,
  } = {}) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.userId = userId;
    this.apiKey = apiKey;
    this.timeoutMs = timeoutMs;
  }

  // ── tools ───────────────────────────────────────────────────────────

  /**
   * Store a fact / observation in long-term memory.
   * @param {string} text
   * @param {object} [metadata]
   * @returns {Promise<{ok:boolean, memory_id?:string, concepts_added?:number, wake_summary_pending?:object}>}
   */
  async addMemory(text, metadata = null) {
    return this._post("/memories", {
      text,
      user_id: this.userId,
      metadata: metadata || undefined,
    });
  }

  /**
   * Retrieve memories relevant to a query (associative recall).
   * @param {string} query
   * @param {number} [limit]
   * @returns {Promise<{ok:boolean, memories:Array, memory_context:string, wake_summary_pending?:object}>}
   */
  async searchMemory(query, limit = 5) {
    return this._post("/memories/search", {
      query,
      user_id: this.userId,
      limit,
    });
  }

  /**
   * Force a sleep cycle. Most callers don't need this — SCM auto-fires
   * sleep when the user has been idle past SCM_IDLE_THRESHOLD_SEC.
   */
  async consolidate(mode = "deep") {
    return this._post("/memories/consolidate", {
      user_id: this.userId,
      mode,
    });
  }

  /**
   * Return the wake summary for the past `sinceHours` hours.
   */
  async wakeSummary(sinceHours = 24) {
    const url = new URL(`${this.baseUrl}/wake-summary`);
    url.searchParams.set("user_id", this.userId);
    url.searchParams.set("since_hours", String(sinceHours));
    return this._fetchJson(url, { method: "GET" });
  }

  /**
   * Permanently remove a specific memory by ID.
   */
  async forget(memoryId) {
    const url = new URL(`${this.baseUrl}/memories/${encodeURIComponent(memoryId)}`);
    url.searchParams.set("user_id", this.userId);
    return this._fetchJson(url, { method: "DELETE" });
  }

  // ── tool-definition exports (for ChatGPT / Claude / Gemini agents) ──

  /**
   * Pull SCM tool definitions in the requested function-calling format.
   * Pass these straight to OpenAI / Anthropic / Gemini SDK calls.
   *
   * @param {"openai"|"anthropic"|"gemini"|"openapi"|"all"} format
   */
  async listTools(format = "openai") {
    const url = new URL(`${this.baseUrl}/tools`);
    url.searchParams.set("format", format);
    return this._fetchJson(url, { method: "GET" });
  }

  /**
   * Health probe.
   */
  async health() {
    return this._fetchJson(new URL(`${this.baseUrl}/health`), { method: "GET" });
  }

  // ── internals ───────────────────────────────────────────────────────

  async _post(path, body) {
    return this._fetchJson(new URL(`${this.baseUrl}${path}`), {
      method: "POST",
      body: JSON.stringify(body),
    });
  }

  async _fetchJson(url, init) {
    const headers = { "Content-Type": "application/json", ...(init.headers || {}) };
    if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;

    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), this.timeoutMs);
    try {
      const r = await fetch(url, {
        ...init,
        headers,
        signal: ctl.signal,
      });
      if (!r.ok) {
        const txt = await r.text();
        throw new SCMError(r.status, txt);
      }
      return await r.json();
    } finally {
      clearTimeout(t);
    }
  }
}

export class SCMError extends Error {
  constructor(status, body) {
    super(`SCM HTTP ${status}: ${body}`);
    this.name = "SCMError";
    this.status = status;
    this.body = body;
  }
}

// Default export for convenience.
export default SCM;
