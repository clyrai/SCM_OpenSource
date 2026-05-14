/**
 * SCM JavaScript SDK — TypeScript definitions.
 */

export interface SCMOptions {
  baseUrl?: string;
  userId?: string;
  apiKey?: string | null;
  timeoutMs?: number;
}

export interface MemoryRecord {
  id: string;
  description: string;
  type: string;
  confidence: number;
  created_at: string | null;
}

export interface WakeSummaryPending {
  narrative: string;
  insights: string[];
  fired_at: string;
  idle_seconds: number;
}

export interface AddMemoryResponse {
  ok: boolean;
  user_id: string;
  memory_id?: string | null;
  concepts_added: number;
  concepts_total: number;
  wake_summary_pending?: WakeSummaryPending;
  error?: string;
}

export interface SearchMemoryResponse {
  ok: boolean;
  user_id: string;
  query: string;
  memories: MemoryRecord[];
  memory_context: string;
  retrieved_count: number;
  wake_summary_pending?: WakeSummaryPending;
}

export interface ConsolidateResponse {
  ok: boolean;
  user_id: string;
  mode: "deep" | "micro";
  schemas_formed: number;
  concepts_consolidated: number;
  concepts_forgotten: number;
  contradictions_resolved: number;
}

export interface WakeSummaryResponse {
  ok: boolean;
  user_id: string;
  since: string;
  narrative: string;
  insights: string[];
  schemas_formed: number;
}

export interface ForgetResponse {
  ok: boolean;
  user_id: string;
  memory_id: string;
}

export type ToolFormat = "openai" | "anthropic" | "gemini" | "openapi" | "all";

export class SCM {
  constructor(options?: SCMOptions);
  baseUrl: string;
  userId: string;
  apiKey: string | null;
  timeoutMs: number;

  addMemory(text: string, metadata?: Record<string, unknown> | null): Promise<AddMemoryResponse>;
  searchMemory(query: string, limit?: number): Promise<SearchMemoryResponse>;
  consolidate(mode?: "deep" | "micro"): Promise<ConsolidateResponse>;
  wakeSummary(sinceHours?: number): Promise<WakeSummaryResponse>;
  forget(memoryId: string): Promise<ForgetResponse>;
  listTools(format?: ToolFormat): Promise<unknown>;
  health(): Promise<{ ok: boolean; active_users: number; auto_sleep: boolean; idle_threshold_sec: number }>;
}

export class SCMError extends Error {
  status: number;
  body: string;
}

export default SCM;
