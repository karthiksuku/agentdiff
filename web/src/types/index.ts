// API Types for AgentDiff

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cached_tokens: number;
  input_cost: number;
  output_cost: number;
  total_cost: number;
}

export interface Span {
  span_id: string;
  trace_id: string;
  parent_span_id: string | null;
  name: string;
  span_type: SpanType;
  status: SpanStatus;
  start_time: string;
  end_time: string | null;
  duration_ms: number | null;
  model: string | null;
  provider: string | null;
  token_usage: TokenUsage | null;
  input_data: Record<string, unknown>;
  output_data: Record<string, unknown>;
  metadata: Record<string, unknown>;
  tags: string[];
  error: string | null;
  error_type: string | null;
  confidence_score: number | null;
  reasoning: string | null;
}

export type SpanType =
  | 'llm_call'
  | 'tool_call'
  | 'memory_access'
  | 'agent_step'
  | 'planning'
  | 'reasoning'
  | 'retrieval'
  | 'embedding'
  | 'custom';

export type SpanStatus = 'running' | 'completed' | 'failed' | 'cancelled';

export interface Trace {
  trace_id: string;
  name: string;
  version: string;
  branch: string;
  parent_trace_id: string | null;
  start_time: string;
  end_time: string | null;
  total_tokens: number;
  total_cost: number;
  total_duration_ms: number;
  metadata: Record<string, unknown>;
  tags: string[];
  commit_message: string | null;
  status: string;
  error: string | null;
  spans?: Span[];
}

export interface SpanDiff {
  diff_type: 'added' | 'removed' | 'modified' | 'unchanged' | 'reordered';
  span_a_id: string | null;
  span_b_id: string | null;
  span_a_name: string | null;
  span_b_name: string | null;
  field_changes: Record<string, { old: unknown; new: unknown }>;
  token_delta: number;
  cost_delta: number;
  latency_delta_ms: number;
  confidence_delta: number;
  is_divergence_point: boolean;
  has_output_change: boolean;
  has_tool_change: boolean;
}

export interface TraceDiff {
  trace_a_id: string;
  trace_b_id: string;
  trace_a_name: string;
  trace_b_name: string;
  structural_similarity: number;
  total_divergences: number;
  token_delta: number;
  cost_delta: number;
  latency_delta_ms: number;
  span_diffs: SpanDiff[];
  divergence_points: SpanDiff[];
  regressions: SpanDiff[];
  improvements: SpanDiff[];
  summary: {
    added_spans: number;
    removed_spans: number;
    modified_spans: number;
    unchanged_spans: number;
  };
}

export interface CostBreakdown {
  trace_id: string;
  trace_name: string;
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  total_input_cost: number;
  total_output_cost: number;
  total_cost: number;
  span_costs: SpanCost[];
  cost_by_model: Record<string, number>;
  cost_by_type: Record<string, number>;
}

export interface SpanCost {
  span_id: string;
  span_name: string;
  span_type: SpanType;
  model: string | null;
  provider: string | null;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  input_cost: number;
  output_cost: number;
  total_cost: number;
  cost_percentage: number;
}

export interface Checkpoint {
  checkpoint_id: string;
  trace_id: string;
  span_id: string;
  name: string;
  state_snapshot: Record<string, unknown>;
  created_at: string;
}

export interface Stats {
  supported: boolean;
  trace_count: number;
  span_count: number;
  total_tokens: number;
  total_cost: number;
  db_path?: string;
  db_size_bytes?: number;
}
