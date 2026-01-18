import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  ArrowLeft,
  GitCompare,
  Play,
  Clock,
  Coins,
  Layers,
  AlertCircle,
} from 'lucide-react';
import { api } from '../lib/api';
import { SpanTree } from '../components/SpanTree';
import { SpanDetail } from '../components/SpanDetail';
import { MetricCard } from '../components/MetricCard';
import { formatDuration, formatCost, formatTokens, formatDate, getStatusColor } from '../lib/utils';
import { cn } from '../lib/utils';
import type { Span } from '../types';

export function TraceDetail() {
  const { traceId } = useParams<{ traceId: string }>();
  const navigate = useNavigate();
  const [selectedSpan, setSelectedSpan] = useState<Span | null>(null);

  const { data: trace, isLoading, error } = useQuery({
    queryKey: ['trace', traceId],
    queryFn: () => api.getTrace(traceId!),
    enabled: !!traceId,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
      </div>
    );
  }

  if (error || !trace) {
    return (
      <div className="space-y-4">
        <button
          onClick={() => navigate('/traces')}
          className="flex items-center gap-2 text-gray-400 hover:text-white"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to traces
        </button>
        <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-8 text-center">
          <AlertCircle className="h-8 w-8 text-red-400 mx-auto mb-3" />
          <p className="text-red-300">Trace not found</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Back button */}
      <button
        onClick={() => navigate('/traces')}
        className="flex items-center gap-2 text-gray-400 hover:text-white"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to traces
      </button>

      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-white">{trace.name}</h1>
            <span className="text-sm text-gray-500 bg-gray-700 px-2 py-0.5 rounded">
              v{trace.version}
            </span>
            {trace.branch && (
              <span className="text-sm text-blue-400 bg-blue-900/30 px-2 py-0.5 rounded">
                {trace.branch}
              </span>
            )}
            <span
              className={cn(
                'text-xs font-medium px-2 py-1 rounded',
                getStatusColor(trace.status)
              )}
            >
              {trace.status}
            </span>
          </div>
          <p className="text-sm text-gray-400 mt-2">{trace.trace_id}</p>
          <p className="text-sm text-gray-500 mt-1">
            Started {formatDate(trace.start_time)}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => navigate(`/compare?source=${trace.trace_id}`)}
            className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-white transition-colors"
          >
            <GitCompare className="h-4 w-4" />
            Compare
          </button>
          <button
            onClick={() => {/* TODO: implement replay */}}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-white transition-colors"
          >
            <Play className="h-4 w-4" />
            Replay
          </button>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard
          title="Duration"
          value={trace.total_duration_ms ? formatDuration(trace.total_duration_ms) : '-'}
          icon={<Clock className="h-5 w-5" />}
        />
        <MetricCard
          title="Spans"
          value={(trace.spans || []).length}
          icon={<Layers className="h-5 w-5" />}
        />
        <MetricCard
          title="Tokens"
          value={formatTokens(trace.total_tokens)}
          subtitle={`Input: ${formatTokens((trace.spans || []).reduce((sum: number, s: Span) => sum + (s.token_usage?.input_tokens || 0), 0))}`}
          icon={<Layers className="h-5 w-5" />}
        />
        <MetricCard
          title="Cost"
          value={formatCost(trace.total_cost)}
          icon={<Coins className="h-5 w-5" />}
        />
      </div>

      {/* Main content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Span tree */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h2 className="text-lg font-medium text-white mb-4">Span Tree</h2>
          <SpanTree
            spans={trace.spans || []}
            selectedSpanId={selectedSpan?.span_id}
            onSpanClick={setSelectedSpan}
          />
        </div>

        {/* Span detail */}
        <div>
          {selectedSpan ? (
            <SpanDetail
              span={selectedSpan}
              onClose={() => setSelectedSpan(null)}
            />
          ) : (
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center h-full flex items-center justify-center">
              <div>
                <Layers className="h-8 w-8 text-gray-500 mx-auto mb-3" />
                <p className="text-gray-400">Select a span to view details</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Metadata */}
      {Object.keys(trace.metadata).length > 0 && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h2 className="text-lg font-medium text-white mb-4">Metadata</h2>
          <pre className="text-sm text-gray-300 bg-gray-900 rounded p-4 overflow-x-auto">
            {JSON.stringify(trace.metadata, null, 2)}
          </pre>
        </div>
      )}

      {/* Tags */}
      {trace.tags.length > 0 && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h2 className="text-lg font-medium text-white mb-4">Tags</h2>
          <div className="flex flex-wrap gap-2">
            {trace.tags.map((tag) => (
              <span
                key={tag}
                className="text-sm bg-gray-700 text-gray-300 px-3 py-1 rounded"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
