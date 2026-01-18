import { ArrowRight, Plus, Minus, RefreshCw } from 'lucide-react';
import type { TraceDiff, SpanDiff } from '../types';
import { cn } from '../lib/utils';

interface DiffViewProps {
  diff: TraceDiff;
}

function getChangeTypeColor(changeType: string): string {
  switch (changeType) {
    case 'added':
      return 'text-green-400 bg-green-900/30';
    case 'removed':
      return 'text-red-400 bg-red-900/30';
    case 'modified':
      return 'text-yellow-400 bg-yellow-900/30';
    case 'unchanged':
    default:
      return 'text-gray-400 bg-gray-700/30';
  }
}

function getChangeTypeIcon(changeType: string) {
  switch (changeType) {
    case 'added':
      return <Plus className="h-4 w-4" />;
    case 'removed':
      return <Minus className="h-4 w-4" />;
    case 'modified':
      return <RefreshCw className="h-4 w-4" />;
    default:
      return null;
  }
}

function SpanDiffItem({ spanDiff }: { spanDiff: SpanDiff }) {
  const changeType = spanDiff.diff_type;
  const colorClass = getChangeTypeColor(changeType);
  const changes = Object.entries(spanDiff.field_changes || {});

  return (
    <div
      className={cn(
        'rounded-lg border p-3 mb-2',
        changeType === 'added' && 'border-green-500/50 bg-green-900/10',
        changeType === 'removed' && 'border-red-500/50 bg-red-900/10',
        (changeType === 'modified' || changeType === 'reordered') && 'border-yellow-500/50 bg-yellow-900/10',
        changeType === 'unchanged' && 'border-gray-700 bg-gray-800/50'
      )}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={cn('p-1 rounded', colorClass)}>
            {getChangeTypeIcon(changeType)}
          </span>
          <span className="font-medium text-white">
            {spanDiff.span_a_name || spanDiff.span_b_name || 'Unknown'}
          </span>
          <span className="text-xs text-gray-500 capitalize">
            {changeType}
          </span>
        </div>
        {spanDiff.cost_delta !== 0 && (
          <span className={cn(
            'text-xs',
            spanDiff.cost_delta > 0 ? 'text-red-400' : 'text-green-400'
          )}>
            {spanDiff.cost_delta > 0 ? '+' : ''}${spanDiff.cost_delta.toFixed(4)}
          </span>
        )}
      </div>

      {changes.length > 0 && (
        <div className="mt-3 space-y-2">
          {changes.map(([field, change], idx) => (
            <div key={idx} className="text-sm bg-gray-900/50 rounded p-2">
              <span className="text-gray-400">{field}:</span>
              <div className="flex items-center gap-2 mt-1">
                {(change as { old: unknown; new: unknown }).old !== undefined && (
                  <span className="text-red-300 line-through text-xs">
                    {typeof (change as { old: unknown }).old === 'object'
                      ? JSON.stringify((change as { old: unknown }).old)
                      : String((change as { old: unknown }).old)}
                  </span>
                )}
                {(change as { old: unknown }).old !== undefined && (change as { new: unknown }).new !== undefined && (
                  <ArrowRight className="h-3 w-3 text-gray-500" />
                )}
                {(change as { new: unknown }).new !== undefined && (
                  <span className="text-green-300 text-xs">
                    {typeof (change as { new: unknown }).new === 'object'
                      ? JSON.stringify((change as { new: unknown }).new)
                      : String((change as { new: unknown }).new)}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function DiffView({ diff }: DiffViewProps) {
  const summary = diff.summary || {
    added_spans: diff.span_diffs.filter((d) => d.diff_type === 'added').length,
    removed_spans: diff.span_diffs.filter((d) => d.diff_type === 'removed').length,
    modified_spans: diff.span_diffs.filter((d) => d.diff_type === 'modified').length,
    unchanged_spans: diff.span_diffs.filter((d) => d.diff_type === 'unchanged').length,
  };

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="flex items-center gap-4 p-4 bg-gray-800 rounded-lg border border-gray-700">
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">Similarity:</span>
          <span className="font-medium text-white">
            {(diff.structural_similarity * 100).toFixed(1)}%
          </span>
        </div>
        <div className="flex-1 bg-gray-700 rounded-full h-2">
          <div
            className="bg-blue-500 h-2 rounded-full transition-all"
            style={{ width: `${diff.structural_similarity * 100}%` }}
          />
        </div>
      </div>

      {/* Change counts */}
      <div className="grid grid-cols-4 gap-2">
        <div className="bg-green-900/20 border border-green-500/30 rounded p-3 text-center">
          <p className="text-2xl font-bold text-green-400">{summary.added_spans}</p>
          <p className="text-xs text-green-300">Added</p>
        </div>
        <div className="bg-red-900/20 border border-red-500/30 rounded p-3 text-center">
          <p className="text-2xl font-bold text-red-400">{summary.removed_spans}</p>
          <p className="text-xs text-red-300">Removed</p>
        </div>
        <div className="bg-yellow-900/20 border border-yellow-500/30 rounded p-3 text-center">
          <p className="text-2xl font-bold text-yellow-400">{summary.modified_spans}</p>
          <p className="text-xs text-yellow-300">Modified</p>
        </div>
        <div className="bg-gray-700/20 border border-gray-600/30 rounded p-3 text-center">
          <p className="text-2xl font-bold text-gray-400">{summary.unchanged_spans}</p>
          <p className="text-xs text-gray-300">Unchanged</p>
        </div>
      </div>

      {/* Span diffs */}
      <div>
        <h3 className="text-sm font-medium text-gray-300 mb-3">Span Changes</h3>
        <div className="space-y-2">
          {diff.span_diffs.map((spanDiff, idx) => (
            <SpanDiffItem key={idx} spanDiff={spanDiff} />
          ))}
        </div>
      </div>

      {/* Token diff */}
      {diff.token_delta !== 0 && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-2">Token Usage</h3>
          <div className="flex items-center gap-2">
            <span
              className={cn(
                'text-lg font-medium',
                diff.token_delta > 0 ? 'text-red-400' : 'text-green-400'
              )}
            >
              {diff.token_delta > 0 ? '+' : ''}
              {diff.token_delta.toLocaleString()}
            </span>
            <span className="text-gray-500">tokens</span>
          </div>
        </div>
      )}

      {/* Cost diff */}
      {diff.cost_delta !== 0 && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-2">Cost Impact</h3>
          <div className="flex items-center gap-2">
            <span
              className={cn(
                'text-lg font-medium',
                diff.cost_delta > 0 ? 'text-red-400' : 'text-green-400'
              )}
            >
              {diff.cost_delta > 0 ? '+' : ''}${diff.cost_delta.toFixed(4)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
