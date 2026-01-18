import { Clock, Coins, Layers } from 'lucide-react';
import type { Trace } from '../types';
import { formatDuration, formatCost, formatTokens, formatDate, getStatusColor } from '../lib/utils';
import { cn } from '../lib/utils';

interface TraceCardProps {
  trace: Trace;
  onClick?: () => void;
}

export function TraceCard({ trace, onClick }: TraceCardProps) {
  return (
    <div
      onClick={onClick}
      className={cn(
        'bg-gray-800 rounded-lg border border-gray-700 p-4 hover:border-gray-600 transition-colors',
        onClick && 'cursor-pointer'
      )}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="font-medium text-white truncate">{trace.name}</h3>
            <span className="text-xs text-gray-500 bg-gray-700 px-2 py-0.5 rounded">
              v{trace.version}
            </span>
            {trace.branch && (
              <span className="text-xs text-blue-400 bg-blue-900/30 px-2 py-0.5 rounded">
                {trace.branch}
              </span>
            )}
          </div>
          <p className="text-sm text-gray-400 mt-1 truncate">
            {trace.trace_id}
          </p>
        </div>
        <span
          className={cn(
            'text-xs font-medium px-2 py-1 rounded',
            getStatusColor(trace.status)
          )}
        >
          {trace.status}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-4">
        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4 text-gray-500" />
          <div>
            <p className="text-xs text-gray-500">Duration</p>
            <p className="text-sm text-white">
              {trace.total_duration_ms ? formatDuration(trace.total_duration_ms) : '-'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Layers className="h-4 w-4 text-gray-500" />
          <div>
            <p className="text-xs text-gray-500">Tokens</p>
            <p className="text-sm text-white">{formatTokens(trace.total_tokens)}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Coins className="h-4 w-4 text-gray-500" />
          <div>
            <p className="text-xs text-gray-500">Cost</p>
            <p className="text-sm text-green-400">{formatCost(trace.total_cost)}</p>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between mt-4 pt-3 border-t border-gray-700">
        <span className="text-xs text-gray-500">
          {(trace.spans || []).length} spans
        </span>
        <span className="text-xs text-gray-500">
          {formatDate(trace.start_time)}
        </span>
      </div>
    </div>
  );
}
