import { X } from 'lucide-react';
import type { Span } from '../types';
import {
  getSpanTypeIcon,
  getSpanTypeColor,
  getStatusColor,
  formatDuration,
  formatTokens,
  formatCost,
  formatDate,
} from '../lib/utils';
import { cn } from '../lib/utils';

interface SpanDetailProps {
  span: Span;
  onClose: () => void;
}

export function SpanDetail({ span, onClose }: SpanDetailProps) {
  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden animate-slide-in">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gray-750 border-b border-gray-700">
        <div className="flex items-center gap-3">
          <span className="text-xl">{getSpanTypeIcon(span.span_type)}</span>
          <div>
            <h3 className="font-medium text-white">{span.name || 'Unnamed Span'}</h3>
            <span
              className={cn(
                'text-xs px-2 py-0.5 rounded',
                getSpanTypeColor(span.span_type),
                'text-white'
              )}
            >
              {span.span_type}
            </span>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1 hover:bg-gray-700 rounded transition-colors"
        >
          <X className="h-5 w-5 text-gray-400" />
        </button>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4 max-h-[500px] overflow-y-auto">
        {/* Status & Timing */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-xs text-gray-400 uppercase">Status</label>
            <p className={cn('font-medium', getStatusColor(span.status))}>
              {span.status}
            </p>
          </div>
          <div>
            <label className="text-xs text-gray-400 uppercase">Duration</label>
            <p className="font-medium text-white">
              {span.duration_ms ? formatDuration(span.duration_ms) : '-'}
            </p>
          </div>
          {span.model && (
            <div>
              <label className="text-xs text-gray-400 uppercase">Model</label>
              <p className="font-medium text-white">{span.model}</p>
            </div>
          )}
          {span.provider && (
            <div>
              <label className="text-xs text-gray-400 uppercase">Provider</label>
              <p className="font-medium text-white">{span.provider}</p>
            </div>
          )}
        </div>

        {/* Token Usage */}
        {span.token_usage && (
          <div>
            <label className="text-xs text-gray-400 uppercase mb-2 block">
              Token Usage
            </label>
            <div className="grid grid-cols-3 gap-2">
              <div className="bg-gray-700/50 rounded p-2">
                <p className="text-xs text-gray-400">Input</p>
                <p className="font-medium text-white">
                  {formatTokens(span.token_usage.input_tokens)}
                </p>
              </div>
              <div className="bg-gray-700/50 rounded p-2">
                <p className="text-xs text-gray-400">Output</p>
                <p className="font-medium text-white">
                  {formatTokens(span.token_usage.output_tokens)}
                </p>
              </div>
              <div className="bg-gray-700/50 rounded p-2">
                <p className="text-xs text-gray-400">Cost</p>
                <p className="font-medium text-green-400">
                  {formatCost(span.token_usage.total_cost)}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Confidence */}
        {span.confidence_score !== null && (
          <div>
            <label className="text-xs text-gray-400 uppercase">Confidence</label>
            <div className="flex items-center gap-2 mt-1">
              <div className="flex-1 bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full"
                  style={{ width: `${span.confidence_score * 100}%` }}
                />
              </div>
              <span className="text-sm text-white">
                {(span.confidence_score * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        )}

        {/* Reasoning */}
        {span.reasoning && (
          <div>
            <label className="text-xs text-gray-400 uppercase mb-1 block">
              Reasoning
            </label>
            <p className="text-sm text-gray-300 bg-gray-700/50 rounded p-2">
              {span.reasoning}
            </p>
          </div>
        )}

        {/* Input Data */}
        {Object.keys(span.input_data).length > 0 && (
          <div>
            <label className="text-xs text-gray-400 uppercase mb-1 block">
              Input
            </label>
            <pre className="text-xs text-gray-300 bg-gray-900 rounded p-3 overflow-x-auto">
              {JSON.stringify(span.input_data, null, 2)}
            </pre>
          </div>
        )}

        {/* Output Data */}
        {Object.keys(span.output_data).length > 0 && (
          <div>
            <label className="text-xs text-gray-400 uppercase mb-1 block">
              Output
            </label>
            <pre className="text-xs text-gray-300 bg-gray-900 rounded p-3 overflow-x-auto">
              {JSON.stringify(span.output_data, null, 2)}
            </pre>
          </div>
        )}

        {/* Error */}
        {span.error && (
          <div>
            <label className="text-xs text-red-400 uppercase mb-1 block">
              Error
            </label>
            <div className="bg-red-900/30 border border-red-500/50 rounded p-3">
              <p className="text-sm text-red-300">{span.error}</p>
              {span.error_type && (
                <p className="text-xs text-red-400 mt-1">{span.error_type}</p>
              )}
            </div>
          </div>
        )}

        {/* Metadata */}
        {Object.keys(span.metadata).length > 0 && (
          <div>
            <label className="text-xs text-gray-400 uppercase mb-1 block">
              Metadata
            </label>
            <pre className="text-xs text-gray-300 bg-gray-900 rounded p-3 overflow-x-auto">
              {JSON.stringify(span.metadata, null, 2)}
            </pre>
          </div>
        )}

        {/* Tags */}
        {span.tags.length > 0 && (
          <div>
            <label className="text-xs text-gray-400 uppercase mb-1 block">
              Tags
            </label>
            <div className="flex flex-wrap gap-1">
              {span.tags.map((tag) => (
                <span
                  key={tag}
                  className="text-xs bg-gray-700 text-gray-300 px-2 py-0.5 rounded"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Timestamps */}
        <div className="text-xs text-gray-400 pt-2 border-t border-gray-700">
          <p>Started: {formatDate(span.start_time)}</p>
          {span.end_time && <p>Ended: {formatDate(span.end_time)}</p>}
          <p className="mt-1">ID: {span.span_id}</p>
        </div>
      </div>
    </div>
  );
}
