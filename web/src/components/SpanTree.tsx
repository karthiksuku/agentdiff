import { useState } from 'react';
import { ChevronRight, ChevronDown } from 'lucide-react';
import type { Span } from '../types';
import {
  cn,
  getSpanTypeColor,
  getSpanTypeIcon,
  getStatusColor,
  formatDuration,
  formatTokens,
  formatCost,
} from '../lib/utils';

interface SpanTreeProps {
  spans: Span[];
  onSpanClick?: (span: Span) => void;
  selectedSpanId?: string;
}

interface SpanNode {
  span: Span;
  children: SpanNode[];
}

function buildSpanTree(spans: Span[]): SpanNode[] {
  const spanMap = new Map<string, SpanNode>();
  const roots: SpanNode[] = [];

  // Create nodes
  for (const span of spans) {
    spanMap.set(span.span_id, { span, children: [] });
  }

  // Build tree
  for (const span of spans) {
    const node = spanMap.get(span.span_id)!;
    if (span.parent_span_id && spanMap.has(span.parent_span_id)) {
      spanMap.get(span.parent_span_id)!.children.push(node);
    } else {
      roots.push(node);
    }
  }

  return roots;
}

function SpanNodeComponent({
  node,
  depth = 0,
  onSpanClick,
  selectedSpanId,
}: {
  node: SpanNode;
  depth?: number;
  onSpanClick?: (span: Span) => void;
  selectedSpanId?: string;
}) {
  const [isExpanded, setIsExpanded] = useState(true);
  const { span, children } = node;
  const hasChildren = children.length > 0;
  const isSelected = selectedSpanId === span.span_id;

  return (
    <div className="animate-fade-in">
      <div
        className={cn(
          'flex items-center gap-2 px-2 py-1.5 rounded cursor-pointer transition-colors',
          isSelected ? 'bg-blue-600/30' : 'hover:bg-gray-700/50'
        )}
        style={{ paddingLeft: `${depth * 20 + 8}px` }}
        onClick={() => onSpanClick?.(span)}
      >
        {/* Expand/Collapse */}
        {hasChildren ? (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setIsExpanded(!isExpanded);
            }}
            className="p-0.5 hover:bg-gray-600 rounded"
          >
            {isExpanded ? (
              <ChevronDown className="h-4 w-4 text-gray-400" />
            ) : (
              <ChevronRight className="h-4 w-4 text-gray-400" />
            )}
          </button>
        ) : (
          <div className="w-5" />
        )}

        {/* Icon */}
        <span className="text-lg">{getSpanTypeIcon(span.span_type)}</span>

        {/* Type badge */}
        <span
          className={cn(
            'px-1.5 py-0.5 text-xs font-medium rounded',
            getSpanTypeColor(span.span_type),
            'text-white'
          )}
        >
          {span.span_type}
        </span>

        {/* Name */}
        <span className="flex-1 text-sm text-gray-200 truncate">
          {span.name || 'unnamed'}
        </span>

        {/* Model */}
        {span.model && (
          <span className="text-xs text-gray-400 bg-gray-700 px-2 py-0.5 rounded">
            {span.model}
          </span>
        )}

        {/* Duration */}
        {span.duration_ms && (
          <span className="text-xs text-gray-400">
            {formatDuration(span.duration_ms)}
          </span>
        )}

        {/* Tokens */}
        {span.token_usage && (
          <span className="text-xs text-gray-400">
            {formatTokens(span.token_usage.total_tokens)} tokens
          </span>
        )}

        {/* Cost */}
        {span.token_usage && span.token_usage.total_cost > 0 && (
          <span className="text-xs text-green-400">
            {formatCost(span.token_usage.total_cost)}
          </span>
        )}

        {/* Status */}
        <span className={cn('text-xs', getStatusColor(span.status))}>
          {span.status === 'completed' ? '✓' : span.status === 'failed' ? '✗' : '⟳'}
        </span>
      </div>

      {/* Children */}
      {isExpanded && hasChildren && (
        <div>
          {children.map((child) => (
            <SpanNodeComponent
              key={child.span.span_id}
              node={child}
              depth={depth + 1}
              onSpanClick={onSpanClick}
              selectedSpanId={selectedSpanId}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function SpanTree({ spans, onSpanClick, selectedSpanId }: SpanTreeProps) {
  const tree = buildSpanTree(spans);

  if (spans.length === 0) {
    return (
      <div className="text-center py-8 text-gray-400">
        No spans to display
      </div>
    );
  }

  return (
    <div className="space-y-0.5">
      {tree.map((node) => (
        <SpanNodeComponent
          key={node.span.span_id}
          node={node}
          onSpanClick={onSpanClick}
          selectedSpanId={selectedSpanId}
        />
      ))}
    </div>
  );
}
