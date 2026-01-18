import { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation } from '@tanstack/react-query';
import { ArrowLeft, GitCompare, AlertCircle } from 'lucide-react';
import { api } from '../lib/api';
import { DiffView } from '../components/DiffView';
import type { Trace } from '../types';

export function Compare() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [sourceId, setSourceId] = useState(searchParams.get('source') || '');
  const [targetId, setTargetId] = useState(searchParams.get('target') || '');

  const { data: traces } = useQuery({
    queryKey: ['traces', 'all'],
    queryFn: () => api.listTraces({ limit: 100 }),
  });

  const compareMutation = useMutation({
    mutationFn: () => api.compareTraces(sourceId, targetId),
  });

  useEffect(() => {
    if (sourceId && targetId) {
      compareMutation.mutate();
    }
  }, [sourceId, targetId]);

  const handleCompare = () => {
    if (sourceId && targetId) {
      compareMutation.mutate();
    }
  };

  return (
    <div className="space-y-6">
      {/* Back button */}
      <button
        onClick={() => navigate(-1)}
        className="flex items-center gap-2 text-gray-400 hover:text-white"
      >
        <ArrowLeft className="h-4 w-4" />
        Back
      </button>

      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Compare Traces</h1>
        <p className="text-gray-400 mt-1">
          Compare two traces to see structural and semantic differences
        </p>
      </div>

      {/* Selection */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">
              Source Trace (Before)
            </label>
            <select
              value={sourceId}
              onChange={(e) => setSourceId(e.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg text-white px-3 py-2 focus:outline-none focus:border-blue-500"
            >
              <option value="">Select a trace...</option>
              {traces?.map((trace) => (
                <option key={trace.trace_id} value={trace.trace_id}>
                  {trace.name} (v{trace.version}) - {new Date(trace.start_time).toLocaleDateString()}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">
              Target Trace (After)
            </label>
            <select
              value={targetId}
              onChange={(e) => setTargetId(e.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg text-white px-3 py-2 focus:outline-none focus:border-blue-500"
            >
              <option value="">Select a trace...</option>
              {traces?.map((trace) => (
                <option key={trace.trace_id} value={trace.trace_id}>
                  {trace.name} (v{trace.version}) - {new Date(trace.start_time).toLocaleDateString()}
                </option>
              ))}
            </select>
          </div>
        </div>

        <button
          onClick={handleCompare}
          disabled={!sourceId || !targetId || compareMutation.isPending}
          className="mt-4 flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg text-white transition-colors"
        >
          <GitCompare className="h-4 w-4" />
          {compareMutation.isPending ? 'Comparing...' : 'Compare'}
        </button>
      </div>

      {/* Results */}
      {compareMutation.isPending && (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
        </div>
      )}

      {compareMutation.isError && (
        <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-8 text-center">
          <AlertCircle className="h-8 w-8 text-red-400 mx-auto mb-3" />
          <p className="text-red-300">Error comparing traces</p>
          <p className="text-sm text-red-400 mt-1">
            {(compareMutation.error as Error)?.message}
          </p>
        </div>
      )}

      {compareMutation.isSuccess && compareMutation.data && (
        <div className="space-y-6">
          {/* Trace headers */}
          <div className="grid grid-cols-2 gap-4">
            <TraceHeader trace={compareMutation.data.source_trace} label="Source" />
            <TraceHeader trace={compareMutation.data.target_trace} label="Target" />
          </div>

          {/* Diff view */}
          <DiffView diff={compareMutation.data} />
        </div>
      )}

      {!sourceId || !targetId ? (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center">
          <GitCompare className="h-8 w-8 text-gray-500 mx-auto mb-3" />
          <p className="text-gray-400">
            Select two traces to compare
          </p>
        </div>
      ) : null}
    </div>
  );
}

function TraceHeader({ trace, label }: { trace: Trace; label: string }) {
  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
      <p className="text-xs text-gray-500 uppercase mb-1">{label}</p>
      <h3 className="font-medium text-white">{trace.name}</h3>
      <div className="flex items-center gap-2 mt-1">
        <span className="text-xs text-gray-500">v{trace.version}</span>
        {trace.branch && (
          <span className="text-xs text-blue-400">{trace.branch}</span>
        )}
      </div>
      <p className="text-xs text-gray-500 mt-2">
        {new Date(trace.start_time).toLocaleString()}
      </p>
    </div>
  );
}
