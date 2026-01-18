import { useQuery } from '@tanstack/react-query';
import { Activity, Coins, Layers, TrendingUp, AlertCircle } from 'lucide-react';
import { api } from '../lib/api';
import { MetricCard } from '../components/MetricCard';
import { TraceCard } from '../components/TraceCard';
import { CostChart, TimeSeriesChart } from '../components/CostChart';
import { formatCost, formatTokens } from '../lib/utils';
import { useNavigate } from 'react-router-dom';

export function Dashboard() {
  const navigate = useNavigate();

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['statistics'],
    queryFn: api.getStatistics,
  });

  const { data: recentTraces, isLoading: tracesLoading } = useQuery({
    queryKey: ['traces', 'recent'],
    queryFn: () => api.listTraces({ limit: 5 }),
  });

  const { data: costBreakdown } = useQuery({
    queryKey: ['costs', 'breakdown'],
    queryFn: () => api.getCostBreakdown({ group_by: 'model' }),
  });

  // Mock time series data for demo
  const timeSeriesData = [
    { date: 'Mon', cost: 1.25, tokens: 45000 },
    { date: 'Tue', cost: 2.10, tokens: 78000 },
    { date: 'Wed', cost: 1.85, tokens: 62000 },
    { date: 'Thu', cost: 3.20, tokens: 98000 },
    { date: 'Fri', cost: 2.75, tokens: 85000 },
    { date: 'Sat', cost: 1.50, tokens: 42000 },
    { date: 'Sun', cost: 0.95, tokens: 28000 },
  ];

  if (statsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-gray-400 mt-1">Overview of your agent traces and costs</p>
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total Traces"
          value={stats?.trace_count || 0}
          icon={<Activity className="h-5 w-5" />}
          trend={{ value: 12, isPositive: true }}
        />
        <MetricCard
          title="Total Spans"
          value={stats?.span_count || 0}
          icon={<Layers className="h-5 w-5" />}
        />
        <MetricCard
          title="Total Tokens"
          value={formatTokens(stats?.total_tokens || 0)}
          icon={<TrendingUp className="h-5 w-5" />}
        />
        <MetricCard
          title="Total Cost"
          value={formatCost(stats?.total_cost || 0)}
          icon={<Coins className="h-5 w-5" />}
          trend={{ value: 8, isPositive: false }}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h2 className="text-lg font-medium text-white mb-4">Cost Over Time</h2>
          <TimeSeriesChart data={timeSeriesData} />
        </div>
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h2 className="text-lg font-medium text-white mb-4">Cost by Model</h2>
          {costBreakdown ? (
            <CostChart data={costBreakdown} type="pie" />
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-500">
              No cost data available
            </div>
          )}
        </div>
      </div>

      {/* Recent traces */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-medium text-white">Recent Traces</h2>
          <button
            onClick={() => navigate('/traces')}
            className="text-sm text-blue-400 hover:text-blue-300"
          >
            View all
          </button>
        </div>
        {tracesLoading ? (
          <div className="flex items-center justify-center h-32">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500" />
          </div>
        ) : recentTraces && recentTraces.length > 0 ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {recentTraces.map((trace) => (
              <TraceCard
                key={trace.trace_id}
                trace={trace}
                onClick={() => navigate(`/traces/${trace.trace_id}`)}
              />
            ))}
          </div>
        ) : (
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center">
            <AlertCircle className="h-8 w-8 text-gray-500 mx-auto mb-3" />
            <p className="text-gray-400">No traces found</p>
            <p className="text-sm text-gray-500 mt-1">
              Start tracing your agents to see data here
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
