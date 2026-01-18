import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Coins, TrendingUp, TrendingDown, AlertCircle } from 'lucide-react';
import { api } from '../lib/api';
import { MetricCard } from '../components/MetricCard';
import { CostChart, TokenChart, TimeSeriesChart } from '../components/CostChart';
import { formatCost, formatTokens } from '../lib/utils';

type GroupBy = 'model' | 'provider' | 'agent' | 'date';

export function Costs() {
  const [groupBy, setGroupBy] = useState<GroupBy>('model');
  const [dateRange, setDateRange] = useState('7d');

  const { data: stats } = useQuery({
    queryKey: ['statistics'],
    queryFn: api.getStatistics,
  });

  const { data: costBreakdown, isLoading } = useQuery({
    queryKey: ['costs', 'breakdown', groupBy],
    queryFn: () => api.getCostBreakdown({ group_by: groupBy }),
  });

  // Mock token breakdown data
  const tokenBreakdown = [
    { category: 'GPT-4', input: 125000, output: 45000 },
    { category: 'GPT-3.5', input: 85000, output: 32000 },
    { category: 'Claude-3', input: 65000, output: 28000 },
    { category: 'Claude-2', input: 45000, output: 18000 },
  ];

  // Mock time series data
  const timeSeriesData = [
    { date: 'Jan 12', cost: 12.50, tokens: 450000 },
    { date: 'Jan 13', cost: 18.75, tokens: 680000 },
    { date: 'Jan 14', cost: 15.20, tokens: 520000 },
    { date: 'Jan 15', cost: 22.30, tokens: 780000 },
    { date: 'Jan 16', cost: 19.80, tokens: 650000 },
    { date: 'Jan 17', cost: 14.50, tokens: 480000 },
    { date: 'Jan 18', cost: 8.90, tokens: 290000 },
  ];

  const totalCost = stats?.total_cost || 0;
  const avgCostPerTrace = stats?.trace_count
    ? totalCost / stats.trace_count
    : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Cost Analysis</h1>
          <p className="text-gray-400 mt-1">
            Track and analyze your AI agent costs
          </p>
        </div>

        <div className="flex items-center gap-2">
          <select
            value={dateRange}
            onChange={(e) => setDateRange(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg text-white px-3 py-2 focus:outline-none focus:border-blue-500"
          >
            <option value="24h">Last 24 hours</option>
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
          </select>
        </div>
      </div>

      {/* Summary metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard
          title="Total Cost"
          value={formatCost(totalCost)}
          icon={<Coins className="h-5 w-5" />}
          trend={{ value: 12.5, isPositive: false }}
        />
        <MetricCard
          title="Avg Cost / Trace"
          value={formatCost(avgCostPerTrace)}
          icon={<TrendingUp className="h-5 w-5" />}
        />
        <MetricCard
          title="Total Tokens"
          value={formatTokens(stats?.total_tokens || 0)}
          icon={<TrendingUp className="h-5 w-5" />}
        />
        <MetricCard
          title="Cost Efficiency"
          value={stats?.total_tokens ? `$${(totalCost / stats.total_tokens * 1000).toFixed(4)}/1K` : '-'}
          subtitle="Cost per 1K tokens"
          icon={<TrendingDown className="h-5 w-5" />}
        />
      </div>

      {/* Time series chart */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <h2 className="text-lg font-medium text-white mb-4">Cost Over Time</h2>
        <TimeSeriesChart data={timeSeriesData} />
      </div>

      {/* Breakdown charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cost breakdown */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-medium text-white">Cost Breakdown</h2>
            <select
              value={groupBy}
              onChange={(e) => setGroupBy(e.target.value as GroupBy)}
              className="bg-gray-900 border border-gray-700 rounded text-sm text-white px-2 py-1 focus:outline-none focus:border-blue-500"
            >
              <option value="model">By Model</option>
              <option value="provider">By Provider</option>
              <option value="agent">By Agent</option>
            </select>
          </div>
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500" />
            </div>
          ) : costBreakdown && costBreakdown.length > 0 ? (
            <CostChart data={costBreakdown} type="bar" />
          ) : (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <AlertCircle className="h-8 w-8 text-gray-500 mx-auto mb-2" />
                <p className="text-gray-400">No cost data available</p>
              </div>
            </div>
          )}
        </div>

        {/* Token breakdown */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h2 className="text-lg font-medium text-white mb-4">Token Usage</h2>
          <TokenChart data={tokenBreakdown} />
        </div>
      </div>

      {/* Cost table */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-700">
          <h2 className="text-lg font-medium text-white">Cost Details</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-900">
              <tr>
                <th className="px-4 py-3 text-left text-xs text-gray-400 uppercase">
                  Category
                </th>
                <th className="px-4 py-3 text-right text-xs text-gray-400 uppercase">
                  Input Tokens
                </th>
                <th className="px-4 py-3 text-right text-xs text-gray-400 uppercase">
                  Output Tokens
                </th>
                <th className="px-4 py-3 text-right text-xs text-gray-400 uppercase">
                  Total Cost
                </th>
                <th className="px-4 py-3 text-right text-xs text-gray-400 uppercase">
                  % of Total
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {costBreakdown?.map((item, idx) => (
                <tr key={idx} className="hover:bg-gray-750">
                  <td className="px-4 py-3 text-white">{item.category}</td>
                  <td className="px-4 py-3 text-right text-gray-300">
                    {formatTokens(item.input_tokens || 0)}
                  </td>
                  <td className="px-4 py-3 text-right text-gray-300">
                    {formatTokens(item.output_tokens || 0)}
                  </td>
                  <td className="px-4 py-3 text-right text-green-400">
                    {formatCost(item.cost)}
                  </td>
                  <td className="px-4 py-3 text-right text-gray-400">
                    {totalCost > 0
                      ? ((item.cost / totalCost) * 100).toFixed(1)
                      : 0}
                    %
                  </td>
                </tr>
              ))}
              {(!costBreakdown || costBreakdown.length === 0) && (
                <tr>
                  <td colSpan={5} className="px-4 py-8 text-center text-gray-500">
                    No cost data available
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
