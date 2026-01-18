import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts';
import type { CostBreakdownItem } from '../lib/api';

interface CostChartProps {
  data: CostBreakdownItem[];
  type?: 'bar' | 'pie';
}

const COLORS = [
  '#3b82f6', // blue
  '#10b981', // green
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // violet
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#f97316', // orange
];

export function CostChart({ data, type = 'bar' }: CostChartProps) {
  // Transform data for recharts compatibility
  const chartData = data.map(item => ({
    ...item,
    category: item.category,
    cost: item.cost,
  }));

  if (type === 'pie') {
    return (
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={chartData}
            dataKey="cost"
            nameKey="category"
            cx="50%"
            cy="50%"
            outerRadius={100}
            label={({ name, percent }) =>
              `${name || ''}: ${((percent || 0) * 100).toFixed(0)}%`
            }
          >
            {chartData.map((_, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #374151',
              borderRadius: '0.5rem',
            }}
            formatter={(value) => [`$${Number(value || 0).toFixed(4)}`, 'Cost']}
          />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="category" stroke="#9ca3af" fontSize={12} />
        <YAxis
          stroke="#9ca3af"
          fontSize={12}
          tickFormatter={(value) => `$${Number(value).toFixed(2)}`}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1f2937',
            border: '1px solid #374151',
            borderRadius: '0.5rem',
          }}
          formatter={(value) => [`$${Number(value || 0).toFixed(4)}`, 'Cost']}
        />
        <Bar dataKey="cost" fill="#3b82f6" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

interface TokenChartProps {
  data: { category: string; input: number; output: number }[];
}

export function TokenChart({ data }: TokenChartProps) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="category" stroke="#9ca3af" fontSize={12} />
        <YAxis
          stroke="#9ca3af"
          fontSize={12}
          tickFormatter={(value) => Number(value).toLocaleString()}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1f2937',
            border: '1px solid #374151',
            borderRadius: '0.5rem',
          }}
          formatter={(value) => [Number(value || 0).toLocaleString(), '']}
        />
        <Legend />
        <Bar
          dataKey="input"
          name="Input Tokens"
          fill="#3b82f6"
          radius={[4, 4, 0, 0]}
        />
        <Bar
          dataKey="output"
          name="Output Tokens"
          fill="#10b981"
          radius={[4, 4, 0, 0]}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}

interface TimeSeriesChartProps {
  data: { date: string; cost: number; tokens: number }[];
}

export function TimeSeriesChart({ data }: TimeSeriesChartProps) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="date" stroke="#9ca3af" fontSize={12} />
        <YAxis
          yAxisId="left"
          stroke="#9ca3af"
          fontSize={12}
          tickFormatter={(value) => `$${Number(value).toFixed(2)}`}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          stroke="#9ca3af"
          fontSize={12}
          tickFormatter={(value) => `${(Number(value) / 1000).toFixed(0)}k`}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1f2937',
            border: '1px solid #374151',
            borderRadius: '0.5rem',
          }}
        />
        <Legend />
        <Bar
          yAxisId="left"
          dataKey="cost"
          name="Cost"
          fill="#10b981"
          radius={[4, 4, 0, 0]}
        />
        <Bar
          yAxisId="right"
          dataKey="tokens"
          name="Tokens"
          fill="#3b82f6"
          radius={[4, 4, 0, 0]}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
