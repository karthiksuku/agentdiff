import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { Search, Filter, SortAsc, SortDesc, AlertCircle } from 'lucide-react';
import { api } from '../lib/api';
import { TraceCard } from '../components/TraceCard';
import { cn } from '../lib/utils';

type SortField = 'start_time' | 'total_cost' | 'total_tokens';
type SortOrder = 'asc' | 'desc';

export function TraceList() {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [sortField, setSortField] = useState<SortField>('start_time');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');

  const { data: traces, isLoading, error } = useQuery({
    queryKey: ['traces', { search, statusFilter, sortField, sortOrder }],
    queryFn: () =>
      api.listTraces({
        limit: 50,
        name: search || undefined,
        status: statusFilter !== 'all' ? statusFilter : undefined,
      }),
  });

  const sortedTraces = traces
    ? [...traces].sort((a, b) => {
        let comparison = 0;
        switch (sortField) {
          case 'start_time':
            comparison = new Date(a.start_time).getTime() - new Date(b.start_time).getTime();
            break;
          case 'total_cost':
            comparison = a.total_cost - b.total_cost;
            break;
          case 'total_tokens':
            comparison = a.total_tokens - b.total_tokens;
            break;
        }
        return sortOrder === 'desc' ? -comparison : comparison;
      })
    : [];

  const filteredTraces = search
    ? sortedTraces.filter(
        (t) =>
          t.name.toLowerCase().includes(search.toLowerCase()) ||
          t.trace_id.toLowerCase().includes(search.toLowerCase())
      )
    : sortedTraces;

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('desc');
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Traces</h1>
        <p className="text-gray-400 mt-1">View and manage all agent traces</p>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        {/* Search */}
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-500" />
          <input
            type="text"
            placeholder="Search traces..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
          />
        </div>

        {/* Status filter */}
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-gray-500" />
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg text-white px-3 py-2 focus:outline-none focus:border-blue-500"
          >
            <option value="all">All Status</option>
            <option value="success">Success</option>
            <option value="error">Error</option>
            <option value="running">Running</option>
          </select>
        </div>

        {/* Sort buttons */}
        <div className="flex items-center gap-2">
          <SortButton
            label="Time"
            field="start_time"
            currentField={sortField}
            order={sortOrder}
            onClick={() => toggleSort('start_time')}
          />
          <SortButton
            label="Cost"
            field="total_cost"
            currentField={sortField}
            order={sortOrder}
            onClick={() => toggleSort('total_cost')}
          />
          <SortButton
            label="Tokens"
            field="total_tokens"
            currentField={sortField}
            order={sortOrder}
            onClick={() => toggleSort('total_tokens')}
          />
        </div>
      </div>

      {/* Traces list */}
      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
        </div>
      ) : error ? (
        <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-8 text-center">
          <AlertCircle className="h-8 w-8 text-red-400 mx-auto mb-3" />
          <p className="text-red-300">Error loading traces</p>
        </div>
      ) : filteredTraces.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {filteredTraces.map((trace) => (
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
          {search && (
            <p className="text-sm text-gray-500 mt-1">
              Try adjusting your search or filters
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function SortButton({
  label,
  field,
  currentField,
  order,
  onClick,
}: {
  label: string;
  field: SortField;
  currentField: SortField;
  order: SortOrder;
  onClick: () => void;
}) {
  const isActive = field === currentField;

  return (
    <button
      onClick={onClick}
      className={cn(
        'flex items-center gap-1 px-3 py-2 rounded-lg text-sm transition-colors',
        isActive
          ? 'bg-blue-600 text-white'
          : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
      )}
    >
      {label}
      {isActive &&
        (order === 'desc' ? (
          <SortDesc className="h-3 w-3" />
        ) : (
          <SortAsc className="h-3 w-3" />
        ))}
    </button>
  );
}
