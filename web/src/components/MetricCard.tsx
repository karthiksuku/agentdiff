import { cn } from '../lib/utils';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  className?: string;
}

export function MetricCard({
  title,
  value,
  subtitle,
  icon,
  trend,
  className,
}: MetricCardProps) {
  return (
    <div
      className={cn(
        'bg-gray-800 rounded-lg border border-gray-700 p-4',
        className
      )}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-400">{title}</p>
          <p className="text-2xl font-semibold text-white mt-1">{value}</p>
          {subtitle && (
            <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
          )}
          {trend && (
            <div
              className={cn(
                'flex items-center gap-1 text-xs mt-2',
                trend.isPositive ? 'text-green-400' : 'text-red-400'
              )}
            >
              <span>{trend.isPositive ? '↑' : '↓'}</span>
              <span>{Math.abs(trend.value)}%</span>
              <span className="text-gray-500">vs last period</span>
            </div>
          )}
        </div>
        {icon && (
          <div className="p-2 bg-gray-700/50 rounded-lg text-gray-400">
            {icon}
          </div>
        )}
      </div>
    </div>
  );
}
