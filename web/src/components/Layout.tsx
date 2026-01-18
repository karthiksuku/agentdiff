import { NavLink, Outlet } from 'react-router-dom';
import {
  LayoutDashboard,
  List,
  GitCompare,
  DollarSign,
  Settings,
  Activity,
} from 'lucide-react';
import { cn } from '../lib/utils';

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Traces', href: '/traces', icon: List },
  { name: 'Compare', href: '/compare', icon: GitCompare },
  { name: 'Costs', href: '/costs', icon: DollarSign },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export function Layout() {
  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* Sidebar */}
      <aside className="fixed inset-y-0 left-0 w-64 bg-gray-800 border-r border-gray-700">
        {/* Logo */}
        <div className="flex items-center gap-2 h-16 px-6 border-b border-gray-700">
          <Activity className="h-6 w-6 text-blue-500" />
          <span className="text-xl font-bold">AgentDiff</span>
        </div>

        {/* Navigation */}
        <nav className="p-4 space-y-1">
          {navigation.map((item) => (
            <NavLink
              key={item.name}
              to={item.href}
              end={item.href === '/'}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-400 hover:bg-gray-700 hover:text-white'
                )
              }
            >
              <item.icon className="h-5 w-5" />
              {item.name}
            </NavLink>
          ))}
        </nav>
      </aside>

      {/* Main content */}
      <main className="pl-64">
        <div className="p-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
