import { useState } from 'react';
import { Save, Server, Database, Key, Bell } from 'lucide-react';

export function Settings() {
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');
  const [darkMode, setDarkMode] = useState(true);
  const [notifications, setNotifications] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30);

  const handleSave = () => {
    // Save settings to localStorage
    localStorage.setItem(
      'agentdiff-settings',
      JSON.stringify({
        apiUrl,
        darkMode,
        notifications,
        autoRefresh,
        refreshInterval,
      })
    );
    alert('Settings saved!');
  };

  return (
    <div className="space-y-6 max-w-2xl">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Settings</h1>
        <p className="text-gray-400 mt-1">Configure your AgentDiff dashboard</p>
      </div>

      {/* API Configuration */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <div className="flex items-center gap-2 mb-4">
          <Server className="h-5 w-5 text-gray-400" />
          <h2 className="text-lg font-medium text-white">API Configuration</h2>
        </div>
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">
              API Base URL
            </label>
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg text-white px-3 py-2 focus:outline-none focus:border-blue-500"
              placeholder="http://localhost:8000"
            />
            <p className="text-xs text-gray-500 mt-1">
              The base URL for the AgentDiff API server
            </p>
          </div>
        </div>
      </div>

      {/* Database */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <div className="flex items-center gap-2 mb-4">
          <Database className="h-5 w-5 text-gray-400" />
          <h2 className="text-lg font-medium text-white">Database</h2>
        </div>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-white">Storage Backend</p>
              <p className="text-xs text-gray-500">Current database type</p>
            </div>
            <span className="text-sm text-blue-400 bg-blue-900/30 px-3 py-1 rounded">
              SQLite
            </span>
          </div>
          <div className="pt-2 border-t border-gray-700">
            <button className="text-sm text-gray-400 hover:text-white">
              View database statistics
            </button>
          </div>
        </div>
      </div>

      {/* Display */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <div className="flex items-center gap-2 mb-4">
          <Bell className="h-5 w-5 text-gray-400" />
          <h2 className="text-lg font-medium text-white">Display & Notifications</h2>
        </div>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-white">Dark Mode</p>
              <p className="text-xs text-gray-500">Use dark theme</p>
            </div>
            <Toggle checked={darkMode} onChange={setDarkMode} />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-white">Notifications</p>
              <p className="text-xs text-gray-500">Show browser notifications</p>
            </div>
            <Toggle checked={notifications} onChange={setNotifications} />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-white">Auto Refresh</p>
              <p className="text-xs text-gray-500">
                Automatically refresh trace list
              </p>
            </div>
            <Toggle checked={autoRefresh} onChange={setAutoRefresh} />
          </div>
          {autoRefresh && (
            <div>
              <label className="block text-sm text-gray-400 mb-2">
                Refresh Interval (seconds)
              </label>
              <input
                type="number"
                value={refreshInterval}
                onChange={(e) => setRefreshInterval(Number(e.target.value))}
                min={5}
                max={300}
                className="w-32 bg-gray-900 border border-gray-700 rounded-lg text-white px-3 py-2 focus:outline-none focus:border-blue-500"
              />
            </div>
          )}
        </div>
      </div>

      {/* API Keys */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <div className="flex items-center gap-2 mb-4">
          <Key className="h-5 w-5 text-gray-400" />
          <h2 className="text-lg font-medium text-white">API Keys</h2>
        </div>
        <p className="text-sm text-gray-400 mb-4">
          API keys are managed through environment variables on the server.
        </p>
        <div className="space-y-2 text-sm">
          <div className="flex items-center justify-between py-2 border-b border-gray-700">
            <span className="text-gray-300">OPENAI_API_KEY</span>
            <span className="text-green-400">Configured</span>
          </div>
          <div className="flex items-center justify-between py-2 border-b border-gray-700">
            <span className="text-gray-300">ANTHROPIC_API_KEY</span>
            <span className="text-gray-500">Not configured</span>
          </div>
          <div className="flex items-center justify-between py-2">
            <span className="text-gray-300">ORACLE_USER</span>
            <span className="text-gray-500">Not configured</span>
          </div>
        </div>
      </div>

      {/* Save button */}
      <button
        onClick={handleSave}
        className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-white transition-colors"
      >
        <Save className="h-4 w-4" />
        Save Settings
      </button>
    </div>
  );
}

function Toggle({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className={`relative w-11 h-6 rounded-full transition-colors ${
        checked ? 'bg-blue-600' : 'bg-gray-600'
      }`}
    >
      <span
        className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
          checked ? 'translate-x-5' : 'translate-x-0'
        }`}
      />
    </button>
  );
}
