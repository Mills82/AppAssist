/*
clients/web/components/Sidebar.tsx

Collapsible left Sidebar used by the dashboard. Client component (uses useState).
Provides: ProjectSelector, CurrentRun status, Approvals badge, RecentRuns list.
*/

"use client";

import React, { useEffect, useState } from "react";

export type RecentRun = {
  id: string;
  title: string;
  status: "running" | "failed" | "success" | "pending";
  time?: string;
};

export interface SidebarProps {
  initialCollapsed?: boolean;
  projects?: string[];
  recentRuns?: RecentRun[];
  approvalsCount?: number;
  currentRun?: { id: string; title: string; status: string } | null;
  // onSelectProject now accepts either a string/id, an object {id,name?}, or null
  onSelectProject?: (project: { id: string; name?: string } | string | null) => void;
  onSelectRun?: (run: RecentRun) => void;
  // optional controlled selected project: either a string (id/name) or an object with id and optional name
  selectedProject?: { id: string; name?: string } | string;
}

const defaultProjects = ["Website Bot", "Infra Upgrade", "Docs Autogen"];
const defaultRecentRuns: RecentRun[] = [
  { id: "r1", title: "Analyze repo for TODOs", status: "success", time: "2h" },
  { id: "r2", title: "Plan migration steps", status: "running", time: "10m" },
  { id: "r3", title: "Create unit tests", status: "pending", time: "1d" },
];

export default function Sidebar({
  initialCollapsed = false,
  projects = defaultProjects,
  recentRuns = defaultRecentRuns,
  approvalsCount = 2,
  currentRun = { id: "r2", title: "Plan migration steps", status: "running" },
  onSelectRun,
  onSelectProject,
  selectedProject: selectedProjectProp,
}: SidebarProps) {
  const [collapsed, setCollapsed] = useState<boolean>(initialCollapsed);

  // Local projects state initialized from prop or defaults; we'll replace with fetched data when available
  const [projectsState, setProjectsState] = useState<string[]>(projects ?? defaultProjects);
  const [isLoadingProjects, setIsLoadingProjects] = useState<boolean>(false);
  const [projectsError, setProjectsError] = useState<string | null>(null);

  // Uncontrolled internal selected project name
  const [internalSelectedProject, setInternalSelectedProject] = useState<string>(projects[0] ?? "Project");

  // Consider the component controlled if a selectedProject prop was provided (either string or object)
  const isControlled = selectedProjectProp !== undefined;

  // Derive a display string for the selection: if the prop is a string, use it; if it's an object, use name ?? id; otherwise fall back to internal selection
  const displayedProject: string = isControlled
    ? typeof selectedProjectProp === "string"
      ? selectedProjectProp
      : selectedProjectProp?.name ?? selectedProjectProp?.id ?? internalSelectedProject
    : internalSelectedProject;

  useEffect(() => {
    let mounted = true;
    const controller = new AbortController();

    async function fetchWorkspaces() {
      setIsLoadingProjects(true);
      setProjectsError(null);
      try {
        // Suggested endpoint: /api/workspaces (adjust if your backend exposes a different path)
        const res = await fetch("/api/workspaces", { signal: controller.signal });
        if (!res.ok) {
          throw new Error(`Failed to fetch workspaces: ${res.status}`);
        }
        const data = await res.json();
        // Defensive parsing: accept array of strings or array of objects with name/id
        if (!mounted) return;
        if (!Array.isArray(data)) {
          throw new Error("Unexpected workspaces response shape");
        }
        const names = data
          .map((item: any) => {
            if (!item && item !== 0) return null;
            if (typeof item === "string") return item;
            if (typeof item === "object") return item.name ?? item.id ?? String(item);
            return String(item);
          })
          .filter(Boolean) as string[];

        if (names.length > 0) {
          setProjectsState(names);
          // if uncontrolled, set internal selection to first workspace when current internal selection isn't present
          if (!isControlled) {
            setInternalSelectedProject((prev) => (names.includes(prev) ? prev : names[0]));
          }
        } else {
          // keep existing fallback projectsState (do not clear to avoid empty UI)
          console.warn("/api/workspaces returned no usable entries; keeping local defaults");
        }
      } catch (err: any) {
        if (err.name === "AbortError") return;
        console.error("Error fetching workspaces:", err);
        if (mounted) setProjectsError(err?.message ?? "Failed to load projects");
      } finally {
        if (mounted) setIsLoadingProjects(false);
      }
    }

    fetchWorkspaces();

    return () => {
      mounted = false;
      controller.abort();
    };
    // We intentionally run this once on mount. Do not include projectsState or isControlled in deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleToggle() {
    setCollapsed((c) => !c);
  }

  function handleProjectChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const p = e.target.value;
    // Keep select option values as strings. Call onSelectProject with the string value so page handlers
    // that expect a string continue to work. Consumers that prefer objects can resolve the id/name themselves.
    if (!isControlled) setInternalSelectedProject(p);
    onSelectProject?.(p);
  }

  function handleRunClick(run: RecentRun) {
    onSelectRun?.(run);
  }

  const containerWidth = collapsed ? "w-20" : "w-64";

  return (
    <aside
      className={`flex flex-col bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 ${containerWidth} transition-all duration-200 shrink-0`}
      aria-label="Sidebar"
    >
      {/* Top: collapse toggle + project selector */}
      <div className="flex items-center justify-between px-3 py-2">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2">
            <div className="text-sm font-semibold text-gray-700 dark:text-gray-200">
              {!collapsed ? "Projects" : "P"}
            </div>
            {!collapsed ? (
              <select
                aria-label="Select project"
                value={displayedProject}
                onChange={handleProjectChange}
                className="bg-transparent text-sm text-gray-700 dark:text-gray-200 focus:outline-none"
              >
                {isLoadingProjects ? (
                  <option disabled>Loading…</option>
                ) : projectsError ? (
                  <option disabled>Failed to load projects</option>
                ) : (
                  projectsState.map((p) => (
                    <option key={p} value={p} className="text-sm">
                      {p}
                    </option>
                  ))
                )}
              </select>
            ) : (
              // collapsed: show small project avatar (first letter)
              <div
                className="h-8 w-8 rounded bg-indigo-500 text-white flex items-center justify-center text-sm font-medium"
                title={displayedProject}
              >
                {displayedProject?.charAt(0) ?? "P"}
              </div>
            )}
          </div>
        </div>

        <button
          onClick={handleToggle}
          aria-expanded={!collapsed}
          aria-controls="sidebar-content"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800"
        >
          {/* simple chevron icon that rotates */}
          <svg
            className={`h-5 w-5 text-gray-600 dark:text-gray-300 transform ${collapsed ? "rotate-180" : "rotate-0"} transition-transform`}
            viewBox="0 0 20 20"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            aria-hidden
          >
            <path d="M6 8l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>

      <div id="sidebar-content" className="px-3 pb-4 overflow-y-auto">
        {/* Current run + approvals */}
        <div className="mt-3 mb-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Current Run</div>
              {!collapsed ? (
                <div className="text-sm font-medium text-gray-800 dark:text-gray-100">{currentRun?.title ?? "—"}</div>
              ) : (
                <div className="text-sm text-gray-800 dark:text-gray-100">{currentRun?.title?.slice(0, 6) ?? "—"}</div>
              )}
            </div>
            <div className="flex items-center gap-2">
              <div className="text-xs text-gray-500 dark:text-gray-400">Approvals</div>
              <div className="bg-red-100 text-red-700 text-xs font-semibold px-2 py-0.5 rounded-full">{approvalsCount}</div>
            </div>
          </div>
          <div className="mt-2">
            <div className="text-xs text-gray-500 dark:text-gray-400">Status</div>
            <div className="mt-1 text-sm text-gray-700 dark:text-gray-200">{currentRun?.status ?? "idle"}</div>
          </div>
        </div>

        {/* Recent runs */}
        <div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">Recent Runs</div>
          <ul className="space-y-2">
            {recentRuns.map((run) => (
              <li key={run.id}>
                <button
                  onClick={() => handleRunClick(run)}
                  className="w-full flex items-center gap-3 text-left rounded p-2 hover:bg-gray-100 dark:hover:bg-gray-800"
                >
                  <div className="h-8 w-8 rounded bg-gray-200 dark:bg-gray-700 flex items-center justify-center text-xs font-medium text-gray-700 dark:text-gray-100">
                    {run.title.charAt(0)}
                  </div>
                  {!collapsed && (
                    <div className="flex-1">
                      <div className="text-sm font-medium text-gray-800 dark:text-gray-100">{run.title}</div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">{run.time} • {run.status}</div>
                    </div>
                  )}
                  {/* status pill visible even when collapsed */}
                  <div className="text-xs">
                    <span
                      className={`px-2 py-0.5 rounded-full text-white text-[10px] ${
                        run.status === "running"
                          ? "bg-yellow-500"
                          : run.status === "success"
                          ? "bg-green-600"
                          : run.status === "failed"
                          ? "bg-red-600"
                          : "bg-gray-500"
                      }`}
                    >
                      {run.status[0].toUpperCase()}
                    </span>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        </div>

        {/* Footer small actions */}
        <div className="mt-6 pt-4 border-t border-gray-100 dark:border-gray-800">
          {!collapsed ? (
            <div className="text-xs text-gray-500 dark:text-gray-400">Shortcuts</div>
          ) : null}
          <div className="mt-2 flex flex-col gap-2">
            <button className="text-sm text-left px-2 py-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800">New Run</button>
            <button className="text-sm text-left px-2 py-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800">Approvals</button>
            <button className="text-sm text-left px-2 py-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800">Settings</button>
          </div>
        </div>
      </div>
    </aside>
  );
}

// Small demo export so other devs can quickly render/test the Sidebar in storybook/dev pages
export function SidebarDemo() {
  return (
    <div className="h-screen">
      <div className="flex">
        <Sidebar />
        <div className="flex-1 p-6">
          <h2 className="text-lg font-semibold">Main content area</h2>
          <p className="text-sm text-gray-600">This area represents the ChatWindow or main dashboard content.</p>
        </div>
      </div>
    </div>
  );
}
