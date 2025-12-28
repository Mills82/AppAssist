/*
clients/web/components/Sidebar.tsx

Collapsible left Sidebar used by the dashboard. Client component (uses useState).
Provides: ProjectSelector, CurrentRun status, Approvals badge, RecentRuns list.
*/

"use client";

import React, { useEffect, useMemo, useState } from "react";

type Project = { id: string; name: string };

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
  selectedProject?: { id: string; name?: string } | string | null;
  // explicit flag to indicate the parent is controlling selection (prevents Sidebar from auto-picking a default)
  selectedProjectControlled?: boolean;
  // optional refresh hook to allow parents (e.g., after create) to request a refresh
  onRefresh?: () => void;
}

const defaultProjects = ["Website Bot", "Infra Upgrade", "Docs Autogen"];
const defaultRecentRuns: RecentRun[] = [
  { id: "r1", title: "Analyze repo for TODOs", status: "success", time: "2h" },
  { id: "r2", title: "Plan migration steps", status: "running", time: "10m" },
  { id: "r3", title: "Create unit tests", status: "pending", time: "1d" },
];

function normalizeProjects(input: any): Project[] {
  if (!Array.isArray(input)) return [];
  const out: Project[] = [];
  for (const item of input) {
    if (!item && item !== 0) continue;
    if (typeof item === "string") {
      out.push({ id: item, name: item });
      continue;
    }
    if (typeof item === "object") {
      const nameRaw = (item as any).name ?? (item as any).title;
      const idRaw = (item as any).id ?? nameRaw;
      const id = idRaw != null ? String(idRaw) : null;
      const name = nameRaw != null ? String(nameRaw) : id != null ? String(id) : null;
      if (id && name) out.push({ id, name });
      continue;
    }
    const s = String(item);
    out.push({ id: s, name: s });
  }
  // de-dupe by id while preserving order
  const seen = new Set<string>();
  return out.filter((p) => {
    if (seen.has(p.id)) return false;
    seen.add(p.id);
    return true;
  });
}

function normalizeProjectsFromStrings(strings: string[]): Project[] {
  return (strings ?? []).map((s) => ({ id: s, name: s }));
}

export default function Sidebar({
  initialCollapsed = false,
  projects = defaultProjects,
  recentRuns = defaultRecentRuns,
  approvalsCount = 2,
  currentRun = { id: "r2", title: "Plan migration steps", status: "running" },
  onSelectRun,
  onSelectProject,
  selectedProject: selectedProjectProp,
  selectedProjectControlled,
  onRefresh,
}: SidebarProps) {
  const [collapsed, setCollapsed] = useState<boolean>(initialCollapsed);

  // Local projects state initialized from prop or defaults; we'll replace with fetched data when available
  const [projectsState, setProjectsState] = useState<Project[]>(normalizeProjectsFromStrings(projects ?? defaultProjects));
  const [isLoadingProjects, setIsLoadingProjects] = useState<boolean>(false);
  const [projectsError, setProjectsError] = useState<string | null>(null);

  // Consider the component controlled if the explicit flag is present, or if the parent passed a value (including null)
  // This ensures that when the parent owns selection and passes `null` to mean "no project selected",
  // Sidebar will not auto-select a default project on mount.
  const isControlled = selectedProjectControlled ?? (selectedProjectProp !== undefined || selectedProjectProp === null);

  // Uncontrolled internal selected project id. If the component is controlled, do not auto-select a default on mount.
  const [internalSelectedProjectId, setInternalSelectedProjectId] = useState<string>(() => {
    if (isControlled) return "";
    const initial = (projects?.[0] ?? defaultProjects[0] ?? "Project").toString();
    return initial;
  });

  const selectedProjectId: string | null = useMemo(() => {
    if (!isControlled) return internalSelectedProjectId;
    if (typeof selectedProjectProp === "string") return selectedProjectProp;
    return selectedProjectProp?.id ?? null;
  }, [internalSelectedProjectId, isControlled, selectedProjectProp]);

  const selectedProjectObj: Project | null = useMemo(() => {
    if (!selectedProjectId) return null;
    return projectsState.find((p) => p.id === selectedProjectId) ?? null;
  }, [projectsState, selectedProjectId]);

  const displayedProjectName: string = useMemo(() => {
    // Show an explicit hint when controlled but no project is selected
    if (isControlled && (selectedProjectProp == null || selectedProjectId == null || selectedProjectId === "")) {
      return "Select a project";
    }

    if (isControlled) {
      if (typeof selectedProjectProp === "string") {
        // Could be id or name; if it matches a known id, show name.
        const found = projectsState.find((p) => p.id === selectedProjectProp);
        return found?.name ?? selectedProjectProp;
      }
      return selectedProjectProp?.name ?? selectedProjectObj?.name ?? selectedProjectProp?.id ?? "Project";
    }
    return selectedProjectObj?.name ?? internalSelectedProjectId ?? "Project";
  }, [internalSelectedProjectId, isControlled, projectsState, selectedProjectObj, selectedProjectProp, selectedProjectId]);

  async function fetchWorkspaces(signal?: AbortSignal) {
    setIsLoadingProjects(true);
    setProjectsError(null);
    try {
      // Suggested endpoint: /api/workspaces (adjust if your backend exposes a different path)
      const res = await fetch("/api/workspaces", { signal });
      if (!res.ok) {
        throw new Error(`Failed to fetch workspaces: ${res.status}`);
      }
      const data = await res.json();

      const parsed = normalizeProjects(data);
      if (parsed.length > 0) {
        setProjectsState(parsed);

        // If uncontrolled, select first project when current selection isn't present.
        if (!isControlled) {
          setInternalSelectedProjectId((prev) => (parsed.some((p) => p.id === prev) ? prev : parsed[0].id));
        }
      } else {
        // keep existing fallback projectsState (do not clear to avoid empty UI)
        console.warn("/api/workspaces returned no usable entries; keeping local defaults");
      }
    } catch (err: any) {
      if (err?.name === "AbortError") return;
      console.error("Error fetching workspaces:", err);
      setProjectsError(err?.message ?? "Failed to load projects");
    } finally {
      setIsLoadingProjects(false);
    }
  }

  useEffect(() => {
    const controller = new AbortController();
    fetchWorkspaces(controller.signal);
    return () => controller.abort();
    // We intentionally run this once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleToggle() {
    setCollapsed((c) => !c);
  }

  function handleProjectChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const projectId = e.target.value;
    if (!isControlled) setInternalSelectedProjectId(projectId);

    const project = projectsState.find((p) => p.id === projectId);
    // Prefer passing a Project object when available; still compatible with callers expecting a string.
    onSelectProject?.(project ?? projectId);
  }

  function handleRunClick(run: RecentRun) {
    onSelectRun?.(run);
  }

  function handleRefreshProjects() {
    onRefresh?.();
    fetchWorkspaces();
  }

  function handleCreateProjectHint() {
    // Sidebar cannot create projects directly; hint the parent flow.
    // We intentionally call onSelectProject(null) so the parent can open the create flow or clear selection.
    onSelectProject?.(null);
    onRefresh?.();
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
            <div className="text-sm font-semibold text-gray-700 dark:text-gray-200">{!collapsed ? "Projects" : "P"}</div>

            {!collapsed ? (
              <div className="flex items-center gap-2">
                <select
                  aria-label="Select project"
                  value={selectedProjectId ?? ""}
                  onChange={handleProjectChange}
                  className="bg-transparent text-sm text-gray-700 dark:text-gray-200 focus:outline-none"
                >
                  {isLoadingProjects ? (
                    <option disabled value="">
                      Loading…
                    </option>
                  ) : projectsError ? (
                    <option disabled value="">
                      Failed to load projects
                    </option>
                  ) : projectsState.length === 0 ? (
                    <option disabled value="">
                      No projects — create one
                    </option>
                  ) : (
                    // When controlled and no project is selected, show a placeholder to instruct the user
                    <>
                      {isControlled && (selectedProjectId == null || selectedProjectId === "") ? (
                        <option value="" disabled>
                          Select a project
                        </option>
                      ) : null}
                      {projectsState.map((p) => (
                        <option key={p.id} value={p.id} className="text-sm">
                          {p.name}
                        </option>
                      ))}
                    </>
                  )}
                </select>

                <button
                  type="button"
                  onClick={handleRefreshProjects}
                  disabled={isLoadingProjects}
                  className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-50"
                  title="Refresh projects"
                  aria-label="Refresh projects"
                >
                  <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4 text-gray-600 dark:text-gray-300" aria-hidden>
                    <path
                      d="M16 10a6 6 0 0 1-10.392 3.978M4 10a6 6 0 0 1 10.392-3.978"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <path
                      d="M6 14H3v3M14 6h3V3"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </button>

                {projectsState.length === 0 && !isLoadingProjects && !projectsError ? (
                  <button
                    type="button"
                    onClick={handleCreateProjectHint}
                    className="text-xs px-2 py-1 rounded bg-indigo-600 text-white hover:bg-indigo-500"
                    title="Create a project"
                  >
                    Create
                  </button>
                ) : null}
              </div>
            ) : (
              // collapsed: show small project avatar (first letter)
              <div
                className="h-8 w-8 rounded bg-indigo-500 text-white flex items-center justify-center text-sm font-medium"
                title={displayedProjectName}
              >
                {displayedProjectName?.charAt(0) ?? "P"}
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
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        {run.time} • {run.status}
                      </div>
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
          {!collapsed ? <div className="text-xs text-gray-500 dark:text-gray-400">Shortcuts</div> : null}
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
