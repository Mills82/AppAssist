"use client";

// clients/web/components/ProjectList.tsx
// Presentational ProjectList component for AppAssist projects page.
// Renders loading / empty / list states and exposes per-project action callbacks.

import React from "react";

/**
 * Minimal import status union used by the projects page.
 * If you have a shared type file, replace this local definition with that import.
 */
export type ImportStatus =
  | "not_connected"
  | "connected"
  | "importing"
  | "imported"
  | "failed";

/** Project shape expected by ProjectList (keeps component presentational). */
export type Project = {
  id: string;
  name: string;
  description?: string;
  importStatus?: ImportStatus;
  repoUrl?: string;
  createdAt?: string; // ISO timestamp
};

export interface ProjectListProps {
  projects: Project[];
  loading?: boolean;
  onView?: (id: string) => void;
  onDelete?: (id: string) => void;
  onRetryImport?: (id: string) => void;
}

const formatDate = (iso?: string) => {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
};

function renderStatusBadge(status?: ImportStatus) {
  const base = "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium";
  switch (status) {
    case "connected":
      return <span className={`${base} bg-blue-100 text-blue-800`}>Connected</span>;
    case "importing":
      return <span className={`${base} bg-yellow-100 text-yellow-800`}>Importing</span>;
    case "imported":
      return <span className={`${base} bg-green-100 text-green-800`}>Imported</span>;
    case "failed":
      return <span className={`${base} bg-red-100 text-red-800`}>Failed</span>;
    case "not_connected":
    default:
      return <span className={`${base} bg-gray-100 text-gray-800`}>Not connected</span>;
  }
}

/**
 * ProjectList
 * Pure presentational component: parent manages data & side-effects.
 */
export default function ProjectList({
  projects,
  loading = false,
  onView,
  onDelete,
  onRetryImport,
}: ProjectListProps) {
  if (loading) {
    // Simple loading skeleton
    return (
      <div className="space-y-3" data-testid="projects-loading">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="animate-pulse flex items-center justify-between bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm"
          >
            <div className="flex-1 space-y-2">
              <div className="h-4 bg-gray-200 rounded w-1/3" />
              <div className="h-3 bg-gray-100 rounded w-1/2" />
            </div>
            <div className="w-32 ml-4">
              <div className="h-8 bg-gray-200 rounded" />
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (!projects || projects.length === 0) {
    return (
      <div
        className="text-center py-12 bg-white dark:bg-slate-800 rounded-lg shadow-sm"
        data-testid="projects-empty"
      >
        <p className="text-lg font-medium">No projects yet</p>
        <p className="mt-2 text-sm text-gray-500">Create a project or connect your GitHub to import repositories.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3" data-testid="projects-list">
      {projects.map((p) => (
        <div
          key={p.id}
          data-testid={`project-row-${p.id}`}
          className="flex items-center justify-between bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm"
        >
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-3">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-semibold truncate">{p.name}</h3>
                  <span data-testid={`project-status-${p.id}`}>{renderStatusBadge(p.importStatus)}</span>
                </div>
                {p.description ? (
                  <p className="mt-1 text-xs text-gray-500 truncate">{p.description}</p>
                ) : null}
                <div className="mt-2 text-xs text-gray-400 flex items-center gap-3">
                  <span title={p.repoUrl || ""} className="truncate">
                    {p.repoUrl ? (
                      <a href={p.repoUrl} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">
                        {p.repoUrl}
                      </a>
                    ) : (
                      <span className="text-gray-400">No repo connected</span>
                    )}
                  </span>
                  <span className="ml-2">Created: {formatDate(p.createdAt)}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="ml-4 flex-shrink-0 flex items-center space-x-2">
            <button
              onClick={() => onView?.(p.id)}
              className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded text-gray-800"
              data-testid={`project-view-${p.id}`}
              aria-label={`View ${p.name}`}
            >
              View
            </button>

            {p.importStatus === "failed" ? (
              <button
                onClick={() => onRetryImport?.(p.id)}
                className="px-3 py-1 text-sm bg-yellow-100 hover:bg-yellow-200 rounded text-yellow-800"
                data-testid={`project-retry-${p.id}`}
                aria-label={`Retry import ${p.name}`}
              >
                Retry
              </button>
            ) : p.importStatus === "not_connected" ? (
              <button
                onClick={() => onRetryImport?.(p.id)}
                className="px-3 py-1 text-sm bg-blue-100 hover:bg-blue-200 rounded text-blue-800"
                data-testid={`project-import-${p.id}`}
                aria-label={`Import ${p.name}`}
              >
                Import
              </button>
            ) : null}

            <button
              onClick={() => onDelete?.(p.id)}
              className="px-3 py-1 text-sm bg-red-50 hover:bg-red-100 rounded text-red-700"
              data-testid={`project-delete-${p.id}`}
              aria-label={`Delete ${p.name}`}
            >
              Delete
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

// Minimal usage example (can be rendered inside a page for quick local dev/demo).
export function DemoProjectList() {
  const sample: Project[] = [
    {
      id: "proj_1",
      name: "Landing Page",
      description: "Marketing landing page with analytics",
      importStatus: "imported",
      repoUrl: "https://github.com/example/landing",
      createdAt: new Date().toISOString(),
    },
    {
      id: "proj_2",
      name: "Backend API",
      description: "Auth + billing API",
      importStatus: "importing",
      repoUrl: "https://github.com/example/api",
      createdAt: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
    },
    {
      id: "proj_3",
      name: "Mobile App",
      importStatus: "not_connected",
      createdAt: new Date(Date.now() - 1000 * 60 * 60 * 48).toISOString(),
    },
  ];

  return (
    <div className="p-4 max-w-3xl mx-auto">
      <h2 className="text-lg font-semibold mb-4">Demo Projects</h2>
      <ProjectList
        projects={sample}
        onView={(id) => console.log("view", id)}
        onDelete={(id) => console.log("delete", id)}
        onRetryImport={(id) => console.log("retry import", id)}
      />
    </div>
  );
}

