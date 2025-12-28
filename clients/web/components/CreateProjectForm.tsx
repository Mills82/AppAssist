/* clients/web/components/CreateProjectForm.tsx
   Simple client-side React component to create a Project (name + description).
   Tries to use a central API client (lib/api) if available, otherwise falls back to a relative fetch to /api/v1/projects.
*/

"use client";

import React, { useState } from "react";

// Minimal Project type (kept intentionally small and compatible with API_CONTRACT.md)
export interface Project {
  id: string;
  name: string;
  description?: string;
  created_at: string;
  // Optional import status used by Projects UI state machine
  importStatus?: "new" | "importing" | "imported" | "failed";
}

export interface CreateProjectFormProps {
  onSuccess?: (project: Project) => void;
  // Temporary: support onCreate in addition to onSuccess during migration.
  // Prefer onSuccess as the canonical callback; call both when present so
  // existing callers using either prop will continue to work.
  onCreate?: (project: Project) => void;
  // Optional hooks for parent components to refresh lists or close modals/popovers.
  // Called after onSuccess/onCreate when a project is successfully created.
  onRefresh?: () => void;
  onClose?: () => void;
  className?: string;
  initialValues?: Partial<{ name: string; description: string }>;
}

export default function CreateProjectForm({
  onSuccess,
  onCreate,
  onRefresh,
  onClose,
  className = "",
  initialValues = {},
}: CreateProjectFormProps) {
  const [name, setName] = useState(initialValues.name ?? "");
  const [description, setDescription] = useState(initialValues.description ?? "");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const resetForm = () => {
    setName("");
    setDescription("");
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setError(null);
    setSuccessMessage(null);

    if (!name.trim()) {
      setError("Project name is required.");
      return;
    }

    setLoading(true);

    try {
      // Try dynamic import of central API client if present
      let project: Project | null = null;

      try {
        // dynamic import so builds don't fail when lib/api isn't present yet
        // Path is relative to this file: components -> ../lib/api
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const apiModule = await import("../lib/api");
        if (apiModule && typeof apiModule.createProject === "function") {
          const result = await apiModule.createProject({ name: name.trim(), description: description.trim() });
          // Accept either { project } or direct project return
          project = (result && (result.project || result)) as Project;
        }
      } catch (err) {
        // ignore dynamic import/import-call errors and fallback to fetch below
      }

      if (!project) {
        // Fallback: call the relative API route
        const resp = await fetch("/api/v1/projects", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: name.trim(), description: description.trim() }),
        });

        if (!resp.ok) {
          // Try to read error message
          let msg = `Request failed: ${resp.status}`;
          try {
            const body = await resp.json();
            if (body && body.error) msg = String(body.error);
          } catch (_err) {
            // ignore
          }
          throw new Error(msg);
        }

        const data = await resp.json();
        // Accept either { ok:true, project } or project directly
        if (data && data.project) project = data.project as Project;
        else project = data as Project;
      }

      if (!project) throw new Error("No project returned from API.");

      setSuccessMessage("Project created");
      resetForm();
      // Call both callbacks if provided to support migration from onCreate -> onSuccess.
      // onSuccess is considered the canonical name going forward.
      onSuccess?.(project);
      onCreate?.(project);
      // Allow parents to refresh project lists and close dialogs/popovers after a successful create.
      onRefresh?.();
      onClose?.();
    } catch (err: any) {
      setError(err?.message ?? String(err) ?? "Failed to create project");
    } finally {
      setLoading(false);
      // clear success message after a short delay
      setTimeout(() => setSuccessMessage(null), 3000);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className={`p-4 bg-white rounded-md shadow-sm max-w-xl w-full ${className}`}
    >
      <h3 className="text-lg font-medium mb-3">Create a project</h3>

      <div className="mb-3">
        <label className="block text-sm font-medium text-gray-700 mb-1" htmlFor="project-name">
          Name
        </label>
        <input
          id="project-name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full rounded-md border-gray-300 shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
          placeholder="My Project"
          disabled={loading}
          aria-required
        />
      </div>

      <div className="mb-3">
        <label className="block text-sm font-medium text-gray-700 mb-1" htmlFor="project-description">
          Description (optional)
        </label>
        <textarea
          id="project-description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          className="w-full rounded-md border-gray-300 shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
          placeholder="A short description about this project"
          rows={3}
          disabled={loading}
        />
      </div>

      {error && <div className="text-sm text-red-600 mb-2">{error}</div>}
      {successMessage && <div className="text-sm text-green-600 mb-2">{successMessage}</div>}

      <div className="flex items-center space-x-2">
        <button
          type="submit"
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 disabled:opacity-60"
          disabled={loading}
        >
          {loading ? "Creating..." : "Create project"}
        </button>

        <button
          type="button"
          onClick={() => {
            resetForm();
            setError(null);
            setSuccessMessage(null);
          }}
          className="inline-flex items-center px-3 py-2 border border-gray-300 text-sm rounded-md bg-white hover:bg-gray-50"
          disabled={loading}
        >
          Reset
        </button>
      </div>

      {/* Small usage example shown in a comment for maintainers:
          <CreateProjectForm onSuccess={(p)=>console.log('created', p)} />
      */}
    </form>
  );
}