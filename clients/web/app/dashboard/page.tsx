'use client';
/* clients/web/app/dashboard/page.tsx */
/**
 * /app/dashboard page - chat-first dashboard entrypoint
 *
 * Composes the Sidebar (left) and ChatWindow (main) components.
 * This page is now a client component so it can hold selection state and
 * forward the selected project to the client children.
 *
 * See clients/web/API_CONTRACT.md for the backend contract and aidev/routes/frontend.py
 * for local dev stub endpoints.
 */

import React, { useEffect, useState } from 'react';
import Sidebar from '../../components/Sidebar';
import ChatWindow from '../../components/ChatWindow';
import CreateProjectForm from '../../components/CreateProjectForm';

type Project = { id: string; name?: string } | null;

// Session storage key used to persist the active project selection across reloads
// within the same browser tab/session.
export const DASHBOARD_SELECTED_PROJECT_STORAGE_KEY = 'dashboard:selectedProject';

export default function Page(): JSX.Element {
  const [selectedProject, setSelectedProject] = useState<Project>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [projectsRefreshCounter, setProjectsRefreshCounter] = useState(0);

  // Hydrate selectedProject from sessionStorage (client-only) to avoid SSR/runtime errors.
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(DASHBOARD_SELECTED_PROJECT_STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object' && typeof parsed.id === 'string') {
        setSelectedProject({ id: parsed.id, name: typeof parsed.name === 'string' ? parsed.name : undefined });
      }
    } catch {
      // Ignore malformed storage data.
    }
  }, []);

  // Persist selection whenever it changes (store minimal safe shape).
  useEffect(() => {
    try {
      sessionStorage.setItem(
        DASHBOARD_SELECTED_PROJECT_STORAGE_KEY,
        JSON.stringify(selectedProject ? { id: selectedProject.id, name: selectedProject.name } : null),
      );
    } catch {
      // Ignore quota/security errors.
    }
  }, [selectedProject]);

  // onSelectProject accepts either a Project object or a string id for convenience.
  const handleSelectProject = (p: Project | string) => {
    if (!p) {
      setSelectedProject(null);
      return;
    }
    if (typeof p === 'string') {
      setSelectedProject({ id: p });
    } else {
      setSelectedProject(p);
    }
  };

  const triggerProjectsRefresh = () => setProjectsRefreshCounter((c) => c + 1);

  // Called when a project is successfully created in the CreateProjectForm.
  // This immediately selects the created project (single source of truth) and
  // closes the creation modal. It also triggers a projects refresh so Sidebar
  // can pick up the new project in its list if it supports onRefresh.
  const handleProjectCreated = (proj: { id: string; name?: string }) => {
    if (!proj || typeof proj.id !== 'string') return;
    setSelectedProject({ id: proj.id, name: proj.name });
    triggerProjectsRefresh();
    setShowCreateModal(false);
  };

  return (
    <div className="min-h-screen h-screen flex bg-gray-50">
      <aside
        aria-label="sidebar"
        className="w-72 max-w-xs border-r border-gray-200 bg-white flex-shrink-0 overflow-y-auto"
      >
        {/* Minimal header + create button to open project creation flow. */}
        <div className="p-3 border-b border-gray-100 flex items-center justify-between">
          <h2 className="text-sm font-medium text-gray-700">Projects</h2>
          <button
            type="button"
            onClick={() => setShowCreateModal(true)}
            className="text-sm text-blue-600 hover:underline"
            aria-label="Create project"
          >
            Create
          </button>
        </div>

        {/*
          Pass selectedProject as a controlled prop even when null to ensure
          Sidebar does not treat the prop as uncontrolled (avoid passing undefined).
          Also provide an onRefresh callback so Sidebar can refresh its list if it
          implements that optional interface.
        */}
        <Sidebar
          selectedProject={selectedProject}
          onSelectProject={handleSelectProject}
          onRefresh={triggerProjectsRefresh}
          projectsRefreshCounter={projectsRefreshCounter}
        />
      </aside>

      <main
        aria-label="chat-window"
        className="flex-1 flex flex-col overflow-hidden"
      >
        {/*
          Pass selectedProject as a controlled prop (nullable). ChatWindow should
          block sending when this prop is null and include selectedProject.id in
          conversation payloads when present.
        */}
        <ChatWindow selectedProject={selectedProject} />
      </main>

      {/* Simple modal for project creation. CreateProjectForm should call onSuccess/onCreate when created. */}
      {showCreateModal ? (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-start justify-center pt-20 px-4"
        >
          <div className="absolute inset-0 bg-black/40" onClick={() => setShowCreateModal(false)} />
          <div className="relative z-10 w-full max-w-lg bg-white rounded shadow-lg overflow-hidden">
            <div className="p-4 border-b flex items-center justify-between">
              <h3 className="text-sm font-medium">Create project</h3>
              <button
                aria-label="Close"
                className="text-sm text-gray-500"
                onClick={() => setShowCreateModal(false)}
              >
                Close
              </button>
            </div>
            <div className="p-4">
              <CreateProjectForm
                // Support both common callback names to maximize compatibility.
                onSuccess={handleProjectCreated}
                onCreate={handleProjectCreated}
                onClose={() => setShowCreateModal(false)}
                onRefresh={triggerProjectsRefresh}
              />
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

/**
 * Exported helper for unit tests / storybook usage.
 * Tests can import { _renderForTest } from './page' to get the JSX element.
 */
export function _renderForTest() {
  return <Page />;
}
