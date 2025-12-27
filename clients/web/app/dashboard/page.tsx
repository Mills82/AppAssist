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

import React, { useState } from 'react';
import Sidebar from '../../components/Sidebar';
import ChatWindow from '../../components/ChatWindow';

type Project = { id: string; name?: string } | null;

export default function Page(): JSX.Element {
  const [selectedProject, setSelectedProject] = useState<Project>(null);

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

  return (
    <div className="min-h-screen h-screen flex bg-gray-50">
      <aside
        aria-label="sidebar"
        className="w-72 max-w-xs border-r border-gray-200 bg-white flex-shrink-0 overflow-y-auto"
      >
        <Sidebar selectedProject={selectedProject ?? undefined} onSelectProject={handleSelectProject} />
      </aside>

      <main
        aria-label="chat-window"
        className="flex-1 flex flex-col overflow-hidden"
      >
        <ChatWindow selectedProject={selectedProject ?? undefined} />
      </main>
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
