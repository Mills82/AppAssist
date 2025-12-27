"use client"

import React, { useEffect, useState } from 'react'
import { listProjects as getProjects, createProject } from '../../lib/api'
import type { Project } from '../../lib/api'
import ProjectList from '../../components/ProjectList'
import CreateProjectForm from '../../components/CreateProjectForm'
import GithubConnectButton from '../../components/GithubConnectButton'

/**
 * clients/web/app/projects/page.tsx
 *
 * Projects page (App Router, client component).
 * Composes CreateProjectForm, GithubConnectButton and ProjectList.
 * Fetches projects from the centralized API client and refreshes on create/import.
 */
export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[] | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [toast, setToast] = useState<string | null>(null)

  useEffect(() => {
    fetchProjects()
  }, [])

  async function fetchProjects() {
    setLoading(true)
    setError(null)
    try {
      const res = await getProjects()
      // Expecting getProjects() to return Project[]
      setProjects(res || [])
    } catch (err: any) {
      console.error('Failed to load projects', err)
      setError(err?.message || 'Failed to load projects')
      setProjects([])
    } finally {
      setLoading(false)
    }
  }

  // CreateProjectForm now calls `onSuccess` with the created Project (or the submitted payload).
  // Accept either a Project (created) or the create payload and handle both.
  async function handleCreate(payload: Project | { name: string; description?: string }) {
    try {
      let created: Project
      // If the form already returned a created project (has an id), don't call the API again.
      if (payload && 'id' in payload) {
        created = payload as Project
      } else {
        // Otherwise call the create API with the provided payload
        created = await createProject(payload as { name: string; description?: string })
      }

      setToast('Project created')
      // Refresh list to pick up server-side state (stubbed backend)
      await fetchProjects()
    } catch (err: any) {
      console.error('Create project failed', err)
      setToast('Failed to create project')
    } finally {
      // auto clear toast
      window.setTimeout(() => setToast(null), 3000)
    }
  }

  function handleImportComplete(message?: string) {
    setToast(message || 'Import completed')
    // Re-fetch projects so imported repos show up
    fetchProjects()
    window.setTimeout(() => setToast(null), 3000)
  }

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold">Projects</h1>
        <p className="text-sm text-gray-600">Create projects or connect/import from GitHub.</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <aside className="md:col-span-1 space-y-4">
          <div className="bg-white p-4 rounded shadow">
            <h2 className="font-medium mb-2">Create a project</h2>
            <CreateProjectForm onSuccess={handleCreate} />
          </div>

          <div className="bg-white p-4 rounded shadow">
            <h2 className="font-medium mb-2">Import from GitHub</h2>
            <p className="text-sm text-gray-600 mb-3">Connect to GitHub and import repositories into your workspace.</p>
            <GithubConnectButton onImportComplete={() => handleImportComplete('Imported from GitHub')} />
          </div>

          {toast && (
            <div className="bg-emerald-50 border border-emerald-200 text-emerald-800 p-2 rounded">{toast}</div>
          )}
        </aside>

        <main className="md:col-span-2">
          <div className="bg-white p-4 rounded shadow">
            <h2 className="font-medium mb-4">Your projects</h2>

            {loading && (
              <div className="text-sm text-gray-600">Loading projects...</div>
            )}

            {error && (
              <div className="text-sm text-red-600">Error: {error}</div>
            )}

            {!loading && !error && projects && projects.length === 0 && (
              <div className="p-6 text-center text-gray-600">
                <p className="mb-2">No projects yet.</p>
                <p className="text-sm">Create your first project or import repositories from GitHub.</p>
              </div>
            )}

            {!loading && !error && projects && projects.length > 0 && (
              <ProjectList projects={projects} onRetryImport={fetchProjects} />
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
