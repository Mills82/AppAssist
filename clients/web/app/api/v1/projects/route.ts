// clients/web/app/api/v1/projects/route.ts
// Dev-only Next.js App Router route handlers for projects.
// Provides an in-memory, session-scoped projects store for frontend dev & testing.

import { NextResponse } from 'next/server'
import type { Project } from '../../../../lib/types'

// Module-scoped store: Map<sessionToken, Project[]>
// NOTE: This is ephemeral and only lives for the running dev server process.
const projectsStore: Map<string, Project[]> = new Map()

function parseCookies(cookieHeader: string | null): Record<string, string> {
  const out: Record<string, string> = {}
  if (!cookieHeader) return out
  const pairs = cookieHeader.split(';')
  for (const p of pairs) {
    const idx = p.indexOf('=')
    if (idx === -1) continue
    const k = p.slice(0, idx).trim()
    const v = p.slice(idx + 1).trim()
    out[k] = decodeURIComponent(v)
  }
  return out
}

/**
 * Extract a session token from Authorization Bearer, x-session-token header, or cookies.
 * Falls back to 'dev-session' so the frontend works without explicit auth.
 */
function getSessionToken(req: Request): string {
  const auth = req.headers.get('authorization')
  if (auth && auth.toLowerCase().startsWith('bearer ')) {
    return auth.slice(7).trim()
  }
  const headerToken = req.headers.get('x-session-token')
  if (headerToken) return headerToken

  const cookies = parseCookies(req.headers.get('cookie'))
  if (cookies.session_token) return cookies.session_token
  if (cookies['session-token']) return cookies['session-token']

  // Dev default token so UI works without signing in
  return 'dev-session'
}

/**
 * Seed a couple of example projects for UX/demo purposes.
 */
function seedProjectsFor(token: string): Project[] {
  const now = new Date().toISOString()
  const seeds: Project[] = [
    {
      id: `proj_seed_1`,
      name: 'Example Project Alpha',
      description: 'A seeded demo project to show the projects list.',
      createdAt: now,
      ownerToken: token,
    },
    {
      id: `proj_seed_2`,
      name: 'Imported Repo: demo-repo',
      description: 'Pretend this was imported from GitHub.',
      createdAt: now,
      ownerToken: token,
    },
  ]
  projectsStore.set(token, seeds)
  return seeds
}

export async function GET(req: Request) {
  try {
    const url = new URL(req.url)
    const seed = url.searchParams.get('seed') === 'true'
    const token = getSessionToken(req)

    const existing = projectsStore.get(token) || []
    if (seed && existing.length === 0) {
      const seeded = seedProjectsFor(token)
      return NextResponse.json(seeded)
    }

    return NextResponse.json(existing)
  } catch (err) {
    return NextResponse.json({ error: 'Failed to list projects' }, { status: 500 })
  }
}

export async function POST(req: Request) {
  const token = getSessionToken(req)
  let body: any
  try {
    body = await req.json()
  } catch (err) {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  if (!body || typeof body.name !== 'string' || body.name.trim() === '') {
    return NextResponse.json({ error: 'Missing required field: name' }, { status: 400 })
  }

  const id = `proj_${Date.now()}_${Math.floor(Math.random() * 100000)}`
  const project: Project = {
    id,
    name: String(body.name).trim(),
    description: body.description ? String(body.description).trim() : undefined,
    createdAt: new Date().toISOString(),
    ownerToken: token,
  }

  const arr = projectsStore.get(token) || []
  arr.push(project)
  projectsStore.set(token, arr)

  return NextResponse.json(project, { status: 201 })
}
