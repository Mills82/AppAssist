/* clients/web/app/api/v1/projects/import/route.ts

Dev-only Next.js App Router route handlers for project imports.
- POST: create a deterministic, in-memory import job and return { importId, status: 'queued' }.
- GET: poll job status by importId; each GET advances a deterministic state machine: queued -> importing -> done.

This file is intentionally in-memory and non-persistent (resets on server restart). It accepts an optional session token via Authorization: Bearer <token> or x-session-token header and stores it with the job (no enforcement).
*/

import { NextResponse } from 'next/server';

// Types
type ImportStatus = 'queued' | 'importing' | 'done';

interface ImportJob {
  importId: string;
  projectId?: string;
  repoUrl?: string;
  status: ImportStatus;
  pollCount: number; // increments on each GET
  createdAt: string;
  sessionToken?: string;
  message?: string;
  result?: { importedRepoName?: string; itemsImported?: number };
}

interface CreateImportRequest {
  repoUrl: string;
  projectId?: string;
}

interface CreateImportResponse {
  importId: string;
  status: ImportStatus;
}

interface ImportStatusResponse {
  importId: string;
  status: ImportStatus;
  progress: number; // 0 | 50 | 100
  createdAt: string;
  message?: string;
  result?: { importedRepoName?: string; itemsImported?: number };
}

// In-memory store and deterministic id generator.
const jobs = new Map<string, ImportJob>();
let nextId = 1;
function makeImportId() {
  return `imp-${nextId++}`;
}

// POST handler: create a new import job.
export async function POST(req: Request) {
  try {
    const body = await req.json().catch(() => ({}));
    const { repoUrl, projectId } = body as CreateImportRequest;

    if (!repoUrl || typeof repoUrl !== 'string') {
      return NextResponse.json({ error: 'Missing required field: repoUrl' }, { status: 400 });
    }

    // Extract optional session token from headers or query params.
    const url = new URL(req.url);
    const sessionToken =
      req.headers.get('authorization')?.replace(/^Bearer\s+/i, '') ||
      req.headers.get('x-session-token') ||
      url.searchParams.get('sessionToken') ||
      undefined;

    const importId = makeImportId();
    const now = new Date().toISOString();

    const job: ImportJob = {
      importId,
      projectId,
      repoUrl,
      status: 'queued',
      pollCount: 0,
      createdAt: now,
      sessionToken,
      message: 'Job queued',
    };

    jobs.set(importId, job);

    const resp: CreateImportResponse = { importId, status: job.status };
    return NextResponse.json(resp, { status: 200 });
  } catch (err) {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }
}

// GET handler: poll job status. Query param: importId
export async function GET(req: Request) {
  const url = new URL(req.url);
  const importId = url.searchParams.get('importId');

  if (!importId) {
    return NextResponse.json({ error: 'Missing query param: importId' }, { status: 400 });
  }

  const job = jobs.get(importId);
  if (!job) {
    return NextResponse.json({ error: 'Import job not found' }, { status: 404 });
  }

  // Deterministic state machine: increment pollCount on each GET.
  // Transitions (based on new pollCount):
  //   pollCount === 1 -> 'importing'
  //   pollCount >= 2 -> 'done'
  // A freshly created job starts with pollCount=0 and status='queued'.
  job.pollCount += 1;

  if (job.pollCount === 1) {
    job.status = 'importing';
    job.message = 'Importing repository...';
  } else if (job.pollCount >= 2) {
    job.status = 'done';
    job.message = 'Import complete';
    // Provide a small deterministic result payload.
    job.result = { importedRepoName: inferRepoName(job.repoUrl), itemsImported: 3 };
  } else {
    job.status = 'queued';
    job.message = 'Job queued';
  }

  const progress = job.status === 'queued' ? 0 : job.status === 'importing' ? 50 : 100;

  const resp: ImportStatusResponse = {
    importId: job.importId,
    status: job.status,
    progress,
    createdAt: job.createdAt,
    message: job.message,
    result: job.result,
  };

  return NextResponse.json(resp, { status: 200 });
}

// Helper: naive repo name inference from a repo URL string.
function inferRepoName(repoUrl?: string) {
  if (!repoUrl) return undefined;
  try {
    const u = new URL(repoUrl);
    const parts = u.pathname.split('/').filter(Boolean);
    return parts[parts.length - 1] || u.hostname;
  } catch {
    const parts = repoUrl.split('/').filter(Boolean);
    return parts[parts.length - 1] || repoUrl;
  }
}

// Export internal jobs map for dev/testing convenience (not for production).
export const _dev_import_jobs = jobs;
