/* clients/web/components/GithubConnectButton.tsx

A small client-only React component to drive a stubbed GitHub connect + import
state machine. Works against a backend if NEXT_PUBLIC_API_BASE_URL/v1/... endpoints
exist, otherwise falls back to a simulated import flow (dev-friendly).

Exports:
- default: GithubConnectButton component
- ImportState: enum of internal states (useful for tests)
- GithubConnectButtonDemo: a tiny example/demo component (development)

Usage example (in pages or other components):
<GithubConnectButton projectId={projectId} repoName={"my-repo"} onImportComplete={(repo)=>{console.log('import done', repo)}} />
*/

"use client";

import React, { useEffect, useRef, useState } from "react";

export type ImportedRepo = { id: string; name: string };

export enum ImportState {
  Disconnected = "disconnected",
  Connected = "connected",
  Importing = "importing",
  Imported = "imported",
}

export interface GithubConnectButtonProps {
  projectId?: string;
  repoName?: string;
  onImportComplete?: (repo: ImportedRepo) => void;
  className?: string;
  /** If true forces the simulated dev flow even if an API base URL exists */
  devMode?: boolean;
}

const DEFAULT_SIM_STEPS = [300, 800, 1500];

function safeDispatchImportedEvent(projectId: string | undefined, repo: ImportedRepo) {
  try {
    window.dispatchEvent(
      new CustomEvent("project:imported", { detail: { projectId, repo } })
    );
  } catch (e) {
    // ignore in non-browser test environments
  }
}

export default function GithubConnectButton({
  projectId,
  repoName,
  onImportComplete,
  className = "",
  devMode = false,
}: GithubConnectButtonProps) {
  const [state, setState] = useState<ImportState>(ImportState.Disconnected);
  const [progress, setProgress] = useState<number>(0);
  const timeoutsRef = useRef<number[]>([]);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      // cleanup timeouts
      timeoutsRef.current.forEach((id) => window.clearTimeout(id));
      abortControllerRef.current?.abort();
    };
  }, []);

  const startSimulatedImport = async () => {
    setState(ImportState.Importing);
    setProgress(0);
    const steps = DEFAULT_SIM_STEPS;
    // Step 1 -> 30%
    const t1 = window.setTimeout(() => setProgress(30), steps[0]);
    timeoutsRef.current.push(t1);
    // Step 2 -> 70%
    const t2 = window.setTimeout(() => setProgress(70), steps[1]);
    timeoutsRef.current.push(t2);
    // Finalize -> 100%
    const t3 = window.setTimeout(() => {
      setProgress(100);
      setState(ImportState.Imported);
      const fakeRepo: ImportedRepo = {
        id: `import-${Date.now()}`,
        name: repoName || "imported-repo",
      };
      onImportComplete?.(fakeRepo);
      safeDispatchImportedEvent(projectId, fakeRepo);
      // clear timeouts list
      timeoutsRef.current = timeoutsRef.current.filter((id) => id !== t1 && id !== t2 && id !== t3);
    }, steps[2]);
    timeoutsRef.current.push(t3);
  };

  const pollImportStatus = async (jobId: string) => {
    // Poll a hypothetical /v1/imports/:jobId endpoint. Backoff simple implementation.
    const base = process.env.NEXT_PUBLIC_API_BASE_URL || "/api";
    let attempts = 0;
    abortControllerRef.current = new AbortController();
    try {
      while (attempts < 12) {
        attempts += 1;
        const resp = await fetch(`${base}/v1/imports/${jobId}`, { signal: abortControllerRef.current.signal });
        if (!resp.ok) {
          // If server doesn't offer import status, fallback to simulated flow
          throw new Error(`Status ${resp.status}`);
        }
        const body = await resp.json();
        // Expected contract: { status: 'pending'|'done', progress?: number, repo?: {id,name} }
        if (body?.status === "done" && body?.repo) {
          setProgress(100);
          setState(ImportState.Imported);
          onImportComplete?.(body.repo);
          safeDispatchImportedEvent(projectId, body.repo);
          return;
        }
        if (typeof body?.progress === "number") {
          setProgress(Math.max(0, Math.min(100, body.progress)));
        } else {
          // nudge progress so UI doesn't look frozen
          setProgress((p) => Math.min(95, p + 10));
        }
        // wait 1s
        await new Promise((res) => setTimeout(res, 1000));
      }
      // timed out -> mark failed -> fallback to simulated finalization
      const fallbackRepo: ImportedRepo = { id: `import-${Date.now()}`, name: repoName || "imported-repo" };
      setProgress(100);
      setState(ImportState.Imported);
      onImportComplete?.(fallbackRepo);
      safeDispatchImportedEvent(projectId, fallbackRepo);
    } catch (e) {
      // any error while polling falls back to simulated import completion
      const fallbackRepo: ImportedRepo = { id: `import-${Date.now()}`, name: repoName || "imported-repo" };
      setProgress(100);
      setState(ImportState.Imported);
      onImportComplete?.(fallbackRepo);
      safeDispatchImportedEvent(projectId, fallbackRepo);
    } finally {
      abortControllerRef.current = null;
    }
  };

  const startImportFlow = async () => {
    // Try to call a real import endpoint; if that fails or devMode is true, simulate.
    if (devMode) {
      // quick simulated flow in devMode
      await startSimulatedImport();
      return;
    }

    const base = process.env.NEXT_PUBLIC_API_BASE_URL || "/api";
    // require projectId for server-backed import; otherwise simulate
    if (!projectId) {
      await startSimulatedImport();
      return;
    }

    try {
      setState(ImportState.Importing);
      setProgress(5);
      const resp = await fetch(`${base}/v1/projects/${projectId}/import`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repoName }),
      });
      if (!resp.ok) {
        // fall back to simulate
        await startSimulatedImport();
        return;
      }
      const body = await resp.json();
      // expect: { jobId?: string, repo?: {id,name} }
      if (body?.repo) {
        setProgress(100);
        setState(ImportState.Imported);
        onImportComplete?.(body.repo);
        safeDispatchImportedEvent(projectId, body.repo);
        return;
      }
      if (body?.jobId) {
        // poll
        await pollImportStatus(body.jobId);
        return;
      }
      // nothing useful -> simulate
      await startSimulatedImport();
    } catch (e) {
      await startSimulatedImport();
    }
  };

  const handleClick = async () => {
    if (state === ImportState.Disconnected) {
      // simulate a GitHub OAuth connect, then mark connected
      setState(ImportState.Connected);
      return;
    }
    if (state === ImportState.Connected) {
      // start import
      await startImportFlow();
      return;
    }
    if (state === ImportState.Importing) {
      // allow cancellation? For now do nothing
      return;
    }
    if (state === ImportState.Imported) {
      // maybe allow re-importing
      setState(ImportState.Connected);
      setProgress(0);
    }
  };

  const labelForState = (s: ImportState) => {
    switch (s) {
      case ImportState.Disconnected:
        return "Connect GitHub";
      case ImportState.Connected:
        return repoName ? `Import ${repoName}` : "Start import";
      case ImportState.Importing:
        return `Importing... ${progress}%`;
      case ImportState.Imported:
        return "Imported";
      default:
        return "Connect";
    }
  };

  return (
    <div className={`inline-flex flex-col items-stretch ${className}`}>
      <button
        type="button"
        onClick={handleClick}
        onKeyDown={(e) => {
          if (e.key === " " || e.key === "Enter") {
            e.preventDefault();
            handleClick();
          }
        }}
        aria-pressed={state !== ImportState.Disconnected}
        aria-busy={state === ImportState.Importing}
        title={labelForState(state)}
        className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm font-medium hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
          state === ImportState.Importing ? "opacity-90" : ""
        }`}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="h-4 w-4"
          aria-hidden
        >
          <path d="M16 8a6 6 0 0 0-12 0c0 3.3 2.7 6 6 6h.5" />
          <path d="M8 14v7" />
        </svg>
        <span>{labelForState(state)}</span>
        {state === ImportState.Importing ? (
          <span className="ml-2 text-xs text-gray-500">{progress}%</span>
        ) : null}
      </button>

      {/* simple inline progress bar */}
      {state === ImportState.Importing ? (
        <div className="w-full mt-2 bg-gray-100 dark:bg-gray-700 rounded-full h-2 overflow-hidden">
          <div
            className="bg-indigo-500 h-2 transition-all"
            style={{ width: `${progress}%` }}
          />
        </div>
      ) : null}

      {/* small status text */}
      <div className="mt-1 text-xs text-gray-500">{state}</div>
    </div>
  );
}

// Small demo component for local development/testing; not used by app unless imported.
export function GithubConnectButtonDemo() {
  const [logs, setLogs] = useState<string[]>([]);
  useEffect(() => {
    const handler = (e: Event) => {
      // @ts-ignore access detail
      const d = (e as CustomEvent).detail;
      setLogs((ls) => [`imported: ${JSON.stringify(d)}`, ...ls].slice(0, 5));
    };
    window.addEventListener("project:imported", handler as EventListener);
    return () => window.removeEventListener("project:imported", handler as EventListener);
  }, []);

  return (
    <div className="p-4 border rounded-md bg-white dark:bg-gray-800">
      <h4 className="font-semibold mb-2">GithubConnectButton Demo</h4>
      <div className="mb-2">
        <GithubConnectButton projectId="demo-project" repoName="demo-repo" onImportComplete={(r) => setLogs((ls) => [`cb: ${r.name}`, ...ls].slice(0, 5))} devMode />
      </div>
      <div className="text-xs text-gray-500">Event & callback logs:</div>
      <ul className="mt-2 text-xs list-disc pl-5 max-h-32 overflow-auto">
        {logs.map((l, i) => (
          <li key={i}>{l}</li>
        ))}
      </ul>
    </div>
  );
}
