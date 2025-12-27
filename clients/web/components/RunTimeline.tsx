"use client";

import React, { useEffect, useMemo, useState } from "react";

/*
clients/web/components/RunTimeline.tsx

A small, accessible, controlled React component that renders a horizontal,
stage-based timeline with badges, a thin progress bar, and an optional
error banner. Visual styling is expected to come from clients/web/app/globals.css
(Target classes: rt-container, rt-badge, rt-badge--active, rt-badge--done,
rt-badge--error, rt-progress, rt-error-banner).

This file provides minimal inline types so the component is compile-safe
without requiring changes to clients/web/lib/types.ts. Once shared types
are added, you can replace these with a type-only import.
*/

// Minimal, local types (replace with `import type { EventEnvelope, StageState } from '../lib/types'` when available)
export type KnownEventType =
  | "stage.start"
  | "stage.progress"
  | "stage.done"
  | "run.error"
  | "unknown";

export interface EventEnvelope {
  type: KnownEventType | string;
  data?: any;
  timestamp?: string; // ISO string
}

export type StageStatus = "pending" | "active" | "done" | "error";

export interface Stage {
  id: string | number;
  label: string;
}

export interface RunTimelineProps {
  stages: Array<string | Stage>;
  // identifies the current stage by index (number) or by id/label (string/number)
  currentStage?: string | number;
  // 0..100 progress percent. undefined -> indeterminate/hidden
  progress?: number;
  // array of error messages (if non-empty, a banner is shown and the current stage is marked error)
  errors?: string[];
  // other/unknown SSE events that shouldn't break the UI
  otherEvents?: EventEnvelope[];
  // optional aria-label for the timeline group
  ariaLabel?: string;
}

// Utility: normalize stage input to Stage[]
function normalizeStages(input: Array<string | Stage>): Stage[] {
  return input.map((s, i) =>
    typeof s === "string"
      ? { id: `s-${i}`, label: s }
      : { id: s.id ?? `s-${i}`, label: (s as Stage).label ?? String(s.id ?? `stage-${i}`) }
  );
}

function clampPercent(v?: number) {
  if (v === undefined || Number.isNaN(v)) return undefined;
  return Math.max(0, Math.min(100, Math.round(v)));
}

export default function RunTimeline({
  stages,
  currentStage,
  progress,
  errors,
  otherEvents,
  ariaLabel = "Run progress",
}: RunTimelineProps) {
  const normalized = useMemo(() => normalizeStages(stages), [stages]);

  // Determine current index
  const currentIndex = useMemo(() => {
    if (currentStage === undefined || currentStage === null) return -1;
    if (typeof currentStage === "number") {
      if (currentStage >= 0 && currentStage < normalized.length) return currentStage;
      return -1;
    }
    // try match by id or label
    const byId = normalized.findIndex((s) => String(s.id) === String(currentStage));
    if (byId >= 0) return byId;
    const byLabel = normalized.findIndex((s) => String(s.label) === String(currentStage));
    return byLabel >= 0 ? byLabel : -1;
  }, [normalized, currentStage]);

  const hasError = Boolean(errors && errors.length > 0) && currentIndex >= 0;
  const pct = clampPercent(progress);

  // Compute state for each stage
  const stageStates: StageStatus[] = normalized.map((_, i) => {
    if (hasError && i === currentIndex) return "error";
    if (i < currentIndex) return "done";
    if (i === currentIndex) return "active";
    return "pending";
  });

  return (
    <div className="rt-container" role="group" aria-label={ariaLabel}>
      {/* Error banner (polite) */}
      {errors && errors.length > 0 ? (
        <div className="rt-error-banner" role="alert" aria-live="polite">
          <strong>Error:</strong> {errors[0]}
          {errors.length > 1 ? <span> (+{errors.length - 1} more)</span> : null}
        </div>
      ) : null}

      {/* Horizontal badges */}
      <div className="rt-badges" style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        {normalized.map((s, i) => {
          const state = stageStates[i];
          const cls = ["rt-badge", `rt-badge--${state}`].join(" ");
          return (
            <div
              key={String(s.id)}
              className={cls}
              aria-current={state === "active" ? "step" : undefined}
              // provide a little semantic text for assistive tech
              title={s.label}
              style={{ display: "flex", alignItems: "center", gap: 6 }}
            >
              <span
                className="rt-badge-dot"
                aria-hidden
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: 999,
                  background: state === "done" ? "var(--success, #2d9a4a)" : state === "error" ? "var(--danger, #d64545)" : state === "active" ? "var(--accent, #2563eb)" : "#e2e8f0",
                }}
              />
              <span className="rt-badge-label" style={{ fontSize: 13 }}>{s.label}</span>
            </div>
          );
        })}
      </div>

      {/* Thin progress bar */}
      <div
        className="rt-progress-wrapper"
        style={{ marginTop: 8, height: 6, background: "var(--muted-2, #f1f5f9)", borderRadius: 999, overflow: "hidden" }}
      >
        {pct === undefined ? (
          // indeterminate: simple animated stripe using CSS gradient would be ideal in globals.css;
          // fall back to a subtle placeholder bar here.
          <div
            className="rt-progress rt-progress--indeterminate"
            style={{ width: "100%", height: "100%", background: "linear-gradient(90deg, rgba(0,0,0,0.04) 25%, rgba(0,0,0,0.06) 50%, rgba(0,0,0,0.04) 75%)", backgroundSize: "200% 100%" }}
            aria-hidden
          />
        ) : (
          <div
            className="rt-progress"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={pct}
            style={{ width: `${pct}%`, height: "100%", background: "var(--accent, #2563eb)", transition: "width 280ms ease" }}
          />
        )}
      </div>

      {/* Collapsible list of other/unknown events so the UI doesn't break on unexpected SSE shapes */}
      {otherEvents && otherEvents.length > 0 ? (
        <details style={{ marginTop: 8 }}>
          <summary style={{ cursor: "pointer", fontSize: 13 }}>
            Other events ({otherEvents.length})
          </summary>
          <ul style={{ marginTop: 6, paddingLeft: 16 }}>
            {otherEvents.map((e, i) => (
              <li key={i} style={{ fontSize: 12, color: "var(--muted, #475569)" }}>
                <code style={{ fontSize: 12 }}>{e.type}</code>
                {e.data ? <span>: {String(typeof e.data === "string" ? e.data : JSON.stringify(e.data))}</span> : null}
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </div>
  );
}

/* Minimal demo harness. This is exported so devs can drop <RunTimelineDemo /> into a page during development.
   It is intentionally small and dependency-free. Remove or ignore in production builds. */
export function RunTimelineDemo() {
  const demoStages = ["Initialize", "Fetch Data", "Process", "Finalize"];
  const [idx, setIdx] = useState(0);
  const [pct, setPct] = useState(0);
  const [errs, setErrs] = useState<string[] | undefined>(undefined);

  useEffect(() => {
    setPct(0);
    setErrs(undefined);
    const tick = setInterval(() => setPct((p) => Math.min(100, p + 7)), 220);
    const stageTimer = setInterval(() => setIdx((i) => {
      if (i >= demoStages.length - 1) {
        clearInterval(stageTimer);
        clearInterval(tick);
        setPct(100);
        return demoStages.length - 1;
      }
      setPct(0);
      return i + 1;
    }), 2600);

    // Example: introduce an error halfway through the second stage (dev only)
    const errTimer = setTimeout(() => {
      // setErrs(["Failed to reach worker node"]);
      // To see error state, uncomment the above line.
    }, 4000);

    return () => {
      clearInterval(tick);
      clearInterval(stageTimer);
      clearTimeout(errTimer);
    };
  }, []);

  return (
    <div style={{ padding: 12, maxWidth: 680 }}>
      <h4 style={{ marginTop: 0 }}>RunTimeline Demo</h4>
      <RunTimeline stages={demoStages} currentStage={idx} progress={pct} errors={errs} otherEvents={[{ type: "debug.info", data: { pct }, timestamp: new Date().toISOString() }]} />
    </div>
  );
}

// trailing newline
