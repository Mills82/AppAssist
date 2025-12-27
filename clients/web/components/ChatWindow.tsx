/* clients/web/components/ChatWindow.tsx
 *
 * Main chat pane for the dashboard (client component).
 * Renders a header with a ModeChip and an Advanced toggle (collapsed by default),
 * a scrollable message list, and a local input area for demo/dev purposes.
 */

'use client';

import React, { useEffect, useRef, useState } from 'react';
import ModeChip from './ModeChip';
import RunTimeline from './RunTimeline';
import type {
  EventEnvelope as SharedEventEnvelope,
  StageState as SharedStageState,
  RunTimelineProps,
} from '../lib/types';

// Lightweight types for messages
type Author = 'user' | 'assistant' | 'system';

interface Message {
  id: string;
  author: Author;
  text: string;
  timestamp: number;
}

// SSE payload shape (best-effort)
interface SSEPayload {
  type?: 'message' | 'ping' | 'system';
  id?: string;
  author?: Author;
  text?: string;
  timestamp?: number;
}

// --- Local run/timeline types (use shared StageState/EventEnvelope from lib/types to avoid duplication) ---
// We keep some local loose typings for incoming payloads to be defensive against backend shape changes.

// Exported endpoints so other components can reuse the same constants (e.g. Sidebar)
export const SSE_URL = '/events/stream';
export const CONVERSATION_ENDPOINT = '/conversation';

const initialMessages: Message[] = [
  {
    id: 'm1',
    author: 'system',
    text: 'Welcome to the AI Dev Bot dashboard. Mode: Auto (Q&A).',
    timestamp: Date.now() - 10000,
  },
  {
    id: 'm2',
    author: 'assistant',
    text: 'Hi — ask me anything about your runs or projects.',
    timestamp: Date.now() - 8000,
  },
];

export default function ChatWindow({
  selectedProject,
}: {
  // optional project context passed from Sidebar/page
  selectedProject?: { id: string; name?: string } | null;
}): JSX.Element {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState<string>('');
  const [advancedShown, setAdvancedShown] = useState<boolean>(false); // collapsed by default
  const [modeOverride, setModeOverride] = useState<string>('Auto');
  const [isSending, setIsSending] = useState<boolean>(false);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'open' | 'reconnecting' | 'closed'>('closed');

  // Run / timeline state (use shared StageState shape)
  const [stages, setStages] = useState<SharedStageState[]>([]);
  const [currentStageId, setCurrentStageId] = useState<string | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [errors, setErrors] = useState<Array<{ message: string; ts: number }>>([]);
  const [otherEvents, setOtherEvents] = useState<any[]>([]);

  const listRef = useRef<HTMLDivElement | null>(null);

  // Refs for SSE lifecycle
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const retryAttemptRef = useRef<number>(0);
  const shouldReconnectRef = useRef<boolean>(true);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (!listRef.current) return;
    listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages]);

  const appendMessage = (author: Author, text: string) => {
    const m: Message = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      author,
      text,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, m]);
  };

  const appendIncoming = (payload: SSEPayload) => {
    const m: Message = {
      id: payload.id ?? `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      author: (payload.author as Author) ?? (payload.type === 'system' ? 'system' : 'assistant'),
      text: payload.text ?? JSON.stringify(payload),
      timestamp: payload.timestamp ? payload.timestamp : Date.now(),
    };
    setMessages((prev) => [...prev, m]);
  };

  // Helpers to manage timeline state (defensive, functional updates)
  const upsertStage = (incoming: Partial<SharedStageState> & { label?: string; title?: string; status?: string; ts?: number }) => {
    const id = incoming.id ?? incoming.label ?? incoming.title ?? `stage-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    const label = incoming.label ?? incoming.title ?? `Stage ${id}`;
    const status = (incoming.status as string) ?? 'pending';

    setStages((prev) => {
      const foundIndex = prev.findIndex((s) => s.id === id);
      let next = prev.slice();
      if (foundIndex >= 0) {
        // update existing — preserve other fields from shared shape
        next[foundIndex] = { ...next[foundIndex], label, title: label, status: status as any, ts: incoming.ts ?? next[foundIndex].ts } as SharedStageState;
      } else {
        // when activating a new stage, mark previous active as done
        if (status === 'active') {
          next = next.map((s) => (s.status === 'active' ? { ...s, status: 'done' } : s));
        }
        const newStage = { id, label, title: label, status: status as any, ts: incoming.ts } as SharedStageState;
        next.push(newStage);
      }

      return next;
    });

    if (status === 'active') {
      setCurrentStageId(id);
    } else if (status === 'done') {
      // if marking done and it matches current, clear current
      setCurrentStageId((cur) => (cur === id ? null : cur));
    } else if (status === 'error') {
      // if error for current stage, keep currentStageId so UI can highlight it
      setCurrentStageId((cur) => (cur === id ? cur : cur));
    }
  };

  const handleProgress = (payload: any) => {
    const raw = payload?.value ?? payload?.progress ?? payload;
    let v = Number(raw ?? 0);
    if (Number.isNaN(v)) v = 0;
    v = Math.max(0, Math.min(100, Math.round(v)));
    setProgress(v);
    if (v >= 100) {
      // mark active stage done
      setStages((prev) => prev.map((s) => (s.status === 'active' ? { ...s, status: 'done' } : s)));
      setCurrentStageId(null);
    }
  };

  const handleError = (payload: any) => {
    const message = payload?.message ?? payload?.text ?? JSON.stringify(payload);
    const ts = Date.now();
    setErrors((prev) => [...prev, { message, ts }]);
    // mark current stage as error if present
    setStages((prev) => prev.map((s) => (s.id === currentStageId ? { ...s, status: 'error' } : s)));
    // optionally clear progress/current stage marker
    setCurrentStageId(null);
  };

  // SSE connect with simple exponential backoff reconnect
  useEffect(() => {
    // Only run on client
    if (typeof window === 'undefined') return;

    shouldReconnectRef.current = true;

    const connectSSE = () => {
      // Prevent duplicate EventSource
      if (eventSourceRef.current) {
        try {
          eventSourceRef.current.close();
        } catch (e) {
          // ignore
        }
        eventSourceRef.current = null;
      }

      setConnectionStatus((prev) => (prev === 'open' ? 'open' : 'connecting'));

      let es: EventSource;
      try {
        es = new EventSource(SSE_URL);
      } catch (err) {
        console.warn('Failed to construct EventSource', err);
        scheduleReconnect();
        return;
      }

      eventSourceRef.current = es;

      es.onopen = () => {
        retryAttemptRef.current = 0;
        if (reconnectTimerRef.current) {
          window.clearTimeout(reconnectTimerRef.current);
          reconnectTimerRef.current = null;
        }
        setConnectionStatus('open');
      };

      es.onmessage = (e: MessageEvent) => {
        // Attempt to parse JSON; fall back to raw text
        try {
          const parsed = JSON.parse(e.data) as SharedEventEnvelope | SharedEventEnvelope[] | any;

          const envelopes = Array.isArray(parsed) ? parsed : [parsed];

          envelopes.forEach((env) => {
            const rawType = (env?.type as string | undefined) ?? (env?.payload?.type as string | undefined) ?? undefined;
            const payload = env?.payload ?? env;

            // Normalize event type to dot-separated form so both 'run:stage' and 'run.stage' are treated equally
            const normType = typeof rawType === 'string' ? rawType.replace(/:/g, '.') : undefined;

            // Message-ish events: either explicit type or legacy payloads
            if (normType === 'message' || payload?.author || payload?.text) {
              // Preserve existing chat rendering behavior
              appendIncoming(payload as SSEPayload);
              return;
            }

            // Timeline / stage updates (accept 'run.stage' and 'stage.update')
            if (normType === 'run.stage' || normType === 'stage.update') {
              try {
                const st = {
                  id: payload?.id ?? payload?.name ?? undefined,
                  label: payload?.label ?? payload?.name ?? payload?.id ?? 'Stage',
                  title: payload?.title ?? payload?.label ?? payload?.name ?? payload?.id ?? 'Stage',
                  status: (payload?.status as any) ?? (payload?.active ? 'active' : 'pending'),
                  ts: env?.ts ?? payload?.ts ?? Date.now(),
                } as Partial<SharedStageState>;
                upsertStage(st);
              } catch (err) {
                // defensive: unknown/malformed stage payloads should not break UI
                setOtherEvents((prev) => [...prev, env]);
              }

              return;
            }

            // Progress events
            if (normType === 'run.progress' || normType === 'progress') {
              handleProgress(payload);
              return;
            }

            // Error events
            if (normType === 'run.error' || normType === 'error') {
              handleError(payload);
              return;
            }

            // System/ping (keep existing small behavior)
            if (normType === 'system' || normType === 'ping') {
              if (normType === 'system') {
                appendIncoming({ type: 'system', text: payload?.text ?? env?.text ?? JSON.stringify(env), id: env?.id, author: 'system', timestamp: env?.ts ?? Date.now() } as SSEPayload);
              }
              // ignore pings
              return;
            }

            // Unknown event types: capture but do not throw
            setOtherEvents((prev) => [...prev, env]);
          });
        } catch (err) {
          // non-JSON message (legacy behavior): append as assistant text
          appendMessage('assistant', e.data);
        }
      };

      es.onerror = () => {
        // onerror can be fired for temporary network issues; close and attempt reconnect
        if (!shouldReconnectRef.current) return;
        setConnectionStatus('reconnecting');
        try {
          es.close();
        } catch (e) {
          // ignore
        }
        eventSourceRef.current = null;
        scheduleReconnect();
      };
    };

    const scheduleReconnect = () => {
      if (!shouldReconnectRef.current) return;
      const attempt = retryAttemptRef.current ?? 0;
      const delay = Math.min(10000, 500 * Math.pow(2, attempt));
      retryAttemptRef.current = attempt + 1;
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      reconnectTimerRef.current = window.setTimeout(() => {
        connectSSE();
      }, delay) as unknown as number;
    };

    // start
    connectSSE();

    return () => {
      // cleanup on unmount
      shouldReconnectRef.current = false;
      setConnectionStatus('closed');
      if (eventSourceRef.current) {
        try {
          eventSourceRef.current.close();
        } catch (e) {
          // ignore
        }
        eventSourceRef.current = null;
      }
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };
  }, []);

  const handleSend = async () => {
    if (!input.trim()) return;
    const userText = input.trim();
    setInput('');

    // Optimistic append
    appendMessage('user', userText);

    // POST to conversation endpoint; server is expected to stream assistant reply via SSE
    setIsSending(true);
    try {
      if (typeof window === 'undefined') {
        // should not happen during SSR but guard anyway
        setIsSending(false);
        return;
      }

      const body: any = { text: userText };
      if (selectedProject?.id) body.projectId = selectedProject.id;

      const res = await fetch(CONVERSATION_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        console.warn('Conversation POST failed', res.status, await res.text());
        appendMessage('system', 'Failed to send message to server (see console).');
      }
      // Do not rely on the response body — assistant replies will come via SSE
    } catch (err) {
      console.warn('Error sending conversation POST', err);
      appendMessage('system', 'Network error while sending message.');
    } finally {
      setIsSending(false);
    }
  };

  // Determine whether to show timeline area
  const shouldRenderTimeline = stages.length > 0 || progress > 0 || errors.length > 0;

  return (
    <div className="flex h-full flex-1 flex-col bg-white dark:bg-slate-900 border-l border-slate-100 dark:border-slate-800">
      {/* Header */}
      <div className="flex items-center justify-between gap-3 px-4 py-3 border-b border-slate-100 dark:border-slate-800">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-200">Conversation</h2>
          {/* Show project context if provided */}
          {selectedProject && (
            <div className="text-xs text-slate-500 dark:text-slate-300 rounded px-2 py-0.5 bg-slate-50 dark:bg-slate-800">
              {selectedProject.name ?? selectedProject.id}
            </div>
          )}
          {/* ModeChip shows inferred mode. Mode override toggles only locally for demo. */}
          <ModeChip mode={modeOverride} />
        </div>

        <div className="flex items-center gap-2">
          {/* Connection status indicator */}
          <div className="text-xs text-slate-500 dark:text-slate-300 mr-2">{connectionStatus}</div>

          <button
            type="button"
            aria-expanded={advancedShown}
            aria-controls="chat-advanced"
            onClick={() => setAdvancedShown((s) => !s)}
            className="rounded-md px-2 py-1 text-sm bg-slate-50 hover:bg-slate-100 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200"
          >
            Advanced
          </button>
        </div>
      </div>

      {/* Timeline (render only when there is timeline/progress/error data) */}
      {shouldRenderTimeline && (
        <RunTimeline
          stages={stages}
          currentStageId={currentStageId ?? undefined}
          progress={progress}
          errors={errors}
          otherEvents={otherEvents}
        />
      )}

      {/* Advanced controls (collapsed by default) */}
      {advancedShown && (
        <div id="chat-advanced" className="border-b border-slate-100 dark:border-slate-800 px-4 py-3 bg-slate-50 dark:bg-slate-800">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-200">
              Mode override:
              <select
                value={modeOverride}
                onChange={(e) => setModeOverride(e.target.value)}
                className="ml-2 rounded-md border px-2 py-1 bg-white dark:bg-slate-900 text-sm"
              >
                <option>Auto</option>
                <option>Q&A</option>
                <option>Analyze</option>
                <option>Plan</option>
                <option>Edit</option>
              </select>
            </label>

            <div className="text-sm text-slate-600 dark:text-slate-300">
              <span className="font-medium">Note:</span> These controls are local-only demo controls and do not call the backend.
            </div>
          </div>
        </div>
      )}

      {/* Message list */}
      <div ref={listRef} className="flex-1 overflow-auto p-4" role="log" aria-live="polite">
        <ul className="flex flex-col gap-3">
          {messages.map((m) => (
            <li key={m.id} className="max-w-[80%]" style={{ alignSelf: m.author === 'user' ? 'flex-end' : 'flex-start' }}>
              <div
                className={`rounded-lg px-4 py-2 text-sm leading-relaxed shadow-sm ${
                  m.author === 'user'
                    ? 'bg-indigo-600 text-white'
                    : m.author === 'assistant'
                    ? 'bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200'
                    : 'bg-amber-50 text-amber-900'
                }`}
              >
                {m.text}
              </div>
              <div className="mt-1 text-[11px] text-slate-400">{new Date(m.timestamp).toLocaleTimeString()}</div>
            </li>
          ))}
        </ul>
      </div>

      {/* Input area */}
      <div className="border-t border-slate-100 dark:border-slate-800 px-4 py-3">
        <div className="flex items-center gap-3">
          <textarea
            aria-label="Message input"
            placeholder="Type a message and press Send"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            rows={1}
            className="min-h-[44px] max-h-36 w-full resize-none rounded-md border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300 dark:bg-slate-900 dark:border-slate-700 dark:text-slate-200"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
          />

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleSend}
              disabled={!input.trim() || isSending}
              className={`rounded-md px-3 py-2 text-sm font-medium ${
                !input.trim() || isSending
                  ? 'bg-slate-200 text-slate-500 dark:bg-slate-700 dark:text-slate-400 cursor-not-allowed'
                  : 'bg-indigo-600 text-white hover:bg-indigo-700'
              }`}
            >
              {isSending ? 'Sending...' : 'Send'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Minimal dev-only mount example:
// If you open a plain HTML file that contains <div id="chat-root"></div> and include React,
// this block will mount ChatWindow for quick local previews. This is development-only and
// guarded by a runtime check so it won't run inside Next.js server rendering.
if (typeof window !== 'undefined') {
  const mountEl = document.getElementById('chat-root');
  if (mountEl) {
    // Dynamic import to avoid requiring react-dom/client in environments that don't have it.
    import('react-dom/client')
      .then((ReactDOM) => {
        const root = ReactDOM.createRoot(mountEl);
        root.render(React.createElement(ChatWindow));
      })
      .catch(() => {
        // If react-dom isn't available (e.g., during some build steps), fail silently.
        // This is only a convenience helper for local dev.
      });
  }
}
