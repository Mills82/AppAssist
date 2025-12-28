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
import type { StageState as SharedStageState } from '../lib/types';
import { postConversation, subscribeToRun } from '../lib/api';

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

// Local extension: SharedStageState currently doesn’t include `ts`,
// but this component uses it for ordering/debug.
type StageStateWithTs = SharedStageState & { ts?: number };

// Exported endpoints so other components can reuse the same constants (e.g. Sidebar)
// NOTE: prefer centralized helpers from ../lib/api. These constants are kept for backwards compatibility.
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

  // Run / timeline state (use shared StageState shape + local ts)
  const [stages, setStages] = useState<StageStateWithTs[]>([]);
  const [currentStageId, setCurrentStageId] = useState<string | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [errors, setErrors] = useState<Array<{ message: string; ts: number }>>([]);
  const [otherEvents, setOtherEvents] = useState<any[]>([]);

  const listRef = useRef<HTMLDivElement | null>(null);

  // Subscription controller for the active run (if any)
  const subscriptionRef = useRef<any | null>(null);
  const currentRunId = useRef<string | null>(null);
  const gotFirstPatch = useRef<boolean>(false);

  function pushMessage(m: Message) {
    // We intentionally do not perform a global message-id dedupe here.
    // The subscribeToRun API is responsible for dedupe/resume semantics.
    setMessages((prev) => [...prev, m]);
  }

  const removeLocalOptimisticIfDuplicate = (incoming: { author?: Author; text?: string; timestamp?: number }) => {
    // Best-effort: remove a local optimistic message if the server echoes it back.
    // Only applies to user-authored messages and uses the local id marker 'local-'.
    if (incoming.author !== 'user' || !incoming.text) return;

    setMessages((prev) => {
      const now = incoming.timestamp ?? Date.now();
      const text = incoming.text ?? '';
      const idx = prev.findIndex((m) => {
        if (!m.id.startsWith('local-')) return false;
        if (m.author !== 'user') return false;
        if (m.text !== text) return false;
        return Math.abs(m.timestamp - now) < 10_000;
      });
      if (idx < 0) return prev;
      const next = prev.slice();
      next.splice(idx, 1);
      return next;
    });
  };

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
    pushMessage(m);
  };

  const appendIncoming = (payload: SSEPayload | any) => {
    const m: Message = {
      id: payload.id ?? `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      author: (payload.author as Author) ?? (payload.type === 'system' ? 'system' : 'assistant'),
      text: payload.text ?? JSON.stringify(payload),
      timestamp: payload.timestamp ? payload.timestamp : Date.now(),
    };

    removeLocalOptimisticIfDuplicate({ author: m.author, text: m.text, timestamp: m.timestamp });
    pushMessage(m);
  };

  // Helpers to manage timeline state (defensive, functional updates)
  const upsertStage = (
    incoming: Partial<StageStateWithTs> & { label?: string; title?: string; status?: string; ts?: number }
  ) => {
    const id = incoming.id ?? incoming.label ?? incoming.title ?? `stage-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    const label = incoming.label ?? incoming.title ?? `Stage ${id}`;
    const status = (incoming.status as string) ?? 'pending';

    setStages((prev) => {
      const foundIndex = prev.findIndex((s) => s.id === id);
      let next = prev.slice();

      if (foundIndex >= 0) {
        // update existing — preserve other fields from shared shape
        next[foundIndex] = {
          ...next[foundIndex],
          label,
          title: label,
          status: status as any,
          ts: incoming.ts ?? next[foundIndex].ts,
        };
      } else {
        // when activating a new stage, mark previous active as done
        if (status === 'active') {
          next = next.map((s) => (s.status === 'active' ? { ...s, status: 'done' as any } : s));
        }
        const newStage: StageStateWithTs = {
          id,
          label,
          title: label,
          status: status as any,
          ts: incoming.ts,
        };
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
      setStages((prev) => prev.map((s) => (s.status === 'active' ? { ...s, status: 'done' as any } : s)));
      setCurrentStageId(null);
    }
  };

  const handleError = (payload: any) => {
    const message = payload?.message ?? payload?.text ?? JSON.stringify(payload);
    const ts = Date.now();
    setErrors((prev) => [...prev, { message, ts }]);

    // mark current stage as error if present
    setStages((prev) => prev.map((s) => (s.id === currentStageId ? { ...s, status: 'error' as any } : s)));

    // run ended in UI state
    setCurrentStageId(null);
  };

  // Apply a RunStatePatch emitted by subscribeToRun — this is the single source of truth for run state
  const applyPatch = (patch: any) => {
    if (!patch || typeof patch !== 'object') return;

    // Messages: trusted canonical list/stream of messages
    if (Array.isArray(patch.messages)) {
      patch.messages.forEach((m: any) => {
        // defensive mapping from canonical message shape to local Message
        const mapped: SSEPayload = {
          id: m.id,
          author: (m.author as Author) ?? (m.role ?? undefined),
          text: m.text ?? m.content ?? JSON.stringify(m),
          timestamp: m.timestamp ?? m.ts,
        };
        removeLocalOptimisticIfDuplicate({ author: mapped.author, text: mapped.text, timestamp: mapped.timestamp });
        appendIncoming(mapped);
      });
    }

    // Stages: stage activation / updates
    if (Array.isArray(patch.stages)) {
      patch.stages.forEach((s: any) => upsertStage(s));
    }

    // Progress
    if (patch.progress !== undefined) {
      handleProgress({ value: patch.progress });
    }

    // Errors array (normalize into our {message, ts} list)
    if (Array.isArray(patch.errors)) {
      setErrors((prev) => [
        ...prev,
        ...patch.errors.map((e: any) => ({
          message: e?.message ?? (typeof e === 'string' ? e : JSON.stringify(e)),
          ts: e?.ts ?? Date.now(),
        })),
      ]);
    }

    // Other miscellaneous events (capture for debugging/RunTimeline)
    if (Array.isArray(patch.otherEvents)) {
      setOtherEvents((prev) => [...prev, ...patch.otherEvents]);
    }

    // Status transitions (ended/error)
    if (patch.status === 'error') {
      handleError({ message: patch.errorMessage ?? 'Run error' });
    }
    if (patch.status === 'done' || patch.status === 'ended') {
      setStages((prev) => prev.map((s) => (s.status === 'active' ? { ...s, status: 'done' as any } : s)));
      setCurrentStageId(null);
      setConnectionStatus('closed');
    }
  };

  // Cleanup subscription when component unmounts
  useEffect(() => {
    return () => {
      if (!subscriptionRef.current) return;
      try {
        if (typeof subscriptionRef.current === 'function') subscriptionRef.current();
        else if (typeof subscriptionRef.current.stop === 'function') subscriptionRef.current.stop();
        else if (typeof subscriptionRef.current.unsubscribe === 'function') subscriptionRef.current.unsubscribe();
      } catch {
        // ignore cleanup errors
      }
      subscriptionRef.current = null;
      currentRunId.current = null;
    };
  }, []);

  const subscribeToRunId = (runId: string) => {
    // Unsubscribe previous if present
    if (subscriptionRef.current) {
      try {
        if (typeof subscriptionRef.current === 'function') subscriptionRef.current();
        else if (typeof subscriptionRef.current.stop === 'function') subscriptionRef.current.stop();
        else if (typeof subscriptionRef.current.unsubscribe === 'function') subscriptionRef.current.unsubscribe();
      } catch {
        // swallow
      }
      subscriptionRef.current = null;
      gotFirstPatch.current = false;
    }

    try {
      setConnectionStatus('connecting');
      const controller = subscribeToRun(runId, {
        onPatch: (patch: any) => {
          if (!gotFirstPatch.current) {
            setConnectionStatus('open');
            gotFirstPatch.current = true;
          }
          applyPatch(patch);
        },
        onError: (err: any) => {
          setConnectionStatus('reconnecting');
          try {
            handleError(err ?? { message: 'Stream error' });
          } catch {
            // ignore
          }
        },
        onDone: () => {
          setConnectionStatus('closed');
          setCurrentStageId(null);
        },
      });

      subscriptionRef.current = controller ?? null;
      currentRunId.current = runId;
    } catch (err) {
      console.warn('Failed to subscribe to run', err);
      setConnectionStatus('closed');
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    const userText = input.trim();
    setInput('');

    const localId = `local-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    pushMessage({ id: localId, author: 'user', text: userText, timestamp: Date.now() });

    setIsSending(true);
    try {
      if (typeof window === 'undefined') {
        setIsSending(false);
        return;
      }

      const body: any = { text: userText };
      if (selectedProject?.id) body.projectId = selectedProject.id;

      const res = await postConversation(body);

      let runId: string | undefined;
      try {
        if (res && typeof (res as any).runId === 'string') {
          runId = (res as any).runId;
        } else if (res && typeof (res as any).json === 'function') {
          const json = await (res as any).json();
          if (json && typeof json.runId === 'string') runId = json.runId;
        }
      } catch {
        // ignore
      }

      if (!runId) {
        console.warn(
          'postConversation did not return runId; cannot subscribe to run stream. Ensure postConversation returns { runId } or similar.'
        );
      } else {
        subscribeToRunId(runId);
      }

      if (res && typeof (res as any).ok === 'boolean' && !(res as any).ok) {
        console.warn('Conversation POST failed', (res as any).status, await (res as any).text?.());
        appendMessage('system', 'Failed to send message to server (see console).');
      }
    } catch (err) {
      console.warn('Error sending conversation POST', err);
      appendMessage('system', 'Network error while sending message.');
    } finally {
      setIsSending(false);
    }
  };

  // Adapt types expected by RunTimeline without changing RunTimeline.tsx
  // RunTimeline expects: stages: (string | Stage)[] where Stage requires label
  const timelineStages = stages.map((s) => ({
    ...s,
    label: (s as any).label ?? (s as any).title ?? s.id,
    title: (s as any).title ?? (s as any).label ?? s.id,
  })) as any[];

  // RunTimeline expects errors: string[]
  const timelineErrors = errors.map((e) => e.message);

  // Determine whether to show timeline area
  const shouldRenderTimeline = timelineStages.length > 0 || progress > 0 || timelineErrors.length > 0;

  return (
    <div className="flex h-full flex-1 flex-col bg-white dark:bg-slate-900 border-l border-slate-100 dark:border-slate-800">
      {/* Header */}
      <div className="flex items-center justify-between gap-3 px-4 py-3 border-b border-slate-100 dark:border-slate-800">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-200">Conversation</h2>
          {selectedProject && (
            <div className="text-xs text-slate-500 dark:text-slate-300 rounded px-2 py-0.5 bg-slate-50 dark:bg-slate-800">
              {selectedProject.name ?? selectedProject.id}
            </div>
          )}
          <ModeChip mode={modeOverride} />
        </div>

        <div className="flex items-center gap-2">
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

      {/* Timeline */}
      {shouldRenderTimeline && (
        <RunTimeline
          stages={timelineStages}
          currentStage={currentStageId ?? undefined} // ✅ RunTimeline expects `currentStage`
          progress={progress}
          errors={timelineErrors}
          otherEvents={otherEvents}
        />
      )}

      {/* Advanced controls */}
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
if (typeof window !== 'undefined') {
  const mountEl = document.getElementById('chat-root');
  if (mountEl) {
    import('react-dom/client')
      .then((ReactDOM) => {
        const root = ReactDOM.createRoot(mountEl);
        root.render(React.createElement(ChatWindow));
      })
      .catch(() => {
        // dev-only convenience
      });
  }
}
