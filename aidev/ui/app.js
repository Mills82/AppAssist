// aidev/ui/app.js

// ---------- tiny utils ----------
const friendlyStages = [
    'Understanding your project',
    'Planning changes',
    'Drafting edits',
    'Waiting for approval',
    'Applying changes'
];
function friendlyStageLabel(raw = '') {
    if (!raw) return 'Working';
    const s = String(raw).toLowerCase();
    if (s.includes('structure') || s.includes('scan') || s.includes('project')) return friendlyStages[0];
    if (s.includes('plan') || s.includes('planning') || s.includes('recommend')) return friendlyStages[1];
    if (s.includes('generate') || s.includes('draft') || s.includes('edit') || s.includes('ai')) return friendlyStages[2];
    if (s.includes('approval') || s.includes('approve') || s.includes('need_plan') || s.includes('approval_requested')) return friendlyStages[3];
    if (s.includes('apply') || s.includes('result') || s.includes('done') || s.includes('end')) return friendlyStages[4];
    return raw.charAt(0).toUpperCase() + raw.slice(1);
}
function stageIndexOf(label) {
    const idx = friendlyStages.indexOf(label);
    return idx === -1 ? null : idx + 1;
}

// Stage map support: read a non-executing JSON block from index.html
// <script id="aidev-stage-map" type="application/json">{...}</script>
let STAGE_MAP = null;
function parseStageMapFromDOM() {
    try {
        const el = document.getElementById('aidev-stage-map');
        if (!el) return null;
        const txt = (el.textContent || el.innerText || '').trim();
        if (!txt) return null;
        const data = JSON.parse(txt);
        STAGE_MAP = data || null;
        log('[stage-map] parsed');
        return STAGE_MAP;
    } catch (e) {
        log('[stage-map] parse failed: ' + (e?.message || e));
        return null;
    }
}

function resolveStageLabel(keyOrLabel) {
    if (!keyOrLabel) return '';
    try {
        if (STAGE_MAP) {
            // Accept shapes: { stages: ["Label", ...] }
            if (Array.isArray(STAGE_MAP.stages)) {
                const s = STAGE_MAP.stages.find((s) => {
                    if (typeof s === 'string') return s.toLowerCase() === String(keyOrLabel).toLowerCase();
                    if (s && s.key) return String(s.key).toLowerCase() === String(keyOrLabel).toLowerCase() || String(s.label || '').toLowerCase() === String(keyOrLabel).toLowerCase();
                    return false;
                });
                if (s) return (typeof s === 'string') ? s : (s.label || s.key || String(keyOrLabel));
            }
            // Accept mapping objects { key: label }
            if (typeof STAGE_MAP === 'object') {
                if (STAGE_MAP[keyOrLabel]) return STAGE_MAP[keyOrLabel];
                if (STAGE_MAP.labels && STAGE_MAP.labels[keyOrLabel]) return STAGE_MAP.labels[keyOrLabel];
            }
        }
    } catch (e) {
        log('[stage-map] resolve failed: ' + (e?.message || e));
    }
    return friendlyStageLabel(keyOrLabel);
}

// ---------- Canonical stage timeline helpers (new) ----------
// Minimal, opt-in: the DOM may provide a #stage-timeline container; if absent
// all timeline operations are no-ops (we still compute progress internally).
const CANONICAL_STAGES = ['Discovery', 'Targets', 'Propose', 'Approval', 'Apply', 'Quality'];
// Map some common event types to canonical stages as a fallback mapping.
const EVENT_TYPE_STAGE_MAP = {
    // discovery / scanning
    scan: 'Discovery', structure: 'Discovery', projects: 'Discovery', project_selected: 'Discovery',
    // planning / analysis
    analysis: 'Targets', analysis_plan: 'Targets', analysis_result: 'Targets',
    // proposing changes
    recommendations: 'Propose', proposed: 'Propose', diff: 'Propose', diffs: 'Propose', diff_ready: 'Propose',
    // approval flow
    approval_requested: 'Approval', approval_summary: 'Approval', approval_summary: 'Approval',
    // apply / result
    apply: 'Apply', result: 'Apply', done: 'Apply', end: 'Apply',
    // quality / validation
    validation_start: 'Quality', validation_done: 'Quality', validation_error: 'Quality', validation_repair: 'Quality', checks_result: 'Quality'
};

// Cache timeline DOM references
let __stageTimelineEl = null;
let __stageBadgeMap = {}; // key -> element

function getStageTimelineContainer() {
    try {
        if (__stageTimelineEl) return __stageTimelineEl;
        // Prefer data-aidev-target then id 'stage-timeline', then 'sse-stage-timeline', then data-aidev-target="sse-stage-timeline"
        let el = null;
        try { el = document.querySelector('[data-aidev-target="stage-timeline"]'); } catch { el = null; }
        if (!el) {
            try { el = document.getElementById('stage-timeline'); } catch { el = null; }
        }
        if (!el) {
            try { el = document.getElementById('sse-stage-timeline'); } catch { el = null; }
        }
        if (!el) {
            try { el = document.querySelector('[data-aidev-target="sse-stage-timeline"]'); } catch { el = null; }
        }
        __stageTimelineEl = el || null;
        return __stageTimelineEl;
    } catch (e) { return null; }
}

function createStageBadge(stageLabel) {
    const el = document.createElement('div');
    el.className = 'stage-badge stage--pending';
    el.dataset.stage = stageLabel;
    el.dataset.state = 'pending';
    el.setAttribute('role', 'status');
    el.setAttribute('aria-live', 'polite');
    el.title = stageLabel;

    const label = document.createElement('div');
    label.className = 'stage-label';
    label.textContent = stageLabel;
    el.appendChild(label);

    const ts = document.createElement('div');
    ts.className = 'stage-ts muted';
    ts.textContent = '';
    el.appendChild(ts);

    return el;
}

function initStageTimeline() {
    try {
        const mount = getStageTimelineContainer();
        __stageBadgeMap = {};
        if (!mount) return;

        // If mount already has badges (from server-rendered HTML), build the map rather than wiping content.
        try {
            const existing = Array.from(mount.querySelectorAll('.stage-badge, .stage'));
            if (existing && existing.length) {
                for (const el of existing) {
                    try {
                        // Determine label from data-stage or attribute or text
                        const label = (el.dataset && el.dataset.stage)
                            || el.getAttribute && el.getAttribute('data-stage')
                            || (el.textContent || '').trim().split('\n')[0] || null;
                        if (!label) continue;
                        const key = String(label).toLowerCase();
                        __stageBadgeMap[key] = el;
                        // Ensure state dataset exists
                        if (!el.dataset) el.dataset = {};
                        if (!el.dataset.state) el.dataset.state = el.classList.contains('stage--done') ? 'done' : (el.classList.contains('stage--active') ? 'active' : (el.classList.contains('stage--error') ? 'error' : 'pending'));
                        // Normalize classes
                        if (!el.classList.contains('stage-badge')) el.classList.add('stage-badge');
                    } catch (e) { /* ignore per-element */ }
                }
            }
        } catch (e) { /* ignore */ }

        // Append any missing canonical stages (idempotent)
        for (const s of CANONICAL_STAGES) {
            try {
                const key = s.toLowerCase();
                if (!__stageBadgeMap[key]) {
                    const badge = createStageBadge(s);
                    mount.appendChild(badge);
                    __stageBadgeMap[key] = badge;
                }
            } catch (e) { /* ignore per-stage */ }
        }
    } catch (e) {
        log('[timeline] init failed: ' + (e?.message || e));
    }
}

function getStageBadgeElement(stageLabel) {
    if (!stageLabel) return null;
    const key = String(stageLabel || '').toLowerCase();
    if (__stageBadgeMap && __stageBadgeMap[key]) return __stageBadgeMap[key];

    // Fallback: try to find in DOM by data-stage, data-stage attr or normalized text
    try {
        const mount = getStageTimelineContainer();
        if (!mount) return null;
        const candidates = Array.from(mount.querySelectorAll('.stage-badge, .stage'));
        for (const el of candidates) {
            try {
                const lab = (el.dataset && el.dataset.stage) || (el.getAttribute && el.getAttribute('data-stage')) || (el.textContent || '').trim();
                if (!lab) continue;
                if (String(lab).toLowerCase() === key || String(lab).toLowerCase().includes(key)) {
                    __stageBadgeMap[key] = el;
                    return el;
                }
            } catch (e) { /* ignore per-candidate */ }
        }
    } catch (e) { /* ignore */ }
    return null;
}

function setStageState(stageLabel, state, meta = {}) {
    try {
        if (!stageLabel) return false;
        const badge = getStageBadgeElement(stageLabel) || (function() {
            // If container exists but badge missing, create a transient badge appended to timeline.
            const mount = getStageTimelineContainer();
            if (!mount) return null;
            const b = createStageBadge(stageLabel);
            mount.appendChild(b);
            __stageBadgeMap[String(stageLabel).toLowerCase()] = b;
            return b;
        })();
        if (!badge) return false;

        const prev = badge.dataset && badge.dataset.state ? badge.dataset.state : 'pending';
        if (badge.dataset) badge.dataset.state = state || 'pending';
        // normalize classes
        try {
            badge.classList.remove('stage--pending', 'stage--active', 'stage--done', 'stage--error');
            badge.classList.add(`stage--${state}`);
        } catch (e) { /* ignore classList failures */ }

        // Update timestamp text (try several selectors)
        try {
            const tsEl = badge.querySelector('.stage-ts, .ts, .stage-time');
            if (tsEl) {
                const when = meta && meta.ts ? new Date(meta.ts).toLocaleTimeString() : new Date().toLocaleTimeString();
                tsEl.textContent = state === 'pending' ? '' : when;
            }
        } catch (e) { /* ignore ts */ }

        // Recompute overall progress: done stages / total
        try {
            const pct = computeProgressFromStates();

            // Update compact SSE progress elements explicitly
            try {
                const sseBar = document.getElementById('sse-progress-bar');
                if (sseBar && ('value' in sseBar)) sseBar.value = pct;
            } catch (e) { /* ignore */ }
            try {
                const ssePct = document.getElementById('sse-progress-percent');
                if (ssePct) ssePct.textContent = String(pct) + '%';
            } catch (e) { /* ignore */ }
            try {
                const els = Array.from(document.querySelectorAll('.sse-progress, .aidev-progress'));
                for (const el of els) {
                    try { el.style.setProperty('--sse-progress', `${pct}%`); } catch (e) { }
                }
            } catch (e) { /* ignore */ }

            updateProgressBar(pct, `${Math.round(pct)}%`);
            try { if (typeof window.updateTopProgress === 'function') window.updateTopProgress(stageLabel, 0, CANONICAL_STAGES.length, pct); } catch (e) { }
        } catch (e) { /* ignore progress errors */ }

        return true;
    } catch (e) {
        log('[timeline] setStageState failed: ' + (e?.message || e));
        return false;
    }
}

function computeProgressFromStates() {
    try {
        const total = CANONICAL_STAGES.length || 1;
        let done = 0;
        for (const s of CANONICAL_STAGES) {
            try {
                const badge = getStageBadgeElement(s);
                if (!badge) continue;
                const st = (badge.dataset && badge.dataset.state) ? badge.dataset.state : 'pending';
                if (st === 'done' || st === 'error') done += 1;
            } catch (e) { /* ignore per-stage */ }
        }
        return Math.round((done / total) * 100);
    } catch (e) { return 0; }
}

function safeResolveStageFromEvent(ev) {
    try {
        if (!ev) return null;
        const t = (ev.type || ev.event || '').toString();
        // 1) explicit payload.stage / payload.where
        const payload = ev.payload || ev.data || ev || {};
        const explicit = payload.stage || payload.where || payload.where_to || payload.step || null;
        if (explicit) {
            const label = resolveStageLabel(explicit);
            // Map to canonical if similar
            for (const s of CANONICAL_STAGES) {
                if (s.toLowerCase() === String(label || '').toLowerCase()) return s;
                if (String(label || '').toLowerCase().includes(s.toLowerCase())) return s;
            }
            return label || null;
        }

        // 2) try mapping from event type
        if (t && EVENT_TYPE_STAGE_MAP[t]) return EVENT_TYPE_STAGE_MAP[t];

        // 3) try schema STAGE_MAP if present (labels array)
        if (STAGE_MAP && Array.isArray(STAGE_MAP.stages)) {
            const found = STAGE_MAP.stages.find(s => String(s).toLowerCase().includes(t.toLowerCase()));
            if (found) return found;
        }

        return null;
    } catch (e) {
        return null;
    }
}

// One-time attach for Other events toggle wiring
let __otherEventsToggleWired = false;
function attachOtherEventsToggle() {
    try {
        if (__otherEventsToggleWired) return;
        __otherEventsToggleWired = true;
        const btn = document.getElementById('other-events-toggle');
        const container = document.getElementById('other-events');
        const content = document.getElementById('other-events-content');
        if (!btn || !container || !content) return;
        // Initialize aria-expanded
        const expanded = container.getAttribute('aria-expanded') === 'true';
        btn.setAttribute('aria-pressed', expanded ? 'true' : 'false');
        btn.addEventListener('click', (e) => {
            try {
                const isOpen = container.getAttribute('aria-expanded') === 'true';
                container.setAttribute('aria-expanded', isOpen ? 'false' : 'true');
                content.hidden = isOpen ? true : false;
                btn.setAttribute('aria-pressed', (!isOpen) ? 'true' : 'false');
            } catch (err) { /* ignore */ }
        });
    } catch (e) { /* ignore */ }
}

function appendUnknownEvent(ev) {
    try {
        // Ensure toggle wiring is present so UI can be expanded when new items arrive
        try { attachOtherEventsToggle(); } catch (e) { }

        const list = document.getElementById('other-events-list') || document.querySelector('[data-aidev-target="other-events-list"]') || document.getElementById('other-events-content') || document.getElementById('other-events');
        const txt = (ev && typeof ev === 'object') ? JSON.stringify(ev, null, 2) : String(ev || '');
        const when = new Date().toLocaleTimeString();
        if (list) {
            const entry = document.createElement('div');
            entry.className = 'unknown-event recent-event';
            entry.setAttribute('role', 'listitem');
            // header with timestamp and event type
            const h = document.createElement('div');
            h.className = 'unknown-event-header muted';
            try { h.textContent = when + ' — ' + String(ev && (ev.type || ev.event) ? (ev.type || ev.event) : '(unknown)'); } catch { h.textContent = when; }
            const pre = document.createElement('pre');
            pre.className = 'unknown-event-payload';
            pre.style.whiteSpace = 'pre-wrap';
            pre.textContent = txt;
            entry.appendChild(h);
            entry.appendChild(pre);
            list.appendChild(entry);
            try { if (typeof list.scrollTop === 'number') list.scrollTop = list.scrollHeight; } catch (e) { }

            // If the container is collapsed, open it so user can see the new event
            try {
                const container = document.getElementById('other-events');
                const content = document.getElementById('other-events-content');
                if (container && content) {
                    container.setAttribute('aria-expanded', 'true');
                    content.hidden = false;
                    const toggle = document.getElementById('other-events-toggle');
                    if (toggle) toggle.setAttribute('aria-pressed', 'true');
                }
            } catch (e) { /* ignore */ }
            return;
        }
        // Fallback to technical run log
        log('[other-event] ' + when + ' ' + (ev && ev.type ? ev.type : '') + '\n' + (txt || '(no data)'));
    } catch (e) { /* ignore */ }
}

function showStageError(stageLabel, message) {
    try {
        if (stageLabel) setStageState(stageLabel, 'error', { ts: Date.now() });
        // Prefer new sse-banner if present
        const sseBanner = document.getElementById('sse-banner');
        if (sseBanner) {
            try {
                const textEl = document.getElementById('sse-banner-text');
                if (textEl) textEl.textContent = String(message || 'An error occurred');
                try { sseBanner.removeAttribute('hidden'); } catch (e) { try { sseBanner.style.display = 'none'; } catch {} }
                sseBanner.classList.remove('status-banner--hidden');
                sseBanner.classList.add('status-banner--error');
                // attach close handler
                const closeBtn = document.getElementById('sse-banner-close');
                if (closeBtn) {
                    // remove previous handlers by cloning the node (safe one-time attach)
                    try {
                        const nb = closeBtn.cloneNode(true);
                        closeBtn.parentNode.replaceChild(nb, closeBtn);
                        nb.addEventListener('click', () => {
                            try { sseBanner.setAttribute('hidden', ''); } catch (e) { try { sseBanner.style.display = 'none'; } catch {} }
                            try { sseBanner.classList.add('status-banner--hidden'); } catch (e) { }
                        });
                    } catch (e) {
                        // best-effort: fallback to direct handler
                        try { closeBtn.onclick = () => { sseBanner.setAttribute('hidden', ''); sseBanner.classList.add('status-banner--hidden'); }; } catch (ee) { }
                    }
                }
                return;
            } catch (e) { /* fall through to legacy handling */ }
        }

        // Fallback to generic showStatusBanner
        try { showStatusBanner(String(message || 'An error occurred'), 'error'); } catch (e) { try { showError(String(message || 'An error occurred'), { variant: 'error' }); } catch { } }
    } catch (e) { /* ignore */ }
}

// Expose for debugging/tests
try { if (typeof window !== 'undefined') { window.setStageState = setStageState; window.safeResolveStageFromEvent = safeResolveStageFromEvent; } } catch (e) { /* ignore */ }

function updateTopProgress(stageKeyOrLabel, stepIndex, totalSteps, pctForTop) {
    try {
        const label = resolveStageLabel(stageKeyOrLabel || '');
        let pct = 0;
        if (typeof pctForTop === 'number' && !isNaN(pctForTop)) pct = clampPct(pctForTop);
        else if (typeof stepIndex === 'number' && typeof totalSteps === 'number' && totalSteps > 0) pct = clampPct(Math.round((stepIndex / totalSteps) * 100));

        // Try a few common ids the HTML may include; fallback to existing progressBar/progressLabel
        const topBar = document.getElementById('top-progress') || document.getElementById('top-progress-bar') || progressBar;
        if (topBar) {
            try {
                const tag = (topBar.tagName || '').toLowerCase();
                if (tag === 'progress' || tag === 'meter') {
                    try { topBar.value = pct; } catch (e) { /* ignore */ }
                } else {
                    try { topBar.style.width = pct + '%'; } catch (e) { /* ignore */ }
                }
            } catch (e) { /* ignore */ }
        }

        const topLabel = document.getElementById('top-progress-label') || document.getElementById('top-progress-text') || progressLabel;
        if (topLabel) {
            try { topLabel.textContent = label ? `${label} — ${pct}%` : `${pct}%`; } catch (e) { /* ignore */ }
        }
    } catch (e) {
        log('[top-progress] failed: ' + (e?.message || e));
    }
}
function updateTopProgressLabel(stageLabel) { updateTopProgress(stageLabel, 0, 0, 0); }

// small debounce helper used for resize handlers and similar
function debounce(fn, wait = 150) {
    let t = null;
    return (...args) => {
        if (t) clearTimeout(t);
        t = setTimeout(() => {
            try { fn(...args); } catch (e) { /* swallow errors */ }
            t = null;
        }, wait);
    };
}

// Centralized selector map and DOM cache (initialized by initUI)
const SELECTOR_MAP = {
    msg: 'msg',
    send: 'send',
    status: 'status',
    progressBar: 'progress-bar',
    progressLabel: 'progress-label',
    progressMini: 'progress-mini',
    runLogPre: 'run-log-pre',
    connDot: 'conn-dot',
    connStatus: 'conn-status',
    session: 'session',
    currentProject: 'current-project',
    approveBtn: 'approve-changes-btn',
    rejectBtn: 'reject-changes-btn',
    btnRefreshCards: 'btn-refresh-cards',
    btnAiSummarizeChanged: 'btn-ai-summarize-changed',
    btnAiDeep: 'btn-ai-deep',
    aiModel: 'ai-model',
    topProgressBar: 'top-progress-bar',
    topProgressStep: 'top-progress-step',
    diffs: 'diffs',
    chatFeed: 'chat-feed',
    botTyping: 'bot-typing',
    recommendationsPanel: 'recommendations-panel'
};
const DOM = {};

// New: cache DOM by preferring data-aidev-target attribute, then id fallback
function cacheDOM() {
    try {
        for (const [k, id] of Object.entries(SELECTOR_MAP)) {
            try {
                // Prefer data-aidev-target first
                const sel = `[data-aidev-target="${id}"]`;
                let el = null;
                try { el = document.querySelector(sel); } catch (err) { el = null; }
                if (!el) {
                    try { el = document.getElementById(id); } catch { el = null; }
                }
                DOM[k] = el || null;
            } catch { DOM[k] = null; }
        }
    } catch (e) { /* ignore */ }
}

const pretty = (obj) => JSON.stringify(obj, null, 2);
// Prefer DOM cache, fall back to data-aidev-target selector, then getElementById for safety
const $ = (id) => {
    try {
        if (DOM && DOM[id]) return DOM[id];
        const sel = `[data-aidev-target="${id}"]`;
        const q = document.querySelector(sel);
        if (q) return q;
        return document.getElementById(id);
    } catch (e) { try { return document.getElementById(id); } catch { return null; } }
};

// ---------- SSE backoff constants (tunable) ----------
const SSE_BACKOFF_BASE_MS = 800; // initial base delay
const SSE_BACKOFF_MAX_MS = 30_000; // cap
const SSE_BACKOFF_JITTER_MS = 600; // jitter
// New constants for attempts cap and debounce
const SSE_BACKOFF_MAX_ATTEMPTS = 8; // maximum reconnect attempts before giving up
const SSE_DEBOUNCE_MS = 200; // coalesce bursty SSE updates to <= 5fps

const log = (t) => {
    // Ensure the technical details panel (collapsed) exists so non-technical users
    // don't see logs by default. When expanded, logs are monospace and grouped
    // by phase headers like "Planning", "Generating edits", etc.
    ensureTechnicalDetailsPanel();
    const el = document.getElementById('log');
    if (!el) return;

    const line = (typeof t === 'string' ? t : JSON.stringify(t));

    // Try to extract a short phase hint like "status", "plan", "diff", etc.
    const m = line.match(/^\s*\[([^\]]+)\]/);
    const phaseRaw = m ? m[1] : null;
    const phase = phaseRaw ? friendlyStageLabel(phaseRaw) : null;

    // If phase changed, insert a header
    if (phase && el.dataset.lastPhase !== phase) {
        el.dataset.lastPhase = phase;
        const hdr = document.createElement('div');
        hdr.className = 'run-log-phase';
        hdr.textContent = `--- ${phase} ---`;
        hdr.style.fontWeight = '600';
        hdr.style.margin = '8px 0 4px 0';
        hdr.style.color = '#333';
        el.appendChild(hdr);
    }

    const lineEl = document.createElement('div');
    lineEl.className = 'run-log-line';
    lineEl.textContent = line;
    el.appendChild(lineEl);
    // Auto-scroll the log when visible
    try { el.scrollTop = el.scrollHeight; } catch { }
};

// ---------- simple showError helper (renders to banner or falls back) ----------
function showError(message, { variant = 'error', target = 'banner' } = {}) {
    try {
        // Prefer explicit data-aidev-target banner element
        const selector = `[data-aidev-target="${target}"]`;
        let el = null;
        try { el = document.querySelector(selector); } catch { el = null; }
        if (el) {
            // Apply a light class for variant if possible
            try { el.textContent = String(message || ''); } catch (e) { /* ignore */ }
            try { el.classList.remove('status-banner--hidden'); } catch { }
            try { el.classList.add(`status-banner--${variant}`); } catch { }
            return;
        }
    } catch (e) { /* ignore */ }

    // Fallback to the older status/banner helper
    try { showStatusBanner(String(message || ''), variant); } catch (e) { /* ignore */ }
}

// Prefer DOM cache, fall back to getElementById for safety

// ---------- SSE announcer helpers (accessibility) ----------
function ensureSSEAnnouncer() {
    try {
        let el = document.getElementById('sse-announcer');
        if (el) return el;
        el = document.createElement('div');
        el.id = 'sse-announcer';
        el.className = 'visually-hidden';
        el.setAttribute('aria-live', 'polite');
        el.setAttribute('aria-atomic', 'true');
        document.body.appendChild(el);
        return el;
    } catch (e) {
        return null;
    }
}
function announceSSE(text, politeness = 'polite') {
    const el = ensureSSEAnnouncer();
    if (!el) return;
    try {
        el.setAttribute('aria-live', politeness);
        el.textContent = String(text || '');
    } catch (e) { /* ignore */ }
}

// ---------- event schema integration ----------

// Default hard-coded list (what you already have in attachSSEHandlers)
const DEFAULT_EVENT_TYPES = [
    "hello", "ping", "assistant", "status", "intent",
    "recommendations",
    "projects", "project_selected",
    "proposed",
    "diff", "diffs", "diff_ready",
    "error", "result", "done", "end",
    "approval_summary",
    "approval_requested",
    "qa_mode_start",
    "qa_answer",
    "qa_mode_done",
    "analysis_plan",
    "analysis_result",
    "analysis",
    "validation_repair",
    "validation_start",
    "validation_done",
    "validation_error",
    "checks_result",
    // New event types we may receive when patch application falls back
    "patch_apply_failed",
    "fallback_full_content"
];

let EVENT_SCHEMA = null;
let KNOWN_EVENT_TYPES = [...DEFAULT_EVENT_TYPES];

function mergeKnownEventTypesFromSchema(schema) {
    if (!schema || typeof schema !== 'object') return;

    // Try top-level: properties.type.enum
    let enums = null;
    if (Array.isArray(schema?.properties?.type?.enum) && schema.properties.type.enum.length) {
        enums = schema.properties.type.enum;
    }

    // Fallback: definitions.event.properties.type.enum
    if (!enums && Array.isArray(schema?.definitions?.event?.properties?.type?.enum)) {
        enums = schema.definitions.event.properties.type.enum;
    }

    if (Array.isArray(enums) && enums.length) {
        const merged = new Set([...KNOWN_EVENT_TYPES, ...enums.map(String)]);
        KNOWN_EVENT_TYPES = Array.from(merged);
        log('[schema] loaded event types from schema: ' + KNOWN_EVENT_TYPES.join(', '));
    } else {
        log('[schema] events.schema.json loaded but no type enum found; using defaults.');
    }
}

async function loadEventSchema() {
    try {
        const res = await fetchWithTimeout('/schemas/events.schema.json', {
            method: 'GET',
            credentials: 'same-origin'
        }, 12_000);
        if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        EVENT_SCHEMA = data;
        mergeKnownEventTypesFromSchema(data);
    } catch (e) {
        log('[schema] failed to load events.schema.json: ' + (e?.message || e));
        // Surface a light error to the user when schema can't be loaded (non-fatal)
        try { showError('Failed to load event schema (non-fatal).', { variant: 'warning' }); } catch (ee) { }
    }
}

function getKnownEventTypes() {
    return KNOWN_EVENT_TYPES;
}

function isKnownEventType(type) {
    if (!type) return false;
    return getKnownEventTypes().includes(String(type));
}

// Ensure a collapsed "Technical details / Run log" panel exists and is collapsed by default.
function ensureTechnicalDetailsPanel() {
    if (document.getElementById('technical-details')) return document.getElementById('technical-details');
    try {
        const details = document.createElement('details');
        details.id = 'technical-details';
        details.className = 'panel technical-details';
        // collapsed by default
        // summary
        const summary = document.createElement('summary');
        summary.textContent = 'Technical details / Run log';
        summary.style.cursor = 'pointer';
        summary.style.fontWeight = '600';
        summary.style.marginBottom = '6px';
        details.appendChild(summary);

        // content container: monospace, subtle background
        const container = document.createElement('div');
        container.id = 'log';
        container.className = 'run-log-container';
        container.style.fontFamily = 'ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", "Courier New", monospace';
        container.style.fontSize = '13px';
        container.style.lineHeight = '1.4';
        container.style.padding = '8px';
        container.style.background = '#f7f7f9';
        container.style.border = '1px solid #e6e6e9';
        container.style.borderRadius = '6px';
        container.style.maxHeight = '28vh';
        container.style.overflow = 'auto';
        container.style.whiteSpace = 'pre-wrap';
        container.dataset.lastPhase = '';
        details.appendChild(container);

        // Run log panel (preformatted) + Refresh control
        const runPanel = document.createElement('div');
        runPanel.id = 'run-log-panel';
        runPanel.className = 'panel run-log-panel';
        runPanel.style.marginTop = '8px';
        runPanel.setAttribute('role', 'region');
        runPanel.setAttribute('aria-live', 'polite');

        // header row: title + refresh button
        const headerRow = document.createElement('div');
        headerRow.style.display = 'flex';
        headerRow.style.justifyContent = 'space-between';
        headerRow.style.alignItems = 'center';
        headerRow.style.marginBottom = '6px';

        const title = document.createElement('div');
        title.textContent = 'Run log';
        title.style.fontWeight = '600';
        headerRow.appendChild(title);

        const refreshBtn = document.createElement('button');
        refreshBtn.id = 'run-log-refresh';
        refreshBtn.type = 'button';
        refreshBtn.className = 'btn btn-xs';
        refreshBtn.textContent = 'Refresh';
        refreshBtn.title = 'Refresh run log';
        refreshBtn.onclick = () => {
            // fire-and-forget, errors are handled inside the fetch
            try { fetchAndRenderRunLog(); } catch (e) { log('[run-log] refresh failed: ' + (e?.message || e)); }
        };
        headerRow.appendChild(refreshBtn);

        runPanel.appendChild(headerRow);

        const pre = document.createElement('pre');
        pre.id = 'run-log-pre';
        pre.className = 'run-log-pre';
        pre.style.whiteSpace = 'pre-wrap';
        pre.style.background = '#fff';
        pre.style.padding = '8px';
        pre.style.border = '1px solid #e6e6e9';
        pre.style.borderRadius = '6px';
        pre.style.maxHeight = '28vh';
        pre.style.overflow = 'auto';
        pre.textContent = '(no run log loaded)';
        runPanel.appendChild(pre);

        details.appendChild(runPanel);

        // When user opens the details panel, try to load the run log
        details.addEventListener('toggle', () => {
            if (details.open) {
                // fetch when opened; swallow errors
                try { fetchAndRenderRunLog().catch(() => {}); } catch { }
            }
        });

        // try to append near the end of main content if present, otherwise body
        const main = document.querySelector('main') || document.body;
        main.appendChild(details);
        return details;
    } catch (e) {
        return null;
    }
}

// Run log fetching and rendering helpers
let __runLogLoading = false;

function getRunLogEndpoint() {
    // Priority: data-run-log-endpoint on #run-log-panel, then any element with data-run-log-endpoint,
    // then window.AIDEV_RUN_LOG_ENDPOINT, then default /api/run-log
    try {
        const panel = document.getElementById('run-log-panel');
        if (panel && panel.dataset && panel.dataset.runLogEndpoint) return panel.dataset.runLogEndpoint;
        const any = document.querySelector('[data-run-log-endpoint]');
        if (any && any.dataset && any.dataset.runLogEndpoint) return any.dataset.runLogEndpoint;
        if (window && window.AIDEV_RUN_LOG_ENDPOINT) return window.AIDEV_RUN_LOG_ENDPOINT;
    } catch { }
    return '/api/run-log';
}

// New: small central fetchWithTimeout using AbortController
async function fetchWithTimeout(url, options = {}, timeoutMs = 12_000) {
    const ctrl = new AbortController();
    const signal = ctrl.signal;
    const opts = { ...(options || {}), signal };
    let id = null;

    try {
        // Inject X-AIDEV-PROJECT header for same-origin requests when a project is selected.
        // Privacy note: only attach activeProjectPath to same-origin requests to avoid
        // leaking local filesystem paths to external hosts.
        try {
            const ap =
                (typeof activeProjectPath !== 'undefined' && typeof activeProjectPath === 'string' && activeProjectPath.trim())
                    ? activeProjectPath
                    : (localStorage.getItem('aidev.activeProjectPath') || '');

            if (ap) {
                try {
                    const resolved = new URL(url, location.origin);
                    if (resolved.origin === location.origin) {
                        const hdrs = new Headers(opts.headers || {});
                        if (!hdrs.has('X-AIDEV-PROJECT')) {
                            hdrs.set('X-AIDEV-PROJECT', ap);
                        }
                        opts.headers = hdrs;
                    }
                } catch (e) {
                    // If URL resolution fails, avoid adding the header
                }
            }
        } catch (e) { /* swallow header injection errors */ }

        if (typeof timeoutMs === 'number' && timeoutMs > 0) {
            id = setTimeout(() => ctrl.abort(), timeoutMs);
        }

        const res = await fetch(url, opts);
        return res;
    } finally {
        if (id !== null) clearTimeout(id);
    }
}

async function fetchRunLogRaw(url, timeoutMs = 60_000) {
    incBusy('Loading run log…');
    try {
        const res = await fetchWithTimeout(url, { credentials: 'same-origin' }, timeoutMs);
        const text = await res.text();
        // even on non-2xx we still return the body so UI can show server message
        return text;
    } catch (err) {
        if (err && err.name === 'AbortError') {
            const e = new Error(`Timed out after ${Math.round(timeoutMs / 1000)}s`);
            e.code = 'ETIMEDOUT';
            // Surface a compact user-facing message
            try { showError(`Run log timed out after ${Math.round(timeoutMs / 1000)}s`, { variant: 'warning' }); } catch (ee) { }
            throw e;
        }
        // Non-timeout failures: surface a small banner and rethrow
        try { showError('Failed to fetch run log (see console)'); } catch (ee) { }
        throw err;
    } finally {
        decBusy();
    }
}

async function fetchAndRenderRunLog({ force = false } = {}) {
    // Avoid concurrent fetches
    if (__runLogLoading && !force) return;
    __runLogLoading = true;
    const pre = document.getElementById('run-log-pre');
    const refreshBtn = document.getElementById('run-log-refresh');
    if (refreshBtn) refreshBtn.disabled = true;
    try {
        const endpoint = getRunLogEndpoint();
        // Build URL; if we have an activeProjectId, add it as a query param
        const url = new URL(endpoint, location.origin);
        if (activeProjectId) url.searchParams.set('project_id', activeProjectId);
        // Show a lightweight loading state immediately
        if (pre) pre.textContent = 'Loading…';

        let text = '';
        try {
            text = await fetchRunLogRaw(url.toString(), 15000);
        } catch (err) {
            log('[run-log] fetch failed: ' + (err?.message || err));
            if (pre) pre.textContent = `Run log is not available right now (${err?.message || err}).`;
            return;
        }

        if (text === null || typeof text !== 'string' || text.trim() === '') {
            if (pre) pre.textContent = 'No run log available';
        } else {
            let rendered = text;
            // Try to pretty-print JSON (what /api/run-log returns)
            try {
                const json = JSON.parse(text);
                rendered = JSON.stringify(json, null, 2);
            } catch {
                // not JSON; keep as-is
            }
            if (pre) pre.textContent = rendered;
        }
    } catch (e) {
        log('[run-log] render failed: ' + (e?.message || e));
    } finally {
        __runLogLoading = false;
        if (refreshBtn) refreshBtn.disabled = false;
    }
}

// Onboarding banner: shown until dismissed (persisted in localStorage)
function ensureOnboardingBanner() {
    // Use the same key as the onboarding modal flow so dismissal is consistent
    const key = 'aidev_onboard_shown';
    try {
        if (safeLocalStorageGet(key) === '1') return null;
    } catch (e) {
        // fall back to trying direct localStorage access but don't throw
        try { if (localStorage.getItem && localStorage.getItem(key) === '1') return null; } catch (ee) { /* ignore */ }
    }
    if (document.getElementById('aidev-onboard-banner')) return document.getElementById('aidev-onboard-banner');

    const banner = document.createElement('div');
    banner.id = 'aidev-onboard-banner';
    banner.setAttribute('role', 'region');
    banner.setAttribute('aria-live', 'polite');
    banner.style.background = '#fff8e6';

    banner.style.border = '1px solid #ffe6a7';
    banner.style.padding = '12px';
    banner.style.borderRadius = '8px';
    banner.style.margin = '8px 0';
    banner.style.display = 'flex';
    banner.style.justifyContent = 'space-between';
    banner.style.alignItems = 'flex-start';

    const text = document.createElement('div');
    text.style.flex = '1';
    text.style.marginRight = '12px';
    text.textContent = 'Welcome! Start by telling the bot what you want to improve. It will propose a few changes, show you what’s going to change, and then ask for your approval before updating your files.';

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = 'Got it';
    btn.className = 'btn';
    btn.onclick = () => {
        try {
            safeLocalStorageSet(key, '1');
            banner.remove();
        } catch (e) { banner.style.display = 'none'; }
    };

    banner.appendChild(text);
    banner.appendChild(btn);

    // Try to insert near the top of main content or body
    const headerMount = document.querySelector('#app-header') || document.body;
    headerMount.insertBefore(banner, headerMount.firstChild);
    return banner;
}

// Safe localStorage helpers (graceful in browsers without localStorage)
function safeLocalStorageGet(key) {
    try { if (typeof localStorage !== 'undefined') return localStorage.getItem(key); } catch (e) { }
    return null;
}
function safeLocalStorageSet(key, value) {
    try { if (typeof localStorage !== 'undefined') { localStorage.setItem(key, String(value)); return true; } } catch (e) { }
    return false;
}

// Advanced mode persistence helpers (use 'advancedMode' with 'true'/'false'; fall back to legacy 'aidev:advanced')
function readAdvancedMode() {
    try {
        const v = safeLocalStorageGet('advancedMode');
        if (v === 'true') return true;
        if (v === 'false') return false;
        const legacy = safeLocalStorageGet('aidev:advanced');
        if (legacy === '1') return true;
        if (legacy === '0') return false;
    } catch (e) { /* ignore */ }
    return false;
}
function writeAdvancedMode(open) {
    try { safeLocalStorageSet('advancedMode', open ? 'true' : 'false'); } catch (e) { /* ignore */ }
    try { safeLocalStorageSet('aidev:advanced', open ? '1' : '0'); } catch (e) { /* ignore */ }
}

// Ensure onboarding modal behavior: open on first visit (no flag), dismiss persists flag.
function initOnboardingModal() {
    try {
        const KEY = 'aidev_onboard_shown';
        const modal = document.getElementById('onboarding-modal');
        if (!modal) return;

        // If the flag is set, do nothing
        const shown = safeLocalStorageGet(KEY) === '1';
        if (shown) return;

        // Wire buttons (if present) to dismiss and persist the flag.
        const getStarted = document.getElementById('onboarding-get-started');
        const skipBtn = document.getElementById('onboarding-skip');

        const dismiss = (focusPrimary = false) => {
            // Attempt to set the flag; if it fails, still close the modal gracefully.
            try { safeLocalStorageSet(KEY, '1'); } catch (e) { /* swallow */ }
            try { closeModal('onboarding-modal'); } catch (e) { modal.classList.remove('open'); modal.setAttribute('aria-hidden', 'true'); }
            if (focusPrimary) {
                try {
                    const primary = normalizePrimaryCTA();
                    if (primary && typeof primary.focus === 'function') primary.focus();
                } catch (e) { /* ignore focus errors */ }
            }
        };

        if (getStarted) {
            getStarted.addEventListener('click', (e) => { e.preventDefault(); dismiss(true); });
            // Ensure keyboard activation for non-button elements handled globally
        }
        if (skipBtn) {
            skipBtn.addEventListener('click', (e) => { e.preventDefault(); dismiss(false); });
        }

        // Open the modal for first-time users. Using openModal helper if available
        try { openModal('onboarding-modal'); } catch (e) { modal.classList.add('open'); modal.setAttribute('aria-hidden', 'false'); }
    } catch (e) {
        // Never throw from onboarding initialization
        log('[onboard] init failed: ' + (e?.message || e));
    }
}

// Ensure advanced panels are collapsed by default and provide a visible toggle
function initCollapsedAdvancedPanels() {
    try {
        const selectors = ['.advanced-panel', '.tools-panel', '.panel.advanced', '[data-advanced-panel]'];
        const panels = new Set();
        selectors.forEach(sel => {
            try {
                document.querySelectorAll(sel).forEach(el => panels.add(el));
            } catch (e) { /* ignore selector errors */ }
        });

        let idCounter = 0;
        panels.forEach((panel) => {
            try {
                // Ensure the collapsed class is present by default
                if (!panel.classList.contains('collapsed')) panel.classList.add('collapsed');

                // Ensure an id so toggle can reference aria-controls
                if (!panel.id) panel.id = `adv-panel-${Date.now().toString(36)}-${++idCounter}`;

                // If a toggle is already present (avoid duplicates), skip creating another
                // Accept either a toggle inside the panel or immediately before it.
                const existing = panel.querySelector('.tools-toggle') || panel.previousElementSibling;
                if (existing && existing.classList && existing.classList.contains('tools-toggle')) return;

                // Create a visible toggle control just before the panel
                const toggle = document.createElement('button');
                toggle.type = 'button';
                toggle.className = 'tools-toggle btn btn-sm ghost';
                toggle.setAttribute('aria-controls', panel.id);
                toggle.setAttribute('aria-expanded', 'false');
                toggle.textContent = 'Show advanced';
                toggle.style.margin = '6px 0';

                toggle.addEventListener('click', () => {
                    const isCollapsed = panel.classList.toggle('collapsed');
                    toggle.setAttribute('aria-expanded', String(!isCollapsed));
                    toggle.textContent = isCollapsed ? 'Show advanced' : 'Hide advanced';
                    // If it's a <details> element, also open/close it for consistency
                    try {
                        if (panel.tagName && panel.tagName.toLowerCase() === 'details') {
                            panel.open = !isCollapsed;
                        }
                    } catch (e) { /* ignore */ }
                });

                // Insert the toggle immediately before the panel so it's visually adjacent
                try { panel.parentNode.insertBefore(toggle, panel); } catch (e) { /* ignore */ }
            } catch (e) { /* swallow per-panel errors */ }
        });
    } catch (e) {
        log('[advanced-panels] init failed: ' + (e?.message || e));
    }
}

// Ensure we register the advanced toggle wiring on DOMContentLoaded as well as via boot
// so that different load orders are robust. The init function guards against double-init.
try {
    if (typeof document !== 'undefined') {
        document.addEventListener('DOMContentLoaded', () => {
            try { initAdvancedToggle(); } catch (e) { log('[advanced-toggle] DOMContentLoaded init failed: ' + (e?.message || e)); }
        });
    }
} catch (e) { /* ignore */ }

// Auto-approve helper
function isAutoApproveEnabled() {
    const el = document.getElementById('auto-approve-changes');
    return !!(el && el.checked);
}

// Run mode helper
function getRunMode() {
    const sel = $('run-mode');
    const val = sel && sel.value ? sel.value : 'auto';
    // Guard against unknown values
    if (val === 'qa' || val === 'analyze' || val === 'edit' || val === 'auto') return val;
    return 'auto';
}

// Safe element builder for dynamic text (prevents HTML injection)
function E(tag, attrs = {}, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === 'className') el.className = v;
        else if (k === 'text') el.textContent = v;
        else if (k === 'html') el.innerHTML = v; // only use with trusted static strings
        else el.setAttribute(k, String(v));
    }
    for (const c of children) {
        if (c == null) continue;
        if (typeof c === 'string') el.appendChild(document.createTextNode(c));
        else el.appendChild(c);
    }
    return el;
}

// Spinner / busy manager
let __busyCount = 0;
let __prevFocused = null;
// Controls to mark disabled during busy states (add more IDs here as UI grows)
// Use a single canonical window-scoped array when possible so multiple script
// inclusions (or other modules) don't cause duplicate-const errors.
try {
    if (typeof window !== 'undefined') {
        window.AIDEV_CONTROL_IDS = window.AIDEV_CONTROL_IDS || [
            'gen-map', 'btn-refresh-cards', 'btn-ai-summarize-changed', 'btn-ai-deep', 'ai-model',
            'btn-update-descriptions', 'btn-run-checks', 'send', 'msg',
            'update-submit', 'checks-approve', 'checks-reject',
            // Primary CTA: prefer new stable id but keep backward-compatible fallback
            'primary-start-cta', 'primary-cta'
        ];
    }
} catch (e) { /* ignore if window isn't writable */ }
const AIDEV_CONTROL_IDS = (typeof window !== 'undefined' && Array.isArray(window.AIDEV_CONTROL_IDS))
    ? window.AIDEV_CONTROL_IDS
    : [
        'gen-map', 'btn-refresh-cards', 'btn-ai-summarize-changed', 'btn-ai-deep', 'ai-model',
        'btn-update-descriptions', 'btn-run-checks', 'send', 'msg',
        'update-submit', 'checks-approve', 'checks-reject',
        'primary-start-cta', 'primary-cta'
    ];

// New helpers to normalize primary CTA id and to hide/de-emphasize secondary header controls
const normalizePrimaryCTA = (typeof window !== 'undefined' && typeof window.normalizePrimaryCTA === 'function')
    ? window.normalizePrimaryCTA
    : function() {
    try {
        const el = document.getElementById('primary-start-cta') || document.getElementById('primary-cta');
        if (!el) return null;
        // Normalize id to the new stable hook so other scripts can rely on it.
        if (el.id !== 'primary-start-cta') {
            try { el.id = 'primary-start-cta'; } catch { /* ignore */ }
        }
        // Make sure it's keyboard friendly: Enter activates
        el.setAttribute('role', el.getAttribute('role') || 'button');
        if (!el.hasAttribute('tabindex')) el.setAttribute('tabindex', '0');
        // Ensure an accessible label exists
        try {
            if (!el.hasAttribute('aria-label') && (el.textContent || '').trim()) {
                el.setAttribute('aria-label', (el.textContent || '').trim());
            }
        } catch (e) { /* ignore */ }
        // Add a subtle class we can target from CSS
        el.classList.add('primary-start-cta');
        // Ensure pressing Enter or Space triggers click for non-button elements
        const handler = (e) => {
            const key = e.key || e.code || '';
            if (key === 'Enter' || key === ' ' || key === 'Spacebar' || key === 'Space') {
                try { e.preventDefault(); el.click(); } catch { }
            }
        };
        el.addEventListener('keydown', handler);
        return el;
    } catch (e) {
        // Never let normalization throw to console
        return null;
    }
};
try { if (typeof window !== 'undefined') window.normalizePrimaryCTA = normalizePrimaryCTA; } catch {}

const enforceHeaderControlVisibility = (typeof window !== 'undefined' && typeof window.enforceHeaderControlVisibility === 'function')
    ? window.enforceHeaderControlVisibility
    : function(limit = 3) {
    try {
        const header = document.querySelector('header');
        if (!header) return;
        const primary = document.getElementById('primary-start-cta') || document.getElementById('primary-cta');
        // Candidate controls (preserve primary always)
        const nodeList = Array.from(header.querySelectorAll('button, [role="button"], a[href], input, select'));
        // Filter to visible and not the primary CTA
        const candidates = nodeList.filter((el) => {
            if (!el) return false;
            if (primary && el === primary) return false;
            if (el.classList && el.classList.contains('secondary-hidden')) return false; // already marked
            try {
                const cs = getComputedStyle(el);
                if (!cs) return true;
                if (cs.display === 'none' || cs.visibility === 'hidden' || parseFloat(cs.opacity || '1') === 0) return false;
            } catch (e) { /* ignore */ }
            return true;
        });

        // Count visible controls including the primary CTA (if present)
        const visibleCount = candidates.length + (primary ? 1 : 0);
        if (visibleCount <= limit) return;

        // Number of non-primary controls allowed
        const allowed = Math.max(0, limit - (primary ? 1 : 0));
        // Hide the extras (keep the earliest in DOM order visible)
        const toHide = candidates.slice(allowed);
        toHide.forEach((el) => {
            try {
                el.classList.add('secondary-hidden');
                // For robustness, also set inline style so missing CSS rules still hide
                el.style.display = 'none';
            } catch (e) { /* ignore */ }
        });
    } catch (e) {
        // swallow errors — do not introduce console noise
    }
};
try { if (typeof window !== 'undefined') window.enforceHeaderControlVisibility = enforceHeaderControlVisibility; } catch {}

function ensureSpinnerElement() {
    let sp = $('aidev-spinner');
    if (!sp) {
        // Build a minimal accessible spinner overlay. Styling/positioning
        // is delegated to style.css (class .aidev-spinner, .aidev-spinner.visible, etc.)
        sp = E('div', { id: 'aidev-spinner', role: 'status', 'aria-live': 'polite', 'aria-atomic': 'true', className: 'aidev-spinner' },
            E('div', { className: 'aidev-spinner-inner' },
                // The actual visual spinner can be implemented in CSS via .aidev-spinner-icon
                E('span', { className: 'aidev-spinner-icon', text: '' }),
                // Visible text that's announced by screen readers and also readable
                E('span', { className: 'aidev-spinner-text', text: 'Thinking…' })
            )
        );
        // Ensure the spinner is appended near the end of body so it overlays UI
        try { document.body.appendChild(sp); } catch (e) { /* ignore if DOM not ready */ }
    }
    return sp;
}

function updateControlsForBusy(disabled) {
    for (const id of AIDEV_CONTROL_IDS) {
        const el = $(id);
        if (!el) continue;
        if (disabled) {
            el.classList.add('aidev-disabled');
            try { el.disabled = true; } catch { el.setAttribute('disabled', 'true'); }
        } else {
            el.classList.remove('aidev-disabled');
            try { el.disabled = false; } catch { el.removeAttribute('disabled'); }
        }
    }
}

function showSpinner(label) {
    const sp = ensureSpinnerElement();
    if (!sp) return;
    if (label) {
        const txt = sp.querySelector('.aidev-spinner-text');
        if (txt) txt.textContent = String(label || 'Thinking…');
    }
    // record focus to restore later if appropriate
    try {
        if (!__prevFocused && document.activeElement && document.activeElement !== document.body) {
            __prevFocused = document.activeElement;
        }
    } catch { /* ignore */ }
    sp.classList.add('visible');
    document.documentElement.classList.add('aidev-spinner-visible');
    updateControlsForBusy(true);
}

function hideSpinner() {
    const sp = $('aidev-spinner');
    if (sp) sp.classList.remove('visible');
    document.documentElement.classList.remove('aidev-spinner-visible');
    updateControlsForBusy(false);
    // Restore focus to the message input if it was previously focused
    try {
        if (__prevFocused && typeof __prevFocused.focus === 'function') {
            __prevFocused.focus();
        } else {
            const msg = $('msg'); if (msg) msg.focus();
        }
    } catch { /* ignore */ }
    __prevFocused = null;
}

function incBusy(label) {
    if (__busyCount === 0) showSpinner(label || 'Thinking…');
    __busyCount += 1;
}
function decBusy() {
    __busyCount = Math.max(0, __busyCount - 1);
    if (__busyCount === 0) hideSpinner();
}

// ---------- status banner (validation / self-heal) ----------
function getStatusBannerElements() {
    const banner = $('status-banner');
    const textEl = $('status-banner-text');
    return { banner, textEl };
}

function showStatusBanner(message, variant = 'warning') {
    const { banner, textEl } = getStatusBannerElements();
    if (!banner || !textEl) return;

    // Reset variant classes and show the banner
    banner.classList.remove(
        'status-banner--hidden',
        'status-banner--info',
        'status-banner--success',
        'status-banner--warning',
        'status-banner--error'
    );
    banner.classList.add(`status-banner--${variant}`);
    textEl.textContent = message || '';
}

function hideStatusBanner() {
    const { banner } = getStatusBannerElements();
    if (!banner) return;
    banner.classList.add('status-banner--hidden');
    // We intentionally keep the last message text so there’s no flicker
}

// endpoint timeouts (ms)
const TIMEOUTS = {
    scan: 20_000,
    structure: 120_000,
    refreshCards: 120_000,
    summarizeChanged: 1_200_000,
    enrich: 1_500_000,
    chat: 60_000,
    apply: 60_000,
    select: 60_000,
    create: 120_000,
    updateDescriptions: 900_000,
    runChecks: 600_000,
    approvePlan: 120_000
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// robust fetch helper so 500s don't blow up JSON.parse
async function fetchJSON(url, options, timeoutMs = 15000) {
    incBusy();
    let res, text;
    try {
        res = await fetchWithTimeout(url, options, timeoutMs);
        const ct = (res.headers.get('content-type') || '').toLowerCase();
        text = await res.text();
        if (ct.includes('application/json')) {
            const data = JSON.parse(text || '{}');
            if (!res.ok) {
                const detail = typeof data?.detail === 'string' ? data.detail : text;
                throw new Error(detail || 'Request failed');
            }
            return data;
        } else {
            if (!res.ok) throw new Error(text || 'Request failed');
            try { return JSON.parse(text || '{}'); } catch { return { raw: text }; }
        }
    } catch (err) {
        if (err && err.name === 'AbortError') {
            const e = new Error(`Timed out after ${Math.round((timeoutMs || 0) / 1000)}s`);
            e.code = 'ETIMEDOUT';
            try { showError(`Request timed out after ${Math.round((timeoutMs || 0) / 1000)}s`); } catch (ee) { }
            throw e;
        }
        // Surface a light user-facing message
        try { showError('Network request failed — see log.'); } catch (ee) { }
        throw err;
    } finally {
        decBusy();
    }
}

async function withBusyButton(btnEl, fn) {
    if (!btnEl) return fn();
    const prevDisabled = btnEl.disabled;
    btnEl.disabled = true;
    try { return await fn(); }
    finally { btnEl.disabled = prevDisabled; }
}

// ---------- global state ----------
let sessionId = null;
let es = null;                // EventSource (job-scoped)
let currentJobId = null;      // Active job we're streaming
let proposed = [];
let cardsIndex = {};

// New: keep last quality-gate / consistency-checks payload
let lastQualityChecks = null;
let lastQualityChecksSummary = null;

// new UX state
let activeProjectId = null;
let activeProjectPath = null;
try { activeProjectId = safeLocalStorageGet('aidev.activeProjectId') || null; } catch (e) { activeProjectId = null; }
try { activeProjectPath = safeLocalStorageGet('aidev.activeProjectPath') || null; } catch (e) { activeProjectPath = null; }

// derived convenience
let projectSelected = !!activeProjectPath;
let projectBooted = false;

// progress UI state
const progressBar = $('progress-bar');
const progressLabel = $('progress-label');
const progressMini = $('progress-mini');
let progressItems = []; // [{file, detail, ts}]
// Track whether the current run should do a post-apply sync
let needsPostApplySync = false;

// focus management for modals
let __lastFocusedBeforeModal = null;

// ---------- project description state (app_descrip + compiled brief) ----------
let currentAppDescription = '';   //.contents of app_descrip.txt (plain text)
let currentCompiledBrief = '';    // optional AI-compiled summary, not user-editable

let lastBundle = null;

function extractAppDescriptionFromPayload(data) {
    if (!data || typeof data !== 'object') return '';
    const p = data.project || data;
    return (
        p.app_description ??
        p.app_descrip ??
        p.description ??
        data.app_description ??
        data.app_descrip ??
        data.description ??
        ''
    ) || '';
}

function extractCompiledBriefFromPayload(data) {
    if (!data || typeof data !== 'object') return '';
    const p = data.project || data;
    return (
        p.compiled_brief ??
        p.project_description ??
        p.project_description_md ??
        data.compiled_brief ??
        data.project_description ??
        data.project_description_md ??
        ''
    ) || '';
}

function setAppDescription(text) {
    currentAppDescription = text || '';
    const preview = document.getElementById('description-preview');
    if (preview) {
        preview.textContent = currentAppDescription ||
            '(No description yet. Click "Edit description" to add one.)';
    }
}

function setCompiledBrief(text) {
    currentCompiledBrief = text || '';
    const pre = document.getElementById('compiled-brief');
    const wrapper = document.getElementById('compiled-brief-section');
    if (pre) {
        pre.textContent = currentCompiledBrief ||
            '(The AI-compiled brief will appear here after you save your description.)';
    }
    if (wrapper) {
        if (!currentCompiledBrief) wrapper.classList.add('empty-brief');
        else wrapper.classList.remove('empty-brief');
    }
}

// ---------- session ----------
async function ensureSession() {
    if (sessionId) return sessionId;
    const data = await fetchJSON('/session/new', { method: 'POST' }, TIMEOUTS.select);
    sessionId = data.session_id || data.id || data.session || data.sessionId;
    const s = $('session');
    if (s && sessionId) s.textContent = (sessionId || '').slice(0, 8);
    // No SSE here anymore; streams are opened per-job (see connectJobStream)
    setControlsEnabled(false);
    return sessionId;
}

function ensureApprovalSummary() {
    let panel = document.getElementById('approval-summary');
    if (panel) return panel;
    panel = document.createElement('div');
    panel.id = 'approval-summary';
    panel.className = 'panel';
    const mount = document.getElementById('diffs');
    if (mount && mount.parentElement) mount.parentElement.insertBefore(panel, mount);
    else document.body.prepend(panel);
    return panel;
}

function riskEmoji(level) {
    return level === 'high' ? '🔴' : (level === 'medium' ? '🟡' : '🟢');
}

async function copyText(t) {
    try { await navigator.clipboard.writeText(t); } catch { }
}

function openInEditor(relPath) {
    const root = (window.localStorage.getItem('aidev.activeProjectPath') || '').replace(/\\/g, '/');
    const full = root ? `${root}/${relPath}` : relPath;
    window.location.href = `vscode://file/${encodeURI(full)}`;
}

function renderApprovalSummary(meta) {
    const panel = ensureApprovalSummary();
    const { summary, risk, files } = meta || {};
    panel.innerHTML = '';

    const head = E('div', { className: 'row' },
        E('span', { className: '' },
            E('strong', { html: `${riskEmoji(risk || 'low')} ${String(risk || 'low').toUpperCase()}` })
        ),
        E('span', { className: 'muted', style: 'margin-left:8px' }, summary || '')
    );
    panel.appendChild(head);

    const list = E('div', { className: 'file-list' });

    if (Array.isArray(files) && files.length) {
        for (const f of files) {
            const details = E('details', { className: 'file-row', open: true });

            // Create buttons first so we can attach handlers without relying on children indices
            const copyBtn = E('button', {
                className: 'btn btn-xs link',
                type: 'button'
            }, 'Copy');

            const openBtn = E('button', {
                className: 'btn btn-xs link',
                type: 'button'
            }, 'Open in editor');

            const s = E('summary', {},
                E('code', {}, f.path || ''),
                ' — ',
                `+${f.added ?? 0} / -${f.removed ?? 0} `,
                copyBtn,
                ' ',
                openBtn
            );

            copyBtn.onclick = (ev) => {
                ev.stopPropagation();
                copyText(f.path || '');
            };

            openBtn.onclick = (ev) => {
                ev.stopPropagation();
                openInEditor(f.path || '');
            };

            details.appendChild(s);

            if (f.why) {
                details.appendChild(
                    E('div', { className: 'muted', style: 'margin-top:2px' }, f.why)
                );
            }

            list.appendChild(details);
        }
    } else {
        list.appendChild(E('div', { className: 'muted' }, '(no files)'));
    }

    panel.appendChild(list);
}

// ---------- SSE helpers ----------
function setConnState(ok) {
    const dot = $('conn-dot');
    if (!dot) return;
    dot.classList.toggle('ok', !!ok);
}

// New: connection status text and richer states
function updateConnStatus(state) {
    // state: 'connected' | 'connecting' | 'disconnected'
    try {
        // Prefer new connection-status target if present
        const el = $('connection-status') || $('conn-status');
        if (el) {
            if (state === 'connected') {
                el.textContent = 'Connected';
                el.setAttribute('aria-live', 'polite');
                el.setAttribute('role', 'status');
            }
            else if (state === 'connecting') {
                el.textContent = 'Reconnecting…';
                el.setAttribute('aria-live', 'polite');
                el.setAttribute('role', 'status');
            }
            else {
                el.textContent = 'Disconnected';
                el.setAttribute('aria-live', 'polite');
                el.setAttribute('role', 'status');
            }
        }
    } catch (e) { /* ignore */ }
    // Keep the simple dot for backward compatibility
    setConnState(state === 'connected');
}

// Keep a per-job processed event id cache to deduplicate repeated events
let __processedEventIds = new Set();
// Reconnect controller for the current EventSource stream
let __esReconnectController = null;

// SSE event coalescing queue (flush debounced)
let __sseEventQueue = [];
const __sseFlushDebounced = debounce(() => flushSSEQueue(), SSE_DEBOUNCE_MS);

function flushSSEQueue() {
    try {
        if (!__sseEventQueue || !__sseEventQueue.length) return;
        // copy and clear early so incoming events aren't blocked on processing
        const items = __sseEventQueue.slice();
        __sseEventQueue.length = 0;
        for (const it of items) {
            try {
                // Ensure we still dedupe by id on flush as well
                if (it && it.__eventId) {
                    if (__processedEventIds.has(it.__eventId)) continue;
                    __processedEventIds.add(it.__eventId);
                }
                handleEvent(it);
            } catch (e) {
                log('[sse] flush handleEvent failed: ' + (e?.message || e));
            }
        }
    } catch (e) {
        log('[sse] flushSSEQueue failed: ' + (e?.message || e));
        // ensure queue cleared so future bursts are processed
        try { __sseEventQueue.length = 0; } catch (_) { }
    }
}

// Safe parse helper: never throws to caller
function safeParseEventData(data) {
    if (!data || typeof data !== 'string') return data;
    try { return JSON.parse(data); } catch (e) { log('[sse] JSON parse failed (falling back to raw): ' + (e?.message || e)); return data; }
}

// Validate an SSE envelope before queueing. Returns normalized object or null.
function validSSEEnvelope(candidate) {
    try {
        if (!candidate || typeof candidate !== 'object') return null;
        // Expect: { type: string, payload: any }
        const t = candidate.type || candidate.event || null;
        if (!t || typeof t !== 'string') return null;
        // Allow empty payloads but ensure a payload property exists
        const payload = (candidate.payload !== undefined) ? candidate.payload : (candidate.data !== undefined ? candidate.data : null);
        // Return normalized shape
        return { type: String(t), payload };
    } catch (e) {
        return null;
    }
}

function attachSSEHandlers(esInstance) {
    // Use schema-derived event types when available; fall back to defaults.
    const eventTypes = getKnownEventTypes();

    // Generic listener that handles named events (via addEventListener) and
    // generic 'message' events. Each listener protects against malformed
    // payloads and ignores duplicates based on a lightweight id.
    for (const t of eventTypes) {
        try {
            esInstance.addEventListener(t, (ev) => {
                try {
                    // Deduplicate: prefer SSE lastEventId, then payload.id, then a quick signature
                    const parsed = safeParseEventData(ev.data);
                    let eventId = ev.lastEventId || (parsed && (parsed.id || parsed.event_id)) || null;
                    if (!eventId) {
                        const raw = typeof ev.data === 'string' ? ev.data : JSON.stringify(ev.data || '');
                        eventId = `${t}:${raw.slice(0, 200)}`;
                    }

                    // Build candidate envelope and validate
                    const candidate = { type: t, payload: parsed };
                    const norm = validSSEEnvelope(candidate);
                    if (!norm) {
                        log(`[sse] ignored invalid named event '${t}'`);
                        return;
                    }

                    // If we've already processed the id, skip queueing
                    if (eventId && __processedEventIds.has(eventId)) return;

                    // Push to coalescing queue
                    const queued = Object.assign({}, norm, { __eventId: eventId });
                    __sseEventQueue.push(queued);
                    __sseFlushDebounced();
                } catch (e) {
                    log('[sse] named listener failed: ' + (e?.message || e));
                }
            });
        } catch (e) {
            // Some environments may not support addEventListener on ES wrapper — ignore
        }
    }

    // Generic onmessage fallback
    esInstance.onmessage = (e) => {
        try {
            const parsed = safeParseEventData(e.data);
            // If parsed is object and has type, pass-through; otherwise create a generic event
            const evt = (parsed && typeof parsed === 'object' && parsed.type) ? parsed : (typeof parsed === 'object' ? parsed : { type: 'message', data: parsed });

            // Deduplicate similar to above
            const eventId = e.lastEventId || (evt && (evt.id || evt.event_id)) || (`message:${String(e.data).slice(0,200)}`);

            const norm = validSSEEnvelope(evt);
            if (!norm) {
                log('[sse] ignored invalid onmessage envelope');
                return;
            }

            if (eventId && __processedEventIds.has(eventId)) return;

            const queued = Object.assign({}, norm, { __eventId: eventId });
            __sseEventQueue.push(queued);
            __sseFlushDebounced();
        } catch (err) {
            // Ensure parsing errors don't bubble up
            log('[sse] onmessage parse failed: ' + (err?.message || err));
        }
    };

    esInstance.onopen = () => {
        log('[sse] connected');
        updateConnStatus('connected');
        // Reset reconnect attempts state if controller exists
        try {
            if (__esReconnectController && typeof __esReconnectController.resetAttempts === 'function') __esReconnectController.resetAttempts();
        } catch (e) { }
    };

    esInstance.onerror = (err) => {
        log('[sse] error (will attempt reconnect)');
        updateConnStatus('connecting');
        // Errors will be surfaced to reconnect controller which may schedule a reconnect
        try {
            if (__esReconnectController && typeof __esReconnectController.onError === 'function') __esReconnectController.onError(err);
        } catch (e) { }
    };
}

// Helper to stop any pending reconnect timer / controller
function stopEsReconnectController() {
    try {
        if (__esReconnectController && typeof __esReconnectController.stop === 'function') __esReconnectController.stop();
    } catch (e) { /* ignore */ }
    __esReconnectController = null;
}

// Job-scoped SSE with exponential backoff + jitter
function connectJobStream(jobId) {
    if (!jobId) return;

    // Stop any previous controller/stream
    stopEsReconnectController();
    try { if (es) { try { es.close(); } catch {} es = null; } } catch {}

    // Reset dedupe set for this job
    __processedEventIds = new Set();

    // Build URL
    const url = new URL(`${location.protocol}//${location.host}/jobs/stream`);
    url.searchParams.set('job_id', jobId);

    // Reconnect controller encapsulates backoff logic
    let attempts = 0;
    let stopped = false;
    let reconnectTimer = null;

    const controller = {
        stop() {
            stopped = true;
            if (reconnectTimer) {
                clearTimeout(reconnectTimer);
                reconnectTimer = null;
            }
        },
        resetAttempts() { attempts = 0; },
        onError(err) {
            // Schedule reconnect depending on attempts; don't reconnect if job changed or stopped
            if (stopped) return;
            attempts += 1;
            // If attempts exceed cap, give up and notify
            if (attempts >= SSE_BACKOFF_MAX_ATTEMPTS) {
                try {
                    log(`[sse] reconnect attempts exceeded (${attempts}); giving up`);
                    updateConnStatus('disconnected');
                    announceSSE('Connection lost; automatic reconnects stopped.');
                    showToast('Connection lost — reconnect attempts exhausted.', 'Dismiss');
                } catch (e) { /* ignore */ }
                controller.stop();
                return;
            }

            const exp = Math.min(SSE_BACKOFF_MAX_MS, Math.pow(2, attempts) * SSE_BACKOFF_BASE_MS);
            // jitter up to SSE_BACKOFF_JITTER_MS
            const jitter = Math.floor(Math.random() * SSE_BACKOFF_JITTER_MS);
            const delay = Math.max(500, Math.min(SSE_BACKOFF_MAX_MS, Math.floor(exp + jitter)));
            log(`[sse] scheduling reconnect attempt #${attempts} in ${delay}ms`);
            updateConnStatus('connecting');

            try { if (es) try { es.close(); } catch {} } catch {}

            if (reconnectTimer) clearTimeout(reconnectTimer);
            reconnectTimer = setTimeout(() => {
                if (stopped) return;
                // Only reconnect if the currentJobId hasn't changed
                if (!currentJobId || String(currentJobId) !== String(jobId)) {
                    log('[sse] not reconnecting because job changed or cleared');
                    return;
                }
                // Attempt a fresh connection
                try {
                    es = new EventSource(url.toString());
                    attachSSEHandlers(es);
                } catch (e) {
                    log('[sse] reconnect failed to create EventSource: ' + (e?.message || e));
                    // schedule next attempt
                    controller.onError(e);
                }
            }, delay);
        }
    };

    __esReconnectController = controller;

    // Open first connection
    try {
        es = new EventSource(url.toString());
        attachSSEHandlers(es);
        // mark status as connecting until onopen fires
        updateConnStatus('connecting');
    } catch (e) {
        log('[sse] failed to open EventSource: ' + (e?.message || e));
        // schedule first reconnect attempt
        controller.onError(e);
    }
}

// Ensure we stop reconnect attempts when a run finishes
function finalizeStreamOnFinish() {
    try {
        stopEsReconnectController();
        if (es) try { es.close(); } catch {}
        es = null;
    } catch (e) { /* ignore */ }
    updateConnStatus('disconnected');
}

// ---------- enable/disable ----------
function setControlsEnabled(enabled) {
    const ids = [
        'approve-changes-btn', 'reject-changes-btn', 'gen-map',
        'btn-refresh-cards', 'btn-ai-summarize-changed', 'btn-ai-deep', 'ai-model',
        'btn-update-descriptions', 'btn-run-checks', 'send', 'msg'
    ];
    ids.forEach(id => {
        const el = $(id);
        if (el) el.disabled = !enabled;
    });
}

// ---------- chat bubbles ----------
function pushBubble(kind, text) {
    const feed = $('chat-feed');
    if (!feed || !text) return;
    const div = document.createElement('div');
    div.className = `bubble ${kind}`;
    // keep assistant messages friendly: strip technical markers when possible
    if (kind === 'assistant') {
        // prefer short plain sentences for user-facing assistant bubbles
        div.innerHTML = text.replace(/\n/g, '<br/>');
    } else {
        div.textContent = text;
    }
    feed.appendChild(div);
    feed.scrollTop = feed.scrollHeight;
}

// Helper: truncate preview/snippet safely to max characters (single-line collapse)
function truncatePreview(text, max = 200) {
    if (!text) return '';
    try {
        let s = String(text).replace(/\s+/g, ' ').trim();
        if (s.length > max) return s.slice(0, max) + '…';
        return s;
    } catch (e) {
        return String(text).slice(0, max);
    }
}

// Helper: render citations (array of {path, snippet, line_start, line_end}) into a container element
function renderCitations(list, container) {
    if (!Array.isArray(list) || !list.length || !container) return;
    try {
        const box = E('div', { className: 'qa-citations', style: 'margin-top:8px' });
        const title = E('div', { className: 'muted', text: 'Cited files:' });
        box.appendChild(title);

        for (const ref of list) {
            try {
                const path = (ref && (ref.path || ref.file)) ? String(ref.path || ref.file) : '';
                if (!path) continue;
                const snippet = truncatePreview(ref.snippet || ref.preview || ref.text || '', 200);

                const row = E('div', { className: 'qa-cite-item', style: 'margin-top:6px' });
                const codeEl = E('code', {}, path);
                codeEl.style.cursor = 'pointer';
                codeEl.title = path;
                codeEl.addEventListener('click', (e) => { e.preventDefault(); try { openInEditor(path); } catch (err) { log('[qa] openInEditor failed: ' + (err?.message || err)); } });
                row.appendChild(codeEl);

                if (ref && (ref.line_start || ref.line_end)) {
                    const ls = ref.line_start ? String(ref.line_start) : '?';
                    const le = ref.line_end ? String(ref.line_end) : '?';
                    const range = E('span', { className: 'muted', style: 'margin-left:8px' }, `lines ${ls}-${le}`);
                    row.appendChild(range);
                }

                if (snippet) {
                    const pre = E('div', { className: 'muted', style: 'margin-top:4px; white-space:pre-wrap' }, snippet);
                    row.appendChild(pre);
                }

                box.appendChild(row);
            } catch (e) {
                log('[qa] renderCitations one ref failed: ' + (e?.message || e));
            }
        }

        container.appendChild(box);
    } catch (e) {
        log('[qa] renderCitations failed: ' + (e?.message || e));
    }
}

// ---------- diff helpers (UI) ----------
function joinPath(a, b) {
    if (!a) return b || '';
    if (!b) return a;
    const sep = a.match(/[\\/]$/) ? '' : '/';
    return (a + sep + b).replace(/[/\\]+/g, '/');
}
function vscodeHref(absPath) {
    try {
        return 'vscode://file/' + encodeURI(absPath);
    } catch {
        return 'vscode://file/' + absPath;
    }
}

// ---------- Analyze Plan (analyze mode, read-only) ----------

function normalizeAnalyzePlan(raw) {
    const plan = raw || {};
    const themes = Array.isArray(plan.themes) ? plan.themes : [];

    const normThemes = themes.map((t, i) => {
        const recs = Array.isArray(t.recommendations) ? t.recommendations : [];
        return {
            id: String(t.id || t.title || `theme-${i + 1}`),
            title: String(t.title || `Theme ${i + 1}`),
            summary: String(t.summary || ''),
            impact: String(t.impact || 'medium'),
            effort: String(t.effort || 'medium'),
            files: Array.isArray(t.files) ? t.files.map(f => String(f || '')) : [],


            notes: String(t.notes || ''),
            recommendations: recs.map((r, j) => ({
                id: String(r.id || r.title || `rec-${i + 1}-${j + 1}`),
                title: String(r.title || 'Recommendation'),
                summary: String(r.summary || ''),
                reason: String(r.reason || ''),
                impact: String(r.impact || 'medium'),
                effort: String(r.effort || 'medium'),
                risk: String(r.risk || 'low'),
                files: Array.isArray(r.files) ? r.files.map(f => String(f || '')) : []
            }))
        };
    });

    return {
        schema_version: plan.schema_version ?? null,
        focus: String(plan.focus || ''),
        overview: String(plan.overview || ''),
        themes: normThemes,
        next_steps: Array.isArray(plan.next_steps)
            ? plan.next_steps.map(s => String(s || ''))
            : []
    };
}

function formatAnalyzePlanForChat(raw) {
    const plan = normalizeAnalyzePlan(raw);
    const lines = [];

    if (plan.focus) {
        lines.push(`Focus: ${plan.focus}`);
    }

    if (plan.overview) {
        if (lines.length) lines.push('');
        lines.push('Overview:');
        lines.push(plan.overview);
    }

    if (plan.themes.length) {
        lines.push('');
        lines.push('Themes:');
        plan.themes.forEach((t, idx) => {
            lines.push(
                `${idx + 1}. ${t.title} [impact: ${t.impact}, effort: ${t.effort}]`
            );
            if (t.summary) {
                lines.push(`   - ${t.summary}`);
            }
            if (t.recommendations && t.recommendations.length) {
                lines.push('   - Recommendations:');
                t.recommendations.slice(0, 3).forEach((r) => {
                    const sub = r.summary || r.reason || '';
                    lines.push(
                        `     • ${r.title}${sub ? ` — ${sub}` : ''}`
                    );
                });
                if (t.recommendations.length > 3) {
                    lines.push(
                        `     • …and ${t.recommendations.length - 3} more`
                    );
                }
            }
            if (t.files && t.files.length) {
                lines.push(
                    `   - Files: ${t.files.join(', ')}`
                );
            }
        });
    }

    if (plan.next_steps.length) {
        lines.push('');
        lines.push('Suggested next steps:');
        plan.next_steps.forEach((step, idx) => {
            lines.push(`  ${idx + 1}. ${step}`);
        });
    }

    if (!lines.length) {
        return 'Here’s a high-level analysis of your project, but the analysis plan was empty.';
    }
    return lines.join('\n');
}

// Give the backend a brief window to send the proposed bundle so the
// "Planned changes" panel can hydrate before we run pre-apply checks.
async function ensurePlannedChangesHydrated({ maxWaitMs = 2000 } = {}) {
    if (proposed && proposed.length) return;
    const start = Date.now();
    while ((!proposed || !proposed.length) && (Date.now() - start) < maxWaitMs) {
        await sleep(150);
    }
    if (!proposed || !proposed.length) {
        log('[approval] planned changes were not ready before checks; continuing anyway.');
    }
}

function setPlannedStatus(text) {
    const el = document.getElementById('planned-changes-status');
    if (el) el.textContent = text || '';
}

function computeDiffStats(diffText) {
    if (!diffText) return { added: 0, removed: 0 };
    let added = 0, removed = 0;
    const lines = diffText.split(/\r?\n/);
    for (const ln of lines) {
        if (!ln) continue;
        if (ln.startsWith('@@') || ln.startsWith('diff ') || ln.startsWith('index ') || ln.startsWith('---') || ln.startsWith('+++')) {
            continue;
        }
        if (ln.startsWith('+')) added += 1;
        else if (ln.startsWith('-')) removed += 1;
    }
    return { added, removed };
}

function renderProposed() {
    const container = document.getElementById('diffs');
    if (!container) return;
    container.innerHTML = '';

    // Optional cross-file notes block (from the last structured bundle, if present)
    if (lastBundle && lastBundle.cross_file_notes) {
        const notes = Array.isArray(lastBundle.cross_file_notes)
            ? lastBundle.cross_file_notes
            : [String(lastBundle.cross_file_notes || '')].filter(Boolean);

        if (notes.length) {
            const box = document.createElement('div');
            box.className = 'cross-file-notes';

            const h = document.createElement('h4');
            h.textContent = 'Cross-file notes';
            box.appendChild(h);

            const ul = document.createElement('ul');
            notes.forEach((n) => {
                const li = document.createElement('li');
                li.textContent = n;
                ul.appendChild(li);
            });
            box.appendChild(ul);

            container.appendChild(box);
        }
    }

    const ap = document.getElementById('approve-changes-btn');
    const rj = document.getElementById('reject-changes-btn');

    if (!proposed.length) {
        if (ap) ap.disabled = true;
        if (rj) rj.disabled = true;
        setPlannedStatus('');
        return;
    }

    // Top "Rationale" box
    const reasons = [];
    for (const it of proposed) {
        const w = (it.why || '').trim();
        if (w && !reasons.includes(w)) reasons.push(w);
    }
    if (reasons.length) {
        const box = document.createElement('div');
        box.className = 'rationale';
        const h = document.createElement('h4');
        h.textContent = 'Rationale';
        const ul = document.createElement('ul');
        reasons.forEach(r => {
            const li = document.createElement('li');
            li.textContent = r;
            ul.appendChild(li);
        });
        box.appendChild(h);
        box.appendChild(ul);
        container.appendChild(box);
    }

    let anyDiffs = false;

    proposed.forEach((it, i) => {
        const path = String(it.path || '');
        const diff = String(it.diff || '');
        const { added, removed } = computeDiffStats(diff);

        if (diff && diff.trim()) anyDiffs = true;

        const details = document.createElement('details');
        details.className = 'diff-panel';
        if (i === 0) details.open = true;

        const summary = document.createElement('summary');
        summary.appendChild(E('strong', {}, path || '(unknown file)'));
        summary.appendChild(document.createTextNode(' '));
        summary.appendChild(
            E('span', { className: 'pill ' + ((added || removed) ? 'changed' : '') },
              `+${added} / -${removed}`)
        );

        const summaryParts = [];
        if (it.summary && String(it.summary).trim()) summaryParts.push(String(it.summary).trim());
        if (it.why && String(it.why).trim()) summaryParts.push(String(it.why).trim());

        if (summaryParts.length) {
            summary.appendChild(document.createTextNode(' '));
            summary.appendChild(
                E('span', { className: 'muted' }, '— ' + summaryParts.join(' • '))
            );
        }

        const controls = document.createElement('div');
        controls.className = 'diff-actions';

        const copyBtn = document.createElement('button');
        copyBtn.className = 'btn btn-xs';
        copyBtn.type = 'button';
        copyBtn.textContent = 'Copy diff';
        copyBtn.onclick = async () => {
            try { await navigator.clipboard.writeText(diff || ''); log('[copy] diff copied: ' + path); }
            catch (e) { log('[copy] failed: ' + (e?.message || e)); }
        };

        const absPath = joinPath(activeProjectPath || '', path);
        const openLink = document.createElement('a');
        openLink.className = 'btn btn-xs link';
        openLink.href = vscodeHref(absPath);
        openLink.textContent = 'Open in editor';
        openLink.title = absPath;

        const approveThis = document.createElement('button');
        approveThis.className = 'btn btn-sm primary';
        approveThis.type = 'button';
        approveThis.textContent = 'Approve this change';
        approveThis.onclick = async () => {
            try {
                if (!currentJobId) throw new Error('No active job to approve');
                await fetchJSON('/jobs/approve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ job_id: currentJobId })
                }, TIMEOUTS.apply);
                showToast('Approved. The change will be applied when ready.');
            } catch (e) { log('[approval] single approve failed: ' + (e?.message || e)); }
        };

        const skipThis = document.createElement('button');
        skipThis.className = 'btn btn-sm ghost';
        skipThis.type = 'button';
        skipThis.textContent = 'Skip this change';
        skipThis.onclick = () => {
            proposed = proposed.filter(p => p.path !== path || p.diff !== diff);
            renderProposed();
            showToast('Skipped this change.');
        };

        const techToggle = document.createElement('button');
        techToggle.className = 'btn btn-xs link';
        techToggle.type = 'button';
        techToggle.textContent = 'Show technical details (code diff)';
        techToggle.onclick = () => {
            const pre = details.querySelector('pre.diff');
            if (!pre) return;
            const isHidden = pre.classList.contains('hidden');
            pre.classList.toggle('hidden');
            techToggle.textContent = isHidden
                ? 'Hide technical details (code diff)'
                : 'Show technical details (code diff)';
        };

        controls.appendChild(copyBtn);
        controls.appendChild(openLink);
        controls.appendChild(approveThis);
        controls.appendChild(skipThis);
        controls.appendChild(techToggle);

        const pre = document.createElement('pre');
        // Use the .hidden helper (CSS-driven) instead of inline style display toggles
        pre.className = 'diff hidden';
        pre.textContent = diff || '(no diff for this file)';

        const body = document.createElement('div');
        body.className = 'diff-body';
        body.appendChild(controls);
        body.appendChild(pre);

        details.appendChild(summary);
        details.appendChild(body);
        container.appendChild(details);
    });

    const ap2 = document.getElementById('approve-changes-btn');
    const rj2 = document.getElementById('reject-changes-btn');
    if (ap2) ap2.disabled = !anyDiffs;
    if (rj2) rj2.disabled = !anyDiffs;

    if (anyDiffs) {
        setPlannedStatus('This run will make ' + proposed.length + ' sets of changes. Review below, then approve or skip.');
    } else {
        setPlannedStatus('');
    }
}

// ---------- progress UI ----------
function clampPct(x) {
    if (typeof x !== 'number' || isNaN(x)) return 0;
    return Math.max(0, Math.min(100, x));
}
function updateProgressBar(pct, label) {
    const v = clampPct(pct);
    if (progressBar) progressBar.value = v;
    if (progressLabel) progressLabel.textContent = label || (v ? `${v}%` : 'Idle');
}
function pushProgressItem(file, detail) {
    if (!file && !detail) return;
    progressItems.unshift({
        file: file || '',
        detail: detail || '',
        ts: Date.now()
    });
    progressItems = progressItems.slice(0, 8);
    renderProgressMini();
}
function renderProgressMini() {
    if (!progressMini) return;
    progressMini.innerHTML = '';
    for (const it of progressItems) {
        const li = document.createElement('li');
        const file = it.file ? `<code>${it.file}</code>` : '';
        const because = it.detail ? it.detail : '';
        li.innerHTML = `${file} ${because ? `<span class="muted">— ${because}</span>` : ''}`;
        progressMini.appendChild(li);
    }
}
function resetProgressSoon() {
    setTimeout(() => updateProgressBar(0, 'Idle'), 800);
}

// unify run finalization for done/end/result
function finishRun(ok) {
    hideStatusBanner();  // clear any validation/self-heal banner

    const label = ok ? 'Finished' : 'Failed';

    // Clear busy state and re-enable chat controls
    setRunBusy(false, label);

    updateProgressBar(100, label);
    resetProgressSoon();
    try { if (es) es.close(); } catch { }
    es = null;
    currentJobId = null;
    planDecision = 'idle';
    planCancelledForRun = false;
    const ap = $('approve-changes-btn'); const rj = $('reject-changes-btn');
    if (ap) ap.disabled = !proposed.length;
    if (rj) rj.disabled = !proposed.length;
    setPlannedStatus('');

    // Stop any reconnect attempts and update connection UI
    try { finalizeStreamOnFinish(); } catch (e) { /* ignore */ }
}

// Quiet helper: refresh cards, update banner, then rebuild project map
async function refreshCardsAndMapQuietly(label = 'post-apply') {
    try {
        await ensureSession();
        const rc = await fetchJSON('/workspaces/refresh-cards', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, force: false })
        }, TIMEOUTS.refreshCards);
        renderProjectMapFromCards(rc.cards_index || {});
        await genProjectMap();
        log(`[sync] ${label}: cards refreshed and project map rebuilt`);
    } catch (e) {
        log(`[sync] ${label} failed: ` + (e?.message || e));
    }
}

// ---------- run-busy / spinner helper ----------
function setRunBusy(isBusy, label) {
    // Status text override (SSE status events can update this later)
    const statusEl = $('status');
    if (statusEl && label) statusEl.textContent = label;

    // Simple CSS hook for spinners etc. (style .job-busy in CSS)
    const root = document.documentElement || document.body;
    if (root) root.classList.toggle('job-busy', !!isBusy);

    // Show/hide the global spinner overlay as appropriate. Note that fetchJSON
    // also increments the global busy count, so the spinner will remain visible
    // while any network operation is in-flight.
    if (isBusy) {
        incBusy(label || 'Thinking…');
    } else {
        // When clearing run-busy, decrement one unit that corresponds to the
        // run-level marker we may have previously incremented.
        decBusy();
    }

    // Keep the per-element send/msg behavior for immediate UX
    const sendEl = $('send');
    const msgEl = $('msg');
    if (sendEl) sendEl.disabled = !!isBusy || !projectSelected;
    if (msgEl) msgEl.disabled = !!isBusy;
}

// ---------- event dispatcher ----------
function handleEvent(ev) {
    const statusEl = $('status');

    // Normalize envelope shapes to be resilient to aligned schema variations.
    // Accept shapes like: { event, data }, { type, payload }, or already-normalized { type, payload }.
    try {
        if (ev && typeof ev === 'object') {
            if (!ev.type && ev.event) ev.type = ev.event;
            if (ev.payload === undefined && ev.data !== undefined) {
                ev.payload = (typeof ev.data === 'string') ? safeParseEventData(ev.data) : ev.data;
            }
            // Keep ev.data present for legacy codepaths
            if (ev.data === undefined && ev.payload !== undefined) ev.data = ev.payload;
        }
    } catch (e) { /* ignore normalization failures */ }

    // Announce important SSE-driven state changes to assistive tech
    try {
        if (ev && ev.type && ['status', 'approval_requested', 'error', 'done', 'result'].includes(ev.type)) {
            const payload = ev.payload || ev.data || ev || {};
            const msg = payload.message || payload.detail || ev.message || ev.msg || ev.type;
            announceSSE(String(msg || ev.type), 'polite');
        }
    } catch (e) { /* ignore announcer failures */ }

    switch (ev?.type) {
        case 'status': {
            const payload = ev.payload || ev.data || ev || {};

            const msg =
                payload.message ??
                payload.detail ??
                ev.msg ??
                ev.message ??
                payload.msg ??
                '';

            const stage =
                payload.stage ||
                payload.where ||
                ev.stage ||
                ev.where ||
                '';

            // Friendly stage label + step numbering
            const friendly = stage ? friendlyStageLabel(stage) : '';
            const idx = friendly ? (stageIndexOf(friendly) || '') : '';
            if (statusEl) {
                if (idx) statusEl.textContent = `Step ${idx} of ${friendlyStages.length} — ${friendly}…`;
                else statusEl.textContent = String(msg || '');
            }

            const pctRaw =
                payload.progress_pct ??
                payload.progress ??
                ev.progress_pct ??
                ev.progress ??
                null;

            if (pctRaw !== null && pctRaw !== undefined) {
                const pctNum = Number(pctRaw);
                const clamped = clampPct(pctNum);
                updateProgressBar(
                    clamped,
                    stage
                        ? `${friendly} — ${Math.round(clamped)}%`
                        : `${Math.round(clamped)}%`
                );

                // Also update compact sse progress (mirror behavior in setStageState)
                try {
                    const sseBar = document.getElementById('sse-progress-bar');
                    if (sseBar && ('value' in sseBar)) sseBar.value = clamped;
                } catch (e) { }
                try {
                    const ssePct = document.getElementById('sse-progress-percent');
                    if (ssePct) ssePct.textContent = String(clamped) + '%';
                } catch (e) { }
                try {
                    const els = Array.from(document.querySelectorAll('.sse-progress, .aidev-progress'));
                    for (const el of els) {
                        try { el.style.setProperty('--sse-progress', `${clamped}%`); } catch (e) { }
                    }
                } catch (e) { }
            }

            // Keep the header top-progress bar in sync, if available
            try {
                if (typeof window.updateTopProgress === 'function') {
                    const totalSteps = friendlyStages.length || 1;
                    const stepIndex = idx || stageIndexOf(friendly) || 1;
                    const pctForTop = (pctRaw !== null && pctRaw !== undefined)
                        ? clampPct(Number(pctRaw))
                        : Math.round((stepIndex / totalSteps) * 100);

                    window.updateTopProgress(
                        stage || friendly || 'working',
                        stepIndex,
                        totalSteps,
                        pctForTop
                    );
                }
            } catch (e) {
                // don't let UI wiring break the stream
                log('[status] updateTopProgress failed: ' + (e?.message || e));
            }

            const file =
                payload.file ||
                payload.path ||
                ev.file ||
                ev.path ||
                '';
            const detail =
                payload.detail ||
                ev.detail ||
                '';

            if (file || detail) pushProgressItem(file, detail);

            // Also update timeline state when stage info is available
            try {
                const resolved = safeResolveStageFromEvent(ev) || (stage ? resolveStageLabel(stage) : null);
                if (resolved) setStageState(resolved, 'active', { ts: Date.now() });
            } catch (e) { /* ignore timeline errors */ }

            log(
                `[status]${stage ? ` [${stage}]` : ''} ${
                    typeof msg === 'string' ? msg : JSON.stringify(msg)
                }`
            );
            break;
        }
        case 'assistant': {
            const text = ev.text ?? ev.data?.text ?? ev.data?.content ?? '';
            if (text) {
                // Strip heavy technical phrasing for the main assistant bubble
                const userFriendly = String(text).replace(/\b(generat\w+|diff|unified|patch)\b/gi, '').trim();
                pushBubble('assistant', userFriendly || text);
                log('[assistant] ' + (userFriendly || text));
            }
            break;
        }
        case 'intent': { log('[intent] ' + pretty(ev.data || ev)); break; }
        case 'projects': {
            const list = Array.isArray(ev.data) ? ev.data : [];
            const wsRoot = localStorage.getItem('aidev.wsroot') || '';
            renderProjects(list, wsRoot);
            populateWizardSelect(list, wsRoot);
            log('[projects] ' + pretty(list.map(p => ({ path: p.path, kind: p.kind, markers: p.markers }))));
            break;
        }
        case 'project_selected': {
            const root = ev.root ?? ev.data?.root ?? ev.data?.path ?? '';
            const pid = ev.project_id ?? ev.data?.project_id ?? null;
            if (pid) activeProjectId = String(pid);
            if (root) activeProjectPath = String(root);

            if (activeProjectPath) localStorage.setItem('aidev.activeProjectPath', activeProjectPath);
            if (activeProjectId) localStorage.setItem('aidev.activeProjectId', activeProjectId);

            const el = $('current-project');
            if (el && root) el.textContent = root;
            projectSelected = true;
            setControlsEnabled(true);
            if (!projectBooted) {
                projectBooted = true;
                postSelectBoot().catch(err => log('[error] postSelectBoot ' + (err?.message || err)));
            }

            // Hydrate description from this event if the backend included it
            const payload = ev.data || ev;
            const desc = extractAppDescriptionFromPayload(payload);
            const brief = extractCompiledBriefFromPayload(payload);

            // Always update, even if empty, so we don't show stale descriptions
            setAppDescription(desc);
            setCompiledBrief(brief);

            const input = $('msg'); if (input) input.focus();

            // Friendlier, less noisy system message
            if (root) {
                pushBubble(
                    'system',
                    `All set — I'm now looking at:\n${root}`
                );
            }

            closeModal('setup-modal');
            log('[project_selected] ' + root);
            break;
        }
        case 'qa_mode_start': {
            const payload = ev.payload || ev.data || ev || {};
            const q = payload.question || payload.prompt || '';

            // Don’t spam the conversation with “Q&A run started…”.
            // Just give a friendly, lightweight status.
            if (statusEl) {
                statusEl.textContent = q
                    ? 'Thinking about your question…'
                    : 'Thinking about your question…';
            }

            log('[qa_mode_start] ' + (q || '(no question attached)'));
            break;
        }
        case 'qa_answer': {
            // Enhanced QA answer rendering: support structured payloads with
            // { answer: string|{answer, file_refs, followups}, file_refs, followups }
            try {
                const payload = ev.payload || ev.data || ev || {};
                // Normalize answer object
                let ansObj = {};
                if (typeof payload.answer === 'string') ansObj.answer = payload.answer;
                else if (payload.answer && typeof payload.answer === 'object') ansObj = payload.answer;
                else if (typeof payload.text === 'string') ansObj.answer = payload.text;
                else ansObj.answer = payload.message || '';

                const answerText = String(ansObj.answer || '').trim();
                const fileRefs = Array.isArray(ansObj.file_refs) ? ansObj.file_refs : (Array.isArray(payload.file_refs) ? payload.file_refs : (Array.isArray(payload.citations) ? payload.citations : []));
                const followups = Array.isArray(ansObj.followups) ? ansObj.followups : (Array.isArray(payload.followups) ? payload.followups : (Array.isArray(payload.suggested_questions) ? payload.suggested_questions : []));

                const feed = $('chat-feed');

                if (!feed || !answerText) {
                    if (answerText) pushBubble('assistant', answerText);
                    else log('[qa_answer] received event with no answer field: ' + JSON.stringify(payload));
                } else {
                    const bubble = document.createElement('div');
                    bubble.className = 'bubble assistant';

                    // Answer text
                    const textEl = E('div', { className: 'qa-answer-text', style: 'white-space:pre-wrap' }, answerText);
                    bubble.appendChild(textEl);

                    // Citations (if any)
                    if (Array.isArray(fileRefs) && fileRefs.length) {
                        renderCitations(fileRefs, bubble);
                        try { log('[qa_answer] file_refs=' + JSON.stringify(fileRefs.map(f => ({ path: f.path || f.file, snippet: truncatePreview(f.snippet || f.preview || '', 200) })))); } catch (e) { /* ignore */ }
                    }

                    // Follow-up quick question buttons (up to 3)
                    if (Array.isArray(followups) && followups.length) {
                        try {
                            const wrap = E('div', { className: 'qa-followups', style: 'margin-top:8px; display:flex; gap:8px; flex-wrap:wrap' });
                            const max = Math.min(3, followups.length);
                            for (let i = 0; i < max; i++) {
                                const q = String(followups[i] || '').trim();
                                if (!q) continue;
                                const btn = E('button', { className: 'btn btn-sm', type: 'button', text: q });

                                // Manual UI smoke test / reproduction steps (developer):
                                // 1) Emit an SSE event of type 'qa_answer' with payload:
                                //    { answer: "...", file_refs: [{ path: "src/x.js", snippet: "..." }], followups: ["Why?", "How?"] }
                                // 2) Confirm the UI displays the cited files (path + snippet) and up to 3 follow-up buttons.
                                // 3) Click a follow-up button and observe an outgoing POST to /conversation with body:
                                //    { session_id: <sessionId>, question: "<button text>" }
                                // This verifies end-to-end wiring for citations and follow-ups.

                                btn.addEventListener('click', async (evBtn) => {
                                    try {
                                        evBtn.preventDefault();
                                    } catch (e) { }
                                    try {
                                        btn.disabled = true;
                                    } catch (e) { }
                                    try {
                                        pushBubble('user', q);
                                    } catch (e) { }
                                    try { setRunBusy(true, 'Thinking about your question…'); } catch (e) { }

                                    // Ensure we have a valid sessionId before sending. If missing,
                                    // surface a user-facing message and re-enable the button.
                                    if (!sessionId) {
                                        try { showToast('Session not initialized — please wait a moment and try again.', 'Dismiss'); } catch (ee) { }
                                        try { btn.disabled = false; } catch (ee) { }
                                        try { setRunBusy(false, 'Idle'); } catch (ee) { }
                                        log('[qa] followup blocked: missing sessionId');
                                        return;
                                    }

                                    try {
                                        // Send follow-up to /conversation (server will stream result back via SSE)
                                        const payloadOut = { session_id: sessionId, question: q };
                                        try { log('[qa] sending follow-up: ' + JSON.stringify(payloadOut)); } catch (e) { }

                                        await fetchJSON('/conversation', {
                                            method: 'POST',
                                            headers: { 'Content-Type': 'application/json' },
                                            body: JSON.stringify(payloadOut)
                                        }, TIMEOUTS.chat);

                                        // On success we leave the run lifecycle to SSE; keep the button disabled to avoid duplicate presses
                                        try { log('[qa] followup sent successfully'); } catch (e) { }
                                    } catch (err) {
                                        // Inform the user and re-enable the button so they can retry
                                        try { showToast('Failed to send follow-up. Please try again.'); } catch (ee) { }
                                        try { btn.disabled = false; } catch (ee) { }
                                        try { setRunBusy(false, 'Idle'); } catch (ee) { }
                                        log('[qa] followup send failed: ' + (err?.message || err) + ' payload=' + q);
                                    }
                                });

                                wrap.appendChild(btn);
                            }
                            bubble.appendChild(wrap);
                        } catch (e) {
                            log('[qa_answer] followups render failed: ' + (e?.message || e));
                        }
                    }

                    feed.appendChild(bubble);
                    try { feed.scrollTop = feed.scrollHeight; } catch (e) { }
                }

                // Clear busy unless a followup triggered a new run (followup click sets busy)
                setRunBusy(false, 'Idle');
                if (statusEl) {
                    statusEl.textContent = 'Done. Ask another question when you’re ready.';
                }

                log('[qa_answer] done');
            } catch (e) {
                log('[qa_answer] render failed: ' + (e?.message || e));
            }
            break;
        }
        case 'qa_mode_done': {
            const payload = ev.payload || ev.data || ev;
            const q = payload.question || '';

            // Previously we added: pushBubble('system', 'Q&A run completed.')
            // That’s redundant with the answer bubble + status text, so drop it
            // to keep the conversation focused.

            // Safety: if for some reason busy wasn’t cleared in qa_answer,
            // make sure it’s cleared here.
            setRunBusy(false, 'Idle');
            log('[qa_mode_done] ' + (q ? `for "${q}"` : ''));
            break;
        }
        case 'analysis_plan':
        case 'analysis_result':
        case 'analysis': {
            const payload = ev.payload || ev.data || ev || {};
            const plan = payload.plan || payload;

            const text = formatAnalyzePlanForChat(plan);

            // Show the analysis as a single assistant message in the chat feed
            pushBubble('assistant', text);

            // Nice to have: log a compact summary
            try {
                const themes = Array.isArray(plan.themes) ? plan.themes.length : 0;
                const steps = Array.isArray(plan.next_steps) ? plan.next_steps.length : 0;
                log(`[analysis] plan received (themes=${themes}, next_steps=${steps})`);
            } catch {
                log('[analysis] plan received');
            }

            // Update timeline: analysis maps to Targets
            try { setStageState('Targets', 'active', { ts: Date.now() }); } catch (e) { }

            // Analyze is read-only; the pipeline will still send result/done to finish the run.
            break;
        }
        case 'recommendations': {
            const payload = ev.payload || ev.data || ev || {};
            const recs =
                payload.recommendations ||
                payload.recs ||
                ev.recommendations ||
                (Array.isArray(payload) ? payload : []);

            const count = Array.isArray(recs) ? recs.length : 0;

            if (count > 0) {
                if (!planCancelledForRun && planDecision !== 'rejected' && planDecision !== 'dismissed') {
                    // Render into the inline recommendations panel instead of a modal
                    try { renderRecommendationsPanel(recs); } catch (e) { log('[recommendations] render failed: ' + (e?.message || e)); }
                } else {
                    log('[recommendations] ignored (user canceled this run)');
                }
            }

            // Mark Propose stage active
            try { setStageState('Propose', 'active', { ts: Date.now() }); } catch (e) { }

            log('[recommendations] ' + count + ' recs');
            break;
        }
        case 'proposed': {
            proposed = Array.isArray(ev.data) ? ev.data : [];
            renderProposed();
            try { setStageState('Propose', 'done', { ts: Date.now() }); } catch (e) { }
            log('[proposed] ' + (proposed.length || 0) + ' items');
            break;
        }
        case 'approval_summary': {
            const data = ev.data || ev.payload || ev;
            renderApprovalSummary(data);
            try { setStageState('Approval', 'done', { ts: Date.now() }); } catch (e) { }
            log('[approval_summary] ' + (data?.summary || ''));
            break;
        }
        case 'approval_requested': {
            const evJobId = ev.job_id ?? ev.data?.job_id ?? null;
            if (evJobId && currentJobId && String(evJobId) !== String(currentJobId)) {
                log(`[approval_requested] ignoring event for job ${evJobId} (current job ${currentJobId})`);
                break;
            }

            const msg = ev.message || ev.data?.message || '';
            log('[approval_requested] ' + (msg || pretty(ev.data || ev)));

            const auto = isAutoApproveEnabled();

            // Give the user an immediate status update while we hydrate diffs + checks.
            if (auto) {
                setPlannedStatus('Running quick checks before auto-applying changes…');
            } else {
                setPlannedStatus('Preparing diffs and running pre-apply checks…');
            }

            // Mark Approval stage active
            try { setStageState('Approval', 'active', { ts: Date.now() }); } catch (e) { }

            // Ensure Planned changes + pre-apply checks hydrate
            // before we expose the approval prompt or auto-approve.
            (async () => {
                try {
                    // 1) Give the backend a brief window to send proposed diffs
                    await ensurePlannedChangesHydrated();

                    // 2) Run pre-apply checks; for auto mode we *don't* open
                    //    the modal automatically, but for manual mode we do.
                    const checks = await doPreapplyChecks({ openModalOnResult: !auto });

                    // Helper to check if we actually have proposed changes.
                    const hasProposed =
                        typeof proposed !== 'undefined' &&
                        Array.isArray(proposed) &&
                        proposed.length > 0;

                    // If we’re in auto-approve mode, only auto-approve when checks pass.
                    if (auto) {
                        if (!checks || !checks.ok) {
                            // Checks failed or missing: fall back to manual approval.
                            if (checks && checks.details) {
                                openChecksModal(checks);
                            }
                            const ap = $('approve-changes-btn');
                            const rj = $('reject-changes-btn');
                            if (ap) ap.disabled = !hasProposed;
                            if (rj) rj.disabled = !hasProposed;
                            setPlannedStatus(
                                'Checks completed, but some issues were found. Review the pre-apply checks and diffs, then approve or skip.'
                            );
                            log('[auto-approve] checks did not pass; falling back to manual approval.');
                            return;
                        }

                        if (statusEl) {
                            statusEl.textContent = 'Automatically applying approved changes…';
                        }
                        setPlannedStatus('Automatically apply changes as they\'re ready (advanced)');
                        log(
                            '[auto-approve] Checkbox is checked; checks passed; auto-approving this job via /jobs/approve.'
                        );
                        approve(true).catch((e) => {
                            log('[auto-approve] approve(true) failed: ' + (e?.message || e));
                        });
                    } else {
                        // Manual mode: enable buttons once checks + planned changes are ready.
                        const ap = $('approve-changes-btn');
                        const rj = $('reject-changes-btn');
                        if (ap) ap.disabled = !hasProposed;
                        if (rj) rj.disabled = !hasProposed;
                        setPlannedStatus(
                            'Checks completed. Review the diffs and pre-apply checks, then Approve Changes or Skip to continue.'
                        );
                    }
                } catch (err) {
                    // If anything in the pre-hydration flow fails, fall back
                    // to the previous simple behavior.
                    log(
                        '[approval_requested] pre-apply wiring failed, falling back: ' +
                            (err?.message || err)
                    );

                    const hasProposed =
                        typeof proposed !== 'undefined' &&
                        Array.isArray(proposed) &&
                        proposed.length > 0;

                    if (auto) {
                        if (statusEl) {
                            statusEl.textContent = 'Automatically applying approved changes…';
                        }
                        setPlannedStatus('Automatically apply changes as they\'re ready (advanced)');
                        approve(true).catch((e) => {
                            log('[auto-approve] approve(true) failed: ' + (e?.message || e));
                        });
                    } else {
                        const ap = $('approve-changes-btn');
                        const rj = $('reject-changes-btn');
                        if (ap) ap.disabled = !hasProposed;
                        if (rj) rj.disabled = !hasProposed;
                        setPlannedStatus(
                            'Waiting for your review. Check the diffs below, then Approve Changes or Skip to continue.'
                        );
                    }
                }
            })();

            break;
        }
        case 'diff':
        case 'diffs':
        case 'diff_ready': {
            const payload = ev.payload || ev.data || ev;
            const d = payload?.unified || ev.unified || '';

            let uni = $('unified-diff');
            if (!uni) {
                uni = document.createElement('pre');
                uni.id = 'unified-diff';
                uni.className = 'diff unified';
                const container = $('diffs');
                if (container) {
                    const header = document.createElement('h4');
                    header.textContent = 'Unified diff (all files)';
                    container.appendChild(header);
                    container.appendChild(uni);
                }
            }
            if (uni) uni.textContent = d || '(no diffs)';

            // If the backend included a structured bundle, hydrate per-file proposals too.
            const bundle = payload?.bundle || ev.bundle;
            if (bundle && Array.isArray(bundle.proposed)) {
                lastBundle = bundle;
                proposed = bundle.proposed;
                renderProposed();
            } else {
                lastBundle = null;
            }

            // Mark Propose stage active/done depending on payload
            try { setStageState('Propose', (d && d.trim()) ? 'done' : 'active', { ts: Date.now() }); } catch (e) { }

            log(`[${ev.type}] ` + (d ? '(received unified diff)' : '(empty)'));
            break;
        }
        case 'validation_repair': {
            const payload = ev.payload || ev.data || ev || {};
            const done = !!(payload.done || payload.finished || payload.ok === true);
            const error = payload.error || payload.message_error || null;

            if (done && !error) {
                hideStatusBanner();
            } else if (error) {
                showStatusBanner(`Self-heal failed: ${error}`, 'error');
            } else {
                const msg =
                    payload.message ||
                    payload.detail ||
                    'Attempting to self-fix issues before applying changes…';
                showStatusBanner(msg, 'warning');
            }

            log('[validation_repair] ' + pretty(payload));
            break;
        }

        case 'validation_start': {
            const payload = ev.payload || ev.data || ev || {};
            const msg =
                payload.message ||
                payload.detail ||
                'Running extra validation before applying changes…';
            showStatusBanner(msg, 'info');
            log('[validation_start] ' + pretty(payload));
            break;
        }

        case 'validation_done': {
            const payload = ev.payload || ev.data || ev || {};
            const ok = !!(payload.ok ?? true);

            if (ok) {
                showStatusBanner('Validation and self-repair complete. Applying changes…', 'success');
                // Hide after a short delay so the user can see the success
                setTimeout(() => hideStatusBanner(), 3000);
            } else {
                showStatusBanner(
                    'Validation finished with issues. Please review the checks panel.',
                    'warning'
                );
            }

            log('[validation_done] ' + pretty(payload));
            break;
        }

        case 'validation_error': {
            const payload = ev.payload || ev.data || ev || {};
            const msg =
                payload.error ||
                payload.message ||
                'Validation encountered an error. Please review the run log.';
            showStatusBanner(msg, 'error');
            // Mark quality stage error
            try { showStageError('Quality', msg); } catch (e) { }
            log('[validation_error] ' + pretty(payload));
            break;
        }
        case 'error': {
            // Normalize payload shape (payload -> data -> ev)
            const payload = (ev && typeof ev === 'object'
                ? (ev.payload || ev.data || ev)
                : {});

            const msg =
                (payload && typeof payload === 'object'
                    ? (
                        payload.error ??
                        payload.message ??
                        payload.detail ??
                        payload.msg
                    )
                    : undefined) ??
                ev.error ??
                ev.message ??
                ev.detail ??
                '(unknown error)';

            const text = typeof msg === 'string' ? msg : JSON.stringify(msg);

            // Surface the error in the UI, similar to validation_error
            try {
                if (typeof showStatusBanner === 'function') {
                    showStatusBanner(text, 'error');
                }
            } catch {
                // Don't let UI wiring break the stream
            }

            // If we can resolve a stage, mark that stage as error
            try {
                const resolved = safeResolveStageFromEvent(ev);
                if (resolved) showStageError(resolved, text);
                else showStageError('Apply', text);
            } catch (e) { /* ignore */ }

            log('[error] ' + text);
            break;
        }
        case 'patch_apply_failed': {
            // New event emitted when a patch failed to apply. Payload may include:
            // { fallback_reason, trace, original_patch, fallback_attempts, full_content }
            try {
                const payload = ev.payload || ev.data || ev || {};
                const reason = payload.fallback_reason || payload.reason || payload.error || payload.message || 'Patch apply failed';
                const trace = payload.trace || payload.trace_log || null;
                const origPatch = payload.original_patch || payload.original || payload.patch || null;
                const attempts = payload.fallback_attempts ?? payload.attempts ?? null;
                const fullContent = payload.full_content || payload.fallback_full_content || null;

                log('[patch_apply_failed] reason=' + String(reason) + ' attempts=' + String(attempts));

                // Surface a visible status and mark the Apply stage as errored
                try { showStatusBanner(`Patch apply failed: ${reason}`, 'error'); } catch (e) { }
                try { showStageError('Apply', reason); } catch (e) { }

                // Append a detailed copy to the technical events panel so users can inspect trace + patch
                try { appendUnknownEvent({ type: ev.type, payload }); } catch (e) { }

                // Also append a compact diagnostic into the run log preformatted area
                try {
                    const pre = document.getElementById('run-log-pre');
                    if (pre) {
                        let entry = '\n--- Patch apply failed ---\n';
                        entry += (reason ? String(reason) + '\n' : '');
                        if (trace) entry += (typeof trace === 'string' ? trace : JSON.stringify(trace, null, 2)) + '\n';
                        if (origPatch) entry += 'Original patch:\n' + String(origPatch) + '\n';
                        if (fullContent) entry += 'Fallback full content (truncated):\n' + String(fullContent).slice(0, 2000) + '\n';
                        pre.textContent = (pre.textContent || '') + entry;
                    }
                } catch (e) { /* ignore */ }

                // If backend provided full_content/fallback, surface it into a dedicated pre block near diffs
                if (fullContent) {
                    try {
                        const container = document.getElementById('diffs');
                        let fb = document.getElementById('fallback-full-content');
                        if (!fb) {
                            fb = document.createElement('pre');
                            fb.id = 'fallback-full-content';
                            fb.className = 'diff fallback';
                            if (container) {
                                const header = document.createElement('h4');
                                header.textContent = 'Fallback full file content (patch failed)';
                                container.appendChild(header);
                                container.appendChild(fb);
                            } else {
                                const runPre = document.getElementById('run-log-pre');
                                if (runPre && runPre.parentNode) runPre.parentNode.appendChild(fb);
                                else document.body.appendChild(fb);
                            }
                        }
                        fb.textContent = String(fullContent || '(no full content)');
                    } catch (e) { /* ignore */ }
                }
            } catch (e) {
                log('[patch_apply_failed] handler failed: ' + (e?.message || e));
            }
            break;
        }
        case 'fallback_full_content': {
            // Emitted when system falls back to providing full file content as a substitute
            // for a patch. Payload should include full_content plus optional metadata.
            try {
                const payload = ev.payload || ev.data || ev || {};
                const content = payload.full_content || payload.content || '';
                const reason = payload.fallback_reason || payload.reason || '';

                log('[fallback_full_content] ' + (reason || 'fallback to full content'));

                try {
                    const container = document.getElementById('diffs');
                    let pre = document.getElementById('fallback-full-content');
                    if (!pre) {
                        pre = document.createElement('pre');
                        pre.id = 'fallback-full-content';
                        pre.className = 'diff fallback';
                        if (container) {
                            const header = document.createElement('h4');
                            header.textContent = 'Fallback: full file content (patch failed)';
                            container.appendChild(header);
                            container.appendChild(pre);
                        } else {
                            const runPre = document.getElementById('run-log-pre');
                            if (runPre && runPre.parentNode) runPre.parentNode.appendChild(pre);
                            else document.body.appendChild(pre);
                        }
                    }
                    pre.textContent = content || '(no content provided)';
                } catch (e) { /* ignore */ }

                showToast('Patch failed; showing fallback full content', 'Dismiss');
            } catch (e) {
                log('[fallback_full_content] handler failed: ' + (e?.message || e));
            }
            break;
        }
        case 'result': {
            const ok = ev.ok ?? ev.data?.ok ?? true;
            // Attempt to mark resolved stage done
            try {
                const resolved = safeResolveStageFromEvent(ev) || 'Apply';
                if (ok) setStageState(resolved, 'done', { ts: Date.now() });
                else setStageState(resolved, 'error', { ts: Date.now() });
            } catch (e) { /* ignore */ }

            finishRun(!!ok);
            if (ok && needsPostApplySync) {
                // Fire-and-forget; don't block UI
                refreshCardsAndMapQuietly('post-apply').catch(() => {});
                needsPostApplySync = false;
            }
            log('[result] ok=' + (!!ok));
            break;
        }
        case 'done':
        case 'end': {
            // Stage-level "done" events fire for things like plan, checks, etc.
            // Only treat as full pipeline completion when `where` (or `stage`)
            // indicates the whole job, not just a single phase.
            const payload = ev.payload || ev.data || {};
            const where =
                payload.where ||
                payload.stage ||
                ev.where ||
                ev.stage ||
                '';

            const ok = payload.ok ?? ev.ok ?? ev.data?.ok ?? true;

            const isPipelineLevel =
                !where || where === 'pipeline' || where === 'orchestrator' || where === 'job';

            try {
                const resolved = safeResolveStageFromEvent(ev) || (where ? resolveStageLabel(where) : null);
                if (resolved) setStageState(resolved, ok ? 'done' : 'error', { ts: Date.now() });
            } catch (e) { /* ignore */ }

            if (isPipelineLevel) {
                finishRun(!!ok);
                if (ok && needsPostApplySync) {
                    // Fire-and-forget; don't block UI
                    refreshCardsAndMapQuietly('post-apply').catch(() => {});
                    needsPostApplySync = false;
                }
            }

            log(`[${ev.type}] where=${where || '(none)'} ok=${!!ok}`);
            break;
        }
        default: {
            if (ev && typeof ev === 'object' && ev.type) {
                if (!isKnownEventType(ev.type)) {
                    log('[event][unknown-type] ' + ev.type + ' ' + pretty(ev));
                    try { appendUnknownEvent(ev); } catch (e) { /* ignore */ }
                } else {
                    log('[event] ' + ev.type + ' ' + pretty(ev));
                }
            } else {
                log(ev);
            }
        }
    }
}

// ---------- Structure & Cards ----------
async function genProjectMap() {
    const statusEl = $('map-status');
    const jsonEl = $('project-map-json');
    const genMapBtn = $('gen-map');
    if (statusEl) statusEl.textContent = `Generating… (timeout ${Math.round(TIMEOUTS.structure / 1000)}s)`;

    if (jsonEl) jsonEl.textContent = '(working…)';
    await withBusyButton(genMapBtn, async () => {
        try {
            await ensureSession();
            const url = new URL('/api/project-map', location.origin);
            url.searchParams.set('session_id', sessionId);
            const data = await fetchJSON(url.toString(), { method: 'GET' }, TIMEOUTS.structure);
            if (statusEl) statusEl.textContent = 'Complete';
            if (jsonEl) jsonEl.textContent = JSON.stringify(data, null, 2);
            log('[project-map] refreshed');
        } catch (e) {
            if (statusEl) statusEl.textContent = '';
            if (jsonEl) jsonEl.textContent = '(failed)';
            log('[error] project-map ' + (e?.message || e));
        }
    });
}

function deriveAiSourceSha(node) {
    if (node && typeof node.ai_summary === 'object' && node.ai_summary !== null) {
        if (node.ai_summary.file_sha) return String(node.ai_summary.file_sha);
    }
    return (node.ai_summary_sha || node.ai_sha || node.ai_file_sha || node.ai_source_sha || '');
}
function currentFileSha(node) { return node.file_sha || node.sha || node.sha256 || ''; }
function computeCounts(idx) {
    const values = Object.values(idx || {});
    const total = values.length;
    let changed = 0, stale = 0;
    for (const v of values) {
        const fileSha = currentFileSha(v);
        const aiSha = deriveAiSourceSha(v);
        const aiSummary = v.ai_summary || null;
        const isChanged = (v.changed === true) || (v.sha_changed === true) || (v.dirty === true);
        if (isChanged) changed++;
        const isStale = !!aiSummary && !!aiSha && !!fileSha && (aiSha !== fileSha);
        if (isStale) stale++;
    }
    return { total, changed, stale };
}
function updateCardsBanner(idx) {
    const { total, changed, stale } = computeCounts(idx);
    const filesEl = $('cards-files');
    const changedEl = $('cards-changed');
    const staleEl = $('cards-stale');
    if (filesEl) filesEl.textContent = String(total);
    if (changedEl) changedEl.textContent = String(changed);
    if (staleEl) staleEl.textContent = String(stale);
}
function renderProjectMapFromCards(idx) {
    cardsIndex = idx || {};
    updateCardsBanner(cardsIndex);
}

// ---------- workspace scanning (wizard only) ----------
const wsRootEl = $('ws-root');          // non-existent in chat-first UI (safe: stays null)
const projectsEl = $('projects');       // non-existent in chat-first UI (safe: stays null)
const currentProjectEl = $('current-project');
const groupMonorepoEl = $('group-monorepo'); // non-existent (safe: null)

function _normSlashes(s) { return s.replaceAll('/', '\\'); }
function _startsWithCI(s, prefix) { return s.toLowerCase().startsWith(prefix.toLowerCase()); }

function shortenPath(fullPath, workspaceRoot) {
    if (!fullPath) return '';
    let s = _normSlashes(fullPath);
    const sLower = s.toLowerCase();
    const idx = sLower.indexOf('\\repos\\');
    if (idx !== -1) return s.slice(idx + 1);
    if (workspaceRoot) {
        let root = _normSlashes(workspaceRoot).replace(/\\+$/, '');
        if (_startsWithCI(s, root)) {
            let rel = s.slice(root.length);
            if (rel.startsWith('\\')) rel = rel.slice(1);
            return rel || s;
        }
    }
    const parts = s.split(/[\/\\]+/).filter(Boolean);
    if (parts.length <= 2) return parts.join('\\');
    return parts.slice(parts.length - 2).join('\\');
}

async function selectProjectPath(projectPath, displayPath, kind, projectId) {
    await ensureSession();

    try {
        const url = new URL('/workspaces/select', location.origin);
        // Keep query param for compatibility with older servers
        url.searchParams.set('session_id', sessionId);

        const payload = {
            // canonical
            session_id: sessionId,
            path: projectPath,
            //legacy fallback (safe to keep while you stabilize versions)
            project_path: projectPath
        };

        const data = await fetchJSON(
            url.toString(),
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            },
            TIMEOUTS.select
        );

        const prettyPath = displayPath || projectPath;

        // Update header badge
        if (currentProjectEl) currentProjectEl.textContent = prettyPath;

        // New server response shape:
        // { session_id, selected: { path, project_id, ... }, project_id }
        const selectedPath =
            data?.selected?.path ||
            data?.path || // ultra-legacy safety
            projectPath;

        const selectedProjectId =
            data?.project_id ||
            data?.selected?.project_id ||
            projectId ||
            selectedPath;

        activeProjectPath = selectedPath;
        activeProjectId = selectedProjectId;

        localStorage.setItem('aidev.activeProjectPath', activeProjectPath);
        localStorage.setItem('aidev.activeProjectId', activeProjectId);

        // Reset description state when switching projects
        const appDescr = extractAppDescriptionFromPayload(data);
        const compiledBrief = extractCompiledBriefFromPayload(data);

        setAppDescription(appDescr);
        setCompiledBrief(compiledBrief);

        projectSelected = true;
        setControlsEnabled(true);

        if (!projectBooted) {
            projectBooted = true;
            postSelectBoot().catch(err => log('[error] postSelectBoot ' + (err?.message || err)));
        }

        const input = $('msg');
        if (input) input.focus();

        pushBubble('system', `Project set to ${activeProjectPath}`);
        closeModal('setup-modal');
        log(`[projects] selected ${prettyPath}`);
    } catch (e) {
        log('[error] select ' + (e?.message || e));
    }
}

function renderProjects(list, workspaceRoot) {
    // Left panel removed — only populate wizard select
    if (!projectsEl) return;
    projectsEl.innerHTML = '';
    list.forEach(p => {
        const li = E('li');
        const left = E('div');

        const display = shortenPath(p.path, workspaceRoot);
        const line1 = E('div', {}, display || p.path || '');
        const line2 = E('div', {},
            E('span', { className: 'pill' }, p.kind || 'unknown'),
            ' ',
            E('small', {}, (p.markers && p.markers.join(', ')) || '')
        );
        left.appendChild(line1);
        left.appendChild(line2);

        const btn = E('button', { className: 'btn', type: 'button' }, 'Select');
        btn.onclick = async () => {
            await withBusyButton(btn, () => selectProjectPath(p.path, display, p.kind, p.project_id));
        };

        li.appendChild(left);
        li.appendChild(btn);
        projectsEl.appendChild(li);
    });
}

async function scan() {
    try {
        await ensureSession();
        const root = wsRootEl?.value || '';
        const group = groupMonorepoEl ? groupMonorepoEl.checked : true;
        localStorage.setItem('aidev.wsroot', root);
        localStorage.setItem('aidev.groupmonorepo', group ? '1' : '0');
        const params = new URLSearchParams();
        if (root) params.set('workspace_root', root);
        params.set('group', group ? '1' : '0');
        params.set('depth', '6');
        const url = `/workspaces/projects?${params.toString()}`;
        const data = await fetchJSON(url, { method: 'GET' }, TIMEOUTS.scan);
        const candidates = data.candidates || [];
        populateWizardSelect(candidates, data.workspace_root || '');
        log(`[projects] ${JSON.stringify(candidates.map(c => ({
            path: shortenPath(c.path, data.workspace_root || ''), kind: c.kind, markers: c.markers, children: c.children_count
        })))}]`);
    } catch (e) {
        log('[error] scan ' + (e?.message || e));
    }
}

// ---------- post-select boot ----------
// Previously this ran /workspaces/refresh-cards + project map on the
// first project select / page load, which could lock the UI if that
// endpoint is slow or broken for a project.
//
// Now we *only* refresh cards & rebuild the project map when:
//   - the user explicitly clicks "Refresh Cards" or "Generate Project Map"
//   - an approved apply run finishes (needsPostApplySync -> refreshCardsAndMapQuietly)
//
// postSelectBoot is kept as a no-op so existing callers don't need to change.
async function postSelectBoot() {
    return;
}

// ---------- AI cost confirm ----------
async function confirmAICostOnce(key = 'aidev.aiConsent') {
    if (sessionStorage.getItem(key)) return true;
    const ok = confirm('This will call the LLM and may incur cost. Proceed?');
    if (ok) sessionStorage.setItem(key, '1');
    return ok;
}

// ---------- Toast (lightweight) ----------
function ensureToastHost() {
    let host = document.getElementById('toast-host');
    if (host) return host;
    host = E('div', { id: 'toast-host', className: 'toast-host' });
    // minimal inline style to avoid CSS dependency
    host.style.position = 'fixed';
    host.style.right = '16px';
    host.style.bottom = '16px';
    host.style.zIndex = '9999';
    host.style.display = 'flex';
    host.style.flexDirection = 'column';
    host.style.gap = '8px';
    document.body.appendChild(host);
    return host;
}
function showToast(text, actionLabel, onAction, kind = 'info', ttlMs = 7000) {
    const host = ensureToastHost();
    const card = E('div', { className: 'toast-card' },
        E('div', { className: 'toast-line' }, text),
        actionLabel ? E('div', { className: 'toast-actions' },
            E('button', { className: 'btn btn-xs', type: 'button' }, actionLabel)
        ) : null
    );
    // minimal style
    card.style.background = '#222';
    card.style.color = '#fff';
    card.style.padding = '10px 12px';
    card.style.borderRadius = '10px';
    card.style.boxShadow = '0 6px 20px rgba(0,0,0,0.25)';
    card.style.maxWidth = '420px';
    card.style.fontSize = '14px';
    card.style.display = 'flex';
    card.style.alignItems = 'center';
    card.style.gap = '10px';
    const btn = card.querySelector('button');
    let closed = false;
    function close() {
        if (closed) return;
        closed = true;
        try { host.removeChild(card); } catch { }
    }
    if (btn && typeof onAction === 'function') {
        btn.onclick = () => { try { onAction(); } finally { close(); } };
    }
    host.appendChild(card);
    const id = setTimeout(close, ttlMs);
    card.addEventListener('click', () => { clearTimeout(id); close(); });
}

// ---------- Summaries Details Modal ----------
function ensureSummariesModal() {
    let modal = $('summaries-modal');
    if (modal) return modal;

    // Use the same modal structure/styles as other modals for consistency
    modal = E('div', {
        id: 'summaries-modal',
        className: 'modal',
        'aria-hidden': 'true',
        role: 'dialog',
        'aria-modal': 'true',
        'aria-labelledby': 'summaries-title'
    },
        E('div', { className: 'modal-backdrop', 'data-close-modal': '1' }),
        E('div', { className: 'modal-card', role: 'document' },
            E('h3', { id: 'summaries-title', text: 'AI Summaries' }),
            E('div', { id: 'summaries-meta', className: 'muted', style: 'margin-bottom:6px' }),
            E('div', {
                id: 'summaries-list',
                className: 'panel',
                style: 'max-height:50vh; overflow:auto; white-space:pre-wrap'
            }),
            E('div', { className: 'row', style: 'margin-top:10px; gap:8px; justify-content:flex-end' },
                E('button', { id: 'summaries-close', className: 'btn ghost', type: 'button' }, 'Close')
            )
        )
    );

    document.body.appendChild(modal);

    const closeBtn = modal.querySelector('#summaries-close');
    if (closeBtn) closeBtn.onclick = () => closeModal('summaries-modal');
    modal.addEventListener('click', (e) => {
        if (e.target && e.target.hasAttribute('data-close-modal')) closeModal('summaries-modal');
    });
    return modal;
}

function firstLines(text, maxLines = 6, maxChars = 600) {
    const t = (text || '').toString();
    const lines = t.split(/\r?\n/).slice(0, maxLines);
    let out = lines.join('\n');
    if (out.length > maxChars) out = out.slice(0, maxChars) + '…';
    return out.trim();
}

function extractSummaryFromNode(node) {
    if (!node || typeof node !== 'object') return ''; 

    // Primary: canonical per-file summary baked onto the card.
    // This is the "one true" summary that the backend owns.
    if (typeof node.summary_text === 'string' && node.summary_text.trim()) {
        return node.summary_text;
    }

    // Backwards-compatible: older card shapes that kept AI data
    // under `ai_summary`.
    const v = node.ai_summary;
    if (typeof v === 'string') return v;
    if (v && typeof v === 'object') {
        return (
            v.summary_text ||   // new-style nested summary
            v.text ||
            v.summary ||
            v.content ||
            v.body ||
            v.preview ||
            ''
        );
    }
    return '';
}

function normalizeRelPathKey(p) {
    if (!p) return '';
    return p.replace(/\\/g, '/').replace(/^\.\/+/, '').replace(/^\/+/, '');
}

function lookupCardNodeByPath(rel) {
    const key = normalizeRelPathKey(rel);
    if (cardsIndex[key]) return cardsIndex[key];
    // try suffix match if exact not found
    const keys = Object.keys(cardsIndex || {});
    const found = keys.find(k => normalizeRelPathKey(k).endsWith('/' + key) || normalizeRelPathKey(k) === key);
    return found ? cardsIndex[found] : null;
}

function renderSummariesDetailsModal(title, counts, files) {
    ensureSummariesModal();
    const meta = $('summaries-meta');
    const ttl = $('summaries-title');
    const list = $('summaries-list');
    if (ttl) ttl.textContent = title || 'AI Summaries';
    if (meta) {
        meta.textContent =
            `Updated ${counts.summarized || 0} • ` +
            `Skipped ${counts.skipped || 0} • ` +
            `Failed ${counts.failed || 0} • ` +
            `Source: per-file cards index (card.summary_text via /summaries/* + /workspaces/refresh-cards)`;
    }
    if (list) {
        list.innerHTML = '';
        if (!Array.isArray(files) || !files.length) {
            list.appendChild(E('div', { className: 'muted' }, '(no files)'));
        } else {
            const frag = document.createDocumentFragment();
            files.forEach(f => {
                const row = E('div', { className: 'sum-row', style: 'padding:8px 6px; border-bottom:1px solid #e5e7eb' });
                const ok = !!f.ok;
                const header = E('div', { className: 'row', style: 'align-items:center; gap:8px; margin-bottom:4px' },
                    E('span', { text: ok ? '✓' : '✗', className: 'pill' }),
                    E('code', {}, normalizeRelPathKey(f.path || '')),
                    E('span', { className: 'muted', style: 'margin-left:8px' }, ` • ${ok ? `len ${f.summary_len || 0}` : (f.error || 'failed')}`)
                );
                frag.appendChild(header);

                if (ok) {
                    const node = lookupCardNodeByPath(f.path || '');
                    const summaryText = extractSummaryFromNode(node);
                    const preview = firstLines(summaryText, 6, 600) || '(no summary text found in cards index yet)';
                    frag.appendChild(E('div', { className: 'muted', style: 'white-space:pre-wrap; margin-left:24px' }, preview));
                } else {
                    frag.appendChild(E('div', { className: 'muted', style: 'margin-left:24px' }, f.error || '(no error)'));
                }
            });
            list.appendChild(frag);
        }
    }
    openModal('summaries-modal');
}

// ---------- cards & summaries ----------
const btnRefreshCards = $('btn-refresh-cards');
const btnAiSummarizeChanged = $('btn-ai-summarize-changed');
const btnAiDeep = $('btn-ai-deep');

const statusRefresh = $('refresh-cards-status');
const statusAi = $('ai-summarize-status');
const statusDeep = $('ai-deep-status');
const modelSel = $('ai-model');

if (btnRefreshCards) btnRefreshCards.onclick = async () => {
    if (!projectSelected) return;
    statusRefresh.textContent = `Refreshing… (timeout ${Math.round(TIMEOUTS.refreshCards / 1000)}s)`;
    btnRefreshCards.disabled = true;
    try {
        await ensureSession();
        const data = await fetchJSON('/workspaces/refresh-cards', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, force: false })
        }, TIMEOUTS.refreshCards);
        statusRefresh.textContent = 'Done';
        renderProjectMapFromCards(data.cards_index || {});
        log('[cards] refreshed');
        // Keep .aidev/project_map.json in sync with refreshed cards
        await genProjectMap();
    } catch (e) {
        statusRefresh.textContent = 'Failed';
        log('[error] refresh-cards ' + (e?.message || e));
    } finally {
        btnRefreshCards.disabled = !projectSelected;
    }
};

async function runSummariesPass(kind, {
    statusEl,
    buttonEl,
    statusLabel,
    toastLabel,
    modalTitle
}) {
    if (!projectSelected) return;
    if (!(await confirmAICostOnce())) return;

    if (statusEl) {
        statusEl.textContent = `${statusLabel}… (timeout ${Math.round(TIMEOUTS.summarizeChanged / 1000)}s)`;
    }
    if (buttonEl) buttonEl.disabled = true;

    try {
        await ensureSession();
        const payload = {
            session_id: sessionId,
            ttl_days: 365,
            enrich_top_k: 24
        };
        const model = modelSel && modelSel.value ? String(modelSel.value) : '';
        if (model) payload.model = model;

        const data = await fetchJSON(`/summaries/${kind}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        }, TIMEOUTS.summarizeChanged);

        const counts = data?.counts || { summarized: 0, skipped: 0, failed: 0 };
        const files = Array.isArray(data?.files) ? data.files : [];

        if (statusEl) {
            statusEl.textContent =
                `${statusLabel} done (${counts.summarized} updated, ${counts.skipped} skipped, ${counts.failed} failed)`;
        }

        // Refresh cards so previews reflect the latest summaries
        try {
            const rc = await fetchJSON('/workspaces/refresh-cards', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, force: false })
            }, TIMEOUTS.refreshCards);
            renderProjectMapFromCards(rc.cards_index || {});
            await genProjectMap();
        } catch (e) {
            log(`[warn] refresh after summaries/${kind} failed: ` + (e?.message || e));
        }

        showToast(
            `${toastLabel}: ${counts.summarized} updated • ${counts.failed} failed — View details`,
            'View details',
            () => {
                renderSummariesDetailsModal(modalTitle, counts, files);
            }
        );

        if (data?.ok === false && data?.error) {
            log(`[ai] summaries/${kind} reported error: ` + data.error);
        }
    } catch (e) {
        if (statusEl) statusEl.textContent = 'Failed';
        log(`[error] summaries/${kind} ` + (e?.message || e));
        showToast(`${toastLabel} failed — View details in log`);
    } finally {
        if (buttonEl) buttonEl.disabled = !projectSelected;
    }
}

if (btnAiSummarizeChanged) {
    btnAiSummarizeChanged.onclick = () =>
        runSummariesPass('changed', {
            statusEl: statusAi,
            buttonEl: btnAiSummarizeChanged,
            statusLabel: 'Summarizing changed files',
            toastLabel: 'Updated summaries',
            modalTitle: 'AI Summaries — Changed files'
        });
}

if (btnAiDeep) {
    btnAiDeep.onclick = () =>
        runSummariesPass('deep', {
            statusEl: statusDeep,
            buttonEl: btnAiDeep,
            statusLabel: 'Deep pass',
            toastLabel: 'Deep summaries',
            modalTitle: 'AI Summaries — Deep pass'
        });
}

// ---------- chat/apply ----------
async function sendChat() {
    if (!projectSelected) {
        openModal('setup-modal');
        return;
    }
    const input = document.getElementById('msg');
    if (!input) return;

    const msg = input.value.trim();
    if (!msg) return;
    input.value = '';

    // Show the user's message immediately in the chat feed
    pushBubble('user', msg);

    await ensureSession();

    const autoApprovePref = isAutoApproveEnabled();
    const runMode = getRunMode();

    // In QA / Analyze modes we don't want edits at all; backend can enforce this,
    // but we also avoid "auto_approve=true" there to keep semantics clean.
    const allowEdits = (runMode === 'edit' || runMode === 'auto');
    const autoApprove = allowEdits ? autoApprovePref : false;

    const modeLabel = (() => {
        switch (runMode) {
            case 'qa':
                return 'Q&A only (no edits)';
            case 'analyze':
                return 'Analyze & plan (no edits)';
            case 'edit':
                return autoApprove
                    ? 'Full edit pass (auto-apply ON)'
                    : 'Full edit pass (manual approval)';
            case 'auto':
            default:
                return autoApprove
                    ? 'Auto (decide per message, auto-apply ON)'
                    : 'Auto (decide per message, manual approval)';
        }
    })();

    const runLabel = `Running — ${modeLabel}`;
    setRunBusy(true, runLabel);

    try {
        const body = {
            message: msg,
            session_id: sessionId,
            mode: runMode,        // tell the backend which mode to use
            allow_edits: allowEdits,
            auto_approve: autoApprove
        };

        pushBubble('system', `Starting run — ${modeLabel}`);

        const data = await fetchJSON('/jobs/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        }, /* timeoutMs */ 0); // or null

        currentJobId = (data && data.job_id) ? String(data.job_id) : null;
        if (!currentJobId) throw new Error('Missing job_id from /jobs/start');

        connectJobStream(currentJobId);

        const approveBtnEl = document.getElementById('approve-changes-btn');
        const rejectBtnEl = document.getElementById('reject-changes-btn');
        if (approveBtnEl && rejectBtnEl) {
            if (!allowEdits) {
                // QA / Analyze modes: there should be no edits to approve.
                approveBtnEl.disabled = true;
                rejectBtnEl.disabled = true;
                approveBtnEl.title = 'This mode does not apply edits.';
                rejectBtnEl.title = 'This mode does not apply edits.';
                setPlannedStatus('This run is configured not to modify files (Q&A / Analyze mode).');
            } else if (autoApprove) {
                approveBtnEl.disabled = true;
                rejectBtnEl.disabled = true;
                const title = 'Automatically apply changes as they\'re ready (advanced).';
                approveBtnEl.title = title;
                rejectBtnEl.title = title;
                setPlannedStatus('Automatically apply changes as they\'re ready (advanced).');
            } else {
                approveBtnEl.disabled = !proposed.length;
                rejectBtnEl.disabled = !proposed.length;
                approveBtnEl.title = '';
                rejectBtnEl.title = '';
                if (proposed.length) {
                    setPlannedStatus('This run will make ' + proposed.length + ' sets of changes. Review below.');
                } else {
                    setPlannedStatus('');
                }
            }
        }

        log('[chat] ' + msg + ` (mode=${runMode}, allow_edits=${allowEdits}, auto_approve=${autoApprove ? 'true' : 'false'})`);
    } catch (e) {
        setRunBusy(false, 'Idle');
        log('[error] chat ' + (e?.message || e));
        showToast('Run failed to start — see log for details');
    }
}

async function approve(yes) {
    await ensureSession();
    const approveBtn = $('approve-changes-btn');
    const rejectBtn = $('reject-changes-btn');
    const btn = yes ? approveBtn : rejectBtn;

    if (approveBtn) approveBtn.disabled = true;
    if (rejectBtn) rejectBtn.disabled = true;

    await withBusyButton(btn, async () => {
        if (!currentJobId) throw new Error('No active job to approve/reject');

        const endpoint = yes ? '/jobs/approve' : '/jobs/reject';
        const data = await fetchJSON(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job_id: currentJobId })
        }, TIMEOUTS.apply);

        log(`[approval] ${yes ? 'approved' : 'rejected'} (job ${currentJobId}) -> ${JSON.stringify(data)}`);

        if (yes) {
            needsPostApplySync = true;
            setPlannedStatus('Approval sent. Applying changes…');
        } else {
            setPlannedStatus('Run rejected. No changes will be applied.');
        }
    }).catch((e) => {
        log('[approval] failed: ' + (e?.message || e));
    });
}

// ---------- Modal helpers ----------
function getFocusableElements(root) {
    if (!root || typeof root.querySelectorAll !== 'function') return [];
    const selector = 'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';
    try {
        return Array.from(root.querySelectorAll(selector)).filter(el => el.offsetParent !== null);
    } catch (e) { return []; }
}

function openModal(id) {
    const el = $(id);
    if (!el) return;
    try {
        // Save last focused element so we can restore on close
        try { __lastFocusedBeforeModal = document.activeElement; } catch (e) { __lastFocusedBeforeModal = null; }

        // Mark open for CSS and ARIA
        el.classList.add('open');
        el.setAttribute('aria-hidden', 'false');

        // Ensure the modal itself is focusable so we can focus it if no focusable children
        if (!el.hasAttribute('tabindex')) el.setAttribute('tabindex', '-1');

        // Focus first focusable element inside modal, otherwise focus the modal
        const focusables = getFocusableElements(el);
        if (focusables.length) {
            try { focusables[0].focus(); } catch (e) { try { el.focus(); } catch { } }
        } else {
            try { el.focus(); } catch { }
        }

        // Attach an Escape key handler specific to this modal so keyboard users can dismiss
        const keydownHandler = (ev) => {
            if (ev.key === 'Escape' || ev.key === 'Esc') {
                try { closeModal(id); } catch (e) { /* ignore */ }
            }
            // Optional: trap tab navigation inside modal (minimal trap)
            if (ev.key === 'Tab') {
                const nodes = getFocusableElements(el);
                if (!nodes.length) return;
                const first = nodes[0];
                const last = nodes[nodes.length - 1];
                if (ev.shiftKey && document.activeElement === first) {
                    ev.preventDefault(); last.focus();
                } else if (!ev.shiftKey && document.activeElement === last) {
                    ev.preventDefault(); first.focus();
                }
            }
        };
        // Store handler so closeModal can remove it
        el.__modalKeydownHandler = keydownHandler;
        el.addEventListener('keydown', keydownHandler);

    } catch (e) {
        // Swallow to avoid console noise
    }
}
function closeModal(id) {
    const el = $(id);
    if (!el) return;
    try {
        el.classList.remove('open');
        el.setAttribute('aria-hidden', 'true');

        // Remove the keydown handler if we added one
        try {
            if (el.__modalKeydownHandler) {
                el.removeEventListener('keydown', el.__modalKeydownHandler);
                delete el.__modalKeydownHandler;
            }
        } catch (e) { /* ignore */ }

        // Restore focus to where it was before the modal opened where possible
        try {
            if (__lastFocusedBeforeModal && typeof __lastFocusedBeforeModal.focus === 'function') {
                __lastFocusedBeforeModal.focus();
            } else {
                // fallback: try to focus the primary CTA or the message input
                const primary = normalizePrimaryCTA();
                if (primary && typeof primary.focus === 'function') primary.focus();
                else { const msg = $('msg'); if (msg) try { msg.focus(); } catch (e) { } }
            }
        } catch (e) { /* ignore */ }
        __lastFocusedBeforeModal = null;
    } catch (e) {
        // Keep silent
    }
}

// Ensure keyboard activation for elements with role="button" (for non-button semantics)
// This helps Enter/Space activate elements that are implemented as div/span with role=button.
(function bindRoleButtonActivation() {
    try {
        document.addEventListener('keydown', (e) => {
            // Don't intercept when typing in inputs or textareas
            const tg = document.activeElement;
            if (!tg) return;
            const tag = (tg.tagName || '').toUpperCase();
            if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag)) return;
            if (tg.isContentEditable) return;

            const role = tg.getAttribute && tg.getAttribute('role');
            if (!role || role !== 'button') return;

            const key = e.key || e.code || '';
            if (key === 'Enter' || key === ' ' || key === 'Spacebar' || key === 'Space') {
                try { e.preventDefault(); tg.click(); } catch (ex) { }
            }
        });
    } catch (e) { /* ignore */ }
})();

// ---------- Setup Wizard ----------
const setupModal = $('setup-modal');
const step1 = $('setup-step-1');
const stepCreate = $('setup-step-create');
const stepSelect = $('setup-step-select');
const btnNext = $('setup-next');
const btnBack1 = $('setup-back-1');
const btnBack2 = $('setup-back-2');
const btnCreate = $('setup-create');
const btnSelect = $('setup-select-btn');
const btnRescan = $('setup-rescan');
const selProjects = $('setup-project-select');
const createStatus = $('create-status');
const selectStatus = $('setup-select-status');

function getSetupMode() {
    const n = setupModal?.querySelector('input[name="setup-mode"]:checked');
    return n ? n.value : 'create';
}
function gotoStep(which) {
    step1.hidden = which !== 1;
    stepCreate.hidden = which !== 'create';
    stepSelect.hidden = which !== 'select';
}
function populateWizardSelect(list, workspaceRoot) {
    if (!selProjects) return;
    selProjects.innerHTML = '';
    (list || []).forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.path;
        opt.textContent = `${shortenPath(p.path, workspaceRoot || '')} — ${p.kind || 'unknown'}`;
        opt.dataset.projectId = p.project_id || '';
        selProjects.appendChild(opt);
    });
}
if (setupModal) {
    setupModal.addEventListener('click', (e) => {
        if (e.target && e.target.hasAttribute('data-close-modal')) closeModal('setup-modal');
    });
}
if (btnNext) btnNext.onclick = () => {
    const mode = getSetupMode();
    if (mode === 'create') gotoStep('create');
    else { gotoStep('select'); scan().catch(() => { }); }
};
if (btnBack1) btnBack1.onclick = () => gotoStep(1);
if (btnBack2) btnBack2.onclick = () => gotoStep(1);

async function doCreateProject() {
    await ensureSession();
    const name = $('create-name')?.value?.trim() || '';
    const baseDir = $('create-basedir')?.value?.trim() || '';
    const brief = $('create-brief')?.value?.trim() || '';
    if (!brief) { createStatus.textContent = 'Please enter a brief.'; return; }

    createStatus.textContent = `Creating… (timeout ${Math.round(TIMEOUTS.create / 1000)}s)`;
    try {
        const body = { brief };
        if (name) body.project_name = name;
        if (baseDir) body.base_dir = baseDir;

        const url = new URL('/projects/create', location.origin);
        url.searchParams.set('session_id', sessionId);
        const data = await fetchJSON(url.toString(), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        }, TIMEOUTS.create);

        createStatus.textContent = 'Created!';
        const projectPath = data?.project?.path;
        const projectId = data?.project?.project_id || data?.project?.project_id || data?.project_id || projectPath;
        if (projectPath) {
            await selectProjectPath(projectPath, projectPath, 'new', projectId);
        }
    } catch (e) {
        createStatus.textContent = 'Failed';
        log('[error] create ' + (e?.message || e));
    }
}
if (btnCreate) btnCreate.onclick = () => withBusyButton(btnCreate, doCreateProject);

async function doSelectExisting() {
    if (!selProjects || !selProjects.value) { selectStatus.textContent = 'Please pick a project.'; return; }
    selectStatus.textContent = 'Selecting…';
    const opt = selProjects.options[selProjects.selectedIndex];
    const pid = opt?.dataset?.projectId || null;
    const display = opt?.textContent || selProjects.value;
    await selectProjectPath(selProjects.value, display, 'existing', pid);
}
if (btnSelect) btnSelect.onclick = () => withBusyButton(btnSelect, doSelectExisting);
if (btnRescan) btnRescan.onclick = () => scan().catch(() => { });

// ---------- Update Descriptions ----------
const btnUpdateDesc = document.getElementById('btn-update-descriptions');
const editDescriptionBtn = document.getElementById('edit-description');
const updateModal = document.getElementById('update-modal');
const updateSubmit = document.getElementById('update-submit');
const updateCancel = document.getElementById('update-cancel');
const updateStatus = document.getElementById('update-status');

function openUpdateDescriptionModal() {
    if (!projectSelected) {
        openModal('setup-modal');
        return;
    }
    const textarea = document.getElementById('update-text');
    if (textarea) {
        textarea.value = currentAppDescription || '';
    }
    if (updateStatus) updateStatus.textContent = '';
    openModal('update-modal');
}

if (btnUpdateDesc) btnUpdateDesc.onclick = openUpdateDescriptionModal;
if (editDescriptionBtn) editDescriptionBtn.onclick = openUpdateDescriptionModal;

if (updateCancel) updateCancel.onclick = () => closeModal('update-modal');
if (updateModal) {
    updateModal.addEventListener('click', (e) => {
        if (e.target && e.target.hasAttribute('data-close-modal')) {
            closeModal('update-modal');
        }
    });
}

async function submitUpdateDescriptions() {
    if (!projectSelected) return;
    await ensureSession();
    const text = document.getElementById('update-text')?.value?.trim() || '';
    if (!text) {
        if (updateStatus) {
            updateStatus.textContent = 'Please enter a description — this is what the AI will read.';
        }
        return;
    }

    if (updateStatus) {
        updateStatus.textContent = 'Saving…';
    }

    try {
        const data = await fetchJSON('/projects/update-descriptions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                app_description: text
            })
        }, TIMEOUTS.updateDescriptions);

        setAppDescription(text);
        const compiled = extractCompiledBriefFromPayload(data);
        setCompiledBrief(compiled);

        // Optional: if backend returns diffs, hydrate per-file proposals too.
        proposed = [];
        for (const [path, rec] of Object.entries(data?.proposed || {})) {
            proposed.push({
                rec_id: 'update-descriptions',
                path,
                diff: String(rec?.diff || ''),
                preview_bytes: Number(rec?.preview_bytes || 0),
                why: 'Update project description and compiled brief',
                summary: String(rec?.summary || 'Update project description and compiled brief')
            });
        }
        renderProposed();

        if (updateStatus) {
            updateStatus.textContent = 'Description saved. Any file edits are shown under Planned Changes.';
        }
        closeModal('update-modal');
    } catch (e) {
        if (updateStatus) updateStatus.textContent = 'Failed to save description.';
        log('[error] update-descriptions ' + (e?.message || e));
    }
}

if (updateSubmit) {
    updateSubmit.onclick = () => withBusyButton(updateSubmit, submitUpdateDescriptions);
}

// ---------- Run checks (pre-apply over proposed patch bundle) ----------
const btnRunChecks = $('btn-run-checks');
const checksStatus = $('checks-status');

// Modal bits
const checksModal = $('checks-modal');
const checksBody = $('checks-body');
const checksApprove = $('checks-approve');
const checksReject = $('checks-reject');

// Keep the last check payload so we can show or re-open the modal
let lastChecksPayload = null;

function openChecksModal(payload) {
    if (!checksModal || !checksBody) { log('[checks] missing modal DOM'); return; }
    const { ok, details = [] } = payload || {};
    checksBody.innerHTML = '';
    const header = E('div', { className: 'row' },
        E('strong', { text: ok ? 'All checks passed ✅' : 'Some checks failed ❌' }),
        E('span', { className: 'muted', style: 'margin-left:8px' }, '(pre-apply validations)')
    );
    checksBody.appendChild(header);

    details.forEach((d) => {
        const wrap = E('details', { className: 'panel', open: !d.ok });
        const ico = d.ok ? '🟢' : '🔴';
        const title = `${ico} ${d.runtime || 'runtime'}`;
        wrap.appendChild(E('summary', { text: title }));
        if (Array.isArray(d.steps) && d.steps.length) {
            d.steps.forEach((s) => {
                const si = s.ok ? '✅' : (s.skipped ? '⏭️' : '❌');
                const head = E('div', { className: 'row' },
                    E('span', { text: `${si} ${s.name}${s.tool ? ` (${s.tool})` : ''}` }),
                    E('span', { className: 'muted', style: 'margin-left:6px' }, `rc=${s.rc}`)
                );
                const pre = E('pre', { className: 'diff' }, (s.stdout || '') + (s.stderr ? `\n${s.stderr}` : ''));
                const inner = E('div', {}, head, pre);
                wrap.appendChild(inner);
            });
        }
        if (d.logs) {
            const pre = E('pre', { className: 'diff' }, d.logs);
            wrap.appendChild(pre);
        }
        checksBody.appendChild(wrap);
    });

    openModal('checks-modal');
    // Default CTA: bias to Reject if any failed
    const allOk = !!ok;
    if (checksApprove) checksApprove.disabled = !allOk;
    if (checksReject) checksReject.disabled = false;
}

function renderQualityChecksSummary(payload) {
    lastQualityChecks = payload || null;

    // We try to render into a dedicated status chip if it exists.
    // If the element is missing, we just log + toast and stay graceful.
    const el = document.getElementById('quality-checks-status');
    const iconEl = document.getElementById('quality-checks-icon');

    if (!payload) {
        if (el) el.textContent = '';
        if (iconEl) iconEl.textContent = '';
        return;
    }

    const ok = payload.ok !== false;
    const results = Array.isArray(payload.results) ? payload.results : [];
    const hasConsistency = results.some(
        (r) =>
            r.tool === 'consistency_checks' ||
            (r.step === 'consistency' || r.step === 'consistency_checks')
    );

    const labelParts = [];
    if (ok) labelParts.push('Checks passed');
    else labelParts.push('Some checks failed');

    if (hasConsistency) {
        labelParts.push('· consistency checks ran');
    }

    const label = labelParts.join(' ');
    if (el) el.textContent = label;

    if (iconEl) {
        iconEl.textContent = ok ? '✅' : '⚠️';
        iconEl.title = label;
    }

    // Stash a compact summary for toast / debugging.
    lastQualityChecksSummary = {
        ok,
        runtimes: results.map((r) => r.tool || r.runtime || 'runtime'),
        hasConsistency
    };

    log('[checks_result] ' + JSON.stringify(lastQualityChecksSummary));

    // Optional UX nicety: a small toast when consistency checks ran
    if (hasConsistency) {
        showToast(
            ok
                ? 'Post-apply checks (format / lint / tests / consistency) all passed.'
                : 'Post-apply checks found issues — see run log for details.'
        );
    }
}

// Core helper that both the button and approval flow can use
async function doPreapplyChecks({ openModalOnResult = true } = {}) {
    if (!projectSelected) {
        openModal('setup-modal');
        throw new Error('No project selected');
    }
    await ensureSession();

    if (checksStatus) checksStatus.textContent = `Running checks… (pre-apply)`;

    try {
        // Build patches from current proposed diffs
        const patches = (proposed || []).map(p => ({
            path: String(p.path || ''),
            diff: String(p.diff || '')
        }));

        const data = await fetchJSON('/checks/preapply', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                job_id: currentJobId || undefined,
                patches
            })
        }, TIMEOUTS.runChecks);

        const passed = !!(data.passed ?? data.ok);
        lastChecksPayload = {
            ok: passed,
            details: data.details || []
        };

        if (checksStatus) checksStatus.textContent = passed ? 'Checks passed' : 'Checks failed';
        if (openModalOnResult) {
            openChecksModal(lastChecksPayload);
        }

        log('[checks] ' + JSON.stringify({
            ok: data.ok,
            runtimes: (data.details || []).map(d => d.runtime)
        }));

        return lastChecksPayload;
    } catch (e) {
        lastChecksPayload = null;
        if (checksStatus) checksStatus.textContent = 'Failed';
        log('[error] pre-apply checks ' + (e?.message || e));
        throw e;
    }
}

async function runChecks() {
    try {
        return await doPreapplyChecks({ openModalOnResult: true });
    } catch {
        return null;
    }
}

if (btnRunChecks) btnRunChecks.onclick = () => withBusyButton(btnRunChecks, runChecks);

// Wire modal approve/reject to new endpoints (unchanged)
if (checksApprove) checksApprove.onclick = async () => {
    try {
        if (!currentJobId) throw new Error('No active job to approve');
        const data = await fetchJSON('/jobs/approve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job_id: currentJobId })
        }, TIMEOUTS.apply);

        if (!data || !data.ok) {
            throw new Error('Server did not accept approval');
        }

        // Mark that this run should sync after it finishes
        needsPostApplySync = true;
        closeModal('checks-modal');
        log('[approval] approved via pre-apply checks modal (job ' + currentJobId + ')');
    } catch (e) {
        log('[approval] approve failed: ' + (e?.message || e));
    }
};

if (checksReject) checksReject.onclick = async () => {
    try {
        if (!currentJobId) throw new Error('No active job to reject');
        const data = await fetchJSON('/jobs/reject', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job_id: currentJobId })
        }, TIMEOUTS.apply);

        if (!data || !data.ok) {
            throw new Error('Server did not accept rejection');
        }

        closeModal('checks-modal');
        log('[approval] rejected via pre-apply checks modal (job ' + currentJobId + ')');
    } catch (e) {
        log('[approval] reject failed: ' + (e?.message || e));
    }
};

if (checksModal) {
    checksModal.addEventListener('click', (e) => {
        if (e.target && e.target.hasAttribute('data-close-modal')) closeModal('checks-modal');
    });
}

// ---------- Plan Approval modal ----------
const planModal = document.getElementById('plan-modal');
const planStatus = document.getElementById('plan-status');
const planSearchEl = document.getElementById('plan-search');
const planApproveBtn = document.getElementById('plan-approve');
const planCancelBtn = document.getElementById('plan-cancel');
const planSelectAll = document.getElementById('plan-select-all');
const planSelectNone = document.getElementById('plan-select-none');
const recListEl = document.getElementById('rec-list');

let planRecs = [];
let selectedRecIds = new Set();
let planModalShown = false;
let planDecision = 'idle';
let planCancelledForRun = false;

function riskPill(risk = 'low') {
    const r = String(risk || 'low').toLowerCase();
    const txt = r.charAt(0).toUpperCase() + r.slice(1);
    const cls = r === 'high' ? 'risk-high' : (r === 'medium' ? 'risk-medium' : 'risk-low');
    return `<span class="pill ${cls}">${txt}</span>`;
}

function normalizeRecs(recs) {
    const out = [];
    (recs || []).forEach((r, i) => {
        const id = String(r.id || r.rec_id || `rec-${i}-${Math.random().toString(36).slice(2, 8)}`);
        let acceptance = [];
        if (Array.isArray(r.acceptance_criteria)) {
            acceptance = r.acceptance_criteria;
        } else if (Array.isArray(r.actions)) {
            acceptance = r.actions;
        }
        const actions = Array.isArray(r.actions) ? r.actions : [];
        out.push({
            id,
            title: String(r.title || r.name || 'Recommendation'),
            summary: String(r.summary || r.short || r.reason || ''),
            risk: (r.risk || 'low'),
            rationale: String(r.rationale || r.why || ''),
            acceptance_criteria: acceptance.map((c) => String(c || '').trim()).filter(Boolean),
            actions: actions.map((a) => String(a || '').trim()).filter(Boolean),
            files: Array.isArray(r.files) ? r.files.map(f => ({
                path: String(f.path || f.file || ''),
                added: Number(f.added || 0),
                removed: Number(f.removed || 0),
                why: String(f.why || '')
            })) : []
        });
    });
    return out;
}

function planMatchesFilter(rec, q) {
    if (!q) return true;
    const needle = q.toLowerCase();
    if (rec.title.toLowerCase().includes(needle)) return true;
    if (rec.summary.toLowerCase().includes(needle)) return true;
    if (rec.rationale.toLowerCase().includes(needle)) return true;
    for (const crit of (rec.acceptance_criteria || [])) {
        if (String(crit).toLowerCase().includes(needle)) return true;
    }
    for (const act of (rec.actions || [])) {
        if (String(act).toLowerCase().includes(needle)) return true;
    }
    for (const f of (rec.files || [])) {
        if ((f.path || '').toLowerCase().includes(needle)) return true;
    }
    return false;
}

function renderRecList() {
    if (!recListEl) return;
    recListEl.innerHTML = '';
    const q = (planSearchEl?.value || '').trim();

    const frag = document.createDocumentFragment();
    planRecs.forEach((rec) => {
        if (!planMatchesFilter(rec, q)) return;
        const id = `rec-cb-${btoa(unescape(encodeURIComponent(rec.id))).replace(/=/g, '')}`;
        const item = document.createElement('div');
        item.className = 'rec-item';

        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.id = id;
        cb.value = rec.id;
        cb.checked = selectedRecIds.has(rec.id);
        cb.addEventListener('change', () => {
            if (cb.checked) selectedRecIds.add(rec.id);
            else selectedRecIds.delete(rec.id);
        });

        const label = document.createElement('label');
        label.htmlFor = id;
        label.style.flex = '1';

        const filesBadge = rec.files && rec.files.length
            ? `<span class="pill">${rec.files.length} file${rec.files.length > 1 ? 's' : ''}</span>`
            : '';
        const head = `
            <div class="rec-title">${rec.title}</div>
            <div class="rec-meta">${riskPill(rec.risk)} ${filesBadge}</div>
            ${rec.summary ? `<div class="rec-summary">${rec.summary}</div>` : ''}
        `;

        const fileRows = (rec.files || []).map(f => {
            const ar = `+${f.added ?? 0} / -${f.removed ?? 0}`;
            const why = f.why ? `<div class="muted">${f.why}</div>` : '';
            return `<div class="rec-files"><code>${f.path}</code> — ${ar}${why ? (' ' + why) : ''}</div>`;
        }).join('');

        const criteria = (rec.acceptance_criteria && rec.acceptance_criteria.length)
            ? rec.acceptance_criteria
            : (rec.actions || []);

        const acHtml = (criteria && criteria.length)
            ? `
                <div class="rec-criteria-label">Acceptance criteria</div>
                <ul class="rec-criteria">
                    ${criteria.map(c => `<li>${c}</li>`).join('')}
                </ul>
            `
            : '';

        const details = `
            <details style="margin-top:6px">
                <summary>Details</summary>
                ${acHtml || ''}
                ${rec.rationale ? `<div class="muted" style="margin:6px 0">${rec.rationale}</div>` : ''}
                ${fileRows || '<div class="muted">(no specific files listed)</div>'}
            </details>
        `;
        label.innerHTML = head + details;

        item.appendChild(cb);
        item.appendChild(label);
        frag.appendChild(item);
    });

    recListEl.appendChild(frag);
}

function openPlanModal(recs = []) {
    if (planModalShown) return;
    try {
        planModalShown = true;
        planDecision = 'pending';
        if (planStatus) planStatus.textContent = '';
        planRecs = normalizeRecs(recs);
        selectedRecIds = new Set(planRecs.map(r => r.id));
        renderRecList();
        openModal('plan-modal');
    } catch (e) {
        planModalShown = false;
        log('[plan-modal] open failed: ' + (e?.message || e));
    }
}

function closePlanModal() {
    planModalShown = false;
    closeModal('plan-modal');
}

async function cancelPlanAndHaltRun(reason = 'plan_modal_cancel') {
    planDecision = 'dismissed';
    planCancelledForRun = true;
    closePlanModal();
    log(`[plan] modal dismissed (reason=${reason}); job left running`);
}

async function submitPlanApproval() {
    if (!projectSelected) return;
    await ensureSession();

    const pick = Array.from(selectedRecIds);
    if (!pick.length) {
        if (planStatus) planStatus.textContent = 'Please select at least one recommendation.';
        return;
    }
    if (planStatus) {
        planStatus.textContent = `Submitting…`;
    }

    try {
        if (!currentJobId) throw new Error('No active job to approve');
        const data = await fetchJSON('/jobs/approve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job_id: currentJobId, selected: pick })
        }, TIMEOUTS.approvePlan);

        if (planStatus) planStatus.textContent = (data && data.ok) ? 'Plan approved!' : 'Submitted.';
        planDecision = 'approved';
        closePlanModal();
        // If an inline panel is present, remove it after approval
        const inline = document.getElementById('recommendations-panel');
        if (inline && inline.parentNode) inline.parentNode.removeChild(inline);
        log('[plan] approved for job ' + currentJobId);
    } catch (e) {
        if (planStatus) planStatus.textContent = 'Failed to submit approval.';
        log('[plan] approval failed: ' + (e?.message || e));
    }
}

if (planSearchEl) planSearchEl.addEventListener('input', renderRecList);
if (planSelectAll) planSelectAll.onclick = () => { planRecs.forEach(r => selectedRecIds.add(r.id)); renderRecList(); };
if (planSelectNone) planSelectNone.onclick = () => { selectedRecIds.clear(); renderRecList(); };

if (planApproveBtn) planApproveBtn.onclick = () => withBusyButton(planApproveBtn, submitPlanApproval);
if (planCancelBtn) planCancelBtn.onclick = () => withBusyButton(planCancelBtn, () => cancelPlanAndHaltRun('plan_modal_cancel'));

if (planModal) {
    planModal.addEventListener('click', (e) => {
        if (e.target && e.target.hasAttribute('data-close-modal')) {
            e.preventDefault();
            cancelPlanAndHaltRun('plan_modal_dismiss');
        }
    });
}

// ---------- INLINE Recommendations panel (replacement for modal) ----------
function ensureRecommendationsPanel() {
    let panel = document.getElementById('recommendations-panel');
    if (panel) return panel;

    panel = document.createElement('section');
    panel.id = 'recommendations-panel';
    panel.className = 'panel recommendations-panel';
    panel.setAttribute('role', 'region');
    panel.setAttribute('aria-label', 'Recommendations');
    panel.setAttribute('aria-live', 'polite');

    const header = document.createElement('div');
    header.className = 'row';
    header.style.justifyContent = 'space-between';
    header.style.alignItems = 'center';

    const title = document.createElement('h4');
    title.textContent = 'Recommendations';
    title.id = 'recommendations-title';
    header.appendChild(title);

    const controls = document.createElement('div');
    controls.className = 'row';
    controls.style.gap = '8px';

    const search = document.createElement('input');
    search.type = 'search';
    search.id = 'recommendations-search';
    search.placeholder = 'Filter recommendations…';
    search.setAttribute('aria-label', 'Filter recommendations');
    search.style.minWidth = '180px';
    controls.appendChild(search);

    const selAll = document.createElement('button');
    selAll.type = 'button';
    selAll.id = 'recommendations-select-all';
    selAll.className = 'btn btn-xs ghost';
    selAll.textContent = 'Select all';
    controls.appendChild(selAll);

    const clear = document.createElement('button');
    clear.type = 'button';
    clear.id = 'recommendations-clear';
    clear.className = 'btn btn-xs ghost';
    clear.textContent = 'Clear';
    controls.appendChild(clear);

    header.appendChild(controls);
    panel.appendChild(header);

    const list = document.createElement('div');
    list.id = 'recommendations-list';
    list.setAttribute('role', 'list');
    list.style.marginTop = '8px';
    panel.appendChild(list);

    const footer = document.createElement('div');
    footer.className = 'row';
    footer.style.justifyContent = 'space-between';
    footer.style.alignItems = 'center';
    footer.style.marginTop = '10px';

    const summary = document.createElement('div');
    summary.id = 'recommendations-selected-summary';
    summary.className = 'muted';
    summary.textContent = '';
    footer.appendChild(summary);

    const actions = document.createElement('div');
    actions.style.display = 'flex';
    actions.style.gap = '8px';

    const approve = document.createElement('button');
    approve.type = 'button';
    approve.id = 'recommendations-approve';
    approve.className = 'btn primary';
    approve.textContent = 'Approve & Apply';
    approve.setAttribute('aria-label', 'Approve selected recommendations and apply changes');
    actions.appendChild(approve);

    const cancel = document.createElement('button');
    cancel.type = 'button';
    cancel.id = 'recommendations-cancel';
    cancel.className = 'btn ghost';
    cancel.textContent = 'Dismiss';
    cancel.setAttribute('aria-label', 'Dismiss recommendations');
    actions.appendChild(cancel);

    footer.appendChild(actions);
    panel.appendChild(footer);

    // Insert the panel near the diffs/review area if present
    const mount = document.getElementById('diffs') || document.getElementById('approval-summary') || document.getElementById('planned-changes') || document.body;
    try {
        if (mount.parentElement) mount.parentElement.insertBefore(panel, mount.nextSibling);
        else document.body.appendChild(panel);
    } catch (e) { document.body.appendChild(panel); }

    // Wire interactions
    search.addEventListener('input', () => renderRecommendationsPanel(planRecs));
    selAll.addEventListener('click', () => { planRecs.forEach(r => selectedRecIds.add(r.id)); renderRecommendationsPanel(planRecs); });
    clear.addEventListener('click', () => { selectedRecIds.clear(); renderRecommendationsPanel(planRecs); });

    approve.addEventListener('click', () => withBusyButton(approve, submitPlanApproval));
    cancel.addEventListener('click', () => cancelPlanAndHaltRun('inline_cancel'));

    return panel;
}

function renderRecommendationsPanel(recs) {
    // recs: array of recommendation objects (backend shape)
    const panel = ensureRecommendationsPanel();
    if (!panel) return;
    planRecs = normalizeRecs(recs || []);
    // default to selecting all
    if (!selectedRecIds || selectedRecIds.size === 0) selectedRecIds = new Set(planRecs.map(r => r.id));

    const list = document.getElementById('recommendations-list');
    if (!list) return;
    list.innerHTML = '';
    const q = (document.getElementById('recommendations-search')?.value || '').trim().toLowerCase();

    planRecs.forEach((rec) => {
        if (q && !planMatchesFilter(rec, q)) return;
        const item = document.createElement('div');
        item.className = 'recommendation-item';
        item.setAttribute('role', 'listitem');
        item.style.display = 'flex';
        item.style.alignItems = 'flex-start';
        item.style.gap = '8px';
        item.style.padding = '8px 0';
        item.style.borderBottom = '1px solid var(--border)';

        const cb = document.createElement('input');
        cb.type = 'checkbox';
        const id = `rec-inline-${btoa(unescape(encodeURIComponent(rec.id))).replace(/=/g, '')}`;
        cb.id = id;
        cb.value = rec.id;
        cb.checked = selectedRecIds.has(rec.id);
        cb.setAttribute('aria-label', `Select recommendation: ${rec.title}`);
        cb.addEventListener('change', () => {
            if (cb.checked) selectedRecIds.add(rec.id);
            else selectedRecIds.delete(rec.id);
            updateRecommendationsSummary();
        });

        const content = document.createElement('div');
        content.style.flex = '1';

        const head = document.createElement('div');
        head.style.display = 'flex';
        head.style.justifyContent = 'space-between';
        head.style.alignItems = 'center';

        const title = document.createElement('div');
        title.className = 'rec-title';
        title.textContent = rec.title || 'Recommendation';
        title.style.fontWeight = '600';

        const meta = document.createElement('div');
        meta.className = 'rec-meta';
        meta.innerHTML = riskPill(rec.risk) + (rec.files && rec.files.length ? ` <span class="pill">${rec.files.length} file${rec.files.length>1?'s':''}</span>` : '');

        head.appendChild(title);
        head.appendChild(meta);

        const summary = document.createElement('div');
        summary.className = 'rec-summary';
        summary.textContent = rec.summary || rec.rationale || '';
        summary.style.marginTop = '6px';
        summary.style.color = 'var(--muted)';

        content.appendChild(head);
        content.appendChild(summary);

        item.appendChild(cb);
        item.appendChild(content);
        list.appendChild(item);
    });

    updateRecommendationsSummary();
}

function updateRecommendationsSummary() {
    const el = document.getElementById('recommendations-selected-summary');
    const total = planRecs.length || 0;
    const selected = selectedRecIds ? selectedRecIds.size : 0;
    if (el) el.textContent = `${selected} of ${total} selected`;
    const approveBtn = document.getElementById('recommendations-approve');
    if (approveBtn) approveBtn.disabled = selected === 0;
}

// ---------- cards & summaries continued ----------

// ---------- Plan wiring re-use ----------
// The inline recommendations reuse the same approval submission function
// (submitPlanApproval) and selection state (planRecs, selectedRecIds) used by
// the plan modal. This keeps network payloads identical: POST /jobs/approve
// with { job_id, selected: [ids...] }.

// ---------- wire controls ----------
const sendBtn = $('send'); if (sendBtn) sendBtn.onclick = sendChat;
const approveBtn = $('approve-changes-btn'); if (approveBtn) approveBtn.onclick = () => approve(true);
const rejectBtn = $('reject-changes-btn'); if (rejectBtn) rejectBtn.onclick = () => approve(false);
const genMapBtn = $('gen-map'); if (genMapBtn) genMapBtn.onclick = genProjectMap;

const copyBtn = $('copy-map');
if (copyBtn) copyBtn.onclick = async () => {
    try {
        const pre = $('project-map-json');
        const txt = pre ? pre.textContent || '' : '';
        await navigator.clipboard.writeText(txt);
        log('[copy] project map JSON copied');
    } catch (e) { log('[copy] failed: ' + (e?.message || e)); }
};

// Header buttons
const projectBtn = $('btn-project');
if (projectBtn) projectBtn.onclick = () => {
    openModal('setup-modal');
    // Populate existing projects when opening
    scan().catch(() => { });
};
const mapToggleBtn = $('btn-toggle-map');
const mapSectionEl = $('map-section');
if (mapToggleBtn && mapSectionEl) {
    mapToggleBtn.onclick = () => {
        const willShow = mapSectionEl.hasAttribute('hidden');
        if (willShow && !projectSelected) {
            openModal('setup-modal');
            return;
        }
        const pre = $('project-map-json');
        if (willShow && !pre) {
            // element isn’t on the page; nothing to do
        }
        mapSectionEl.toggleAttribute('hidden');
    };
}

// persist auto-approve preference
(function initAutoApprove() {
    const key = 'aidev.autoApprove';
    const cb = $('auto-approve-changes');
    if (!cb) return;

    // Restore last value
    const saved = localStorage.getItem(key);
    if (saved === '1') cb.checked = true;
    if (saved === '0') cb.checked = false;

    cb.addEventListener('change', () => {
        localStorage.setItem(key, cb.checked ? '1' : '0');
        log(`[auto-approve] ${cb.checked ? 'enabled' : 'disabled'} (persisted)`);
    });
})();

// persist run mode preference
(function initRunModePreference() {
    const sel = $('run-mode');
    if (!sel) return;
    const key = 'aidev.runMode';
    const saved = localStorage.getItem(key);
    if (saved && Array.from(sel.options || []).some(o => o.value === saved)) {
        sel.value = saved;
    }
    sel.addEventListener('change', () => {
        localStorage.setItem(key, sel.value || 'auto');
        log(`[run-mode] set to ${sel.value || 'auto'} (persisted)`);
    });
})();

// persist model
(function persistInit() {
    const modelSelEl = $('ai-model');
    const savedModel = localStorage.getItem('aidev.model');
    if (savedModel && modelSelEl) modelSelEl.value = savedModel;
    if (modelSelEl) modelSelEl.onchange = () => localStorage.setItem('aidev.model', modelSelEl.value || '');
})();

// keyboard shortcuts
document.addEventListener('keydown', (e) => {
    const targetIsMsg = document.activeElement && document.activeElement.id === 'msg';
    if (e.key === 'Enter' && targetIsMsg && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
        e.preventDefault(); sendChat();
    } else if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (planModalShown && $('plan-approve') && !$('plan-approve').disabled) {
            $('plan-approve').click();
            return;
        }
        const approveBtnEl = $('approve-changes-btn');
        if (approveBtnEl && !approveBtnEl.disabled) {
            approve(true);
        } else {
            log('[key] Ctrl/Cmd+Enter ignored (approve disabled)');
        }
    }
});

// ---------- Lightweight delegated event handling & public init ----------
function _delegatedClickHandler(e) {
    try {
        const el = e.target && e.target.closest ? e.target.closest('button, [role="button"], a, [data-action]') : null;
        if (!el) return;

        // Respect explicit data-action when present
        const action = el.getAttribute('data-action') || el.id || (el.dataset && el.dataset.action) || null;
        if (!action) {
            // Close modals when clicking background/backdrop elements marked with data-close-modal
            if (el.hasAttribute('data-close-modal')) {
                const modal = el.closest('.modal');
                if (modal && modal.id) closeModal(modal.id);
            }
            return;
        }

        // Map common top-level actions to functions here (delegated)
        switch (action) {
            case 'send':
            case 'send-btn':
            case 'send-button':
            case 'send':
                e.preventDefault(); sendChat(); break;
            case 'approve-changes-btn':
            case 'approve':
                e.preventDefault(); approve(true); break;
            case 'reject-changes-btn':
            case 'reject':
                e.preventDefault(); approve(false); break;
            case 'gen-map':
                e.preventDefault(); genProjectMap(); break;
            case 'btn-refresh-cards':
                e.preventDefault(); if (btnRefreshCards) btnRefreshCards.click(); break;
            case 'btn-ai-summarize-changed':
                e.preventDefault(); if (btnAiSummarizeChanged) btnAiSummarizeChanged.click(); break;
            case 'btn-ai-deep':
                e.preventDefault(); if (btnAiDeep) btnAiDeep.click(); break;
            case 'btn-update-descriptions':
            case 'edit-description':
                e.preventDefault(); openUpdateDescriptionModal(); break;
            case 'primary-start-cta':
            case 'primary-cta':
                e.preventDefault(); openModal('setup-modal'); break;
            case 'btn-project':
                e.preventDefault(); openModal('setup-modal'); break;
            case 'btn-toggle-map':
                e.preventDefault(); if (mapSectionEl) mapSectionEl.toggleAttribute('hidden'); break;
            // Allow fallthrough to default behavior for other actions
            default:
                // If element has data-close-modal, attempt to close its modal ancestor
                if (el.hasAttribute('data-close-modal')) {
                    const modal = el.closest('.modal');
                    if (modal && modal.id) closeModal(modal.id);
                }
                break;
        }
    } catch (err) {
        log('[delegate] click handler error: ' + (err?.message || err));
    }
}

// Theme helpers: persist and apply theme (light/dark)
function applyTheme(theme) {
    try {
        const root = document.documentElement || document.body;
        if (!root) return;
        if (String(theme) === 'dark') {
            root.classList.add('theme-dark');
        } else {
            root.classList.remove('theme-dark');
        }
        try { safeLocalStorageSet('aidev.theme', String(theme)); } catch (e) { /* ignore */ }
        // update toggle UI if present
        try {
            const toggle = $('theme-toggle') || document.querySelector('[data-aidev-target="theme-toggle"]') || document.getElementById('theme-toggle');
            if (toggle) {
                const pressed = String(theme) === 'dark';
                toggle.setAttribute('aria-pressed', pressed ? 'true' : 'false');
                if (toggle.tagName && toggle.tagName.toLowerCase() === 'button') {
                    toggle.textContent = pressed ? 'Dark' : 'Light';
                }
            }
        } catch (e) { /* ignore */ }
    } catch (e) { log('[theme] applyTheme failed: ' + (e?.message || e)); }
}
function toggleTheme() {
    try {
        const current = safeLocalStorageGet('aidev.theme') || (document.documentElement.classList.contains('theme-dark') ? 'dark' : 'light');
        const next = current === 'dark' ? 'light' : 'dark';
        applyTheme(next);
    } catch (e) { log('[theme] toggle failed: ' + (e?.message || e)); }
}

function initThemeToggle() {
    try {
        // Restore saved preference
        const saved = safeLocalStorageGet('aidev.theme');
        if (saved) applyTheme(saved);
        // Wire toggle if present
        const toggle = $('theme-toggle') || document.querySelector('[data-aidev-target="theme-toggle"]') || document.getElementById('theme-toggle');
        if (!toggle) return;
        // Make accessible
        if (!toggle.hasAttribute('role')) toggle.setAttribute('role', 'button');
        if (!toggle.hasAttribute('tabindex')) toggle.setAttribute('tabindex', '0');
        toggle.addEventListener('click', (e) => { e.preventDefault(); toggleTheme(); });
        toggle.addEventListener('keydown', (e) => {
            const key = e.key || e.code || '';
            if (key === 'Enter' || key === ' ' || key === 'Spacebar' || key === 'Space') { e.preventDefault(); toggleTheme(); }
        });
    } catch (e) { log('[theme] init failed: ' + (e?.message || e)); }
}

function init(root = document) {
    try {
        if (init.__inited) return;
        init.__inited = true;
        const mount = (root && root.addEventListener) ? root : document;
        // Attach delegated click handler at document/root level to capture top-level UI actions
        mount.addEventListener('click', _delegatedClickHandler);
        // Ensure the SSE announcer exists (idempotent)
        ensureSSEAnnouncer();
        // Provide a global hook for programmatic init in tests
        try { if (typeof window !== 'undefined') window.aidevInit = init; } catch (e) { /* ignore */ }

        // Parse a declarative stage map in the DOM (if present) and expose top-progress helpers
        try {
            parseStageMapFromDOM();
            if (typeof window !== 'undefined' && typeof window.updateTopProgress !== 'function') window.updateTopProgress = updateTopProgress;
            if (typeof window !== 'undefined' && typeof window.updateTopProgressLabel !== 'function') window.updateTopProgressLabel = updateTopProgressLabel;
        } catch (e) { log('[init] stage-map wiring failed: ' + (e?.message || e)); }

        // Debounced resize: keep header controls tidy on viewport changes
        try {
            if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
                window.addEventListener('resize', debounce(() => {
                    try { enforceHeaderControlVisibility(3); } catch (e) { /* ignore */ }
                }, 150));
            }
        } catch (e) { /* ignore */ }
    } catch (e) {
        log('[init] failed: ' + (e?.message || e));
    }
}

// ---------- initUI (new public boot) ----------
async function initUI(rootDocument = document) {
    if (initUI.__inited) return;
    initUI.__inited = true;
    try {
        // Cache commonly-used DOM nodes for faster lookups and consistent access
        cacheDOM();

        // Parse stage map, ensure announcer and spinner exist, and wire UI helpers
        try { parseStageMapFromDOM(); } catch (e) { log('[initUI] parseStageMap failed: ' + (e?.message || e)); }
        try { ensureSSEAnnouncer(); } catch (e) { /* ignore */ }
        try { ensureSpinnerElement(); } catch (e) { /* ignore */ }
        try { ensureTechnicalDetailsPanel(); } catch (e) { /* ignore */ }

        // Init onboarding, advanced panels and toggle (idempotent guards inside)
        try { initOnboardingModal(); } catch (e) { log('[initUI] onboarding init failed: ' + (e?.message || e)); }
        try { initCollapsedAdvancedPanels(); } catch (e) { log('[initUI] collapsed advanced panels failed: ' + (e?.message || e)); }
        try { initAdvancedToggle(); } catch (e) { log('[initUI] advanced toggle failed: ' + (e?.message || e)); }

        // Try to load event schema asynchronously (non-blocking)
        try { loadEventSchema().catch(() => {}); } catch (e) { /* ignore */ }

        // Debounced resize handler for header controls
        try {
            if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
                window.addEventListener('resize', debounce(() => {
                    try { enforceHeaderControlVisibility(3); } catch (e) { /* ignore */ }
                }, 200));
            }
        } catch (e) { /* ignore */ }

        // Autosave minimal UI draft state for the message input
        try {
            const msgEl = DOM.msg || document.getElementById('msg');
            if (msgEl) {
                // Restore any prior draft safely
                try {
                    const raw = safeLocalStorageGet('aidev.ui.state');
                    if (raw) {
                        const st = JSON.parse(raw || '{}');
                        if (st.lastMsgDraft && (!msgEl.value || msgEl.value === '')) msgEl.value = st.lastMsgDraft;
                    }
                } catch (e) { /* ignore parse errors */ }

                msgEl.addEventListener('input', debounce(() => {
                    try {
                        const raw = safeLocalStorageGet('aidev.ui.state');
                        const s = raw ? JSON.parse(raw || '{}') : {};
                        s.lastMsgDraft = msgEl.value || '';
                        s.advancedMode = !!readAdvancedMode();
                        s.activeProjectId = activeProjectId || null;
                        safeLocalStorageSet('aidev.ui.state', JSON.stringify(s));
                    } catch (e) { log('[initUI] autosave failed: ' + (e?.message || e)); }
                }, 500));
            }
        } catch (e) { /* ignore */ }

        // mark global UI ready so lightweight defensive stubs in index.html can detect real loader
        try { if (typeof window !== 'undefined') window.__AIDEV_APP_LOADED = true; } catch (e) { /* ignore */ }

        // Initialize theme toggle wiring and restore theme preference
        try { initThemeToggle(); } catch (e) { log('[initUI] theme init failed: ' + (e?.message || e)); }

        // Initialize stage timeline UI if available
        try { initStageTimeline(); } catch (e) { log('[initUI] stage timeline init failed: ' + (e?.message || e)); }

    } catch (e) {
        log('[initUI] failed: ' + (e?.message || e));
    }
}

// Expose initUI for external callers (tests, embed wrappers)
try { if (typeof window !== 'undefined') window.initUI = initUI; } catch (e) { /* ignore */ }
try { if (typeof module !== 'undefined' && module.exports) module.exports.initUI = initUI; } catch (e) { /* ignore */ }

// Ensure initUI runs on DOMContentLoaded (idempotent)
try {
    if (typeof document !== 'undefined') {
        if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => initUI(document));
        else setTimeout(() => initUI(document), 0);
    }
} catch (e) { /* ignore */ }

// ---------- boot ----------
async function boot() {
    // Ensure spinner DOM exists early so ARIA is present for screen readers
    try { ensureSpinnerElement(); } catch { }

    // Ensure onboarding banner is shown for first-time users
    try { ensureOnboardingBanner(); } catch { }

    // Normalize primary CTA id and hide extra header controls so the header is calm on first load.
    try {
        normalizePrimaryCTA();
        enforceHeaderControlVisibility(3);
    } catch { }

    // Best-effort load of events.schema.json; failures just fall back to defaults.
    try {
        await loadEventSchema();
    } catch (e) {
        log('[boot] events schema load failed: ' + (e?.message || e));
    }

    try {
        await ensureSession();
    } catch (e) {
        log('[boot] ensureSession failed: ' + (e?.message || e));
    }
    try { await scan(); } catch { }
    setControlsEnabled(projectSelected);

    // Initialize onboarding modal (separate from the lightweight banner). This
    // will open the modal on first load if the 'aidev_onboard_shown' flag is not set.
    try { initOnboardingModal(); } catch (e) { log('[onboard] boot init failed: ' + (e?.message || e)); }

    // Collapse advanced panels by default and add visible toggles so users can
    // reveal them when ready.
    try { initCollapsedAdvancedPanels(); } catch (e) { log('[advanced-panels] boot init failed: ' + (e?.message || e)); }

    // Initialize the single global Advanced toggle (keeps UI minimal by default)
    try { initAdvancedToggle(); } catch (e) { log('[advanced-toggle] boot init failed: ' + (e?.message || e)); }

    if (!projectSelected) {
        gotoStep(1);
        openModal('setup-modal');
    } else {
        if (activeProjectPath && currentProjectEl) {
            currentProjectEl.textContent = activeProjectPath;
            setControlsEnabled(true);

            // Rehydrate server-side selection so endpoints don't fall back to "example"
            try {
                await selectProjectPath(
                    activeProjectPath,
                    activeProjectPath,     // display
                    'rehydrate',           // kind (for logs only)
                    activeProjectId || null
                );
            } catch (e) {
                log('[rehydrate] failed: ' + (e?.message || e));
            }
        }
    }

    // Make sure the summaries modal exists so "View details" works immediately
    ensureSummariesModal();
}

// Run boot after DOMContentLoaded to avoid DOM race conditions and ensure
// handlers/querying occur with a ready DOM.
try {
    if (typeof document !== 'undefined') {
        document.addEventListener('DOMContentLoaded', () => {
            try { init(document); boot().catch(e => log('[boot] ' + (e?.message || e))); } catch (e) { log('[boot] DOMContentLoaded handler failed: ' + (e?.message || e)); }
        });
    }
} catch (e) { /* ignore */ }
