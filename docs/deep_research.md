# Deep Research (Repository-Local)

Deep Research is **repository-local research**: it analyzes the code and files in this repo and **does not browse the web**, fetch URLs, or crawl external sources.

This page describes how to invoke Deep Research, budget profiles, cache layout, determinism guarantees, and how it integrates with orchestration flows.

## Purpose

Deep Research is a structured, repeatable way to collect evidence from the repository (code, docs, configs) and produce a **research bundle** that downstream flows (e.g., analyze/recommend) can use.

Typical outputs include:
- A ranked set of relevant files/sections
- Extracted snippets/quotes with stable references
- Summary notes and supporting metadata sufficient for follow-on reasoning

## Invocation

Public entrypoint: `DevBotAPI.deep_research`

This document defines a single canonical parameter set for `DevBotAPI.deep_research`. Other docs (e.g., README.md) should use the same calling convention to avoid confusion.

Key parameters (high-level):
- `budget`: `'quick' | 'standard' | 'deep'` (controls runtime vs coverage)
- `targets`: what to research (commonly `['repo']` to indicate repository scope)
- `include_patterns`: glob patterns of files to include (e.g., `['src/**', 'docs/**']`)
- `exclude_patterns`: glob patterns to exclude (e.g., `['**/node_modules/**', '**/.venv/**']`)
- `dry_run`: if true, plans work and reports what would be done without writing results
- `log_level`: controls verbosity of logs for debugging and reproducibility checks

High-level return shape (conceptual):
- `summary`: brief findings
- `artifacts`: references to cached/serialized artifacts (paths/ids)
- `citations`: structured references to files/line ranges (or equivalent stable anchors)
- `stats`: counts/timings useful for comparing runs across budgets

## Example

```python
DevBotAPI.deep_research(
    budget='standard',
    targets=['repo'],
    include_patterns=['src/**', 'docs/**'],
    exclude_patterns=['**/node_modules/**'],
    dry_run=False,
)
```

Notes:
- Deep Research is repository-local: it does not fetch the web.
- Prefer explicit include/exclude patterns to make runs smaller and more deterministic.

## Budget Profiles

Budgets are intended to be **predictable** presets. Choose based on speed vs coverage.

### `quick`
- Meaning: fast, shallow scan for high-signal entrypoints and obvious references.
- Tradeoff: fastest runtime, lowest coverage; may miss deep call chains or peripheral docs.
- Use when: iterating locally, validating wiring, or doing a first pass.

### `standard`
- Meaning: balanced scan; enough depth to support typical analyze/recommend workflows.
- Tradeoff: moderate runtime with good coverage; still bounded to avoid exhaustive indexing.
- Use when: default choice for CI-like research or routine engineering tasks.

### `deep`
- Meaning: maximum coverage within repository-local constraints; broader indexing and cross-linking.
- Tradeoff: slowest runtime, highest coverage; produces larger caches and more artifacts.
- Use when: hard-to-trace behaviors, large refactors, or when quick/standard are missing evidence.

Guidance:
- Increase budget when you see gaps like “no references found” for known symbols, or missing cross-module linkages.
- Decrease budget when you only need a file shortlist or to confirm the location of an interface.

## Cache & Stored Artifacts

Cache root: `.aidev/cache/research/`

At a high level, this cache may store:
- File discovery indices (file lists, timestamps, sizes, normalized paths)
- Content-derived indices (e.g., tokenized text, symbol/location maps)
- Intermediate representations used for retrieval (e.g., embeddings or similar vectors)
- Result bundles (the final research output, plus supporting metadata)
- Metadata for reproducibility (budget used, include/exclude patterns, tool versions, etc.)

Cache behavior expectations:
- Cache keys are **deterministic** (see Determinism Guarantees).
- Cache invalidation is typically driven by input changes (file contents, include/exclude patterns, budget) and/or implementation-defined TTL/eviction.
- If you suspect stale results, clear `.aidev/cache/research/` and re-run with the same inputs to confirm reproducibility.

## Determinism Guarantees

Deep Research is intended to be reproducible for the same repository state and inputs. Implementations should uphold:

1. **Stable sort/order**
   - Always sort files, matches, and outputs using a stable ordering (e.g., normalized path then position).
   - Rationale: prevents nondeterministic differences caused by filesystem traversal order or hash iteration order.

2. **Stable ids**
   - Assign stable identifiers to artifacts/citations derived from deterministic inputs (e.g., path + location + content hash).
   - Rationale: allows caching and downstream references to remain consistent across runs.

3. **Deterministic truncation**
   - When limits apply (max files, max tokens, top-K results), truncate deterministically using explicit ranking + stable tie-breakers.
   - Rationale: avoids “randomly missing” items when multiple candidates score similarly.

4. **Deterministic cache keys**
   - Cache keys must be derived from normalized inputs (budget, include/exclude patterns, target set, and relevant configuration) and deterministic content fingerprints.
   - Rationale: ensures cache hits/misses are explainable and repeatable.

## Integration & Engine

Deep Research is implemented in:
- `aidev/orchestration/deep_research_engine.py`

Cache support is commonly implemented in (if present):
- `aidev/orchestration/deep_research_cache.py`

Integration points (high level):
- Orchestration flows typically call the engine to produce a research bundle.
- The research bundle is then consumed by **analyze** (to interpret evidence) and **recommend** (to propose changes grounded in repo facts).

Validation / smoke testing:
- If present, see `scripts/smoke_deep_research.py` for a simple end-to-end run that can be used to verify configuration, determinism, and cache writes. If that script/module is not available in your workspace, run the engine programmatically or consult orchestration tests/examples for a similar verification flow.

## Troubleshooting / FAQ

### I see different outputs across runs with the same inputs
Check for common sources of nondeterminism:
- Unstable ordering (filesystem traversal order, unordered dict/set iteration)
- Non-deterministic truncation (ties without stable tie-breakers)
- Cache key differences due to unnormalized patterns or configuration

Recommended checks:
- Re-run with the exact same `budget`, `targets`, `include_patterns`, and `exclude_patterns`.
- Inspect `.aidev/cache/research/` to confirm which artifacts were reused vs regenerated.
- If needed, clear `.aidev/cache/research/` and rerun to confirm the engine is deterministic given a clean cache.

### The cache looks “wrong” or out of date
- Confirm the repo state (branch/commit) matches your expectations.
- Clear `.aidev/cache/research/` to force regeneration.
- If the issue persists, compare the deterministic cache key inputs (patterns, budget, target set) between runs.
