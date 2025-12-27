# aidev/config.py
from __future__ import annotations
"""
Configuration loader for AI-Dev-Bot.

Environment variables (OpenAI & Azure OpenAI):

# OpenAI (default)
- OPENAI_API_KEY=...            # required when using OpenAI endpoints
- OPENAI_BASE_URL=...           # optional, for custom/proxy base URL
- AIDEV_MODEL=gpt-5-mini        # default model unless overridden
- AIDEV_TEMP=0.0
- AIDEV_MAX_TOKENS=80000
- AIDEV_TIMEOUT_SEC=1800

# Azure OpenAI (auto-detected if present)
- AZURE_OPENAI_API_KEY=...      # key
- AZURE_OPENAI_ENDPOINT=...     # e.g. https://<resource>.openai.azure.com/
- AZURE_OPENAI_API_VERSION=...  # e.g. 2024-10-01-preview
- AZURE_DEPLOYMENT=...          # your deployment name (also used as 'model')

# Deep Research
- AIDEV_DEEP_RESEARCH_DEFAULT_PROFILE=medium
- AIDEV_DEEP_RESEARCH_MAX_EVIDENCE=...
- AIDEV_DEEP_RESEARCH_MAX_FINDINGS=...
- AIDEV_DEEP_RESEARCH_MAX_CHARS=...

Notes:
- The client should read the API key from the env var specified in cfg["llm"]["api_key_env"].
- If AZURE_* vars are present, we override base_url/api_key_env and optionally the model (deployment).
- Deep Research profile/caps are non-secret and are safe to be in config.json and/or env vars.
"""

import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

# Optional: load .env without clobbering existing env
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

# Optional: soft JSON Schema validation
try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None  # type: ignore


def load_env_variables() -> None:
    """Load a local .env if present (non-destructive)."""
    if load_dotenv is not None:
        load_dotenv(override=False)


def _get_env_or(names: list[str]) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v and str(v).strip():
            return v.strip()
    return None


# ---------- Defaults (safe, OpenAI-first; Azure auto-overrides if env is present) ----------

DEFAULT_CONFIG: Dict[str, Any] = {
    "llm": {
        # OpenAI-style defaults (works with the new OpenAI SDK). Azure overrides below if env is set.
        "model": os.getenv("AIDEV_MODEL", "gpt-5-mini"),
        "temperature": float(os.getenv("AIDEV_TEMP", "0.0")),
        "max_output_tokens": int(os.getenv("AIDEV_MAX_TOKENS", "80000")),
        "timeout_sec": float(os.getenv("AIDEV_TIMEOUT_SEC", "1800")),
        "base_url": os.getenv("OPENAI_BASE_URL", None),
        # Which env var to read the key from (do NOT write secrets to config.json)
        "api_key_env": os.getenv("AIDEV_API_KEY_ENV", "OPENAI_API_KEY"),
        # Extra/advanced fields (kept generic so the client can pass them through)
        "extra": {},
        # Per-stage model mapping (optional). Keys are stage names, values are model strings.
        # Env override: AIDEV_MODEL_<STAGE> (e.g. AIDEV_MODEL_PROJECT_CREATE) will override entries.
        "model_map": {"PROJECT_CREATE": "gpt-5-mini"},
    },
    "discovery": {
        "scan_depth": int(os.getenv("AIDEV_SCAN_DEPTH", "2")),
        "includes": [],
        "excludes": ["**/.aidev/**", "**/node_modules/**", "**/.git/**"],
    },
    "cards": {
        "default_budget": int(os.getenv("AIDEV_CARDS_BUDGET", "4096")),
        "default_top_k": int(os.getenv("AIDEV_CARDS_TOPK", "20")),
        # Use a hard-coded default here so config-file overrides (.aidev/config.json) work as intended.
        "summarize_concurrency": 5,
    },
    "recommendations": {
        # How many related KB cards to attach to the recommendations payload by default.
        "related_cards_top_k": int(os.getenv("RECOMMENDATIONS_RELATED_CARDS_TOP_K", "6")),
        # Maximum total characters across concatenated related card summaries (truncation enforced by caller).
        "related_cards_char_cap": int(os.getenv("RECOMMENDATIONS_RELATED_CARDS_CHAR_CAP", "4000")),
    },
    "deep_research": {
        # Budget profile selection (stable default, can be overridden in .aidev/config.json and/or env).
        "default_profile": "medium",
        # Named profiles (non-secret). Callers can add profile-specific knobs later; for now we use caps.
        "profiles": {
            "low": {"max_evidence": 5, "max_findings": 4, "max_chars": 4000},
            "medium": {"max_evidence": 12, "max_findings": 8, "max_chars": 8000},
            "high": {"max_evidence": 30, "max_findings": 20, "max_chars": 15000},
        },
        # Global caps: applied as defaults/fallbacks and merged with profile-specific settings.
        "caps": {"max_evidence": 50, "max_findings": 30, "max_chars": 15000},
    },
    "quality_gates": {
        # Toggle quality gates per runtime
        "python": {"format": True, "lint": True, "test": True},
        "node":   {"format": True, "lint": True, "test": True},
        "flutter":{"format": True, "lint": True, "test": True},
        "php":    {"format": False, "lint": False, "test": False},
        # Optional custom command overrides
        "commands": {
            "python": {
                "format": ["black", "-q", "."],
                "lint":   ["ruff", "check", "."],
                "test":   ["pytest", "-q"],
            },
            "node": {
                "format": ["prettier", "--write", "."],
                "lint":   ["eslint", ".", "--fix"],
                "test":   ["npm", "test", "--silent"],
            },
            "flutter": {
                "format": ["dart", "format", "-o", "write", "."],
                "lint":   ["flutter", "analyze"],
                "test":   ["flutter", "test"],
            },
            "php": {
                # Example if you add later:
                # "lint": ["php", "-l", "."],
            },
        },
    },
    "trace": {
        "max_bytes": int(os.getenv("AIDEV_TRACE_MAX_BYTES", str(10 * 1024 * 1024))),  # 10 MiB
        "keep_generations": int(os.getenv("AIDEV_TRACE_KEEP", "7")),
    },
    "ui": {
        "host": os.getenv("AIDEV_UI_HOST", "127.0.0.1"),
        "port": int(os.getenv("AIDEV_UI_PORT", "8765")),
        "require_approval": True,
    },
    # Whether to refresh cards between applying each recommendation (can be overridden in .aidev/config.json)
    "refresh_cards_between_recs": True,
}


# Module-level convenience constants for easy import by runtime code.
# These constants are intentionally module-level so other modules (e.g., the recommendations
# payload builder) can import them directly instead of reading the full CONFIG at runtime.
# Normalize and validate environment values to sensible defaults so tests and runtimes
# are resilient to malformed environment variables.
def _env_int(name: str, default: int, min_value: int | None = None) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        ival = int(val)
    except Exception:
        return default
    if min_value is not None and ival < min_value:
        return default
    return ival


def _env_bool(name: str, default: bool) -> bool:
    """Read an environment variable as a boolean with sensible string parsing.

    Accepts (case-insensitive): '1','true','yes','on' -> True; '0','false','no','off' -> False.
    If the variable is not set or the value is unrecognized, returns the provided default.
    """
    val = os.getenv(name)
    if val is None:
        return default
    v = str(val).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


RECOMMENDATIONS_RELATED_CARDS_TOP_K: int = _env_int("RECOMMENDATIONS_RELATED_CARDS_TOP_K", 6, min_value=1)
RECOMMENDATIONS_RELATED_CARDS_CHAR_CAP: int = _env_int("RECOMMENDATIONS_RELATED_CARDS_CHAR_CAP", 4000, min_value=0)
# Environment variable wins over config file; default true allows per-recommendation refreshes.
REFRESH_CARDS_BETWEEN_RECS: bool = _env_bool("REFRESH_CARDS_BETWEEN_RECS", DEFAULT_CONFIG.get("refresh_cards_between_recs", True))


CONFIG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "llm": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "temperature": {"type": "number"},
                "max_output_tokens": {"type": "integer"},
                "timeout_sec": {"type": "number"},
                "base_url": {"type": ["string", "null"]},
                "api_key_env": {"type": "string"},
                "extra": {"type": "object"},
                "model_map": {"type": "object", "additionalProperties": {"type": "string"}},
            },
            "required": ["model", "temperature", "max_output_tokens", "timeout_sec", "api_key_env"],
        },
        "discovery": {
            "type": "object",
            "properties": {
                "scan_depth": {"type": "integer"},
                "includes": {"type": "array", "items": {"type": "string"}},
                "excludes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["scan_depth", "includes", "excludes"],
        },
        "cards": {
            "type": "object",
            "properties": {
                "default_budget": {"type": "integer"},
                "default_top_k": {"type": "integer"},
                "summarize_concurrency": {"type": "integer"},
            },
            "required": ["default_budget", "default_top_k"],
        },
        "recommendations": {
            "type": "object",
            "properties": {
                "related_cards_top_k": {"type": "integer"},
                "related_cards_char_cap": {"type": "integer"},
            },
            "required": [],
        },
        "deep_research": {
            "type": "object",
            "properties": {
                "default_profile": {"type": "string"},
                "profiles": {"type": "object"},
                "caps": {
                    "type": "object",
                    "properties": {
                        "max_evidence": {"type": "integer"},
                        "max_findings": {"type": "integer"},
                        "max_chars": {"type": "integer"},
                    },
                    "required": [],
                    "additionalProperties": True,
                },
            },
            "required": [],
            "additionalProperties": True,
        },
        "quality_gates": {"type": "object"},
        "trace": {
            "type": "object",
            "properties": {
                "max_bytes": {"type": "integer"},
                "keep_generations": {"type": "integer"},
            },
            "required": ["max_bytes", "keep_generations"],
        },
        "ui": {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer"},
                "require_approval": {"type": "boolean"},
            },
            "required": ["host", "port", "require_approval"],
        },
        "refresh_cards_between_recs": {"type": "boolean"},
    },
    "additionalProperties": True,
}


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _validate(cfg: Dict[str, Any]) -> None:
    if jsonschema is None:
        return
    try:
        jsonschema.validate(cfg, CONFIG_SCHEMA)  # type: ignore[attr-defined]
    except Exception as e:
        logging.getLogger(__name__).debug("config schema validation failed: %s", e)


def _apply_azure_overrides(cfg: Dict[str, Any]) -> None:
    """
    Auto-detect Azure and override OpenAI defaults if present.
    """
    endpoint = _get_env_or(["AZURE_ENDPOINT", "AZURE_OPENAI_ENDPOINT"])
    key = _get_env_or(["AZURE_OPENAI_API_KEY"])
    api_version = _get_env_or(["API_VERSION", "AZURE_OPENAI_API_VERSION"])
    deployment = _get_env_or(["AZURE_DEPLOYMENT"])

    if endpoint and key:
        llm = cfg.setdefault("llm", {})
        llm["base_url"] = endpoint
        llm["api_key_env"] = "AZURE_OPENAI_API_KEY"
        # Use deployment as model if provided
        if deployment:
            llm["model"] = deployment
        # Surface api_version in extra so the client can pass it through when supported
        extra = dict(llm.get("extra") or {})
        if api_version:
            extra["api_version"] = api_version
        llm["extra"] = extra


def _apply_model_env_overrides(cfg: Dict[str, Any]) -> None:
    """
    Apply model-related environment overrides:
      - Ensure cfg['llm']['model_map'] exists.
      - AIDEV_DEFAULT_MODEL overrides cfg['llm']['model'] when present.
      - Any envs starting with AIDEV_MODEL_ (e.g. AIDEV_MODEL_card_summarize) are merged into cfg['llm']['model_map'] with the suffix as the stage key.
    This function only writes into llm.model_map and llm.model when envs are present.
    """
    llm = cfg.setdefault("llm", {})
    model_map = llm.setdefault("model_map", {}) or {}

    # AIDEV_DEFAULT_MODEL overrides the default llm.model (env precedence)
    default_model = os.getenv("AIDEV_DEFAULT_MODEL")
    if default_model and default_model.strip():
        llm["model"] = default_model.strip()

    # Scan for AIDEV_MODEL_<stage> env vars and merge into model_map.
    prefix = "AIDEV_MODEL_"
    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue
        stage = k[len(prefix):]
        if not stage:
            continue
        if v and v.strip():
            model_map[stage] = v.strip()

    llm["model_map"] = model_map


def _apply_deep_research_env_overrides(cfg: Dict[str, Any]) -> None:
    """Apply Deep Research env overrides into cfg (mutating cfg in-place).

    Env vars (non-secret):
      - AIDEV_DEEP_RESEARCH_DEFAULT_PROFILE
      - AIDEV_DEEP_RESEARCH_MAX_EVIDENCE
      - AIDEV_DEEP_RESEARCH_MAX_FINDINGS
      - AIDEV_DEEP_RESEARCH_MAX_CHARS

    Precedence: env wins over project config/defaults.
    """
    dr = cfg.setdefault("deep_research", {})

    # default_profile override
    env_profile = os.getenv("AIDEV_DEEP_RESEARCH_DEFAULT_PROFILE")
    if env_profile and env_profile.strip():
        dr["default_profile"] = env_profile.strip()

    # caps overrides
    caps = dict(dr.get("caps") or {})

    def _cap_from_env(name: str, min_value: int) -> int | None:
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            return None
        try:
            val = int(str(raw).strip())
        except Exception:
            return None
        if val < min_value:
            return None
        return val

    v = _cap_from_env("AIDEV_DEEP_RESEARCH_MAX_EVIDENCE", min_value=0)
    if v is not None:
        caps["max_evidence"] = v

    v = _cap_from_env("AIDEV_DEEP_RESEARCH_MAX_FINDINGS", min_value=0)
    if v is not None:
        caps["max_findings"] = v

    v = _cap_from_env("AIDEV_DEEP_RESEARCH_MAX_CHARS", min_value=0)
    if v is not None:
        caps["max_chars"] = v

    dr["caps"] = caps
    cfg["deep_research"] = dr


def get_deep_research_profile(
    profile_name: str | None = None,
    cfg: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Return the effective Deep Research profile settings and metadata.

    This helper is deterministic and side-effect free: it does not mutate the provided cfg.

    Resolution order:
      1) profile_name argument (if provided)
      2) env AIDEV_DEEP_RESEARCH_DEFAULT_PROFILE
      3) cfg['deep_research']['default_profile']
      4) DEFAULT_CONFIG['deep_research']['default_profile']

    Caps are merged as: global caps (cfg/default) overlaid by profile-specific values.
    Env caps (AIDEV_DEEP_RESEARCH_MAX_*) always override merged values.

    Returns: (settings, meta) where settings includes:
      - profile_name (string)
      - max_evidence/max_findings/max_chars (ints)
    and meta includes: source ('env'|'config'|'default') and profile_name.
    """
    # Base config blocks (do not mutate caller cfg)
    base_dr: Dict[str, Any] = dict((DEFAULT_CONFIG.get("deep_research") or {}))
    cfg_dr: Dict[str, Any] = {}
    if cfg:
        try:
            cfg_dr = dict((cfg.get("deep_research") or {}))
        except Exception:
            cfg_dr = {}

    # Determine chosen profile name and its source
    source = "default"
    chosen = None

    if profile_name and str(profile_name).strip():
        chosen = str(profile_name).strip()
        # Caller-provided choice is treated as config-level selection for metadata purposes.
        source = "config"
    else:
        env_profile = os.getenv("AIDEV_DEEP_RESEARCH_DEFAULT_PROFILE")
        if env_profile and env_profile.strip():
            chosen = env_profile.strip()
            source = "env"
        else:
            cfg_default = cfg_dr.get("default_profile")
            if cfg_default and str(cfg_default).strip():
                chosen = str(cfg_default).strip()
                source = "config"
            else:
                chosen = str(base_dr.get("default_profile") or "medium")
                source = "default"

    # Global caps from cfg then default
    base_caps = base_dr.get("caps") or {}
    cfg_caps = cfg_dr.get("caps") or {}
    merged_caps: Dict[str, Any] = {}
    if isinstance(base_caps, dict):
        merged_caps.update(base_caps)
    if isinstance(cfg_caps, dict):
        merged_caps.update(cfg_caps)

    # Profile-specific settings from cfg then default (cfg wins, then default)
    base_profiles = base_dr.get("profiles") or {}
    cfg_profiles = cfg_dr.get("profiles") or {}

    prof_settings: Dict[str, Any] = {}
    if isinstance(base_profiles, dict):
        maybe = base_profiles.get(chosen)
        if isinstance(maybe, dict):
            prof_settings.update(maybe)
    if isinstance(cfg_profiles, dict):
        maybe = cfg_profiles.get(chosen)
        if isinstance(maybe, dict):
            prof_settings.update(maybe)

    # Merge caps + profile
    out: Dict[str, Any] = dict(merged_caps)
    out.update(prof_settings)

    # Canonicalize to ints and fill required keys with safe fallbacks
    def _to_int(v: Any, default: int) -> int:
        try:
            iv = int(v)
            if iv < 0:
                return default
            return iv
        except Exception:
            return default

    # Use the resolved values if present, else fallback to hard defaults from DEFAULT_CONFIG
    hard_caps = (DEFAULT_CONFIG.get("deep_research") or {}).get("caps") or {}
    hard_max_evidence = _to_int((hard_caps or {}).get("max_evidence", 50), 50)
    hard_max_findings = _to_int((hard_caps or {}).get("max_findings", 30), 30)
    hard_max_chars = _to_int((hard_caps or {}).get("max_chars", 15000), 15000)

    out_max_evidence = _to_int(out.get("max_evidence", hard_max_evidence), hard_max_evidence)
    out_max_findings = _to_int(out.get("max_findings", hard_max_findings), hard_max_findings)
    out_max_chars = _to_int(out.get("max_chars", hard_max_chars), hard_max_chars)

    # Env caps override merged values (highest precedence)
    env_evidence = os.getenv("AIDEV_DEEP_RESEARCH_MAX_EVIDENCE")
    if env_evidence is not None and str(env_evidence).strip() != "":
        out_max_evidence = _to_int(env_evidence, out_max_evidence)

    env_findings = os.getenv("AIDEV_DEEP_RESEARCH_MAX_FINDINGS")
    if env_findings is not None and str(env_findings).strip() != "":
        out_max_findings = _to_int(env_findings, out_max_findings)

    env_chars = os.getenv("AIDEV_DEEP_RESEARCH_MAX_CHARS")
    if env_chars is not None and str(env_chars).strip() != "":
        out_max_chars = _to_int(env_chars, out_max_chars)

    settings = {
        "profile_name": str(chosen),
        "max_evidence": int(out_max_evidence),
        "max_findings": int(out_max_findings),
        "max_chars": int(out_max_chars),
    }
    meta = {"source": source, "profile_name": str(chosen)}
    return settings, meta


def deep_research_profile_for_cache(
    cfg: Dict[str, Any] | None = None,
    profile_name: str | None = None,
) -> Tuple[str, Tuple[Tuple[str, int], ...]]:
    """Return a normalized Deep Research profile representation suitable for cache keys.

    Output is deterministic and composed only of canonical primitives:
      (profile_name, (('max_chars', 6000), ('max_evidence', 12), ('max_findings', 8)))

    Note: callers should include BOTH the profile_name and the normalized items in their cache key
    so different budget profiles do not reuse the same cached artifact.
    """
    settings, _meta = get_deep_research_profile(profile_name=profile_name, cfg=cfg)
    pn = str(settings.get("profile_name") or "")
    items = tuple(
        sorted(
            (
                ("max_chars", int(settings.get("max_chars", 0))),
                ("max_evidence", int(settings.get("max_evidence", 0))),
                ("max_findings", int(settings.get("max_findings", 0))),
            ),
            key=lambda kv: kv[0],
        )
    )
    return pn, items


def load_model_map_from_env(cfg: Dict[str, Any] | None = None) -> Dict[str, str]:
    """
    Non-mutating helper: return a model_map derived from the provided cfg (if any)
    merged with any AIDEV_MODEL_<STAGE> environment variables.

    The returned mapping keys are the stage suffixes from env vars or the keys present
    in cfg['llm']['model_map']. This function does not mutate the provided cfg.

    Example env:
      AIDEV_MODEL_RECOMMENDATIONS=gpt-5
      AIDEV_MODEL_ANALYZE=gpt-5-mini

    Precedence (for a given stage key): env AIDEV_MODEL_<STAGE> wins over cfg['llm']['model_map'].
    """
    base_map: Dict[str, str] = {}
    if cfg:
        try:
            llm = cfg.get("llm") or {}
            mm = llm.get("model_map") or {}
            if isinstance(mm, dict):
                # copy to avoid mutating caller-provided cfg structures
                for k, v in mm.items():
                    if v is None:
                        continue
                    base_map[str(k)] = str(v)
        except Exception:
            base_map = {}

    prefix = "AIDEV_MODEL_"
    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue
        stage = k[len(prefix):]
        if not stage:
            continue
        if v and v.strip():
            base_map[stage] = v.strip()
    return base_map


def get_model_for_stage(stage: str, cfg: Dict[str, Any] | None = None) -> Tuple[str, Dict[str, str]]:
    """
    Resolve the effective model for a named stage.

    Resolution order (highest -> lowest):
      1. Explicit env AIDEV_MODEL_<STAGE> (case-insensitive stage matching)
      2. cfg['llm']['model_map'][stage] (case-insensitive key match)
      3. Env AIDEV_DEFAULT_MODEL
      4. cfg['llm']['model'] (from file defaults or .aidev/config.json, possibly overridden by Azure)
      5. DEFAULT_CONFIG['llm']['model'] (hard fallback)

    Returns (model_string, metadata) where metadata contains keys: 'source' ('env'|'model_map'|'default') and 'stage'.

    Note: callers who only need the model string should access the first tuple element (e.g. model, meta = get_model_for_stage(...); use model).
    """
    logger = logging.getLogger(__name__)
    if cfg is None:
        cfg = dict(DEFAULT_CONFIG)

    # Normalize for env lookup: use uppercased stage for environment variable names.
    stage_original = stage
    stage_upper = stage.upper()

    # 1) Check explicit env for this stage (case-insensitive)
    env_key = f"AIDEV_MODEL_{stage_upper}"
    env_val = os.getenv(env_key)
    if env_val and env_val.strip():
        model = env_val.strip()
        source = "env"
        logger.debug("resolved model for stage=%s -> %s (source=%s)", stage_original, model, source)
        return model, {"source": source, "stage": stage_original}

    # 2) Check cfg['llm']['model_map'][stage] with case-insensitive matching
    llm_cfg = cfg.get("llm") or {}
    model_map = llm_cfg.get("model_map") or {}
    if isinstance(model_map, dict):
        # Try exact, upper, lower keys, then a case-insensitive search over keys.
        # exact key as provided
        if stage in model_map and model_map[stage]:
            model = str(model_map[stage])
            source = "model_map"
            logger.debug("resolved model for stage=%s -> %s (source=%s)", stage_original, model, source)
            return model, {"source": source, "stage": stage_original}
        # uppercased key
        if stage_upper in model_map and model_map[stage_upper]:
            model = str(model_map[stage_upper])
            source = "model_map"
            logger.debug("resolved model for stage=%s -> %s (source=%s)", stage_original, model, source)
            return model, {"source": source, "stage": stage_original}
        # lowercased key
        stage_lower = stage.lower()
        if stage_lower in model_map and model_map[stage_lower]:
            model = str(model_map[stage_lower])
            source = "model_map"
            logger.debug("resolved model for stage=%s -> %s (source=%s)", stage_original, model, source)
            return model, {"source": source, "stage": stage_original}
        # final attempt: case-insensitive match against all keys
        for k, v in model_map.items():
            try:
                if k is not None and v and str(k).lower() == stage_lower:
                    model = str(v)
                    source = "model_map"
                    logger.debug("resolved model for stage=%s -> %s (source=%s)", stage_original, model, source)
                    return model, {"source": source, "stage": stage_original}
            except Exception:
                continue

    # 3) Check AIDEV_DEFAULT_MODEL env
    env_default = os.getenv("AIDEV_DEFAULT_MODEL")
    if env_default and env_default.strip():
        model = env_default.strip()
        source = "env"
        logger.debug("resolved model for stage=%s -> %s (source=%s)", stage_original, model, source)
        return model, {"source": source, "stage": stage_original}

    # 4) cfg['llm']['model']
    cfg_model = llm_cfg.get("model")
    if cfg_model and str(cfg_model).strip():
        model = str(cfg_model).strip()
        source = "default"
        logger.debug("resolved model for stage=%s -> %s (source=%s)", stage_original, model, source)
        return model, {"source": source, "stage": stage_original}

    # 5) Fallback to the hard-coded DEFAULT_CONFIG
    model = str(DEFAULT_CONFIG.get("llm", {}).get("model", ""))
    source = "default"
    logger.debug("resolved model for stage=%s -> %s (source=%s)", stage_original, model, source)
    return model, {"source": source, "stage": stage_original}


def get_model_for(stage: str) -> str:
    """
    Public convenience wrapper that returns only the resolved model string for a stage.

    This function normalizes stage matching so callers can pass e.g. 'recommendations'
    and environment variables like AIDEV_MODEL_RECOMMENDATIONS will be respected.

    The returned value is never an empty string: if resolution yields an empty string, the
    DEFAULT_CONFIG['llm']['model'] is returned as a final fallback.
    """
    model, _meta = get_model_for_stage(stage, cfg=globals().get("CONFIG"))
    if not model or not str(model).strip():
        return str(DEFAULT_CONFIG.get("llm", {}).get("model", ""))
    return model

# Convenience alias for callers that prefer a shorter name: get_model('STAGE').
# This flow honors AIDEV_MODEL_<STAGE> env vars (e.g., AIDEV_MODEL_PROJECT_CREATE) via get_model_for_stage.
get_model = get_model_for


def get_project_create_model() -> str:
    """Return the resolved model string for the PROJECT_CREATE stage, honoring AIDEV_MODEL_PROJECT_CREATE.

    This helper is an ergonomic convenience for the project-create flow.
    """
    return get_model("PROJECT_CREATE")


def load_project_config(project_root: Path, explicit_path: str | None = None) -> Tuple[Dict[str, Any], Path]:
    """
    Load `.aidev/config.json` if present, deep-merge onto defaults,
    then apply Azure env overrides (if AZURE_* present).
    Returns (config, path_used).
    """
    root = Path(project_root).resolve()
    path = Path(explicit_path).resolve() if explicit_path else (root / ".aidev" / "config.json")

    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cfg = _deep_merge(DEFAULT_CONFIG, data)
        except Exception:
            cfg = dict(DEFAULT_CONFIG)
    else:
        cfg = dict(DEFAULT_CONFIG)

    _apply_azure_overrides(cfg)
    # Apply model-related environment overrides (AIDEV_DEFAULT_MODEL and AIDEV_MODEL_<stage>)
    _apply_model_env_overrides(cfg)
    # Apply deep research env overrides (AIDEV_DEEP_RESEARCH_*)
    _apply_deep_research_env_overrides(cfg)
    _validate(cfg)
    return cfg, path


def save_default_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")


# Load .env early when this module is imported
load_env_variables()

# Populate module-level CONFIG and MODEL_MAP so other modules can import them without
# having to call load_project_config themselves. This call is lightweight and resilient to missing files.
try:
    CONFIG, CONFIG_PATH = load_project_config(Path("."))
except Exception:
    CONFIG = dict(DEFAULT_CONFIG)
    CONFIG_PATH = Path(".")

# Resolve CARD_SUMMARIZE_CONCURRENCY with precedence: explicit ENV -> loaded CONFIG -> DEFAULT_CONFIG
def _resolve_card_summarize_concurrency() -> int:
    default = int(DEFAULT_CONFIG.get("cards", {}).get("summarize_concurrency", 5) or 5)

    # Env override wins when present and valid
    env_raw = os.getenv("AIDEV_CARD_SUMMARIZE_CONCURRENCY")
    if env_raw is not None and str(env_raw).strip() != "":
        try:
            val = int(env_raw)
            if val >= 1:
                return val
        except Exception:
            # fall through to config/default
            pass

    # Next, respect the loaded CONFIG (e.g., .aidev/config.json) if present and valid
    try:
        cfg_cards = (CONFIG or {}).get("cards") or {}
        cfg_val = int(cfg_cards.get("summarize_concurrency", default))
        if cfg_val >= 1:
            return cfg_val
    except Exception:
        pass

    return int(default)

CARD_SUMMARIZE_CONCURRENCY: int = _resolve_card_summarize_concurrency()

# MODEL_MAP reflects the effective model_map after merging config and environment variables.
MODEL_MAP: Dict[str, str] = dict((CONFIG.get("llm", {}) or {}).get("model_map") or {})

# ---------------------------------------------------------------------------
# Export module-level deep research convenience constants so other modules can
# import these values directly (per the cross-file contract). These are
# resolved after CONFIG is loaded so they reflect env overrides and project
# config. Defaults from DEFAULT_CONFIG ensure stable behavior when env/config
# are unset.
# ---------------------------------------------------------------------------
try:
    _dr_settings, _dr_meta = get_deep_research_profile(cfg=globals().get("CONFIG"))
except Exception:
    # Fall back to defaults if resolution fails for any reason
    _dr_settings, _dr_meta = get_deep_research_profile(cfg=None)

AIDEV_DEEP_RESEARCH_DEFAULT_PROFILE: str = str(
    _dr_settings.get("profile_name")
    or os.getenv("AIDEV_DEEP_RESEARCH_DEFAULT_PROFILE")
    or (DEFAULT_CONFIG.get("deep_research") or {}).get("default_profile", "medium")
)
AIDEV_DEEP_RESEARCH_MAX_EVIDENCE: int = int(_dr_settings.get("max_evidence") or (DEFAULT_CONFIG.get("deep_research") or {}).get("caps", {}).get("max_evidence", 50))
AIDEV_DEEP_RESEARCH_MAX_FINDINGS: int = int(_dr_settings.get("max_findings") or (DEFAULT_CONFIG.get("deep_research") or {}).get("caps", {}).get("max_findings", 30))
AIDEV_DEEP_RESEARCH_MAX_CHARS: int = int(_dr_settings.get("max_chars") or (DEFAULT_CONFIG.get("deep_research") or {}).get("caps", {}).get("max_chars", 15000))
# Legacy alias supported by other modules: treat max_bytes as an alias for max_chars
AIDEV_DEEP_RESEARCH_MAX_BYTES: int = AIDEV_DEEP_RESEARCH_MAX_CHARS
