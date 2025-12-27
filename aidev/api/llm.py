from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from fastapi import APIRouter
from pydantic import BaseModel, Field
import hashlib

from ..llm_client import (
    LLMClient,
    summarize_file,
    ChatResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/llm", tags=["llm"])

# -------- Defaults (env-tunable) --------
PLAN_MAX = int(os.getenv("AIDEV_PLAN_MAX_OUTPUT_TOKENS", "8192"))             # ~800–1200
DIFFS_MAX = int(os.getenv("AIDEV_DIFFS_MAX_OUTPUT_TOKENS", "60000"))          # 4096–8192 when needed
SUMMARY_MAX = int(os.getenv("AIDEV_SUMMARY_MAX_OUTPUT_TOKENS", "4096"))       # 1028–4096 typical
DEFAULT_MODEL = os.getenv("OPENAI_MODEL") or "gpt-5-mini"


# -------- Schemas --------
class PlanRequest(BaseModel):
    context_text: str = Field(..., description="Natural-language description / context to plan against.")
    model: Optional[str] = Field(None, description="Model.")
    stage: Optional[str] = Field(None, description="Optional stage key used to resolve per-stage model routing.")
    max_output_tokens: Optional[int] = Field(None, ge=128, description="Override tokens for this call only.")

class PlanResponse(BaseModel):
    plan: Dict[str, Any]
    usage: Dict[str, int]
    resolved_model: Optional[str] = None

class DiffsRequest(BaseModel):
    spec_text: str = Field(..., description="Change request / spec for generating diffs.")
    model: Optional[str] = None
    stage: Optional[str] = Field(None, description="Optional stage key used to resolve per-stage model routing.")
    max_output_tokens: Optional[int] = Field(None, ge=512)
    prefer_patches: bool = Field(False, description="If true, encourage patches over full-file contents.")

class DiffsResponse(BaseModel):
    diffs: Dict[str, Any]
    usage: Dict[str, int]
    resolved_model: Optional[str] = None

class SummarizeFileRequest(BaseModel):
    rel_path: str
    content: str
    model: Optional[str] = None
    stage: Optional[str] = Field(None, description="Optional stage key used to resolve per-stage model routing.")
    max_output_tokens: Optional[int] = Field(None, ge=64)

class SummarizeFileResponse(BaseModel):
    summary: str
    resolved_model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


# -------- Helpers --------

def _usage_dict(resp: Optional[ChatResponse]) -> Dict[str, int]:
    if not resp:
        return {}
    return {
        "prompt_tokens": int(resp.prompt_tokens or 0),
        "completion_tokens": int(resp.completion_tokens or 0),
        "total_tokens": int(resp.total_tokens or 0),
    }


# -------- Endpoints --------

@router.post("/summarize-file", response_model=SummarizeFileResponse)
def summarize_file(req: SummarizeFileRequest) -> SummarizeFileResponse:
    """
    Concise per-file summary. Output budget defaults ~512, tunable per call.
    Accepts optional `stage` to allow per-stage model routing in the llm client. Returns resolved_model and usage metadata.
    """
    # Call the helper. It may return either a string or a (summary, resp) tuple depending on llm_client implementation.
    summary_result = summarize_file(
        path=Path(req.rel_path),
        content=req.content or "",
        context=None,
        model=req.model or None,
        stage=req.stage,
        max_tokens=int(req.max_output_tokens or SUMMARY_MAX),
    )

    summary = None
    resp: Optional[ChatResponse] = None

    # helper might return (summary, resp) or just summary string
    if isinstance(summary_result, tuple) and len(summary_result) == 2:
        summary, resp = summary_result
    else:
        summary = summary_result

    resolved_model = None
    if resp is not None:
        resolved_model = getattr(resp, "model", None) or getattr(resp, "model_name", None)
    if not resolved_model:
        resolved_model = req.model or DEFAULT_MODEL

    logger.info("LLM call: stage=%r resolved_model=%r", req.stage, resolved_model)

    return SummarizeFileResponse(summary=summary or "", resolved_model=resolved_model, usage=_usage_dict(resp))
