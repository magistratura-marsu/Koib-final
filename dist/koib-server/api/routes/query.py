# -*- coding: utf-8 -*-
from __future__ import annotations

import hmac
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from api.deps import get_pipeline

logger = logging.getLogger("koib.api.query")
router = APIRouter()

class QueryRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    query: str = Field(..., min_length=1, max_length=3000)
    user_id: str = Field("api", min_length=1, max_length=128)
    top_k: int = Field(4, ge=1, le=10)
    model_filter: Literal["", "koib2010", "koib2017a", "koib2017b"] = ""
    use_memory: bool = False
    validate_answer: bool = Field(True, alias="validate")

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    figures: List[Dict[str, Any]] = Field(default_factory=list)
    status: str
    latency: float

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest, x_api_key: str | None = Header(default=None)) -> Dict[str, Any]:
    expected = os.getenv("KOIB_API_KEY", "").strip()
    if not expected:
        if os.getenv("KOIB_ALLOW_UNAUTHENTICATED", "false").lower() != "true":
            raise HTTPException(status_code=503, detail="API key is not configured")
    elif not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid API key")
    try:
        result = await get_pipeline().answer(
            query=payload.query, user_id=payload.user_id,
            k=payload.top_k, model_filter=payload.model_filter,
            use_memory=payload.use_memory, validate=payload.validate_answer,
        )
        # Do not expose absolute host filesystem paths through the public API.
        result["figures"] = [
            {**{key: value for key, value in figure.items() if key != "image_path"},
             "file_name": Path(figure.get("image_path", "")).name}
            for figure in result.get("figures", []) if isinstance(figure, dict)
        ]
        return result
    except Exception:
        logger.exception("Ошибка обработки запроса")
        raise HTTPException(status_code=503, detail="Retrieval service unavailable") from None
