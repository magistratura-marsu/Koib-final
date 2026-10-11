# -*- coding: utf-8 -*-
"""Liveness and readiness based on real persisted SQLite index artifacts."""
from __future__ import annotations

import sqlite3
import os
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from config import APP_NAME, APP_VERSION, INDEX_DIR, DOCSTORE_DIR

router = APIRouter()
_start_time = time.monotonic()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    return {"status": "ok", "uptime_seconds": round(time.monotonic() - _start_time, 1), "version": APP_VERSION}

@router.get("/ready")
async def readiness_check():
    bm25 = Path(INDEX_DIR) / "bm25_fts.db"
    docstore = Path(DOCSTORE_DIR) / "docstore.db"
    text_index = Path(INDEX_DIR) / "text_index.faiss"
    summary_index = Path(INDEX_DIR) / "summary_index.faiss"
    fts_ok = docstore_ok = False
    for path, table in ((bm25, "chunks_fts"), (docstore, "docstore")):
        if not path.is_file():
            continue
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                ok = bool(conn.execute("SELECT 1 FROM " + table + " LIMIT 1").fetchone())
            if table == "chunks_fts":
                fts_ok = ok
            else:
                docstore_ok = ok
        except sqlite3.Error:
            pass
    ready = fts_ok and docstore_ok
    # Detect that vector artifacts are trusted by deployment configuration;
    # existence alone doesn't mean retrieval will deserialize them.
    vector_present = (os.getenv("KOIB_TRUST_FAISS_PICKLE", "false").lower() == "true"
                      and (text_index.is_file() or summary_index.is_file()))
    body = {
        "status": ("ready" if vector_present else "degraded") if ready else "not_ready",
        "version": APP_VERSION,
        "indexes": {"text_index": text_index.is_file(), "summary_index": summary_index.is_file(),
                    "bm25_index": fts_ok, "docstore": docstore_ok},
    }
    return JSONResponse(status_code=200 if ready else 503, content=body)

@router.get("/")
async def root() -> Dict[str, str]:
    return {"name": APP_NAME, "version": APP_VERSION,
            "description": "Поиск и ответы по разнородным пользовательским документам",
            "docs": "/docs"}
