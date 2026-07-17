"""SQLite persistence for fleet final results.

Keeps JSON files as-is and mirrors each final payload into ``results.db``.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Optional


DEFAULT_DB_PATH = Path("work_fleet/results.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    finished_at REAL,
    request_path TEXT,
    work_root TEXT,
    strategy TEXT,
    job_count INTEGER DEFAULT 0,
    ok_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    job_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    strategy TEXT,
    input_path TEXT,
    output_path TEXT,
    elapsed_sec REAL,
    uploaded INTEGER DEFAULT 0,
    error TEXT,
    use_gpu INTEGER DEFAULT 0,
    encoder TEXT,
    recipe TEXT,
    mode TEXT,
    crf REAL,
    vmaf REAL,
    vmaf_base REAL,
    vmaf_delta REAL,
    s_f REAL,
    compression_rate REAL,
    compression_ratio REAL,
    features_json TEXT,
    stage_timings_json TEXT,
    best_json TEXT,
    payload_json TEXT NOT NULL,
    UNIQUE(run_id, job_id)
);

CREATE INDEX IF NOT EXISTS idx_results_job_id ON results(job_id);
CREATE INDEX IF NOT EXISTS idx_results_created_at ON results(created_at);
CREATE INDEX IF NOT EXISTS idx_results_crf ON results(crf);
"""


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def start_run(
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    request_path: Optional[str] = None,
    work_root: Optional[str] = None,
    strategy: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    rid = run_id or new_run_id()
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs(
                run_id, created_at, request_path, work_root, strategy
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (rid, time.time(), request_path, work_root, strategy),
        )
        conn.commit()
    return rid


def finish_run(
    run_id: str,
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    job_count: int = 0,
    ok_count: int = 0,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE runs
            SET finished_at = ?, job_count = ?, ok_count = ?
            WHERE run_id = ?
            """,
            (time.time(), int(job_count), int(ok_count), run_id),
        )
        conn.commit()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def upsert_result(
    payload: dict[str, Any],
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    run_id: Optional[str] = None,
) -> int:
    """Insert or replace one final job payload. Returns row id."""
    best = payload.get("best") if isinstance(payload.get("best"), dict) else {}
    features = payload.get("features") if isinstance(payload.get("features"), dict) else {}
    timings = (
        payload.get("stage_timings")
        if isinstance(payload.get("stage_timings"), dict)
        else {}
    )
    recipes = payload.get("recipes") or []
    recipe = recipes[0] if isinstance(recipes, list) and recipes else best.get("recipe")

    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO results(
                run_id, job_id, created_at, strategy, input_path, output_path,
                elapsed_sec, uploaded, error, use_gpu, encoder, recipe, mode,
                crf, vmaf, vmaf_base, vmaf_delta, s_f, compression_rate,
                compression_ratio, features_json, stage_timings_json, best_json,
                payload_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
            ON CONFLICT(run_id, job_id) DO UPDATE SET
                created_at=excluded.created_at,
                strategy=excluded.strategy,
                input_path=excluded.input_path,
                output_path=excluded.output_path,
                elapsed_sec=excluded.elapsed_sec,
                uploaded=excluded.uploaded,
                error=excluded.error,
                use_gpu=excluded.use_gpu,
                encoder=excluded.encoder,
                recipe=excluded.recipe,
                mode=excluded.mode,
                crf=excluded.crf,
                vmaf=excluded.vmaf,
                vmaf_base=excluded.vmaf_base,
                vmaf_delta=excluded.vmaf_delta,
                s_f=excluded.s_f,
                compression_rate=excluded.compression_rate,
                compression_ratio=excluded.compression_ratio,
                features_json=excluded.features_json,
                stage_timings_json=excluded.stage_timings_json,
                best_json=excluded.best_json,
                payload_json=excluded.payload_json
            """,
            (
                run_id,
                str(payload.get("job_id") or ""),
                time.time(),
                payload.get("strategy"),
                payload.get("input_path"),
                payload.get("output_path"),
                payload.get("elapsed_sec"),
                1 if payload.get("uploaded") else 0,
                payload.get("error") or "",
                1 if payload.get("use_gpu") else 0,
                best.get("encoder"),
                recipe,
                best.get("mode"),
                best.get("crf"),
                best.get("vmaf"),
                best.get("vmaf_base"),
                best.get("vmaf_delta"),
                best.get("s_f"),
                best.get("compression_rate"),
                best.get("compression_ratio"),
                _json_dumps(features),
                _json_dumps(timings),
                _json_dumps(best),
                _json_dumps(payload),
            ),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def import_result_json(
    path: Path | str,
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    run_id: Optional[str] = None,
) -> Optional[int]:
    p = Path(path)
    if not p.is_file():
        return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("job_id"):
        return None
    return upsert_result(payload, db_path=db_path, run_id=run_id)


def import_work_root(
    work_root: Path | str = "work_fleet",
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    run_id: Optional[str] = None,
) -> tuple[str, int]:
    """Import all ``*/result.json`` under work_root. Returns (run_id, count)."""
    root = Path(work_root)
    rid = run_id or start_run(
        db_path=db_path,
        work_root=str(root),
        strategy="import",
    )
    count = 0
    seen: set[str] = set()
    for path in sorted(root.glob("*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload.get("job_id"):
            continue
        job_id = str(payload["job_id"])
        upsert_result(payload, db_path=db_path, run_id=rid)
        if job_id not in seen:
            seen.add(job_id)
            count += 1
    batch = root / "batch_results.json"
    if batch.is_file() and not seen:
        data = json.loads(batch.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("job_id"):
                    upsert_result(item, db_path=db_path, run_id=rid)
                    count += 1
    finish_run(rid, db_path=db_path, job_count=count, ok_count=count)
    return rid, count


def list_latest_results(
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    limit: int = 50,
) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT job_id, crf, vmaf, vmaf_base, s_f, compression_rate,
                   compression_ratio, encoder, error, created_at, run_id
            FROM results
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]
