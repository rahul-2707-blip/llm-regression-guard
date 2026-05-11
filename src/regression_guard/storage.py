"""SQLite-backed storage for eval runs. Simple, portable, git-friendly."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
from .contracts import EvalRun, CaseResult, ClassifierOutput


DB_PATH = Path(__file__).resolve().parents[2] / "eval_runs" / "runs.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                prompt_fingerprint TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                model TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                overall_accuracy REAL NOT NULL,
                mean_summary_score REAL NOT NULL,
                mean_latency_ms REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS case_results (
                run_id TEXT NOT NULL,
                case_id TEXT NOT NULL,
                raw_output TEXT,
                parsed_json TEXT,
                parse_error TEXT,
                category_match INTEGER NOT NULL,
                summary_score INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                PRIMARY KEY (run_id, case_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(timestamp);
            """
        )


def save_run(run: EvalRun) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run.run_id,
                run.prompt_fingerprint,
                run.prompt_version,
                run.model,
                run.timestamp.isoformat(),
                run.overall_accuracy,
                run.mean_summary_score,
                run.mean_latency_ms,
            ),
        )
        c.executemany(
            "INSERT OR REPLACE INTO case_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    run.run_id,
                    r.case_id,
                    r.raw_output,
                    r.parsed.model_dump_json() if r.parsed else None,
                    r.parse_error,
                    int(r.category_match),
                    r.summary_score,
                    r.latency_ms,
                    r.input_tokens,
                    r.output_tokens,
                )
                for r in run.case_results
            ],
        )


def get_run(run_id: str) -> Optional[EvalRun]:
    with _conn() as c:
        row = c.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        cases = c.execute(
            "SELECT * FROM case_results WHERE run_id = ?", (run_id,)
        ).fetchall()
    return EvalRun(
        run_id=row["run_id"],
        prompt_fingerprint=row["prompt_fingerprint"],
        prompt_version=row["prompt_version"],
        model=row["model"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        case_results=[
            CaseResult(
                case_id=c["case_id"],
                raw_output=c["raw_output"],
                parsed=ClassifierOutput(**json.loads(c["parsed_json"])) if c["parsed_json"] else None,
                parse_error=c["parse_error"],
                category_match=bool(c["category_match"]),
                summary_score=c["summary_score"],
                latency_ms=c["latency_ms"],
                input_tokens=c["input_tokens"],
                output_tokens=c["output_tokens"],
            )
            for c in cases
        ],
    )


def latest_runs(limit: int = 10) -> list[EvalRun]:
    with _conn() as c:
        rows = c.execute(
            "SELECT run_id FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [r for r in (get_run(row["run_id"]) for row in rows) if r is not None]


def previous_run(before_run_id: str) -> Optional[EvalRun]:
    """The run immediately before this one (by timestamp)."""
    with _conn() as c:
        row = c.execute(
            "SELECT timestamp FROM runs WHERE run_id = ?", (before_run_id,)
        ).fetchone()
        if not row:
            return None
        prev = c.execute(
            "SELECT run_id FROM runs WHERE timestamp < ? ORDER BY timestamp DESC LIMIT 1",
            (row["timestamp"],),
        ).fetchone()
    return get_run(prev["run_id"]) if prev else None
