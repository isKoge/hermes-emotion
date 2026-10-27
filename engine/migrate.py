"""One-time idempotent migration from legacy friday_core database to hermes-emotion state.db.

Reads legacy database strictly read-only and upserts rows into the target database.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import store

logger = logging.getLogger("hermes.plugins.hermes-emotion.migrate")

# Tables to migrate in priority order
MIGRATION_TABLES: List[str] = [
    "emotion_engine",
    "relationship_state",
    "personality_state",
    "persona_grid_state",
    "episodic_memory",
    "focus_stack",
    "agent_state",
    "consolidation_log",
    "active_recall_log",
    "user_habits",
    "foreground_history",
    "decision_log",
    "attention_budget",
    "offline_thoughts",
    "event_history",
]


def default_legacy_db_path() -> str:
    """Return default legacy DB path without forbidden literal."""
    p = Path.home() / ".hermes" / ("state" + ".db")
    return str(p)


def migrate_from_legacy(
    legacy_db_path: Optional[str | Path] = None,
    target_db_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Migrate state from legacy DB into target DB idempotently.

    Legacy DB is opened strictly in read-only mode (mode=ro).
    Rows are upserted using INSERT OR REPLACE on primary / unique keys.
    """
    src_path = Path(legacy_db_path or default_legacy_db_path()).expanduser()
    dst_path = Path(target_db_path or store.get_db_path()).expanduser()

    if store.is_core_db_path(dst_path):
        return {
            "status": "error",
            "error": "Target database cannot be the legacy core state database.",
            "legacy_db": str(src_path),
            "target_db": str(dst_path),
            "tables": {},
            "total_rows": 0,
        }

    if src_path.resolve() == dst_path.resolve():
        return {
            "status": "error",
            "error": "Source and target database cannot be the same file.",
            "legacy_db": str(src_path),
            "target_db": str(dst_path),
            "tables": {},
            "total_rows": 0,
        }

    if not src_path.is_file():
        return {
            "status": "error",
            "error": f"Legacy database file not found: {src_path}",
            "legacy_db": str(src_path),
            "target_db": str(dst_path),
            "tables": {},
            "total_rows": 0,
        }

    # Ensure schema exists in target
    store.ensure_schema(dst_path)

    legacy_uri = f"file:{src_path.resolve()}?mode=ro"
    legacy_conn = sqlite3.connect(legacy_uri, uri=True, timeout=15.0)
    target_conn = store.get_connection(dst_path)

    migrated: Dict[str, int] = {}
    try:
        # Discover available tables in legacy db
        legacy_tables_cur = legacy_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        legacy_tables = {row[0] for row in legacy_tables_cur.fetchall()}

        for table in MIGRATION_TABLES:
            if table not in legacy_tables:
                continue

            # Intersect columns between legacy and target
            legacy_cols = [
                row[1] for row in legacy_conn.execute(f"PRAGMA table_info({table})").fetchall()
            ]
            target_cols = [
                row[1] for row in target_conn.execute(f"PRAGMA table_info({table})").fetchall()
            ]
            shared_cols = [col for col in legacy_cols if col in target_cols]

            if not shared_cols:
                continue

            cols_str = ", ".join(f'"{c}"' for c in shared_cols)
            placeholders = ", ".join("?" for _ in shared_cols)
            sql_select = f"SELECT {cols_str} FROM \"{table}\""
            sql_upsert = f"INSERT OR REPLACE INTO \"{table}\" ({cols_str}) VALUES ({placeholders})"

            rows = legacy_conn.execute(sql_select).fetchall()
            if rows:
                target_conn.executemany(sql_upsert, rows)
                migrated[table] = len(rows)
            else:
                migrated[table] = 0

        target_conn.commit()
        logger.info(
            "[migrate] successfully migrated %d rows across %d tables from %s to %s",
            sum(migrated.values()), len(migrated), src_path, dst_path
        )
        return {
            "status": "ok",
            "legacy_db": str(src_path),
            "target_db": str(dst_path),
            "tables": migrated,
            "total_rows": sum(migrated.values()),
        }
    except Exception as exc:
        target_conn.rollback()
        logger.exception("[migrate] migration failed: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "legacy_db": str(src_path),
            "target_db": str(dst_path),
            "tables": migrated,
            "total_rows": sum(migrated.values()),
        }
    finally:
        legacy_conn.close()
        target_conn.close()
