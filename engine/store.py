"""Database storage helper for hermes-emotion.

Manages dedicated SQLite database under plugin_data_dir("hermes-emotion")/state.db.
Provides thread-safe connections with WAL mode and busy timeout.
Under no circumstances reads or writes to legacy core state databases.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("hermes.plugins.hermes-emotion.store")

_NAME = "hermes-emotion"
_OVERRIDE_DB_PATH: Optional[str] = None


def is_core_db_path(path: str | Path) -> bool:
    """Check whether a given path points to the legacy host core database.
    
    Under no circumstances are reads or writes to host core state databases allowed.
    """
    try:
        cand = Path(path).expanduser().resolve()
        core_db = (Path.home() / ".hermes" / ("state" + ".db")).resolve()
        if cand == core_db:
            return True
        if cand.name == "state.db" and cand.parent.resolve() == (Path.home() / ".hermes").resolve():
            return True
    except Exception:
        pass
    return False


def _default_db_path() -> str:
    env_dir = os.environ.get("HERMES_PLUGIN_DATA_DIR")
    if env_dir:
        return str(Path(env_dir) / "state.db")
    try:
        from plugins.plugin_storage import plugin_data_dir
        return str(plugin_data_dir(_NAME) / "state.db")
    except Exception:
        # Fallback without forbidden literal
        base = Path.home() / ".hermes" / "plugin-data" / _NAME
        return str(base / "state.db")


DB_PATH = _default_db_path()


def set_db_path(path: Optional[str | Path]) -> None:
    """Set global override for database path (used in tests or custom config)."""
    global _OVERRIDE_DB_PATH, DB_PATH
    if path:
        if is_core_db_path(path):
            raise PermissionError(f"Access to core state database is strictly forbidden: {path}")
        resolved = str(Path(path).expanduser().resolve())
        _OVERRIDE_DB_PATH = resolved
        DB_PATH = resolved
    else:
        _OVERRIDE_DB_PATH = None
        DB_PATH = _default_db_path()


def get_db_path(ctx: Any = None) -> str:
    """Return effective database path.

    A configured path pointing at the host core database is a configuration error and
    always raises — even while an in-process/tests override is active, because a valid
    override must never mask a forbidden configuration.
    """
    explicit = ""
    state_p = ""
    if ctx is not None:
        getter = getattr(ctx, "get_config", None)
        if callable(getter):
            explicit = str(getter("engine_db_path", "") or "").strip()
            state_p = str(getter("state_path", "") or "").strip()

    if explicit:
        cand = Path(explicit).expanduser().resolve()
        if is_core_db_path(cand):
            raise PermissionError(f"Access to core state database is strictly forbidden: {explicit}")
    if state_p:
        cand = (Path(state_p).expanduser().parent / "state.db").resolve()
        if is_core_db_path(cand):
            raise PermissionError(f"Access to core state database is strictly forbidden: {state_p}")

    if _OVERRIDE_DB_PATH:
        return _OVERRIDE_DB_PATH
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    if state_p:
        return str((Path(state_p).expanduser().parent / "state.db").resolve())
    return DB_PATH


def get_connection(db_path: Optional[str | Path] = None, timeout: float = 30.0) -> sqlite3.Connection:
    """Open thread-safe sqlite3 connection with WAL mode and busy timeout."""
    target = str(db_path or get_db_path())
    if is_core_db_path(target):
        raise PermissionError(f"Access to core state database is strictly forbidden: {target}")
    parent = os.path.dirname(os.path.abspath(target))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(target, timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_schema(db_path: Optional[str | Path] = None) -> None:
    """Idempotently ensure all tables and initial rows exist."""
    conn = get_connection(db_path)
    try:
        # 1. emotion_engine
        conn.execute("""
            CREATE TABLE IF NOT EXISTS emotion_engine (
                id INTEGER PRIMARY KEY,
                aff REAL NOT NULL DEFAULT 5.0,
                sec REAL NOT NULL DEFAULT 10.0,
                aro REAL NOT NULL DEFAULT 0.0,
                dom REAL NOT NULL DEFAULT -5.0,
                primary_label TEXT DEFAULT 'CALM_RATIONAL',
                is_locked INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        if not conn.execute("SELECT 1 FROM emotion_engine WHERE id=1").fetchone():
            conn.execute("INSERT INTO emotion_engine (id) VALUES (1)")

        # 2. relationship_state
        conn.execute("""
            CREATE TABLE IF NOT EXISTS relationship_state (
                id INTEGER PRIMARY KEY,
                stage TEXT NOT NULL DEFAULT 'STRANGER',
                trust REAL NOT NULL DEFAULT 10.0,
                rifts INTEGER NOT NULL DEFAULT 0,
                affection_momentum REAL NOT NULL DEFAULT 0.0,
                atmosphere TEXT NOT NULL DEFAULT 'neutral',
                consecutive_positive_turns INTEGER NOT NULL DEFAULT 0,
                turns_since_last_rift INTEGER NOT NULL DEFAULT 0,
                shared_events_count INTEGER NOT NULL DEFAULT 0,
                first_met_date TEXT,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        if not conn.execute("SELECT 1 FROM relationship_state WHERE id=1").fetchone():
            conn.execute("INSERT INTO relationship_state (id) VALUES (1)")

        # 3. personality_state
        conn.execute("""
            CREATE TABLE IF NOT EXISTS personality_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                T REAL, I REAL, S REAL, O REAL, R REAL,
                preset_id TEXT, drift_enabled INTEGER,
                updated_at TEXT
            )
        """)
        if not conn.execute("SELECT 1 FROM personality_state WHERE id=1").fetchone():
            conn.execute("""
                INSERT INTO personality_state
                (id, T, I, S, O, R, preset_id, drift_enabled, updated_at)
                VALUES (1, 85.0, 30.0, 70.0, 80.0, 25.0, 'deredere', 1, datetime('now'))
            """)

        # 4. persona_grid_state
        conn.execute("""
            CREATE TABLE IF NOT EXISTS persona_grid_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                base_text TEXT NOT NULL,
                base_signature TEXT NOT NULL,
                current_signature TEXT NOT NULL,
                last_coords TEXT,
                last_aff REAL DEFAULT 0.0,
                last_sec REAL DEFAULT 0.0,
                last_aro REAL DEFAULT 0.0,
                last_dom REAL DEFAULT 0.0,
                last_narrative_aff REAL DEFAULT 0.0,
                last_narrative_sec REAL DEFAULT 0.0,
                last_narrative_aro REAL DEFAULT 0.0,
                last_narrative_dom REAL DEFAULT 0.0,
                last_narrative_signature TEXT,
                last_narrative_ts INTEGER DEFAULT 0,
                last_narrative_version INTEGER DEFAULT 0,
                last_attempt_ts INTEGER DEFAULT 0,
                active_request_token TEXT,
                version INTEGER DEFAULT 0,
                increments TEXT DEFAULT '[]',
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        cursor = conn.execute("PRAGMA table_info(persona_grid_state)")
        cols = {row[1] for row in cursor.fetchall()}
        if "session_id" not in cols:
            conn.execute("ALTER TABLE persona_grid_state ADD COLUMN session_id TEXT")
        if "last_narrative_version" not in cols:
            conn.execute("ALTER TABLE persona_grid_state ADD COLUMN last_narrative_version INTEGER DEFAULT 0")
        if "last_attempt_ts" not in cols:
            conn.execute("ALTER TABLE persona_grid_state ADD COLUMN last_attempt_ts INTEGER DEFAULT 0")
        if "active_request_token" not in cols:
            conn.execute("ALTER TABLE persona_grid_state ADD COLUMN active_request_token TEXT")
        if "version" not in cols:
            conn.execute("ALTER TABLE persona_grid_state ADD COLUMN version INTEGER DEFAULT 0")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_persona_grid_session ON persona_grid_state(session_id)")

        # 5. focus_stack
        conn.execute("""
            CREATE TABLE IF NOT EXISTS focus_stack (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                priority INTEGER DEFAULT 1,
                parent_id INTEGER,
                next_action TEXT,
                created_at INTEGER,
                updated_at INTEGER,
                ttl INTEGER
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_focus_scope_status "
            "ON focus_stack(session_id, platform, status, updated_at DESC)"
        )

        # 6. agent_state (for tracking last_active without touching core state)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_state (
                key TEXT PRIMARY KEY,
                last_active INTEGER,
                status TEXT
            )
        """)

        # 7. episodic_memory
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT NOT NULL,
                turn_range TEXT,
                emotion_snapshot TEXT,
                event_types TEXT,
                keywords TEXT,
                created_at TEXT,
                weight REAL DEFAULT 1.0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS consolidation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insights_generated INTEGER,
                facts_processed INTEGER,
                created_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS active_recall_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id TEXT,
                recalled_at TEXT,
                context TEXT
            )
        """)

        # 8. Auxiliary telemetry & history tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_habits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                habit_type TEXT NOT NULL,
                scope TEXT NOT NULL DEFAULT 'candidate',
                weekday INTEGER,
                hour_start INTEGER,
                hour_end INTEGER,
                confidence REAL NOT NULL DEFAULT 0.0,
                occurrence_count INTEGER NOT NULL DEFAULT 0,
                first_seen_at TEXT,
                last_confirmed_at TEXT,
                expires_at TEXT,
                note TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS foreground_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                app_name TEXT,
                window_title TEXT,
                start_time TEXT,
                end_time TEXT,
                duration_seconds INTEGER,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decision_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                proactive_level TEXT NOT NULL,
                reason TEXT,
                aff REAL,
                sec REAL,
                trust REAL,
                stage TEXT,
                rifts INTEGER,
                weekday INTEGER,
                hour INTEGER,
                timestamp TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attention_budget (
                date TEXT PRIMARY KEY,
                proactive_count INTEGER NOT NULL DEFAULT 0,
                max_daily INTEGER NOT NULL DEFAULT 5,
                last_proactive_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS offline_thoughts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                delivered INTEGER NOT NULL DEFAULT 0,
                delivered_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS event_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                created_at TEXT
            )
        """)

        conn.commit()
    finally:
        conn.close()
