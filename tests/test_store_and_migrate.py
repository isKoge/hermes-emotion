"""Tests for storage isolation and migration idempotence."""

from __future__ import annotations

import os
import sqlite3
import pytest
from pathlib import Path

from engine import store
from engine import migrate


def test_store_schema_creation_and_defaults(tmp_path):
    target_db = tmp_path / "test_store.db"
    store.ensure_schema(target_db)

    conn = store.get_connection(target_db)
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        for expected in [
            "emotion_engine",
            "relationship_state",
            "personality_state",
            "persona_grid_state",
            "focus_stack",
            "episodic_memory",
            "agent_state",
        ]:
            assert expected in tables, f"Expected table {expected} to exist"

        # Check default row initialization
        emo = conn.execute("SELECT aff, sec, aro, dom, primary_label FROM emotion_engine WHERE id=1").fetchone()
        assert emo == (5.0, 10.0, 0.0, -5.0, "CALM_RATIONAL")

        rel = conn.execute("SELECT stage, trust, rifts FROM relationship_state WHERE id=1").fetchone()
        assert rel == ("STRANGER", 10.0, 0)

        pers = conn.execute("SELECT preset_id, drift_enabled FROM personality_state WHERE id=1").fetchone()
        assert pers == ("deredere", 1)
    finally:
        conn.close()


def test_store_wal_and_busy_timeout(tmp_path):
    target_db = tmp_path / "wal_test.db"
    conn = store.get_connection(target_db)
    try:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal_mode.lower() == "wal"
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert busy_timeout == 30000
    finally:
        conn.close()


def test_migration_success_and_idempotence(tmp_path):
    legacy_db = tmp_path / "legacy.db"
    target_db = tmp_path / "target.db"

    # Populate legacy DB
    conn = sqlite3.connect(legacy_db)
    conn.execute("""
        CREATE TABLE emotion_engine (
            id INTEGER PRIMARY KEY,
            aff REAL, sec REAL, aro REAL, dom REAL,
            primary_label TEXT, is_locked INTEGER, updated_at TEXT
        )
    """)
    conn.execute("INSERT INTO emotion_engine VALUES (1, 62.5, 45.0, 15.0, 8.0, 'SWEET_ATTACHMENT', 0, '2026-10-27')")

    conn.execute("""
        CREATE TABLE relationship_state (
            id INTEGER PRIMARY KEY,
            stage TEXT, trust REAL, rifts INTEGER, affection_momentum REAL,
            atmosphere TEXT, consecutive_positive_turns INTEGER, turns_since_last_rift INTEGER,
            shared_events_count INTEGER, first_met_date TEXT, updated_at TEXT
        )
    """)
    conn.execute("INSERT INTO relationship_state VALUES (1, 'INTIMATE', 78.0, 2, 0.4, 'warm', 15, 8, 42, '2026-01-01', '2026-10-27')")

    conn.execute("""
        CREATE TABLE personality_state (
            id INTEGER PRIMARY KEY,
            T REAL, I REAL, S REAL, O REAL, R REAL,
            preset_id TEXT, drift_enabled INTEGER, updated_at TEXT
        )
    """)
    conn.execute("INSERT INTO personality_state VALUES (1, 75.0, 40.0, 40.0, 90.0, 20.0, 'genki', 1, '2026-10-27')")

    conn.execute("""
        CREATE TABLE persona_grid_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            base_text TEXT, base_signature TEXT, current_signature TEXT,
            increments TEXT, updated_at TEXT
        )
    """)
    conn.execute("INSERT INTO persona_grid_state (session_id, base_text, base_signature, current_signature, increments) VALUES ('default', '心情很好', '近·稳·热·进', '近·稳·热·进', '[\"加深羁绊\"]')")

    conn.execute("""
        CREATE TABLE episodic_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary TEXT, turn_range TEXT, emotion_snapshot TEXT,
            event_types TEXT, keywords TEXT, created_at TEXT, weight REAL
        )
    """)
    conn.execute("INSERT INTO episodic_memory (id, summary, turn_range, emotion_snapshot, event_types, keywords, created_at, weight) VALUES (1, '聊起了过去的旅行', '1-6', '{}', '[]', '[]', '2026-10-27', 1.0)")
    conn.execute("INSERT INTO episodic_memory (id, summary, turn_range, emotion_snapshot, event_types, keywords, created_at, weight) VALUES (2, '分享了一首喜欢的歌', '7-12', '{}', '[]', '[]', '2026-10-27', 1.0)")

    conn.commit()
    conn.close()

    mtime_before = os.path.getmtime(legacy_db)

    # First migration run
    res1 = migrate.migrate_from_legacy(legacy_db, target_db)
    assert res1["status"] == "ok"
    assert res1["tables"]["emotion_engine"] == 1
    assert res1["tables"]["relationship_state"] == 1
    assert res1["tables"]["personality_state"] == 1
    assert res1["tables"]["persona_grid_state"] == 1
    assert res1["tables"]["episodic_memory"] == 2

    # Verify content in target
    tconn = store.get_connection(target_db)
    emo_row = tconn.execute("SELECT aff, sec, primary_label FROM emotion_engine WHERE id=1").fetchone()
    assert emo_row == (62.5, 45.0, "SWEET_ATTACHMENT")

    rel_row = tconn.execute("SELECT stage, trust, rifts FROM relationship_state WHERE id=1").fetchone()
    assert rel_row == ("INTIMATE", 78.0, 2)

    pers_row = tconn.execute("SELECT preset_id FROM personality_state WHERE id=1").fetchone()
    assert pers_row == ("genki",)

    ep_count = tconn.execute("SELECT count(*) FROM episodic_memory").fetchone()[0]
    assert ep_count == 2
    tconn.close()

    # Second migration run (idempotence check)
    res2 = migrate.migrate_from_legacy(legacy_db, target_db)
    assert res2["status"] == "ok"
    assert res2["total_rows"] == res1["total_rows"]

    tconn2 = store.get_connection(target_db)
    ep_count2 = tconn2.execute("SELECT count(*) FROM episodic_memory").fetchone()[0]
    assert ep_count2 == 2
    tconn2.close()

    # Check that legacy db was strictly read-only and unchanged
    mtime_after = os.path.getmtime(legacy_db)
    assert mtime_before == mtime_after


def test_migration_missing_file_fails_gracefully(tmp_path):
    missing_db = tmp_path / "does_not_exist.db"
    target_db = tmp_path / "target.db"
    res = migrate.migrate_from_legacy(missing_db, target_db)
    assert res["status"] == "error"
    assert "not found" in res["error"]


def test_forbidden_core_db_access_blocked():
    """Ensure accessing host core state database raises PermissionError."""
    forbidden = Path.home() / ".hermes" / ("state" + ".db")
    with pytest.raises(PermissionError):
        store.set_db_path(forbidden)
    with pytest.raises(PermissionError):
        store.get_connection(forbidden)


def test_migration_guards_against_identical_or_core_target(tmp_path):
    """Ensure source == target and core target are blocked."""
    dummy_db = tmp_path / "dummy.db"
    dummy_db.touch()

    # Source == target
    res_same = migrate.migrate_from_legacy(dummy_db, dummy_db)
    assert res_same["status"] == "error"
    assert "cannot be the same file" in res_same["error"]

    # Target is core DB
    forbidden = Path.home() / ".hermes" / ("state" + ".db")
    res_core = migrate.migrate_from_legacy(dummy_db, forbidden)
    assert res_core["status"] == "error"
    assert "core state database" in res_core["error"]


def test_dynamic_db_path_propagation(tmp_path):
    """Ensure store.set_db_path immediately propagates to personality and focus modules."""
    db1 = tmp_path / "dynamic1.db"
    db2 = tmp_path / "dynamic2.db"

    store.set_db_path(db1)
    store.ensure_schema(db1)
    from engine import personality, focus_stack
    p1 = personality.load_personality()
    p1.preset_id = "test_preset_1"
    personality.save_personality(p1)

    store.set_db_path(db2)
    store.ensure_schema(db2)
    p2 = personality.load_personality()
    # In db2, it should be the default, not test_preset_1
    assert p2.preset_id == "deredere"


def test_all_entry_points_reject_core_db(tmp_path):
    """Verify that all state reading, writing, external source loading and injection
    strictly reject core database path and do not modify it.
    """
    import engine.emotion_grid as eg
    import __init__ as hermes_emotion

    forbidden = Path.home() / ".hermes" / ("state" + ".db")
    mtime_before = forbidden.stat().st_mtime_ns if forbidden.exists() else None

    class Ctx:
        def __init__(self, cfg):
            self.cfg = cfg
        def get_config(self, k, d=None):
            return self.cfg.get(k, d)

    with pytest.raises(PermissionError):
        store.set_db_path(forbidden)
    with pytest.raises(PermissionError):
        store.get_connection(forbidden)
    with pytest.raises(PermissionError):
        store.get_db_path(Ctx({"engine_db_path": str(forbidden)}))
    with pytest.raises(PermissionError):
        store.get_db_path(Ctx({"state_path": str(forbidden)}))
    with pytest.raises(PermissionError):
        eg.load_state(forbidden)
    with pytest.raises(PermissionError):
        eg.save_state(forbidden, {})
    with pytest.raises(PermissionError):
        eg._read_json_source(forbidden)
    with pytest.raises(PermissionError):
        eg._read_sqlite_source(forbidden)
    with pytest.raises(PermissionError):
        eg.resolve_scores({}, source="sqlite", path=forbidden)
    with pytest.raises(PermissionError):
        hermes_emotion._state_path(Ctx({"state_path": str(forbidden)}))
    with pytest.raises(PermissionError):
        hermes_emotion._inject(Ctx({"external_state_path": str(forbidden)}), "s1", "hi")

    # Symlink to core DB is also blocked
    symlink_core = tmp_path / "symlink_core.db"
    try:
        symlink_core.symlink_to(forbidden)
        with pytest.raises(PermissionError):
            store.get_connection(symlink_core)
        with pytest.raises(PermissionError):
            eg.load_state(symlink_core)
    except OSError:
        pass

    if forbidden.exists() and mtime_before is not None:
        assert forbidden.stat().st_mtime_ns == mtime_before


def test_mode_switching_and_snapshot_consistency(tmp_path):
    """Verify snapshot score consistency between SQLite and state.json across mode switches."""
    import __init__ as hermes_emotion
    import engine.emotion_grid as eg

    db_path = tmp_path / "switch_store.db"
    json_path = tmp_path / "switch_state.json"
    store.set_db_path(db_path)
    store.ensure_schema(db_path)

    class Ctx:
        def __init__(self, cfg):
            self.cfg = cfg
            self.hooks = {}
            self.commands = {}
        def get_config(self, k, d=None):
            return self.cfg.get(k, d)
        def register_hook(self, name, fn):
            self.hooks[name] = fn
        def register_command(self, name, handler, *a):
            self.commands[name] = handler

    ctx = Ctx({
        "state_path": str(json_path),
        "engine_db_path": str(db_path),
        "state_source": "internal",
    })
    hermes_emotion.register(ctx)

    # Turn 1 in internal mode
    ctx.hooks["pre_llm_call"](session_id="switch_sess", user_message="你好啊，今天真好！")
    l1_1, l2_1, _ = hermes_emotion._load_db_state(str(db_path))
    json_1 = eg.load_state(json_path)

    # Internal mode authoritative scores match json snapshot
    assert round(l2_1["aff"], 4) == round(json_1["scores"]["aff"], 4)
    assert round(l2_1["sec"], 4) == round(json_1["scores"]["sec"], 4)
    assert round(l2_1["aro"], 4) == round(json_1["scores"]["aro"], 4)
    assert round(l2_1["dom"], 4) == round(json_1["scores"]["dom"], 4)

    # Switch to manual mode
    ctx.cfg["state_source"] = "manual"
    ctx.hooks["pre_llm_call"](session_id="switch_sess", user_message="继续聊天")
    l1_2, l2_2, _ = hermes_emotion._load_db_state(str(db_path))
    json_2 = eg.load_state(json_path)

    # Scores must remain unchanged and consistent in manual mode
    assert round(l2_2["aff"], 4) == round(l2_1["aff"], 4)
    assert round(json_2["scores"]["aff"], 4) == round(json_1["scores"]["aff"], 4)

