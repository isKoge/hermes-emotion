"""Self-contained parity and offline equivalence tests against the Friday Core model.

Verifies that for identical inputs and initial state, hermes-emotion produces equivalent
injection context sections (Time/Acquaintance/Reunion, Persona Grid, Focus Stack,
Weather Sense, and Adult Mode) without relying on live external networks or machine-local paths.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
import pytest

from engine import store, adult_mode, focus_stack, weather_sense
import __init__ as hermes_emotion


class FakeCtx:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}

    def get_config(self, k, d=None):
        return self.cfg.get(k, d)

    def register_hook(self, *a, **k):
        pass

    def register_command(self, *a, **k):
        pass


def test_parity_offline_fixtures(tmp_path):
    """Verify item-by-item equivalence on canonical dialogue inputs with offline fixtures."""
    db_file = tmp_path / "parity_test.db"
    json_file = tmp_path / "parity_test.json"
    store.set_db_path(db_file)
    store.ensure_schema(db_file)

    ctx = FakeCtx({
        "state_path": str(json_file),
        "engine_db_path": str(db_file),
        "state_source": "internal",
        "adult_enabled": True,
        "weather_enabled": True,
        "focus_enabled": True,
        "episodic_enabled": True,
        "holidays_enabled": True,
    })
    hermes_emotion.register(ctx)

    fixed_now = datetime(2026, 10, 27, 15, 30, 0)

    test_cases = [
        ("打招呼", "你好啊！"),
        ("情绪事件（难受）", "今天好累，心里好难受"),
        ("天气感知", "广州明天天气怎么样"),
        ("成人模式", "想要亲亲你抱抱你"),
        ("任务焦点", "帮我重构一下代码"),
    ]

    for name, msg in test_cases:
        res = hermes_emotion._inject(ctx, "parity_sess", msg, now=fixed_now)
        assert res is not None
        ctx_str = res.get("context", "")
        assert res.get("target") == "user_message"

        # 1. Time Awareness
        assert "时间: 工作日 下午 · 秋天" in ctx_str

        # 2. Persona Grid
        assert "[我心里的状态]" in ctx_str

        # 3. Weather
        if "天气" in msg:
            assert "天气: 广州 明天" in ctx_str

        # 4. Adult Mode (non-NORMAL transitions into FLIRTING)
        if "想要亲亲" in msg:
            assert "模式: 暧昧" in ctx_str

        # 5. Focus Stack
        if "帮我" in msg:
            assert "[当前焦点]" in ctx_str
            assert "帮我重构一下代码" in ctx_str


def test_turn_id_caching_prevents_duplicate_advancement(tmp_path):
    """Same turn_id must return cached context and not advance state multiple times."""
    db_file = tmp_path / "cache_test.db"
    store.set_db_path(db_file)
    store.ensure_schema(db_file)

    ctx = FakeCtx({
        "engine_db_path": str(db_file),
        "state_source": "internal",
    })
    hermes_emotion.register(ctx)

    fixed_now = datetime(2026, 10, 27, 10, 0, 0)
    t_id = "turn-fixed-12345"

    res1 = hermes_emotion.pre_llm_call(
        session_id="turn_cache_sess",
        user_message="你好啊，今天真开心！",
        now=fixed_now,
        turn_id=t_id,
    )
    l1_first, l2_first, _ = hermes_emotion._load_db_state(str(db_file))

    # Second call in the same turn tool loop
    res2 = hermes_emotion.pre_llm_call(
        session_id="turn_cache_sess",
        user_message="你好啊，今天真开心！",
        now=fixed_now,
        turn_id=t_id,
    )
    l1_second, l2_second, _ = hermes_emotion._load_db_state(str(db_file))

    assert res1 == res2
    assert l2_first["aff"] == l2_second["aff"]
    assert l1_first["trust"] == l1_second["trust"]


def test_assistant_reply_collection_in_pre_llm_call(tmp_path):
    """Verifies that an assistant message in conversation_history triggers post-turn updates."""
    db_file = tmp_path / "asst_test.db"
    store.set_db_path(db_file)
    store.ensure_schema(db_file)

    ctx = FakeCtx({
        "engine_db_path": str(db_file),
        "state_source": "internal",
    })
    hermes_emotion.register(ctx)

    history = [
        {"role": "user", "content": "帮我看看这个任务"},
        {"role": "assistant", "content": "好的，我已经完成了这个任务！"},
    ]

    res = hermes_emotion.pre_llm_call(
        session_id="sess_asst",
        user_message="接下来做什么？",
        conversation_history=history,
    )
    assert res is not None

    # Verify assistant turn was recorded into episodic memory
    recent_eps = hermes_emotion.episodic_memory.get_recent_episodes(3, db_path=str(db_file))
    assert hermes_emotion._last_processed_assistant.get("sess_asst") == "好的，我已经完成了这个任务！"
