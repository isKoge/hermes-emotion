"""Tests for engine parity, injection ordering, switch toggling, and crash degradation."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
import pytest

from engine import store, adult_mode, focus_stack, weather_sense
from engine.store import set_db_path, ensure_schema
import __init__ as hermes_emotion


class FakeCtx:
    def __init__(self, config=None, llm=None):
        self.config = config or {}
        self.llm = llm
        self.hooks = {}
        self.commands = {}

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_hook(self, name, fn):
        self.hooks[name] = fn

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands[name] = handler


@pytest.fixture
def isolated_db(tmp_path):
    db_file = tmp_path / "isolated_test.db"
    json_file = tmp_path / "isolated_test.json"
    set_db_path(db_file)
    ensure_schema(db_file)
    adult_mode.reset_adult_mode()
    yield {"db": str(db_file), "json": str(json_file)}
    set_db_path(None)


def test_injection_order_matches_spec(isolated_db):
    """Verify sections are assembled in the exact order:
    1. 时间/相识/重聚
    2. [我心里的状态]
    3. 焦点
    4. 天气
    5. 成人状态
    """
    ctx = FakeCtx({
        "state_path": isolated_db["json"],
        "engine_db_path": isolated_db["db"],
        "state_source": "internal",
        "adult_enabled": True,
        "weather_enabled": True,
        "focus_enabled": True,
        "holidays_enabled": True,
    })
    hermes_emotion.register(ctx)

    # First turn sets up intimacy / adult flirting
    fixed_now = datetime(2026, 10, 27, 14, 0, 0)
    hook = ctx.hooks["pre_llm_call"]

    # Pre-populate focus and adult flirting
    adult_state = adult_mode.get_adult_state()
    adult_state.state = "FLIRTING"
    adult_state.intensity_budget = 25.0

    focus_conn = store.get_connection(isolated_db["db"])
    focus_conn.execute("""
        INSERT INTO focus_stack (session_id, platform, title, status, priority, created_at, updated_at)
        VALUES ('sess_order', 'unknown', '实现数据迁移模块', 'active', 1, 1000, 1000)
    """)
    focus_conn.commit()
    focus_conn.close()

    result = hook(
        session_id="sess_order",
        user_message="广州明天天气怎么样",
        now=fixed_now,
    )

    assert result is not None
    assert "context" in result
    assert result["target"] == "user_message"
    context = result["context"]

    # Check that all 5 parts appear in the exact required relative order
    idx_time = context.index("时间: 工作日 下午 · 秋天")
    idx_persona = context.index("[我心里的状态]")
    idx_focus = context.index("[当前焦点]")
    idx_weather = context.index("天气: 广州 明天")
    idx_adult = context.index("模式: 暧昧 | 预算: 23/60")

    assert idx_time < idx_persona, "时间 must precede [我心里的状态]"
    assert idx_persona < idx_focus, "[我心里的状态] must precede 焦点"
    assert idx_focus < idx_weather, "焦点 must precede 天气"
    assert idx_weather < idx_adult, "天气 must precede 成人状态"


def test_adult_switch_disabled(isolated_db):
    """When adult_enabled=False, adult state machine is not triggered and not injected."""
    ctx = FakeCtx({
        "state_path": isolated_db["json"],
        "engine_db_path": isolated_db["db"],
        "state_source": "internal",
        "adult_enabled": False,
    })
    hermes_emotion.register(ctx)

    # User sends explicit flirting
    res = ctx.hooks["pre_llm_call"](
        session_id="s_no_adult",
        user_message="想要抱抱你亲一下",
    )
    assert res is not None
    assert "模式:" not in res["context"]
    assert adult_mode.get_adult_state().state == "NORMAL"


def test_weather_switch_disabled(isolated_db):
    """When weather_enabled=False, weather queries do not trigger network and are not injected."""
    ctx = FakeCtx({
        "state_path": isolated_db["json"],
        "engine_db_path": isolated_db["db"],
        "state_source": "internal",
        "weather_enabled": False,
    })
    hermes_emotion.register(ctx)

    res = ctx.hooks["pre_llm_call"](
        session_id="s_no_weather",
        user_message="广州明天天气怎么样",
    )
    assert res is not None
    assert "天气: 广州" not in res["context"]


def test_focus_switch_disabled(isolated_db):
    """When focus_enabled=False, focus stack is not tracked or injected."""
    focus_conn = store.get_connection(isolated_db["db"])
    focus_conn.execute("""
        INSERT INTO focus_stack (session_id, platform, title, status, priority, created_at, updated_at)
        VALUES ('s_no_focus', 'unknown', '任务A', 'active', 1, 1000, 1000)
    """)
    focus_conn.commit()
    focus_conn.close()

    ctx = FakeCtx({
        "state_path": isolated_db["json"],
        "engine_db_path": isolated_db["db"],
        "state_source": "internal",
        "focus_enabled": False,
    })
    hermes_emotion.register(ctx)

    res = ctx.hooks["pre_llm_call"](
        session_id="s_no_focus",
        user_message="做点别的事情",
    )
    assert res is not None
    assert "[当前焦点]" not in res["context"]


def test_holidays_switch_disabled(isolated_db):
    """When holidays_enabled=False, special dates are not computed or injected."""
    ctx_off = FakeCtx({
        "state_path": isolated_db["json"],
        "engine_db_path": isolated_db["db"],
        "state_source": "internal",
        "holidays_enabled": False,
    })
    hermes_emotion.register(ctx_off)

    new_year_day = datetime(2026, 1, 1, 10, 0, 0)
    res_off = ctx_off.hooks["pre_llm_call"](session_id="s_hol", user_message="早", now=new_year_day)
    assert "特殊日: 元旦" not in res_off["context"]

    # When enabled, New Year holiday is detected
    ctx_on = FakeCtx({
        "state_path": isolated_db["json"],
        "engine_db_path": isolated_db["db"],
        "state_source": "internal",
        "holidays_enabled": True,
    })
    hermes_emotion.register(ctx_on)
    res_on = ctx_on.hooks["pre_llm_call"](session_id="s_hol2", user_message="早", now=new_year_day)
    assert "特殊日: 元旦" in res_on["context"]


def test_engine_crash_safety_robustness(isolated_db, monkeypatch):
    """When an internal engine component throws an unexpected exception,
    the plugin gracefully degrades to the fallback persona block without crashing.
    """
    ctx = FakeCtx({
        "state_path": isolated_db["json"],
        "engine_db_path": isolated_db["db"],
        "state_source": "internal",
    })
    hermes_emotion.register(ctx)

    def crash_interpret(*a, **k):
        raise ValueError("simulated unexpected engine failure")

    monkeypatch.setattr(hermes_emotion.EE, "interpret_event", crash_interpret)

    res = ctx.hooks["pre_llm_call"](session_id="s_crash", user_message="你好")
    assert res is not None
    assert "context" in res
    assert res["target"] == "user_message"
    assert res["context"].startswith("[我心里的状态]")
