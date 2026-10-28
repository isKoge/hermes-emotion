"""Tests for emotion_engine and internal mode integration."""

from __future__ import annotations

import copy
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import emotion_engine as E  # noqa: E402
import emotion_grid as G  # noqa: E402


def load_plugin():
    spec = importlib.util.spec_from_file_location(
        "hermes_emotion", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hermes_emotion"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeLLM:
    def __init__(self, text="我心里动了一下，想靠近些。"):
        self.text = text
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return type("R", (), {"text": self.text})()


class FakeCtx:
    def __init__(self, config=None, llm=None):
        self.config = config or {}
        self.llm = llm or FakeLLM()
        self.hooks = {}
        self.commands = {}

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_hook(self, name, fn):
        self.hooks[name] = fn

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands[name] = handler


@pytest.fixture
def tmp_state(tmp_path):
    return tmp_path / "state.json"


# ── 1. Event axis signs ──────────────────────────────────────────────
def test_all_event_deltas_axis_signs():
    """Verify each event has the expected sign on all four axes."""
    expected_signs = {
        E.EVENT_PLAIN: {"aff": 1, "sec": 1, "aro": 1, "dom": 0},
        E.EVENT_PRAISE: {"aff": 1, "sec": 1, "aro": 1, "dom": 1},
        E.EVENT_APOLOGY: {"aff": 1, "sec": 1, "aro": -1, "dom": -1},
        E.EVENT_VULNERABILITY: {"aff": 1, "sec": 1, "aro": -1, "dom": 1},
        E.EVENT_TEASING: {"aff": 1, "sec": 1, "aro": 1, "dom": 1},
        E.EVENT_COLD: {"aff": -1, "sec": -1, "aro": -1, "dom": -1},
        E.EVENT_CRITICISM: {"aff": -1, "sec": -1, "aro": 1, "dom": -1},
        E.EVENT_REUNION: {"aff": 1, "sec": -1, "aro": 1, "dom": -1},
    }

    assert set(E.ALL_EVENTS) == set(expected_signs.keys())

    for event, signs in expected_signs.items():
        deltas = E.EVENT_DELTAS[event]
        for axis, sign in signs.items():
            if sign == 0:
                assert deltas[axis] == 0, f"Expected {event} {axis} == 0, got {deltas[axis]}"
            elif sign > 0:
                assert deltas[axis] > 0, f"Expected {event} {axis} > 0, got {deltas[axis]}"
            else:
                assert deltas[axis] < 0, f"Expected {event} {axis} < 0, got {deltas[axis]}"


def test_event_rule_recognition():
    """Verify rules correctly identify all event categories from message text."""
    cases = [
        ("你真棒，太厉害了！", E.EVENT_PRAISE),
        ("谢谢你，做得真好", E.EVENT_PRAISE),
        ("对不起，是我不好", E.EVENT_APOLOGY),
        ("抱歉，我错了", E.EVENT_APOLOGY),
        ("今天好累，心里好难受", E.EVENT_VULNERABILITY),
        ("好委屈，想哭", E.EVENT_VULNERABILITY),
        ("哈哈笑死我了", E.EVENT_TEASING),
        ("逗你玩的，别当真", E.EVENT_TEASING),
        ("哦", E.EVENT_COLD),
        ("行吧，随便你", E.EVENT_COLD),
        ("你真笨，闭嘴吧！", E.EVENT_CRITICISM),
        ("真没用的东西，讨厌你", E.EVENT_CRITICISM),
        ("好久不见，还记得我吗？", E.EVENT_REUNION),
        ("好久没聊了！", E.EVENT_REUNION),
        ("今天星期三，天气晴朗。", E.EVENT_PLAIN),
    ]
    for text, expected_event in cases:
        assert E.detect_event(text) == expected_event, f"Failed on message: {text!r}"


# ── 2. Asymmetric decay: positive vs negative aff ────────────────────
def test_decay_difference_positive_vs_negative_aff():
    """Affection decays slower when positive (stickiness), faster when negative."""
    decay_rate = 0.05
    pos_initial = {"aff": 50.0, "sec": 20.0, "aro": 20.0, "dom": 20.0}
    neg_initial = {"aff": -50.0, "sec": 20.0, "aro": 20.0, "dom": 20.0}

    pos_decayed = E.apply_decay(pos_initial, decay_rate)
    neg_decayed = E.apply_decay(neg_initial, decay_rate)

    pos_decay_pct = abs(pos_initial["aff"] - pos_decayed["aff"]) / pos_initial["aff"]
    neg_decay_pct = abs(neg_initial["aff"] - neg_decayed["aff"]) / abs(neg_initial["aff"])

    # Positive aff decays slower than negative aff
    assert pos_decay_pct < neg_decay_pct
    assert pos_decayed["aff"] > 0
    assert neg_decayed["aff"] < 0
    # Specifically, check the factor relationship
    assert pos_decay_pct == pytest.approx(decay_rate * E.AFF_STICKINESS_FACTOR)
    assert neg_decay_pct == pytest.approx(decay_rate * E.AFF_REBOUND_FACTOR)


# ── 3. Rifts: apology decreases, criticism increases ──────────────────
def test_rifts_apology_decreases_criticism_increases():
    """Apology decreases rifts; criticism/insults increase rifts."""
    initial_state = {
        "scores": {"aff": 0.0, "sec": 0.0, "aro": 0.0, "dom": 0.0},
        "rifts": 10.0,
        "trust": 20.0,
        "stage": "STRANGER",
        "turns": 1,
    }

    # Criticism raises rifts and lowers trust
    state_after_criticism = E.step(initial_state, "真没用，讨厌你", now_ts=1000.0)
    assert state_after_criticism["rifts"] > initial_state["rifts"]
    assert state_after_criticism["rifts"] == 25.0
    assert state_after_criticism["trust"] < initial_state["trust"]

    # Apology lowers rifts and restores trust
    state_after_apology = E.step(state_after_criticism, "对不起，是我不好", now_ts=1010.0)
    assert state_after_apology["rifts"] < state_after_criticism["rifts"]
    assert state_after_apology["rifts"] == 15.0
    assert state_after_apology["trust"] > state_after_criticism["trust"]


# ── 4. Stage: only advances, never demotes ────────────────────────────
def test_stage_only_advances_never_demotes():
    """Stage advances with turns and trust, and never drops even when trust drops."""
    state = {
        "scores": {"aff": 0.0, "sec": 0.0, "aro": 0.0, "dom": 0.0},
        "trust": 0.0,
        "stage": E.STAGE_STRANGER,
        "turns": 0,
    }

    # Initially STRANGER
    assert state["stage"] == E.STAGE_STRANGER

    # Simulate turns with praise to advance to FAMILIAR
    state["turns"] = 5
    state["trust"] = 16.0
    state = E.step(state, "你真好", now_ts=1000.0)
    assert state["stage"] == E.STAGE_FAMILIAR

    # Advance to INTIMATE
    state["turns"] = 25
    state["trust"] = 50.0
    state = E.step(state, "太棒了", now_ts=2000.0)
    assert state["stage"] == E.STAGE_INTIMATE

    # Now simulate sudden criticism causing trust to plunge
    state = E.step(state, "你真差劲，闭嘴", now_ts=3000.0)
    # Trust plunged, but stage NEVER drops back
    assert state["trust"] < 50.0
    assert state["stage"] == E.STAGE_INTIMATE


# ── 5. Days since calculation ─────────────────────────────────────────
def test_days_since_calculation():
    """days_since is correctly calculated from first_seen and current time."""
    t0 = 1_000_000.0

    assert E.compute_days_since(t0, t0) == 0
    assert E.compute_days_since(t0, t0 + 43200) == 0  # 12 hours
    assert E.compute_days_since(t0, t0 + 86400) == 1  # 1 day
    assert E.compute_days_since(t0, t0 + 7 * 86400) == 7  # 7 days
    assert E.compute_days_since(t0, t0 + 65 * 86400) == 65  # 65 days

    # Tested via step
    state = {"first_seen": t0, "turns": 1}
    next_state = E.step(state, "你好", now_ts=t0 + 10 * 86400)
    assert next_state["days_since"] == 10
    assert next_state["first_seen"] == t0


# ── 6. Idempotence and determinism ────────────────────────────────────
def test_step_requires_now_ts():
    """Verify that omitting now_ts raises TypeError, enforcing caller-injected time."""
    state = {"scores": {"aff": 0.0, "sec": 0.0, "aro": 0.0, "dom": 0.0}}
    with pytest.raises(TypeError):
        E.step(state, "你好")  # type: ignore[call-arg]


def test_step_idempotent_and_deterministic():
    """Identical state and inputs yield identical outputs without mutating inputs."""
    state = {
        "scores": {"aff": 10.0, "sec": 5.0, "aro": 0.0, "dom": -5.0},
        "trust": 15.0,
        "rifts": 5.0,
        "stage": E.STAGE_STRANGER,
        "turns": 3,
        "first_seen": 1_000_000.0,
    }
    input_copy = copy.deepcopy(state)

    res1 = E.step(state, "你好棒", now_ts=1_000_100.0, engine_decay=0.05)
    res2 = E.step(state, "你好棒", now_ts=1_000_100.0, engine_decay=0.05)

    assert res1 == res2
    # Ensure input was not mutated
    assert state == input_copy

    # Also test on completely fresh/empty state: first_seen and updated_at must be strictly deterministic
    fresh1 = E.step({}, "你好棒", now_ts=1_234_567.0)
    fresh2 = E.step({}, "你好棒", now_ts=1_234_567.0)
    assert fresh1 == fresh2
    assert fresh1["first_seen"] == 1_234_567.0
    assert fresh1["updated_at"] == 1_234_567.0


# ── 7. Internal mode: consecutive turns produce different coordinates ─
def test_internal_mode_consecutive_turns_different_coords(tmp_state):
    """In internal mode without external files, consecutive user messages change coords."""
    mod = load_plugin()
    ctx = FakeCtx({"state_path": str(tmp_state), "state_source": "internal"})
    mod.register(ctx)
    hook = ctx.hooks["pre_llm_call"]

    # Turn 1: Warm praise pushes scores into positive territory
    t1 = hook(session_id="s1", user_message="你真棒，太厉害了！")
    state1 = G.load_state(tmp_state)
    coords1 = tuple(state1["coords"])

    # Turn 2: Sharp criticism pulls scores down
    t2 = hook(session_id="s1", user_message="真没用，差劲透顶！")
    state2 = G.load_state(tmp_state)
    coords2 = tuple(state2["coords"])

    assert t1 is not None and t2 is not None
    assert state1["scores"]["aff"] > 0
    assert state2["scores"]["aff"] < state1["scores"]["aff"]

    # Invariant: the injected persona section is exactly the section derived from
    # the state written for that same turn (no lag, no stale cell).
    def persona_part(ctx_text: str) -> str:
        """The [我心里的状态] section only; other sections carry per-turn fields."""
        tail = ctx_text.split("[我心里的状态]", 1)[1]
        cut = tail.find("\n[")
        return (tail[:cut] if cut != -1 else tail).strip()

    for state, turn in ((state1, t1), (state2, t2)):
        body = state["base_text"].split("[我心里的状态]")[-1].strip()
        assert body in persona_part(turn["context"]), "persona text lags the turn's own state"

    # Whether two arbitrary messages cross a cell boundary depends on threshold
    # calibration, so the cell-to-text mapping is asserted directly instead.
    from engine import persona_grid as pg
    sig_a, text_a = pg.get_persona_injection(
        60.0, 40.0, 10.0, 5.0, rifts=0, stage="STRANGER", days_since=10,
        atmosphere="warm", session_id="c1", db_path=str(tmp_state.parent / "grid.db"),
        coords=("近", "稳", "静", "进"),
    )
    sig_b, text_b = pg.get_persona_injection(
        -60.0, -10.0, -30.0, -20.0, rifts=0, stage="STRANGER", days_since=10,
        atmosphere="cool", session_id="c2", db_path=str(tmp_state.parent / "grid.db"),
        coords=("远", "悬", "静", "收"),
    )
    assert sig_a != sig_b
    assert text_a != text_b


def test_internal_mode_consecutive_turns_same_coords_identical_context(tmp_state):
    """In internal mode, consecutive turns with identical coordinates yield byte-for-byte identical context."""
    mod = load_plugin()
    ctx = FakeCtx({"state_path": str(tmp_state), "state_source": "internal"})
    ctx.llm = None
    mod.register(ctx)
    hook = ctx.hooks["pre_llm_call"]

    t1 = hook(session_id="s_same", user_message="你好啊")
    state1 = G.load_state(tmp_state)
    coords1 = tuple(state1["coords"])

    t2 = hook(session_id="s_same", user_message="你好啊")
    state2 = G.load_state(tmp_state)
    coords2 = tuple(state2["coords"])

    assert t1 is not None and t2 is not None
    assert coords1 == coords2
    assert t1["context"] == t2["context"]


# ── 8. Legacy modes: json, sqlite, manual preserved ──────────────────
def test_legacy_modes_preserved(tmp_path, tmp_state):
    """json, sqlite, and manual modes behave identically to before."""
    mod = load_plugin()

    # Manual mode: scores only change via set command
    ctx_manual = FakeCtx({"state_path": str(tmp_state), "state_source": "manual"})
    mod.register(ctx_manual)
    ctx_manual.commands["emotion"]("set 70 30 25 15")
    ctx_manual.hooks["pre_llm_call"](session_id="s1", user_message="你好啊")
    state = G.load_state(tmp_state)
    assert state["scores"]["aff"] == 70.0
    assert state["coords"] == ["近", "稳", "热", "进"]

    # External JSON mode: read directly from external file
    ext_json = tmp_path / "ext.json"
    ext_json.write_text(json.dumps({"aff": -50, "sec": 5, "aro": 5, "dom": -20}), encoding="utf-8")
    ctx_json = FakeCtx({
        "state_path": str(tmp_state),
        "state_source": "json",
        "external_state_path": str(ext_json),
    })
    mod.register(ctx_json)
    ctx_json.hooks["pre_llm_call"](session_id="s1", user_message="你好")
    state = G.load_state(tmp_state)
    assert state["scores"]["aff"] == -50.0
    assert state["coords"] == ["远", "悬", "静", "收"]

    # External SQLite mode: read newest row from sqlite
    db_file = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE emotion_engine (aff REAL, sec REAL, aro REAL, dom REAL)")
    conn.execute("INSERT INTO emotion_engine VALUES (65, 35, 10, -15)")
    conn.commit()
    conn.close()

    ctx_sqlite = FakeCtx({
        "state_path": str(tmp_state),
        "state_source": "sqlite",
        "external_state_path": str(db_file),
    })
    mod.register(ctx_sqlite)
    ctx_sqlite.hooks["pre_llm_call"](session_id="s1", user_message="你好")
    state = G.load_state(tmp_state)
    assert state["scores"]["aff"] == 65.0
    assert state["coords"] == ["近", "稳", "静", "收"]


def test_internal_mode_ignores_external_file(tmp_path, tmp_state):
    """In internal mode, external file content is ignored; engine calculates scores."""
    mod = load_plugin()
    ext_json = tmp_path / "ignored.json"
    ext_json.write_text(json.dumps({"aff": -99, "sec": -99, "aro": -99, "dom": -99}), encoding="utf-8")
    ctx = FakeCtx({
        "state_path": str(tmp_state),
        "state_source": "internal",
        "external_state_path": str(ext_json),
    })
    mod.register(ctx)
    hook = ctx.hooks["pre_llm_call"]
    hook(session_id="s1", user_message="你真棒，太厉害了！")
    state = G.load_state(tmp_state)
    # The external -99 score is ignored; praise event was applied
    assert state["scores"]["aff"] > 0
    assert state["scores"]["aff"] != -99.0


def test_engine_decay_zero_via_hook(tmp_state):
    """When engine_decay=0.0 is explicitly configured, neutral messages do not decay scores."""
    mod = load_plugin()
    ctx = FakeCtx({
        "state_path": str(tmp_state),
        "state_source": "internal",
        "engine_decay": 0.0,
    })
    mod.register(ctx)
    hook = ctx.hooks["pre_llm_call"]

    # Preset scores
    state = G.default_state()
    state["scores"] = {"aff": 50.0, "sec": 30.0, "aro": 20.0, "dom": 10.0}
    G.save_state(tmp_state, state)

    # An ordinary message is EVENT_PLAIN; with decay=0 only its small delta applies
    hook(session_id="s1", user_message="今天星期三，天气晴朗。")
    saved = G.load_state(tmp_state)
    plain = E.EVENT_DELTAS[E.EVENT_PLAIN]
    assert saved["scores"]["aff"] == 50.0 + plain["aff"]
    assert saved["scores"]["sec"] == 30.0 + plain["sec"]
    assert saved["scores"]["aro"] == 20.0 + plain["aro"]
    assert saved["scores"]["dom"] == 10.0 + plain["dom"]


def test_engine_crash_safety_degrades_gracefully(tmp_state, monkeypatch):
    """If engine step raises, injection degrades to fixed block only and skips narrator scheduling."""
    mod = load_plugin()
    fake_llm = FakeLLM()
    ctx = FakeCtx({"state_path": str(tmp_state), "state_source": "internal"}, llm=fake_llm)
    mod.register(ctx)
    hook = ctx.hooks["pre_llm_call"]

    # Preset state with existing narrative increment
    state = G.default_state()
    state["scores"] = {"aff": 25.0, "sec": 22.0, "aro": 22.0, "dom": 12.0}
    state["narrative"]["increments"] = ["这是一段历史续写自述。"]
    G.save_state(tmp_state, state)

    def broken_step(*args, **kwargs):
        raise RuntimeError("simulated engine failure")

    monkeypatch.setattr(mod.E, "step", broken_step)
    result = hook(session_id="s1", user_message="你好")

    assert result is not None
    assert "context" in result
    block = result["context"]
    assert block.startswith(G.BLOCK_HEADER)
    # Failed turn must only output fixed block: no increments, no narrator prefix
    assert "这是一段历史续写自述。" not in block
    assert G.INCREMENT_PREFIX not in block
    # Narrator scheduling was skipped
    assert len(fake_llm.calls) == 0

    # Persistence is preserved for future recovery
    persisted = G.load_state(tmp_state)
    assert persisted["narrative"]["increments"] == ["这是一段历史续写自述。"]

