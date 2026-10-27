"""Tests for hermes-emotion. Run with: python -m pytest tests/ -q"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import emotion_grid as G  # noqa: E402

BANDS = dict(G.DEFAULT_BANDS)


def load_plugin():
    spec = importlib.util.spec_from_file_location(
        "hermes_emotion", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hermes_emotion"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeLLM:
    def __init__(self, text="我心里动了一下，想靠近些。", raises=None):
        self.text = text
        self.raises = raises
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
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


# ── grid ─────────────────────────────────────────────────────────────
def test_all_16_cells_have_text():
    assert len(G.GRID_TEXTS) == 16
    for coords, text in G.GRID_TEXTS.items():
        assert len(coords) == 4 and text.strip()


def test_grid_text_has_no_digits_or_field_names():
    for text in G.GRID_TEXTS.values():
        assert not G._DIGIT_RE.search(text), text
        assert not G._FIELD_RE.search(text), text


def test_hysteresis_holds_inside_band():
    first = G.coords_of([0, 0, 0, 0], BANDS)
    near = G.coords_of([21, 0, 0, 0], BANDS)
    assert near[0] == "近"
    # inside the band, the previous side wins
    back = G.coords_of([2, 0, 0, 0], BANDS, near)
    assert back[0] == "近"
    far = G.coords_of([-25, 0, 0, 0], BANDS, near)
    assert far[0] == "远"


def test_same_scores_same_signature_and_text():
    a = G.coords_of([80, 20, 25, 5], BANDS)
    b = G.coords_of([80, 20, 25, 5], BANDS)
    mods = G.modifiers(rifts=39, stage="STRANGER", days_since=122)
    assert a == b
    assert G.signature(a, mods) == G.signature(b, mods)
    assert G.base_text(a, mods) == G.base_text(b, mods)


def test_modifiers_are_conditional():
    assert G.modifiers(rifts=0, stage="FAMILIAR", days_since=200) == []
    mods = G.modifiers(rifts=39, stage="STRANGER", days_since=122, atmosphere="cool")
    assert len(mods) == 3
    assert G.modifiers(rifts=39, stage="STRANGER", days_since=5) == ["中间有些事我一直没放下。"]


def test_increment_validation():
    assert G.valid_increment("我心里安静下来，也想让他多待一会儿。")
    assert not G.valid_increment("")
    assert not G.valid_increment("好感是 80，我很开心。")
    assert not G.valid_increment("我" * 300 + "。")
    assert not G.valid_increment("没有句号的话")


# ── trigger ──────────────────────────────────────────────────────────
def test_first_time_always_generates():
    narrative = {"increments": [], "last_success_ts": 0, "last_attempt_ts": 0}
    assert G.should_generate(narrative, "sig", [0, 0, 0, 0], now_ts=1000)


def test_cooldown_blocks_then_allows():
    narrative = {"increments": ["x"], "last_success_ts": 1000, "last_attempt_ts": 1000,
                 "last_signature": "sig", "last_scores": [0, 0, 0, 0]}
    # same signature, no score movement: nothing to regenerate even after 7 hours
    assert not G.should_generate(narrative, "sig", [0, 0, 0, 0], now_ts=1000 + 7 * 3600)
    # a cell change inside the cooldown window is still blocked
    assert not G.should_generate(narrative, "other", [0, 0, 0, 0], now_ts=1000 + 3600)
    # ... and allowed once the window has passed
    assert G.should_generate(narrative, "other", [0, 0, 0, 0], now_ts=1000 + 7 * 3600)


def test_axis_delta_respects_configured_threshold():
    narrative = {"increments": ["x"], "last_success_ts": 1000, "last_attempt_ts": 1000,
                 "last_signature": "sig", "last_scores": [50, 0, 0, 0]}
    later = 1000 + 7 * 3600
    assert not G.should_generate(narrative, "sig", [64, 0, 0, 0], now_ts=later, axis_delta=15)
    assert G.should_generate(narrative, "sig", [65, 0, 0, 0], now_ts=later, axis_delta=15)


def test_pending_blocks_duplicate_request():
    narrative = {"increments": [], "last_success_ts": 0, "last_attempt_ts": 0, "pending": "t1"}
    assert not G.should_generate(narrative, "sig", [0, 0, 0, 0], now_ts=10_000)


def test_failure_sets_retry_gap():
    # never succeeded yet: the retry window alone decides, not the success cooldown
    narrative = {"increments": ["x"], "last_success_ts": 0, "last_attempt_ts": 1000,
                 "last_signature": "sig", "last_scores": [0, 0, 0, 0]}
    assert not G.should_generate(narrative, "sig", [0, 0, 0, 0], now_ts=1030, retry_seconds=60)
    assert G.should_generate(narrative, "sig", [0, 0, 0, 0], now_ts=1100, retry_seconds=60)


# ── plugin wiring ────────────────────────────────────────────────────
def test_injection_is_stable_and_digit_free(tmp_state):
    mod = load_plugin()
    ctx = FakeCtx({"state_path": str(tmp_state), "state_source": "manual",
                   "narrator_model": "", "narrator_provider": ""})
    ctx.config["state_path"] = str(tmp_state)
    mod.register(ctx)
    hook = ctx.hooks["pre_llm_call"]

    state = G.load_state(tmp_state) or G.default_state()
    state["scores"] = {"aff": 80.0, "sec": 20.0, "aro": 25.0, "dom": 5.0}
    state["rifts"] = 39
    state["stage"] = "STRANGER"
    state["days_since"] = 122
    state["narrative"] = {"increments": [], "last_success_ts": 9e12,
                          "last_attempt_ts": 9e12, "pending": ""}
    G.save_state(tmp_state, state)

    first = hook(session_id="s1", user_message="在干嘛")
    second = hook(session_id="s1", user_message="在干嘛")
    assert first["context"] == second["context"]
    assert first["target"] == "user_message"
    block = first["context"]
    assert block.startswith(G.BLOCK_HEADER)
    persona = block.split("\n", 1)[1]
    assert not G._DIGIT_RE.search(persona.replace("（以上状态以最后一条变更为准，与当前基准状态保持一致）", ""))


def test_increment_is_appended_not_replaced(tmp_state):
    mod = load_plugin()
    ctx = FakeCtx({"state_path": str(tmp_state), "state_source": "manual"})
    mod.register(ctx)
    hook = ctx.hooks["pre_llm_call"]

    state = G.default_state()
    state["scores"] = {"aff": 80.0, "sec": 20.0, "aro": 25.0, "dom": 5.0}
    state["narrative"] = {"increments": ["旧的一句。"], "last_success_ts": 9e12,
                          "last_attempt_ts": 9e12, "pending": ""}
    G.save_state(tmp_state, state)

    block = hook(session_id="s1", user_message="嗯")["context"]
    assert "旧的一句。" in block
    assert block.index(G.BLOCK_HEADER) < block.index(G.INCREMENT_PREFIX)


def test_stale_increment_is_marked_when_cell_moves(tmp_state):
    mod = load_plugin()
    ctx = FakeCtx({"state_path": str(tmp_state), "state_source": "manual"})
    mod.register(ctx)
    hook = ctx.hooks["pre_llm_call"]

    state = G.default_state()
    state["scores"] = {"aff": 80.0, "sec": 20.0, "aro": 25.0, "dom": 5.0}
    state["narrative"] = {"increments": ["那时候我很想靠近。"], "last_success_ts": 9e12,
                          "last_attempt_ts": 9e12, "pending": "", "last_signature": ""}
    G.save_state(tmp_state, state)

    block = hook(session_id="s1", user_message="嗯")["context"]
    state = G.load_state(tmp_state)
    state["narrative"]["last_signature"] = state["signature"]   # the increment belongs to this cell
    G.save_state(tmp_state, state)
    block = hook(session_id="s1", user_message="嗯")["context"]
    assert G.STALE_NOTE not in block          # same cell as the increment
    state = G.load_state(tmp_state)
    state["scores"] = {"aff": -80.0, "sec": 80.0, "aro": 5.0, "dom": -30.0}
    G.save_state(tmp_state, state)
    block2 = hook(session_id="s1", user_message="嗯")["context"]
    assert G.STALE_NOTE in block2             # cell moved: old text is only history
    assert "那时候我很想靠近。" in block2


def test_narrator_failure_degrades_silently(tmp_state):
    mod = load_plugin()
    ctx = FakeCtx({"state_path": str(tmp_state), "state_source": "manual"},
                  llm=FakeLLM(raises=RuntimeError("boom")))
    mod.register(ctx)
    hook = ctx.hooks["pre_llm_call"]

    block = hook(session_id="s1", user_message="嗯")["context"]
    assert block.startswith(G.BLOCK_HEADER)


def test_narrator_override_denied_falls_back(tmp_state):
    class Denied(FakeLLM):
        def complete(self, **kwargs):
            if "model" in kwargs:
                self.calls.append(kwargs)
                raise PermissionError("override denied")
            return super().complete(**kwargs)

    mod = load_plugin()
    llm = Denied()
    ctx = FakeCtx({"state_path": str(tmp_state), "state_source": "manual",
                   "narrator_model": "gpt-6-luna", "narrator_provider": "openai-codex"},
                  llm=llm)
    mod.register(ctx)
    hook = ctx.hooks["pre_llm_call"]
    hook(session_id="s1", user_message="嗯")
    assert any("model" not in c for c in llm.calls)


def test_external_json_source_is_read(tmp_path, tmp_state):
    mod = load_plugin()
    ext = tmp_path / "scores.json"
    ext.write_text(json.dumps({"aff": -60, "sec": 80, "aro": 5, "dom": -30}), encoding="utf-8")
    ctx = FakeCtx({"state_path": str(tmp_state), "state_source": "json",
                   "external_state_path": str(ext)})
    mod.register(ctx)
    hook = ctx.hooks["pre_llm_call"]
    hook(session_id="s1", user_message="嗯")
    state = G.load_state(tmp_state)
    assert state["coords"] == ["远", "稳", "静", "收"]


def test_sqlite_source_reads_latest_row(tmp_path, tmp_state):
    import sqlite3
    db = tmp_path / "ext.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE emotion_engine (aff REAL, sec REAL, aro REAL, dom REAL)")
    conn.execute("INSERT INTO emotion_engine VALUES (10, 10, 10, 10)")
    conn.execute("INSERT INTO emotion_engine VALUES (-70, 75, 40, 30)")
    conn.commit()
    conn.close()

    mod = load_plugin()
    ctx = FakeCtx({"state_path": str(tmp_state), "state_source": "sqlite",
                   "external_state_path": str(db)})
    mod.register(ctx)
    ctx.hooks["pre_llm_call"](session_id="s1", user_message="嗯")
    state = G.load_state(tmp_state)
    assert state["scores"]["aff"] == -70.0
    assert state["coords"] == ["远", "稳", "热", "进"]


def test_missing_sqlite_file_keeps_stored_scores(tmp_path, tmp_state):
    mod = load_plugin()
    ctx = FakeCtx({"state_path": str(tmp_state), "state_source": "sqlite",
                   "external_state_path": str(tmp_path / "nope.db")})
    mod.register(ctx)
    state = G.default_state()
    state["scores"] = {"aff": 50.0, "sec": 50.0, "aro": 50.0, "dom": 50.0}
    G.save_state(tmp_state, state)
    ctx.hooks["pre_llm_call"](session_id="s1", user_message="嗯")
    assert G.load_state(tmp_state)["scores"]["aff"] == 50.0


def test_emotion_command_sets_scores(tmp_state):
    mod = load_plugin()
    ctx = FakeCtx({"state_path": str(tmp_state), "state_source": "manual"})
    mod.register(ctx)
    out = ctx.commands["emotion"]("set 85 15 40 -5")
    assert "85" in out
    assert G.load_state(tmp_state)["scores"]["aff"] == 85.0
