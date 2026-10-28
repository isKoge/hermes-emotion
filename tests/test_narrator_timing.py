"""Timing contract: narrator runs async in the background and lands on the NEXT turn.

These lock the two halves the user cares about:
  * generation must never block the turn it was triggered from;
  * a landed increment must show up in the next turn's injected block — exactly once.
"""

from __future__ import annotations

import json

import pytest

from test_emotion_grid import FakeCtx, FakeLLM, load_plugin


@pytest.fixture()
def tmp_state(tmp_path):
    return tmp_path / "state.json"


def _turns(mod, ctx, tmp_state, messages):
    hook = ctx.hooks["pre_llm_call"]
    out = []
    for msg in messages:
        out.append(hook(session_id="s-timing", user_message=msg))
    return out


def test_increment_lands_async_and_shows_up_next_turn(tmp_state):
    mod = load_plugin()
    calls = {"n": 0}

    class CountingLLM(FakeLLM):
        def complete(self, **kwargs):
            calls["n"] += 1
            return super().complete(**kwargs)

    ctx = FakeCtx({"state_path": str(tmp_state), "state_source": "internal"}, llm=CountingLLM())
    mod.register(ctx)

    # turn 1 triggers generation but its own block is already rendered: no blocking,
    # no half-written sentence leaking into the turn that caused it
    t1 = _turns(mod, ctx, tmp_state, ["你真棒，太厉害了！"])[0]
    assert "[我心里的状态]" in t1["context"]
    assert "人设现在变更为：" not in t1["context"]

    mod.wait_for_narrator_threads(timeout_s=5.0)
    assert calls["n"] >= 1, "narrator was never invoked"

    state = json.loads(tmp_state.read_text())
    assert state.get("coords"), "coords missing from state"

    # turn 2 consumes what landed in the background — exactly once
    t2 = _turns(mod, ctx, tmp_state, ["在呢"])[0]
    assert t2["context"].count("[我心里的状态]") == 1
    assert "人设现在变更为：" in t2["context"], "landed increment did not reach the next turn"
    assert t2["context"].count("我心里动了一下，想靠近些。") == 1


def test_empty_narrator_output_is_not_injected(tmp_state):
    mod = load_plugin()
    ctx = FakeCtx(
        {"state_path": str(tmp_state), "state_source": "internal"},
        llm=FakeLLM(text=""),
    )
    mod.register(ctx)

    _turns(mod, ctx, tmp_state, ["你真棒，太厉害了！"])
    mod.wait_for_narrator_threads(timeout_s=5.0)

    t2 = _turns(mod, ctx, tmp_state, ["在呢"])[0]
    assert "[我心里的状态]" in t2["context"]
    assert "人设现在变更为：" not in t2["context"], "empty model output leaked into the block"
