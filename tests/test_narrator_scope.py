from __future__ import annotations

from test_emotion_grid import load_plugin, FakeCtx


def test_narrator_call_still_works_without_host_scope_module():
    """Host may not expose agent.secret_scope (tests/older builds); the call must still go through."""
    mod = load_plugin()
    seen = {}

    class FakeLLM:
        def complete(self, **kwargs):
            seen["max_tokens"] = kwargs.get("max_tokens")
            return type("R", (), {"text": "就这样吧。"})()

    ctx = FakeCtx({})
    ctx.llm = FakeLLM()
    mod._plugin_ctx = ctx
    out = mod._complete(ctx, "写两句", 5.0)
    assert out == "就这样吧。"
    assert seen["max_tokens"] == 800
