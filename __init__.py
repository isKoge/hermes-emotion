"""hermes-emotion — give Hermes a mood that actually changes.

A four-axis score (affection / security / arousal / dominance) is folded into a
16-cell grid. Each cell is one short first-person self-description. The mapping
is deterministic: the same scores always produce the same text, so the injected
block does not churn the prompt cache.

Only when the cell changes (or a score moves far enough, or nothing has been
written for days) does a small model write one new sentence group. That write
happens on a background thread, lands in the state file, and shows up on the
NEXT turn — never mid-turn, never blocking the reply.

Nothing here touches SOUL.md or the system prompt: the block is appended to the
current turn's user message, which is what `pre_llm_call` is for.
"""

from __future__ import annotations

import contextvars
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

try:  # loaded as a package by Hermes
    from . import emotion_grid as G
    from . import emotion_engine as E
except ImportError:  # loaded flat by pytest / a plain python session
    import emotion_grid as G  # type: ignore[no-redef]
    import emotion_engine as E  # type: ignore[no-redef]

logger = logging.getLogger("hermes.plugins.hermes-emotion")

_NAME = "hermes-emotion"
_LOCK = threading.Lock()


# ── config helpers ───────────────────────────────────────────────────
def _cfg(ctx: Any, key: str, default: Any) -> Any:
    getter = getattr(ctx, "get_config", None)
    if callable(getter):
        try:
            value = getter(key, default)
        except Exception:  # host older than config_schema support
            return default
        return default if value is None else value
    return default


def _state_path(ctx: Any) -> Path:
    explicit = str(_cfg(ctx, "state_path", "") or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    try:
        from plugins.plugin_storage import plugin_data_dir  # host-provided
        return plugin_data_dir(_NAME) / "state.json"
    except Exception:
        return Path.home() / ".hermes" / "plugin-data" / _NAME / "state.json"


def _bands(ctx: Any) -> Dict[str, float]:
    bands = dict(G.DEFAULT_BANDS)
    for key in bands:
        bands[key] = float(_cfg(ctx, key, bands[key]))
    return bands


# ── narrator ─────────────────────────────────────────────────────────
def _complete(ctx: Any, prompt: str, timeout_s: float) -> Optional[str]:
    """One host-owned LLM call. Overrides are only sent when configured."""
    llm = getattr(ctx, "llm", None)
    if llm is None:
        logger.warning("[emotion] host exposes no ctx.llm; skipping narrator")
        return None
    provider = str(_cfg(ctx, "narrator_provider", "") or "").strip()
    model = str(_cfg(ctx, "narrator_model", "") or "").strip()
    kwargs: Dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 220,
        "purpose": f"{_NAME}.narrator",
    }
    if provider:
        kwargs["provider"] = provider
    if model:
        kwargs["model"] = model
    try:
        result = llm.complete(**kwargs)
        return getattr(result, "text", None) or (result if isinstance(result, str) else None)
    except Exception as exc:
        if provider or model:
            logger.info("[emotion] override denied/failed (%s); retrying on the active model", exc)
            try:
                result = llm.complete(messages=kwargs["messages"], max_tokens=220,
                                      purpose=kwargs["purpose"])
                return getattr(result, "text", None) or None
            except Exception as exc2:
                logger.warning("[emotion] narrator call failed: %s", exc2)
                return None
        logger.warning("[emotion] narrator call failed: %s", exc)
        return None


def _run_narrator(ctx: Any, path: Path, token: str, prompt: str, timeout_s: float) -> None:
    text = _complete(ctx, prompt, timeout_s)
    ok = G.valid_increment(text)
    with _LOCK:
        state = G.load_state(path)
        narrative = state.get("narrative") or {}
        if narrative.get("pending") != token:
            return  # superseded by a newer request
        narrative["pending"] = ""
        narrative["last_attempt_ts"] = G.now_ts()
        if ok:
            increments = [t for t in (narrative.get("increments") or []) if t]
            increments.append(text.strip())
            narrative["increments"] = increments[-int(_cfg(ctx, "max_increments", 3)):]
            narrative["last_success_ts"] = G.now_ts()
            narrative["last_signature"] = state.get("signature", "")
            narrative["last_scores"] = [
                float(state.get("scores", {}).get(k, 0.0)) for k in ("aff", "sec", "aro", "dom")
            ]
            logger.info("[emotion] narrator wrote a new increment (%d chars)", len(text.strip()))
        else:
            logger.warning("[emotion] narrator output rejected (empty/digits/too long/field names)")
        state["narrative"] = narrative
        G.save_state(path, state)


# ── hook ─────────────────────────────────────────────────────────────
def pre_llm_call(session_id: str = "", user_message: str = "", conversation_history=None,
                 is_first_turn: bool = False, model: str = "", platform: str = "",
                 **kwargs: Any):
    try:
        return _inject(None, session_id or "default", user_message or "")
    except Exception:
        logger.exception("[emotion] injection failed; skipping this turn")
        return None


def register(ctx: Any) -> None:
    """Wire the plugin into the host."""
    if not bool(_cfg(ctx, "enabled", True)):
        logger.info("[emotion] disabled by config; no hooks registered")
        return

    def inject(session_id: str = "", user_message: str = "", **_kw: Any):
        try:
            return _inject(ctx, session_id or "default", user_message or "")
        except Exception:
            logger.exception("[emotion] injection failed; skipping this turn")
            return None

    ctx.register_hook("pre_llm_call", inject)

    def show(*_a: Any, **_kw: Any) -> str:
        state = G.load_state(_state_path(ctx))
        scores = state.get("scores") or {}
        coords = state.get("coords") or ()
        narrative = state.get("narrative") or {}
        lines = [
            f"scores: aff={scores.get('aff', 0):.1f} sec={scores.get('sec', 0):.1f} "
            f"aro={scores.get('aro', 0):.1f} dom={scores.get('dom', 0):.1f}",
            f"grid: {'·'.join(coords) if coords else '(未定)'}  signature={state.get('signature', '')}",
            f"increments: {len(narrative.get('increments') or [])}  pending={bool(narrative.get('pending'))}",
            "",
            G.build_block(state.get("base_text") or "（还没有生成人设段）",
                          narrative.get("increments") or []),
        ]
        return "\n".join(lines)

    def set_scores(raw: str = "") -> str:
        parts = [p for p in str(raw).replace(",", " ").split() if p]
        if len(parts) != 4:
            return "用法：/emotion set <aff> <sec> <aro> <dom>  例如 /emotion set 60 20 25 5"
        try:
            values = [float(p) for p in parts]
        except ValueError:
            return "四个值都要是数字。"
        path = _state_path(ctx)
        state = G.load_state(path)
        state.setdefault("scores", {})
        for key, value in zip(("aff", "sec", "aro", "dom"), values):
            state["scores"][key] = max(-100.0, min(100.0, value))
        state["updated_at"] = G.now_ts()
        state["signature"] = ""  # force the grid to be re-read next turn
        G.save_state(path, state)
        return "已写入：" + " ".join(f"{k}={v:.0f}" for k, v in zip(("aff", "sec", "aro", "dom"), values))

    def command(raw: str = "") -> str:
        text = str(raw or "").strip()
        if text.lower().startswith("set"):
            return set_scores(text[3:])
        return show()

    register_command = getattr(ctx, "register_command", None)
    if callable(register_command):
        try:
            register_command(
                name="emotion",
                handler=command,
                description="查看当前情绪人设；/emotion set <aff> <sec> <aro> <dom> 手动设置分数",
                args_hint="[set aff sec aro dom]",
            )
        except Exception:
            logger.debug("[emotion] register_command failed", exc_info=True)

    logger.info("[emotion] ready (state=%s)", _state_path(ctx))


def _inject(ctx: Any, session_id: str, user_message: str) -> Optional[Dict[str, str]]:
    path = _state_path(ctx)
    source = str(_cfg(ctx, "state_source", "internal") or "internal")
    external = Path(str(_cfg(ctx, "external_state_path", "") or "")).expanduser() \
        if str(_cfg(ctx, "external_state_path", "") or "").strip() else None

    with _LOCK:
        state = G.load_state(path) or G.default_state()
        engine_failed = False
        if source == "internal":
            try:
                raw_decay = _cfg(ctx, "engine_decay", E.DEFAULT_ENGINE_DECAY)
                if raw_decay is None or raw_decay == "":
                    engine_decay = E.DEFAULT_ENGINE_DECAY
                else:
                    try:
                        engine_decay = float(raw_decay)
                    except (TypeError, ValueError):
                        engine_decay = E.DEFAULT_ENGINE_DECAY
                state = E.step(state, user_message, now_ts=G.now_ts(), engine_decay=engine_decay)
            except Exception:
                logger.exception("[emotion] internal engine step failed; keeping state")
                engine_failed = True
        else:
            state = G.resolve_scores(
                state, source=source, path=external,
                decay_per_turn=float(_cfg(ctx, "decay_per_turn", 0.0) or 0.0),
                sqlite_table=str(_cfg(ctx, "sqlite_table", "emotion_engine") or "emotion_engine"),
            )
        scores = [float(state["scores"].get(k, 0.0)) for k in ("aff", "sec", "aro", "dom")]
        previous = tuple(state["coords"]) if state.get("coords") else None
        if previous and len(previous) != 4:
            previous = None
        coords = G.coords_of(scores, _bands(ctx), previous)  # type: ignore[arg-type]
        mods = G.modifiers(
            rifts=float(state.get("rifts") or 0),
            stage=str(state.get("stage") or ""),
            days_since=int(state.get("days_since") or 0),
            atmosphere=str(state.get("atmosphere") or ""),
            rifts_threshold=float(_cfg(ctx, "rifts_threshold", 20) or 20),
        )
        sig = G.signature(coords, mods)

        if sig != state.get("signature") or not state.get("base_text"):
            state["base_text"] = G.base_text(coords, mods)
            state["signature"] = sig
        state["coords"] = list(coords)
        state["updated_at"] = G.now_ts()

        narrative = state.setdefault("narrative", {})
        max_increments = int(_cfg(ctx, "max_increments", 3) or 3)
        token = ""
        prompt = ""

        if engine_failed:
            # Degrade to fixed base block only, skip narrator scheduling
            block = G.build_block(state["base_text"], [])
            trigger = False
        else:
            stale = bool(narrative.get("increments")) and narrative.get("last_signature") not in ("", sig)
            block = G.build_block(state["base_text"], narrative.get("increments") or [],
                                  max_increments, stale=stale)
            trigger = G.should_generate(
                narrative, sig, scores, G.now_ts(),
                axis_delta=float(_cfg(ctx, "axis_delta", 15) or 15),
                cooldown_seconds=float(_cfg(ctx, "cooldown_hours", 6) or 6) * 3600.0,
                refresh_seconds=float(_cfg(ctx, "refresh_days", 7) or 7) * 86400.0,
                retry_seconds=float(_cfg(ctx, "retry_seconds", 60) or 60),
            )
            if trigger:
                token = f"{session_id}-{int(G.now_ts() * 1000)}"
                narrative["pending"] = token
                prompt = G.narrator_prompt(coords, mods, _first_line(user_message))

        G.save_state(path, state)

    if trigger:
        timeout_s = float(_cfg(ctx, "narrator_timeout_s", 20) or 20)
        ctx_run = contextvars.copy_context().run
        threading.Thread(
            target=ctx_run,
            args=(_run_narrator, ctx, path, token, prompt, timeout_s),
            name="hermes-emotion-narrator",
            daemon=True,
        ).start()

    return {"context": block, "target": "user_message"}


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line and not line.startswith("["):
            return line[:120]
    return ""
