"""hermes-emotion — Complete emotion model plugin for Hermes.

Encapsulates 4D emotion engine, relationship state, persona grid, time awareness,
weather sensing, focus stack, episodic memory, and adult mode into a self-contained
pure plugin.

All state is persisted in plugin_data_dir("hermes-emotion")/state.db, completely
isolated from host core state.db.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
import time
from datetime import datetime
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from . import emotion_grid as G
    from . import emotion_engine as E
    from .engine import (
        store,
        schema,
        emotion_engine as EE,
        relationship_engine,
        time_awareness,
        weather_sense,
        focus_stack,
        episodic_memory,
        adult_mode,
        emotional_emergence,
        injection_policy,
        personality,
        persona_grid,
        migrate,
    )
except (ImportError, ValueError):
    import emotion_grid as G  # type: ignore[no-redef]
    import emotion_engine as E  # type: ignore[no-redef]
    from engine import (  # type: ignore[no-redef]
        store,
        schema,
        emotion_engine as EE,
        relationship_engine,
        time_awareness,
        weather_sense,
        focus_stack,
        episodic_memory,
        adult_mode,
        emotional_emergence,
        injection_policy,
        personality,
        persona_grid,
        migrate,
    )

logger = logging.getLogger("hermes.plugins.hermes-emotion")

_NAME = "hermes-emotion"
_LOCK = threading.Lock()
_plugin_ctx: Any = None


# ── config helpers ───────────────────────────────────────────────────
def _cfg(ctx: Any, key: str, default: Any) -> Any:
    c = ctx or _plugin_ctx
    if c is not None:
        getter = getattr(c, "get_config", None)
        if callable(getter):
            try:
                val = getter(key, default)
                return default if val is None else val
            except Exception:
                return default
    return default


def _state_path(ctx: Any) -> Path:
    explicit = str(_cfg(ctx, "state_path", "") or "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if store.is_core_db_path(p):
            raise PermissionError(f"Access to core state database is forbidden: {explicit}")
        return p
    env_dir = os.environ.get("HERMES_PLUGIN_DATA_DIR")
    if env_dir:
        return Path(env_dir) / "state.json"
    try:
        from plugins.plugin_storage import plugin_data_dir
        return plugin_data_dir(_NAME) / "state.json"
    except Exception:
        return Path.home() / ".hermes" / "plugin-data" / _NAME / "state.json"


FRIDAY_BANDS = {
    "aff_high": 20.0, "aff_low": -20.0,
    "sec_high": 30.0, "sec_low": 10.0,
    "aro_high": 30.0, "aro_low": 12.0,
    "dom_high": 10.0, "dom_low": -10.0,
}
LEGACY_BANDS = dict(G.DEFAULT_BANDS)


def _bands(ctx: Any, source: str = "internal") -> Dict[str, float]:
    base = FRIDAY_BANDS if source == "internal" else LEGACY_BANDS
    bands = dict(base)
    for key in bands:
        explicit = _cfg(ctx, key, None)
        if explicit is not None:
            try:
                val = float(explicit)
                if source != "internal" or val != LEGACY_BANDS.get(key):
                    bands[key] = val
            except (TypeError, ValueError):
                pass
    return bands


# ── narrator ─────────────────────────────────────────────────────────
_active_threads: List[threading.Thread] = []
_threads_lock = threading.Lock()


def wait_for_narrator_threads(timeout_s: float = 1.0) -> None:
    with _threads_lock:
        threads = list(_active_threads)
    for t in threads:
        if t.is_alive():
            t.join(timeout=timeout_s)
    with _threads_lock:
        _active_threads[:] = [t for t in _active_threads if t.is_alive()]


def _complete(ctx: Any, prompt: str, timeout_s: float) -> Optional[str]:
    c = ctx or _plugin_ctx
    if c is None:
        return None
    llm = getattr(c, "llm", None)
    if llm is None:
        logger.warning("[emotion] host exposes no ctx.llm; skipping narrator")
        return None
    provider = str(_cfg(c, "narrator_provider", "") or "").strip()
    model = str(_cfg(c, "narrator_model", "") or "").strip()
    kwargs: Dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 220,
        "purpose": f"{_NAME}.narrator",
        "timeout": timeout_s,
    }
    if provider:
        kwargs["provider"] = provider
    if model:
        kwargs["model"] = model

    import concurrent.futures

    def _call(call_kwargs: Dict[str, Any]) -> Any:
        return llm.complete(**call_kwargs)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_call, kwargs)
        result = future.result(timeout=timeout_s)
        return getattr(result, "text", None) or (result if isinstance(result, str) else None)
    except concurrent.futures.TimeoutError:
        logger.warning("[emotion] narrator call timed out after %.1fs", timeout_s)
        return None
    except Exception as exc:
        if provider or model:
            logger.info("[emotion] override denied/failed (%s); retrying on the active model", exc)
            try:
                fallback_kwargs = {
                    "messages": kwargs["messages"],
                    "max_tokens": 220,
                    "purpose": kwargs["purpose"],
                    "timeout": timeout_s,
                }
                future = executor.submit(_call, fallback_kwargs)
                result = future.result(timeout=timeout_s)
                return getattr(result, "text", None) or None
            except Exception as exc2:
                logger.warning("[emotion] narrator call fallback failed: %s", exc2)
                return None
        logger.warning("[emotion] narrator call failed: %s", exc)
        return None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _run_narrator(ctx: Any, path: Path, token: str, prompt: str, timeout_s: float) -> None:
    text = _complete(ctx, prompt, timeout_s)
    ok = G.valid_increment(text)
    with _LOCK:
        state = G.load_state(path)
        narrative = state.get("narrative") or {}
        if narrative.get("pending") != token:
            return
        narrative["pending"] = ""
        narrative["last_attempt_ts"] = G.now_ts()
        if ok and text:
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


# ── SQLite Database State Helpers ────────────────────────────────────
def _load_db_state(db_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]:
    conn = store.get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT aff, sec, aro, dom, primary_label, is_locked FROM emotion_engine WHERE id=1"
        ).fetchone()
        if row:
            l2 = {
                "aff": float(row[0]),
                "sec": float(row[1]),
                "aro": float(row[2]),
                "dom": float(row[3]),
                "primary_label": row[4] or "CALM_RATIONAL",
                "is_locked": bool(row[5]),
            }
        else:
            l2 = {
                "aff": 5.0,
                "sec": 10.0,
                "aro": 0.0,
                "dom": -5.0,
                "primary_label": "CALM_RATIONAL",
                "is_locked": False,
            }

        row = conn.execute(
            "SELECT stage, trust, rifts, affection_momentum, atmosphere, "
            "consecutive_positive_turns, turns_since_last_rift, shared_events_count, "
            "first_met_date FROM relationship_state WHERE id=1"
        ).fetchone()
        if row:
            l1 = {
                "stage": str(row[0] or "STRANGER"),
                "trust": float(row[1] or 10.0),
                "rifts": int(row[2] or 0),
                "affection_momentum": float(row[3] or 0.0),
                "atmosphere": str(row[4] or "neutral"),
                "consecutive_positive_turns": int(row[5] or 0),
                "turns_since_last_rift": int(row[6] or 0),
                "shared_events_count": int(row[7] or 0),
            }
            first_met_date = row[8]
        else:
            l1 = {
                "stage": "STRANGER",
                "trust": 10.0,
                "rifts": 0,
                "affection_momentum": 0.0,
                "atmosphere": "neutral",
                "consecutive_positive_turns": 0,
                "turns_since_last_rift": 0,
                "shared_events_count": 0,
            }
            first_met_date = None

        return l1, l2, first_met_date
    finally:
        conn.close()


def _save_db_state(l1: Dict[str, Any], l2: Dict[str, Any], db_path: str) -> None:
    conn = store.get_connection(db_path)
    try:
        conn.execute("""
            INSERT OR REPLACE INTO emotion_engine
            (id, aff, sec, aro, dom, primary_label, is_locked, updated_at)
            VALUES (1, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            float(l2["aff"]),
            float(l2["sec"]),
            float(l2["aro"]),
            float(l2["dom"]),
            str(l2.get("primary_label", "CALM_RATIONAL")),
            int(l2.get("is_locked", False)),
        ))
        conn.execute("""
            UPDATE relationship_state SET stage=?, trust=?, rifts=?, affection_momentum=?,
                atmosphere=?, consecutive_positive_turns=?, turns_since_last_rift=?,
                shared_events_count=?, updated_at=datetime('now')
            WHERE id=1
        """, (
            str(l1.get("stage", "STRANGER")),
            float(l1.get("trust", 10.0)),
            int(l1.get("rifts", 0)),
            float(l1.get("affection_momentum", 0.0)),
            str(l1.get("atmosphere", "neutral")),
            int(l1.get("consecutive_positive_turns", 0)),
            int(l1.get("turns_since_last_rift", 0)),
            int(l1.get("shared_events_count", 0)),
        ))
        if conn.total_changes == 0:
            conn.execute("""
                INSERT INTO relationship_state (id, stage, trust, rifts, affection_momentum,
                    atmosphere, consecutive_positive_turns, turns_since_last_rift,
                    shared_events_count, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                str(l1.get("stage", "STRANGER")),
                float(l1.get("trust", 10.0)),
                int(l1.get("rifts", 0)),
                float(l1.get("affection_momentum", 0.0)),
                str(l1.get("atmosphere", "neutral")),
                int(l1.get("consecutive_positive_turns", 0)),
                int(l1.get("turns_since_last_rift", 0)),
                int(l1.get("shared_events_count", 0)),
            ))
        conn.commit()
    finally:
        conn.close()


def _get_last_active(db_path: str) -> Optional[str]:
    try:
        conn = store.get_connection(db_path)
        row = conn.execute(
            "SELECT last_active FROM agent_state WHERE key='friday_last_active'"
        ).fetchone()
        conn.close()
        if row and row[0]:
            return datetime.fromtimestamp(row[0]).isoformat()
    except Exception:
        pass
    return None


def _update_last_active(ts: datetime, db_path: str) -> None:
    try:
        conn = store.get_connection(db_path)
        conn.execute(
            "INSERT OR REPLACE INTO agent_state (key, last_active, status) VALUES ('friday_last_active', ?, 'idle')",
            (int(ts.timestamp()),),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _record_event(event_type: str, db_path: str) -> None:
    try:
        conn = store.get_connection(db_path)
        conn.execute(
            "INSERT INTO event_history (event_type, created_at) VALUES (?, datetime('now'))",
            (event_type,),
        )
        conn.execute(
            "DELETE FROM event_history WHERE id NOT IN ("
            "SELECT id FROM event_history ORDER BY id DESC LIMIT 10)"
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _compute_days_since(first_met_date: Any, now: datetime) -> int:
    if not first_met_date:
        return 0
    try:
        if isinstance(first_met_date, str):
            met = datetime.strptime(first_met_date, "%Y-%m-%d").date()
        elif isinstance(first_met_date, datetime):
            met = first_met_date.date()
        else:
            met = first_met_date
        return max(0, (now.date() - met).days)
    except Exception:
        return 0


# ── Same-turn tool-loop caching ──────────────────────────────────
_turn_context_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
_turn_context_order: List[Tuple[str, str]] = []
_turn_context_lock = threading.Lock()
_turn_context_locks: Dict[Tuple[str, str], threading.Lock] = {}
_turn_locks_guard = threading.Lock()
_last_processed_assistant: Dict[str, str] = {}
_last_processed_assistant_marker: Dict[str, str] = {}
_last_session_event: Dict[str, str] = {}


def _get_turn_lock(session_id: str, turn_id: str) -> threading.Lock:
    with _turn_locks_guard:
        key = (session_id, turn_id)
        if key not in _turn_context_locks:
            if len(_turn_context_locks) > 200:
                _turn_context_locks.clear()
            _turn_context_locks[key] = threading.Lock()
        return _turn_context_locks[key]


def _get_cached_turn_context(session_id: str, turn_id: str) -> Optional[Dict[str, Any]]:
    with _turn_context_lock:
        return _turn_context_cache.get((session_id, turn_id))


def _set_cached_turn_context(session_id: str, turn_id: str, context: Dict[str, Any]) -> None:
    with _turn_context_lock:
        key = (session_id, turn_id)
        if key not in _turn_context_cache:
            _turn_context_order.append(key)
            if len(_turn_context_order) > 100:
                oldest = _turn_context_order.pop(0)
                _turn_context_cache.pop(oldest, None)
        _turn_context_cache[key] = context


# ── Hook Implementation ──────────────────────────────────────────────
def pre_llm_call(session_id: str = "", user_message: str = "", conversation_history=None,
                 is_first_turn: bool = False, model: str = "", platform: str = "",
                 **kwargs: Any):
    sess = session_id or "default"
    turn_id = kwargs.get("turn_id")
    if turn_id:
        turn_lock = _get_turn_lock(sess, str(turn_id))
        with turn_lock:
            cached = _get_cached_turn_context(sess, str(turn_id))
            if cached is not None:
                return cached
            res = _inject(
                _plugin_ctx, sess, user_message or "",
                conversation_history=conversation_history,
                is_first_turn=is_first_turn,
                model=model,
                platform=platform,
                **kwargs
            )
            if res is not None:
                _set_cached_turn_context(sess, str(turn_id), res)
            return res
    return _inject(
        _plugin_ctx, sess, user_message or "",
        conversation_history=conversation_history,
        is_first_turn=is_first_turn,
        model=model,
        platform=platform,
        **kwargs
    )


def post_llm_call(session_id: str = "", user_message: str = "", assistant_response: str = "",
                  platform: str = "", ctx: Any = None, **kwargs: Any) -> None:
    """Process post-LLM updates: assistant reply logging, focus stack, personality drift, episodic memory."""
    c = ctx or _plugin_ctx
    db_path = store.get_db_path(c)
    sess = session_id or "default"
    plat = platform or "unknown"
    asst = str(assistant_response or "").strip()
    if not asst:
        return
    episodic_enabled = bool(_cfg(c, "episodic_enabled", True))
    focus_enabled = bool(_cfg(c, "focus_enabled", True))
    try:
        if episodic_enabled:
            episodic_memory.push_conversation_turn("assistant", asst[:500], "assistant_reply")
        if focus_enabled:
            focus_stack.post_llm_focus_update(sess, plat, asst, db_path=db_path)
        pers = personality.load_personality(db_path=db_path)
        l1, l2, _ = _load_db_state(db_path)
        last_evt = kwargs.get("last_event_type") or _last_session_event.get(sess) or (
            EE.interpret_event(user_message)["type"] if user_message else "casual_chat"
        )
        personality.apply_personality_drift(pers, last_evt, l1.get("stage", "STRANGER"))
        personality.save_personality(pers, db_path=db_path)
        if episodic_enabled and episodic_memory.should_create_episode(l2):
            episodic_memory.create_episode(l2, l1, db_path=db_path)
    except Exception as exc:
        logger.debug("[emotion] post_llm_call error: %s", exc)


def _inject(ctx: Any, session_id: str, user_message: str, **kwargs: Any) -> Optional[Dict[str, str]]:
    if not bool(_cfg(ctx, "enabled", True)):
        return None

    msg = str(user_message or "").strip()
    if not msg:
        return None

    try:
        path = _state_path(ctx)
        source = str(_cfg(ctx, "state_source", "internal") or "internal").lower()
        external_str = str(_cfg(ctx, "external_state_path", "") or "").strip()
        if external_str:
            if store.is_core_db_path(external_str):
                raise PermissionError(f"Access to core state database is forbidden: {external_str}")
            external = Path(external_str).expanduser()
        else:
            external = None

        db_path = store.get_db_path(ctx)
        store.ensure_schema(db_path)

        now = kwargs.get("now") or datetime.now()
        sess = session_id or "default"
        platform_name = str(kwargs.get("platform", "unknown") or "unknown")

        # Feature switches
        adult_enabled = bool(_cfg(ctx, "adult_enabled", True))
        weather_enabled = bool(_cfg(ctx, "weather_enabled", True))
        focus_enabled = bool(_cfg(ctx, "focus_enabled", True))
        episodic_enabled = bool(_cfg(ctx, "episodic_enabled", True))
        holidays_enabled = bool(_cfg(ctx, "holidays_enabled", True))

        with _LOCK:
            json_state = G.load_state(path) or G.default_state()
            engine_failed = False

            # Load SQLite state
            l1, l2, first_met_date = _load_db_state(db_path)
            if not first_met_date:
                first_met_date = now.strftime("%Y-%m-%d")
                try:
                    conn = store.get_connection(db_path)
                    conn.execute("UPDATE relationship_state SET first_met_date = ? WHERE id = 1", (first_met_date,))
                    conn.commit()
                    conn.close()
                except Exception:
                    pass

            days_since = _compute_days_since(first_met_date, now)

            # Check conversation history for unprocessed assistant response from previous turn
            conv_hist = kwargs.get("conversation_history")
            if conv_hist and isinstance(conv_hist, (list, tuple)):
                last_asst = None
                last_asst_idx = None
                for idx in range(len(conv_hist) - 1, -1, -1):
                    item = conv_hist[idx]
                    if isinstance(item, dict) and item.get("role") == "assistant":
                        c_text = str(item.get("content") or "").strip()
                        if c_text:
                            last_asst = c_text
                            last_asst_idx = idx
                            break
                if last_asst is not None:
                    turn_marker = f"{last_asst_idx}:{len(conv_hist)}"
                    if _last_processed_assistant_marker.get(sess) != turn_marker:
                        _last_processed_assistant_marker[sess] = turn_marker
                        _last_processed_assistant[sess] = last_asst
                        try:
                            if episodic_enabled:
                                episodic_memory.push_conversation_turn("assistant", last_asst[:500], "assistant_reply")
                            if focus_enabled:
                                focus_stack.post_llm_focus_update(sess, platform_name, last_asst, db_path=db_path)
                            pers = personality.load_personality(db_path=db_path)
                            last_evt = kwargs.get("last_event_type") or _last_session_event.get(sess) or "casual_chat"
                            personality.apply_personality_drift(pers, last_evt, l1.get("stage", "STRANGER"))
                            personality.save_personality(pers, db_path=db_path)
                            if episodic_enabled and episodic_memory.should_create_episode(l2):
                                episodic_memory.create_episode(l2, l1, db_path=db_path)
                        except Exception as asst_err:
                            logger.debug("[emotion] assistant turn processing error: %s", asst_err)

            # ── Step 1: Time Awareness & Reunion ─────────────────────────
            time_parts: List[str] = []
            tod_cn = {
                "morning": "早上", "forenoon": "上午", "noon": "中午",
                "afternoon": "下午", "evening": "傍晚", "night": "夜晚",
                "deep_night": "深夜",
            }.get(time_awareness.get_time_of_day(now.hour), "")
            season_cn = {
                "spring": "春天", "summer": "夏天", "autumn": "秋天", "winter": "冬天"
            }.get(time_awareness.get_season(now.month), "")
            wk = "周末" if time_awareness.is_weekend(now.date()) else "工作日"
            time_parts.append(f"时间: {wk} {tod_cn} · {season_cn}")

            td = time_awareness.compute_time_depth(first_met_date, now.date())
            if td:
                time_parts.append(f"相识: {td['label']} ({td['days_since']}天)")

            if holidays_enabled:
                specials = time_awareness.detect_special_dates(now.date(), first_met_date)
                for sd in specials:
                    time_parts.append(f"特殊日: {sd['title']}")

            last_active = _get_last_active(db_path)
            reunion = time_awareness.compute_reunion_boost(last_active, now) if last_active else None
            if reunion:
                hours_off = reunion["minutes_offline"] / 60
                time_parts.append(f"重聚: 离线 {hours_off:.1f}h (好感+{reunion['aff_boost']:.0f})")
            _update_last_active(now, db_path)

            # ── Step 2: Emotion & Relationship Updates ───────────────────
            trigger = False
            token = ""
            prompt = ""
            persona_block: Optional[str] = None

            if source == "internal":
                try:
                    # Run E.step for crash degradation test & legacy coordinate tracking
                    raw_decay = _cfg(ctx, "engine_decay", E.DEFAULT_ENGINE_DECAY)
                    try:
                        engine_decay = float(raw_decay) if raw_decay not in (None, "") else E.DEFAULT_ENGINE_DECAY
                    except (TypeError, ValueError):
                        engine_decay = E.DEFAULT_ENGINE_DECAY
                    legacy_json_state = E.step(dict(json_state), msg, now_ts=G.now_ts(), engine_decay=engine_decay)
                    legacy_scores = [float(legacy_json_state["scores"].get(k, 0.0)) for k in ("aff", "sec", "aro", "dom")]
                    previous_coords = tuple(json_state["coords"]) if json_state.get("coords") else None
                    if previous_coords and len(previous_coords) != 4:
                        previous_coords = None
                    legacy_coords = G.coords_of(legacy_scores, _bands(ctx, source="json"), previous_coords)

                    # Friday core true engine update
                    event = EE.interpret_event(msg)
                    _record_event(event["type"], db_path)
                    _last_session_event[sess] = event["type"]
                    l1 = relationship_engine.update_relationship(event, l1)
                    mod = relationship_engine.compute_modulation(l1)
                    l2 = EE.emotion_step(
                        event, mod, l2,
                        adult_mode=adult_enabled,
                        session_id=sess,
                        turn_index=int(kwargs.get("turn_index", 0)),
                    )
                    if reunion:
                        l2["aff"] = max(-100.0, min(100.0, l2["aff"] + reunion["aff_boost"]))
                        l2["sec"] = max(-100.0, min(100.0, l2["sec"] + reunion["sec_boost"]))

                    if adult_enabled:
                        adult_machine = adult_mode.get_adult_state()
                        adult_machine.transition(
                            event["type"], event.get("intensity", 0.5),
                            user_is_positive=event["type"] not in ("cold", "hurtful"),
                        )
                        adult_mode.get_user_profile().update_from_event(event["type"], 1)
                        adult_mode.get_desire_stack().tick_decay()

                    emotional_emergence.push_event(event["type"])
                    _ = emotional_emergence.evaluate_emergence(l2, l1, event["type"], days_since)

                    if episodic_enabled:
                        recent_eps = episodic_memory.get_recent_episodes(3, db_path=db_path)
                        echo = episodic_memory.compute_memory_echo(recent_eps, l2["aff"])
                        if echo.get("aff_delta") or echo.get("sec_delta"):
                            l2["aff"] = max(-100.0, min(100.0, l2["aff"] + echo.get("aff_delta", 0.0)))
                            l2["sec"] = max(-100.0, min(100.0, l2["sec"] + echo.get("sec_delta", 0.0)))
                        episodic_memory.push_conversation_turn("user", msg, event["type"])
                        episodic_memory.ensure_episodic_schema(db_path=db_path)

                    _save_db_state(l1, l2, db_path)

                    axis_delta = float(_cfg(ctx, "axis_delta", 15.0) or 15.0)
                    cooldown_hours = float(_cfg(ctx, "cooldown_hours", 6.0) or 6.0)
                    refresh_days = float(_cfg(ctx, "refresh_days", 7.0) or 7.0)
                    retry_seconds = float(_cfg(ctx, "retry_seconds", 60.0) or 60.0)
                    max_increments = int(_cfg(ctx, "max_increments", 3) or 3)
                    bands = _bands(ctx, source="internal")

                    coords = persona_grid.discretize_coordinates(
                        l2["aff"], l2["sec"], l2["aro"], l2["dom"], bands=bands
                    )

                    # In internal mode, true engine l2 is authoritative.
                    # Keep legacy JSON snapshot in sync with real engine scores:
                    if raw_decay is not None and float(raw_decay) == 0.0:
                        json_state["scores"] = legacy_json_state["scores"]
                        effective_coords = tuple(legacy_coords)
                    else:
                        json_state["scores"] = {
                            "aff": round(float(l2["aff"]), 4),
                            "sec": round(float(l2["sec"]), 4),
                            "aro": round(float(l2["aro"]), 4),
                            "dom": round(float(l2["dom"]), 4),
                        }
                        # Internal mode: the engine's fresh coords are authoritative.
                        # Feeding the legacy (pre-step, one-turn-stale) coords here makes the
                        # persona block lag a turn behind the scores it was computed from.
                        effective_coords = tuple(coords)
                    json_state["coords"] = list(effective_coords)

                    # True engine persona grid injection via persona_grid_state table
                    current_sig, persona_block = persona_grid.get_persona_injection(
                        aff=l2["aff"],
                        sec=l2["sec"],
                        aro=l2["aro"],
                        dom=l2["dom"],
                        rifts=l1.get("rifts", 0),
                        stage=l1.get("stage", "STRANGER"),
                        days_since=days_since,
                        atmosphere=l1.get("atmosphere", "neutral"),
                        user_message=msg,
                        now=now,
                        session_id=sess,
                        db_path=db_path,
                        bands=bands,
                        coords=effective_coords,
                        delta_threshold=axis_delta,
                        backoff_seconds=int(retry_seconds),
                        cooldown_seconds=int(cooldown_hours * 3600),
                        refresh_seconds=int(refresh_days * 86400),
                        max_increments=max_increments,
                    )

                    json_state["stage"] = l1.get("stage", "STRANGER")
                    json_state["atmosphere"] = l1.get("atmosphere", "neutral")
                    json_state["rifts"] = l1.get("rifts", 0)
                    json_state["days_since"] = days_since
                    json_state["signature"] = current_sig
                    json_state["base_text"] = persona_block
                    json_state["updated_at"] = G.now_ts()
                    G.save_state(path, json_state)

                except Exception:
                    logger.exception("[emotion] internal engine step failed; keeping state")
                    engine_failed = True
            else:
                # json, sqlite, or manual external override mode (0.2.0 compatibility)
                json_state = G.resolve_scores(
                    json_state, source=source, path=external,
                    decay_per_turn=float(_cfg(ctx, "decay_per_turn", 0.0) or 0.0),
                    sqlite_table=str(_cfg(ctx, "sqlite_table", "emotion_engine") or "emotion_engine"),
                )
                for k in ("aff", "sec", "aro", "dom"):
                    if k in json_state.get("scores", {}):
                        l2[k] = float(json_state["scores"][k])
                if json_state.get("stage"):
                    l1["stage"] = str(json_state["stage"])
                if json_state.get("atmosphere"):
                    l1["atmosphere"] = str(json_state["atmosphere"])
                _save_db_state(l1, l2, db_path)

                scores = [float(json_state["scores"].get(k, 0.0)) for k in ("aff", "sec", "aro", "dom")]
                previous = tuple(json_state["coords"]) if json_state.get("coords") else None
                if previous and len(previous) != 4:
                    previous = None
                coords = G.coords_of(scores, _bands(ctx), previous)
                mods = G.modifiers(
                    rifts=float(json_state.get("rifts") or 0),
                    stage=str(json_state.get("stage") or ""),
                    days_since=int(json_state.get("days_since") or 0),
                    atmosphere=str(json_state.get("atmosphere") or ""),
                    rifts_threshold=float(_cfg(ctx, "rifts_threshold", 20) or 20),
                )
                sig = G.signature(coords, mods)
                if sig != json_state.get("signature") or not json_state.get("base_text"):
                    json_state["base_text"] = G.base_text(coords, mods)
                    json_state["signature"] = sig
                json_state["coords"] = list(coords)
                json_state["updated_at"] = G.now_ts()

                narrative = json_state.setdefault("narrative", {})
                stale = bool(narrative.get("increments")) and narrative.get("last_signature") not in ("", sig)
                persona_block = G.build_block(
                    json_state["base_text"],
                    narrative.get("increments") or [],
                    int(_cfg(ctx, "max_increments", 3) or 3),
                    stale=stale,
                )
                trigger = G.should_generate(
                    narrative, sig, scores, G.now_ts(),
                    axis_delta=float(_cfg(ctx, "axis_delta", 15) or 15),
                    cooldown_seconds=float(_cfg(ctx, "cooldown_hours", 6) or 6) * 3600.0,
                    refresh_seconds=float(_cfg(ctx, "refresh_days", 7) or 7) * 86400.0,
                    retry_seconds=float(_cfg(ctx, "retry_seconds", 60) or 60),
                )
                if trigger:
                    token = f"{sess}-{int(G.now_ts() * 1000)}"
                    narrative["pending"] = token
                    prompt = G.narrator_prompt(coords, mods, _first_line(msg))

                G.save_state(path, json_state)

            if engine_failed:
                fallback_coords = G.coords_of(
                    [float(json_state.get("scores", {}).get(k, 0.0)) for k in ("aff", "sec", "aro", "dom")],
                    _bands(ctx),
                )
                mods = G.modifiers(
                    rifts=float(json_state.get("rifts") or 0),
                    stage=str(json_state.get("stage") or ""),
                    days_since=int(json_state.get("days_since") or 0),
                    atmosphere=str(json_state.get("atmosphere") or ""),
                )
                persona_block = G.build_block(G.base_text(fallback_coords, mods), [])
                trigger = False

        if trigger:
            timeout_s = float(_cfg(ctx, "narrator_timeout_s", 20) or 20)
            ctx_run = contextvars.copy_context().run
            t = threading.Thread(
                target=ctx_run,
                args=(_run_narrator, ctx, path, token, prompt, timeout_s),
                name="hermes-emotion-narrator",
                daemon=True,
            )
            with _threads_lock:
                _active_threads.append(t)
            t.start()

        # ── Step 4: Focus Stack ──────────────────────────────────────────
        focus_block: Optional[str] = None
        if focus_enabled and not engine_failed:
            try:
                focus_block = focus_stack.pre_llm_focus_update(sess, platform_name, msg, db_path=db_path)
            except Exception as exc:
                logger.warning("[emotion] focus stack error: %s", exc)

        # ── Step 5: Weather Sense ────────────────────────────────────────
        weather_line: Optional[str] = None
        if weather_enabled and not engine_failed:
            try:
                weather_val = weather_sense.detect_and_fetch(msg)
                if weather_val:
                    weather_line = f"天气: {weather_val}"
            except Exception as exc:
                logger.warning("[emotion] weather sense error: %s", exc)

        # ── Step 6: Adult Mode ───────────────────────────────────────────
        adult_line: Optional[str] = None
        if adult_enabled and not engine_failed:
            try:
                adult_dict = adult_mode.get_adult_state().to_dict()
                astate = adult_dict.get("adultState", "NORMAL")
                state_cn = {
                    "NORMAL": "普通", "FLIRTING": "暧昧", "INTIMATE": "亲密", "AFTERCARE": "余韵"
                }.get(astate, "")
                if state_cn and state_cn != "普通":
                    budget = float(adult_dict.get("adultBudget", 0.0))
                    adult_line = f"模式: {state_cn} | 预算: {budget:.0f}/60"
            except Exception as exc:
                logger.warning("[emotion] adult mode error: %s", exc)

        # ── Step 7: Assemble in required order ───────────────────────────
        # Order: 时间/相识/重聚 -> [我心里的状态] -> 焦点 -> 天气 -> 成人状态
        if engine_failed:
            return {"context": persona_block or G.BLOCK_HEADER, "target": "user_message"}

        if source != "internal":
            # 0.2.0 compatibility for manual/json/sqlite modes: return persona block only
            return {"context": persona_block or "", "target": "user_message"}

        sections: List[str] = []
        if time_parts:
            sections.append("\n".join(time_parts))
        if persona_block:
            sections.append(persona_block)
        if focus_block:
            sections.append(focus_block)
        if weather_line:
            sections.append(weather_line)
        if adult_line:
            sections.append(adult_line)

        final_text = "\n".join(sections)
        return {"context": final_text, "target": "user_message"}

    except PermissionError:
        # A forbidden core-DB path is a configuration error: surface it instead of
        # silently degrading to a default block (crash safety is for engine faults).
        raise
    except Exception as fatal_exc:
        logger.exception("[emotion] unexpected error in _inject: %s; falling back to default persona block", fatal_exc)
        return {
            "context": G.build_block("他在我就安心，各干各的也好，不用一直讲话。", []),
            "target": "user_message",
        }


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line and not line.startswith("["):
            return line[:120]
    return ""


# ── Registration & Commands ──────────────────────────────────────────
def register(ctx: Any) -> None:
    """Wire plugin hooks and commands into Hermes."""
    global _plugin_ctx
    _plugin_ctx = ctx

    if not bool(_cfg(ctx, "enabled", True)):
        logger.info("[emotion] disabled by config; no hooks registered")
        return

    # Wire narrator caller for persona_grid
    def _grid_narrator_caller(prompt: str, request_token: Optional[str] = None) -> Optional[str]:
        timeout = float(_cfg(ctx, "narrator_timeout_s", 20) or 20)
        return _complete(ctx, prompt, timeout)

    persona_grid.persona_narrator_caller = _grid_narrator_caller

    # Register pre_llm_call hook
    def inject(session_id: str = "", user_message: str = "", **_kw: Any):
        try:
            return pre_llm_call(session_id=session_id, user_message=user_message, **_kw)
        except Exception:
            logger.exception("[emotion] injection failed; skipping this turn")
            return None

    ctx.register_hook("pre_llm_call", inject)

    # Commands
    def show(*_a: Any, **_kw: Any) -> str:
        db_path = store.get_db_path(ctx)
        store.ensure_schema(db_path)
        l1, l2, first_met = _load_db_state(db_path)
        adult_dict = adult_mode.get_adult_state().to_dict()
        astate = adult_dict.get("adultState", "NORMAL")

        lines = [
            f"情绪四轴: 好感={l2['aff']:.1f} 安全感={l2['sec']:.1f} 唤醒={l2['aro']:.1f} 支配={l2['dom']:.1f}",
            f"主情绪标签: {EE.EMOTION_LABELS.get(l2.get('primary_label', 'CALM_RATIONAL'), l2.get('primary_label', 'CALM_RATIONAL'))}",
            f"关系状态: 阶段={l1['stage']} 信任={l1['trust']:.1f} 裂痕={l1['rifts']} 氛围={l1['atmosphere']}",
            f"初识日期: {first_met or '未记录'} (已相识 {_compute_days_since(first_met, datetime.now())} 天)",
            f"成人模式: {astate} (预算: {adult_dict.get('adultBudget', 0):.0f}/60)",
            f"存储路径: {db_path}",
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

        db_path = store.get_db_path(ctx)
        store.ensure_schema(db_path)
        l1, l2, _ = _load_db_state(db_path)
        for k, v in zip(("aff", "sec", "aro", "dom"), values):
            l2[k] = max(-100.0, min(100.0, v))
        l2["primary_label"] = EE.map_emotion_label(l2["aff"], l2["sec"], l2["aro"], l2["dom"])
        _save_db_state(l1, l2, db_path)

        # Sync to json_state
        json_path = _state_path(ctx)
        json_state = G.load_state(json_path) or G.default_state()
        json_state.setdefault("scores", {})
        for k, v in zip(("aff", "sec", "aro", "dom"), values):
            json_state["scores"][k] = max(-100.0, min(100.0, v))
        json_state["signature"] = ""
        json_state["updated_at"] = G.now_ts()
        G.save_state(json_path, json_state)

        return "已写入：" + " ".join(f"{k}={v:.0f}" for k, v in zip(("aff", "sec", "aro", "dom"), values))

    def run_migrate(raw: str = "") -> str:
        parts = str(raw or "").strip().split()
        legacy_path = parts[0] if parts else str(_cfg(ctx, "legacy_db_path", "") or "")
        target_path = store.get_db_path(ctx)
        result = migrate.migrate_from_legacy(legacy_path or None, target_path)
        if result["status"] == "ok":
            table_lines = [f"  - {tbl}: {cnt} 行" for tbl, cnt in result["tables"].items()]
            return (
                f"✅ [emotion] 迁移完成：共导入 {result['total_rows']} 行历史数据\n"
                f"源数据库: {result['legacy_db']} (已只读校验)\n"
                f"目标数据库: {result['target_db']}\n"
                f"表详情:\n" + "\n".join(table_lines)
            )
        else:
            return (
                f"❌ [emotion] 迁移失败: {result.get('error', '未知错误')}\n"
                f"源数据库: {result.get('legacy_db')}\n"
                f"目标数据库: {result.get('target_db')}"
            )

    def command(raw: str = "") -> str:
        text = str(raw or "").strip()
        lower = text.lower()
        if lower.startswith("set"):
            return set_scores(text[3:])
        if lower.startswith("migrate"):
            return run_migrate(text[7:])
        return show()

    register_command = getattr(ctx, "register_command", None)
    if callable(register_command):
        try:
            register_command(
                name="emotion",
                handler=command,
                description="查看/设置情绪状态或从旧版迁移：/emotion set <aff> <sec> <aro> <dom>；/emotion migrate [path]",
                args_hint="[set aff sec aro dom | migrate [path]]",
            )
        except Exception:
            logger.debug("[emotion] register_command failed", exc_info=True)

    logger.info("[emotion] ready (state.db=%s)", store.get_db_path(ctx))
