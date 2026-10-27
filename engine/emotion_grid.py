"""Emotion → persona text: deterministic grid, hysteresis, guarded increments.

Pure logic only — no Hermes imports, no I/O outside the small state helpers, so
it can be unit-tested without a host process.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .store import is_core_db_path

Coords = Tuple[str, str, str, str]

# ── 16 格人设文本 ────────────────────────────────────────────────────
# 顺序：好感(近/远) · 安全感(稳/悬) · 唤醒(热/静) · 支配(进/收)
GRID_TEXTS: Dict[Coords, str] = {
    ("近", "稳", "热", "进"): "我心情好得藏不住，就想凑到跟前，让他看我一眼。",
    ("近", "稳", "热", "收"): "心里是甜的，可我不敢先开口，等他自己过来。",
    ("近", "稳", "静", "进"): "我踏实得很，慢慢挪过去挨着他坐，不着急说话。",
    ("近", "稳", "静", "收"): "他在我就安心，各干各的也好，不用一直讲话。",
    ("近", "悬", "热", "进"): "我心里烧得慌，又没底，想立刻问清楚他到底怎么想。",
    ("近", "悬", "热", "收"): "我特别想凑过去，一想到他可能嫌我黏，话又咽回去了。",
    ("近", "悬", "静", "进"): "我对他还是热乎的，只是没底，就轻轻问一句他在忙什么。",
    ("近", "悬", "静", "收"): (
        "我对他还是热乎的，可心里一直悬着，他一忙我就不知道该往哪站。"
        "想凑近，又怕显得太急，所以话到嘴边总收一半。"
    ),
    ("远", "稳", "热", "进"): "我对他没什么热乎气了，但也不慌，有件事我想挑明说。",
    ("远", "稳", "热", "收"): "我没什么心思热络，他要说错话，我大概会冷冷刺一句。",
    ("远", "稳", "静", "进"): "公事公办，我不多想，把该说的说清楚就行。",
    ("远", "稳", "静", "收"): "我跟他客气着，各退一步，谁也不往里走。",
    ("远", "悬", "热", "进"): "我心里憋着火，话已经冲到嘴边了。",
    ("远", "悬", "热", "收"): "我委屈，可我不想闹，就堵着不说。",
    ("远", "悬", "静", "进"): "我冷着，账我记着，等他哪天自己撞上来。",
    ("远", "悬", "静", "收"): "我不想说话，缩在一边，等他先开口。",
}

DEFAULT_BANDS = {
    "aff_high": 8.0, "aff_low": -8.0,
    "sec_high": 18.0, "sec_low": 5.0,
    "aro_high": 18.0, "aro_low": 5.0,
    "dom_high": 6.0, "dom_low": -6.0,
}

BLOCK_HEADER = "[我心里的状态]"
INCREMENT_PREFIX = "人设现在变更为："
INCREMENT_TAIL = "（以上状态以最后一条变更为准，与当前基准状态保持一致）"
MAX_INCREMENT_CHARS = 200


def _pick(value: float, high: float, low: float, previous: Optional[str], hi: str, lo: str) -> str:
    """Hysteresis: only flip past the band, else keep the previous side."""
    if value >= high:
        return hi
    if value <= low:
        return lo
    if previous in (hi, lo):
        return previous
    # No history yet: the band's midpoint decides the first reading.
    return hi if value >= (high + low) / 2 else lo


def coords_of(scores: Sequence[float], bands: Dict[str, float], previous: Optional[Coords] = None) -> Coords:
    aff, sec, aro, dom = (float(x) for x in scores)
    prev = previous or (None, None, None, None)
    return (
        _pick(aff, bands["aff_high"], bands["aff_low"], prev[0], "近", "远"),
        _pick(sec, bands["sec_high"], bands["sec_low"], prev[1], "稳", "悬"),
        _pick(aro, bands["aro_high"], bands["aro_low"], prev[2], "热", "静"),
        _pick(dom, bands["dom_high"], bands["dom_low"], prev[3], "进", "收"),
    )


def modifiers(rifts: float = 0, stage: str = "", days_since: int = 0, atmosphere: str = "",
              *, rifts_threshold: float = 20) -> List[str]:
    out: List[str] = []
    if rifts >= rifts_threshold:
        out.append("中间有些事我一直没放下。")
    if stage.upper() == "STRANGER" and days_since >= 60:
        out.append("我们该熟了，可界面上还写着初识。")
    if atmosphere.lower() == "cool":
        out.append("屋里现在是凉的。")
    return out


def signature(coords: Coords, mods: Sequence[str]) -> str:
    return "·".join(coords) + "|" + "|".join(str(i) for i, _ in enumerate(mods) if _)


def base_text(coords: Coords, mods: Sequence[str]) -> str:
    return GRID_TEXTS[coords] + "".join(mods)


STALE_NOTE = "（以上是更早的心情记录，只作参考；以当前这段为准）"


def build_block(base: str, increments: Sequence[str], max_increments: int = 3,
                stale: bool = False) -> str:
    kept = [t for t in increments if t][-max_increments:]
    parts = [BLOCK_HEADER]
    if stale and kept:
        # the grid cell moved on: older self-descriptions must not read as current
        parts.append(STALE_NOTE)
    parts.append(base)
    parts.extend(f"{INCREMENT_PREFIX}{t}" for t in kept)
    if len(kept) > 1:
        parts.append(INCREMENT_TAIL)
    return "\n".join(parts)


# ── increment validation ─────────────────────────────────────────────
_DIGIT_RE = re.compile(r"[0-9０-９]")
_FIELD_RE = re.compile(r"(好感|安全感|信任|裂痕|aff|sec|aro|dom)", re.I)
_SENTENCE_END = re.compile(r"[。！？!?]")


def valid_increment(text: Optional[str]) -> bool:
    if not text:
        return False
    body = text.strip()
    if not body or len(body) > MAX_INCREMENT_CHARS:
        return False
    if _DIGIT_RE.search(body):
        return False
    if _FIELD_RE.search(body):
        return False
    return bool(_SENTENCE_END.search(body))


# ── narrator prompt ──────────────────────────────────────────────────
def narrator_prompt(coords: Coords, mods: Sequence[str], user_line: str = "") -> str:
    mood = "·".join(coords)
    extra = "；".join(mods) if mods else "无"
    user_part = f"对方刚说：{user_line}\n" if user_line else ""
    return (
        "你在以第一人称记录自己此刻的心境，不是写对白。\n"
        f"心境坐标：{mood}（依次为：亲疏·安心与否·情绪高低·主动与否）。\n"
        f"附加状态：{extra}\n"
        f"{user_part}"
        "写两到三句中文，只描述此刻的感受与细微变化；"
        "不要出现任何数字或「好感/安全感/信任/裂痕」这类词；"
        "不要编造过去发生过什么。"
    )


# ── trigger ──────────────────────────────────────────────────────────
def should_generate(narrative: Dict[str, Any], sig: str, scores: Sequence[float],
                    now_ts: float, *, axis_delta: float = 15.0,
                    cooldown_seconds: float = 6 * 3600,
                    refresh_seconds: float = 7 * 86400,
                    retry_seconds: float = 60.0) -> bool:
    last_success = float(narrative.get("last_success_ts") or 0)
    last_attempt = float(narrative.get("last_attempt_ts") or 0)
    has_story = bool(narrative.get("increments"))

    if narrative.get("pending"):
        return False
    since_attempt = now_ts - last_attempt
    if since_attempt < retry_seconds:
        return False
    if last_success and now_ts - last_success < cooldown_seconds:
        return False
    if not has_story or not last_success:
        return True
    if sig != narrative.get("last_signature"):
        return True
    ref = narrative.get("last_scores") or []
    if len(ref) == len(scores):
        if any(abs(float(a) - float(b)) >= axis_delta for a, b in zip(scores, ref)):
            return True
    return now_ts - last_success >= refresh_seconds


# ── state file ───────────────────────────────────────────────────────
def load_state(path: Path | str) -> Dict[str, Any]:
    if is_core_db_path(path):
        raise PermissionError(f"Access to core state database is forbidden: {path}")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(path: Path | str, state: Dict[str, Any]) -> None:
    if is_core_db_path(path):
        raise PermissionError(f"Access to core state database is forbidden: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".emotion-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def default_state() -> Dict[str, Any]:
    return {
        "scores": {"aff": 0.0, "sec": 0.0, "aro": 0.0, "dom": 0.0},
        "rifts": 0,
        "stage": "",
        "days_since": 0,
        "atmosphere": "",
        "narrative": {"increments": [], "last_signature": "", "last_scores": [],
                      "last_success_ts": 0, "last_attempt_ts": 0, "pending": ""},
        "updated_at": 0,
    }


def _read_json_source(path: Path | str) -> Dict[str, Any]:
    if is_core_db_path(path):
        raise PermissionError(f"Access to core state database is forbidden: {path}")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    flat = data.get("scores") if isinstance(data.get("scores"), dict) else data
    out: Dict[str, Any] = {}
    for key in ("aff", "sec", "aro", "dom"):
        try:
            out[key] = float(flat.get(key))
        except (TypeError, ValueError):
            out[key] = 0.0
    for key in ("rifts", "days_since"):
        try:
            out[key] = int(float(flat.get(key) or 0))
        except (TypeError, ValueError):
            out[key] = 0
    out["stage"] = str(flat.get("stage") or "")
    out["atmosphere"] = str(flat.get("atmosphere") or "")
    return out


def _read_sqlite_source(path: Path | str, table: str = "emotion_engine",
                        columns: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Read the newest row of `table` and map its columns onto the four axes."""
    if is_core_db_path(path):
        raise PermissionError(f"Access to core state database is forbidden: {path}")
    cols = {"aff": "aff", "sec": "sec", "aro": "aro", "dom": "dom",
            "rifts": "rifts", "stage": "stage", "days_since": "days_since",
            "atmosphere": "atmosphere"}
    if columns:
        cols.update(columns)
    import sqlite3
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return {}
    try:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        wanted = [c for k, c in cols.items() if c in existing]
        if not wanted:
            return {}
        row = conn.execute(
            f"SELECT {', '.join(wanted)} FROM {table} ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    if not row:
        return {}
    got = dict(zip(wanted, row))
    out: Dict[str, Any] = {}
    for axis in ("aff", "sec", "aro", "dom"):
        try:
            out[axis] = float(got.get(cols[axis]) or 0.0)
        except (TypeError, ValueError):
            out[axis] = 0.0
    for key in ("rifts", "days_since"):
        try:
            out[key] = int(float(got.get(cols[key]) or 0))
        except (TypeError, ValueError):
            out[key] = 0
    out["stage"] = str(got.get(cols["stage"]) or "")
    out["atmosphere"] = str(got.get(cols["atmosphere"]) or "")
    return out


def resolve_scores(state: Dict[str, Any], *, source: str, path: Optional[Path | str],
                   decay_per_turn: float = 0.0,
                   sqlite_table: str = "emotion_engine") -> Dict[str, Any]:
    """Merge the configured source over the plugin's own stored scores."""
    if path is not None and is_core_db_path(path):
        raise PermissionError(f"Access to core state database is forbidden: {path}")
    own = dict(state.get("scores") or {})
    if source in ("json", "sqlite") and path is not None:
        external = (_read_sqlite_source(path, sqlite_table) if source == "sqlite"
                    else _read_json_source(path))
        if external:
            own.update({k: external[k] for k in ("aff", "sec", "aro", "dom") if k in external})
            for key in ("rifts", "stage", "days_since", "atmosphere"):
                if key in external:
                    state[key] = external[key]
    if decay_per_turn:
        for key in ("aff", "sec", "aro", "dom"):
            value = float(own.get(key) or 0.0)
            own[key] = value * (1.0 - decay_per_turn)
    state["scores"] = own
    return state


def now_ts() -> float:
    return time.time()
