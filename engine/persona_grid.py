"""Friday Persona Grid & Incremental Narrative Module.

1. Discretizes (aff, sec, aro, dom) into 4-axis binary coordinates with hysteresis.
2. Maps coordinates to a 16-grid first-person persona text.
3. Assembles modifiers (rifts, long-term stranger, cool atmosphere).
4. Computes persona signature key.
5. Injects [我心里的状态] block with immutable base text + up to 3 increments.
6. Asynchronously requests auxiliary slot `persona_narrator` on signature change,
   large axis delta, or 7+ days without update, respecting 6h cooldown.
7. Silently degrades on small model failure/timeout/invalid output.
"""

import contextvars
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from .store import get_connection, get_db_path

# ═══════════════════════════════════════════════════════════════════
# Constants & Thresholds
# ═══════════════════════════════════════════════════════════════════

# Thresholds with hysteresis bands
AFF_HIGH = 20.0
AFF_LOW = -20.0
SEC_HIGH = 30.0
SEC_LOW = 10.0
ARO_HIGH = 30.0
ARO_LOW = 12.0
DOM_HIGH = 10.0
DOM_LOW = -10.0

DEFAULT_AXIS_DELTA_THRESHOLD = 15.0
try:
    AXIS_DELTA_THRESHOLD = float(os.environ.get("PERSONA_DELTA_THRESHOLD", DEFAULT_AXIS_DELTA_THRESHOLD))
except (ValueError, TypeError):
    AXIS_DELTA_THRESHOLD = DEFAULT_AXIS_DELTA_THRESHOLD

COOLDOWN_SECONDS = 6 * 3600  # 6 hours
MAX_UPDATE_INTERVAL_SECONDS = 7 * 86400  # 7 days
DEFAULT_BACKOFF_SECONDS = 60  # 60 seconds failure backoff
try:
    BACKOFF_SECONDS = int(os.environ.get("PERSONA_BACKOFF_SECONDS", DEFAULT_BACKOFF_SECONDS))
except (ValueError, TypeError):
    BACKOFF_SECONDS = DEFAULT_BACKOFF_SECONDS
MAX_INCREMENTS = 3
MAX_INCREMENT_CHARS = 200
INFLIGHT_TIMEOUT_SECONDS = 60.0

# 16-grid texts - word-for-word constants
PERSONA_GRID_TEXTS: Dict[Tuple[str, str, str, str], str] = {
    ("近", "稳", "热", "进"): "我心情好得藏不住，就想凑到跟前，让他看我一眼。",
    ("近", "稳", "热", "收"): "心里是甜的，可我不敢先开口，等他自己过来。",
    ("近", "稳", "静", "进"): "我踏实得很，慢慢挪过去挨着他坐，不着急说话。",
    ("近", "稳", "静", "收"): "他在我就安心，各干各的也好，不用一直讲话。",
    ("近", "悬", "热", "进"): "我心里烧得慌，又没底，想立刻问清楚他到底怎么想。",
    ("近", "悬", "热", "收"): "我特别想凑过去，一想到他可能嫌我黏，话又咽回去了。",
    ("近", "悬", "静", "进"): "我对他还是热乎的，只是没底，就轻轻问一句他在忙什么。",
    ("近", "悬", "静", "收"): "我对他还是热乎的，可心里一直悬着，他一忙我就不知道该往哪站。想凑近，又怕显得太急，所以话到嘴边总收一半。",
    ("远", "稳", "热", "进"): "我对他没什么热乎气了，但也不慌，有件事我想挑明说。",
    ("远", "稳", "热", "收"): "我没什么心思热络，他要说错话，我大概会冷冷刺一句。",
    ("远", "稳", "静", "进"): "公事公办，我不多想，把该说的说清楚就行。",
    ("远", "稳", "静", "收"): "我跟他客气着，各退一步，谁也不往里走。",
    ("远", "悬", "热", "进"): "我心里憋着火，话已经冲到嘴边了。",
    ("远", "悬", "热", "收"): "我委屈，可我不想闹，就堵着不说。",
    ("远", "悬", "静", "进"): "我冷着，账我记着，等他哪天自己撞上来。",
    ("远", "悬", "静", "收"): "我不想说话，缩在一边，等他先开口。",
}

# Modifiers
MODIFIER_RIFT = "中间有些事我一直没放下。"
MODIFIER_STRANGER = "我们该熟了，可界面上还写着初识。"
MODIFIER_COOL = "屋里现在是凉的。"

FORBIDDEN_TERMS = ("好感", "安全感", "信任", "裂痕")


# ═══════════════════════════════════════════════════════════════════
# Coordinate Discretization with Hysteresis
# ═══════════════════════════════════════════════════════════════════

def discretize_coordinates(
    aff: float,
    sec: float,
    aro: float,
    dom: float,
    last_coords: Optional[Tuple[str, str, str, str]] = None,
    bands: Optional[Dict[str, float]] = None,
) -> Tuple[str, str, str, str]:
    """Discretize (aff, sec, aro, dom) into 4 binary coordinates with hysteresis."""
    b = bands or {}
    aff_h = float(b.get("aff_high", AFF_HIGH))
    aff_l = float(b.get("aff_low", AFF_LOW))
    sec_h = float(b.get("sec_high", SEC_HIGH))
    sec_l = float(b.get("sec_low", SEC_LOW))
    aro_h = float(b.get("aro_high", ARO_HIGH))
    aro_l = float(b.get("aro_low", ARO_LOW))
    dom_h = float(b.get("dom_high", DOM_HIGH))
    dom_l = float(b.get("dom_low", DOM_LOW))

    # aff: 近 / 远
    if aff >= aff_h:
        aff_c = "近"
    elif aff <= aff_l:
        aff_c = "远"
    elif last_coords:
        aff_c = last_coords[0]
    else:
        aff_c = "近" if aff >= (aff_h + aff_l) / 2 else "远"

    # sec: 稳 / 悬
    if sec >= sec_h:
        sec_c = "稳"
    elif sec <= sec_l:
        sec_c = "悬"
    elif last_coords:
        sec_c = last_coords[1]
    else:
        sec_c = "稳" if sec >= (sec_h + sec_l) / 2 else "悬"

    # aro: 热 / 静
    if aro >= aro_h:
        aro_c = "热"
    elif aro <= aro_l:
        aro_c = "静"
    elif last_coords:
        aro_c = last_coords[2]
    else:
        aro_c = "热" if aro >= (aro_h + aro_l) / 2 else "静"

    # dom: 进 / 收
    if dom >= dom_h:
        dom_c = "进"
    elif dom <= dom_l:
        dom_c = "收"
    elif last_coords:
        dom_c = last_coords[3]
    else:
        dom_c = "进" if dom >= (dom_h + dom_l) / 2 else "收"

    return (aff_c, sec_c, aro_c, dom_c)


# ═══════════════════════════════════════════════════════════════════
# Modifiers & Signature
# ═══════════════════════════════════════════════════════════════════

def compute_modifiers(
    rifts: int = 0,
    stage: str = "STRANGER",
    days_since: int = 0,
    atmosphere: str = "neutral",
) -> List[str]:
    """Return active modifier sentences in defined order."""
    mods = []
    if rifts >= 20:
        mods.append(MODIFIER_RIFT)
    if stage == "STRANGER" and days_since >= 60:
        mods.append(MODIFIER_STRANGER)
    if atmosphere == "cool":
        mods.append(MODIFIER_COOL)
    return mods


def compute_signature(
    coords: Tuple[str, str, str, str],
    rifts: int = 0,
    stage: str = "STRANGER",
    days_since: int = 0,
    atmosphere: str = "neutral",
) -> str:
    """Compute persona signature from coordinates and modifier flags."""
    m_rift = 1 if rifts >= 20 else 0
    m_stranger = 1 if (stage == "STRANGER" and days_since >= 60) else 0
    m_cool = 1 if atmosphere == "cool" else 0
    coord_str = "·".join(coords)
    return f"{coord_str}|r:{m_rift}|s:{m_stranger}|a:{m_cool}"


def resolve_persona_grid(
    aff: float,
    sec: float,
    aro: float,
    dom: float,
    rifts: int = 0,
    stage: str = "STRANGER",
    days_since: int = 0,
    atmosphere: str = "neutral",
    last_coords: Optional[Tuple[str, str, str, str]] = None,
) -> Tuple[str, str]:
    """Resolve 4-axis coordinates, modifiers, signature, and fixed segment text.

    Returns:
        (signature, fixed_segment_text)
    """
    coords = discretize_coordinates(aff, sec, aro, dom, last_coords)
    sig = compute_signature(coords, rifts, stage, days_since, atmosphere)
    grid_text = PERSONA_GRID_TEXTS[coords]
    mods = compute_modifiers(rifts, stage, days_since, atmosphere)
    fixed_text = grid_text + "".join(mods)
    return sig, fixed_text


def build_persona_injection_text(
    base_text: str,
    increments: Optional[List[str]] = None,
    current_signature: Optional[str] = None,
    last_narrative_signature: Optional[str] = None,
    max_increments: Optional[int] = None,
) -> str:
    """Build the full [我心里的状态] injection text block.

    If current coordinate/signature has shifted away from the last generated narrative
    (e.g., during cooldown), historical increments are explicitly marked as past reference
    and cannot override the current base status.
    """
    limit = max_increments if max_increments is not None else MAX_INCREMENTS
    lines = ["[我心里的状态]", base_text]
    if increments:
        kept = [inc for inc in increments if inc][-limit:]
        is_signature_diverged = (
            current_signature is not None
            and last_narrative_signature is not None
            and current_signature != last_narrative_signature
        )
        for inc in kept:
            lines.append(f"人设现在变更为：{inc}")
        if is_signature_diverged:
            lines.append("（注意：当前内心坐标已发生迁移，最新基准状态优先，上述历史增量仅供过渡参考，不得推翻当前基准状态）")
        elif len(kept) > 1:
            lines.append("（以上状态以最后一条变更为准，与当前基准状态保持一致）")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Persistence (state.db)
# ═══════════════════════════════════════════════════════════════════

def ensure_persona_schema(db_path: Optional[str] = None) -> None:
    """Ensure persona_grid_state table exists in state.db with correct schema and migration."""
    path = db_path or get_db_path()
    dir_name = os.path.dirname(os.path.abspath(path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    conn = get_connection(path, timeout=30.0)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS persona_grid_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                base_text TEXT NOT NULL,
                base_signature TEXT NOT NULL,
                current_signature TEXT NOT NULL,
                last_coords TEXT,
                last_aff REAL DEFAULT 0.0,
                last_sec REAL DEFAULT 0.0,
                last_aro REAL DEFAULT 0.0,
                last_dom REAL DEFAULT 0.0,
                last_narrative_aff REAL DEFAULT 0.0,
                last_narrative_sec REAL DEFAULT 0.0,
                last_narrative_aro REAL DEFAULT 0.0,
                last_narrative_dom REAL DEFAULT 0.0,
                last_narrative_signature TEXT,
                last_narrative_ts INTEGER DEFAULT 0,
                last_narrative_version INTEGER DEFAULT 0,
                last_attempt_ts INTEGER DEFAULT 0,
                active_request_token TEXT,
                version INTEGER DEFAULT 0,
                increments TEXT DEFAULT '[]',
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        # Safe migration for existing schemas
        cursor = conn.execute("PRAGMA table_info(persona_grid_state)")
        cols = {row[1] for row in cursor.fetchall()}
        if "session_id" not in cols:
            conn.execute("ALTER TABLE persona_grid_state ADD COLUMN session_id TEXT")
        if "last_narrative_version" not in cols:
            conn.execute("ALTER TABLE persona_grid_state ADD COLUMN last_narrative_version INTEGER DEFAULT 0")
        if "last_attempt_ts" not in cols:
            conn.execute("ALTER TABLE persona_grid_state ADD COLUMN last_attempt_ts INTEGER DEFAULT 0")
        if "active_request_token" not in cols:
            conn.execute("ALTER TABLE persona_grid_state ADD COLUMN active_request_token TEXT")
        if "version" not in cols:
            conn.execute("ALTER TABLE persona_grid_state ADD COLUMN version INTEGER DEFAULT 0")
        # Safely migrate legacy rows so unique index won't fail if multiple rows exist
        conn.execute("""
            UPDATE persona_grid_state
            SET session_id = CASE
                WHEN id = (SELECT MIN(id) FROM persona_grid_state WHERE session_id IS NULL OR session_id = '')
                     AND NOT EXISTS (SELECT 1 FROM persona_grid_state WHERE session_id = 'default')
                THEN 'default'
                ELSE 'legacy_' || id
            END
            WHERE session_id IS NULL OR session_id = ''
        """)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_persona_grid_session ON persona_grid_state(session_id)")
        conn.commit()
    finally:
        conn.close()


def load_persona_state(
    session_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load persona grid state from DB for a given session.
    
    Supports both load_persona_state(db_path) and load_persona_state(session_id, db_path).
    """
    if session_id and db_path is None and ('/' in session_id or session_id.endswith('.db')):
        db_path = session_id
        session_id = "default"

    target_session = session_id or "default"
    path = db_path or get_db_path()
    ensure_persona_schema(path)
    conn = get_connection(path, timeout=30.0)
    try:
        row = conn.execute("""
            SELECT session_id, base_text, base_signature, current_signature, last_coords,
                   last_aff, last_sec, last_aro, last_dom,
                   last_narrative_aff, last_narrative_sec, last_narrative_aro, last_narrative_dom,
                   last_narrative_signature, last_narrative_ts, last_narrative_version, last_attempt_ts, version, increments, active_request_token
            FROM persona_grid_state WHERE session_id = ?
        """, (target_session,)).fetchone()
        if not row:
            return None
        coords_val = tuple(row[4].split(",")) if row[4] else None
        inc_val = json.loads(row[18]) if row[18] else []
        return {
            "session_id": row[0],
            "base_text": row[1],
            "base_signature": row[2],
            "current_signature": row[3],
            "last_coords": coords_val,
            "last_aff": row[5],
            "last_sec": row[6],
            "last_aro": row[7],
            "last_dom": row[8],
            "last_narrative_aff": row[9],
            "last_narrative_sec": row[10],
            "last_narrative_aro": row[11],
            "last_narrative_dom": row[12],
            "last_narrative_signature": row[13],
            "last_narrative_ts": row[14],
            "last_narrative_version": row[15] or 0,
            "last_attempt_ts": row[16] or 0,
            "version": row[17] or 0,
            "increments": inc_val,
            "active_request_token": row[19] if len(row) > 19 else None,
        }
    finally:
        conn.close()


def save_persona_state(
    state: Dict[str, Any],
    db_path: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """Save persona grid state to DB without overwriting worker-managed attempt and narrative fields."""
    target_session = session_id or state.get("session_id") or "default"
    state["session_id"] = target_session
    path = db_path or get_db_path()
    ensure_persona_schema(path)
    conn = get_connection(path, timeout=30.0)
    try:
        conn.execute("BEGIN IMMEDIATE")
        coords_str = ",".join(state["last_coords"]) if state.get("last_coords") else None
        inc_json = json.dumps(state.get("increments", []), ensure_ascii=False)
        conn.execute("""
            INSERT INTO persona_grid_state
            (session_id, base_text, base_signature, current_signature, last_coords,
             last_aff, last_sec, last_aro, last_dom,
             last_narrative_aff, last_narrative_sec, last_narrative_aro, last_narrative_dom,
             last_narrative_signature, last_narrative_ts, last_attempt_ts, version, increments, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, datetime('now'))
            ON CONFLICT(session_id) DO UPDATE SET
                base_text = excluded.base_text,
                base_signature = excluded.base_signature,
                current_signature = excluded.current_signature,
                last_coords = excluded.last_coords,
                last_aff = excluded.last_aff,
                last_sec = excluded.last_sec,
                last_aro = excluded.last_aro,
                last_dom = excluded.last_dom,
                version = persona_grid_state.version + 1,
                updated_at = datetime('now')
        """, (
            target_session,
            state["base_text"],
            state["base_signature"],
            state["current_signature"],
            coords_str,
            state.get("last_aff", 0.0),
            state.get("last_sec", 0.0),
            state.get("last_aro", 0.0),
            state.get("last_dom", 0.0),
            state.get("last_narrative_aff", 0.0),
            state.get("last_narrative_sec", 0.0),
            state.get("last_narrative_aro", 0.0),
            state.get("last_narrative_dom", 0.0),
            state.get("last_narrative_signature", state["base_signature"]),
            state.get("last_narrative_ts", 0),
            state.get("last_attempt_ts", 0),
            inc_json,
        ))
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def record_narrator_attempt(
    session_id: str = "default",
    attempt_ts: int = 0,
    db_path: Optional[str] = None,
    request_token: Optional[str] = None,
) -> None:
    """Record an auxiliary call attempt timestamp and token to enforce failure backoff across restarts."""
    path = db_path or get_db_path()
    ensure_persona_schema(path)
    conn = get_connection(path, timeout=30.0)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("""
            UPDATE persona_grid_state
            SET last_attempt_ts = ?,
                active_request_token = COALESCE(?, active_request_token),
                version = version + 1,
                updated_at = datetime('now')
            WHERE session_id = ?
        """, (attempt_ts, request_token, session_id))
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def validate_narrative_increment(text: Any) -> bool:
    """Validate candidate narrative text from small model."""
    if not text or not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    if len(s) > MAX_INCREMENT_CHARS:
        return False
    # Check digits (both ascii and full-width)
    if re.search(r"[\d０-９]", s):
        return False
    # Check forbidden field names
    for term in FORBIDDEN_TERMS:
        if term in s:
            return False
    return True


def append_persona_increment(
    increment: str,
    new_signature: str,
    aff: float,
    sec: float,
    aro: float,
    dom: float,
    now_ts: int,
    session_id: str = "default",
    request_version: Optional[int] = None,
    request_snapshot_ts: Optional[int] = None,
    request_token: Optional[str] = None,
    db_path: Optional[str] = None,
    delta_threshold: Optional[float] = None,
    max_increments: Optional[int] = None,
) -> bool:
    """Validate and append an increment to DB under an atomic transaction, rejecting stale/superseded results."""
    token_str = request_token or "unknown"
    if not validate_narrative_increment(increment):
        logger.warning(
            "[persona_grid] narrator request rejected: category=non_compliant_text token=%s session=%s text=%r",
            token_str, session_id, increment
        )
        return False

    path = db_path or get_db_path()
    ensure_persona_schema(path)
    conn = get_connection(path, timeout=30.0)
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("""
            SELECT increments, last_narrative_ts, version, last_narrative_version, active_request_token,
                   current_signature, last_aff, last_sec, last_aro, last_dom
            FROM persona_grid_state WHERE session_id = ?
        """, (session_id,)).fetchone()

        if row:
            db_increments_raw = row[0]
            db_last_narrative_ts = row[1] or 0
            db_version = row[2] or 0
            db_last_narrative_ver = row[3] or 0
            db_active_token = row[4]
            db_current_sig = row[5]
            db_last_aff = row[6]
            db_last_sec = row[7]
            db_last_aro = row[8]
            db_last_dom = row[9]

            # 1. Superseded check: if a newer request has started and claimed ownership
            if request_token and db_active_token and db_active_token != request_token:
                logger.info(
                    "[persona_grid] narrator request rejected: category=superseded token=%s session=%s (active_token=%s)",
                    token_str, session_id, db_active_token
                )
                conn.execute("ROLLBACK")
                return False

            # 2. Version stale check: if a newer narrative has already landed
            if request_version is not None and db_last_narrative_ver > request_version:
                logger.info(
                    "[persona_grid] narrator request rejected: category=stale_version token=%s session=%s (db_ver=%s > req_ver=%s)",
                    token_str, session_id, db_last_narrative_ver, request_version
                )
                conn.execute("ROLLBACK")
                return False

            # 3. Snapshot timestamp check
            if request_snapshot_ts is not None and db_last_narrative_ts > request_snapshot_ts:
                logger.info(
                    "[persona_grid] narrator request rejected: category=stale_snapshot token=%s session=%s (db_ts=%s > req_ts=%s)",
                    token_str, session_id, db_last_narrative_ts, request_snapshot_ts
                )
                conn.execute("ROLLBACK")
                return False

            # 4. Subsequent turn state drift checks:
            # If subsequent turns updated scores or signatures before worker finished,
            # reject stale increment so we don't apply an obsolete narrative or trigger a false 6h cooldown.
            if db_current_sig and new_signature and db_current_sig != new_signature:
                logger.warning(
                    "[persona_grid] narrator request rejected: category=stale_signature token=%s session=%s (db_sig=%s != req_sig=%s)",
                    token_str, session_id, db_current_sig, new_signature
                )
                conn.execute("ROLLBACK")
                return False

            if db_last_aff is not None and aff is not None:
                d_aff = abs(float(db_last_aff) - float(aff))
                d_sec = abs(float(db_last_sec if db_last_sec is not None else 0.0) - float(sec if sec is not None else 0.0))
                d_aro = abs(float(db_last_aro if db_last_aro is not None else 0.0) - float(aro if aro is not None else 0.0))
                d_dom = abs(float(db_last_dom if db_last_dom is not None else 0.0) - float(dom if dom is not None else 0.0))
                max_drift = max(d_aff, d_sec, d_aro, d_dom)
                thresh = delta_threshold if delta_threshold is not None else AXIS_DELTA_THRESHOLD
                if max_drift >= thresh:
                    logger.warning(
                        "[persona_grid] narrator request rejected: category=stale_scores token=%s session=%s (max_drift=%.1f >= %.1f)",
                        token_str, session_id, max_drift, thresh
                    )
                    conn.execute("ROLLBACK")
                    return False

            increments = json.loads(db_increments_raw) if db_increments_raw else []
            increments.append(increment.strip())
            limit = max_increments if max_increments is not None else MAX_INCREMENTS
            if len(increments) > limit:
                increments = increments[-limit:]

            conn.execute("""
                UPDATE persona_grid_state
                SET increments = ?,
                    last_narrative_signature = ?,
                    last_narrative_ts = ?,
                    last_narrative_version = version + 1,
                    last_narrative_aff = ?,
                    last_narrative_sec = ?,
                    last_narrative_aro = ?,
                    last_narrative_dom = ?,
                    version = version + 1,
                    updated_at = datetime('now')
                WHERE session_id = ?
            """, (
                json.dumps(increments, ensure_ascii=False),
                new_signature,
                now_ts,
                aff,
                sec,
                aro,
                dom,
                session_id,
            ))
            conn.execute("COMMIT")
            logger.info(
                "[persona_grid] narrator request landed: category=success token=%s session=%s ver=%s",
                token_str, session_id, db_version + 1
            )
            return True
        else:
            increments = [increment.strip()]
            conn.execute("""
                INSERT INTO persona_grid_state
                (session_id, base_text, base_signature, current_signature, last_coords,
                 last_aff, last_sec, last_aro, last_dom,
                 last_narrative_aff, last_narrative_sec, last_narrative_aro, last_narrative_dom,
                 last_narrative_signature, last_narrative_ts, last_narrative_version, last_attempt_ts, active_request_token, version, increments, updated_at)
                VALUES (?, '', ?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, 1, ?, datetime('now'))
            """, (
                session_id,
                new_signature,
                new_signature,
                aff, sec, aro, dom,
                aff, sec, aro, dom,
                new_signature,
                now_ts,
                now_ts,
                request_token,
                json.dumps(increments, ensure_ascii=False),
            ))
            conn.execute("COMMIT")
            logger.info(
                "[persona_grid] narrator request landed initial row: category=success token=%s session=%s",
                token_str, session_id
            )
            return True
    except Exception as e:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        logger.warning(
            "[persona_grid] narrator request rejected: category=db_error token=%s session=%s error=%s",
            token_str, session_id, e
        )
        return False
    finally:
        conn.close()


_inflight_sessions: Dict[str, Tuple[str, float]] = {}
_inflight_lock = threading.Lock()
_request_counter = 0
_counter_lock = threading.Lock()


def _generate_request_token(session_id: str) -> str:
    global _request_counter
    with _counter_lock:
        _request_counter += 1
        cnt = _request_counter
    return f"{session_id}_{int(time.time() * 1000)}_{cnt}_{uuid.uuid4().hex[:6]}"


def _mark_inflight(session_id: str) -> Tuple[bool, Optional[str]]:
    """Mark session as having an inflight narrative request.
    
    Returns (acquired, request_token). If already inflight and not timed out, returns (False, None).
    If timed out, supersedes the previous task with a new request token.
    """
    with _inflight_lock:
        now = time.monotonic()
        if session_id in _inflight_sessions:
            old_token, started = _inflight_sessions[session_id]
            if now - started < INFLIGHT_TIMEOUT_SECONDS:
                return False, None
            logger.warning(
                "[persona_grid] inflight request %s for %s timed out after %.1fs, superseding",
                old_token, session_id, now - started,
            )
        token = _generate_request_token(session_id)
        _inflight_sessions[session_id] = (token, now)
        return True, token


def _clear_inflight(session_id: str, request_token: Optional[str] = None) -> bool:
    """Clear inflight state if token matches or token is omitted."""
    with _inflight_lock:
        if session_id not in _inflight_sessions:
            return False
        curr_token, _ = _inflight_sessions[session_id]
        if request_token is None or curr_token == request_token:
            _inflight_sessions.pop(session_id, None)
            return True
        logger.info(
            "[persona_grid] ignoring clear_inflight for superseded token %s (current is %s)",
            request_token, curr_token,
        )
        return False


def _is_inflight(session_id: str) -> bool:
    with _inflight_lock:
        if session_id not in _inflight_sessions:
            return False
        _, started = _inflight_sessions[session_id]
        if time.monotonic() - started >= INFLIGHT_TIMEOUT_SECONDS:
            _inflight_sessions.pop(session_id, None)
            return False
        return True


def should_trigger_narrative(
    state: Dict[str, Any],
    current_sig: str,
    aff: float,
    sec: float,
    aro: float,
    dom: float,
    now_ts: int,
    session_id: str = "default",
    ignore_cooldown: bool = False,
    delta_threshold: Optional[float] = None,
    backoff_seconds: Optional[int] = None,
    return_reason: bool = False,
    cooldown_seconds: Optional[int] = None,
    refresh_seconds: Optional[int] = None,
) -> Any:
    """Check if incremental narrative generation should be requested.

    Priority order:
    1. Inflight check (dedup; timeout prevents deadlock)
    2. Failure backoff check (attempts separated from success, backoff default 60s)
    3. First-time missing narrative (immediately requested, not blocked by 6h cooldown)
    4. 6-hour cooldown check (successful narrative cooldown)
    5. Triggers: 7-day staleness, or cumulative delta >= threshold (15.0) relative to last success snapshot
    6. Below threshold -> do not request
    """
    # 1. Inflight dedup
    if not ignore_cooldown and _is_inflight(session_id):
        if return_reason:
            return False, "inflight_pending"
        return False

    last_attempt = state.get("last_attempt_ts", 0) or 0
    last_success_ts = state.get("last_narrative_ts", 0) or 0
    increments = state.get("increments") or []
    has_success = bool(last_success_ts > 0 and len(increments) > 0)

    # 2. Failure backoff check
    backoff = backoff_seconds if backoff_seconds is not None else BACKOFF_SECONDS
    if not ignore_cooldown:
        if has_success and last_attempt > last_success_ts and (now_ts - last_attempt) < backoff:
            if return_reason:
                return False, "failure_backoff"
            return False
        elif not has_success and last_attempt > 0 and (now_ts - last_attempt) < backoff:
            if return_reason:
                return False, "failure_backoff"
            return False

    # 3. First-time missing narrative: trigger immediately (not blocked by cooldown)
    if not has_success:
        if return_reason:
            return True, "first_missing_narrative"
        return True

    # 4. Success cooldown
    cool_sec = cooldown_seconds if cooldown_seconds is not None else COOLDOWN_SECONDS
    if not ignore_cooldown and (now_ts - last_success_ts < cool_sec):
        if return_reason:
            return False, "in_cooldown"
        return False

    # 5. Refresh / change triggers
    # 5a. Stale interval
    ref_sec = refresh_seconds if refresh_seconds is not None else MAX_UPDATE_INTERVAL_SECONDS
    if (now_ts - last_success_ts) >= ref_sec:
        if return_reason:
            return True, "interval_expired_7d"
        return True

    # 5b. Cumulative delta relative to last successful generation snapshot
    threshold = delta_threshold if delta_threshold is not None else AXIS_DELTA_THRESHOLD
    last_aff = state.get("last_narrative_aff")
    last_sec = state.get("last_narrative_sec")
    last_aro = state.get("last_narrative_aro")
    last_dom = state.get("last_narrative_dom")
    if last_aff is None:
        last_aff = aff
    if last_sec is None:
        last_sec = sec
    if last_aro is None:
        last_aro = aro
    if last_dom is None:
        last_dom = dom

    d_aff = abs(aff - last_aff)
    d_sec = abs(sec - last_sec)
    d_aro = abs(aro - last_aro)
    d_dom = abs(dom - last_dom)
    max_delta = max(d_aff, d_sec, d_aro, d_dom)

    if max_delta >= threshold:
        if return_reason:
            return True, f"axis_delta_reached_{max_delta:.1f}"
        return True

    # 6. Below threshold -> do not request
    if return_reason:
        return False, "delta_below_threshold"
    return False


# ═══════════════════════════════════════════════════════════════════
# Small Model Auxiliary Call & Prompt
# ═══════════════════════════════════════════════════════════════════

def build_narrator_prompt(
    coords: Tuple[str, str, str, str],
    rifts: int = 0,
    stage: str = "STRANGER",
    days_since: int = 0,
    atmosphere: str = "neutral",
    user_first_line: str = "",
    aff: float = 0.0,
    sec: float = 0.0,
    aro: float = 0.0,
    dom: float = 0.0,
    last_narrative_aff: Optional[float] = None,
    last_narrative_sec: Optional[float] = None,
    last_narrative_aro: Optional[float] = None,
    last_narrative_dom: Optional[float] = None,
) -> str:
    """Build prompt for auxiliary slot persona_narrator.

    Supplies continuous 4-axis values, ranges, meanings, current snapshot,
    cumulative delta relative to last success generation snapshot, and user context.
    Strictly forbids numbers, system field names, dialogues, and fabricating historical grievances.
    """
    coord_str = "·".join(coords)
    mods = []
    if rifts >= 20:
        mods.append("心中存有未解心结")
    else:
        mods.append("心境坦然无结")

    if stage == "STRANGER" and days_since >= 60:
        mods.append("相识虽久但仍如初见")
    else:
        mods.append("相处阶段自然")

    if atmosphere == "cool":
        mods.append("当前气氛微凉疏离")
    else:
        mods.append("当前气氛正常")

    mod_str = "，".join(mods)
    user_part = f"对方刚才说：{user_first_line}\n" if user_first_line else ""

    if (last_narrative_aff is not None and last_narrative_sec is not None and
        last_narrative_aro is not None and last_narrative_dom is not None):
        d_aff = aff - last_narrative_aff
        d_sec = sec - last_narrative_sec
        d_aro = aro - last_narrative_aro
        d_dom = dom - last_narrative_dom
        delta_str = (
            f"相对上次成功生成的变化：亲近感 {d_aff:+.1f}，安全感 {d_sec:+.1f}，"
            f"唤醒度 {d_aro:+.1f}，掌控度 {d_dom:+.1f}"
        )
    else:
        delta_str = "相对上次生成：首次建立人设（无历史相对变化）"

    return (
        "你现在的四维内心情感状态如下：\n"
        f"- 亲近感：{aff:.1f}（范围 -100 到 +100，正值亲近依恋，负值疏离冷淡）\n"
        f"- 安全感：{sec:.1f}（范围 -100 到 +100，正值笃定踏实，负值悬心未定）\n"
        f"- 唤醒度：{aro:.1f}（范围 -100 到 +100，正值活跃热切，负值平静安稳）\n"
        f"- 掌控度：{dom:.1f}（范围 -100 到 +100，正值主动进取，负值收敛退守）\n"
        f"心境坐标：{coord_str}。\n"
        f"背景修饰：{mod_str}。\n"
        f"{delta_str}。\n"
        f"{user_part}"
        "请以第一人称写两到三句内心情感与状态叙述，表达此刻对对方的心境与细微状态变动。\n"
        "要求：\n"
        "1. 必须是第一人称内心感受人设，绝不要写成对白台词或向对方说的话。\n"
        "2. 严禁凭空捏造被伤害、被冷落等具体历史故事或因果。\n"
        "3. 全文严格禁止包含任何数字（包括阿拉伯数字与全角数字），严格禁止包含“好感”、“安全感”、“信任”、“裂痕”等字眼。\n"
        "4. 字数控制在100字左右。"
    )


def extract_final_content(response: Any) -> str:
    """Extract strictly final message content from an LLM response or string.
    Strips inline reasoning tags (e.g. <think>...</think>).
    Strictly ignores reasoning/reasoning_content fields. Returns empty string if no valid content.
    """
    if response is None:
        return ""
    if isinstance(response, str):
        raw = response
    elif isinstance(response, dict):
        if "choices" in response:
            choices = response.get("choices") or []
            msg = choices[0].get("message") if choices and isinstance(choices[0], dict) else None
            raw = msg.get("content") if isinstance(msg, dict) else (response.get("content") or "")
        else:
            raw = response.get("content") or ""
    else:
        # Object response (e.g. ChatCompletion or message object)
        choices = getattr(response, "choices", None)
        if choices and len(choices) > 0:
            first_choice = choices[0]
            msg = getattr(first_choice, "message", None) if not isinstance(first_choice, dict) else first_choice.get("message")
            raw = getattr(msg, "content", None) if msg and not isinstance(msg, dict) else (msg.get("content") if isinstance(msg, dict) else None)
        else:
            raw = getattr(response, "content", None)

    if not raw or not isinstance(raw, str):
        return ""

    content = raw.strip()
    if not content:
        return ""

    # Strip think/reasoning block patterns
    content = re.sub(r"<(think|thinking|reasoning|thought|思考|反思|推理|推敲)>.*?</\1>", "", content, flags=re.DOTALL | re.IGNORECASE)

    return content.strip()


def inspect_response(response: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Inspect LLM response for valid completion status, stripping reasoning and detecting failures.

    Returns:
        (clean_content, category, detail)
        - On success: (clean_content, None, None)
        - On abnormal completion status: (None, "abnormal_finish_reason", finish_reason)
        - On reasoning only (empty content): (None, "empty_response", "reasoning_only")
        - On empty response: (None, "empty_response", None)
    """
    if response is None:
        return None, "empty_response", None

    if isinstance(response, str):
        clean = extract_final_content(response)
        if clean:
            return clean, None, None
        if re.search(r"<(think|thinking|reasoning|thought|思考|反思|推理|推敲)>", response, flags=re.IGNORECASE):
            return None, "empty_response", "reasoning_only"
        return None, "empty_response", None

    # Handle dict or object
    choice = None
    choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
    if choices and len(choices) > 0:
        choice = choices[0]

    # 1. Check finish_reason
    finish_reason = None
    if choice is not None:
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else getattr(choice, "finish_reason", None)
    if not finish_reason:
        finish_reason = response.get("finish_reason") if isinstance(response, dict) else getattr(response, "finish_reason", None)

    # 2. Check message / tool_calls
    msg = None
    if choice is not None:
        msg = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
    if msg is None:
        msg = response

    tool_calls = None
    if isinstance(msg, dict):
        tool_calls = msg.get("tool_calls")
    else:
        tool_calls = getattr(msg, "tool_calls", None)

    if not tool_calls and choice is not None:
        tool_calls = choice.get("tool_calls") if isinstance(choice, dict) else getattr(choice, "tool_calls", None)

    fr_str = str(finish_reason).lower().strip() if finish_reason else None
    if (tool_calls and len(tool_calls) > 0) or fr_str == "tool_calls":
        return None, "abnormal_finish_reason", "tool_calls"

    if fr_str and fr_str not in ("stop", "none", "", "end_turn"):
        return None, "abnormal_finish_reason", fr_str

    # 3. Check raw content and reasoning
    raw_content = None
    if isinstance(msg, dict):
        raw_content = msg.get("content")
    else:
        raw_content = getattr(msg, "content", None)
    if raw_content is None:
        raw_content = response.get("content") if isinstance(response, dict) else getattr(response, "content", None)

    reasoning_val = None
    for fld in ("reasoning", "reasoning_content", "thought", "thinking"):
        val = msg.get(fld) if isinstance(msg, dict) else getattr(msg, fld, None)
        if val:
            reasoning_val = val
            break
    if not reasoning_val:
        for fld in ("reasoning", "reasoning_content", "thought", "thinking"):
            val = response.get(fld) if isinstance(response, dict) else getattr(response, fld, None)
            if val:
                reasoning_val = val
                break

    clean = extract_final_content(raw_content) if raw_content else ""
    if clean:
        return clean, None, None

    if reasoning_val or (raw_content and re.search(r"<(think|thinking|reasoning|thought|思考|反思|推理|推敲)>", str(raw_content), flags=re.IGNORECASE)):
        return None, "empty_response", "reasoning_only"

    return None, "empty_response", None


def _default_call_persona_narrator(prompt: str, request_token: Optional[str] = None) -> Optional[str]:
    """Default fallback caller for persona narrator (overridden by plugin host narrator caller)."""
    return None


persona_narrator_caller = _default_call_persona_narrator

_last_narrator_thread: Optional[threading.Thread] = None
_narrator_threads: List[threading.Thread] = []
_threads_lock = threading.Lock()
_narrator_cancel_event = threading.Event()


def cancel_all_narrators() -> None:
    """Signal all active narrator worker threads to cancel/abort."""
    _narrator_cancel_event.set()


def reset_narrator_cancel() -> None:
    """Reset the cancellation signal for narrator worker threads."""
    _narrator_cancel_event.clear()


def join_all_narrator_threads(timeout: float = 3.0) -> List[threading.Thread]:
    """Join all active narrator worker threads with a bounded timeout.
    Returns the list of threads that remain alive after the timeout.
    """
    with _threads_lock:
        threads = [th for th in _narrator_threads if th.is_alive()]
    deadline = time.monotonic() + timeout
    for th in threads:
        remaining = max(0.0, deadline - time.monotonic())
        if th.is_alive():
            th.join(timeout=remaining)
    with _threads_lock:
        still_alive = [th for th in _narrator_threads if th.is_alive()]
        return still_alive


def trigger_narrator_async(
    coords: Tuple[str, str, str, str],
    current_sig: str,
    aff: float,
    sec: float,
    aro: float,
    dom: float,
    rifts: int,
    stage: str,
    days_since: int,
    atmosphere: str,
    user_first_line: str,
    now_ts: int,
    session_id: str = "default",
    request_version: Optional[int] = None,
    last_narrative_aff: Optional[float] = None,
    last_narrative_sec: Optional[float] = None,
    last_narrative_aro: Optional[float] = None,
    last_narrative_dom: Optional[float] = None,
    db_path: Optional[str] = None,
    delta_threshold: Optional[float] = None,
) -> Optional[threading.Thread]:
    """Launch background daemon thread to query auxiliary model and persist increment."""
    global _last_narrator_thread

    acquired, token = _mark_inflight(session_id)
    if not acquired or token is None:
        logger.debug("[persona_grid] async narrator for %s already inflight, skipping", session_id)
        return None

    try:
        # Persist attempt timestamp and active token immediately
        record_narrator_attempt(
            session_id=session_id, attempt_ts=now_ts, db_path=db_path, request_token=token
        )

        def worker():
            try:
                if _narrator_cancel_event.is_set():
                    return
                prompt = build_narrator_prompt(
                    coords=coords,
                    rifts=rifts,
                    stage=stage,
                    days_since=days_since,
                    atmosphere=atmosphere,
                    user_first_line=user_first_line,
                    aff=aff,
                    sec=sec,
                    aro=aro,
                    dom=dom,
                    last_narrative_aff=last_narrative_aff,
                    last_narrative_sec=last_narrative_sec,
                    last_narrative_aro=last_narrative_aro,
                    last_narrative_dom=last_narrative_dom,
                )
                if _narrator_cancel_event.is_set():
                    return
                try:
                    import inspect
                    sig = inspect.signature(persona_narrator_caller)
                    if "request_token" in sig.parameters:
                        raw_text = persona_narrator_caller(prompt, request_token=token)
                    else:
                        raw_text = persona_narrator_caller(prompt)
                except Exception as call_err:
                    err_str = str(call_err)
                    cat = "timeout" if ("timeout" in err_str.lower() or "timed out" in err_str.lower()) else "model_exception"
                    logger.warning(
                        "[persona_grid] narrator request failed: category=%s token=%s session=%s error=%s",
                        cat, token, session_id, err_str,
                    )
                    return

                if _narrator_cancel_event.is_set():
                    return

                if raw_text is None:
                    wired = persona_narrator_caller is not _default_call_persona_narrator
                    logger.warning(
                        "[persona_grid] narrator request failed: category=empty_response token=%s session=%s caller_wired=%s",
                        token, session_id, wired,
                    )
                    return

                clean_text, cat, detail = inspect_response(raw_text)
                if not clean_text:
                    if cat == "abnormal_finish_reason":
                        logger.warning(
                            "[persona_grid] narrator request failed: category=abnormal_finish_reason finish_reason=%s token=%s session=%s",
                            detail, token, session_id,
                        )
                    elif detail == "reasoning_only":
                        logger.warning(
                            "[persona_grid] narrator request failed: category=empty_response reason=reasoning_only token=%s session=%s",
                            token, session_id,
                        )
                    else:
                        logger.warning(
                            "[persona_grid] narrator request failed: category=%s token=%s session=%s",
                            cat or "empty_response", token, session_id,
                        )
                    return

                append_persona_increment(
                    increment=clean_text,
                    new_signature=current_sig,
                    aff=aff,
                    sec=sec,
                    aro=aro,
                    dom=dom,
                    now_ts=now_ts,
                    session_id=session_id,
                    request_version=request_version,
                    request_snapshot_ts=now_ts,
                    request_token=token,
                    db_path=db_path,
                    delta_threshold=delta_threshold,
                )
            except Exception as e:
                logger.warning(
                    "[persona_grid] narrator worker uncaught error: category=worker_exception token=%s session=%s error=%s",
                    token, session_id, e,
                )
            finally:
                _clear_inflight(session_id, token)

        ctx = contextvars.copy_context()
        t = threading.Thread(
            target=ctx.run,
            args=(worker,),
            name=f"persona-narrator-{session_id}",
            daemon=True,
        )
        _last_narrator_thread = t
        with _threads_lock:
            _narrator_threads[:] = [th for th in _narrator_threads if th.is_alive()]
            _narrator_threads.append(t)
        t.start()
        return t
    except Exception:
        _clear_inflight(session_id, token)
        raise


# ═══════════════════════════════════════════════════════════════════
# High-level Entry Point
# ═══════════════════════════════════════════════════════════════════

def get_persona_injection(
    aff: float,
    sec: float,
    aro: float,
    dom: float,
    rifts: int = 0,
    stage: str = "STRANGER",
    days_since: int = 0,
    atmosphere: str = "neutral",
    user_message: Optional[str] = None,
    now: Optional[datetime] = None,
    db_path: Optional[str] = None,
    ignore_cooldown: bool = False,
    sync_narrator: bool = False,
    session_id: Optional[str] = None,
    delta_threshold: Optional[float] = None,
    backoff_seconds: Optional[int] = None,
    bands: Optional[Dict[str, float]] = None,
    axis_delta: Optional[float] = None,
    retry_seconds: Optional[int] = None,
    cooldown_hours: Optional[float] = None,
    refresh_days: Optional[float] = None,
    max_increments: Optional[int] = None,
    cooldown_seconds: Optional[int] = None,
    refresh_seconds: Optional[int] = None,
    coords: Optional[Tuple[str, str, str, str]] = None,
) -> Tuple[str, str]:
    """Get current persona signature and [我心里的状态] injection text for a session.

    Manages hysteresis, base persona text initialization, narrative triggers,
    and returns immediately without blocking.
    """
    if axis_delta is not None and delta_threshold is None:
        delta_threshold = axis_delta
    if retry_seconds is not None and backoff_seconds is None:
        backoff_seconds = int(retry_seconds)
    if cooldown_hours is not None and cooldown_seconds is None:
        cooldown_seconds = int(cooldown_hours * 3600)
    if refresh_days is not None and refresh_seconds is None:
        refresh_seconds = int(refresh_days * 86400)

    path = db_path or get_db_path()
    ensure_persona_schema(path)
    target_session = session_id or "default"
    state = load_persona_state(session_id=target_session, db_path=path)
    now_dt = now or datetime.now()
    now_ts = int(now_dt.timestamp())

    last_coords = state["last_coords"] if state else None
    if coords is None:
        coords = discretize_coordinates(aff, sec, aro, dom, last_coords, bands=bands)
    else:
        coords = tuple(coords)
    current_sig = compute_signature(coords, rifts, stage, days_since, atmosphere)

    user_first_line = ""
    if user_message:
        for line in user_message.splitlines():
            if line.strip():
                user_first_line = line.strip()
                break

    # First time: establish base persona text
    if not state:
        grid_text = PERSONA_GRID_TEXTS.get(coords)
        if grid_text is None:
            logger.warning("[persona_grid] failed to find grid text for coords: %s", coords)
            grid_text = PERSONA_GRID_TEXTS.get(("近", "稳", "静", "收"), "他在我就安心，各干各的也好，不用一直讲话。")
        mods = compute_modifiers(rifts, stage, days_since, atmosphere)
        base_text = grid_text + "".join(mods)
        state = {
            "session_id": target_session,
            "base_text": base_text,
            "base_signature": current_sig,
            "current_signature": current_sig,
            "last_coords": coords,
            "last_aff": aff,
            "last_sec": sec,
            "last_aro": aro,
            "last_dom": dom,
            "last_narrative_aff": aff,
            "last_narrative_sec": sec,
            "last_narrative_aro": aro,
            "last_narrative_dom": dom,
            "last_narrative_signature": None,
            "last_narrative_ts": 0,  # 0 indicates no successful narrative yet!
            "last_attempt_ts": 0,
            "version": 0,
            "increments": [],
        }
        save_persona_state(state, db_path=path, session_id=target_session)

        # First time without successful narrative: trigger immediately
        if sync_narrator:
            try:
                token = _generate_request_token(target_session)
                record_narrator_attempt(session_id=target_session, attempt_ts=now_ts, db_path=path, request_token=token)
                prompt = build_narrator_prompt(
                    coords=coords, rifts=rifts, stage=stage, days_since=days_since,
                    atmosphere=atmosphere, user_first_line=user_first_line,
                    aff=aff, sec=sec, aro=aro, dom=dom,
                    last_narrative_aff=None, last_narrative_sec=None,
                    last_narrative_aro=None, last_narrative_dom=None,
                )
                try:
                    import inspect
                    sig_inspect = inspect.signature(persona_narrator_caller)
                    if "request_token" in sig_inspect.parameters:
                        raw = persona_narrator_caller(prompt, request_token=token)
                    else:
                        raw = persona_narrator_caller(prompt)
                except Exception:
                    raw = None
                if raw:
                    clean, cat, detail = inspect_response(raw)
                    if clean:
                        append_persona_increment(
                            increment=clean, new_signature=current_sig,
                            aff=aff, sec=sec, aro=aro, dom=dom,
                            now_ts=now_ts, session_id=target_session,
                            request_version=0, request_snapshot_ts=now_ts,
                            request_token=token, db_path=path,
                            delta_threshold=delta_threshold,
                            max_increments=max_increments,
                        )
                        new_state = load_persona_state(session_id=target_session, db_path=path)
                        if new_state:
                            return current_sig, build_persona_injection_text(
                                base_text, new_state.get("increments", []),
                                current_signature=current_sig,
                                last_narrative_signature=new_state.get("last_narrative_signature"),
                                max_increments=max_increments,
                            )
            except Exception as e:
                logger.debug("[persona_grid] sync narrator error: %s", e)
        else:
            trigger_narrator_async(
                coords=coords, current_sig=current_sig,
                aff=aff, sec=sec, aro=aro, dom=dom,
                rifts=rifts, stage=stage, days_since=days_since, atmosphere=atmosphere,
                user_first_line=user_first_line, now_ts=now_ts,
                session_id=target_session, request_version=0,
                last_narrative_aff=None, last_narrative_sec=None,
                last_narrative_aro=None, last_narrative_dom=None,
                db_path=path,
                delta_threshold=delta_threshold,
            )

        return current_sig, build_persona_injection_text(
            base_text, [],
            current_signature=current_sig,
            last_narrative_signature=None,
            max_increments=max_increments,
        )

    # Subsequent turns:
    # Check narrative trigger before mutating state
    trigger_needed = should_trigger_narrative(
        state, current_sig, aff, sec, aro, dom, now_ts,
        session_id=target_session, ignore_cooldown=ignore_cooldown,
        delta_threshold=delta_threshold, backoff_seconds=backoff_seconds,
        cooldown_seconds=cooldown_seconds, refresh_seconds=refresh_seconds,
    )

    if current_sig == state.get("base_signature"):
        # Cache hit: signature identical, base_text remains byte-for-byte identical
        base_text = state["base_text"]
    else:
        # Cache miss: signature changed, recompute fixed segment for new coordinates & modifiers
        grid_text = PERSONA_GRID_TEXTS.get(coords)
        if grid_text is not None:
            mods = compute_modifiers(rifts, stage, days_since, atmosphere)
            base_text = grid_text + "".join(mods)
            state["base_text"] = base_text
            state["base_signature"] = current_sig
        else:
            logger.warning(
                "[persona_grid] failed to find grid text for coords: %s, falling back to previous base_text",
                coords,
            )
            base_text = state.get("base_text")
            if not base_text:
                base_text = PERSONA_GRID_TEXTS.get(("近", "稳", "静", "收"), "他在我就安心，各干各的也好，不用一直讲话。")
            state["base_text"] = base_text

    increments = state.get("increments", [])

    # Update current signature and coordinates in DB
    state["current_signature"] = current_sig
    state["last_coords"] = coords
    state["last_aff"] = aff
    state["last_sec"] = sec
    state["last_aro"] = aro
    state["last_dom"] = dom
    save_persona_state(state, db_path=path, session_id=target_session)

    # Narrative generation if triggered
    if trigger_needed:
        req_ver = state.get("version", 0)
        last_n_aff = state.get("last_narrative_aff")
        last_n_sec = state.get("last_narrative_sec")
        last_n_aro = state.get("last_narrative_aro")
        last_n_dom = state.get("last_narrative_dom")

        if sync_narrator:
            try:
                token = _generate_request_token(target_session)
                record_narrator_attempt(session_id=target_session, attempt_ts=now_ts, db_path=path, request_token=token)
                prompt = build_narrator_prompt(
                    coords=coords, rifts=rifts, stage=stage, days_since=days_since,
                    atmosphere=atmosphere, user_first_line=user_first_line,
                    aff=aff, sec=sec, aro=aro, dom=dom,
                    last_narrative_aff=last_n_aff, last_narrative_sec=last_n_sec,
                    last_narrative_aro=last_n_aro, last_narrative_dom=last_n_dom,
                )
                try:
                    import inspect
                    sig_inspect = inspect.signature(persona_narrator_caller)
                    if "request_token" in sig_inspect.parameters:
                        raw = persona_narrator_caller(prompt, request_token=token)
                    else:
                        raw = persona_narrator_caller(prompt)
                except Exception:
                    raw = None
                if raw:
                    clean, cat, detail = inspect_response(raw)
                    if clean:
                        append_persona_increment(
                            increment=clean, new_signature=current_sig,
                            aff=aff, sec=sec, aro=aro, dom=dom,
                            now_ts=now_ts, session_id=target_session,
                            request_version=req_ver, request_snapshot_ts=now_ts,
                            request_token=token, db_path=path,
                            delta_threshold=delta_threshold,
                            max_increments=max_increments,
                        )
                        new_state = load_persona_state(session_id=target_session, db_path=path)
                        if new_state:
                            increments = new_state.get("increments", [])
            except Exception as e:
                logger.debug("[persona_grid] sync narrator error: %s", e)
        else:
            trigger_narrator_async(
                coords=coords, current_sig=current_sig,
                aff=aff, sec=sec, aro=aro, dom=dom,
                rifts=rifts, stage=stage, days_since=days_since, atmosphere=atmosphere,
                user_first_line=user_first_line, now_ts=now_ts,
                session_id=target_session, request_version=req_ver,
                last_narrative_aff=last_n_aff, last_narrative_sec=last_n_sec,
                last_narrative_aro=last_n_aro, last_narrative_dom=last_n_dom,
                db_path=path,
                delta_threshold=delta_threshold,
            )

    return current_sig, build_persona_injection_text(
        base_text, increments,
        current_signature=current_sig,
        last_narrative_signature=state.get("last_narrative_signature"),
        max_increments=max_increments,
    )
