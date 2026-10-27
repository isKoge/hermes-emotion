"""Built-in emotion engine: rule-based L0 event detection, asymmetric decay, relationship tracking.

Pure logic only — no Hermes imports, no I/O, no network, no LLM calls.
All transitions are deterministic and idempotent for identical inputs.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Optional

# ── Event Types ──────────────────────────────────────────────────────
EVENT_PLAIN = "plain"
EVENT_PRAISE = "praise"
EVENT_APOLOGY = "apology"
EVENT_VULNERABILITY = "vulnerability"
EVENT_TEASING = "teasing"
EVENT_COLD = "cold"
EVENT_CRITICISM = "criticism"
EVENT_REUNION = "reunion"

ALL_EVENTS = (
    EVENT_PLAIN,
    EVENT_PRAISE,
    EVENT_APOLOGY,
    EVENT_VULNERABILITY,
    EVENT_TEASING,
    EVENT_COLD,
    EVENT_CRITICISM,
    EVENT_REUNION,
)

# ── Four-axis Deltas per Event ───────────────────────────────────────
# Axes: aff (好感: 亲/远), sec (安全感: 稳/悬), aro (唤醒: 热/静), dom (支配: 进/收)
EVENT_DELTAS: Dict[str, Dict[str, float]] = {
    EVENT_PLAIN: {"aff": 2.0, "sec": 1.0, "aro": 3.0, "dom": 0.0},
    EVENT_PRAISE: {"aff": 25.0, "sec": 22.0, "aro": 35.0, "dom": 12.0},
    EVENT_APOLOGY: {"aff": 15.0, "sec": 18.0, "aro": -15.0, "dom": -12.0},
    EVENT_VULNERABILITY: {"aff": 18.0, "sec": 12.0, "aro": -16.0, "dom": 12.0},
    EVENT_TEASING: {"aff": 12.0, "sec": 8.0, "aro": 16.0, "dom": 10.0},
    EVENT_COLD: {"aff": -25.0, "sec": -18.0, "aro": -16.0, "dom": -12.0},
    EVENT_CRITICISM: {"aff": -35.0, "sec": -25.0, "aro": 15.0, "dom": -22.0},
    EVENT_REUNION: {"aff": 20.0, "sec": -12.0, "aro": 18.0, "dom": -10.0},
}

# ── Relationship Stages ──────────────────────────────────────────────
STAGE_STRANGER = "STRANGER"
STAGE_FAMILIAR = "FAMILIAR"
STAGE_INTIMATE = "INTIMATE"

STAGE_ORDER: Dict[str, int] = {
    STAGE_STRANGER: 0,
    STAGE_FAMILIAR: 1,
    STAGE_INTIMATE: 2,
}

ORDER_TO_STAGE: Dict[int, str] = {
    0: STAGE_STRANGER,
    1: STAGE_FAMILIAR,
    2: STAGE_INTIMATE,
}

# ── Tuning Constants ─────────────────────────────────────────────────
DEFAULT_ENGINE_DECAY: float = 0.08
AFF_STICKINESS_FACTOR: float = 0.5  # Decay slower when aff > 0
AFF_REBOUND_FACTOR: float = 1.5     # Decay faster toward 0 when aff < 0

RIFTS_CRITICISM_DELTA: float = 15.0
RIFTS_APOLOGY_DELTA: float = -10.0

TRUST_DELTAS: Dict[str, float] = {
    EVENT_PLAIN: 0.5,
    EVENT_PRAISE: 4.0,
    EVENT_APOLOGY: 3.0,
    EVENT_VULNERABILITY: 5.0,
    EVENT_TEASING: 2.0,
    EVENT_REUNION: 3.0,
    EVENT_COLD: -2.0,
    EVENT_CRITICISM: -12.0,
}

# ── Regex Rules for L0 Event Detection ──────────────────────────────
_RE_CRITICISM = re.compile(
    r"(笨|蠢|没用|垃圾|差劲|讨厌你|烦死|闭嘴|滚开|去死|废物|恶心|神[经病]+|不要脸|要你何用|什么玩意|"
    r"你(根本|完全|总是|压根)?不懂|说了也?白说|跟你(说|讲)不(通|明白)|"
    r"shut up|idiot|hate you|useless|stupid)",
    re.I,
)
_RE_APOLOGY = re.compile(
    r"(对不起|抱歉|不好意思|是我不好|怪我|我的错|我错了|对不住|对唔住|原谅我|请原谅|别生气|别怪我|"
    r"sorry|apologize|my bad)",
    re.I,
)
_RE_REUNION = re.compile(
    r"(好久不见|好久没见|很久没见|好久没聊|很久没聊|好久没找你|隔了这么久|很久没联系|好久没来|好久没上线|"
    r"long time no see)",
    re.I,
)
_RE_VULNERABILITY = re.compile(
    r"(好难过|我难过|太难受|好难受|心里难受|我好累|太累了|好累啊|想哭|破防|压力好大|压力大|好委屈|委屈|"
    r"emo了|好烦躁|撑不住|好迷茫|好绝望|好孤独|好孤单)",
    re.I,
)
_RE_PRAISE = re.compile(
    r"(真棒|太棒|你好棒|厉害|太强|真强|好强|真聪明|太聪明|你真好|好喜欢你|喜欢你|爱你|夸夸|做得好|干得漂亮|"
    r"太神了|牛逼|好牛|优秀|真贴心|太懂我|多亏了你|谢谢你|多谢|感谢|"
    r"good job|great job|awesome|well done|thank you|thanks)",
    re.I,
)
_RE_TEASING = re.compile(
    r"(哈哈|逗你|开玩笑|笑死|夺笋|皮一下|幽默|hhhh|xswl|lol|haha)",
    re.I,
)
_RE_COLD = re.compile(
    r"(随便[你吧啦]?|随你[便吧]?|无所谓|行吧|懒得理你|没空理你|爱咋咋地|呵呵|算了|不说了|没事)|"
    r"^[\s\W]*(哦|嗯|ok|好的|收到|知道了|哦哦|嗯嗯)[\s\W]*$",
    re.I,
)


def detect_event(user_message: str) -> Optional[str]:
    """Identify event category from user message text using pure rule matching."""
    if not user_message:
        return None
    text = user_message.strip()
    if not text:
        return None

    is_teasing = bool(_RE_TEASING.search(text))

    # Explicit criticism (unless playful teasing marker present)
    if _RE_CRITICISM.search(text) and not is_teasing:
        return EVENT_CRITICISM
    if _RE_APOLOGY.search(text):
        return EVENT_APOLOGY
    if _RE_REUNION.search(text):
        return EVENT_REUNION
    if _RE_VULNERABILITY.search(text):
        return EVENT_VULNERABILITY
    if _RE_PRAISE.search(text):
        return EVENT_PRAISE
    if is_teasing:
        return EVENT_TEASING
    if _RE_COLD.search(text):
        return EVENT_COLD
    return EVENT_PLAIN if len(text) >= 2 else None


def apply_decay(scores: Dict[str, float], decay_rate: float) -> Dict[str, float]:
    """Pull scores toward 0 per turn.

    Affection decays slower when positive (stickiness), faster when negative (faster rebound).
    """
    if decay_rate <= 0:
        return dict(scores)
    rate = min(1.0, max(0.0, float(decay_rate)))
    out: Dict[str, float] = {}
    for key in ("aff", "sec", "aro", "dom"):
        val = float(scores.get(key, 0.0))
        if key == "aff":
            if val > 0:
                axis_rate = min(1.0, rate * AFF_STICKINESS_FACTOR)
            elif val < 0:
                axis_rate = min(1.0, rate * AFF_REBOUND_FACTOR)
            else:
                axis_rate = 0.0
        else:
            axis_rate = rate
        out[key] = val * (1.0 - axis_rate)
    return out


def apply_event_deltas(scores: Dict[str, float], event: Optional[str]) -> Dict[str, float]:
    """Apply four-axis deltas for the identified event and clamp to [-100.0, 100.0]."""
    out = dict(scores)
    if not event or event not in EVENT_DELTAS:
        return out
    deltas = EVENT_DELTAS[event]
    for key in ("aff", "sec", "aro", "dom"):
        current = float(out.get(key, 0.0))
        delta = float(deltas.get(key, 0.0))
        out[key] = max(-100.0, min(100.0, current + delta))
    return out


def compute_atmosphere(aff: float, aro: float) -> str:
    """Derive atmosphere (warm / neutral / cool) from aff and aro scores."""
    if aff >= 15.0 and aro >= 0.0:
        return "warm"
    if aff <= -15.0 or (aff < 0.0 and aro <= -10.0):
        return "cool"
    return "neutral"


def compute_stage(current_stage: str, turns: int, trust: float) -> str:
    """Advance relationship stage based on turns and trust. Only ascends, never demotes."""
    cur = str(current_stage or STAGE_STRANGER).upper()
    current_rank = STAGE_ORDER.get(cur, 0)

    target_rank = 0
    if turns >= 20 and trust >= 30.0:
        target_rank = 2  # INTIMATE
    elif turns >= 5 and trust >= 10.0:
        target_rank = 1  # FAMILIAR

    new_rank = max(current_rank, target_rank)
    return ORDER_TO_STAGE.get(new_rank, STAGE_STRANGER)


def compute_days_since(first_seen: float, now_ts: float) -> int:
    """Calculate days elapsed since first encounter."""
    if first_seen <= 0:
        return 0
    diff = max(0.0, now_ts - first_seen)
    return int(diff // 86400)


def step(
    state: Dict[str, Any],
    user_message: str,
    now_ts: float,
    *,
    engine_decay: float = DEFAULT_ENGINE_DECAY,
) -> Dict[str, Any]:
    """Pure, deterministic state transition function.

    Takes current state, user_message, and injected now_ts. Performs event recognition,
    asymmetric decay, delta updates, relationship advancement, and atmosphere calculation.
    Returns a new state dict without modifying the input state.
    """
    out = copy.deepcopy(state) if state else {}
    current_time = float(now_ts)

    # 1. Base fields
    turns = int(out.get("turns") or 0) + 1
    first_seen = float(out.get("first_seen") or 0.0)
    if first_seen <= 0:
        first_seen = current_time
    days_since = compute_days_since(first_seen, current_time)

    trust = float(out.get("trust") or 0.0)
    rifts = float(out.get("rifts") or 0.0)
    current_stage = str(out.get("stage") or STAGE_STRANGER)

    scores_dict = dict(out.get("scores") or {})
    for k in ("aff", "sec", "aro", "dom"):
        scores_dict[k] = float(scores_dict.get(k, 0.0))

    # 2. L0 Event recognition
    event = detect_event(user_message)

    # 3. Decay & Event Deltas
    scores_dict = apply_decay(scores_dict, engine_decay)
    scores_dict = apply_event_deltas(scores_dict, event)

    # 4. Relationship updates
    if event and event in TRUST_DELTAS:
        trust = max(0.0, min(100.0, trust + TRUST_DELTAS[event]))

    if event == EVENT_CRITICISM:
        rifts = max(0.0, min(100.0, rifts + RIFTS_CRITICISM_DELTA))
    elif event == EVENT_APOLOGY:
        rifts = max(0.0, min(100.0, rifts + RIFTS_APOLOGY_DELTA))

    stage = compute_stage(current_stage, turns, trust)

    # 5. Atmosphere
    atmosphere = compute_atmosphere(scores_dict["aff"], scores_dict["aro"])

    # 6. Assemble state
    out["scores"] = scores_dict
    out["trust"] = trust
    out["rifts"] = rifts
    out["stage"] = stage
    out["turns"] = turns
    out["first_seen"] = first_seen
    out["days_since"] = days_since
    out["atmosphere"] = atmosphere
    out["last_event"] = event
    out["updated_at"] = current_time

    return out
