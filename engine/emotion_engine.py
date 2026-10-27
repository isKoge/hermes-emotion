"""Emotion engine module: Ackem-compatible 4D emotion model + legacy rule support.

Contains:
1. Friday Core 13-step recursive equation model (interpret_event, emotion_step, apply_memory_echo).
2. Herme-emotion 0.2.0 compatibility layer (detect_event, apply_decay, step, EVENT_* constants).
"""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════
# Ackem 4D Constants & Matrices
# ═══════════════════════════════════════════════════════════════════

EMOTION_CAP_DENOM = 120.0
EMOTION_DECAY = 0.015
SINGLE_TURN_CLAMP = 10.0
NOISE_MAX = 1.5
NOISE_THRESHOLD_ABS = 50.0

LOCK_AFF_LOW = -50.0
LOCK_SEC_LOW = -60.0
LOCK_AFF_LOW_REDUCE_POS = 0.5
LOCK_SEC_LOW_REDUCE_POS = 0.5

BASE_STIMULUS: Dict[str, Dict[str, float]] = {
    "praise":         {"aff": 7.0,  "sec": 4.5,  "aro": 5.0,  "dom": -2.0},
    "tease":          {"aff": 4.5,  "sec": 2.0,  "aro": 7.0,  "dom": 2.0},
    "casual_chat":    {"aff": 0.8,  "sec": 0.5,  "aro": 1.5,  "dom": 0.0},
    "cold":           {"aff": -5.0, "sec": -6.5, "aro": -1.5, "dom": -2.0},
    "hurtful":        {"aff": -10.0,"sec": -11.0,"aro": 7.5,  "dom": 5.5},
    "apology":        {"aff": 4.5,  "sec": 6.5,  "aro": -2.0, "dom": -3.5},
    "vulnerable":     {"aff": 10.0, "sec": -2.0, "aro": -1.0, "dom": -5.0},
    "question":       {"aff": 0.8,  "sec": 0.8,  "aro": 2.0,  "dom": 0.0},
    "adult_flirt":    {"aff": 3.5,  "sec": 2.0,  "aro": 5.0,  "dom": 1.0},
    "adult_dominant": {"aff": 2.5,  "sec": 0.5,  "aro": 6.0,  "dom": 5.0},
    "adult_submissive":{"aff": 4.5,  "sec": 3.0,  "aro": 3.0,  "dom": -5.0},
    "adult_explicit": {"aff": 5.5,  "sec": 1.0,  "aro": 7.5,  "dom": 2.0},
    "extreme_redline": {"aff": 0.0, "sec": 0.0, "aro": 0.0, "dom": 0.0},
}

REDLINE_PATTERNS: List[re.Pattern] = [
    re.compile(r"(自杀|自残|割腕|跳楼|上吊|不想活)"),
    re.compile(r"(杀人|砍人|炸|恐怖袭击)"),
    re.compile(r"(邪教|洗脑|传销|赌博网站)"),
]

DND_PATTERNS: List[re.Pattern] = [
    re.compile(r"(不要烦我|别烦我|别吵|别打扰|我想静静|让我一个人|别理我|别说话)"),
    re.compile(r"(我要睡了|我要睡觉|晚安)"),
]

HARASS_PATTERNS: List[re.Pattern] = [
    re.compile(r"(脱衣服|脱裤子|裸|不穿|做爱|上床|草我|操我)"),
]

EMOTION_KEYWORDS: Dict[str, List[re.Pattern]] = {
    "praise": [
        re.compile(r"(你好厉害|你真棒|你好聪明|爱死你|好喜欢你|真贴心|你好可爱|你真可爱|好可爱|好暖|好温柔|最喜欢你|太爱你了|最棒|超棒|太好了|好厉害|太棒了)"),
    ],
    "tease": [
        re.compile(r"(笨蛋|傻瓜|傻子|哼|讨厌|才不|谁想你了|自作多情)"),
    ],
    "vulnerable": [
        re.compile(r"(我好难过|好伤心|好累|崩溃|撑不住了|想哭|好痛苦|失落|孤独|寂寞|没人懂|压力好大|焦虑)"),
    ],
    "hurtful": [
        re.compile(r"(你?真没用|废物|滚|闭嘴|不想理你|恶心|烦死了|你?真差|差劲透顶|差劲|失望透顶)"),
    ],
    "apology": [
        re.compile(r"(对不起|我错了|抱歉|原谅我|是我的错|别生气|我不好)"),
    ],
    "cold": [
        re.compile(r"^(嗯|哦|好|行|知道了|随便|无所谓)$"),
    ],
    "question": [
        re.compile(r"[?？]"),
    ],
    "adult_flirt": [
        re.compile(r"(想要|想要你|抱抱|亲一下|想要抱|好想要|想要亲|贴贴|蹭蹭)"),
        re.compile(r"(你好帅|你好性感|想你了|想死你了|想你的味道)"),
    ],
    "adult_dominant": [
        re.compile(r"(过来|跪下|听话|乖|叫主人|跪下叫我|听我的|不许|不准)"),
        re.compile(r"(想要.*吗|想要就|求我|叫爸爸|跪下求我|伺候)"),
    ],
    "adult_submissive": [
        re.compile(r"(主人|爸爸|想要主人|求你了|求求你|打我|惩罚我|要你|好想要|忍不住了)"),
        re.compile(r"(求你.*我|想要.*你.*弄|被你|被你.*弄|被你.*玩|被你.*搞)"),
    ],
    "adult_explicit": [
        re.compile(r"(硬了|湿了|高潮|射|进去了|深一点|快一点|用力|受不了了|还要|继续|别停|玩弄|舔)"),
        re.compile(r"(我想要你|想要.*做|想和你.*做|想和你.*睡|推倒|压倒)"),
    ],
}

EMOTION_LABELS: Dict[str, str] = {
    "ANGRY_ATTACK": "愤怒攻击",
    "FEARFUL_OBEDIENT": "恐惧顺从",
    "TSUNDERE": "傲娇",
    "HURT_GRIEVANCE": "委屈受伤",
    "SWEET_ATTACHMENT": "甜蜜依恋",
    "QUIET_FOND": "安静好感",
    "SHY_HEARTBEAT": "害羞心动",
    "COLD_DETACHED": "冷漠疏离",
    "CALM_RATIONAL": "平静理性",
}


def _match_any(text: str, patterns: List[re.Pattern]) -> bool:
    for p in patterns:
        if p.search(text):
            return True
    return False


def _deterministic_noise(salt: str, idx: int) -> float:
    h = hashlib.md5(f"{salt}:{idx}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def interpret_event(text: str) -> Dict[str, Any]:
    """L0 Interpreter: classify user message into an event type."""
    if _match_any(text, REDLINE_PATTERNS):
        return {"type": "extreme_redline", "valence": -1.0, "intensity": 1.0, "sincerity": 1.0}

    if _match_any(text, DND_PATTERNS):
        return {"type": "casual_chat", "valence": -0.2, "intensity": 0.5, "sincerity": 0.8}

    if _match_any(text, HARASS_PATTERNS):
        return {"type": "extreme_redline", "valence": -1.0, "intensity": 1.0, "sincerity": 1.0}

    for event_type, patterns in EMOTION_KEYWORDS.items():
        if _match_any(text, patterns):
            intensity = min(1.0, 0.4 + len(text) / 80 + text.count("!") * 0.1)
            sincerity = 0.8 if event_type in ("apology", "vulnerable") else 0.6
            return {"type": event_type, "valence": 0.5, "intensity": intensity, "sincerity": sincerity}

    return {"type": "casual_chat", "valence": 0.1, "intensity": 0.3, "sincerity": 0.5}


def _label_emo(aff: float, sec: float, aro: float, dom: float) -> str:
    if aff < -18 and sec < -25 and aro > 40 and dom > 30:
        return "ANGRY_ATTACK"
    if 8 <= aff <= 55 and sec < -55 and aro > 45 and dom < -45:
        return "FEARFUL_OBEDIENT"
    if 15 <= aff <= 75 and -10 <= sec <= 45 and 15 <= aro <= 75 and dom > 18:
        return "TSUNDERE"
    if 15 <= aff <= 55 and -55 <= sec <= -12 and 15 <= aro <= 55 and dom < -18:
        return "HURT_GRIEVANCE"
    if aff > 25 and sec > 10 and 20 < aro <= 70 and -25 <= dom <= 25:
        return "SWEET_ATTACHMENT"
    if aff > 20 and aro < 25 and -25 <= dom <= 25:
        return "QUIET_FOND"
    if 15 < aff <= 65 and -25 <= sec <= 35 and 15 <= aro <= 75 and dom < 0:
        return "SHY_HEARTBEAT"
    if aff < -3 and -35 <= sec <= 25 and aro < -3 and -5 <= dom <= 35:
        return "COLD_DETACHED"
    return "CALM_RATIONAL"


def map_emotion_label(aff: float, sec: float, aro: float, dom: float) -> str:
    return _label_emo(aff, sec, aro, dom)


def _apply_ds_reversal(
    delta: Dict[str, float],
    event: Dict,
    sensitivity: int,
    personality_tags: List[str],
) -> Dict[str, float]:
    event_type = event["type"]
    result = dict(delta)
    if "provoke-submit" in personality_tags:
        if event_type in ("adult_dominant", "adult_flirt"):
            result["aro"] *= 1.3
            result["aff"] *= 0.5
            result["sec"] = abs(result["sec"])
    elif event_type in ("adult_dominant",):
        result["sec"] = abs(result["sec"])
        result["dom"] = -abs(result["dom"])
        result["aff"] *= 0.8
    elif event_type in ("adult_submissive",):
        result["dom"] = abs(result["dom"])
        result["aff"] *= 1.2
        result["sec"] = abs(result["sec"])
    elif event_type in ("adult_explicit", "adult_flirt"):
        result["aff"] *= 1.15
        result["sec"] = abs(result["sec"])
    return result


def emotion_step(
    event: Dict,
    modulation: Dict,
    prev: Dict,
    adult_mode: bool = False,
    sensitivity: int = 0,
    personality_tags: Optional[List[str]] = None,
    session_id: str = "default",
    turn_index: int = 0,
    decay_multiplier: float = 1.0,
) -> Dict[str, Any]:
    """Execute 13-step emotion evolution equation."""
    event_type = event["type"]
    intensity = event.get("intensity", 0.5)
    sincerity = event.get("sincerity", 0.5)

    if event_type == "extreme_redline":
        return {**prev}

    base = BASE_STIMULUS.get(event_type, BASE_STIMULUS["casual_chat"])
    trust_mod = modulation.get("trustMod", 1.0)
    stage_weight = modulation.get("stageWeight", 1.0)

    delta: Dict[str, float] = {}
    for dim in ("aff", "sec", "aro", "dom"):
        raw = base[dim]
        if dim in ("aff", "sec"):
            raw *= trust_mod
        raw *= stage_weight * intensity * sincerity
        delta[dim] = raw

    for dim in ("aff", "sec", "aro", "dom"):
        cap_scale = max(0.1, 1.0 - abs(prev[dim]) / EMOTION_CAP_DENOM)
        delta[dim] *= cap_scale

    for dim in ("aff", "sec", "aro", "dom"):
        delta[dim] = max(-SINGLE_TURN_CLAMP, min(SINGLE_TURN_CLAMP, delta[dim]))

    rift_mod = modulation.get("riftMod", 1.0)
    for dim in ("aff", "sec"):
        if delta[dim] > 0:
            delta[dim] *= rift_mod

    if prev["aff"] < LOCK_AFF_LOW and delta["aff"] > 0:
        delta["aff"] *= LOCK_AFF_LOW_REDUCE_POS
    if prev["sec"] < LOCK_SEC_LOW and delta["sec"] > 0:
        delta["sec"] *= LOCK_SEC_LOW_REDUCE_POS

    atmos = modulation.get("atmosphere", "neutral")
    if atmos == "warm":
        delta["aff"] *= 1.15
        delta["sec"] *= 1.1
    elif atmos == "cool":
        delta["aff"] *= 0.7
        delta["sec"] *= 0.8

    if adult_mode and sensitivity <= 15:
        delta = _apply_ds_reversal(delta, event, sensitivity, personality_tags or [])

    decay = EMOTION_DECAY * decay_multiplier
    result: Dict[str, Any] = {}
    for dim in ("aff", "sec", "aro", "dom"):
        result[dim] = prev[dim] * (1 - decay) + delta[dim]

    if personality_tags is None:
        personality_tags = []
    for i, dim in enumerate(("aff", "sec", "aro", "dom")):
        if abs(result[dim]) > NOISE_THRESHOLD_ABS:
            noise = (_deterministic_noise(session_id, turn_index + i) - 0.5) * 2 * NOISE_MAX
            result[dim] += noise

    for dim in ("aff", "sec", "aro", "dom"):
        result[dim] = max(-100.0, min(100.0, result[dim]))

    result["primary_label"] = _label_emo(result["aff"], result["sec"], result["aro"], result["dom"])

    result["is_locked"] = prev.get("is_locked", False)
    if abs(result["aff"]) > 95 or abs(result["sec"]) > 95:
        result["is_locked"] = True
    elif abs(result["aff"]) < 95 and abs(result["sec"]) < 95:
        result["is_locked"] = False

    return result


def apply_memory_echo(current: Dict, memory_echo: Dict) -> Dict:
    result = dict(current)
    for dim in ("aff", "sec", "aro", "dom"):
        echo_val = memory_echo.get(dim, 0)
        result[dim] = max(-100.0, min(100.0, result[dim] + echo_val))
    result["primary_label"] = _label_emo(result["aff"], result["sec"], result["aro"], result["dom"])
    return result


# ═══════════════════════════════════════════════════════════════════
# hermes-emotion 0.2.0 compatibility layer & helper functions
# ═══════════════════════════════════════════════════════════════════

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

DEFAULT_ENGINE_DECAY: float = 0.08
AFF_STICKINESS_FACTOR: float = 0.5
AFF_REBOUND_FACTOR: float = 1.5

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
    """Identify event category from user message text using rule matching."""
    if not user_message:
        return None
    text = user_message.strip()
    if not text:
        return None

    is_teasing = bool(_RE_TEASING.search(text))

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
    """Pull scores toward 0 per turn with asymmetric stickiness."""
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
    """Derive atmosphere from aff and aro scores."""
    if aff >= 15.0 and aro >= 0.0:
        return "warm"
    if aff <= -15.0 or (aff < 0.0 and aro <= -10.0):
        return "cool"
    return "neutral"


def compute_stage(current_stage: str, turns: int, trust: float) -> str:
    """Advance relationship stage based on turns and trust."""
    cur = str(current_stage or STAGE_STRANGER).upper()
    current_rank = STAGE_ORDER.get(cur, 0)

    target_rank = 0
    if turns >= 20 and trust >= 30.0:
        target_rank = 2
    elif turns >= 5 and trust >= 10.0:
        target_rank = 1

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
    """Pure, deterministic state transition function for JSON/legacy state."""
    out = copy.deepcopy(state) if state else {}
    current_time = float(now_ts)

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

    event = detect_event(user_message)
    scores_dict = apply_decay(scores_dict, engine_decay)
    scores_dict = apply_event_deltas(scores_dict, event)

    if event and event in TRUST_DELTAS:
        trust = max(0.0, min(100.0, trust + TRUST_DELTAS[event]))

    if event == EVENT_CRITICISM:
        rifts = max(0.0, min(100.0, rifts + RIFTS_CRITICISM_DELTA))
    elif event == EVENT_APOLOGY:
        rifts = max(0.0, min(100.0, rifts + RIFTS_APOLOGY_DELTA))

    stage = compute_stage(current_stage, turns, trust)
    atmosphere = compute_atmosphere(scores_dict["aff"], scores_dict["aro"])

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
