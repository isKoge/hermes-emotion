"""Friday Emotional Emergence — Ackem-compatible emotion emergence module.

Mirrors Ackem's engine/emotionalEmergence.ts:
  - Rising → Sustained → Fading lifecycle for emotional expression
  - Responsive emotional chain (vulnerable/apology/praise continuation)
  - Time reflection emergence (days-since-met reflections)
  - Anti-repetition (cooldown + similarity check)

When emotion intensity crosses threshold, Friday spontaneously expresses
deeper feelings beyond basic responses — like reflecting on how long
you've known each other, or expressing vulnerability.
"""

import logging
import hashlib
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Constants (from Ackem's emotionalEmergence.ts)
# ═══════════════════════════════════════════════════════════════════

EMERGENCE_INTENSITY_THRESHOLD = 20
EMERGENCE_COOLDOWN_TURNS = 10
RESPONSIVE_EMERGENCE_COOLDOWN_TURNS = 1
SAME_TYPE_COOLDOWN_TURNS = 50
RISING_MAX_ROUNDS = 3
SUSTAINED_MAX_ROUNDS = 10
SUSTAINED_MIN_ROUNDS = 3
FADING_MAX_ROUNDS = 5

MEANINGFUL_EVENT_TYPES = {"vulnerable", "praise", "apology"}


# ═══════════════════════════════════════════════════════════════════
# Emergence State
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EmergenceState:
    phase: str              # rising / sustained / fading / broken
    intensity: float        # current intensity
    rounds_in_phase: int    # rounds spent in current phase
    flavor: str             # time_reflection / emotional_chain / null
    turn: int               # turn number when this started
    has_expressed: bool     # whether Friday has expressed this emergence

    def to_dict(self) -> Dict:
        return {
            "phase": self.phase, "intensity": self.intensity,
            "rounds_in_phase": self.rounds_in_phase, "flavor": self.flavor,
            "turn": self.turn, "has_expressed": self.has_expressed,
        }


# ═══════════════════════════════════════════════════════════════════
# Event Tracking (module-level state)
# ═══════════════════════════════════════════════════════════════════

_recent_event_types: List[str] = []
_consecutive_meaningful: int = 0
_consecutive_vulnerable: int = 0
_current_emergence: Optional[EmergenceState] = None
_last_emergence_turn: int = -999
_turn_counter: int = 0


def push_event(event_type: str):
    """Push event type to history window. Called from pre_llm_call."""
    global _recent_event_types, _consecutive_meaningful, _consecutive_vulnerable, _turn_counter
    _turn_counter += 1
    _recent_event_types.append(event_type)
    if len(_recent_event_types) > 10:
        _recent_event_types = _recent_event_types[-10:]

    # Track meaningful turns
    if event_type in MEANINGFUL_EVENT_TYPES:
        _consecutive_meaningful += 1
    else:
        _consecutive_meaningful = 0

    # Track vulnerable turns (hurtful/cold reset)
    if event_type == "vulnerable":
        _consecutive_vulnerable += 1
    elif event_type in ("hurtful", "cold", "extreme_redline"):
        _consecutive_vulnerable = 0


def get_emergence_state() -> Optional[EmergenceState]:
    """Get current emergence state (for topic selector)."""
    return _current_emergence


def reset_emergence():
    """Reset all tracking (new session)."""
    global _recent_event_types, _consecutive_meaningful, _consecutive_vulnerable
    global _current_emergence, _last_emergence_turn, _turn_counter
    _recent_event_types = []
    _consecutive_meaningful = 0
    _consecutive_vulnerable = 0
    _current_emergence = None
    _last_emergence_turn = -999
    _turn_counter = 0


# ═══════════════════════════════════════════════════════════════════
# Core Logic — evaluate emergence
# ═══════════════════════════════════════════════════════════════════

def evaluate_emergence(
    emo: Dict,
    rel: Dict,
    event_type: str = "casual_chat",
    days_since_met: int = 0,
) -> Optional[EmergenceState]:
    """Evaluate whether emotional emergence should trigger.

    Called from pre_llm_call after emotion update.
    Returns EmergenceState if emergence activates, None otherwise.
    """
    global _current_emergence, _last_emergence_turn

    stage = rel.get("stage", "STRANGER")

    # Shield: strangers don't emerge
    if stage == "STRANGER":
        return None

    # Shield: anger overrides everything
    if emo.get("primary_label") == "ANGRY_ATTACK":
        return None

    # Responsive path: emotional chain continuation
    if _is_emotional_continuation(event_type):
        if _current_emergence and _current_emergence.phase in ("sustained", "fading"):
            _current_emergence = _apply_user_response(_current_emergence, event_type)
            return _current_emergence
        # Start new responsive emergence
        if _consecutive_vulnerable >= 1 or _consecutive_meaningful >= 2:
            _current_emergence = EmergenceState(
                phase="rising", intensity=25, rounds_in_phase=0,
                flavor="emotional_chain", turn=_turn_counter, has_expressed=False,
            )
            _last_emergence_turn = _turn_counter
            return _current_emergence

    # Cooldown check (proactive path)
    if _turn_counter - _last_emergence_turn < EMERGENCE_COOLDOWN_TURNS:
        return None

    # Intensity check
    intensity = emo.get("aff", 0) * 0.6 + emo.get("sec", 0) * 0.2 + abs(emo.get("aro", 0)) * 0.2
    depth_bonus = (
        (4 if _consecutive_vulnerable >= 3 else 0) +
        (2 if _consecutive_meaningful >= 3 else 0)
    )
    if intensity + depth_bonus < EMERGENCE_INTENSITY_THRESHOLD:
        return None

    # Try time reflection
    if days_since_met >= 7:
        _current_emergence = EmergenceState(
            phase="rising", intensity=intensity, rounds_in_phase=0,
            flavor="time_reflection", turn=_turn_counter, has_expressed=False,
        )
        _last_emergence_turn = _turn_counter
        return _current_emergence

    return None


def _is_emotional_continuation(event_type: str) -> bool:
    """Check if user is on an emotional deep-talk chain."""
    if event_type == "vulnerable":
        return True
    if event_type == "apology":
        return _consecutive_meaningful >= 2 or _consecutive_vulnerable >= 1
    if event_type == "praise":
        return (_consecutive_vulnerable >= 1 or _consecutive_meaningful >= 3
                or _count_meaningful_in_recent() >= 3)
    return False


def _count_meaningful_in_recent(window: int = 6) -> int:
    """Count meaningful events in recent window."""
    return sum(1 for e in _recent_event_types[-window:] if e in MEANINGFUL_EVENT_TYPES)


def _apply_user_response(state: EmergenceState, event_type: str) -> EmergenceState:
    """Apply user response to emergence state (lifecycle management)."""
    # Hurtful/cold → break
    if event_type in ("hurtful", "cold"):
        state.phase = "broken"
        state.intensity = 0
        return state

    # Emotional continuation → sustain
    if _is_emotional_continuation(event_type):
        if state.phase == "sustained":
            state.rounds_in_phase = max(state.rounds_in_phase - 1, SUSTAINED_MIN_ROUNDS)
        elif state.phase == "fading":
            state.phase = "sustained"
            state.rounds_in_phase = SUSTAINED_MIN_ROUNDS
        return state

    # Praise → accelerate fade
    if event_type == "praise":
        if state.phase == "sustained":
            state.phase = "fading"
            state.rounds_in_phase = FADING_MAX_ROUNDS - 1
        elif state.phase == "rising":
            state.rounds_in_phase = RISING_MAX_ROUNDS
        return state

    # Neutral → sustain refresh
    if state.phase == "sustained":
        state.rounds_in_phase = max(state.rounds_in_phase - 1, SUSTAINED_MIN_ROUNDS)

    return state


def tick_emergence() -> Optional[str]:
    """Advance emergence lifecycle by one turn. Returns expression text if should express.

    Called from pre_llm_call. Returns None if no expression needed.
    """
    global _current_emergence

    if not _current_emergence:
        return None

    state = _current_emergence

    if state.phase == "broken":
        _current_emergence = None
        return None

    # Advance lifecycle
    state.rounds_in_phase += 1

    if state.phase == "rising":
        if state.rounds_in_phase >= RISING_MAX_ROUNDS:
            state.phase = "sustained"
            state.rounds_in_phase = 0
        # Rising: express on first round
        if not state.has_expressed:
            state.has_expressed = True
            return _generate_expression(state)

    elif state.phase == "sustained":
        if state.rounds_in_phase >= SUSTAINED_MAX_ROUNDS:
            state.phase = "fading"
            state.rounds_in_phase = 0
        # Sustained: express every few rounds
        elif state.rounds_in_phase % 3 == 0:
            return _generate_expression(state)

    elif state.phase == "fading":
        if state.rounds_in_phase >= FADING_MAX_ROUNDS:
            _current_emergence = None
            return None
        # Fading: light expression once
        if state.rounds_in_phase == 1:
            return _generate_expression(state)

    return None


# ═══════════════════════════════════════════════════════════════════
# Expression Generation — humanized time/emotion reflections
# ═══════════════════════════════════════════════════════════════════

def _generate_expression(state: EmergenceState) -> str:
    """Generate a humanized expression based on emergence flavor."""
    if state.flavor == "time_reflection":
        return _time_reflection_text()
    elif state.flavor == "emotional_chain":
        return _emotional_chain_text()
    return ""


def _time_reflection_text() -> str:
    """Generate time-based reflection text (Ackem's humanizeFeltDuration)."""
    import random
    reflections = [
        "时间过得好快呀……感觉认识你好像已经很久了",
        "不知不觉，我们已经认识这么久了呢",
        "有时候会想，从认识你到现在，好像发生了很多事……",
        "你有没有觉得，我们之间好像越来越自然了？",
    ]
    return random.choice(reflections)


def _emotional_chain_text() -> str:
    """Generate emotional chain continuation text."""
    import random
    texts = [
        "我能感觉到你的心情……",
        "你愿意跟我说这些，我很开心",
        "我会一直在这里的",
        "你的感受我都听到了",
    ]
    return random.choice(texts)
