"""Friday Adult Mode — Ackem-compatible NSFW interaction subsystem.

Features:
  - Adult state machine: NORMAL → FLIRTING → INTIMATE → AFTERCARE
  - User profile auto-detection (sexualDirectness, dominancePreference)
  - Desire stack (P2-1)
  - Adult intensity budget + negative lock
  - Memory privacy filter (normal / intimate / explicit)
  - Hidden personality ratio (gap_moe)

All constants from Ackem's ackemParams.ts + types.ts.
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Adult State Machine
# ═══════════════════════════════════════════════════════════════════

ADULT_STATES = ["NORMAL", "FLIRTING", "INTIMATE", "AFTERCARE"]

# Intensity budget (Ackem: 0-60)
BUDGET_MAX = 60
BUDGET_PER_FLIRT = 5
BUDGET_PER_DOMINANT = 8
BUDGET_PER_SUBMISSIVE = 8
BUDGET_PER_EXPLICIT = 15
BUDGET_DECAY_PER_TURN = 2     # passive decay
BUDGET_AFTERCARE_RESET = 30   # reset to this after aftercare

# Negative lock
NEGATIVE_LOCK_TURNS = 3       # turns to lock after negative event
VULNERABLE_LOCK_THRESHOLD = 2 # consecutive vulnerable turns to lock
REJECTED_COOLDOWN_TURNS = 5   # turns before re-approach after rejection


class AdultStateMachine:
    """Tracks and manages the adult interaction state."""

    def __init__(self, state: str = "NORMAL"):
        self.state = state
        self.intensity_budget = 0.0
        self.negative_lock_turns = 0
        self.consecutive_vulnerable = 0
        self.last_rejected_turn = -999

    def is_active(self) -> bool:
        return self.state != "NORMAL"

    def can_advance(self) -> bool:
        """Check if adult interaction is allowed right now."""
        if self.negative_lock_turns > 0:
            return False
        if self.intensity_budget >= BUDGET_MAX:
            return False
        return True

    def transition(self, event_type: str, intensity: float,
                   user_is_positive: bool) -> str:
        """Process an adult event and return the new state.

        Called from pre_llm_call after emotion analysis.
        """
        old_state = self.state

        # ── Negative lock countdown ────────────────────────────
        if self.negative_lock_turns > 0:
            self.negative_lock_turns -= 1
            logger.debug("[adult] negative lock: %d turns remaining",
                        self.negative_lock_turns)
            return self.state

        # ── User rejected / cold → cooldown ───────────────────
        if not user_is_positive and self.state in ("FLIRTING", "INTIMATE"):
            self.last_rejected_turn = 0  # will be incremented next turn
            self.state = "NORMAL"
            self.intensity_budget = max(0, self.intensity_budget - 10)
            logger.info("[adult] rejection: %s → NORMAL", old_state)
            return self.state

        # ── Process event ───────────────────────────────────────
        if event_type in ("adult_flirt",):
            self.intensity_budget += BUDGET_PER_FLIRT * intensity
            if self.state == "NORMAL":
                self.state = "FLIRTING"
            elif self.state == "AFTERCARE":
                self.state = "FLIRTING"
            self.consecutive_vulnerable = 0

        elif event_type in ("adult_dominant", "adult_submissive"):
            self.intensity_budget += BUDGET_PER_DOMINANT * intensity
            if self.state in ("NORMAL", "FLIRTING"):
                self.state = "INTIMATE"
            self.consecutive_vulnerable = 0

        elif event_type in ("adult_explicit",):
            self.intensity_budget += BUDGET_PER_EXPLICIT * intensity
            self.state = "INTIMATE"
            self.consecutive_vulnerable = 0
            if self.intensity_budget >= BUDGET_MAX:
                self.state = "AFTERCARE"

        elif event_type in ("vulnerable",):
            self.consecutive_vulnerable += 1
            if self.state in ("INTIMATE", "FLIRTING") and \
               self.consecutive_vulnerable >= VULNERABLE_LOCK_THRESHOLD:
                self.state = "AFTERCARE"

        # ── Budget cap + aftercare trigger ─────────────────────
        if self.intensity_budget >= BUDGET_MAX:
            self.state = "AFTERCARE"

        # ── Decay ──────────────────────────────────────────────
        self.intensity_budget = max(0, self.intensity_budget - BUDGET_DECAY_PER_TURN)

        # ── Aftercare → reset ──────────────────────────────────
        if self.state == "AFTERCARE" and self.intensity_budget < BUDGET_AFTERCARE_RESET:
            self.state = "NORMAL"
            self.intensity_budget = 0
            logger.info("[adult] aftercare complete → NORMAL")

        if old_state != self.state:
            logger.info("[adult] %s → %s (budget=%.0f)",
                       old_state, self.state, self.intensity_budget)

        return self.state

    def to_dict(self) -> Dict:
        return {
            "adultState": self.state,
            "adultBudget": self.intensity_budget,
            "adultNegativeLock": self.negative_lock_turns,
        }


# ═══════════════════════════════════════════════════════════════════
# User Profile Auto-Detection (Ackem LLM-based, simplified here)
# ═══════════════════════════════════════════════════════════════════

class UserProfile:
    """Auto-detected user intimacy preferences."""

    def __init__(self):
        self.dominant_archetype = "unknown"
        self.sexual_directness = 0.5   # 0-1
        self.dominance_preference = 0.0 # -1~1
        self.emotional_neediness = 0.5  # 0-1
        self.trust_trajectory = "building"
        self.last_updated = datetime.now().isoformat()

    def update_from_event(self, event_type: str, count: int):
        """Heuristic update based on user's event patterns."""
        if event_type in ("adult_dominant",):
            self.dominance_preference = min(
                1.0, self.dominance_preference + 0.1)
            self.dominant_archetype = "repressed_release"
        elif event_type in ("adult_submissive",):
            self.dominance_preference = max(
                -1.0, self.dominance_preference - 0.1)
            self.dominant_archetype = "romantic_submissive"
        elif event_type in ("adult_explicit",):
            self.sexual_directness = min(1.0, self.sexual_directness + 0.1)
        elif event_type in ("vulnerable",):
            self.emotional_neediness = min(1.0, self.emotional_neediness + 0.05)
            self.dominant_archetype = "emotional_seeker"
        elif event_type in ("adult_flirt",):
            self.dominant_archetype = "playful"

        self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "archetype": self.dominant_archetype,
            "directness": self.sexual_directness,
            "dominance": self.dominance_preference,
            "neediness": self.emotional_neediness,
        }


# ═══════════════════════════════════════════════════════════════════
# Desire Stack (P2-1)
# ═══════════════════════════════════════════════════════════════════

class Desire:
    def __init__(self, did: str, topic: str, category: str, urgency: float):
        self.id = did
        self.topic = topic
        self.category = category  # curiosity|concern|share|tease|suggest
        self.urgency = urgency    # 0-10
        self.status = "latent"

    def to_dict(self) -> Dict:
        return {"id": self.id, "topic": self.topic,
                "category": self.category, "urgency": self.urgency,
                "status": self.status}


class DesireStack:
    """Active desires — max 5 slots."""

    MAX_SLOTS = 5
    DECAY_PER_TURN = 0.3

    def __init__(self):
        self.slots: List[Optional[Desire]] = [None] * self.MAX_SLOTS

    def add(self, topic: str, category: str, urgency: float) -> bool:
        for i, slot in enumerate(self.slots):
            if slot is None:
                self.slots[i] = Desire(f"d_{i}", topic, category, urgency)
                return True
        return False

    def tick_decay(self):
        for slot in self.slots:
            if slot and slot.status == "active":
                slot.urgency = max(0, slot.urgency - self.DECAY_PER_TURN)
                if slot.urgency <= 0:
                    slot.status = "settled"

    def to_list(self) -> List[Dict]:
        return [s.to_dict() for s in self.slots if s]


# ═══════════════════════════════════════════════════════════════════
# Memory Privacy Filter (Ackem adultMode)
# ═══════════════════════════════════════════════════════════════════

def filter_facts_by_privacy(facts: List[Dict], adult_mode: bool = True) -> List[Dict]:
    """Filter facts by privacy level. When adultMode is True, show all."""
    if adult_mode:
        return facts
    return [f for f in facts
            if f.get("privacyLevel", "normal") == "normal"]


# ═══════════════════════════════════════════════════════════════════
# Global instances (lazy initialized)
# ═══════════════════════════════════════════════════════════════════

_adult_state = AdultStateMachine()
_user_profile = UserProfile()
_desire_stack = DesireStack()

ADULT_MODE_ENABLED = True  # 成人模式默认开启


def get_adult_state() -> AdultStateMachine:
    return _adult_state

def get_user_profile() -> UserProfile:
    return _user_profile

def get_desire_stack() -> DesireStack:
    return _desire_stack

def reset_adult_mode():
    global _adult_state, _user_profile, _desire_stack
    _adult_state = AdultStateMachine()
    _user_profile = UserProfile()
    _desire_stack = DesireStack()
