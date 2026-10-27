"""Friday Relationship Engine — Ackem-compatible L1 relationship FSM.

Features:
  - 3-stage FSM: STRANGER → FAMILIAR → INTIMATE
  - Trust system (0-100) with per-event deltas
  - Rift mechanism (hurtful → rift, apology → repair)
  - Affection momentum (EMA) driving atmosphere
  - Ice-break mechanic (sincere apology at low trust)
  - Modulation coefficients fed to emotion engine

References:
  Ackem src/main/engine/relationship.ts — 100% algorithm parity.
"""

from typing import Dict, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

# Trust deltas per event type
TRUST_DELTA: Dict[str, float] = {
    "praise": 1.5,
    "apology": 2.0,
    "vulnerable": 1.0,
    "tease": 0.8,
    "cold": -1.5,
    "hurtful": -3.0,
    "casual_chat": 0.0,
    "question": 0.0,
}

# Positive event types (for momentum + consecutive turns)
POSITIVE_TYPES = {"praise", "tease", "vulnerable", "apology"}
NEGATIVE_TYPES = {"cold", "hurtful"}

# Modulation coefficients
TRUST_MOD_MIN = 0.5
TRUST_MOD_MAX = 1.5
RIFT_MOD_MIN = 0.3
RIFT_MOD_DECAY_PER_RIFT = 0.15

# Stage weights
STAGE_WEIGHT_STRANGER = 0.8
STAGE_WEIGHT_FAMILIAR = 1.0
STAGE_WEIGHT_INTIMATE = 1.4

# Stage transitions
STAGE_WARMUP_TURNS = 10       # STRANGER → FAMILIAR
STAGE_INTIMATE_TRUST = 60      # FAMILIAR → INTIMATE (trust threshold)
STAGE_INTIMATE_EVENTS = 3      # FAMILIAR → INTIMATE (shared events)
STAGE_DOWNGRADE_RIFTS = 5      # INTIMATE → FAMILIAR
STAGE_DOWNGRADE_TRUST = 30     # INTIMATE → FAMILIAR (alt condition)
STAGE_DOWNGRADE_INTIMATE_RIFTS = 8   # INTIMATE → STRANGER (hard)
STAGE_DOWNGRADE_INTIMATE_TRUST = 15  # INTIMATE → STRANGER (hard)

# Atmosphere
ATMOSPHERE_WARM_THRESHOLD = 0.5
ATMOSPHERE_COOL_THRESHOLD = -0.3

# Ice break
ICE_BREAK_TRUST_THRESHOLD = 15.0
ICE_BREAK_SINCERITY_THRESHOLD = 0.7
ICE_BREAK_TRUST_BONUS = 3.0

# Momentum
MOMENTUM_ALPHA = 0.7  # EMA alpha for affection momentum

# External atmosphere
EXTERNAL_MOMENTUM_ALPHA = 0.95
EXTERNAL_WARM_THRESHOLD = 0.4
EXTERNAL_COOL_THRESHOLD = -0.2

# Rift mechanics
RIFT_HURTFUL_COOLDOWN = 2   # turns between rift increments
RIFT_REPAIR_POSITIVE_STREAK = 4  # consecutive positive turns to repair


# ═══════════════════════════════════════════════════════════════════
# Core Functions
# ═══════════════════════════════════════════════════════════════════

def trust_delta(event: Dict) -> float:
    """Return trust change for a given event type."""
    return TRUST_DELTA.get(event["type"], 0.0)


def sign_for_momentum(event: Dict) -> int:
    """Return +1 for positive, -1 for negative, 0 for neutral events."""
    if event["type"] in POSITIVE_TYPES:
        return 1
    if event["type"] in NEGATIVE_TYPES:
        return -1
    return 0


def compute_modulation(l1: Dict) -> Dict:
    """Compute modulation coefficients from relationship state.

    These are fed to the emotion engine's emotionStep().
    """
    trust = l1["trust"]
    rifts = l1["rifts"]
    stage = l1["stage"]

    # trustMod: linear interpolation from 0.5 to 1.5
    trust_mod = TRUST_MOD_MIN + (trust / 100.0) * (TRUST_MOD_MAX - TRUST_MOD_MIN)

    # riftMod: decays linearly per rift
    rift_mod = max(RIFT_MOD_MIN, 1.0 - rifts * RIFT_MOD_DECAY_PER_RIFT)

    # stageWeight
    if stage == "STRANGER":
        stage_weight = STAGE_WEIGHT_STRANGER
    elif stage == "INTIMATE":
        stage_weight = STAGE_WEIGHT_INTIMATE
    else:
        stage_weight = STAGE_WEIGHT_FAMILIAR

    return {
        "trustMod": trust_mod,
        "riftMod": rift_mod,
        "stageWeight": stage_weight,
        "atmosphere": l1.get("atmosphere", "neutral"),
    }


def evolve_stage(l1: Dict) -> str:
    """Determine the next relationship stage based on FSM rules."""
    stage = l1["stage"]
    trust = l1["trust"]
    rifts = l1["rifts"]
    consecutive = l1.get("consecutive_positive_turns", 0)
    shared = l1.get("shared_events_count", 0)

    if stage == "STRANGER":
        if consecutive >= STAGE_WARMUP_TURNS:
            return "FAMILIAR"
        return "STRANGER"

    if stage == "FAMILIAR":
        if trust >= STAGE_INTIMATE_TRUST and shared >= STAGE_INTIMATE_EVENTS:
            return "INTIMATE"
        if rifts >= STAGE_DOWNGRADE_INTIMATE_RIFTS or trust < STAGE_DOWNGRADE_INTIMATE_TRUST:
            return "STRANGER"
        return "FAMILIAR"

    if stage == "INTIMATE":
        if rifts >= STAGE_DOWNGRADE_RIFTS or trust < STAGE_DOWNGRADE_TRUST:
            return "FAMILIAR"
        return "INTIMATE"

    return stage


def update_relationship(event: Dict, prev: Dict) -> Dict:
    """Main update function — run once per turn.

    Args:
        event: {"type", "intensity", "sincerity", ...}
        prev:  Previous L1State dict

    Returns:
        Updated L1State dict
    """
    event_type = event["type"]

    # Skip extreme redline events
    if event_type == "extreme_redline":
        return {**prev}

    l1 = {**prev}  # shallow copy

    # ── Step 1: Trust update ──────────────────────────────────
    trust_change = trust_delta(event)
    l1["trust"] = max(0.0, min(100.0, l1["trust"] + trust_change))

    # ── Step 2: Ice break ─────────────────────────────────────
    ice_broken = False
    if (event_type == "apology" and
        l1["trust"] <= ICE_BREAK_TRUST_THRESHOLD and
        event.get("sincerity", 0) >= ICE_BREAK_SINCERITY_THRESHOLD):
        l1["trust"] += ICE_BREAK_TRUST_BONUS
        l1["atmosphere"] = "neutral"
        ice_broken = True

    # ── Step 3: Rifts ─────────────────────────────────────────
    # Increment rift on hurtful (with cooldown)
    if event_type == "hurtful" and l1.get("turns_since_last_rift", 0) >= RIFT_HURTFUL_COOLDOWN:
        l1["rifts"] += 1
        l1["turns_since_last_rift"] = 0
    else:
        l1["turns_since_last_rift"] = l1.get("turns_since_last_rift", 0) + 1

    # Repair rift on apology with positive streak
    if (event_type == "apology" and
        l1["rifts"] > 0 and
        l1.get("consecutive_positive_turns", 0) >= RIFT_REPAIR_POSITIVE_STREAK):
        l1["rifts"] = max(0, l1["rifts"] - 1)

    # ── Step 4: Consecutive positive turns ────────────────────
    if event_type in POSITIVE_TYPES:
        l1["consecutive_positive_turns"] = l1.get("consecutive_positive_turns", 0) + 1
    elif event_type in NEGATIVE_TYPES:
        l1["consecutive_positive_turns"] = 0

    # ── Step 5: Affection momentum (EMA) ──────────────────────
    sign = sign_for_momentum(event)
    intensity = event.get("intensity", 0.5)
    l1["affection_momentum"] = (
        MOMENTUM_ALPHA * l1.get("affection_momentum", 0) +
        (1 - MOMENTUM_ALPHA) * intensity * sign
    )

    # ── Step 6: Atmosphere ────────────────────────────────────
    if not ice_broken:
        momentum = l1["affection_momentum"]
        if momentum > ATMOSPHERE_WARM_THRESHOLD:
            l1["atmosphere"] = "warm"
        elif momentum < ATMOSPHERE_COOL_THRESHOLD:
            l1["atmosphere"] = "cool"
        else:
            l1["atmosphere"] = "neutral"

    # ── Step 7: Stage evolution ───────────────────────────────
    new_stage = evolve_stage(l1)
    if new_stage != l1["stage"]:
        l1["stage"] = new_stage
        # Count shared events on stage transitions
        if event_type not in ("casual_chat", "question"):
            l1["shared_events_count"] = l1.get("shared_events_count", 0) + 1

    # Ensure all fields exist
    l1.setdefault("shared_events_count", 0)
    l1.setdefault("consecutive_positive_turns", 0)
    l1.setdefault("turns_since_last_rift", 0)

    return l1


def update_external_atmosphere(
    sign: float,
    intensity: float,
    prev: Optional[Dict] = None,
) -> Dict:
    """Update the external atmosphere (long-term trend).

    Used for sensing the overall tone across many sessions.
    Independent EMA with alpha=0.95.
    """
    alpha = EXTERNAL_MOMENTUM_ALPHA
    prev_level = prev.get("level", 0.0) if prev else 0.0

    delta = (1 - alpha) * intensity * sign
    level = max(-1.0, min(1.0, prev_level * alpha + delta))

    if level > EXTERNAL_WARM_THRESHOLD:
        label = "warm"
    elif level < EXTERNAL_COOL_THRESHOLD:
        label = "cool"
    else:
        label = "neutral"

    return {"level": level, "label": label}


# ═══════════════════════════════════════════════════════════════════
# Stage transition detection (for Growth Unlock extension)
# ═══════════════════════════════════════════════════════════════════

def detect_stage_transition(old_stage: str, new_stage: str) -> Optional[str]:
    """Detect if a stage upgrade occurred. Returns direction or None."""
    ORDER = {"STRANGER": 0, "FAMILIAR": 1, "INTIMATE": 2}
    old_order = ORDER.get(old_stage, 0)
    new_order = ORDER.get(new_stage, 0)

    if new_order > old_order:
        return "upgrade"
    elif new_order < old_order:
        return "downgrade"
    return None
