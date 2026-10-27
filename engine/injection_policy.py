"""Friday Injection Policy — Ackem-compatible injection slot resolution.

Mirrors Ackem's strategy/injectionPolicy.ts:
  - Proactive injection (when ProactiveGate allows)
  - Responsive injection (when user mentions special dates/emotional topics)
  - Special date priority override (high-priority dates bypass whisper gate)

Determines WHAT gets injected into the LLM prompt and WHEN.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class InjectionSlots:
    temporal: str       # proactive / responsive / none
    emergence: str      # proactive / responsive / none
    reason: str         # why this decision was made


@dataclass
class InjectionPolicyContext:
    proactive_level: str        # silent / whisper / casual / proactive
    silent: bool
    event_type: Optional[str] = None
    special_date_hit: Optional[Dict] = None
    consecutive_meaningful_turns: int = 0
    consecutive_vulnerable_turns: int = 0
    recent_event_types: list = None
    has_temporal_signal: bool = False

    def __post_init__(self):
        if self.recent_event_types is None:
            self.recent_event_types = []


USER_INITIATED_EVENTS = {"question", "recall", "vulnerable", "apology", "praise"}


def can_proactive_arbitrate(ctx: InjectionPolicyContext) -> bool:
    """Check if proactive injection is allowed."""
    if ctx.silent:
        return False
    if ctx.proactive_level in ("silent", "whisper"):
        return False
    return True


def is_user_initiated_temporal_interest(ctx: InjectionPolicyContext) -> bool:
    """Check if user is showing interest in temporal/special topics."""
    if ctx.has_temporal_signal:
        return True
    if ctx.special_date_hit and ctx.event_type in USER_INITIATED_EVENTS:
        return True
    if ctx.event_type == "vulnerable" and ctx.special_date_hit:
        return True
    return False


def resolve_injection_slots(ctx: InjectionPolicyContext) -> InjectionSlots:
    """Determine injection slots based on context.

    Returns what should be injected: temporal hints, emergence hints, or nothing.
    """
    # Proactive path: gate allows full injection
    if can_proactive_arbitrate(ctx):
        return InjectionSlots(
            temporal="proactive",
            emergence="proactive",
            reason="proactive_arbitrate",
        )

    # Responsive path: user-initiated temporal interest
    user_initiated = is_user_initiated_temporal_interest(ctx)

    if user_initiated and ctx.special_date_hit:
        priority = ctx.special_date_hit.get("priority", "normal")
        if priority != "low":
            return InjectionSlots(
                temporal="responsive",
                emergence=_responsive_emergence_slot(ctx),
                reason="responsive_special_date",
            )

    if user_initiated and ctx.has_temporal_signal:
        return InjectionSlots(
            temporal="responsive",
            emergence=_responsive_emergence_slot(ctx),
            reason="responsive_temporal_signal",
        )

    # Whisper + high-priority special date → proactive temporal
    if (not ctx.silent and ctx.proactive_level == "whisper"
            and ctx.special_date_hit and ctx.special_date_hit.get("priority") == "high"):
        return InjectionSlots(
            temporal="proactive",
            emergence="none",
            reason="whisper_high_priority_date",
        )

    # Silent but emotional chain → responsive emergence only
    responsive_emergence = _responsive_emergence_slot(ctx)
    if ctx.silent and responsive_emergence == "responsive":
        return InjectionSlots(
            temporal="none",
            emergence="responsive",
            reason="responsive_emotional_arc",
        )

    return InjectionSlots(temporal="none", emergence="none", reason="blocked")


def _responsive_emergence_slot(ctx: InjectionPolicyContext) -> str:
    """Check if emotional continuation warrants responsive emergence."""
    if not ctx.event_type:
        return "none"
    if ctx.event_type == "vulnerable":
        return "responsive"
    if ctx.event_type == "apology" and ctx.consecutive_meaningful_turns >= 2:
        return "responsive"
    if ctx.event_type == "praise" and (
        ctx.consecutive_vulnerable_turns >= 1 or ctx.consecutive_meaningful_turns >= 3
    ):
        return "responsive"
    return "none"
