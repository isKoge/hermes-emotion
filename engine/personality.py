"""Friday Personality System — Ackem-compatible personality presets + drift.

Mirrors Ackem's personalityPresets + P1-1~P1-6 drift system:
  - 5 personality presets (deredere/tsundere/kuudere/genki/dandere)
  - 5-dimension personality model (Tenderness/Independence/Sensitivity/Openness/Rationality)
  - Slow drift based on interaction patterns
  - Personality influences: expression style, message generation, topic selection
"""

import logging
import sqlite3
import os
import math
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
from .store import get_connection, get_db_path

# ═══════════════════════════════════════════════════════════════════
# 5 Personality Presets (from Ackem's personalityPresets.ts)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PersonalityPreset:
    id: str
    name: str
    T: float   # Tenderness [0,100]
    I: float   # Independence [0,100]
    S: float   # Sensitivity [0,100]
    O: float   # Openness [0,100]
    R: float   # Rationality [0,100]
    tags: List[str] = field(default_factory=list)
    description: str = ""


PRESETS = {
    "deredere": PersonalityPreset(
        id="deredere", name="甜蜜型", T=85, I=30, S=70, O=80, R=25,
        tags=["warm", "clingy"], description="黏人、甜蜜、热情、直白表达爱意",
    ),
    "tsundere": PersonalityPreset(
        id="tsundere", name="傲娇型", T=60, I=65, S=55, O=40, R=50,
        tags=["aloof_caring"], description="嘴上说不要，身体很诚实",
    ),
    "kuudere": PersonalityPreset(
        id="kuudere", name="冷娇型", T=45, I=80, S=30, O=35, R=75,
        tags=["cool_exterior"], description="表面冷淡，内心在乎",
    ),
    "genki": PersonalityPreset(
        id="genki", name="元气型", T=75, I=40, S=40, O=90, R=20,
        tags=["energetic"], description="开朗、活泼、充满活力",
    ),
    "dandere": PersonalityPreset(
        id="dandere", name="安静型", T=55, I=50, S=85, O=30, R=60,
        tags=["quiet_observer"], description="安静、细腻、善于观察",
    ),
}


# ═══════════════════════════════════════════════════════════════════
# 5-Dimension Personality State
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PersonalityState:
    T: float = 70.0   # Tenderness
    I: float = 50.0   # Independence
    S: float = 60.0   # Sensitivity
    O: float = 65.0   # Openness
    R: float = 40.0   # Rationality
    preset_id: str = "deredere"
    drift_enabled: bool = True

    def to_dict(self) -> Dict:
        return {"T": self.T, "I": self.I, "S": self.S, "O": self.O, "R": self.R,
                "preset_id": self.preset_id, "drift_enabled": self.drift_enabled}

    def style_hint(self) -> str:
        """Generate a style hint for LLM prompt injection."""
        preset = PRESETS.get(self.preset_id, PRESETS["deredere"])
        traits = []
        if self.T > 70:
            traits.append("温柔黏人")
        elif self.T < 40:
            traits.append("独立冷静")
        if self.S > 70:
            traits.append("敏感细腻")
        if self.O > 70:
            traits.append("开放直白")
        elif self.O < 40:
            traits.append("含蓄内敛")
        if self.R > 60:
            traits.append("理性分析")
        return f"人格风格：{preset.name}（{', '.join(traits)}）"


# ═══════════════════════════════════════════════════════════════════
# Personality Drift (Ackem's P1-1~P1-6)
# ═══════════════════════════════════════════════════════════════════

# Drift rates — very slow (Ackem's design: personality changes over months)
DRIFT_RATE = 0.1          # max change per meaningful event
DRIFT_DECAY = 0.005       # passive drift back toward preset
DIMENSION_CLAMP = (10, 95)  # never reach extremes


def apply_personality_drift(
    state: PersonalityState,
    event_type: str,
    relationship_stage: str,
) -> PersonalityState:
    """Apply personality drift based on interaction event.

    Called from post_llm_call. Very slow — changes accumulate over hundreds of turns.
    """
    if not state.drift_enabled:
        return state
    if relationship_stage == "STRANGER":
        return state  # no drift with strangers

    preset = PRESETS.get(state.preset_id, PRESETS["deredere"])

    # Event-based drift
    if event_type == "vulnerable":
        # User is vulnerable → increase tenderness, sensitivity
        state.T = _clamp(state.T + DRIFT_RATE * 1.5)
        state.S = _clamp(state.S + DRIFT_RATE * 0.8)

    elif event_type == "praise":
        # User praises → increase openness, tenderness
        state.O = _clamp(state.O + DRIFT_RATE * 1.2)
        state.T = _clamp(state.T + DRIFT_RATE * 0.5)

    elif event_type == "hurtful":
        # User is hurtful → increase independence, decrease openness
        state.I = _clamp(state.I + DRIFT_RATE * 1.0)
        state.O = _clamp(state.O - DRIFT_RATE * 0.8)

    elif event_type == "tease":
        # Playful interaction → increase openness
        state.O = _clamp(state.O + DRIFT_RATE * 0.6)

    elif event_type == "apology":
        # User apologizes → increase tenderness
        state.T = _clamp(state.T + DRIFT_RATE * 0.8)

    # Passive decay back toward preset (very slow)
    state.T = _decay_toward(state.T, preset.T, DRIFT_DECAY)
    state.I = _decay_toward(state.I, preset.I, DRIFT_DECAY)
    state.S = _decay_toward(state.S, preset.S, DRIFT_DECAY)
    state.O = _decay_toward(state.O, preset.O, DRIFT_DECAY)
    state.R = _decay_toward(state.R, preset.R, DRIFT_DECAY)

    return state


def _clamp(v: float) -> float:
    return max(DIMENSION_CLAMP[0], min(DIMENSION_CLAMP[1], v))


def _decay_toward(current: float, target: float, rate: float) -> float:
    """Slowly drift current value toward target."""
    diff = target - current
    return current + diff * rate


# ═══════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════

def load_personality(db_path: Optional[str] = None) -> PersonalityState:
    """Load personality state from state.db."""
    conn = None
    try:
        conn = get_connection(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS personality_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                T REAL, I REAL, S REAL, O REAL, R REAL,
                preset_id TEXT, drift_enabled INTEGER,
                updated_at TEXT
            )
        """)
        row = conn.execute("SELECT T,I,S,O,R,preset_id,drift_enabled FROM personality_state WHERE id=1").fetchone()
        if row:
            return PersonalityState(T=row[0], I=row[1], S=row[2], O=row[3], R=row[4],
                                    preset_id=row[5], drift_enabled=bool(row[6]))
    except Exception as e:
        logger.debug("[personality] load failed: %s", e)
    finally:
        if conn:
            conn.close()
    return PersonalityState()  # default: deredere


def save_personality(state: PersonalityState, db_path: Optional[str] = None):
    """Save personality state to state.db."""
    conn = None
    try:
        conn = get_connection(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS personality_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                T REAL, I REAL, S REAL, O REAL, R REAL,
                preset_id TEXT, drift_enabled INTEGER,
                updated_at TEXT
            )
        """)
        conn.execute("""
            INSERT OR REPLACE INTO personality_state
            (id, T, I, S, O, R, preset_id, drift_enabled, updated_at)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (state.T, state.I, state.S, state.O, state.R,
              state.preset_id, int(state.drift_enabled)))
        conn.commit()
    except Exception as e:
        logger.debug("[personality] save failed: %s", e)
    finally:
        if conn:
            conn.close()


def set_preset(preset_id: str) -> bool:
    """Change personality preset."""
    if preset_id not in PRESETS:
        return False
    preset = PRESETS[preset_id]
    state = PersonalityState(
        T=preset.T, I=preset.I, S=preset.S, O=preset.O, R=preset.R,
        preset_id=preset_id,
    )
    save_personality(state)
    return True
