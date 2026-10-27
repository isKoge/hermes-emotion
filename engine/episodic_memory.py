"""Friday Episodic Memory — Ackem-compatible episodic memory + consolidation + contradiction.

Mirrors Ackem's memory subsystems:
  - Episodic memory: auto-summarize conversation segments every N turns
  - Memory consolidation: periodic LLM-based fact merging/dedup
  - Contradiction detection: find conflicting facts in memory
  - Active recall: random memory injection (15% probability when FAMILIAR+)
  - Memory echo: read memories to influence emotion (aff/sec modulation)
"""

import logging
import sqlite3
import os
import json
import random
import hashlib
import re
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
from .store import get_connection, get_db_path

# ═══════════════════════════════════════════════════════════════════
# Constants (from Ackem's ackemParams)
# ═══════════════════════════════════════════════════════════════════

EPISODE_INTERVAL_TURNS = 6
EPISODE_INTERVAL_TURNS_LOW = 10
EPISODE_EMOTION_INTENSITY_THRESHOLD = 0.5
EPISODE_SUMMARY_MAX_CHARS = 200
EPISODE_RETRIEVAL_MAX = 3
EPISODE_CHAR_BUDGET = 1200

CONSOLIDATION_INTERVAL_TURNS = 30
CONSOLIDATION_MIN_TURNS = 20
CONSOLIDATION_MAX_TURNS = 60
CONSOLIDATION_MIN_FACTS = 6
CONSOLIDATION_MAX_INSIGHTS = 4
CONSOLIDATION_INSIGHT_WEIGHT = 4.0

ACTIVE_RECALL_MIN_INTERVAL = 8
ACTIVE_RECALL_PROBABILITY = 0.15
ACTIVE_RECALL_MIN_STAGE = "FAMILIAR"

MEMORY_ECHO_CAP = 2.0
MEMORY_ECHO_AFF_WEIGHT = 0.5
MEMORY_ECHO_SEC_POSITIVE = 0.3
MEMORY_ECHO_SEC_NEGATIVE = -0.3
MEMOIR_TRUST_FLOOR = 25

CONTRADICTION_SIMILARITY_THRESHOLD = 0.35
CONTRADICTION_MIN_WEIGHT = 1.5


# ═══════════════════════════════════════════════════════════════════
# Schema — episodic tables
# ═══════════════════════════════════════════════════════════════════

_schema_ensured = False


def ensure_episodic_schema(db_path: Optional[str] = None):
    """Create episodic memory tables if they don't exist. Only runs once."""
    global _schema_ensured
    if _schema_ensured:
        return
    conn = None
    try:
        conn = get_connection(db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT NOT NULL,
                turn_range TEXT,
                emotion_snapshot TEXT,
                event_types TEXT,
                keywords TEXT,
                created_at TEXT,
                weight REAL DEFAULT 1.0
            );
            CREATE TABLE IF NOT EXISTS consolidation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insights_generated INTEGER,
                facts_processed INTEGER,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS active_recall_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id TEXT,
                recalled_at TEXT,
                context TEXT
            );
        """)
        conn.commit()
        _schema_ensured = True
    except Exception as e:
        logger.debug("[episodic] schema creation failed: %s", e)
    finally:
        if conn:
            conn.close()


# ═══════════════════════════════════════════════════════════════════
# Episodic Memory — auto-summarize conversation segments
# ═══════════════════════════════════════════════════════════════════

_turn_counter = 0
_last_episode_turn = 0
_recent_messages: List[Dict] = []  # {role, content, event_type}

_NOISE_MARKERS = (
    "[Friday 内部状态]", "[System note:", "[OUT-OF-BAND", "AGENTS.md",
    "tool_calls", "relationship_state", "emotion_engine", "TICK", "```",
)
_GENERIC_ASSISTANT = ("好哒老板", "马上帮你", "我先", "让我", "已经搞定")


def _clean_message(role: str, content: str) -> str:
    """Remove injected state, Markdown scaffolding, URLs and assistant boilerplate."""
    if not isinstance(content, str):
        return ""
    text = content
    for marker in _NOISE_MARKERS:
        if marker in text and role == "assistant":
            return ""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]+>|\[[^\]]{0,40}\]", " ", text)
    text = re.sub(r"[*_`#>|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if role == "assistant" and any(text.startswith(x) for x in _GENERIC_ASSISTANT):
        return ""
    return text[:500] if len(text) >= 8 else ""


def _is_quality_summary(summary: str) -> bool:
    if not summary or len(summary) < 18:
        return False
    if any(marker.lower() in summary.lower() for marker in _NOISE_MARKERS):
        return False
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]{2,}", summary)
    return len(set(tokens)) >= 3


def push_conversation_turn(role: str, content: str, event_type: str = "casual_chat"):
    """Push a conversation turn for episodic summarization."""
    global _turn_counter, _recent_messages
    cleaned = _clean_message(role, content)
    if not cleaned:
        return
    _turn_counter += 1
    _recent_messages.append({
        "role": role, "content": cleaned,
        "event_type": event_type,
    })
    # Keep last 20 turns
    if len(_recent_messages) > 20:
        _recent_messages = _recent_messages[-20:]


def should_create_episode(emo: Dict) -> bool:
    """Check if it's time to create an episode summary."""
    global _last_episode_turn
    turns_since = _turn_counter - _last_episode_turn

    # Adaptive interval based on emotion intensity
    intensity = abs(emo.get("aff", 0)) / 100
    interval = EPISODE_INTERVAL_TURNS if intensity > EPISODE_EMOTION_INTENSITY_THRESHOLD else EPISODE_INTERVAL_TURNS_LOW

    if turns_since >= interval and len(_recent_messages) >= 3:
        return True
    return False


def create_episode(emo: Dict, rel: Dict, db_path: Optional[str] = None) -> Optional[str]:
    """Create an episode summary from recent conversation.

    Returns the summary text. Called from post_llm_call when should_create_episode().
    """
    global _last_episode_turn

    if not _recent_messages:
        return None

    # User statements carry the episode topic; assistant text only supports it.
    keywords = []
    event_types = set()
    recent = _recent_messages[-8:]
    for msg in recent:
        event_types.add(msg.get("event_type", "casual_chat"))
        if msg.get("role") != "user":
            continue
        for word in re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_-]{2,}", msg["content"]):
            if word not in keywords and word.lower() not in {"system", "assistant", "friday", "internal"}:
                keywords.append(word)

    # Build summary (rule-based, Ackem-style)
    event_summary = ", ".join(sorted(event_types))
    event_types.add("__v2__")
    user_texts = [m["content"] for m in recent if m.get("role") == "user"]
    if not user_texts or len(keywords) < 2:
        _last_episode_turn = _turn_counter
        return None
    topic = "；".join(user_texts[-3:])
    summary = f"[{event_summary}] 用户讨论：{topic}"
    if len(summary) > EPISODE_SUMMARY_MAX_CHARS:
        summary = summary[:EPISODE_SUMMARY_MAX_CHARS].rstrip("，；、 ") + "。"
    if not _is_quality_summary(summary):
        _last_episode_turn = _turn_counter
        return None

    # Store
    conn = None
    try:
        conn = get_connection(db_path)
        ensure_episodic_schema(db_path)
        conn.execute("""
            INSERT INTO episodic_memory (summary, turn_range, emotion_snapshot,
                                          event_types, keywords, created_at, weight)
            VALUES (?, ?, ?, ?, ?, datetime('now'), 1.0)
        """, (
            summary,
            f"{_turn_counter - len(_recent_messages)}-{_turn_counter}",
            json.dumps({"aff": emo.get("aff", 0), "label": emo.get("primary_label", "")}),
            json.dumps(list(event_types)),
            json.dumps(keywords[:20], ensure_ascii=False),
        ))
        conn.commit()
    except Exception as e:
        logger.debug("[episodic] store failed: %s", e)
    finally:
        if conn:
            conn.close()

    _last_episode_turn = _turn_counter
    return summary


def get_recent_episodes(limit: int = EPISODE_RETRIEVAL_MAX, db_path: Optional[str] = None) -> List[Dict]:
    """Retrieve recent episodic memories."""
    conn = None
    try:
        conn = get_connection(db_path)
        ensure_episodic_schema(db_path)
        
        result = []
        offset = 0
        batch_size = 20
        while len(result) < limit:
            rows = conn.execute("""
                SELECT id, summary, emotion_snapshot, created_at, weight, event_types
                FROM episodic_memory
                ORDER BY id DESC LIMIT ? OFFSET ?
            """, (batch_size, offset)).fetchall()
            
            if not rows:
                break
                
            for r in rows:
                if _is_quality_summary(r[1]) and r[5] and "__v2__" in r[5]:
                    result.append({"id": r[0], "summary": r[1], "emotion": r[2], "created_at": r[3], "weight": r[4]})
                    if len(result) >= limit:
                        break
            offset += batch_size

        return result
    except Exception:
        return []
    finally:
        if conn:
            conn.close()


# ═══════════════════════════════════════════════════════════════════
# Active Recall — random memory injection (15% probability)
# ═══════════════════════════════════════════════════════════════════

_last_recall_turn = 0


def try_active_recall(
    stage: str,
    trust: float,
) -> Optional[str]:
    """Try to actively recall a memory for injection.

    Returns a memory hint for LLM injection, or None.
    Ackem: 15% probability per turn, only when FAMILIAR+, min 8-turn interval.
    """
    global _last_recall_turn

    # Gate checks
    stage_order = {"STRANGER": 0, "FAMILIAR": 1, "INTIMATE": 2}
    if stage_order.get(stage, 0) < stage_order.get(ACTIVE_RECALL_MIN_STAGE, 1):
        return None
    if trust < MEMOIR_TRUST_FLOOR:
        return None
    if _turn_counter - _last_recall_turn < ACTIVE_RECALL_MIN_INTERVAL:
        return None
    if random.random() > ACTIVE_RECALL_PROBABILITY:
        return None

    # Get recent episodes
    episodes = get_recent_episodes(15)  # fetch more to find unrecalled ones
    if not episodes:
        return None

    try:
        conn = get_connection()
        recalled_rows = conn.execute("SELECT fact_id FROM active_recall_log").fetchall()
        conn.close()
        recalled_ids = {r[0] for r in recalled_rows}
    except Exception:
        recalled_ids = set()

    unrecalled = [ep for ep in episodes if f"episode:{ep.get('id', '')}" not in recalled_ids]
    if not unrecalled:
        return None

    # Pick a random episode
    episode = random.choice(unrecalled)
    _last_recall_turn = _turn_counter

    try:
        conn = get_connection()
        conn.execute("INSERT INTO active_recall_log (fact_id,recalled_at,context) VALUES (?,datetime('now'),?)",
                     (f"episode:{episode.get('id', '')}", episode["summary"][:200]))
        conn.commit(); conn.close()
    except Exception:
        logger.debug("[episodic] active recall log failed", exc_info=True)

    return f"【可以自然提起的旧事】{episode['summary']}"


# ═══════════════════════════════════════════════════════════════════
# Memory Echo — influence emotion from memories
# ═══════════════════════════════════════════════════════════════════

def compute_memory_echo(
    recent_episodes: List[Dict],
    current_aff: float,
) -> Dict[str, float]:
    """Compute emotion modulation from memory echoes.

    Returns {"aff_delta": float, "sec_delta": float}.
    Positive memories boost aff/sec, negative ones reduce them.
    """
    if not recent_episodes:
        return {"aff_delta": 0, "sec_delta": 0}

    total_aff_delta = 0
    total_sec_delta = 0

    for ep in recent_episodes:
        try:
            emo_snapshot = json.loads(ep.get("emotion", "{}"))
            ep_aff = emo_snapshot.get("aff", 0)
        except Exception:
            ep_aff = 0

        # Echo: past emotions influence current state
        aff_echo = ep_aff * MEMORY_ECHO_AFF_WEIGHT * 0.01  # scale down
        total_aff_delta += max(-MEMORY_ECHO_CAP, min(MEMORY_ECHO_CAP, aff_echo))

        # Security: positive past → security boost, negative → security reduction
        if ep_aff > 0:
            total_sec_delta += MEMORY_ECHO_SEC_POSITIVE * 0.1
        elif ep_aff < 0:
            total_sec_delta += MEMORY_ECHO_SEC_NEGATIVE * 0.1

    return {
        "aff_delta": max(-MEMORY_ECHO_CAP, min(MEMORY_ECHO_CAP, total_aff_delta)),
        "sec_delta": max(-MEMORY_ECHO_CAP, min(MEMORY_ECHO_CAP, total_sec_delta)),
    }


# ═══════════════════════════════════════════════════════════════════
# Memory Consolidation — periodic fact merging (LLM-based)
# ═══════════════════════════════════════════════════════════════════

_last_consolidation_turn = 0


def should_consolidate() -> bool:
    """Check if it's time for memory consolidation."""
    global _last_consolidation_turn
    turns_since = _turn_counter - _last_consolidation_turn
    if turns_since < CONSOLIDATION_MIN_TURNS:
        return False
    if turns_since >= CONSOLIDATION_MAX_TURNS:
        return True
    # Meaningful density check
    meaningful_count = sum(1 for m in _recent_messages[-20:]
                          if m.get("event_type") in ("vulnerable", "praise", "apology"))
    density = meaningful_count / max(1, len(_recent_messages[-20:]))
    return turns_since >= CONSOLIDATION_INTERVAL_TURNS and density >= 0.4


def get_consolidation_context() -> Optional[str]:
    """Get recent episodes for consolidation prompt.

    Returns a context string for LLM consolidation, or None if not enough data.
    """
    global _last_consolidation_turn
    episodes = get_recent_episodes(CONSOLIDATION_MAX_INSIGHTS * 2)
    if len(episodes) < CONSOLIDATION_MIN_FACTS:
        return None

    _last_consolidation_turn = _turn_counter
    lines = ["需要整合的最近记忆："]
    for i, ep in enumerate(episodes):
        lines.append(f"{i+1}. {ep['summary']}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Contradiction Detection — find conflicting facts
# ═══════════════════════════════════════════════════════════════════

def _jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two strings."""
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def check_contradictions(episodes: List[Dict]) -> List[Tuple[Dict, Dict, float]]:
    """Check for contradictions among recent episodes.

    Returns list of (ep1, ep2, similarity) pairs that might contradict.
    """
    contradictions = []
    for i, ep1 in enumerate(episodes):
        for ep2 in episodes[i+1:]:
            sim = _jaccard_similarity(ep1["summary"], ep2["summary"])
            if sim >= CONTRADICTION_SIMILARITY_THRESHOLD:
                # Check if emotions are opposite
                try:
                    emo1 = json.loads(ep1.get("emotion", "{}"))
                    emo2 = json.loads(ep2.get("emotion", "{}"))
                    aff1 = emo1.get("aff", 0)
                    aff2 = emo2.get("aff", 0)
                    # Opposite valence = potential contradiction
                    if (aff1 > 20 and aff2 < -20) or (aff1 < -20 and aff2 > 20):
                        contradictions.append((ep1, ep2, sim))
                except Exception:
                    pass
    return contradictions


def get_contradiction_context() -> Optional[str]:
    """Check for contradictions and return context if found."""
    episodes = get_recent_episodes(20)
    if len(episodes) < 4:
        return None

    contradictions = check_contradictions(episodes)
    if not contradictions:
        return None

    lines = ["注意：检测到以下记忆可能存在矛盾："]
    for ep1, ep2, sim in contradictions[:3]:
        lines.append(f"- 「{ep1['summary'][:50]}」vs「{ep2['summary'][:50]}」(相似度{sim:.0%})")
    return "\n".join(lines)
