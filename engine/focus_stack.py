"""Focus Stack - Bounded context tracker for current tasks/topics."""

import sqlite3
import os
import time
import re
from typing import Dict, List, Optional

from .store import get_connection, get_db_path
MAX_STACK_DEPTH = 5
DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60

_CONTINUATION_PREFIXES = (
    "继续", "然后", "另外", "还有", "再", "顺便", "对", "是的", "不是", "改成",
    "加上", "补上", "这个", "那个", "刚才", "上面", "接着", "and ", "also ",
)
_NEW_TASK_PREFIXES = ("帮我", "请帮", "新任务", "另外一件事", "换个话题", "写一个", "实现一个")

def ensure_focus_schema(db_path: Optional[str] = None):
    conn = get_connection(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS focus_stack (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                priority INTEGER DEFAULT 1,
                parent_id INTEGER,
                next_action TEXT,
                created_at INTEGER,
                updated_at INTEGER,
                ttl INTEGER
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_focus_scope_status "
            "ON focus_stack(session_id, platform, status, updated_at DESC)"
        )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()

def _is_trivial(msg: str) -> bool:
    msg = msg.strip().lower()
    if len(msg) < 10:
        for w in ["hi", "hello", "hey", "早", "晚安", "在吗", "你好", "哈喽", "嗨", "嗯", "哦", "好"]:
            if w in msg:
                return True
    return False

def _is_completion(msg: str) -> bool:
    msg = msg.strip().lower()
    if len(msg) < 15:
        for w in ["完成", "做完了", "搞定", "解决", "done", "finished", "ok", "好的", "没问题", "算了", "取消", "不用了", "cancel", "nevermind"]:
            if w in msg:
                return True
    return False

def _detect_switch(msg: str, current_focus: Dict) -> bool:
    msg = msg.strip()
    lowered = msg.lower()
    if _is_completion(msg):
        return False
    if lowered.startswith(_NEW_TASK_PREFIXES):
        return True
    if lowered.startswith(_CONTINUATION_PREFIXES):
        return False

    # A long message is not automatically a topic switch. Compare meaningful
    # lexical anchors first; this keeps follow-up constraints on the same task.
    prior = " ".join((str(current_focus.get("title") or ""), str(current_focus.get("summary") or "")))
    token_re = r"[\u4e00-\u9fff]{2,6}|[A-Za-z][A-Za-z0-9_-]{2,}"
    old_tokens = {t.lower() for t in re.findall(token_re, prior)}
    new_tokens = {t.lower() for t in re.findall(token_re, msg)}
    if old_tokens & new_tokens:
        return False
    return len(msg) >= 24 and any(x in msg for x in ("做一个", "查一下", "看一下", "创建", "开发", "设计"))

def get_active_focus(session_id: str, platform: str, db_path: Optional[str] = None) -> Optional[Dict]:
    conn = get_connection(db_path)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM focus_stack WHERE session_id = ? AND platform = ? AND status = 'active' ORDER BY updated_at DESC, id DESC LIMIT 1",
            (session_id, platform)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def get_stack(session_id: str, platform: str, limit: int = 5, db_path: Optional[str] = None) -> List[Dict]:
    conn = get_connection(db_path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM focus_stack WHERE session_id = ? AND platform = ? AND status IN ('active', 'suspended', 'unresolved') ORDER BY updated_at DESC, id DESC LIMIT ?",
            (session_id, platform, limit)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

def _enforce_bound(conn, session_id: str, platform: str):
    rows = conn.execute(
        "SELECT id FROM focus_stack WHERE session_id = ? AND platform = ? AND status IN ('active', 'suspended', 'unresolved') ORDER BY updated_at DESC, id DESC LIMIT -1 OFFSET ?",
        (session_id, platform, MAX_STACK_DEPTH)
    ).fetchall()
    for r in rows:
        conn.execute("UPDATE focus_stack SET status = 'resolved' WHERE id = ?", (r[0],))

def pre_llm_focus_update(session_id: str, platform: str, user_message: str, db_path: Optional[str] = None) -> Optional[str]:
    if not session_id or not platform or not user_message:
        return None
        
    ensure_focus_schema(db_path)
    
    if _is_trivial(user_message):
        return _build_injection(session_id, platform, db_path=db_path)

    now = int(time.time())
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE focus_stack SET status='resolved', updated_at=? "
            "WHERE status IN ('active','suspended','unresolved') AND ttl IS NOT NULL AND ttl < ?",
            (now, now),
        )
        conn.commit()
        current = get_active_focus(session_id, platform, db_path=db_path)
        
        if _is_completion(user_message) and current:
            conn.execute("UPDATE focus_stack SET status = 'resolved', updated_at = ? WHERE id = ?", (now, current['id']))
            conn.commit()
            return _build_injection(session_id, platform, db_path=db_path)
            
        if current and _detect_switch(user_message, current):
            conn.execute("UPDATE focus_stack SET status = 'suspended', updated_at = ? WHERE id = ?", (now, current['id']))
            conn.commit()
            current = None

        if not current:
            title = user_message[:20] + "..." if len(user_message) > 20 else user_message
            conn.execute(
                "INSERT INTO focus_stack (session_id, platform, title, summary, status, created_at, updated_at, ttl) VALUES (?, ?, ?, ?, 'active', ?, ?, ?)",
                (session_id, platform, title, user_message[:100], now, now, now + DEFAULT_TTL_SECONDS)
            )
            _enforce_bound(conn, session_id, platform)
        else:
            summary = current.get("summary") or ""
            addition = user_message.strip()
            if addition not in summary:
                summary = (summary + "；" + addition).strip("；")[-240:]
            conn.execute(
                "UPDATE focus_stack SET summary=?, updated_at=?, ttl=? WHERE id=?",
                (summary, now, now + DEFAULT_TTL_SECONDS, current['id']),
            )

        # get_active_focus uses a short-lived read connection; publish writes first.
        conn.commit()
        try:
            active = get_active_focus(session_id, platform, db_path=db_path)
            if active:
                conn.execute("UPDATE channel_sessions SET last_topic = ? WHERE channel = ? AND session_id = ?", (active['title'], platform, session_id))
        except Exception:
            pass
            
        conn.commit()
    finally:
        conn.close()
        
    return _build_injection(session_id, platform, db_path=db_path)

def post_llm_focus_update(session_id: str, platform: str, assistant_message: str, db_path: Optional[str] = None):
    if not session_id or not platform or not assistant_message:
        return
        
    ensure_focus_schema(db_path)
    now = int(time.time())
    
    msg = assistant_message.lower()
    is_done = any(w in msg for w in ["完成", "搞定", "结束", "已解决", "i have finished", "done"])
    needs_input = any(w in msg for w in ["你需要", "请告诉我", "请提供", "你想要", "what do you think", "please provide"])
    
    conn = get_connection(db_path)
    try:
        current = get_active_focus(session_id, platform, db_path=db_path)
        if current:
            if is_done:
                conn.execute("UPDATE focus_stack SET status = 'resolved', next_action = 'completed', updated_at = ? WHERE id = ?", (now, current['id']))
            elif needs_input:
                conn.execute("UPDATE focus_stack SET next_action = 'waiting_for_user', updated_at = ? WHERE id = ?", (now, current['id']))
            else:
                conn.execute("UPDATE focus_stack SET next_action = 'in_progress', updated_at = ? WHERE id = ?", (now, current['id']))
            conn.commit()
    finally:
        conn.close()

def _build_injection(session_id: str, platform: str, db_path: Optional[str] = None) -> Optional[str]:
    stack = get_stack(session_id, platform, limit=3, db_path=db_path)
    if not stack:
        return None
        
    lines = ["[当前焦点]"]
    for item in stack:
        status_cn = {"active": "活跃", "suspended": "挂起", "unresolved": "未决"}.get(item["status"], item["status"])
        act = f" -> {item['next_action']}" if item['next_action'] else ""
        lines.append(f"- [{status_cn}] {item['title']}{act}")
        
    return "\n".join(lines)
