"""Database schema initialization for hermes-emotion.

Delegates directly to store.ensure_schema(), guaranteeing state isolation.
"""

from __future__ import annotations

from .store import DB_PATH, ensure_schema, get_connection, get_db_path

__all__ = ["DB_PATH", "ensure_schema", "get_connection", "get_db_path"]
