"""Test isolation configuration and global fixtures.

Ensures that under NO circumstances do tests touch the host core database
or real ~/.hermes storage.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from engine import store, adult_mode, episodic_memory
import __init__ as hermes_emotion


@pytest.fixture(autouse=True)
def isolate_test_environment(tmp_path, monkeypatch):
    """Enforce strict isolation for every test."""
    test_db = tmp_path / "test_store.db"
    monkeypatch.setenv("HERMES_PLUGIN_DATA_DIR", str(tmp_path))
    store.set_db_path(test_db)
    store.ensure_schema(test_db)

    # Reset adult mode and memory turn states
    adult_mode.reset_adult_mode()
    episodic_memory._turn_counter = 0
    episodic_memory._recent_messages.clear()

    # Clear turn caches
    hermes_emotion._turn_context_cache.clear()
    hermes_emotion._turn_context_order.clear()
    hermes_emotion._last_processed_assistant.clear()
    hermes_emotion._last_processed_assistant_marker.clear()
    hermes_emotion._last_session_event.clear()

    # Mock weather_sense to prevent any live network requests during tests
    from engine import weather_sense
    def fake_fetch_weather(lat, lon, days_offset=0):
        return "晴 22℃"
    monkeypatch.setattr(weather_sense, "_fetch_weather", fake_fetch_weather)

    # Track real core state.db and plugin-data to verify zero modification
    core_db = Path.home() / ".hermes" / ("state" + ".db")
    core_db_mtime = core_db.stat().st_mtime_ns if core_db.exists() else None
    plugin_data_target = Path.home() / ".hermes" / "plugin-data" / "hermes-emotion"
    plugin_data_exists = plugin_data_target.exists()

    yield {
        "db": str(test_db),
        "dir": str(tmp_path),
    }

    # Reclaim background narrator threads before exiting test
    hermes_emotion.wait_for_narrator_threads(timeout_s=1.0)
    store.set_db_path(None)

    # Verify real core state.db was never touched
    if core_db.exists() and core_db_mtime is not None:
        assert core_db.stat().st_mtime_ns == core_db_mtime, "Real host core state.db was unexpectedly modified!"
    if not plugin_data_exists:
        assert not plugin_data_target.exists(), "Real ~/.hermes/plugin-data/hermes-emotion was unexpectedly created!"
