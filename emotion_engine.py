"""Top-level shim re-exporting engine.emotion_engine for backward compatibility."""

from __future__ import annotations

import engine.emotion_engine as _mod
import sys

# Copy all attributes including internal helpers used in tests
for _k, _v in _mod.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
