"""Shared state store for API/Worker process split.

The worker process writes state after each bg-fetch cycle. API processes in
APP_MODE=api read this file to serve /health without relying on local thread state.

Write is atomic: write to .tmp then rename, so readers never see a partial write.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_STATE_PATH = Path(os.environ.get("WORKER_STATE_PATH", "artifacts/worker_state.json"))


def write_state(state: dict[str, Any]) -> None:
    """Atomically overwrite the shared state file."""
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_PATH.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(state), encoding="utf-8")
        # On Windows, replace() can fail with PermissionError if another
        # process briefly holds the target open.  One retry is enough.
        try:
            tmp.replace(_STATE_PATH)
        except PermissionError:
            time.sleep(0.05)
            tmp.replace(_STATE_PATH)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def read_state() -> dict[str, Any] | None:
    """Return the latest state dict, or None if the file is missing or corrupt."""
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
