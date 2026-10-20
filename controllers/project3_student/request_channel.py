"""Typed-input channel for development: send_goal.py writes a request file,
the controller polls it without blocking (same idea as the Lab 5 channel).

Typed input is for debugging only; the assessed input is speech.
"""
import json
import os
import uuid
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[2] / "runtime"
REQUEST_PATH = RUNTIME / "goal_request.json"


def write_request(text, path=REQUEST_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    request = {"request_id": uuid.uuid4().hex, "text": str(text).strip()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(request), encoding="utf-8")
    os.replace(tmp, path)   # atomic: the controller never reads half a file
    return request


class RequestChannel:
    def __init__(self, path=REQUEST_PATH):
        self.path = Path(path)
        self.seen = self._current_id()   # ignore a request left from last run

    def _current_id(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8")).get("request_id")
        except (OSError, ValueError):
            return None

    def poll(self):
        """Return new request text, or None."""
        try:
            request = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(request, dict) or request.get("request_id") == self.seen:
            return None
        self.seen = request.get("request_id")
        text = request.get("text")
        return text if isinstance(text, str) and text.strip() else None
