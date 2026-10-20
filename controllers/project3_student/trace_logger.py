"""Development trace of robot motion and task events (JSON Lines).

Writes logs/run_<date>-<time>.jsonl with one JSON object per line:
  {"type": "meta", ...}      once, at start-up
  {"type": "sample", ...}    every SAMPLE_PERIOD_S: pose, wheel speeds,
                             proximity readings, current step and waypoint
  {"type": "event", ...}     task status changes and skill outcomes

save_frame() also writes annotated camera images to logs/frames/<run>/ during
searches (overview every 60 deg, each candidate, each confirmation).

Only the robot's own sensors and controller state are recorded. Ground truth
from the setup Supervisor is never read here; any comparison with the true
ball position is done afterwards by a separate analysis script.

Called from the Webots main loop only (it reads devices). Disable with
TRACE_ENABLED in project3_student.py; delete logs/*.jsonl before submission.
"""
import json
import math
import time
from pathlib import Path

SAMPLE_PERIOD_S = 0.1
FLUSH_EVERY = 50  # lines; buffered so logging does not slow the simulation


class TraceLogger:
    def __init__(self, robot, navigator, world_state, executive, meta=None,
                 log_dir=None):
        self.robot = robot
        self.navigator = navigator
        self.world_state = world_state
        self.executive = executive
        logs = Path(log_dir) if log_dir else (
            Path(__file__).resolve().parents[2] / "logs")
        logs.mkdir(exist_ok=True)
        self.path = logs / f"run_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        self._file = self.path.open("w", encoding="utf-8")
        self._pending = 0
        self._last_sample = -math.inf
        self._last_status = None
        self._trace_seen = 0
        self._nav_evidence = None
        self._frames_dir = None
        self._write({"type": "meta", "wall_clock": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "time_step_ms": int(robot.getBasicTimeStep()),
                     **(meta or {})})
        print(f"[TRACE] writing {self.path}")

    def update(self):
        now = self.robot.getTime()
        self._log_events(now)
        if now - self._last_sample >= SAMPLE_PERIOD_S:
            self._last_sample = now
            self._log_sample(now)

    def event(self, name, **details):
        """Record a one-off event from elsewhere in the controller."""
        self._write({"type": "event", "t": round(self.robot.getTime(), 3),
                     "event": name, **details})

    def save_frame(self, tag, bgr, detections):
        """Save an annotated camera frame under logs/frames/<run>/ (PNG)."""
        import cv2
        from perception import annotate
        if self._frames_dir is None:
            self._frames_dir = self.path.parent / "frames" / self.path.stem
            self._frames_dir.mkdir(parents=True, exist_ok=True)
        t = self.robot.getTime()
        name = f"t{t:07.2f}_{tag}.png"
        cv2.imwrite(str(self._frames_dir / name), annotate(bgr, detections))
        self.event("frame_saved", file=name,
                   detections=[d.summary() for d in detections])

    def close(self):
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    # ------------------------------------------------------------------
    def _log_sample(self, now):
        nav = self.navigator
        pose = nav.pose()
        waypoint = None
        if nav.active_location is not None and nav.waypoints:
            index = min(nav.waypoint_index, len(nav.waypoints) - 1)
            waypoint = nav.waypoints[index][0]
        step = self.world_state.current_step
        self._write({
            "type": "sample",
            "t": round(now, 3),
            "x": None if pose is None else round(pose[0], 4),
            "y": None if pose is None else round(pose[1], 4),
            "yaw_deg": None if pose is None else round(math.degrees(pose[2]), 2),
            "wheels": [round(nav.left_motor.getVelocity(), 3),
                       round(nav.right_motor.getVelocity(), 3)],
            "ps": [round(s.getValue(), 1) for s in nav.proximity],
            "step": step.get("action") if isinstance(step, dict) else None,
            "nav_phase": nav.phase if nav.active_location is not None else None,
            "waypoint": waypoint,
            "location": self.world_state.current_location,
        })

    def _log_events(self, now):
        status = self.world_state.execution_status
        if status != self._last_status:
            self._write({"type": "event", "t": round(now, 3),
                         "event": "task_status", "status": status,
                         "goal": self.world_state.current_goal})
            self._last_status = status

        trace = self.executive.trace
        if len(trace) < self._trace_seen:
            self._trace_seen = 0  # a new plan cleared the trace
        for entry in trace[self._trace_seen:]:
            if entry.get("status") != "RUNNING":
                self._trace_seen += 1
                self._write({"type": "event", "t": round(now, 3),
                             "event": "step_outcome", **_jsonable(entry)})
            else:
                break

        evidence = self.navigator.evidence
        if evidence is not None and evidence is not self._nav_evidence:
            self._nav_evidence = evidence
            self._write({"type": "event", "t": round(now, 3),
                         "event": "navigation_outcome", **_jsonable(evidence)})

    def _write(self, record):
        self._file.write(json.dumps(record) + "\n")
        self._pending += 1
        if self._pending >= FLUSH_EVERY:
            self._file.flush()
            self._pending = 0


def _jsonable(value):
    """Copy of value that json.dumps accepts (sets and objects become text)."""
    return json.loads(json.dumps(value, default=str))
