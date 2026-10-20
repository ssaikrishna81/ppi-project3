"""Bounded, non-blocking camera search for an object at the current location.

Search policy (finite by construction)
--------------------------------------
For the robot's current location, visit each viewpoint listed in
config/search_viewpoints.json in order (the pad first, then a second spot
that sees around furniture). At each viewpoint:

  SCAN     rotate on the spot through one full turn (360 deg, measured by
           summing Compass heading changes), analysing every camera frame.
  CENTER   when a frame shows a usable target candidate, stop and turn
           towards it (at most CENTER_MAX_S) so it is away from the edges.
  CONFIRM  hold still for CONFIRM_FRAMES frames. FOUND needs at least
           CONFIRM_REQUIRED of them to show the target within
           CONFIRM_BEARING_DEG of the same direction. Otherwise that
           direction is marked rejected and the scan continues.

The search stops at the first FOUND. Its outcome is:
  FOUND      confirmed positive evidence (above).
  UNCERTAIN  no confirmation, and either a target-like blob was seen but
             could not be confirmed (too small, failed confirmation), or
             the policy could not be completed (viewpoint unreachable,
             scan stalled, time budget used up).
  NOT_FOUND  every viewpoint was scanned completely and nothing target-like
             remained unexplained. It means "not seen from these
             viewpoints", never "absent from unsearched places".

A FOUND ball may be outside the current room (seen through a doorway); its
estimated position and room are reported. Unconfirmed blobs whose size puts
them outside the current room are "far sightings": reported, but they do not
make the current room UNCERTAIN.

A finished search is SUCCEEDED (the outcome is the result). FAILED only
means the camera or pose was unusable.
"""
import json
import math
from pathlib import Path

from perception import matches

RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"

FOUND = "FOUND"
NOT_FOUND = "NOT_FOUND"
UNCERTAIN = "UNCERTAIN"
OBSERVED = "OBSERVED"           # observe action: no target, list of objects

# Rotation: wheels at +/-1.2 rad/s turn the e-puck at about 0.9 rad/s
# (0.02 m wheels, 0.052 m axle), one full turn in about 7 s, i.e. about
# 1.7 deg per 32 ms frame, so a 2 m-away ball (32 px) stays in view for
# many frames.
SCAN_WHEEL_SPEED = 1.2
FULL_TURN_RAD = 2.0 * math.pi
SCAN_STALL_S = 3.0               # no heading change for this long -> stalled
SCAN_STALL_RAD = 0.1

CANDIDATE_MIN_CONFIDENCE = 0.35  # weaker blobs are noted, not chased
CENTER_TOLERANCE_DEG = 4.0
CENTER_MAX_S = 3.0
CENTER_GAIN = 3.0
SETTLE_FRAMES = 2
CONFIRM_FRAMES = 5
CONFIRM_REQUIRED = 4
CONFIRM_BEARING_DEG = 6.0
REJECTED_BEARING_DEG = 8.0       # ignore a direction that failed confirmation

VIEWPOINT_REACHED_M = 0.30       # already close enough: scan here
SEARCH_BUDGET_S = 160.0          # 3 viewpoints; executive backup limit is 200 s
OVERVIEW_EVERY_RAD = math.radians(60)

CONFIG = Path(__file__).resolve().parents[2] / "config" / "search_viewpoints.json"


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def load_search_config(path=CONFIG):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def room_of(x, y, rooms):
    for name, (xmin, xmax, ymin, ymax) in rooms.items():
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return name
    return "central_area"


class SearchController:
    def __init__(self, navigator, perception, world_state, config=None,
                 frame_saver=None):
        self.navigator = navigator
        self.perception = perception
        self.world_state = world_state
        config = config or load_search_config()
        self.viewpoints = config["viewpoints"]
        self.rooms = config["rooms"]
        # frame_saver(tag, bgr, detections) stores debug images (optional).
        self.frame_saver = frame_saver
        self.evidence = None
        self.outcome = None
        self.phase = "IDLE"

    # ------------------------------------------------------------ public
    def start(self, object_type, colour=None):
        return self._start((object_type, colour), observe=False)

    def start_observe(self):
        """One 360 deg scan where the robot stands, recording every object.

        Used by the 'observe' action: no target, no moving to other
        viewpoints, outcome OBSERVED with the list of objects seen.
        """
        return self._start((None, None), observe=True)

    def _start(self, target, observe):
        self.navigator.cancel()
        self.target = target
        self.observe = observe
        self.observations = {}
        self.started_at = self.navigator.robot.getTime()
        location = self.world_state.current_location
        if observe:
            self.plan = [None]      # scan here, wherever "here" is
        else:
            # Outside a named place (e.g. stopped between rooms): scan here.
            self.plan = list(self.viewpoints.get(location) or [None])
        self.plan_index = -1
        self.outcome = None
        self.evidence = None
        self.frames = 0
        self.unresolved = []        # target-like blobs never confirmed
        self.far_sightings = []     # same, but estimated outside this room
        self.center_distance = None
        self.incomplete = []        # parts of the policy not completed
        self.viewpoint_log = []
        self.rejected_dirs = []
        self.best = None
        self.distractors = {}
        self._next_viewpoint()
        return RUNNING

    def update(self):
        if self.phase in ("IDLE", "DONE"):
            return FAILED if self.outcome is None else SUCCEEDED
        now = self.navigator.robot.getTime()
        if now - self.started_at > SEARCH_BUDGET_S:
            self.incomplete.append(f"time budget {SEARCH_BUDGET_S:.0f} s used up "
                                   f"during {self.phase}")
            return self._finish()

        if self.phase == "MOVE":
            return self._update_move()

        pose = self.navigator.pose()
        if pose is None:
            self.navigator.stop()
            self.evidence = {"reason": "no GPS/Compass reading during search"}
            self.phase = "DONE"
            return FAILED
        if self.phase == "SCAN":
            return self._update_scan(now, pose)
        if self.phase == "CENTER":
            return self._update_center(now, pose)
        if self.phase in ("SETTLE", "CONFIRM"):
            return self._update_confirm(now, pose)
        return RUNNING

    def cancel(self):
        self.navigator.cancel()
        self.navigator.stop()
        self.phase = "IDLE"

    # ------------------------------------------------------------ phases
    def _next_viewpoint(self):
        self.plan_index += 1
        if self.plan_index >= len(self.plan):
            return self._finish()
        node = self.plan[self.plan_index]
        if node is None:            # observe: scan where the robot stands
            self._begin_scan()
            return RUNNING
        pose = self.navigator.pose()
        target = self.navigator.routes.nodes.get(node)
        if (pose is not None and target is not None and
                math.hypot(target[0] - pose[0], target[1] - pose[1]) < VIEWPOINT_REACHED_M):
            self._begin_scan()
        else:
            status = self.navigator.start(node)
            if status != RUNNING:
                self.incomplete.append(f"viewpoint {node} unreachable ({status})")
                return self._next_viewpoint()
            self.phase = "MOVE"
        return RUNNING

    def _update_move(self):
        status = self.navigator.update()
        if status == RUNNING:
            return RUNNING
        node = self.plan[self.plan_index]
        if status != SUCCEEDED:
            reason = (self.navigator.evidence or {}).get("reason", status)
            self.incomplete.append(f"viewpoint {node} not reached: {reason}")
            self.viewpoint_log.append({"viewpoint": node, "scanned": False,
                                       "reason": reason})
            return self._next_viewpoint()
        self._begin_scan()
        return RUNNING

    def _begin_scan(self):
        self.phase = "SCAN"
        self.turned = 0.0
        self.last_yaw = None
        self.stall_since = self.navigator.robot.getTime()
        self.stall_turned = 0.0
        self.next_overview = 0.0
        self.rejected_dirs = []
        self.viewpoint_log.append({"viewpoint": self.plan[self.plan_index] or "here",
                                   "scanned": False})

    def _update_scan(self, now, pose):
        yaw = pose[2]
        if self.last_yaw is not None:
            self.turned += abs(wrap_angle(yaw - self.last_yaw))
        self.last_yaw = yaw

        detections = self.perception.detect()
        self.frames += 1
        if self.turned >= self.next_overview:
            self._save("overview", detections)
            self.next_overview += OVERVIEW_EVERY_RAD
        candidate = self._pick_candidate(detections, yaw, pose)

        if candidate is not None:
            self.navigator.stop()
            self.center_dir = wrap_angle(yaw + candidate.bearing_rad)
            self.center_distance = candidate.distance_m
            self.center_started = now
            self.phase = "CENTER"
            self._save("candidate", detections)
            return RUNNING

        if self.turned >= FULL_TURN_RAD:
            self.navigator.stop()
            self.viewpoint_log[-1].update({"scanned": True,
                                           "turned_deg": round(math.degrees(self.turned))})
            return self._next_viewpoint()

        if self.turned - self.stall_turned > SCAN_STALL_RAD:
            self.stall_turned = self.turned
            self.stall_since = now
        elif now - self.stall_since > SCAN_STALL_S:
            self.navigator.stop()
            node = self.plan[self.plan_index]
            self.incomplete.append(f"scan at {node} stalled after "
                                   f"{math.degrees(self.turned):.0f} deg")
            self.viewpoint_log[-1].update({"scanned": False,
                                           "turned_deg": round(math.degrees(self.turned))})
            return self._next_viewpoint()

        self.navigator.drive(-SCAN_WHEEL_SPEED, SCAN_WHEEL_SPEED)
        return RUNNING

    def _update_center(self, now, pose):
        error = wrap_angle(self.center_dir - pose[2])
        self._track_turn(pose[2])
        if abs(math.degrees(error)) < CENTER_TOLERANCE_DEG or \
                now - self.center_started > CENTER_MAX_S:
            self.navigator.stop()
            self.phase = "SETTLE"
            self.settle_left = SETTLE_FRAMES
            self.confirm_hits = []
            self.confirm_seen = 0
            return RUNNING
        speed = max(0.3, min(SCAN_WHEEL_SPEED, CENTER_GAIN * abs(error)))
        direction = 1.0 if error > 0 else -1.0
        self.navigator.drive(-direction * speed, direction * speed)
        return RUNNING

    def _update_confirm(self, now, pose):
        self.navigator.stop()
        detections = self.perception.detect()
        self.frames += 1
        if self.phase == "SETTLE":
            self.settle_left -= 1
            if self.settle_left <= 0:
                self.phase = "CONFIRM"
            return RUNNING

        self.confirm_seen += 1
        hit = self._target_near(detections, pose, self.center_dir)
        if hit is not None:
            self.confirm_hits.append(hit)
        if self.confirm_seen < CONFIRM_FRAMES:
            return RUNNING

        self._save("confirm", detections)
        if len(self.confirm_hits) >= CONFIRM_REQUIRED:
            best = max(self.confirm_hits, key=lambda d: d.confidence)
            self.best = self._located(best, pose)
            self.best["confirm_frames"] = f"{len(self.confirm_hits)}/{CONFIRM_FRAMES}"
            self.best["confidence"] = round(
                sum(d.confidence for d in self.confirm_hits) / CONFIRM_FRAMES, 2)
            self.viewpoint_log[-1].update({"found": True,
                                           "turned_deg": round(math.degrees(self.turned))})
            return self._finish()

        self._note_unresolved(
            f"confirmation failed ({len(self.confirm_hits)}/{CONFIRM_FRAMES} frames)",
            self.center_dir, self.center_distance, pose)
        self.rejected_dirs.append(self.center_dir)
        self.last_yaw = pose[2]
        self.phase = "SCAN"
        self.stall_since = now
        return RUNNING

    # ------------------------------------------------------------ helpers
    def _pick_candidate(self, detections, yaw, pose):
        if self.observe:
            self._record_observations(detections, pose)
            return None
        object_type, colour = self.target
        best = None
        for det in detections:
            is_target = matches(det, object_type, colour)
            if not is_target:
                if det.usable and det.object_type != "unclear":
                    key = f"{det.colour}_{det.object_type}"
                    self.distractors[key] = self.distractors.get(key, 0) + 1
                continue
            direction = wrap_angle(yaw + det.bearing_rad)
            if any(abs(wrap_angle(direction - d)) < math.radians(REJECTED_BEARING_DEG)
                   for d in self.rejected_dirs):
                continue
            if det.rejected == "too_small":
                self._note_unresolved("target-like blob too small to confirm",
                                      direction, det.distance_m, pose)
                continue
            if not det.usable or det.confidence < CANDIDATE_MIN_CONFIDENCE:
                continue
            if best is None or det.confidence > best.confidence:
                best = det
        return best

    def _record_observations(self, detections, pose):
        """Keep the clearest view of each kind of object (observe mode)."""
        for det in detections:
            if not det.usable or det.object_type == "unclear":
                continue
            key = f"{det.colour} {det.object_type}"
            old = self.observations.get(key)
            if old is None or det.diameter_px > old["diameter_px"]:
                self.observations[key] = self._located(det, pose)

    def _note_unresolved(self, reason, direction, distance, pose):
        """Record a target-like blob that could not be confirmed.

        If its size puts it outside the room the robot is in (e.g. seen far
        away through a doorway), it says nothing about this room: it is kept
        as a far sighting and does not make this room's result UNCERTAIN.
        """
        node = self.plan[self.plan_index]
        item = {"reason": reason, "viewpoint": node,
                "direction_deg": round(math.degrees(direction), 1)}
        target = self.unresolved
        if distance is not None:
            ex = pose[0] + distance * math.cos(direction)
            ey = pose[1] + distance * math.sin(direction)
            item["estimated_position"] = [round(ex, 2), round(ey, 2)]
            here = room_of(pose[0], pose[1], self.rooms)
            there = room_of(ex, ey, self.rooms)
            item["estimated_room"] = there
            if there != here:
                target = self.far_sightings
        for old in target:
            if (old["viewpoint"] == node and old["reason"] == reason and
                    abs(math.radians(old["direction_deg"]) - direction) < 0.2):
                return
        target.append(item)

    def _target_near(self, detections, pose, direction):
        object_type, colour = self.target
        best = None
        for det in detections:
            if not (det.usable and matches(det, object_type, colour)):
                continue
            offset = wrap_angle(pose[2] + det.bearing_rad - direction)
            if abs(math.degrees(offset)) > CONFIRM_BEARING_DEG:
                continue
            if best is None or det.confidence > best.confidence:
                best = det
        return best

    def _located(self, det, pose):
        direction = wrap_angle(pose[2] + det.bearing_rad)
        info = det.summary()
        info["direction_deg"] = round(math.degrees(direction), 1)
        info["robot"] = [round(pose[0], 3), round(pose[1], 3)]
        if det.distance_m is not None:
            ex = pose[0] + det.distance_m * math.cos(direction)
            ey = pose[1] + det.distance_m * math.sin(direction)
            info["estimated_position"] = [round(ex, 2), round(ey, 2)]
            info["estimated_room"] = room_of(ex, ey, self.rooms)
        return info

    def _track_turn(self, yaw):
        if self.last_yaw is not None:
            self.turned += abs(wrap_angle(yaw - self.last_yaw))
        self.last_yaw = yaw

    def _save(self, tag, detections):
        if self.frame_saver is not None and self.perception.last_frame is not None:
            self.frame_saver(f"{self.plan[self.plan_index] or 'here'}_{tag}",
                             self.perception.last_frame, detections)

    def _finish(self):
        self.navigator.stop()
        if self.observe:
            self.outcome = OBSERVED
            self.evidence = {
                "outcome": OBSERVED,
                "location": self.world_state.current_location,
                "viewpoints": self.viewpoint_log,
                "frames_examined": self.frames,
                "elapsed_s": round(self.navigator.robot.getTime() - self.started_at, 1),
                "objects": self.observations,
                "incomplete": self.incomplete,
            }
            self.phase = "DONE"
            return SUCCEEDED
        if self.best is not None:
            outcome = FOUND
        elif self.unresolved or self.incomplete:
            outcome = UNCERTAIN
        else:
            outcome = NOT_FOUND
        object_type, colour = self.target
        self.outcome = outcome
        self.evidence = {
            "outcome": outcome,
            "target": " ".join(v for v in (colour, object_type) if v),
            "location": self.world_state.current_location,
            "viewpoints": self.viewpoint_log,
            "frames_examined": self.frames,
            "elapsed_s": round(self.navigator.robot.getTime() - self.started_at, 1),
            "detection": self.best,
            "unresolved": self.unresolved[:5],
            "far_sightings": self.far_sightings[:5],
            "incomplete": self.incomplete,
            # Frames in which each non-target blob was seen and ignored.
            "distractor_frames": dict(self.distractors),
        }
        self.phase = "DONE"
        return SUCCEEDED
