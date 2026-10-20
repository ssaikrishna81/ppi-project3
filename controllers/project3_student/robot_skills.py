"""Deterministic, non-blocking skills available to validated plans.

Each action maps to one skill. start(step) begins it and may finish at once;
update() advances it by one Webots step. Both return RUNNING, SUCCEEDED,
FAILED, TIMEOUT or BLOCKED, and leave a plain dict in self.evidence saying
what happened. Outcomes come from sensors and controller state only.

navigate / return_home  Navigator (GPS + Compass + proximity).
search                  SearchController: FOUND / NOT_FOUND / UNCERTAIN.
observe                 one 360 deg scan here; objects seen go to semantic
                        memory and WorldState notes. Claims nothing about
                        the task.
ask                     speaks the question; the executive then waits for
                        the user's answer.
report                  speaks WorldState.report_text(), built from
                        measured results. The planner's message is kept in
                        the evidence but not spoken, so the robot never
                        repeats a claim it has no evidence for.
stop                    stops all motion; the executive ends the task.
"""
from search import FOUND, SearchController

RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"


class RobotSkills:
    def __init__(self, navigator, perception, memory, speech, world_state,
                 search=None):
        self.navigator = navigator
        self.perception = perception
        self.memory = memory
        self.speech = speech
        self.world_state = world_state
        self.search = search or SearchController(navigator, perception, world_state)
        self.active_step = None
        self.evidence = None

    def start(self, step: dict) -> str:
        """Start one already-validated symbolic action."""
        self.active_step = step
        self.evidence = None
        action = step["action"]
        if action == "navigate":
            return self.navigator.start(step["location"])
        if action == "return_home":
            return self.navigator.start("reception")
        if action == "search" and step.get("locations"):
            return self._start_room_search(step)
        if action == "search":
            return self.search.start(step["object"], step.get("colour"))
        if action == "observe":
            return self.search.start_observe()
        if action == "ask":
            self.speech.speak(step["question"])
            self.evidence = {"spoken": step["question"]}
            return SUCCEEDED
        if action == "report":
            text = self.world_state.report_text()
            self.speech.speak(text)
            self.evidence = {"spoken": text, "planner_message": step["message"]}
            return SUCCEEDED
        if action == "stop":
            self.navigator.cancel()
            self.search.cancel()
            self.evidence = {"stopped_at": self.world_state.current_location}
            return SUCCEEDED
        self.evidence = {"reason": f"no skill for action {action!r}"}
        return FAILED

    def update(self) -> str:
        """Advance the active skill by one Webots time step."""
        if not self.active_step:
            return FAILED
        action = self.active_step["action"]
        if action in {"navigate", "return_home"}:
            status = self.navigator.update()
            self.evidence = self.navigator.evidence
            if status == SUCCEEDED:
                self.world_state.current_location = (
                    self.active_step.get("location") or "reception"
                )
            return status
        if action == "search" and self.active_step.get("locations"):
            return self._update_room_search()
        if action == "search":
            status = self.search.update()
            self.evidence = self.search.evidence
            if status == SUCCEEDED:
                self._record_search()
            return status
        if action == "observe":
            status = self.search.update()
            self.evidence = self.search.evidence
            if status == SUCCEEDED:
                self._record_observation()
            return status
        # ask, report and stop finish in start().
        return SUCCEEDED

    def cancel(self):
        self.navigator.cancel()
        self.search.cancel()
        self.active_step = None

    # ---------------------------------------------- search with locations
    # Contract extension: visit each listed room in order, search it, stop
    # at the first FOUND. A room that cannot be reached is skipped and noted.
    # Overall outcome: FOUND if any room found it; NOT_FOUND only if every
    # room was searched and none found it; otherwise UNCERTAIN.
    def _start_room_search(self, step):
        self.rooms_left = list(step["locations"])
        self.room_results = []
        return self._next_room()

    def _next_room(self):
        if not self.rooms_left:
            return self._finish_room_search()
        self.room = self.rooms_left.pop(0)
        self.room_phase = "go"
        status = self.navigator.start(self.room)
        return RUNNING if status == RUNNING else self._room_unreachable(status)

    def _update_room_search(self):
        if self.room_phase == "go":
            status = self.navigator.update()
            if status == RUNNING:
                return RUNNING
            if status != SUCCEEDED:
                return self._room_unreachable(status)
            self.world_state.current_location = self.room
            self.world_state.record_route("CLEAR", self.room)
            self.room_phase = "look"
            step = self.active_step
            return self.search.start(step["object"], step.get("colour"))
        status = self.search.update()
        if status == RUNNING:
            return RUNNING
        if status != SUCCEEDED:
            self.room_results.append({"room": self.room, "outcome": "FAILED"})
            return self._next_room()
        self._record_search()
        self.room_results.append({"room": self.room, "outcome": self.search.outcome})
        if self.search.outcome == FOUND:
            self.rooms_left = []
        return self._next_room()

    def _room_unreachable(self, status):
        reason = (self.navigator.evidence or {}).get("reason", status)
        if status == "BLOCKED":
            self.world_state.record_route("BLOCKED", self.room, reason)
        self.room_results.append({"room": self.room, "outcome": "UNREACHABLE",
                                  "reason": reason})
        return self._next_room()

    def _finish_room_search(self):
        outcomes = [r["outcome"] for r in self.room_results]
        if FOUND in outcomes:
            final = FOUND
        elif outcomes and all(o == "NOT_FOUND" for o in outcomes):
            final = "NOT_FOUND"
        else:
            final = "UNCERTAIN"
        self.world_state.last_search_result = final
        found = next((r for r in self.room_results if r["outcome"] == FOUND), None)
        self.evidence = {"outcome": final, "rooms": self.room_results,
                         "detection": (self.search.evidence or {}).get("detection")
                         if found else None}
        print(f"[SEARCH] rooms {[r['room'] + ':' + r['outcome'] for r in self.room_results]}"
              f" -> {final}")
        return SUCCEEDED

    # ------------------------------------------------------------------
    def _record_search(self):
        outcome = self.search.outcome
        evidence = self.search.evidence
        detection = evidence.get("detection") or {}
        found_in = detection.get("estimated_room") if outcome == FOUND else None
        if found_in == "central_area":
            found_in = None
        self.world_state.record_search(outcome, evidence, found_in)
        print(_search_line(self.world_state.current_location, evidence))
        if outcome == FOUND:
            key = " ".join(
                value for value in (
                    self.active_step.get("colour"),
                    self.active_step.get("object"),
                ) if value
            )
            self.memory.update(key, found_in or self.world_state.current_location,
                               detection.get("confidence", 0.0))

    def _record_observation(self):
        objects = self.search.evidence.get("objects", {})
        here = self.world_state.current_location
        for key, info in objects.items():
            room = info.get("estimated_room")
            place = here if room in (None, "central_area") else room
            self.memory.update(key, place, info.get("confidence", 0.0))
        seen = ", ".join(sorted(objects)) or "nothing recognisable"
        self.world_state.add_note(f"observed at {here}: {seen}")
        print(f"[OBSERVE] {here}: {seen}")


def _search_line(location, evidence):
    """One readable console line; the full evidence goes to the trace log."""
    text = f"[SEARCH] {location}: {evidence['outcome']} ({evidence['elapsed_s']} s)"
    det = evidence.get("detection")
    if det:
        text += (f" - {evidence['target']} at {det.get('estimated_position')}"
                 f" in {det.get('estimated_room')}, {det.get('distance_m')} m away,"
                 f" confidence {det.get('confidence')}")
    if evidence.get("unresolved"):
        text += f" - unconfirmed: {evidence['unresolved'][0]['reason']}"
    if evidence.get("incomplete"):
        text += f" - incomplete: {evidence['incomplete'][0]}"
    if evidence.get("far_sightings"):
        far = evidence["far_sightings"][0]
        text += f" - far sighting in {far.get('estimated_room')}"
    if evidence.get("distractor_frames"):
        text += f" - ignored {evidence['distractor_frames']}"
    return text
