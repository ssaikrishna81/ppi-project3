"""TaskExecutive and WorldState tests using scripted fake skills.

FakeSkills stands in for RobotSkills so the executive can be tested without
Webots. Each action gets a script: the number of RUNNING updates before it
finishes and the final status. Search steps also write a scripted outcome
(FOUND / NOT_FOUND / UNCERTAIN) per location into WorldState.
"""
import copy
import sys
import unittest
from pathlib import Path

CONTROLLER_DIR = (
    Path(__file__).resolve().parents[1] / "controllers" / "project3_student"
)
sys.path.insert(0, str(CONTROLLER_DIR))

from plan_validator import PlanValidator  # noqa: E402
from task_executive import (  # noqa: E402
    AWAITING_ANSWER,
    BLOCKED,
    CANCELLED,
    FAILED,
    IDLE,
    RUNNING,
    SUCCEEDED,
    TIMEOUT,
    TaskExecutive,
)
from world_state import WorldState  # noqa: E402


class FakeSkills:
    def __init__(self, world_state, durations=None, outcomes=None,
                 search_results=None):
        self.world_state = world_state
        self.durations = durations or {}        # action -> RUNNING updates
        self.outcomes = outcomes or {}          # (action, location) or action -> status
        self.search_results = search_results or {}  # location -> outcome
        self.started = []
        self.update_calls = 0
        self.cancel_calls = 0
        self.evidence = None
        self._step = None
        self._remaining = 0

    def start(self, step):
        self._step = step
        self.started.append(step)
        self._remaining = self.durations.get(step["action"], 1)
        return self._advance()

    def update(self):
        self.update_calls += 1
        return self._advance()

    def _advance(self):
        if self._remaining > 0:
            self._remaining -= 1
            return RUNNING
        action = self._step["action"]
        status = self.outcomes.get(
            (action, self._step.get("location")),
            self.outcomes.get(action, SUCCEEDED),
        )
        self.evidence = f"fake {action} -> {status}"
        if status == SUCCEEDED and action == "navigate":
            self.world_state.current_location = self._step["location"]
        if status == SUCCEEDED and action == "return_home":
            self.world_state.current_location = "reception"
        if status == SUCCEEDED and action == "search":
            outcome = self.search_results.get(
                self.world_state.current_location, "NOT_FOUND")
            self.world_state.record_search(outcome, self.evidence)
        return status

    def cancel(self):
        self.cancel_calls += 1


def run(executive, max_updates=500):
    """Call update() like the Webots loop until the task ends."""
    for _ in range(max_updates):
        status = executive.update()
        if status != RUNNING:
            return status
    raise AssertionError("executive did not finish")


def visited(skills):
    return [s.get("location", s["action"]) for s in skills.started
            if s["action"] in ("navigate", "return_home")]


SEARCH_RED = {"action": "search", "object": "ball", "colour": "red"}


def find_plan():
    """Spec section 7 example: lab first, meeting room only if not found."""
    return {
        "version": "1.0",
        "goal": "find the red ball; check the laboratory first",
        "steps": [
            {"action": "navigate", "location": "laboratory"},
            dict(SEARCH_RED),
            {"action": "conditional",
             "condition": {"source": "last_search_result",
                           "operator": "found", "value": False},
             "then": [{"action": "navigate", "location": "meeting_room"},
                      dict(SEARCH_RED)],
             "else": []},
            {"action": "return_home"},
            {"action": "report", "message": "Report the evidence-based result."},
        ],
    }


class ExecutiveTests(unittest.TestCase):
    def setUp(self):
        self.ws = WorldState()

    def make(self, **kwargs):
        clock = kwargs.pop("clock", None)
        skills = FakeSkills(self.ws, **kwargs)
        return TaskExecutive(skills, self.ws, clock=clock), skills

    # ------------------------------------------------------- basic flow
    def test_idle_before_start(self):
        executive, _ = self.make()
        self.assertEqual(executive.update(), IDLE)

    def test_sequence_runs_in_order(self):
        executive, skills = self.make(durations={"navigate": 3})
        executive.start({"steps": [
            {"action": "navigate", "location": "laboratory"},
            {"action": "navigate", "location": "office_a"},
            {"action": "return_home"},
        ]})
        self.assertEqual(run(executive), SUCCEEDED)
        self.assertEqual(visited(skills), ["laboratory", "office_a", "return_home"])
        self.assertEqual(self.ws.current_location, "reception")
        self.assertEqual([e["status"] for e in executive.trace], [SUCCEEDED] * 3)
        self.assertEqual(self.ws.execution_status, SUCCEEDED)

    def test_one_unit_of_work_per_update(self):
        executive, skills = self.make(durations={"navigate": 5})
        executive.start({"steps": [{"action": "navigate", "location": "laboratory"},
                                   {"action": "navigate", "location": "office_a"}]})
        executive.update()
        self.assertEqual(len(skills.started), 1)
        self.assertEqual(skills.update_calls, 0)
        executive.update()
        self.assertEqual(len(skills.started), 1)
        self.assertEqual(skills.update_calls, 1)

    def test_finishing_a_step_does_not_start_the_next_in_same_update(self):
        executive, skills = self.make(durations={"navigate": 0})
        executive.start({"steps": [{"action": "navigate", "location": "laboratory"},
                                   {"action": "navigate", "location": "office_a"}]})
        executive.update()
        self.assertEqual(len(skills.started), 1)

    def test_plan_dict_is_not_modified(self):
        plan = find_plan()
        original = copy.deepcopy(plan)
        executive, _ = self.make(search_results={"laboratory": "NOT_FOUND"})
        executive.start(plan)
        run(executive)
        self.assertEqual(plan, original)

    # ----------------------------------------------------- conditionals
    def test_not_found_takes_then_branch(self):
        executive, skills = self.make(search_results={"meeting_room": "FOUND"})
        executive.start(find_plan())
        self.assertEqual(run(executive), SUCCEEDED)
        self.assertEqual(visited(skills),
                         ["laboratory", "meeting_room", "return_home"])
        self.assertEqual(self.ws.search_results,
                         {"laboratory": "NOT_FOUND", "meeting_room": "FOUND"})

    def test_found_takes_else_branch(self):
        executive, skills = self.make(search_results={"laboratory": "FOUND"})
        executive.start(find_plan())
        self.assertEqual(run(executive), SUCCEEDED)
        self.assertEqual(visited(skills), ["laboratory", "return_home"])

    def test_uncertain_counts_as_not_found(self):
        executive, skills = self.make(search_results={"laboratory": "UNCERTAIN"})
        executive.start(find_plan())
        run(executive)
        self.assertIn("meeting_room", visited(skills))

    def test_equals_condition_matches_exact_outcome(self):
        executive, skills = self.make(search_results={"laboratory": "UNCERTAIN"})
        executive.start({"steps": [
            {"action": "navigate", "location": "laboratory"},
            dict(SEARCH_RED),
            {"action": "conditional",
             "condition": {"source": "last_search_result",
                           "operator": "equals", "value": "UNCERTAIN"},
             "then": [{"action": "observe"}],
             "else": [{"action": "return_home"}]},
        ]})
        run(executive)
        self.assertEqual([s["action"] for s in skills.started],
                         ["navigate", "search", "observe"])

    def test_conditional_decision_is_traced(self):
        executive, _ = self.make(search_results={"laboratory": "NOT_FOUND"})
        executive.start(find_plan())
        run(executive)
        decisions = [e for e in executive.trace
                     if e["step"]["action"] == "conditional"]
        self.assertEqual(len(decisions), 1)
        self.assertIn("'NOT_FOUND'", decisions[0]["evidence"])
        self.assertIn("then", decisions[0]["evidence"])

    def test_condition_without_value_fails_safely(self):
        executive, _ = self.make()
        executive.start({"steps": [
            {"action": "conditional",
             "condition": {"source": "last_search_result",
                           "operator": "found", "value": False},
             "then": [{"action": "observe"}], "else": []},
        ]})
        self.assertEqual(run(executive), FAILED)

    # ---------------------------------------------------------- failures
    def test_blocked_navigation_ends_plan_with_evidence(self):
        executive, skills = self.make(
            outcomes={("navigate", "office_a"): BLOCKED})
        executive.start({"steps": [
            {"action": "navigate", "location": "laboratory"},
            {"action": "navigate", "location": "office_a"},
            dict(SEARCH_RED),
        ]})
        self.assertEqual(run(executive), BLOCKED)
        self.assertEqual(self.ws.route_status, "BLOCKED")
        self.assertEqual(self.ws.blocked_routes,
                         [{"from": "laboratory", "to": "office_a"}])
        self.assertEqual(self.ws.last_failure["status"], BLOCKED)
        self.assertGreaterEqual(skills.cancel_calls, 1)
        evidence = executive.failure_evidence()
        self.assertEqual(evidence["remaining_steps"][0]["location"], "office_a")
        self.assertEqual(len(evidence["remaining_steps"]), 2)
        self.assertLessEqual(len(evidence["recent_trace"]), 5)

    def test_blocked_route_handled_by_route_conditional(self):
        executive, skills = self.make(
            outcomes={("navigate", "office_a"): BLOCKED})
        executive.start({"steps": [
            {"action": "navigate", "location": "office_a"},
            {"action": "conditional",
             "condition": {"source": "route_status",
                           "operator": "equals", "value": "BLOCKED"},
             "then": [{"action": "navigate", "location": "laboratory"}],
             "else": []},
        ]})
        self.assertEqual(run(executive), SUCCEEDED)
        self.assertEqual(visited(skills), ["office_a", "laboratory"])

    def test_failed_skill_ends_plan(self):
        executive, _ = self.make(outcomes={"search": FAILED})
        executive.start({"steps": [dict(SEARCH_RED), {"action": "return_home"}]})
        self.assertEqual(run(executive), FAILED)

    def test_skill_timeout_status_is_reported(self):
        executive, _ = self.make(outcomes={"navigate": TIMEOUT})
        executive.start({"steps": [{"action": "navigate", "location": "office_b"}]})
        self.assertEqual(run(executive), TIMEOUT)

    def test_backup_timeout_when_skill_never_finishes(self):
        now = [0.0]
        executive, skills = self.make(durations={"observe": 10**9},
                                      clock=lambda: now[0])
        executive.start({"steps": [{"action": "observe"}]})
        executive.update()
        now[0] = 29.0
        self.assertEqual(executive.update(), RUNNING)
        now[0] = 31.0
        self.assertEqual(executive.update(), TIMEOUT)
        self.assertGreaterEqual(skills.cancel_calls, 1)
        self.assertEqual(executive.trace[-1]["status"], TIMEOUT)

    def test_unknown_skill_status_fails(self):
        executive, _ = self.make(outcomes={"observe": "MAYBE"})
        executive.start({"steps": [{"action": "observe"}]})
        self.assertEqual(run(executive), FAILED)

    def test_search_not_found_is_not_a_failure(self):
        executive, _ = self.make(search_results={"reception": "NOT_FOUND"})
        executive.start({"steps": [dict(SEARCH_RED)]})
        self.assertEqual(run(executive), SUCCEEDED)
        self.assertEqual(self.ws.last_search_result, "NOT_FOUND")

    # ----------------------------------------------- cancel, stop, ask
    def test_cancel_mid_step(self):
        executive, skills = self.make(durations={"navigate": 100})
        executive.start({"steps": [{"action": "navigate", "location": "office_b"},
                                   {"action": "return_home"}]})
        executive.update()
        executive.update()
        executive.cancel()
        self.assertFalse(executive.active)
        self.assertEqual(self.ws.execution_status, CANCELLED)
        self.assertEqual(executive.trace[-1]["status"], CANCELLED)
        self.assertEqual(skills.cancel_calls, 1)
        self.assertEqual(executive.update(), IDLE)
        self.assertEqual(len(skills.started), 1)

    def test_cancel_when_idle_does_nothing(self):
        executive, skills = self.make()
        executive.cancel()
        self.assertEqual(skills.cancel_calls, 0)

    def test_stop_step_ends_task(self):
        executive, skills = self.make()
        executive.start({"steps": [
            {"action": "navigate", "location": "laboratory"},
            dict(SEARCH_RED),
            {"action": "conditional",
             "condition": {"source": "last_search_result",
                           "operator": "found", "value": True},
             "then": [{"action": "stop"}], "else": []},
            {"action": "return_home"},
        ]})
        self.ws.current_location = "reception"
        skills.search_results = {"laboratory": "FOUND"}
        self.assertEqual(run(executive), CANCELLED)
        self.assertNotIn("return_home", visited(skills))

    def test_ask_waits_for_answer(self):
        executive, skills = self.make()
        executive.start({"steps": [
            {"action": "ask", "question": "Which colour of ball?"},
            dict(SEARCH_RED),
        ]})
        self.assertEqual(run(executive), AWAITING_ANSWER)
        self.assertEqual(self.ws.pending_question, "Which colour of ball?")
        self.assertEqual(len(skills.started), 1)

    # ----------------------------------------- validator + executive
    def test_validated_spec_plan_runs(self):
        plan = find_plan()
        valid, errors = PlanValidator().validate(plan)
        self.assertTrue(valid, errors)
        executive, skills = self.make(search_results={"meeting_room": "FOUND"})
        executive.start(plan)
        self.assertEqual(run(executive), SUCCEEDED)
        self.assertEqual(skills.started[-1]["action"], "report")


class WorldStateTests(unittest.TestCase):
    def test_reset_clears_task_evidence_but_keeps_location(self):
        ws = WorldState()
        ws.current_location = "laboratory"
        ws.record_search("FOUND", "ok")
        ws.record_route("BLOCKED", "office_a", "front sensors")
        ws.reset_for_new_goal("Find the red ball.")
        self.assertIsNone(ws.last_search_result)
        self.assertEqual(ws.search_results, {})
        self.assertEqual(ws.blocked_routes, [])
        self.assertIsNone(ws.route_status)
        self.assertEqual(ws.current_location, "laboratory")
        self.assertEqual(ws.current_goal, "Find the red ball.")

    def test_notes_are_bounded(self):
        ws = WorldState()
        for i in range(100):
            ws.add_note(f"note {i}")
        self.assertEqual(len(ws.notes), 20)
        self.assertEqual(ws.notes[-1], "note 99")

    def test_describe_uses_stored_state(self):
        ws = WorldState()
        self.assertIn("idle", ws.describe())
        ws.reset_for_new_goal("Find the red ball.")
        ws.execution_status = "RUNNING"
        ws.current_step = {"action": "navigate", "location": "meeting_room"}
        ws.current_location = "laboratory"
        ws.record_search("NOT_FOUND")
        text = ws.describe()
        self.assertIn("Find the red ball", text)
        self.assertIn("going to meeting room", text)
        self.assertIn("laboratory not found", text)
        # Moving: say where it is coming from, not "I am at" (Webots test).
        self.assertIn("on my way from the laboratory", text)

    def test_as_dict_is_a_copy(self):
        ws = WorldState()
        snapshot = ws.as_dict()
        snapshot["blocked_routes"].append("x")
        self.assertEqual(ws.blocked_routes, [])


if __name__ == "__main__":
    unittest.main()
