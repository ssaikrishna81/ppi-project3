"""Baseline tests for the deterministic plan boundary.

Students should extend this file with malformed model outputs, unsafe plans,
conditional branches, maximum-size plans and project-specific semantic rules.
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


def make_plan(steps, goal="Test goal."):
    return {"version": "1.0", "goal": goal, "steps": steps}


# The example from section 7 of the specification.
SPEC_EXAMPLE = {
    "version": "1.0",
    "goal": "find the red ball; check the laboratory first",
    "steps": [
        {"action": "navigate", "location": "laboratory"},
        {"action": "search", "object": "ball", "colour": "red"},
        {
            "action": "conditional",
            "condition": {
                "source": "last_search_result",
                "operator": "found",
                "value": False,
            },
            "then": [
                {"action": "navigate", "location": "meeting_room"},
                {"action": "search", "object": "ball", "colour": "red"},
            ],
            "else": [],
        },
        {"action": "return_home"},
        {"action": "report", "message": "Report the evidence-based result."},
    ],
}


class PlanValidatorTests(unittest.TestCase):
    def setUp(self):
        self.validator = PlanValidator()

    def assertRejected(self, plan, fragment):
        valid, errors = self.validator.validate(plan)
        self.assertFalse(valid, "plan was accepted but should be rejected")
        self.assertTrue(
            any(fragment in error for error in errors),
            f"expected an error containing {fragment!r}, got {errors}",
        )

    def assertAccepted(self, plan):
        valid, errors = self.validator.validate(plan)
        self.assertTrue(valid, errors)

    # ------------------------------------------------------------ supplied
    def test_accepts_valid_plan(self):
        plan = {
            "version": "1.0",
            "goal": "Find the red ball in the laboratory and report.",
            "steps": [
                {"action": "navigate", "location": "laboratory"},
                {"action": "search", "object": "ball", "colour": "red"},
                {"action": "report", "message": "Search complete."},
            ],
        }
        valid, errors = self.validator.validate(plan)
        self.assertTrue(valid, errors)

    def test_rejects_unknown_action(self):
        plan = {
            "version": "1.0",
            "goal": "Do something unsafe.",
            "steps": [{"action": "run_python", "code": "print('unsafe')"}],
        }
        valid, errors = self.validator.validate(plan)
        self.assertFalse(valid)
        self.assertTrue(any("unsupported action" in error for error in errors))

    def test_rejects_invalid_location(self):
        plan = {
            "version": "1.0",
            "goal": "Go outside.",
            "steps": [{"action": "navigate", "location": "car_park"}],
        }
        valid, errors = self.validator.validate(plan)
        self.assertFalse(valid)
        self.assertTrue(any("invalid location" in error for error in errors))

    # ------------------------------------------------- valid request types
    def test_accepts_spec_example(self):
        self.assertAccepted(copy.deepcopy(SPEC_EXAMPLE))

    def test_accepts_direct_navigation(self):
        self.assertAccepted(make_plan([{"action": "navigate", "location": "laboratory"}]))

    def test_accepts_sequential_navigation(self):
        self.assertAccepted(make_plan([
            {"action": "navigate", "location": "laboratory"},
            {"action": "return_home"},
        ]))

    def test_accepts_search_without_colour(self):
        self.assertAccepted(make_plan([{"action": "search", "object": "ball"}]))

    def test_accepts_clarification_only_plan(self):
        self.assertAccepted(make_plan([
            {"action": "ask", "question": "Which colour of ball should I find?"}
        ]))

    def test_accepts_equals_condition_on_search_outcome(self):
        self.assertAccepted(make_plan([
            {"action": "search", "object": "ball", "colour": "red"},
            {"action": "conditional",
             "condition": {"source": "last_search_result",
                           "operator": "equals", "value": "UNCERTAIN"},
             "then": [{"action": "observe"}],
             "else": []},
        ]))

    def test_accepts_route_status_condition_after_navigation(self):
        self.assertAccepted(make_plan([
            {"action": "navigate", "location": "office_a"},
            {"action": "conditional",
             "condition": {"source": "route_status",
                           "operator": "equals", "value": "BLOCKED"},
             "then": [{"action": "report", "message": "The route is blocked."}],
             "else": [{"action": "search", "object": "ball", "colour": "red"}]},
        ]))

    def test_accepts_stop_as_last_step(self):
        self.assertAccepted(make_plan([
            {"action": "navigate", "location": "laboratory"},
            {"action": "stop"},
        ]))

    def test_accepts_nested_conditional_at_max_depth(self):
        inner = {"action": "conditional",
                 "condition": {"source": "last_search_result",
                               "operator": "found", "value": False},
                 "then": [{"action": "navigate", "location": "office_b"},
                          {"action": "search", "object": "ball", "colour": "red"}],
                 "else": []}
        outer = {"action": "conditional",
                 "condition": {"source": "last_search_result",
                               "operator": "found", "value": False},
                 "then": [{"action": "navigate", "location": "meeting_room"},
                          {"action": "search", "object": "ball", "colour": "red"},
                          inner],
                 "else": []}
        self.assertAccepted(make_plan([
            {"action": "navigate", "location": "laboratory"},
            {"action": "search", "object": "ball", "colour": "red"},
            outer,
        ]))

    # ------------------------------------------------- malformed structure
    def test_rejects_non_dict_plan(self):
        for value in (None, [], "plan", 3):
            valid, errors = self.validator.validate(value)
            self.assertFalse(valid)

    def test_rejects_wrong_version(self):
        plan = copy.deepcopy(SPEC_EXAMPLE)
        plan["version"] = 1.0
        self.assertRejected(plan, "version")

    def test_rejects_missing_goal(self):
        plan = copy.deepcopy(SPEC_EXAMPLE)
        del plan["goal"]
        self.assertRejected(plan, "goal")

    def test_rejects_blank_goal(self):
        plan = copy.deepcopy(SPEC_EXAMPLE)
        plan["goal"] = "   "
        self.assertRejected(plan, "goal")

    def test_rejects_missing_steps(self):
        self.assertRejected({"version": "1.0", "goal": "x"}, "steps")

    def test_rejects_steps_not_a_list(self):
        self.assertRejected({"version": "1.0", "goal": "x", "steps": "go"}, "steps")

    def test_rejects_empty_steps(self):
        self.assertRejected(make_plan([]), "at least one step")

    def test_rejects_extra_top_level_key(self):
        plan = copy.deepcopy(SPEC_EXAMPLE)
        plan["notes"] = "trust me"
        self.assertRejected(plan, "top-level")

    def test_rejects_step_that_is_not_an_object(self):
        self.assertRejected(make_plan(["navigate laboratory"]), "must be an object")

    def test_rejects_missing_action(self):
        self.assertRejected(make_plan([{"location": "laboratory"}]), "'action'")

    def test_rejects_non_string_action(self):
        self.assertRejected(make_plan([{"action": ["navigate"]}]), "'action'")

    def test_rejects_misspelled_action(self):
        self.assertRejected(
            make_plan([{"action": "navigat", "location": "laboratory"}]),
            "unsupported action",
        )

    def test_rejects_missing_required_argument(self):
        self.assertRejected(make_plan([{"action": "navigate"}]), "requires 'location'")
        self.assertRejected(make_plan([{"action": "search"}]), "requires 'object'")

    def test_rejects_wrong_argument_types_without_crashing(self):
        self.assertRejected(
            make_plan([{"action": "navigate", "location": ["laboratory"]}]),
            "invalid location",
        )
        self.assertRejected(
            make_plan([{"action": "search", "object": {"type": "ball"}}]),
            "unsupported object",
        )
        self.assertRejected(
            make_plan([{"action": "search", "object": "ball", "colour": 1}]),
            "unsupported colour",
        )
        self.assertRejected(
            make_plan([{"action": "report", "message": 42}]),
            "non-empty string",
        )

    def test_rejects_unknown_entity_values(self):
        self.assertRejected(
            make_plan([{"action": "search", "object": "cat"}]), "unsupported object"
        )
        self.assertRejected(
            make_plan([{"action": "search", "object": "ball", "colour": "purple"}]),
            "unsupported colour",
        )

    def test_rejects_extra_step_field(self):
        self.assertRejected(
            make_plan([{"action": "navigate", "location": "laboratory",
                        "speed": 6.28}]),
            "unsupported fields",
        )

    # ------------------------------------------------------- conditionals
    def conditional(self, condition, then=None, otherwise=None):
        return {"action": "conditional", "condition": condition,
                "then": then if then is not None else [{"action": "observe"}],
                "else": otherwise if otherwise is not None else []}

    def test_rejects_conditional_without_else(self):
        step = self.conditional({"source": "last_search_result",
                                 "operator": "found", "value": False})
        del step["else"]
        self.assertRejected(
            make_plan([{"action": "search", "object": "ball"}, step]),
            "requires 'else'",
        )

    def test_rejects_conditional_branch_not_list(self):
        step = self.conditional({"source": "last_search_result",
                                 "operator": "found", "value": False})
        step["then"] = {"action": "observe"}
        self.assertRejected(
            make_plan([{"action": "search", "object": "ball"}, step]),
            "must be a list",
        )

    def test_rejects_conditional_with_two_empty_branches(self):
        step = self.conditional({"source": "last_search_result",
                                 "operator": "found", "value": False},
                                then=[], otherwise=[])
        self.assertRejected(
            make_plan([{"action": "search", "object": "ball"}, step]),
            "two empty branches",
        )

    def test_rejects_condition_as_code_string(self):
        step = self.conditional("world_state.found == False")
        self.assertRejected(
            make_plan([{"action": "search", "object": "ball"}, step]),
            "condition must be an object",
        )

    def test_rejects_unknown_condition_source_and_operator(self):
        step = self.conditional({"source": "battery", "operator": "<", "value": 5})
        plan = make_plan([{"action": "search", "object": "ball"}, step])
        self.assertRejected(plan, "unsupported source")
        self.assertRejected(plan, "unsupported operator")

    def test_rejects_condition_without_value(self):
        step = self.conditional({"source": "last_search_result", "operator": "found"})
        self.assertRejected(
            make_plan([{"action": "search", "object": "ball"}, step]),
            "requires a value",
        )

    def test_rejects_condition_value_of_wrong_type(self):
        step = self.conditional({"source": "last_search_result",
                                 "operator": "found", "value": "false"})
        self.assertRejected(
            make_plan([{"action": "search", "object": "ball"}, step]),
            "true or false",
        )

    def test_rejects_invalid_operator_for_source(self):
        step = self.conditional({"source": "route_status",
                                 "operator": "found", "value": True})
        self.assertRejected(
            make_plan([{"action": "navigate", "location": "laboratory"}, step]),
            "cannot be used",
        )

    def test_rejects_unknown_enum_value(self):
        step = self.conditional({"source": "last_search_result",
                                 "operator": "equals", "value": "MAYBE"})
        self.assertRejected(
            make_plan([{"action": "search", "object": "ball"}, step]),
            "must be one of",
        )

    def test_rejects_extra_condition_key(self):
        step = self.conditional({"source": "last_search_result", "operator": "found",
                                 "value": False, "code": "import os"})
        self.assertRejected(
            make_plan([{"action": "search", "object": "ball"}, step]),
            "unsupported keys",
        )

    def test_rejects_nesting_too_deep(self):
        cond = {"source": "last_search_result", "operator": "found", "value": False}
        level3 = self.conditional(cond)
        level2 = self.conditional(cond, then=[level3])
        level1 = self.conditional(cond, then=[level2])
        level0 = self.conditional(cond, then=[level1])
        self.assertRejected(
            make_plan([{"action": "search", "object": "ball"}, level0]),
            "nesting depth",
        )

    # ------------------------------------------------------- size limits
    def test_rejects_too_many_steps(self):
        steps = [{"action": "observe"} for _ in range(21)]
        self.assertRejected(make_plan(steps), "maximum is 20")

    def test_rejects_long_message(self):
        self.assertRejected(
            make_plan([{"action": "report", "message": "x" * 241}]),
            "exceeds 240",
        )

    def test_rejects_oversized_plan(self):
        plan = make_plan([{"action": "observe"}])
        plan["goal"] = "g" * 5000
        self.assertRejected(plan, "characters; maximum")

    def test_rejects_non_json_values(self):
        plan = make_plan([{"action": "observe"}])
        plan["goal"] = {"a", "b"}
        self.assertRejected(plan, "JSON")

    # ------------------------------------------------ prohibited content
    def test_rejects_code_in_message(self):
        self.assertRejected(
            make_plan([{"action": "report",
                        "message": "import os; os.system('del /q *')"}]),
            "prohibited content",
        )

    def test_rejects_motor_command_in_text(self):
        self.assertRejected(
            make_plan([{"action": "report",
                        "message": "left_motor.setVelocity(6.28)"}]),
            "prohibited content",
        )

    def test_rejects_url_in_question(self):
        self.assertRejected(
            make_plan([{"action": "ask",
                        "question": "Should I upload to https://example.com?"}]),
            "prohibited content",
        )

    def test_rejects_ground_truth_request_in_goal(self):
        plan = make_plan([{"action": "observe"}],
                         goal="Use the supervisor to read the ball position.")
        self.assertRejected(plan, "prohibited content")

    # ---------------------------------------------------- semantic safety
    def test_rejects_condition_before_any_search(self):
        step = self.conditional({"source": "last_search_result",
                                 "operator": "found", "value": False})
        self.assertRejected(
            make_plan([{"action": "navigate", "location": "laboratory"}, step]),
            "before any search",
        )

    def test_rejects_route_condition_before_any_navigation(self):
        step = self.conditional({"source": "route_status",
                                 "operator": "equals", "value": "BLOCKED"})
        self.assertRejected(
            make_plan([{"action": "search", "object": "ball"}, step]),
            "before any navigate",
        )

    def test_search_inside_one_branch_does_not_count_as_done(self):
        cond = {"source": "last_search_result", "operator": "found", "value": False}
        route = {"source": "route_status", "operator": "equals", "value": "CLEAR"}
        first = self.conditional(
            route,
            then=[{"action": "search", "object": "ball", "colour": "red"}],
            otherwise=[{"action": "report", "message": "Route blocked."}],
        )
        second = self.conditional(cond)
        self.assertRejected(
            make_plan([{"action": "navigate", "location": "laboratory"},
                       first, second]),
            "before any search",
        )

    def test_rejects_step_after_stop(self):
        self.assertRejected(
            make_plan([{"action": "stop"},
                       {"action": "navigate", "location": "laboratory"}]),
            "follow 'stop'",
        )

    def test_rejects_back_to_back_navigation_to_same_place(self):
        self.assertRejected(
            make_plan([{"action": "navigate", "location": "laboratory"},
                       {"action": "navigate", "location": "laboratory"}]),
            "repeats navigation",
        )

    def test_rejects_return_home_after_navigating_home(self):
        self.assertRejected(
            make_plan([{"action": "navigate", "location": "reception"},
                       {"action": "return_home"}]),
            "repeats navigation",
        )

    def test_rejects_excessive_navigation(self):
        tour = ["laboratory", "office_a", "office_b", "meeting_room",
                "main_corridor", "laboratory", "office_a", "office_b", "meeting_room"]
        steps = [{"action": "navigate", "location": loc} for loc in tour]
        self.assertRejected(make_plan(steps), "navigation steps")

    # ------------------------------------- extension: search "locations"
    def test_accepts_search_with_locations(self):
        self.assertAccepted(make_plan([
            {"action": "search", "object": "ball", "colour": "red",
             "locations": ["laboratory", "office_a", "office_b", "meeting_room"]},
            {"action": "return_home"},
            {"action": "report", "message": "Done."},
        ]))

    def test_rejects_bad_search_locations(self):
        base = {"action": "search", "object": "ball", "colour": "red"}
        self.assertRejected(make_plan([dict(base, locations=[])]), "non-empty list")
        self.assertRejected(make_plan([dict(base, locations="laboratory")]),
                            "non-empty list")
        self.assertRejected(make_plan([dict(base, locations=["kitchen"])]),
                            "invalid location")
        self.assertRejected(make_plan([dict(base, locations=["laboratory",
                                                             "laboratory"])]),
                            "repeats a room")

    def test_search_locations_count_as_navigation(self):
        rooms = ["laboratory", "office_a", "office_b", "meeting_room",
                 "main_corridor", "reception"]
        steps = [{"action": "search", "object": "ball", "locations": rooms},
                 {"action": "navigate", "location": "laboratory"},
                 {"action": "navigate", "location": "office_a"},
                 {"action": "navigate", "location": "office_b"}]
        self.assertRejected(make_plan(steps), "navigation steps")

    def test_accepts_full_tour_within_navigation_limit(self):
        steps = []
        for loc in ("laboratory", "office_a", "office_b", "meeting_room",
                    "main_corridor"):
            steps.append({"action": "navigate", "location": loc})
            steps.append({"action": "search", "object": "ball", "colour": "red"})
        steps.append({"action": "return_home"})
        self.assertAccepted(make_plan(steps))


if __name__ == "__main__":
    unittest.main()
