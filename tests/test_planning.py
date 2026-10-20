"""Grounding, JSON extraction/repair and the correction loop.

The LLM is replaced by scripted responses (a test fixture, not model
evidence) so the pipeline logic can be checked deterministically.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "controllers" / "project3_student"))

from grounding import check_consistency, ground  # noqa: E402
from llm_interface import LLMError, LLMInterface, extract_json_object  # noqa: E402
from plan_validator import PlanValidator  # noqa: E402
from planning import plan_request, replan_request  # noqa: E402


def scripted(*responses):
    """client(system, user) returning the given strings in turn."""
    queue = list(responses)
    calls = []

    def client(system, user):
        calls.append(user)
        return queue.pop(0)
    client.calls = calls
    return client


def plan(*steps, goal="g"):
    return json.dumps({"version": "1.0", "goal": goal, "steps": list(steps)})


NAV_LAB = {"action": "navigate", "location": "laboratory"}
SEARCH_RED = {"action": "search", "object": "ball", "colour": "red"}


class GroundingTests(unittest.TestCase):
    def test_locations_in_spoken_order_with_synonyms(self):
        g = ground("Visit office B, then the lab and finally the hallway")
        self.assertEqual(g["locations"], ["office_b", "laboratory", "main_corridor"])

    def test_target_and_condition(self):
        g = ground("Check the lab first; only search Office A if the red ball is absent")
        self.assertEqual((g["object"], g["colour"]), ("ball", "red"))
        self.assertTrue(g["conditional"])
        self.assertTrue(g["search"])

    def test_unsupported(self):
        self.assertIn("fly", ground("Fly through the wall to Office B")["unsupported"])
        self.assertEqual(ground("go to the lab")["unsupported"], [])

    def test_return_home(self):
        self.assertTrue(ground("go to the lab and then come back")["return_home"])
        # Seen failing in the real-model benchmark before the pattern fix.
        self.assertTrue(ground("head to the lab and come straight back")["return_home"])
        self.assertTrue(ground("go to office A, then go right back home")["return_home"])
        self.assertFalse(ground("go to the lab")["return_home"])
        # Webots test: "go back to office b" was planned as return_home.
        g = ground("go back to office b")
        self.assertFalse(g["return_home"])
        self.assertEqual(g["locations"], ["office_b"])

    def test_nearest_rooms_first(self):
        from planning import nearest_rooms_first
        plan = {"steps": [{"action": "search", "object": "ball", "colour": "red",
                           "locations": ["laboratory", "office_a", "office_b",
                                         "meeting_room"]}]}
        # Webots test: stopped near office_b_entry, then "find red ball".
        note = nearest_rooms_first(plan, [1.83, 0.38])
        self.assertTrue(note)
        self.assertEqual(plan["steps"][0]["locations"][0], "office_b")
        self.assertIsNone(nearest_rooms_first(plan, None))

    def test_consistency_catches_missing_room(self):
        g = ground("go to office A")
        problems = check_consistency({"steps": [NAV_LAB]}, g)
        self.assertTrue(any("office_a" in p for p in problems))

    def test_consistency_catches_wrong_ask(self):
        g = ground("please drive over to the meeting room")
        problems = check_consistency(
            {"steps": [{"action": "ask", "question": "I cannot fly"}]}, g)
        self.assertTrue(problems)

    def test_consistency_requires_ask_for_unsupported(self):
        g = ground("jump over the table")
        self.assertTrue(check_consistency({"steps": [NAV_LAB]}, g))
        self.assertEqual(check_consistency(
            {"steps": [{"action": "ask", "question": "I cannot jump."}]}, g), [])

    def test_consistency_accepts_good_search_plan(self):
        g = ground("look in the lab for the red ball, if not there check the meeting room")
        good = {"steps": [NAV_LAB, SEARCH_RED,
                          {"action": "conditional",
                           "condition": {"source": "last_search_result",
                                         "operator": "found", "value": False},
                           "then": [{"action": "navigate", "location": "meeting_room"},
                                    SEARCH_RED], "else": []}]}
        self.assertEqual(check_consistency(good, g), [])


class ExampleBankTests(unittest.TestCase):
    def test_every_example_is_valid_and_consistent(self):
        from grounding import pattern_of
        from planning import load_examples
        validator = PlanValidator()
        for example in load_examples():
            g = ground(example["request"])
            self.assertEqual(pattern_of(g), example["pattern"], example["request"])
            self.assertEqual(validator.validate(example["plan"]), (True, []))
            self.assertEqual(check_consistency(example["plan"], g), [],
                             example["request"])

    def test_best_match_is_last(self):
        from planning import select_examples
        shots = select_examples("search_all")
        self.assertEqual(shots[-1][0], "find the red ball")
        self.assertEqual(len(shots), 2)


class ExtractionTests(unittest.TestCase):
    def test_code_fence_and_prose(self):
        value = extract_json_object('Sure!\n```json\n{"a": 1}\n```')
        self.assertEqual(value, {"a": 1})

    def test_no_json(self):
        with self.assertRaises(LLMError):
            extract_json_object("I will go to the lab.")

    def test_repairs_aliases(self):
        llm = LLMInterface(client=scripted(plan(
            {"action": "Navigate", "location": "lab"},
            {"action": "search", "object": "ball", "color": "red"})))
        result = llm.generate_plan("go to the lab", {})
        self.assertEqual(result["steps"][0], NAV_LAB)
        self.assertEqual(result["steps"][1], SEARCH_RED)
        self.assertTrue(llm.last["repairs"])


class ActionAliasTests(unittest.TestCase):
    def test_drive_becomes_navigate(self):
        # Seen in Webots testing: {"action": "drive", "location": "office_b"}.
        llm = LLMInterface(client=scripted(plan(
            {"action": "drive", "location": "office_b"})))
        result = llm.generate_plan("drive to office B", {})
        self.assertEqual(result["steps"], [{"action": "navigate", "location": "office_b"}])

    def test_destination_key_repaired(self):
        llm = LLMInterface(client=scripted(plan(
            {"action": "go", "destination": "lab"})))
        result = llm.generate_plan("go to the lab", {})
        self.assertEqual(result["steps"], [NAV_LAB])

    def test_unknown_verb_left_for_validator(self):
        llm = LLMInterface(client=scripted(plan(
            {"action": "fly", "location": "office_b"})))
        result = llm.generate_plan("fly to office B", {})
        self.assertEqual(result["steps"][0]["action"], "fly")
        self.assertFalse(PlanValidator().validate(result)[0])

    def test_alias_without_arguments_not_repaired(self):
        llm = LLMInterface(client=scripted(plan({"action": "drive"})))
        result = llm.generate_plan("drive", {})
        self.assertEqual(result["steps"][0]["action"], "drive")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.validator = PlanValidator()

    def test_valid_first_time(self):
        llm = LLMInterface(client=scripted(plan(NAV_LAB)))
        result = plan_request(llm, self.validator, "go to the lab", {})
        self.assertEqual(result["kind"], "plan")
        self.assertEqual(len(result["attempts"]), 1)

    def test_invalid_then_corrected(self):
        llm = LLMInterface(client=scripted(
            plan({"action": "teleport", "location": "laboratory"}),
            plan(NAV_LAB)))
        result = plan_request(llm, self.validator, "go to the lab", {})
        self.assertEqual(result["kind"], "plan")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertIn("unsupported action", llm.client.calls[1])

    def test_inconsistent_then_corrected(self):
        llm = LLMInterface(client=scripted(
            plan({"action": "ask", "question": "I cannot fly."}),
            plan({"action": "navigate", "location": "meeting_room"})))
        result = plan_request(llm, self.validator, "drive to the meeting room", {})
        self.assertEqual(result["kind"], "plan")
        self.assertIn("Hints:", llm.client.calls[0])

    def test_retry_limit_enforced(self):
        bad = plan({"action": "teleport"})
        llm = LLMInterface(client=scripted(bad, bad, bad, bad))
        result = plan_request(llm, self.validator, "go to the lab", {}, max_retries=1)
        self.assertEqual(result["kind"], "rejected")
        self.assertEqual(len(llm.client.calls), 2)

    def test_garbage_output_asks_again_from_scratch(self):
        llm = LLMInterface(client=scripted("I am not sure.", plan(NAV_LAB)))
        result = plan_request(llm, self.validator, "go to the lab", {})
        self.assertEqual(result["kind"], "plan")
        self.assertNotIn("previous plan", llm.client.calls[1])

    def test_replan_situation_and_examples(self):
        llm = LLMInterface(client=scripted(plan(
            {"action": "search", "object": "ball", "colour": "red",
             "locations": ["office_b", "meeting_room"]},
            {"action": "return_home"}, {"action": "report", "message": "x"})))
        evidence = {"failure": {"status": "BLOCKED",
                                "step": {"action": "navigate", "location": "office_a"},
                                "reason": "no route avoids the blocked edge"}}
        snapshot = {"current_location": "laboratory",
                    "search_results": {"laboratory": "NOT_FOUND"},
                    "blocked_routes": [{"from": "laboratory", "to": "office_a"}]}
        result = replan_request(llm, self.validator, "find the red ball",
                                snapshot, evidence)
        self.assertEqual(result["kind"], "plan")
        prompt = llm.client.calls[0]
        self.assertIn("Cannot reach: office_a", prompt)
        self.assertIn("Already searched, not found: laboratory", prompt)
        self.assertIn("Rooms still to search: office_b, meeting_room", prompt)
        self.assertIn("Examples of revised plans", prompt)

    def test_replan_rejects_blocked_and_repeated_rooms(self):
        from planning import check_replan, replan_situation
        _, facts = replan_situation(
            "find the red ball",
            {"search_results": {"laboratory": "NOT_FOUND"},
             "blocked_routes": [{"to": "office_a"}]},
            {"failure": {"status": "BLOCKED",
                         "step": {"action": "navigate", "location": "office_a"}}})
        bad = {"steps": [{"action": "search", "object": "ball", "colour": "red",
                          "locations": ["office_a", "laboratory", "office_b",
                                        "meeting_room"]},
                         {"action": "report", "message": "x"}]}
        problems = check_replan(bad, facts)
        self.assertTrue(any("office_a cannot be reached" in p for p in problems))
        self.assertTrue(any("laboratory was already searched" in p for p in problems))
        good = {"steps": [{"action": "search", "object": "ball", "colour": "red",
                           "locations": ["office_b", "meeting_room"]},
                          {"action": "return_home"},
                          {"action": "report", "message": "x"}]}
        self.assertEqual(check_replan(good, facts), [])

    def test_replan_after_found_only_reports(self):
        from planning import check_replan, replan_situation
        _, facts = replan_situation(
            "find the red ball", {"search_results": {"office_a": "FOUND"}},
            {"failure": {"status": "BLOCKED",
                         "step": {"action": "navigate", "location": "reception"}}})
        self.assertEqual(facts["remaining"], [])
        self.assertTrue(check_replan({"steps": [{"action": "navigate",
                                                 "location": "office_b"}]}, facts))


if __name__ == "__main__":
    unittest.main()
