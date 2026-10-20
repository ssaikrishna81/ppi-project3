"""RobotSkills tests on the fake robot and camera, plus one full plan run
through PlanValidator -> TaskExecutive -> RobotSkills."""
import math
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "controllers" / "project3_student"))
sys.path.insert(0, str(TESTS_DIR))

import navigation  # noqa: E402
from fake_webots import FakeCamera, FakeRobot  # noqa: E402
from navigation import Navigator  # noqa: E402
from perception import ObjectPerception  # noqa: E402
from plan_validator import PlanValidator  # noqa: E402
from robot_skills import RobotSkills  # noqa: E402
from semantic_memory import SemanticMemory  # noqa: E402
from task_executive import RUNNING, SUCCEEDED, TaskExecutive  # noqa: E402
from world_state import WorldState  # noqa: E402

navigation.DEBUG = False

RED_BALL = {"shape": "ball", "size": 0.18, "bgr": (13, 13, 242)}
RED_BOX = {"shape": "box", "size": 0.12, "bgr": (15, 15, 217)}


class FakeSpeech:
    def __init__(self):
        self.said = []

    def speak(self, text):
        self.said.append(text)


def build(objects, x=-0.8, y=-2.8, location="reception"):
    robot = FakeRobot(x=x, y=y)
    camera = FakeCamera(robot, objects)
    nav = Navigator(robot, robot.left, robot.right, 32)
    ws = WorldState()
    ws.current_location = location
    speech = FakeSpeech()
    memory = SemanticMemory()
    skills = RobotSkills(nav, ObjectPerception(camera, 32), memory, speech, ws)
    return robot, skills, ws, speech, memory


def run_step(robot, skills, step, max_seconds=400):
    status = skills.start(step)
    for _ in range(int(max_seconds / robot.dt)):
        if status != RUNNING:
            return status
        robot.step()
        status = skills.update()
    raise AssertionError("skill did not finish")


class RobotSkillTests(unittest.TestCase):
    def test_navigate_updates_location_and_evidence(self):
        robot, skills, ws, _, _ = build([])
        status = run_step(robot, skills, {"action": "navigate", "location": "laboratory"})
        self.assertEqual(status, SUCCEEDED)
        self.assertEqual(ws.current_location, "laboratory")
        self.assertIn("arrived", skills.evidence["reason"])

    def test_search_records_result_and_memory(self):
        robot, skills, ws, _, memory = build([dict(RED_BALL, x=-4.35, y=-1.35)],
                                             x=-3.5, y=-2.0, location="laboratory")
        run_step(robot, skills, {"action": "search", "object": "ball", "colour": "red"})
        self.assertEqual(ws.last_search_result, "FOUND")
        self.assertEqual(ws.search_results, {"laboratory": "FOUND"})
        self.assertEqual(memory.query("red ball")["location"], "laboratory")

    def test_observe_lists_objects_without_a_search_result(self):
        robot, skills, ws, _, memory = build(
            [dict(RED_BALL, x=3.65, y=1.15), dict(RED_BOX, x=4.25, y=2.2)],
            x=3.5, y=2.0, location="office_b")
        status = run_step(robot, skills, {"action": "observe"})
        self.assertEqual(status, SUCCEEDED)
        self.assertIsNone(ws.last_search_result)
        self.assertIn("red ball", skills.evidence["objects"])
        self.assertIn("red upright", skills.evidence["objects"])
        self.assertEqual(memory.query("red ball")["location"], "office_b")
        # Observe never leaves its spot.
        self.assertLess(math.hypot(robot.x - 3.5, robot.y - 2.0), 0.05)

    def test_report_ignores_unsupported_planner_claim(self):
        robot, skills, ws, speech, _ = build([])
        ws.current_location = "laboratory"
        ws.record_search("NOT_FOUND", {"target": "red ball"})
        run_step(robot, skills, {"action": "report",
                                 "message": "I found the red ball in the lab!"})
        spoken = speech.said[-1]
        self.assertIn("did not see the red ball in the laboratory", spoken)
        self.assertNotIn("I found", spoken)
        self.assertEqual(skills.evidence["planner_message"],
                         "I found the red ball in the lab!")

    def test_ask_speaks_question(self):
        robot, skills, _, speech, _ = build([])
        run_step(robot, skills, {"action": "ask", "question": "Which ball?"})
        self.assertEqual(speech.said, ["Which ball?"])

    def test_stop_halts_motion(self):
        robot, skills, _, _, _ = build([])
        skills.start({"action": "navigate", "location": "office_b"})
        for _ in range(60):
            robot.step()
            skills.update()
        self.assertEqual(skills.start({"action": "stop"}), SUCCEEDED)
        self.assertEqual((robot.left.velocity, robot.right.velocity), (0.0, 0.0))

    def test_search_with_locations_stops_at_found(self):
        robot, skills, ws, _, _ = build([dict(RED_BALL, x=3.65, y=1.15)])
        status = run_step(robot, skills, {
            "action": "search", "object": "ball", "colour": "red",
            "locations": ["office_b", "meeting_room", "laboratory"]}, max_seconds=900)
        self.assertEqual(status, SUCCEEDED)
        self.assertEqual(skills.evidence["outcome"], "FOUND")
        self.assertEqual([r["room"] for r in skills.evidence["rooms"]], ["office_b"])
        self.assertEqual(ws.last_search_result, "FOUND")
        self.assertEqual(ws.current_location, "office_b")

    def test_search_with_locations_not_found_anywhere(self):
        robot, skills, ws, _, _ = build([])
        run_step(robot, skills, {"action": "search", "object": "ball", "colour": "red",
                                 "locations": ["laboratory", "meeting_room"]},
                 max_seconds=900)
        self.assertEqual(skills.evidence["outcome"], "NOT_FOUND")
        self.assertEqual(ws.search_results, {"laboratory": "NOT_FOUND",
                                             "meeting_room": "NOT_FOUND"})

    def test_unknown_action_fails(self):
        _, skills, _, _, _ = build([])
        self.assertEqual(skills.start({"action": "fly"}), "FAILED")


class FullPlanTests(unittest.TestCase):
    def test_spec_example_end_to_end(self):
        """Lab first; ball in the meeting room -> conditional takes 'then'."""
        plan = {
            "version": "1.0",
            "goal": "find the red ball; check the laboratory first",
            "steps": [
                {"action": "navigate", "location": "laboratory"},
                {"action": "search", "object": "ball", "colour": "red"},
                {"action": "conditional",
                 "condition": {"source": "last_search_result",
                               "operator": "found", "value": False},
                 "then": [{"action": "navigate", "location": "meeting_room"},
                          {"action": "search", "object": "ball", "colour": "red"}],
                 "else": []},
                {"action": "return_home"},
                {"action": "report", "message": "Report the evidence-based result."},
            ],
        }
        self.assertTrue(PlanValidator().validate(plan)[0])
        robot, skills, ws, speech, _ = build([dict(RED_BALL, x=4.55, y=-1.35)])
        executive = TaskExecutive(skills, ws, clock=robot.getTime)
        ws.reset_for_new_goal(plan["goal"])
        executive.start(plan)
        status = RUNNING
        while status == RUNNING and robot.time < 1500:
            robot.step()
            status = executive.update()
        self.assertEqual(status, SUCCEEDED, executive.failure_evidence())
        self.assertEqual(ws.search_results.get("meeting_room"), "FOUND")
        self.assertNotEqual(ws.search_results.get("laboratory"), "FOUND")
        self.assertEqual(ws.current_location, "reception")
        self.assertIn("found the red ball in the meeting room", speech.said[-1])
        self.assertEqual(robot.collisions, 0)


if __name__ == "__main__":
    unittest.main()
