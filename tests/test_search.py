"""SearchController tests: fake robot + fake pinhole camera.

The scene objects are placed by each test (the fake camera has no walls),
so these check the policy logic - scan, centre, confirm, second viewpoint,
outcome rules - not Webots image quality.
"""
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
from search import (  # noqa: E402
    FOUND,
    NOT_FOUND,
    RUNNING,
    SUCCEEDED,
    UNCERTAIN,
    SearchController,
)
from world_state import WorldState  # noqa: E402

navigation.DEBUG = False

RED_BALL = {"shape": "ball", "size": 0.18, "bgr": (13, 13, 242)}
RED_BOX = {"shape": "box", "size": 0.12, "bgr": (15, 15, 217)}
GREEN_BALL = {"shape": "ball", "size": 0.18, "bgr": (38, 204, 13)}


def place(template, x, y):
    return dict(template, x=x, y=y)


def setup(location, objects, **robot_kwargs):
    nodes = {"laboratory": (-3.5, -2.0), "office_b": (3.5, 2.0),
             "meeting_room": (3.5, -2.0)}
    x, y = nodes[location]
    robot = FakeRobot(x=x, y=y, **robot_kwargs)
    camera = FakeCamera(robot, objects)
    robot.devices["camera"] = camera
    nav = Navigator(robot, robot.left, robot.right, 32)
    ws = WorldState()
    ws.current_location = location
    search = SearchController(nav, ObjectPerception(camera, 32), ws)
    return robot, search


def run(robot, search, max_seconds=150):
    status = search.start("ball", "red")
    for _ in range(int(max_seconds / robot.dt)):
        if status != RUNNING:
            return status
        robot.step()
        status = search.update()
    raise AssertionError("search did not finish")


class SearchTests(unittest.TestCase):
    def test_finds_ball_in_room(self):
        robot, search = setup("laboratory", [place(RED_BALL, -4.35, -1.35)])
        self.assertEqual(run(robot, search), SUCCEEDED)
        self.assertEqual(search.outcome, FOUND, search.evidence)
        det = search.evidence["detection"]
        ex, ey = det["estimated_position"]
        self.assertLess(math.hypot(ex + 4.35, ey + 1.35), 0.4, det)
        self.assertEqual(det["estimated_room"], "laboratory")
        # Found at the first viewpoint, so the robot never left the pad.
        self.assertEqual(len(search.evidence["viewpoints"]), 1)
        self.assertEqual(robot.collisions, 0)

    def test_finds_ball_behind_the_robot(self):
        robot, search = setup("laboratory", [place(RED_BALL, -2.75, -2.85)],
                              yaw=math.pi / 2)
        run(robot, search)
        self.assertEqual(search.outcome, FOUND, search.evidence)

    def test_rejects_red_box_and_green_ball(self):
        robot, search = setup("office_b", [place(RED_BOX, 4.25, 2.2),
                                           place(GREEN_BALL, 3.15, 1.55)])
        self.assertEqual(run(robot, search), SUCCEEDED)
        self.assertEqual(search.outcome, NOT_FOUND, search.evidence)
        seen = search.evidence["distractor_frames"]
        self.assertIn("red_upright", seen)
        # Both viewpoints were scanned completely before NOT_FOUND.
        views = search.evidence["viewpoints"]
        self.assertEqual([v["viewpoint"] for v in views],
                         ["office_b", "office_b_inner", "office_b_far"])
        self.assertTrue(all(v["scanned"] for v in views))

    def test_ball_next_to_red_box(self):
        robot, search = setup("office_b", [place(RED_BOX, 4.25, 2.2),
                                           place(RED_BALL, 3.65, 1.15)])
        run(robot, search)
        self.assertEqual(search.outcome, FOUND, search.evidence)
        ex, ey = search.evidence["detection"]["estimated_position"]
        self.assertLess(math.hypot(ex - 3.65, ey - 1.15), 0.4)

    def test_empty_room_is_not_found(self):
        robot, search = setup("meeting_room", [])
        run(robot, search)
        self.assertEqual(search.outcome, NOT_FOUND, search.evidence)
        self.assertEqual(search.evidence["incomplete"], [])

    def test_stalled_rotation_is_uncertain(self):
        robot, search = setup("meeting_room", [], wheel_efficiency=0.0)
        run(robot, search, max_seconds=200)
        self.assertEqual(search.outcome, UNCERTAIN, search.evidence)
        self.assertTrue(any("stalled" in s or "not reached" in s
                            for s in search.evidence["incomplete"]))

    def test_far_ball_in_another_room_is_a_far_sighting(self):
        # About 7 m away (in the lab), too small to confirm: says nothing
        # about the meeting room, but is reported.
        robot, search = setup("meeting_room", [place(RED_BALL, -3.5, -2.0)])
        run(robot, search)
        self.assertEqual(search.outcome, NOT_FOUND, search.evidence)
        self.assertTrue(search.evidence["far_sightings"])
        self.assertEqual(search.evidence["unresolved"], [])

    def test_ball_seen_through_doorway_reports_its_room(self):
        # Seen in Webots: from the meeting-room pad the ball at office_b_south
        # is visible through the doorway at x = 3.3-3.9.
        robot, search = setup("meeting_room", [place(RED_BALL, 3.65, 1.15)])
        run(robot, search)
        self.assertEqual(search.outcome, FOUND, search.evidence)
        self.assertEqual(search.evidence["detection"]["estimated_room"], "office_b")

    def test_cancel_stops_motors(self):
        robot, search = setup("laboratory", [])
        search.start("ball", "red")
        for _ in range(50):
            robot.step()
            search.update()
        search.cancel()
        self.assertEqual(robot.left.velocity, 0.0)
        self.assertEqual(robot.right.velocity, 0.0)


if __name__ == "__main__":
    unittest.main()
