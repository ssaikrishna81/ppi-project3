"""Navigator tests on the fake 2-D e-puck (tests/fake_webots.py).

These check route planning, heading convention, arrival, obstacle handling,
the progress watchdog and the time budget. They do not replace runs in
Webots; they catch logic errors before tuning there.
"""
import math
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
CONTROLLER_DIR = TESTS_DIR.parent / "controllers" / "project3_student"
sys.path.insert(0, str(CONTROLLER_DIR))
sys.path.insert(0, str(TESTS_DIR))

import navigation  # noqa: E402
from fake_webots import FakeRobot  # noqa: E402
from navigation import (  # noqa: E402
    BLOCKED,
    FAILED,
    RUNNING,
    SUCCEEDED,
    TIMEOUT,
    Navigator,
    RouteGraph,
    heading_from_compass,
)

navigation.DEBUG = False


def make(**kwargs):
    robot = FakeRobot(**kwargs)
    nav = Navigator(robot, robot.left, robot.right, 32)
    return robot, nav


def drive(robot, nav, location, max_seconds=400):
    status = nav.start(location)
    steps = int(max_seconds / robot.dt)
    for _ in range(steps):
        if status != RUNNING:
            return status
        robot.step()
        status = nav.update()
    return status


class RouteGraphTests(unittest.TestCase):
    def setUp(self):
        self.graph = RouteGraph(_load_graph())

    def test_direct_route_to_corridor_uses_centre_edge(self):
        path = self.graph.shortest_path("reception", "main_corridor")
        self.assertEqual(path, ["reception", "centre_south", "centre",
                                "centre_north", "main_corridor"])

    def test_office_route_avoids_centre_north(self):
        path = self.graph.shortest_path("reception", "office_b")
        self.assertNotIn("centre_north", path)
        self.assertEqual(path[-1], "office_b")

    def test_blocked_edge_uses_bypass(self):
        blocked = {frozenset(("centre", "centre_north"))}
        path = self.graph.shortest_path("reception", "main_corridor", blocked)
        self.assertTrue(any(n.startswith("centre_bypass") for n in path), path)

    def test_no_route_returns_none(self):
        blocked = {frozenset(("reception", "centre_south"))}
        self.assertIsNone(self.graph.shortest_path("reception", "laboratory",
                                                   blocked))

    def test_backup_doors_never_used_on_normal_routes(self):
        for a in self.graph.nodes:
            for b in self.graph.nodes:
                if a.startswith("door_") or b.startswith("door_"):
                    continue
                path = self.graph.shortest_path(a, b)
                self.assertFalse(any(n.startswith("door_") for n in path), (a, b, path))

    def test_backup_door_when_main_doorway_blocked(self):
        blocked = {frozenset(("centre", "office_a_entry")),
                   frozenset(("centre_north", "office_a_entry"))}
        path = self.graph.shortest_path("reception", "office_a", blocked)
        self.assertIn("door_west", path)

    def test_every_location_reachable_from_reception(self):
        for location in ("laboratory", "office_a", "office_b", "meeting_room",
                         "main_corridor", "reception"):
            self.assertIsNotNone(self.graph.shortest_path("reception", location))


def _load_graph():
    import json
    root = TESTS_DIR.parent
    return json.loads((root / "config" / "navigation_graph.json").read_text())


class HeadingTests(unittest.TestCase):
    def test_compass_convention(self):
        for degrees in (0, 45, 90, 135, 180, -90, -30):
            yaw = math.radians(degrees)
            north = [math.sin(yaw), math.cos(yaw), 0.0]
            self.assertAlmostEqual(
                navigation.wrap_angle(heading_from_compass(north) - yaw), 0.0)


class NavigatorSimTests(unittest.TestCase):
    def test_reaches_laboratory(self):
        robot, nav = make()
        self.assertEqual(drive(robot, nav, "laboratory"), SUCCEEDED)
        self.assertEqual(robot.collisions, 0)
        lx, ly = nav.locations["laboratory"]
        self.assertLess(math.hypot(robot.x - lx, robot.y - ly), 0.12)
        self.assertLessEqual(robot.left.max_seen, navigation.MAX_WHEEL_SPEED)
        self.assertLessEqual(robot.right.max_seen, navigation.MAX_WHEEL_SPEED)

    def test_starts_facing_any_direction(self):
        robot, nav = make(yaw=math.radians(-120))
        self.assertEqual(drive(robot, nav, "meeting_room"), SUCCEEDED)
        self.assertEqual(robot.collisions, 0)

    def test_full_tour_without_collisions(self):
        robot, nav = make()
        for location in ("laboratory", "office_a", "office_b", "meeting_room",
                         "main_corridor", "reception"):
            status = drive(robot, nav, location)
            self.assertEqual(status, SUCCEEDED, (location, nav.evidence))
        self.assertEqual(robot.collisions, 0)

    def test_detects_obstacle_and_reroutes(self):
        robot, nav = make()
        status = drive(robot, nav, "main_corridor")
        self.assertEqual(status, SUCCEEDED, nav.evidence)
        self.assertEqual(robot.collisions, 0)
        self.assertGreaterEqual(nav.evidence["reroutes"], 1)
        self.assertIn("centre-centre_north", nav.evidence["blocked_edges"])

    def test_remembers_blocked_edge_on_next_trip(self):
        robot, nav = make()
        drive(robot, nav, "main_corridor")
        drive(robot, nav, "reception")
        status = drive(robot, nav, "main_corridor")
        self.assertEqual(status, SUCCEEDED, nav.evidence)
        self.assertEqual(nav.evidence["reroutes"], 0)

    def test_blocked_when_no_route_remains(self):
        robot, nav = make()
        # A wall across the only corridor out of reception.
        robot.obstacles.append(("TEST_WALL", -1.5, 1.5, -2.2, -2.0))
        status = drive(robot, nav, "laboratory")
        self.assertEqual(status, BLOCKED, nav.evidence)
        self.assertEqual(robot.collisions, 0)
        self.assertIn("reroutes", nav.evidence)

    def test_progress_watchdog_catches_stuck_wheels(self):
        robot, nav = make(wheel_efficiency=0.0)
        robot.yaw = math.atan2(-1.15 + 2.8, 0.8)  # already facing the route
        status = drive(robot, nav, "laboratory")
        self.assertEqual(status, BLOCKED, nav.evidence)
        self.assertTrue(nav.evidence["block_reasons"])
        self.assertTrue(all("no progress" in r for r in nav.evidence["block_reasons"]),
                        nav.evidence)

    def test_time_budget_gives_timeout(self):
        robot, nav = make(wheel_efficiency=0.3)
        status = drive(robot, nav, "office_b", max_seconds=2000)
        self.assertEqual(status, TIMEOUT, nav.evidence)

    def test_unknown_location_fails(self):
        robot, nav = make()
        self.assertEqual(nav.start("car_park"), FAILED)

    def test_missing_gps_fails(self):
        robot, nav = make()
        robot.gps_valid = False
        self.assertEqual(drive(robot, nav, "laboratory"), FAILED)
        self.assertIn("GPS", nav.evidence["reason"])

    def test_cancel_stops_motors(self):
        robot, nav = make()
        nav.start("office_b")
        for _ in range(100):
            robot.step()
            nav.update()
        nav.cancel()
        self.assertEqual(robot.left.velocity, 0.0)
        self.assertEqual(robot.right.velocity, 0.0)
        self.assertEqual(nav.update(), FAILED)

    def test_restart_after_cancel_does_not_turn_back(self):
        # Webots test 2026-10-20: stopped between office_b_entry and centre,
        # then asked for reception; the old code drove back to the nearest
        # node (centre_bypass_east) first.
        robot, nav = make(x=3.39, y=1.96)
        robot.yaw = math.pi
        nav.start("reception")
        while robot.time < 300:
            robot.step()
            nav.update()
            if nav.edge == ("office_b_entry", "centre") and robot.x < 1.2:
                break
        nav.cancel()
        self.assertEqual(nav.edge, ("office_b_entry", "centre"))
        status = drive(robot, nav, "reception")
        self.assertEqual(status, SUCCEEDED)
        self.assertEqual(nav.evidence["route"][0], "centre")

    def test_backup_door_used_only_when_main_doorway_blocked(self):
        # Same blockage as worlds/test_blocked_office_a.wbt: the bar turned
        # across office A's main doorway.
        from fake_webots import load_obstacles
        obstacles = [o for o in load_obstacles() if o[0] != "BLOCKING_OBSTACLE"]
        obstacles.append(("BAR_ACROSS_DOORWAY", -2.15, -1.85, -0.025, 1.125))
        robot, nav = make(obstacles=obstacles)
        status = drive(robot, nav, "office_a", max_seconds=600)
        self.assertEqual(status, SUCCEEDED, nav.evidence)
        self.assertIn("door_west", nav.evidence["route"])
        self.assertEqual(robot.collisions, 0)

    def test_already_at_destination(self):
        robot, nav = make()
        self.assertEqual(drive(robot, nav, "reception"), SUCCEEDED)


if __name__ == "__main__":
    unittest.main()
