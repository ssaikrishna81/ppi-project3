"""TraceLogger writes valid JSON Lines with samples and events."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "controllers" / "project3_student"))
sys.path.insert(0, str(TESTS_DIR))

import navigation  # noqa: E402
from fake_webots import FakeRobot  # noqa: E402
from navigation import Navigator  # noqa: E402
from task_executive import RUNNING, TaskExecutive  # noqa: E402
from trace_logger import TraceLogger  # noqa: E402
from world_state import WorldState  # noqa: E402

navigation.DEBUG = False


class NavOnlySkills:
    """Just enough of RobotSkills to drive navigate steps."""

    def __init__(self, navigator, world_state):
        self.navigator = navigator
        self.world_state = world_state
        self.evidence = None
        self.step = None

    def start(self, step):
        self.step = step
        return self.navigator.start(step["location"])

    def update(self):
        status = self.navigator.update()
        self.evidence = self.navigator.evidence
        if status == "SUCCEEDED":
            self.world_state.current_location = self.step["location"]
        return status

    def cancel(self):
        self.navigator.cancel()


class TraceLoggerTests(unittest.TestCase):
    def test_writes_samples_and_events(self):
        robot = FakeRobot()
        nav = Navigator(robot, robot.left, robot.right, 32)
        ws = WorldState()
        executive = TaskExecutive(NavOnlySkills(nav, ws), ws, clock=robot.getTime)
        with tempfile.TemporaryDirectory() as tmp:
            trace = TraceLogger(robot, nav, ws, executive, meta={"test": True},
                                log_dir=tmp)
            ws.reset_for_new_goal("go to lab")
            executive.start({"steps": [{"action": "navigate",
                                        "location": "laboratory"}]})
            status = RUNNING
            while status == RUNNING and robot.time < 300:
                robot.step()
                status = executive.update()
                trace.update()
            trace.update()
            trace.close()
            lines = [json.loads(line) for line in
                     trace.path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(lines[0]["type"], "meta")
        samples = [r for r in lines if r["type"] == "sample"]
        events = [r for r in lines if r["type"] == "event"]
        self.assertGreater(len(samples), 100)
        self.assertEqual(len(samples[0]["ps"]), 8)
        names = {e["event"] for e in events}
        self.assertIn("navigation_outcome", names)
        self.assertIn("step_outcome", names)
        self.assertIn("SUCCEEDED", [e.get("status") for e in events])
        # Samples are about 0.1 s apart.
        gap = samples[11]["t"] - samples[10]["t"]
        self.assertAlmostEqual(gap, 0.1, delta=0.04)


if __name__ == "__main__":
    unittest.main()
