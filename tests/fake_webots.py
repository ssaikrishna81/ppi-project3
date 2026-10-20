"""A tiny 2-D stand-in for the Webots e-puck, for navigation tests only.

It integrates differential-drive kinematics, reports GPS and Compass the way
Webots does (ENU, compass = north in the robot frame), and fakes the eight
proximity sensors by ray-casting against the boxes, spheres and cylinders
parsed from the real world file. Contact with an object stops the robot.

This is a test fixture: it is not evidence of behaviour in Webots.
"""
import math
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORLD = PROJECT_ROOT / "worlds" / "project_genai_embodied_assistant.wbt"

WHEEL_RADIUS = 0.02
AXLE = 0.052
ROBOT_RADIUS = 0.037
# Sensor directions relative to the robot heading (ps0..ps7), from the
# E-puck.proto layout: front pair, 45 deg pair, sides, rear pair.
SENSOR_ANGLES = [math.radians(a) for a in (-17, -45, -90, -150, 150, 90, 45, 17)]
SENSOR_RANGE = 0.06
FREE_VALUE = 65.0


def load_obstacles(skip=("FLOOR", "RED_BALL")):
    """Axis-aligned rectangles (name, xmin, xmax, ymin, ymax) from the world."""
    text = WORLD.read_text(encoding="utf-8")
    obstacles = []
    for match in re.finditer(r"DEF (\w+) Solid \{(.*?)\n\}", text, re.S):
        name, body = match.group(1), match.group(2)
        if name in skip:
            continue
        t = re.search(r"translation ([-\d.]+) ([-\d.]+)", body)
        x, y = float(t.group(1)), float(t.group(2))
        bounding = body[body.find("boundingObject"):]
        box = re.search(r"Box \{\s*size ([-\d.]+) ([-\d.]+)", bounding)
        sphere = re.search(r"Sphere \{\s*radius ([-\d.]+)", bounding)
        cylinder = re.search(r"Cylinder \{\s*height [-\d.]+\s*radius ([-\d.]+)",
                             bounding)
        if box:
            hx, hy = float(box.group(1)) / 2, float(box.group(2)) / 2
        elif sphere:
            hx = hy = float(sphere.group(1))
        elif cylinder:
            hx = hy = float(cylinder.group(1))
        else:
            continue
        obstacles.append((name, x - hx, x + hx, y - hy, y + hy))
    return obstacles


def _inside(px, py, rect):
    return rect[1] <= px <= rect[2] and rect[3] <= py <= rect[4]


def _circle_hits(px, py, radius, rect):
    dx = max(rect[1] - px, 0.0, px - rect[2])
    dy = max(rect[3] - py, 0.0, py - rect[4])
    return math.hypot(dx, dy) < radius


class FakeMotor:
    def __init__(self):
        self.velocity = 0.0
        self.max_seen = 0.0

    def setPosition(self, _):
        pass

    def setVelocity(self, value):
        self.velocity = value
        self.max_seen = max(self.max_seen, abs(value))

    def getVelocity(self):
        return self.velocity


class _Device:
    def __init__(self, read):
        self._read = read

    def enable(self, _):
        pass

    def getValues(self):
        return self._read()

    def getValue(self):
        return self._read()


class FakeCamera:
    """Pinhole rendering of balls and boxes as seen from the fake robot.

    objects: list of dicts {x, y, shape: "ball"|"box", size (m), bgr}.
    No occlusion by walls; nearer objects are drawn over farther ones.
    """
    WIDTH, HEIGHT = 320, 240
    HEIGHT_M = 0.028        # camera height above the floor
    FOV = 0.84

    def __init__(self, robot, objects):
        self.robot = robot
        self.objects = objects
        self.focal = (self.WIDTH / 2) / math.tan(self.FOV / 2)

    def enable(self, _):
        pass

    def getWidth(self):
        return self.WIDTH

    def getHeight(self):
        return self.HEIGHT

    def getImage(self):
        import cv2
        import numpy as np
        img = np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8)
        img[: self.HEIGHT // 2] = (205, 200, 200)
        img[self.HEIGHT // 2:] = (85, 70, 60)
        cos_y, sin_y = math.cos(-self.robot.yaw), math.sin(-self.robot.yaw)
        visible = []
        for obj in self.objects:
            dx, dy = obj["x"] - self.robot.x, obj["y"] - self.robot.y
            forward = dx * cos_y - dy * sin_y
            left = dx * sin_y + dy * cos_y
            if forward < 0.1:
                continue
            visible.append((forward, left, obj))
        for forward, left, obj in sorted(visible, key=lambda v: -v[0]):
            u = self.WIDTH / 2 - self.focal * left / forward
            centre_z = obj["size"] / 2
            v = self.HEIGHT / 2 - self.focal * (centre_z - self.HEIGHT_M) / forward
            half = self.focal * obj["size"] / 2 / forward
            if obj["shape"] == "ball":
                cv2.circle(img, (int(round(u)), int(round(v))),
                           max(1, int(round(half))), obj["bgr"], -1)
            else:
                cv2.rectangle(img, (int(u - half), int(v - half)),
                              (int(u + half), int(v + half)), obj["bgr"], -1)
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return bgra.tobytes()


class FakeRobot:
    def __init__(self, x=-0.8, y=-2.8, yaw=0.0, obstacles=None,
                 wheel_efficiency=1.0, time_step=32):
        self.x, self.y, self.yaw = x, y, yaw
        self.time = 0.0
        self.dt = time_step / 1000.0
        self.obstacles = load_obstacles() if obstacles is None else obstacles
        self.wheel_efficiency = wheel_efficiency
        self.left = FakeMotor()
        self.right = FakeMotor()
        self.collisions = 0
        self.gps_valid = True
        self.path = []
        self.devices = {
            "gps": _Device(self._gps),
            "compass": _Device(self._compass),
        }
        for i, angle in enumerate(SENSOR_ANGLES):
            self.devices[f"ps{i}"] = _Device(lambda a=angle: self._proximity(a))

    def getDevice(self, name):
        return self.devices[name]

    def getTime(self):
        return self.time

    def getBasicTimeStep(self):
        return self.dt * 1000.0

    def _gps(self):
        if not self.gps_valid:
            return [math.nan, math.nan, math.nan]
        return [self.x, self.y, 0.0]

    def _compass(self):
        if not self.gps_valid:
            return [math.nan, math.nan, math.nan]
        return [math.sin(self.yaw), math.cos(self.yaw), 0.0]

    def _proximity(self, angle):
        direction = self.yaw + angle
        for step in range(13):
            d = step * SENSOR_RANGE / 12
            px = self.x + (ROBOT_RADIUS + d) * math.cos(direction)
            py = self.y + (ROBOT_RADIUS + d) * math.sin(direction)
            if any(_inside(px, py, r) for r in self.obstacles):
                return FREE_VALUE + (SENSOR_RANGE - d) / SENSOR_RANGE * 2000.0
        return FREE_VALUE

    def step(self):
        wl = self.left.velocity * self.wheel_efficiency
        wr = self.right.velocity * self.wheel_efficiency
        v = WHEEL_RADIUS * (wl + wr) / 2.0
        w = WHEEL_RADIUS * (wr - wl) / AXLE
        yaw = self.yaw + w * self.dt
        nx = self.x + v * math.cos(yaw) * self.dt
        ny = self.y + v * math.sin(yaw) * self.dt
        if any(_circle_hits(nx, ny, ROBOT_RADIUS, r) for r in self.obstacles):
            self.collisions += 1
            self.yaw = yaw
        else:
            self.x, self.y, self.yaw = nx, ny, yaw
        self.time += self.dt
        self.path.append((self.x, self.y))
