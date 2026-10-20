"""Non-blocking navigation support for named locations.

The supplied coordinates are only high-level targets. Students must implement
a safe feedback-based navigation method rather than teleporting the robot.

How it works
------------
start(location) plans a shortest route over config/navigation_graph.json
(Dijkstra, Euclidean edge length) from the graph node nearest the robot.
update() then does one control step towards the current waypoint using GPS
position and Compass heading, and returns RUNNING until an outcome is known:

SUCCEEDED  within ARRIVAL_TOLERANCE_M of the destination (GPS).
BLOCKED    front sensors confirmed an obstacle, or the GPS progress watchdog
           saw no movement, and no route avoiding the blocked edge remains
           (after at most MAX_REROUTES re-plans).
TIMEOUT    the route took longer than its time budget.
FAILED     unknown destination, or GPS/Compass gave no valid reading.

Each outcome leaves a plain dict in self.evidence describing why.
"""
import heapq
import json
import math
from pathlib import Path

RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
BLOCKED = "BLOCKED"
TIMEOUT = "TIMEOUT"

# E-puck geometry from E-puck.proto (Webots R2025a).
WHEEL_RADIUS_M = 0.02
MAX_WHEEL_SPEED = 6.28          # rad/s, motor maxVelocity

# Speed limit: 80% of the motor limit (about 0.10 m/s) so steering
# corrections never need more than the motor can give.
CRUISE_SPEED = 5.0
TURN_SPEED = 2.5                # rad/s per wheel when rotating in place
MIN_TURN_SPEED = 0.8            # avoids stalling on tiny heading errors
HEADING_GAIN = 4.0              # wheel rad/s per rad of heading error
TURN_IN_PLACE_RAD = 0.5         # above ~29 deg, stop and rotate first
SLOWDOWN_DISTANCE_M = 0.25      # ease off over the last 25 cm
MIN_SLOWDOWN_SCALE = 0.3

# Pads are 0.75 m squares, so 0.12 m puts the robot well inside its pad.
ARRIVAL_TOLERANCE_M = 0.12
# If an obstacle stops the robot this close to the destination it is still
# on the 0.75 m pad (half-width 0.375 m), e.g. the meeting-room pad is only
# 0.1 m from the table. Counts as arrival, and the evidence says so.
NEAR_ARRIVAL_M = 0.30
# Intermediate nodes only keep the robot on the route; a looser tolerance
# avoids circling a node that sits near a wall.
WAYPOINT_TOLERANCE_M = 0.15

# Proximity: ps0/ps7 face forward, ps1/ps6 about 45 deg to the sides.
# The starter uses 80 as the front threshold; readings in open space are
# lower (to be confirmed from the [NAV] log and recorded for the report).
FRONT_SENSORS = (0, 1, 6, 7)
REAR_SENSORS = (3, 4)
OBSTACLE_THRESHOLD = 80.0
OBSTACLE_CONFIRM_STEPS = 3      # consecutive readings (~0.1 s) to ignore noise

# Back away from a confirmed obstacle before turning towards a new route.
BACKOFF_SPEED = 2.0
BACKOFF_SECONDS = 1.5           # about 6 cm at this speed
MAX_REROUTES = 2

# Progress watchdog, independent of the proximity sensors: while driving
# forward the robot must move PROGRESS_MIN_M within PROGRESS_WINDOW_S.
PROGRESS_WINDOW_S = 4.0
PROGRESS_MIN_M = 0.05

# Time budget: route length at half the cruise speed, plus fixed slack for
# turning on the spot at each waypoint.
BUDGET_SPEED_MPS = 0.05
BUDGET_SLACK_S = 20.0
BUDGET_PER_WAYPOINT_S = 6.0

GPS_WAIT_S = 1.0
# The robot counts as "on" an edge if within this distance of its line.
EDGE_TOLERANCE_M = 0.35                # GPS/Compass need a step or two after enable
DEBUG = True
DEBUG_PERIOD_S = 1.0


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def heading_from_compass(values):
    """Robot yaw in the world frame (0 = facing +x, counter-clockwise).

    In Webots' ENU frame north is +y. The compass returns that direction in
    the robot frame, which is (sin yaw, cos yaw) for a robot facing +x at
    yaw 0, so yaw = atan2(x, y).
    """
    return math.atan2(values[0], values[1])


def _distance_to_segment(x, y, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length2))
    return math.hypot(x - (ax + t * dx), y - (ay + t * dy))


class RouteGraph:
    """Waypoint graph with Dijkstra search; no Webots dependency.

    "backup_edges" (the internal doors lab <-> office A and meeting room <->
    office B) are only used when no route exists without them, so they never
    change a normal trip; they only give a second way in when a room's main
    doorway is blocked.
    """

    def __init__(self, graph):
        self.nodes = {name: tuple(xy) for name, xy in graph["nodes"].items()}
        self.adjacent = {name: set() for name in self.nodes}
        self.backup = set()
        for a, b in graph["edges"]:
            self.adjacent[a].add(b)
            self.adjacent[b].add(a)
        for a, b in graph.get("backup_edges", []):
            self.adjacent[a].add(b)
            self.adjacent[b].add(a)
            self.backup.add(frozenset((a, b)))

    def shortest_path(self, start, goal, blocked=()):
        """Return the node list from start to goal, or None.

        blocked holds frozenset({a, b}) edges that must not be used. Backup
        edges are tried only if the normal graph has no route.
        """
        path = self._dijkstra(start, goal, set(blocked) | self.backup)
        if path is None and self.backup:
            path = self._dijkstra(start, goal, blocked)
        return path

    def distance(self, a, b):
        (ax, ay), (bx, by) = self.nodes[a], self.nodes[b]
        return math.hypot(bx - ax, by - ay)

    def nearest_node(self, x, y):
        return min(self.nodes, key=lambda n: math.hypot(self.nodes[n][0] - x,
                                                        self.nodes[n][1] - y))

    def _dijkstra(self, start, goal, blocked):
        if start not in self.nodes or goal not in self.nodes:
            return None
        best = {start: 0.0}
        previous = {}
        queue = [(0.0, start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if node == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(previous[path[-1]])
                return path[::-1]
            if cost > best.get(node, math.inf):
                continue
            for nxt in self.adjacent[node]:
                if frozenset((node, nxt)) in blocked:
                    continue
                new_cost = cost + self.distance(node, nxt)
                if new_cost < best.get(nxt, math.inf):
                    best[nxt] = new_cost
                    previous[nxt] = node
                    heapq.heappush(queue, (new_cost, nxt))
        return None

    def path_length(self, path):
        return sum(self.distance(a, b) for a, b in zip(path, path[1:]))


class Navigator:
    def __init__(self, robot, left_motor, right_motor, timestep):
        self.robot = robot
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.timestep = timestep
        config = Path(__file__).resolve().parents[2] / "config" / "locations.json"
        graph_config = (
            Path(__file__).resolve().parents[2] / "config" / "navigation_graph.json"
        )
        self.locations = json.loads(config.read_text(encoding="utf-8"))
        self.graph = json.loads(graph_config.read_text(encoding="utf-8"))
        self.routes = RouteGraph(self.graph)
        self.gps = robot.getDevice("gps")
        self.compass = robot.getDevice("compass")
        self.gps.enable(timestep)
        self.compass.enable(timestep)
        self.proximity = []
        for index in range(8):
            sensor = robot.getDevice(f"ps{index}")
            sensor.enable(timestep)
            self.proximity.append(sensor)
        # Edges found blocked during this run. Kept between navigate calls so
        # the robot does not drive into the same obstacle twice; cleared if
        # no route is left, in case the obstacle has been moved.
        self.blocked_edges = set()
        self.evidence = None
        # Graph edge the robot was last driving along, kept after a cancel so
        # the next trip can start from a node it can reach in a straight line.
        self.edge = None
        self._reset_task()

    # ------------------------------------------------------------ public
    def stop(self):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def start(self, location: str) -> str:
        """Start a named-location task without advancing the simulation.

        Any graph node is accepted, so the search skill can move to extra
        viewpoints; plans can only name locations (PlanValidator checks).
        """
        self._reset_task()
        self.stop()
        self.started_at = self.robot.getTime()
        if location not in self.routes.nodes:
            return self._finish(FAILED, f"unknown location {location!r}")
        self.active_location = location
        self.phase = "WAIT_SENSORS"
        return RUNNING

    def update(self) -> str:
        """Perform one feedback-control update and return a status."""
        if self.active_location is None:
            return FAILED
        now = self.robot.getTime()
        pose = self.pose()

        if self.phase == "WAIT_SENSORS":
            if pose is None:
                if now - self.started_at > GPS_WAIT_S:
                    return self._finish(FAILED, "no valid GPS/Compass reading")
                return RUNNING
            if not self._plan_from(self._choose_start(pose), pose):
                return self._finish(BLOCKED, "no route to destination")
            self.phase = "DRIVE"

        if pose is None:
            self.stop()
            return self._finish(FAILED, "lost GPS/Compass reading")
        x, y, yaw = pose

        if now - self.started_at > self.time_budget:
            return self._finish(TIMEOUT, f"route not completed within "
                                         f"{self.time_budget:.0f} s", pose)

        if self.phase == "BACKOFF":
            return self._update_backoff(now, pose)

        return self._update_drive(now, x, y, yaw, pose)

    def drive(self, left, right):
        """Direct wheel command for other skills (search scanning), clamped."""
        self._set_wheels(left, right)

    def front_is_blocked(self, threshold=OBSTACLE_THRESHOLD) -> bool:
        return max(self.proximity[i].getValue() for i in FRONT_SENSORS) > threshold

    def cancel(self):
        self.stop()
        self.active_location = None
        self.waypoints = []

    def forget_blocked_edges(self):
        self.blocked_edges.clear()

    # ------------------------------------------------------------ driving
    def _update_drive(self, now, x, y, yaw, pose):
        name, (wx, wy) = self.waypoints[self.waypoint_index]
        dist = math.hypot(wx - x, wy - y)
        last = self.waypoint_index == len(self.waypoints) - 1
        tolerance = ARRIVAL_TOLERANCE_M if last else WAYPOINT_TOLERANCE_M

        if self.last_node is not None and self.last_node != name:
            self.edge = (self.last_node, name)      # where the robot is now
        if dist < tolerance:
            self.last_node = name
            self.edge = (name, name)
            if last:
                self.stop()
                return self._finish(SUCCEEDED, f"arrived {dist:.3f} m from "
                                               f"{self.active_location}", pose)
            self.waypoint_index += 1
            return RUNNING

        error = wrap_angle(math.atan2(wy - y, wx - x) - yaw)
        self._debug(now, name, x, y, yaw, dist, error)

        if abs(error) > TURN_IN_PLACE_RAD:
            # Rotating on the spot: no forward motion, so no front check and
            # the progress window restarts when driving resumes.
            speed = max(MIN_TURN_SPEED, min(TURN_SPEED, HEADING_GAIN * abs(error)))
            direction = 1.0 if error > 0 else -1.0
            self._set_wheels(-direction * speed, direction * speed)
            self.obstacle_count = 0
            self._restart_progress(now, x, y)
            return RUNNING

        if self.front_is_blocked():
            self.obstacle_count += 1
            if self.obstacle_count >= OBSTACLE_CONFIRM_STEPS:
                if last and dist < NEAR_ARRIVAL_M:
                    return self._finish(SUCCEEDED, f"stopped {dist:.3f} m from "
                                        f"{self.active_location} by an obstacle "
                                        f"at the destination {self._front_values()}",
                                        pose)
                return self._on_blocked(now, pose, "front proximity sensors "
                                        f"{self._front_values()} > {OBSTACLE_THRESHOLD}")
            self.stop()
            return RUNNING
        self.obstacle_count = 0

        if self.progress_origin is None:
            self._restart_progress(now, x, y)
        elif now - self.progress_since > PROGRESS_WINDOW_S:
            moved = math.hypot(x - self.progress_origin[0], y - self.progress_origin[1])
            if moved < PROGRESS_MIN_M:
                return self._on_blocked(now, pose, f"no progress: moved {moved:.3f} m "
                                        f"in {PROGRESS_WINDOW_S:.0f} s")
            self._restart_progress(now, x, y)

        self._check_heading(x, y, yaw, error)

        scale = 1.0
        if last:
            scale = max(MIN_SLOWDOWN_SCALE, min(1.0, dist / SLOWDOWN_DISTANCE_M))
        forward = CRUISE_SPEED * scale
        turn = HEADING_GAIN * error
        self._set_wheels(forward - turn, forward + turn)
        return RUNNING

    def _on_blocked(self, now, pose, reason):
        self.stop()
        self.reroutes += 1
        self.block_reasons.append(reason)
        current = self.waypoints[self.waypoint_index][0]
        previous = self.last_node
        if previous is not None and previous != current:
            self.blocked_edges.add(frozenset((previous, current)))
            self.blocked_edge_names.append(f"{previous}-{current}")
        if self.reroutes > MAX_REROUTES:
            return self._finish(BLOCKED, f"blocked {self.reroutes} times; last: "
                                         f"{reason}", pose)
        self.phase = "BACKOFF"
        self.backoff_until = now + BACKOFF_SECONDS
        return RUNNING

    def _update_backoff(self, now, pose):
        rear_clear = max(self.proximity[i].getValue()
                         for i in REAR_SENSORS) <= OBSTACLE_THRESHOLD
        if now < self.backoff_until and rear_clear:
            self._set_wheels(-BACKOFF_SPEED, -BACKOFF_SPEED)
            return RUNNING
        self.stop()
        restart = self.last_node or self.routes.nearest_node(pose[0], pose[1])
        if not self._plan_from(restart, pose):
            return self._finish(BLOCKED, "no route avoids the blocked edge(s) "
                                         f"{self.blocked_edge_names}", pose)
        self.phase = "DRIVE"
        self._restart_progress(now, pose[0], pose[1])
        return RUNNING

    # ------------------------------------------------------------ planning
    def _choose_start(self, pose):
        """Node to start the route from.

        If the robot was stopped part-way along a graph edge, both ends of
        that edge are reachable in a straight line; take the one that gives
        the shorter total trip (e.g. keep going forward rather than drive
        back to a node behind it). Otherwise use the nearest node.
        """
        x, y = pose[0], pose[1]
        if self.edge and self.edge[0] != self.edge[1]                 and _distance_to_segment(x, y, self.routes.nodes[self.edge[0]],
                                         self.routes.nodes[self.edge[1]]) < EDGE_TOLERANCE_M:
            best, best_cost = None, math.inf
            for node in self.edge:
                path = self.routes.shortest_path(node, self.active_location,
                                                 self.blocked_edges)
                if path is None:
                    continue
                nx, ny = self.routes.nodes[node]
                cost = math.hypot(nx - x, ny - y) + self.routes.path_length(path)
                if cost < best_cost:
                    best, best_cost = node, cost
            if best is not None:
                return best
        return self.routes.nearest_node(x, y)

    def _plan_from(self, start_node, pose):
        path = self.routes.shortest_path(start_node, self.active_location,
                                         self.blocked_edges)
        if path is None and self.blocked_edges:
            # Every route is marked blocked. The obstacle may have moved, so
            # forget old blockages once rather than give up immediately.
            if self.forgot_once:
                return False
            self.forgot_once = True
            self.blocked_edges = {e for e in self.blocked_edges
                                  if e in self._edges_blocked_this_task()}
            path = self.routes.shortest_path(start_node, self.active_location,
                                             self.blocked_edges)
        if path is None:
            return False
        self.route = path
        self.waypoints = [(n, self.routes.nodes[n]) for n in path]
        self.waypoint_index = 0
        length = self.routes.path_length(path) + math.hypot(
            self.routes.nodes[start_node][0] - pose[0],
            self.routes.nodes[start_node][1] - pose[1])
        elapsed = self.robot.getTime() - self.started_at
        self.time_budget = (elapsed + length / BUDGET_SPEED_MPS + BUDGET_SLACK_S
                            + BUDGET_PER_WAYPOINT_S * len(path))
        if DEBUG:
            print(f"[NAV] route to {self.active_location}: {' -> '.join(path)} "
                  f"({length:.2f} m, budget {self.time_budget:.0f} s)")
        return True

    def _edges_blocked_this_task(self):
        return {frozenset(name.split("-", 1)) for name in self.blocked_edge_names}

    # ------------------------------------------------------------ helpers
    def pose(self):
        """(x, y, yaw) from GPS and Compass, or None before valid readings."""
        position = self.gps.getValues()
        north = self.compass.getValues()
        values = (position[0], position[1], north[0], north[1])
        if any(v is None or math.isnan(v) for v in values):
            return None
        return position[0], position[1], heading_from_compass(north)

    def _set_wheels(self, left, right):
        left = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, left))
        right = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, right))
        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)

    def _front_values(self):
        return [round(self.proximity[i].getValue(), 1) for i in FRONT_SENSORS]

    def _check_heading(self, x, y, yaw, error):
        """Warn once if the compass heading disagrees with GPS motion.

        Only checked while driving almost straight, so the direction of
        travel should match the compass heading.
        """
        if self.heading_warned or abs(error) > 0.15:
            self.heading_ref = None
            return
        if self.heading_ref is None:
            self.heading_ref = (x, y)
            return
        dx, dy = x - self.heading_ref[0], y - self.heading_ref[1]
        if math.hypot(dx, dy) < 0.15:
            return
        diff = abs(wrap_angle(math.atan2(dy, dx) - yaw))
        if diff > math.radians(45):
            self.heading_warned = True
            print(f"[NAV] WARNING: compass heading {math.degrees(yaw):.0f} deg but "
                  f"GPS motion {math.degrees(math.atan2(dy, dx)):.0f} deg - "
                  "check heading_from_compass()")
        self.heading_ref = (x, y)

    def _restart_progress(self, now, x, y):
        self.progress_origin = (x, y)
        self.progress_since = now

    def _reset_task(self):
        self.active_location = None
        self.waypoints = []
        self.waypoint_index = 0
        self.route = []
        self.phase = "IDLE"
        self.started_at = 0.0
        self.time_budget = math.inf
        self.last_node = None
        self.obstacle_count = 0
        self.progress_origin = None
        self.progress_since = 0.0
        self.backoff_until = 0.0
        self.reroutes = 0
        self.block_reasons = []
        self.blocked_edge_names = []
        self.forgot_once = False
        self.heading_ref = None
        self.heading_warned = False
        self._last_debug = -math.inf

    def _finish(self, status, reason, pose=None):
        self.stop()
        self.evidence = {
            "status": status,
            "destination": self.active_location,
            "reason": reason,
            "route": list(self.route),
            "reroutes": self.reroutes,
            "blocked_edges": list(self.blocked_edge_names),
            "block_reasons": list(self.block_reasons),
            "elapsed_s": round(self.robot.getTime() - self.started_at, 2),
        }
        if pose is not None:
            self.evidence["position"] = [round(pose[0], 3), round(pose[1], 3)]
            self.evidence["heading_deg"] = round(math.degrees(pose[2]), 1)
        if DEBUG:
            print(f"[NAV] {status}: {reason}")
        self.active_location = None
        self.waypoints = []
        return status

    def _debug(self, now, name, x, y, yaw, dist, error):
        if not DEBUG or now - self._last_debug < DEBUG_PERIOD_S:
            return
        self._last_debug = now
        print(f"[NAV] t={now:6.1f} wp={name} pos=({x:.2f},{y:.2f}) "
              f"yaw={math.degrees(yaw):6.1f} dist={dist:.2f} "
              f"err={math.degrees(error):6.1f} front={self._front_values()}")
