"""Cooperative executor for plans that have already passed validation.

One call to update() does at most one unit of work: start a step, advance the
running skill once, or resolve a conditional. Nothing here blocks or sleeps.

Outcome policy
--------------
SUCCEEDED   step finished; record evidence and move to the next step.
FAILED,
TIMEOUT,
BLOCKED     stop the skill, record the reason and end the plan with that
            status so the controller can decide whether to replan.
            Exception: a BLOCKED navigation directly followed by a
            route_status conditional is handled by that conditional.
CANCELLED   stop all motion and drop the remaining steps (cancel() or a
            plan 'stop' step).
AWAITING_ANSWER
            an 'ask' step was spoken; remaining steps are dropped because
            they may depend on the answer. The controller replans with the
            original goal plus the user's answer.

Search outcomes (FOUND / NOT_FOUND / UNCERTAIN) are results, not failures:
a completed search is SUCCEEDED and its outcome lives in WorldState.
"""

IDLE = "IDLE"
RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
TIMEOUT = "TIMEOUT"
BLOCKED = "BLOCKED"
CANCELLED = "CANCELLED"
AWAITING_ANSWER = "AWAITING_ANSWER"

FAILURE_STATUSES = {FAILED, TIMEOUT, BLOCKED}
STEP_TERMINAL = {SUCCEEDED} | FAILURE_STATUSES

# Backup limit per action in simulated seconds. Each skill enforces its own,
# tighter limit; this only catches a skill that never reports an outcome.
# Longest graph route (reception -> office_b) is about 9 m; the e-puck tops
# out near 0.12 m/s, so ~80 s of driving plus turning. A trip that meets a
# blocked doorway, tries the second approach and then a backup door took
# 229 s in simulation, so the backup limit is 400 s.
STEP_TIME_LIMITS = {
    "navigate": 400.0,
    "return_home": 400.0,
    "search": 200.0,
    "observe": 30.0,
    "ask": 20.0,
    "report": 20.0,
    "stop": 5.0,
}
DEFAULT_STEP_LIMIT = 60.0

NAVIGATION_ACTIONS = {"navigate", "return_home"}
MAX_TRACE = 50
FAILURE_EVIDENCE_ENTRIES = 5


class TaskExecutive:
    """Advance at most one small unit of work per Webots controller step."""

    def __init__(self, skills, world_state, clock=None, time_step=0.032):
        self.skills = skills
        self.world_state = world_state
        # clock() returns simulated seconds (robot.getTime in Webots). Without
        # one, time is counted as update calls multiplied by time_step.
        self.clock = clock
        self.time_step = time_step
        self._ticks = 0
        self.trace = []
        self.steps = []
        self.step_index = 0
        self.active = False
        self.status = IDLE
        self._step_running = False
        self._step_started_at = 0.0

    # ------------------------------------------------------------ control
    def start(self, plan: dict):
        self.trace.clear()
        self.steps = list(plan.get("steps", []))
        self.step_index = 0
        self.active = True
        self.status = RUNNING
        self._step_running = False
        self.world_state.execution_status = RUNNING
        self.world_state.last_failure = None

    def cancel(self, reason="cancelled by user"):
        """Stop motion now and drop all remaining work."""
        if not self.active:
            return
        self.skills.cancel()
        if self._step_running:
            self._close_step(CANCELLED, reason)
        self._finish(CANCELLED, reason)

    # ------------------------------------------------------------- update
    def update(self) -> str:
        """Run one non-blocking executive update."""
        self._ticks += 1
        if not self.active:
            return IDLE
        if self.step_index >= len(self.steps):
            return self._finish(SUCCEEDED, "all steps completed")

        step = self.steps[self.step_index]
        self.world_state.current_step = step

        if step["action"] == "conditional":
            return self._expand_conditional(step)

        if not self._step_running:
            return self._start_step(step)

        if self._now() - self._step_started_at > self._limit(step):
            self.skills.cancel()
            evidence = f"no outcome after {self._limit(step):.0f} s"
            self._close_step(TIMEOUT, evidence)
            return self._fail(step, TIMEOUT, evidence)

        status = self.skills.update()
        return self._handle_step_status(step, status)

    def _start_step(self, step):
        self._step_running = True
        self._step_started_at = self._now()
        self._append_trace({
            "index": self.step_index,
            "step": step,
            "status": RUNNING,
            "started_at": round(self._step_started_at, 2),
        })
        status = self.skills.start(step)
        return self._handle_step_status(step, status)

    def _handle_step_status(self, step, status):
        if status == RUNNING:
            return RUNNING
        if status not in STEP_TERMINAL:
            evidence = f"skill returned unknown status {status!r}"
            self.skills.cancel()
            self._close_step(FAILED, evidence)
            return self._fail(step, FAILED, evidence)

        evidence = self._skill_evidence()
        action = step["action"]
        if action in NAVIGATION_ACTIONS:
            destination = step.get("location", "reception")
            if status == SUCCEEDED:
                self.world_state.record_route("CLEAR", destination)
            elif status == BLOCKED:
                self.world_state.record_route("BLOCKED", destination, evidence)

        self._close_step(status, evidence)

        if status == SUCCEEDED:
            self.step_index += 1
            if action == "stop":
                return self._finish(CANCELLED, "plan requested stop")
            if action == "ask":
                self.world_state.pending_question = step.get("question")
                return self._finish(AWAITING_ANSWER, "waiting for the user's answer")
            return RUNNING

        if status == BLOCKED and self._next_checks_route():
            self.skills.cancel()
            self.step_index += 1
            return RUNNING
        return self._fail(step, status, evidence)

    # -------------------------------------------------------- conditional
    def _expand_conditional(self, step):
        """Replace a conditional with the selected validated branch.

        The condition is looked up in a fixed table of (source, operator)
        pairs that PlanValidator already approved. Nothing is evaluated as code.
        """
        condition = step["condition"]
        source = condition["source"]
        operator = condition["operator"]
        expected = condition["value"]

        if source == "last_search_result":
            observed = self.world_state.last_search_result
        elif source == "route_status":
            observed = self.world_state.route_status
        else:
            observed = None

        if observed is None:
            evidence = f"condition source {source!r} has no value yet"
            self._append_trace({"index": self.step_index, "step": step,
                                "status": FAILED, "evidence": evidence})
            return self._fail(step, FAILED, evidence)

        if operator == "found":
            # "found" means positive evidence only; UNCERTAIN counts as not
            # found so the robot keeps looking rather than claiming success.
            result = (observed == "FOUND") == expected
        else:  # "equals"
            result = observed == expected

        branch = step["then"] if result else step["else"]
        self._append_trace({
            "index": self.step_index,
            "step": {"action": "conditional", "condition": condition},
            "status": SUCCEEDED,
            "evidence": (f"{source}={observed!r}, {operator} {expected!r} -> "
                         f"{'then' if result else 'else'} "
                         f"({len(branch)} steps)"),
        })
        self.steps[self.step_index:self.step_index + 1] = list(branch)
        return RUNNING

    def _next_checks_route(self):
        if self.step_index + 1 >= len(self.steps):
            return False
        nxt = self.steps[self.step_index + 1]
        return (nxt.get("action") == "conditional"
                and nxt["condition"]["source"] == "route_status")

    # ------------------------------------------------------------ helpers
    def _fail(self, step, status, evidence):
        self.world_state.last_failure = {
            "status": status,
            "step": step,
            "reason": evidence,
        }
        return self._finish(status, evidence)

    def _finish(self, status, reason=""):
        if status in FAILURE_STATUSES:
            self.skills.cancel()
        self.active = False
        self._step_running = False
        self.status = status
        self.world_state.execution_status = status
        self.world_state.current_step = None
        if reason:
            self.world_state.add_note(f"task {status.lower()}: {reason}")
        return status

    def _close_step(self, status, evidence):
        self._step_running = False
        for entry in reversed(self.trace):
            if entry.get("status") == RUNNING:
                entry["status"] = status
                entry["ended_at"] = round(self._now(), 2)
                if evidence is not None:
                    entry["evidence"] = evidence
                break

    def _append_trace(self, entry):
        self.trace.append(entry)
        del self.trace[:-MAX_TRACE]

    def _skill_evidence(self):
        return getattr(self.skills, "evidence", None)

    def _limit(self, step):
        limit = STEP_TIME_LIMITS.get(step["action"], DEFAULT_STEP_LIMIT)
        if step["action"] == "search" and step.get("locations"):
            # One trip plus one room search per listed room.
            limit = len(step["locations"]) * (STEP_TIME_LIMITS["navigate"]
                                              + STEP_TIME_LIMITS["search"])
        return limit

    def _now(self):
        if self.clock is not None:
            return float(self.clock())
        return self._ticks * self.time_step

    def remaining_steps(self):
        """Steps not yet started (the failed step included, if any)."""
        return list(self.steps[self.step_index:])

    def failure_evidence(self):
        """Bounded summary for the replanning prompt and the status answer."""
        return {
            "status": self.status,
            "failure": self.world_state.last_failure,
            "recent_trace": self.trace[-FAILURE_EVIDENCE_ENTRIES:],
            "remaining_steps": self.remaining_steps(),
            "location": self.world_state.current_location,
            "search_results": dict(self.world_state.search_results),
            "blocked_routes": list(self.world_state.blocked_routes),
        }
