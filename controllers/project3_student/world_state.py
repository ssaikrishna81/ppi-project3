"""World/task state used for explanation and replanning.

Everything here comes from robot sensors or controller state, never from
claims made by the language model. TaskExecutive reads it to choose
conditional branches; the planner and the status answer read as_dict().
"""
MAX_NOTES = 20


class WorldState:
    def __init__(self):
        self.current_location = "reception"
        self.current_goal = None
        self.current_step = None
        # Outcome of the most recent search: "FOUND", "NOT_FOUND",
        # "UNCERTAIN" or None before any search in this task.
        self.last_search_result = None
        self.last_search_evidence = None
        # Search outcome per location for the current task only. Cleared on a
        # new goal, so a previous trial's result is never reused.
        self.search_results = {}
        # Outcome of the most recent navigate/return_home: "CLEAR",
        # "BLOCKED" or None before any navigation in this task.
        self.route_status = None
        self.blocked_routes = []
        self.notes = []
        self.replan_count = 0
        self.execution_status = "IDLE"
        self.pending_question = None
        self.last_failure = None

    def reset_for_new_goal(self, goal):
        """Forget task-specific evidence. Location is physical, so it stays."""
        self.current_goal = goal
        self.current_step = None
        self.last_search_result = None
        self.last_search_evidence = None
        self.search_results = {}
        self.route_status = None
        self.blocked_routes = []
        self.notes = []
        self.replan_count = 0
        self.pending_question = None
        self.last_failure = None

    def record_search(self, outcome, evidence=None, found_in=None):
        """Store a search outcome.

        found_in is the room a FOUND target was estimated to be in; it can
        differ from the current location when seen through a doorway.
        """
        self.last_search_result = outcome
        self.last_search_evidence = evidence
        place = found_in or self.current_location
        self.search_results[place] = outcome
        where = (f" (seen in {found_in} from {self.current_location})"
                 if found_in and found_in != self.current_location else "")
        self.add_note(f"search at {self.current_location}: {outcome}{where}")

    def record_route(self, status, destination, evidence=None):
        self.route_status = status
        if status == "BLOCKED":
            route = {"from": self.current_location, "to": destination}
            if route not in self.blocked_routes:
                self.blocked_routes.append(route)
        self.add_note(f"route to {destination}: {status}"
                      + (f" ({evidence})" if evidence else ""))

    def add_note(self, text):
        self.notes.append(str(text))
        del self.notes[:-MAX_NOTES]

    def report_text(self):
        """What the robot can truthfully say about the task so far.

        Built only from stored search outcomes, routes and location, so a
        'report' step can never repeat an unsupported claim from the planner.
        """
        target = "the target"
        if isinstance(self.last_search_evidence, dict):
            target = "the " + self.last_search_evidence.get("target", "target")
        found = [p for p, r in self.search_results.items() if r == "FOUND"]
        absent = [p for p, r in self.search_results.items() if r == "NOT_FOUND"]
        unsure = [p for p, r in self.search_results.items() if r == "UNCERTAIN"]
        parts = []
        if found:
            parts.append(f"I found {target} in {_places(found)}.")
        if absent:
            parts.append(f"I did not see {target} in {_places(absent)}.")
        if unsure:
            parts.append(f"In {_places(unsure)} I could not confirm whether "
                         f"{target} was there.")
        if self.blocked_routes:
            blocked = _places(sorted({r['to'] for r in self.blocked_routes}))
            parts.append(f"The route to {blocked} was blocked.")
        if not parts:
            parts.append("I have no search results yet.")
        parts.append(f"I am now at {_place(self.current_location)}.")
        return " ".join(parts)

    def describe(self):
        """Plain-language status built only from stored state."""
        if self.current_goal is None:
            return "I am idle and waiting for a goal."
        parts = [f"My goal is: {self.current_goal}."]
        step = self.current_step
        moving = (self.execution_status == "RUNNING" and isinstance(step, dict)
                  and step.get("action") in ("navigate", "return_home"))
        if self.execution_status == "RUNNING" and isinstance(step, dict):
            parts.append(f"I am currently {_step_text(step)}.")
        else:
            parts.append(f"The task status is {self.execution_status.lower()}.")
        if moving:
            parts.append(f"I am on my way from {_place(self.current_location)}.")
        else:
            parts.append(f"I am at {_place(self.current_location)}.")
        if self.search_results:
            found = ", ".join(
                f"{place.replace('_', ' ')} {result.lower().replace('_', ' ')}"
                for place, result in self.search_results.items()
            )
            parts.append(f"Search results so far: {found}.")
        if self.last_failure:
            parts.append(f"Last problem: {self.last_failure.get('reason')}.")
        return " ".join(parts)

    def as_dict(self):
        return {
            "current_location": self.current_location,
            "current_goal": self.current_goal,
            "current_step": self.current_step,
            "last_search_result": self.last_search_result,
            "search_results": dict(self.search_results),
            "route_status": self.route_status,
            "blocked_routes": list(self.blocked_routes),
            "notes": list(self.notes),
            "replan_count": self.replan_count,
            "execution_status": self.execution_status,
            "pending_question": self.pending_question,
            "last_failure": self.last_failure,
        }


SPOKEN_PLACES = {
    "office_a": "office A", "office_b": "office B", "laboratory": "the laboratory",
    "meeting_room": "the meeting room", "main_corridor": "the main corridor",
    "reception": "reception",
    "central_area": "the central area between the rooms",
}


def _place(name):
    return SPOKEN_PLACES.get(name, "the " + name.replace("_", " "))


def spoken_place(name):
    """Room name as the robot says it ("office A", "the laboratory")."""
    return _place(name)


def spoken_list(names):
    """"the laboratory, office A and office B"."""
    return _places(names)


def _places(names):
    names = [_place(n) for n in names]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _step_text(step):
    action = step.get("action")
    if action == "navigate":
        return f"going to {step['location'].replace('_', ' ')}"
    if action == "return_home":
        return "returning to reception"
    if action == "search":
        target = " ".join(v for v in (step.get("colour"), step.get("object")) if v)
        return f"searching for the {target}"
    if action == "observe":
        return "looking around"
    if action == "report":
        return "reporting"
    if action == "ask":
        return "asking a question"
    return str(action)
