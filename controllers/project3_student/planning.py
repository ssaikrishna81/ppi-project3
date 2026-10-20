"""Language-side planning pipeline, run in the worker thread.

plan_request(...)   goal -> grounding hints -> LLM plan -> PlanValidator ->
                    consistency check against the request -> if either finds
                    problems, up to max_retries corrections -> result dict.
replan_request(...) same, starting from LLMInterface.replan() with bounded
                    failure evidence (validator only: a replan legitimately
                    differs from the original request).

Nothing here touches Webots devices; the inputs are plain data snapshots
taken on the main thread. Every model output, first or corrected, goes
through the same PlanValidator. The result dict records every attempt (raw
text, latency, repairs, errors) for the trace log and the report.
"""
import json
from pathlib import Path

from grounding import SEARCH_ROOMS, check_consistency, ground, hint_text, pattern_of
from llm_interface import LLMError

RETRY_TEMPERATURE = 0.4


def drop_leading_noop(plan, here, repairs=None):
    """Remove a first 'navigate' to where the robot already is.

    The model often copies "The robot is at reception" into a first step.
    Driving to the current place does nothing, so removing it changes no
    behaviour; it is recorded as a repair.
    """
    steps = plan.get("steps") if isinstance(plan, dict) else None
    if (here and isinstance(steps, list) and len(steps) > 1
            and isinstance(steps[0], dict)
            and steps[0].get("action") == "navigate"
            and steps[0].get("location") == here
            # "navigate here; search" may be a wrong room the consistency
            # check must see, so only drop it before non-search steps.
            and isinstance(steps[1], dict) and steps[1].get("action") != "search"):
        plan["steps"] = steps[1:]
        if repairs is not None:
            repairs.append(f"removed first step: navigate to {here} (already there)")
    return plan


EXAMPLES_PATH = Path(__file__).resolve().parents[2] / "prompts" / "plan_examples.json"
GRAPH_PATH = Path(__file__).resolve().parents[2] / "config" / "navigation_graph.json"
_EXAMPLES = None


def load_examples():
    global _EXAMPLES
    if _EXAMPLES is None:
        _EXAMPLES = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))["examples"]
    return _EXAMPLES


def select_examples(pattern, count=2):
    """Few-shot examples for this request pattern, best match last.

    The 0.5B model tends to copy the last example's structure, so the last
    one is always of the same pattern as the request. If only one example
    has that pattern, a simple navigation example goes first.
    """
    bank = load_examples()
    same = [e for e in bank if e["pattern"] == pattern]
    # Short plans are cheap to show, so use every example of the pattern
    # (up to 3); the variety of wording matters most for plain trips.
    count = max(count, min(3, len(same)))
    if len(same) < count:
        # Never show fewer examples than asked: pad with other patterns
        # (in front, so the matching example stays last).
        filler = [e for e in bank if e not in same]
        same = filler[: count - len(same)] + same
    return [(e["request"], e["plan"]) for e in same[-count:]]


def describe_plan(plan):
    """One spoken sentence saying what the robot is about to do.

    Built from the validated plan itself, so it always matches what will be
    executed. None for plans that only ask (the ask step speaks).
    """
    from world_state import spoken_list, spoken_place
    steps = plan.get("steps", [])
    if not steps or all(s.get("action") == "ask" for s in steps):
        return None
    parts = []
    for step in steps:
        action = step.get("action")
        if action == "navigate":
            parts.append(f"go to {spoken_place(step['location'])}")
        elif action == "search":
            target = " ".join(v for v in (step.get("colour"), step.get("object")) if v)
            if step.get("locations"):
                parts.append(f"search {spoken_list(step['locations'])} for the "
                             f"{target}, stopping when I find it")
            else:
                parts.append(f"look for the {target}")
        elif action == "conditional":
            then = [s for s in step.get("then", []) if s.get("action") == "navigate"]
            if then:
                where = spoken_list([s["location"] for s in then])
                cond = step.get("condition", {})
                if cond.get("operator") == "found" and cond.get("value") is False:
                    parts.append(f"if it is not there, try {where}")
                else:
                    parts.append(f"depending on the result, go to {where}")
        elif action == "observe":
            parts.append("look around")
        elif action == "return_home":
            parts.append("come back to reception")
        elif action == "report":
            parts.append("tell you what I found")
        elif action == "stop":
            parts.append("stop")
    if not parts:
        return None
    text = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
    return f"I will {text}."


def plan_request(llm, validator, goal, world_snapshot, max_retries=1):
    grounded = ground(goal)
    hints = hint_text(grounded)
    examples = select_examples(pattern_of(grounded))

    here = world_snapshot.get("current_location")

    def first():
        plan = llm.generate_plan(goal, world_snapshot, hints, examples)
        return drop_leading_noop(plan, here, llm.last.get("repairs"))

    def correct(plan, errors):
        # The small model tends to repeat a plan it is shown, so a correction
        # is a fresh attempt with the problems listed as extra hints, at a
        # small temperature so it does not reproduce the same output.
        extra = hints + "\nA previous attempt was wrong: " + "; ".join(errors[:4]) + "."
        plan = llm.generate_plan(goal, world_snapshot, extra, examples,
                                 temperature=RETRY_TEMPERATURE)
        return drop_leading_noop(plan, here, llm.last.get("repairs"))

    def check(plan):
        valid, errors = validator.validate(plan)
        if valid:
            errors = check_consistency(plan, grounded)
            valid = not errors
        return valid, errors

    result = _with_corrections(llm, first, correct, check, max_retries, "goal")
    result["grounding"] = grounded
    result["pattern"] = pattern_of(grounded)
    if result["kind"] == "plan" and result["pattern"] == "search_all":
        note = nearest_rooms_first(result["plan"], world_snapshot.get("position"))
        if note:
            result["attempts"][-1].setdefault("repairs", []).append(note)
    return result


def nearest_rooms_first(plan, position):
    """Reorder a no-room-given search so the closest rooms come first.

    Only used when the user named no room (pattern search_all), so no order
    the user asked for is changed. Distance is the route length over the
    waypoint graph from the node nearest the robot. Returns a note, or None.
    """
    if not position:
        return None
    from navigation import RouteGraph
    graph = RouteGraph(json.loads(GRAPH_PATH.read_text(encoding="utf-8")))
    start = graph.nearest_node(position[0], position[1])
    for step in plan.get("steps", []):
        rooms = step.get("locations") if step.get("action") == "search" else None
        if not rooms or len(rooms) < 2:
            continue

        def cost(room):
            path = graph.shortest_path(start, room)
            return graph.path_length(path) if path else float("inf")
        ordered = sorted(rooms, key=cost)
        if ordered != rooms:
            step["locations"] = ordered
            return f"search order {rooms} -> {ordered} (nearest first from {start})"
    return None


def replan_situation(goal, world_snapshot, failure_evidence):
    """Summarise what is left after a failure, in the replan examples' format.

    Returns (situation_text, facts). facts holds the sets used to check the
    revised plan: rooms that cannot be reached, rooms already searched
    without success, rooms still worth searching, and the target.
    """
    g = ground(goal)
    blocked = {r.get("to") for r in world_snapshot.get("blocked_routes", [])
               if r.get("to")}
    failure = (failure_evidence or {}).get("failure") or {}
    step = failure.get("step") or {}
    if failure.get("status") in ("BLOCKED", "TIMEOUT", "FAILED") \
            and step.get("action") == "navigate" and step.get("location"):
        blocked.add(step["location"])
    results = world_snapshot.get("search_results") or {}
    searched = {room for room, outcome in results.items() if outcome == "NOT_FOUND"}
    found = [room for room, outcome in results.items() if outcome == "FOUND"]
    target = (" ".join(v for v in (g["colour"], g["object"]) if v)
              if g["object"] else None)
    explicit = [loc for loc in g["locations"] if loc != "reception"]
    candidates = explicit if (g["search"] and explicit) else (
        list(SEARCH_ROOMS) if g["search"] else [])
    remaining = [] if found else [r for r in candidates
                                  if r not in searched and r not in blocked]
    situation = "\n".join([
        f"Original request: {goal}",
        f"Already searched, not found: {', '.join(sorted(searched)) or 'none'}",
        f"Cannot reach: {', '.join(sorted(blocked)) or 'none'}",
        f"Rooms still to search: {', '.join(remaining) or 'none'}",
        f"Target: {target or 'none'}",
    ])
    facts = {"blocked": blocked, "searched": searched, "remaining": remaining,
             "target": (g["object"], g["colour"]), "found": found}
    return situation, facts


def check_replan(plan, facts):
    """Problems that make a revised plan unsuitable, or []."""
    problems = []
    steps = list(_walk_steps(plan.get("steps", [])))
    navs, searched_rooms = [], []
    for s in steps:
        if s.get("action") == "navigate":
            navs.append(s.get("location"))
        if s.get("action") == "search":
            searched_rooms.extend(s.get("locations") or [])
    for room in sorted(set(navs + searched_rooms) & facts["blocked"]):
        problems.append(f"{room} cannot be reached; do not go there again")
    for room in sorted(set(searched_rooms) & facts["searched"]):
        problems.append(f"{room} was already searched; do not search it again")
    if facts["remaining"]:
        missing = [r for r in facts["remaining"] if r not in navs + searched_rooms]
        if missing:
            problems.append("search the rooms still to search: "
                            + ", ".join(facts["remaining"]))
        obj, colour = facts["target"]
        for s in steps:
            if s.get("action") == "search" and (
                    s.get("object") != obj or (colour and s.get("colour") != colour)):
                problems.append(f"search for the {colour or ''} {obj}".replace("  ", " "))
                break
    elif any(s.get("action") in ("navigate", "search") for s in steps):
        problems.append("nothing is left to search; return_home and report")
    if not any(s.get("action") == "report" for s in steps):
        problems.append("end with a report step")
    return problems


def _walk_steps(steps):
    for step in steps:
        yield step
        if step.get("action") == "conditional":
            yield from _walk_steps(step.get("then", []))
            yield from _walk_steps(step.get("else", []))


def replan_request(llm, validator, goal, world_snapshot, failure_evidence,
                   max_retries=1):
    situation, facts = replan_situation(goal, world_snapshot, failure_evidence)
    pattern = "replan_search" if facts["remaining"] else "replan_give_up"
    bank = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8")).get("replan_examples", [])
    same = [e for e in bank if e["pattern"] == pattern]
    other = [e for e in bank if e["pattern"] != pattern]
    examples = [(e["situation"], e["plan"]) for e in (other[:1] + same[-2:])]

    def first():
        return llm.replan(goal, world_snapshot, failure_evidence,
                          situation=situation, examples=examples)

    def correct(plan, errors):
        extra = situation + "\nA previous attempt was wrong: " + "; ".join(errors[:4]) + "."
        return llm.replan(goal, world_snapshot, failure_evidence, situation=extra,
                          examples=examples, temperature=RETRY_TEMPERATURE)

    def check(plan):
        valid, errors = validator.validate(plan)
        if valid:
            errors = check_replan(plan, facts)
            valid = not errors
        return valid, errors

    result = _with_corrections(llm, first, correct, check, max_retries, "replan")
    result["situation"] = situation
    return result


def _with_corrections(llm, first_call, correct_call, check, max_retries, purpose):
    attempts = []
    plan, errors = None, []
    for attempt in range(1 + max_retries):
        try:
            if plan is None:
                # First try, or the last output had no usable JSON: there is
                # nothing to correct, so ask from scratch.
                plan = first_call()
            else:
                plan = correct_call(plan, errors)
            valid, errors = check(plan)
        except LLMError as error:
            plan, valid, errors = None, False, [str(error)]
        attempts.append({
            "attempt": attempt,
            "latency_s": llm.last.get("latency_s"),
            "raw": llm.last.get("raw"),
            "repairs": llm.last.get("repairs"),
            "valid": valid,
            "errors": errors[:5],
        })
        if valid:
            return {"kind": "plan", "purpose": purpose, "plan": plan,
                    "attempts": attempts}
    return {"kind": "rejected", "purpose": purpose, "plan": plan,
            "errors": errors, "attempts": attempts}
