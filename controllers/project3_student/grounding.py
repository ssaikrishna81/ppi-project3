"""Deterministic grounding of a request to known entities, and a check that a
generated plan is consistent with it.

ground(text) finds, with word-boundary patterns:
  locations   named places in the order they were said (synonyms mapped to
              the configured names, e.g. "lab" -> laboratory)
  object/colour  the search target, if any
  conditional  "if ... not / only if / absent / otherwise" style wording
  return_home  "come back / return / go home" style wording
  unsupported  abilities the robot does not have, or unknown places

The result is given to the LLM as a hint (the model still writes the plan)
and used by check_consistency() to catch plans that ignore the request, e.g.
a missing room or a search for something the user never mentioned. Such a
plan is sent back to the model for correction like a validator error.
"""
import re

LOCATION_PATTERNS = [
    ("laboratory", r"\blab(oratory)?\b"),
    ("meeting_room", r"\bmeeting(\s*room)?\b|\bconference room\b"),
    ("office_a", r"\boffice[\s_-]*a\b|\bfirst office\b"),
    ("office_b", r"\boffice[\s_-]*b\b|\bsecond office\b"),
    ("main_corridor", r"\b(main\s+)?corridor\b|\bhall(way)?\b"),
    ("reception", r"\breception\b|\bfront desk\b|\blobby\b|\bentrance\b"),
]
OBJECT_PATTERNS = [("ball", r"\bballs?\b|\bsphere\b"),
                   ("box", r"\bbox(es)?\b|\bcubes?\b"),
                   ("cylinder", r"\bcylinders?\b|\bcan\b")]
COLOURS = ("red", "green", "blue", "yellow")
CONDITIONAL = r"\bif\b|\bonly\b.*\b(absent|not|missing)\b|\botherwise\b|\bunless\b|\belse\b"
# "come back", "come straight back", "go right back", "head back home"...
RETURN = (r"\b(come|go|head|get|drive|turn)\b(\s+\w+){0,2}\s+back\b|\breturn\b"
          r"|\bback to (the )?(start|reception|front desk|home)\b|\bhome\b")
BACK_TO_ROOM = (r"\bback to (the )?(lab(oratory)?|office|meeting|conference|"
                r"(main )?corridor|hall(way)?)\b")
SEARCH_ROOMS = ("laboratory", "office_a", "office_b", "meeting_room")
OBSERVE =r"\blook around\b|\bobserve\b|\bwhat (can|do) you see\b|\bscan\b"
SEARCH_WORDS =r"\bfind\b|\bsearch\b|\blook (for|in|around)\b|\bcheck\b|\blocate\b|\bwhere is\b|\bis the\b"
UNSUPPORTED = [
    ("fly", r"\bfly(ing)?\b"), ("jump", r"\bjump(ing)?\b"), ("climb", r"\bclimb"),
    ("go through walls", r"\bthrough (the )?walls?\b"),
    ("pick up or carry objects", r"\bpick (it |the \w+ )?up\b|\bcarry\b|\bgrab\b|\bbring\b|\bfetch\b"),
    ("open doors", r"\bopen\b"), ("swim", r"\bswim"), ("teleport", r"\bteleport"),
    ("go upstairs or outside", r"\bupstairs\b|\boutside\b|\bgarden\b|\bcar park\b|\bkitchen\b|\broof\b"),
]


def ground(text):
    t = " " + text.lower() + " "
    found = []
    for name, pattern in LOCATION_PATTERNS:
        for match in re.finditer(pattern, t):
            found.append((match.start(), name))
    found.sort()
    locations = []
    for _, name in found:
        if not locations or locations[-1] != name:
            locations.append(name)
    obj = next((name for name, p in OBJECT_PATTERNS if re.search(p, t)), None)
    colour = next((c for c in COLOURS if re.search(rf"\b{c}\b", t)), None)
    return {
        "locations": locations,
        "object": obj,
        "colour": colour,
        "search": bool(obj) and bool(re.search(SEARCH_WORDS, t)) or (
            bool(obj) and not locations),
        "conditional": bool(re.search(CONDITIONAL, t)),
        # "go back to office B" means office B, not reception.
        "return_home": bool(re.search(RETURN, t)) and not re.search(BACK_TO_ROOM, t),
        "observe": bool(re.search(OBSERVE, t)),
        "unsupported": [name for name, p in UNSUPPORTED if re.search(p, t)],
    }


def pattern_of(g):
    """Which kind of request this is, to choose matching few-shot examples."""
    explicit = [loc for loc in g["locations"] if loc != "reception"]
    if g["unsupported"]:
        return "unsupported"
    if g["search"]:
        if not explicit:
            return "search_all"
        return "search_room" if len(explicit) == 1 else "search_conditional"
    if g["observe"]:
        return "observe"
    if g["return_home"] and (explicit or not g["locations"]):
        return "nav_return" if explicit else "nav_single"
    return "nav_sequence" if len(g["locations"]) > 1 else "nav_single"


def hint_text(g):
    """One short line for the LLM prompt."""
    parts = [f"places mentioned in order: {', '.join(g['locations']) or 'none'}"]
    if g["object"]:
        parts.append(f"target: {g['colour'] or 'unspecified colour'} {g['object']}")
    explicit = [loc for loc in g["locations"] if loc != "reception"]
    if g["search"] and not explicit:
        parts.append("no room given, so use one search step with locations "
                     "laboratory, office_a, office_b, meeting_room")
    if g["conditional"]:
        parts.append("contains a condition")
    if g["return_home"]:
        parts.append("asks to return to reception")
    if g["unsupported"]:
        parts.append(f"asks for something the robot cannot do: {', '.join(g['unsupported'])}")
    return "Hints: " + "; ".join(parts) + "."


def _walk(steps):
    for step in steps:
        yield step
        if step.get("action") == "conditional":
            yield from _walk(step.get("then", []))
            yield from _walk(step.get("else", []))


def check_consistency(plan, g):
    """Problems that make a (valid) plan not match the request, or []."""
    steps = list(_walk(plan.get("steps", [])))
    actions = [s.get("action") for s in steps]
    navs = []
    for s in steps:
        if s.get("action") == "navigate":
            navs.append(s.get("location"))
        elif s.get("action") == "search" and isinstance(s.get("locations"), list):
            navs.extend(s["locations"])
    searches = [s for s in steps if s.get("action") == "search"]
    problems = []

    if g["unsupported"]:
        if actions != ["ask"]:
            problems.append(f"the request asks for something the robot cannot do "
                            f"({', '.join(g['unsupported'])}); the plan must be a "
                            "single ask step that says so")
        return problems
    if "ask" in actions:
        problems.append("the request can be done; do not use ask, plan the steps")
        return problems

    explicit = [loc for loc in g["locations"] if loc != "reception"]
    for loc in explicit:
        if loc not in navs:
            problems.append(f"the user mentioned {loc} but the plan never navigates there")
    order = [loc for loc in navs if loc in explicit]
    first_seen = [loc for i, loc in enumerate(order) if loc not in order[:i]]
    if first_seen and first_seen != [l for l in explicit if l in first_seen]:
        problems.append(f"keep the user's order: {', '.join(explicit)}")
    if explicit:
        extra = [loc for loc in navs if loc not in explicit and loc != "reception"]
        if extra:
            problems.append(f"the plan visits {', '.join(sorted(set(extra)))} which the "
                            "user did not ask for")
    if not g["search"] and not g["return_home"] and "return_home" in actions:
        problems.append("the user did not ask to come back; remove return_home")
    if "conditional" in actions and not g["conditional"] and not (
            g["search"] and not explicit):
        problems.append("the request has no condition; do not use conditional steps")
    if g["search"]:
        if not searches:
            problems.append(f"the user asked to find the {g['colour'] or ''} "
                            f"{g['object']}; add search steps".replace("  ", " "))
        for s in searches:
            if s.get("object") != g["object"] or (g["colour"] and s.get("colour") != g["colour"]):
                problems.append(f"search for the {g['colour'] or ''} {g['object']}, "
                                "not something else".replace("  ", " "))
                break
        before_search = []
        for s in plan.get("steps", []):
            if s.get("action") == "search":
                if s.get("locations"):
                    before_search.append(s["locations"][0])
                break
            if s.get("action") == "navigate":
                before_search.append(s.get("location"))
        if not explicit:
            missing = [r for r in SEARCH_ROOMS if r not in navs]
            if missing:
                problems.append("no room was given, so search all rooms: "
                                + ", ".join(SEARCH_ROOMS))
        if ((not before_search or before_search[-1] == "reception")
                and "reception" not in g["locations"]):
            first = explicit[0] if explicit else "laboratory"
            problems.append(f"do not search the reception; navigate to {first} "
                            "before the first search")
    elif searches:
        problems.append("the user did not ask to search for anything")
    if g["conditional"] and "conditional" not in actions and len(explicit) > 1:
        problems.append("the request has a condition; use a conditional step")
    if g["return_home"] and "return_home" not in actions and "reception" not in navs:
        problems.append("the user asked to come back; add return_home")
    return problems
