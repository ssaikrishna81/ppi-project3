"""Shared symbolic-plan definitions for the group project.

Students may extend the schema, but should keep the representation explicit,
machine-readable and independently validatable.
"""
ALLOWED_ACTIONS = {
    "navigate",
    "search",
    "observe",
    "ask",
    "report",
    "return_home",
    "stop",
}

PLAN_VERSION = "1.0"
MAX_TOTAL_STEPS = 20
MAX_NESTING_DEPTH = 2
MAX_TEXT_LENGTH = 240

# Upper bound on the whole plan once serialised to JSON. A plan of 20 short
# steps is well under 2000 characters, so anything near this limit is padded
# or runaway generation rather than a real task.
MAX_PLAN_CHARS = 4000

# navigate + return_home steps counted over the whole plan, both branches
# included. Six named locations plus one return home is 7, so 8 allows a
# full tour and rejects loops such as visiting the same rooms again and again.
MAX_NAVIGATION_STEPS = 8

ACTION_REQUIRED_FIELDS = {
    "navigate": {"location"},
    "search": {"object"},
    "observe": set(),
    "ask": {"question"},
    "report": {"message"},
    "return_home": set(),
    "stop": set(),
}

# Every key a step may contain. Anything else (speed, code, retries, url...)
# is rejected, so the model cannot smuggle extra instructions into a step.
ACTION_ALLOWED_FIELDS = {
    "navigate": {"action", "location"},
    "search": {"action", "object", "colour", "locations"},
    "observe": {"action"},
    "ask": {"action", "question"},
    "report": {"action", "message"},
    "return_home": {"action"},
    "stop": {"action"},
    "conditional": {"action", "condition", "then", "else"},
}

PLAN_ALLOWED_FIELDS = {"version", "goal", "steps"}

# Contract extension (documented in the README and report): a search step may
# carry "locations", a list of rooms to search in that order. The search
# skill navigates to each, searches, and stops at the first FOUND. Each room
# counts as one navigation towards MAX_NAVIGATION_STEPS. This replaces a
# chain of conditionals the 0.5B model could not write reliably.
MAX_SEARCH_LOCATIONS = len({"laboratory", "office_a", "office_b",
                            "meeting_room", "main_corridor", "reception"})

ALLOWED_CONDITION_SOURCES = {"last_search_result", "route_status"}
ALLOWED_CONDITION_OPERATORS = {"found", "equals"}

# Search outcomes written to WorldState by the search skill.
SEARCH_OUTCOMES = {"FOUND", "NOT_FOUND", "UNCERTAIN"}
# Outcome of the most recent navigate/return_home step.
ROUTE_STATUSES = {"CLEAR", "BLOCKED"}

# Condition grammar: (source, operator) -> rule for the value. This is the
# complete list; any other combination is rejected. Conditions are evaluated
# by table lookup in TaskExecutive, never with eval().
CONDITION_GRAMMAR = {
    ("last_search_result", "found"): ("bool", None),
    ("last_search_result", "equals"): ("enum", SEARCH_OUTCOMES),
    ("route_status", "equals"): ("enum", ROUTE_STATUSES),
}

# Which earlier action must have run before a condition on this source makes
# sense. "Is the ball found?" before any search is a contradictory plan.
CONDITION_SOURCE_REQUIRES = {
    "last_search_result": {"search"},
    "route_status": {"navigate", "return_home"},
}

VALID_LOCATIONS = {
    "reception",
    "office_a",
    "office_b",
    "laboratory",
    "meeting_room",
    "main_corridor",
}
HOME_LOCATION = "reception"

SUPPORTED_OBJECTS = {"ball", "box", "cylinder"}
SUPPORTED_COLOURS = {"red", "green", "blue", "yellow"}

# Substrings that never belong in a goal, question or message. The validator
# already rejects unknown actions and fields, so this only guards the free
# text that will be spoken or logged.
FORBIDDEN_TEXT_PATTERNS = (
    "import ",
    "eval(",
    "exec(",
    "__",
    "os.system",
    "subprocess",
    "setvelocity",
    "setposition",
    "getfromdef",
    "supervisor",
    "rm -",
    "del /",
    "powershell",
    "cmd.exe",
    "http://",
    "https://",
    "open(",
    "<script",
)
