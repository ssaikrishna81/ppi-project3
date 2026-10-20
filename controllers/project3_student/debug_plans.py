"""Fixed plans for testing robot skills in Webots without the LLM.

These are hand-written test fixtures, not model output, and must not be
reported as generative-planning results. They still pass through
PlanValidator before execution, exactly like a generated plan.

Select one with DEBUG_PLAN in project3_student.py; set it to None for the
demonstration and the submission.
"""

DEBUG_PLANS = {
    # Every named location once, then home. The first trip north meets
    # BLOCKING_OBSTACLE on centre -> centre_north and must reroute.
    "nav_tour": {
        "version": "1.0",
        "goal": "Debug: visit every location and return home.",
        "steps": [
            {"action": "navigate", "location": "laboratory"},
            {"action": "navigate", "location": "office_a"},
            {"action": "navigate", "location": "office_b"},
            {"action": "navigate", "location": "meeting_room"},
            {"action": "navigate", "location": "main_corridor"},
            {"action": "return_home"},
        ],
    },
    # Search every room in turn regardless of earlier results, so one run
    # tests detection in all four rooms (the ball is in exactly one).
    "search_tour": {
        "version": "1.0",
        "goal": "Debug: search every room for the red ball.",
        "steps": [
            {"action": "navigate", "location": "laboratory"},
            {"action": "search", "object": "ball", "colour": "red"},
            {"action": "navigate", "location": "office_a"},
            {"action": "search", "object": "ball", "colour": "red"},
            {"action": "navigate", "location": "office_b"},
            {"action": "search", "object": "ball", "colour": "red"},
            {"action": "navigate", "location": "meeting_room"},
            {"action": "search", "object": "ball", "colour": "red"},
            {"action": "return_home"},
        ],
    },
    # Section 7 example from the specification (conditional on the result).
    "spec_example": {
        "version": "1.0",
        "goal": "Debug: find the red ball; check the laboratory first.",
        "steps": [
            {"action": "navigate", "location": "laboratory"},
            {"action": "search", "object": "ball", "colour": "red"},
            {"action": "conditional",
             "condition": {"source": "last_search_result",
                           "operator": "found", "value": False},
             "then": [{"action": "navigate", "location": "meeting_room"},
                      {"action": "search", "object": "ball", "colour": "red"}],
             "else": []},
            {"action": "return_home"},
            {"action": "report", "message": "Report the evidence-based result."},
        ],
    },
    # Short single trip for quick checks.
    "nav_lab": {
        "version": "1.0",
        "goal": "Debug: go to the laboratory.",
        "steps": [{"action": "navigate", "location": "laboratory"}],
    },
}
