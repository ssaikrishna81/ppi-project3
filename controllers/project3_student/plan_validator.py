"""Deterministic validation gate between generative planning and execution.

This baseline checks the supplied plan contract. Groups may extend it, but an
LLM must never be the only component deciding whether its own output is safe.

Checks run in two passes:
1. structure: types, allowed keys, allowed actions, known values, sizes;
2. semantics: order-dependent rules (a condition needs the action it reads,
   no step after stop, no back-to-back trips to the same place, a cap on
   total navigation).
The validator never raises on bad input; every problem becomes an error string.
"""
import json

from plan_schema import (
    ACTION_ALLOWED_FIELDS,
    ACTION_REQUIRED_FIELDS,
    ALLOWED_ACTIONS,
    ALLOWED_CONDITION_OPERATORS,
    ALLOWED_CONDITION_SOURCES,
    CONDITION_GRAMMAR,
    CONDITION_SOURCE_REQUIRES,
    FORBIDDEN_TEXT_PATTERNS,
    HOME_LOCATION,
    MAX_SEARCH_LOCATIONS,
    MAX_NAVIGATION_STEPS,
    MAX_NESTING_DEPTH,
    MAX_PLAN_CHARS,
    MAX_TEXT_LENGTH,
    MAX_TOTAL_STEPS,
    PLAN_ALLOWED_FIELDS,
    PLAN_VERSION,
    SUPPORTED_COLOURS,
    SUPPORTED_OBJECTS,
    VALID_LOCATIONS,
)

NAVIGATION_ACTIONS = {"navigate", "return_home"}


class PlanValidator:
    def validate(self, plan: dict):
        errors = []
        if not isinstance(plan, dict):
            return False, ["Plan must be a dictionary/object."]

        try:
            size = len(json.dumps(plan))
        except (TypeError, ValueError):
            return False, ["Plan must contain only JSON values."]
        if size > MAX_PLAN_CHARS:
            return False, [
                f"Plan is {size} characters; maximum is {MAX_PLAN_CHARS}."
            ]

        extra = set(plan) - PLAN_ALLOWED_FIELDS
        if extra:
            errors.append(f"Plan has unsupported top-level keys {sorted(extra)}.")

        if plan.get("version") != PLAN_VERSION:
            errors.append(f"Plan version must be {PLAN_VERSION!r}.")

        goal = plan.get("goal")
        if not isinstance(goal, str) or not goal.strip():
            errors.append("Plan must contain a non-empty string named 'goal'.")
        else:
            self._check_text(goal, "goal", errors)

        steps = plan.get("steps")
        if not isinstance(steps, list):
            return False, errors + ["Plan must contain a list named 'steps'."]

        if not steps:
            errors.append("Plan must contain at least one step.")

        counter = [0]
        self._validate_steps(steps, errors, path="steps", depth=0, counter=counter)
        if counter[0] > MAX_TOTAL_STEPS:
            errors.append(
                f"Plan contains {counter[0]} steps; maximum is {MAX_TOTAL_STEPS}."
            )

        # Semantic rules assume a well-formed plan, so only run them when the
        # structure is clean; otherwise they would repeat the same problems.
        if not errors:
            self._check_semantics(steps, errors)
        return len(errors) == 0, errors

    # ------------------------------------------------------------------
    # Pass 1: structure
    # ------------------------------------------------------------------
    def _validate_steps(self, steps, errors, path, depth, counter):
        if depth > MAX_NESTING_DEPTH:
            errors.append(f"{path}: nesting depth exceeds {MAX_NESTING_DEPTH}.")
            return

        for i, step in enumerate(steps):
            counter[0] += 1
            here = f"{path}[{i}]"
            if not isinstance(step, dict):
                errors.append(f"{here}: step must be an object.")
                continue

            action = step.get("action")
            if not isinstance(action, str):
                errors.append(f"{here}: 'action' must be a string.")
                continue

            if action == "conditional":
                self._check_fields(step, action, errors, here)
                self._validate_condition(step.get("condition"), errors, here)
                branches = {}
                for branch in ("then", "else"):
                    if branch not in step:
                        errors.append(f"{here}: conditional requires {branch!r}.")
                        continue
                    nested = step[branch]
                    if not isinstance(nested, list):
                        errors.append(f"{here}.{branch} must be a list.")
                        continue
                    branches[branch] = nested
                    self._validate_steps(
                        nested, errors, f"{here}.{branch}", depth + 1, counter
                    )
                if len(branches) == 2 and not branches["then"] and not branches["else"]:
                    errors.append(f"{here}: conditional has two empty branches.")
                continue

            if action not in ALLOWED_ACTIONS:
                errors.append(f"{here}: unsupported action {action!r}.")
                continue

            self._check_fields(step, action, errors, here)

            for field in ACTION_REQUIRED_FIELDS[action]:
                if field not in step:
                    errors.append(f"{here}: {action!r} requires {field!r}.")

            if action == "navigate" and "location" in step:
                loc = step["location"]
                if not isinstance(loc, str) or loc not in VALID_LOCATIONS:
                    errors.append(f"{here}: invalid location {loc!r}.")

            if action == "search" and "locations" in step:
                self._validate_search_locations(step["locations"], errors, here)

            if action == "search":
                obj = step.get("object")
                if "object" in step and (
                    not isinstance(obj, str) or obj not in SUPPORTED_OBJECTS
                ):
                    errors.append(f"{here}: unsupported object {obj!r}.")
                if "colour" in step:
                    colour = step["colour"]
                    if not isinstance(colour, str) or colour not in SUPPORTED_COLOURS:
                        errors.append(f"{here}: unsupported colour {colour!r}.")

            if action in {"ask", "report"}:
                field = "question" if action == "ask" else "message"
                value = step.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{here}.{field} must be a non-empty string.")
                else:
                    self._check_text(value, f"{here}.{field}", errors)

    @staticmethod
    def _validate_search_locations(locations, errors, here):
        if not isinstance(locations, list) or not locations:
            errors.append(f"{here}.locations must be a non-empty list.")
            return
        if len(locations) > MAX_SEARCH_LOCATIONS:
            errors.append(f"{here}.locations has more than {MAX_SEARCH_LOCATIONS} rooms.")
        for loc in locations:
            if not isinstance(loc, str) or loc not in VALID_LOCATIONS:
                errors.append(f"{here}.locations: invalid location {loc!r}.")
        names = [l for l in locations if isinstance(l, str)]
        if len(set(names)) != len(names):
            errors.append(f"{here}.locations repeats a room.")

    @staticmethod
    def _check_fields(step, action, errors, here):
        extra = set(step) - ACTION_ALLOWED_FIELDS[action]
        if extra:
            errors.append(f"{here}: {action!r} has unsupported fields {sorted(extra)}.")

    @staticmethod
    def _check_text(text, where, errors):
        if len(text) > MAX_TEXT_LENGTH:
            errors.append(f"{where} exceeds {MAX_TEXT_LENGTH} characters.")
        lowered = text.lower()
        for pattern in FORBIDDEN_TEXT_PATTERNS:
            if pattern in lowered:
                errors.append(f"{where} contains prohibited content {pattern!r}.")
                break

    @staticmethod
    def _validate_condition(condition, errors, here):
        if not isinstance(condition, dict):
            errors.append(f"{here}.condition must be an object.")
            return
        extra = set(condition) - {"source", "operator", "value"}
        if extra:
            errors.append(f"{here}.condition has unsupported keys {sorted(extra)}.")

        source = condition.get("source")
        operator = condition.get("operator")
        source_ok = isinstance(source, str) and source in ALLOWED_CONDITION_SOURCES
        operator_ok = (
            isinstance(operator, str) and operator in ALLOWED_CONDITION_OPERATORS
        )
        if not source_ok:
            errors.append(f"{here}.condition has an unsupported source.")
        if not operator_ok:
            errors.append(f"{here}.condition has an unsupported operator.")
        if "value" not in condition:
            errors.append(f"{here}.condition requires a value.")
            return
        if not (source_ok and operator_ok):
            return

        rule = CONDITION_GRAMMAR.get((source, operator))
        if rule is None:
            errors.append(
                f"{here}.condition: operator {operator!r} cannot be used "
                f"with source {source!r}."
            )
            return
        kind, allowed = rule
        value = condition["value"]
        if kind == "bool" and not isinstance(value, bool):
            errors.append(f"{here}.condition value must be true or false.")
        if kind == "enum" and (not isinstance(value, str) or value not in allowed):
            errors.append(
                f"{here}.condition value must be one of {sorted(allowed)}."
            )

    # ------------------------------------------------------------------
    # Pass 2: semantics (order-dependent)
    # ------------------------------------------------------------------
    def _check_semantics(self, steps, errors):
        nav_count = self._count_navigation(steps)
        if nav_count > MAX_NAVIGATION_STEPS:
            errors.append(
                f"Plan contains {nav_count} navigation steps; "
                f"maximum is {MAX_NAVIGATION_STEPS}."
            )
        self._check_sequence(steps, "steps", set(), errors)

    def _count_navigation(self, steps):
        total = 0
        for step in steps:
            if step["action"] in NAVIGATION_ACTIONS:
                total += 1
            elif step["action"] == "search":
                total += len(step.get("locations", []))
            elif step["action"] == "conditional":
                total += self._count_navigation(step["then"])
                total += self._count_navigation(step["else"])
        return total

    def _check_sequence(self, steps, path, done, errors):
        """Walk one step list in execution order.

        `done` holds the actions guaranteed to have run before this list. It
        is returned updated so the caller can continue after a conditional.
        Only actions that run on both branches count as guaranteed.
        """
        done = set(done)
        last_destination = None
        for i, step in enumerate(steps):
            here = f"{path}[{i}]"
            action = step["action"]

            if action == "stop" and i != len(steps) - 1:
                errors.append(f"{here}: no step may follow 'stop'.")

            if action == "conditional":
                source = step["condition"]["source"]
                if not done & CONDITION_SOURCE_REQUIRES[source]:
                    needed = " or ".join(sorted(CONDITION_SOURCE_REQUIRES[source]))
                    errors.append(
                        f"{here}: condition reads {source!r} before any "
                        f"{needed} step has run."
                    )
                after_then = self._check_sequence(
                    step["then"], f"{here}.then", done, errors
                )
                after_else = self._check_sequence(
                    step["else"], f"{here}.else", done, errors
                )
                done |= after_then & after_else
                last_destination = None
                continue

            if action in NAVIGATION_ACTIONS:
                destination = (
                    HOME_LOCATION if action == "return_home" else step["location"]
                )
                if destination == last_destination:
                    errors.append(
                        f"{here}: repeats navigation to {destination!r} "
                        "with nothing in between."
                    )
                last_destination = destination
            else:
                last_destination = None
            done.add(action)
        return done
