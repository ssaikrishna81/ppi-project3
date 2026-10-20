"""Isolate all generative-model-specific code behind one interface.

Backend: the Lab 5 Qwen2.5-0.5B-Instruct GGUF model, run locally with
llama-cpp-python (settings in config/llm_config.json). A "mock" backend
returns canned JSON for software tests only; it is labelled as such and is
not evidence of model performance.

generate_plan(goal, world_state) -> plan dict
    Prompt = prompts/planning_prompt_template.txt (fixed system text, so
    llama.cpp can reuse its cached prefix between calls) + a short user
    message with the robot's location and the request.
correct_plan(goal, world_state, previous, errors) -> plan dict
    Asks the model to fix a plan that PlanValidator rejected.
replan(goal, world_state, evidence) -> plan dict
    Asks for the remaining steps after a failure, with bounded evidence.

Every result is untrusted: the caller must pass it through PlanValidator.
Parsing failures raise LLMError. The raw text, parsed plan, repairs and
latency of the last call are kept in self.last for logging and the report.
"""
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict


class LLMError(RuntimeError):
    """The model could not be loaded, failed, or returned no usable JSON."""


# Ordinary naming slips the small model makes. Applied to the parsed plan
# before validation; the validator still checks every value afterwards.
LOCATION_ALIASES = {
    "lab": "laboratory", "the lab": "laboratory", "laboratory_entry": "laboratory",
    "meeting room": "meeting_room", "meetingroom": "meeting_room",
    "meeting": "meeting_room",
    "office a": "office_a", "officea": "office_a", "office_1": "office_a",
    "office b": "office_b", "officeb": "office_b", "office_2": "office_b",
    "corridor": "main_corridor", "main corridor": "main_corridor",
    "hallway": "main_corridor",
    "home": "reception", "front desk": "reception", "start": "reception",
}
COLOUR_ALIASES = {"colour_red": "red", "color_red": "red"}
# Verb slips for known actions (seen: "drive"). Only applied when the step
# has the arguments of the real action; anything else stays unknown and the
# validator rejects it.
ACTION_ALIASES = {
    "navigate": {"drive", "go", "goto", "go_to", "move", "move_to", "travel",
                 "drive_to", "visit", "walk", "head"},
    "search": {"find", "look_for", "search_for", "locate", "look"},
    "return_home": {"go_home", "return", "return_to_reception", "home",
                    "come_back", "go_back"},
}
# Argument-name slips for navigate.
LOCATION_KEYS = ("destination", "room", "place", "to", "target")


class LLMInterface:
    def __init__(self, config_path=None, client=None):
        project_root = Path(__file__).resolve().parents[2]
        self.project_root = project_root
        self.config_path = Path(config_path) if config_path else (
            project_root / "config" / "llm_config.json"
        )
        self.prompt_path = project_root / "prompts" / "planning_prompt_template.txt"
        self.config = self._load_config()
        self.system_prompt = self.prompt_path.read_text(encoding="utf-8")
        # client(system, user) -> str. Injected in tests; otherwise created
        # lazily on first use so the controller starts without waiting.
        self.client = client
        self._model = None
        self.last = {}

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            print(
                "[LLM] No config/llm_config.json found. Copy and edit "
                "config/llm_config.example.json before enabling planning."
            )
            return {}
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------ public
    def generate_plan(self, user_goal: str, world_state: dict,
                      hints: str = "", examples=None,
                      temperature=None) -> Dict[str, Any]:
        """Generate a structured symbolic plan, not robot motor commands.

        examples: [(request, plan_dict)] chosen for this request; shown
        before it, best match last.
        """
        shots = ""
        if examples:
            shots = "Examples:\n\n" + "\n\n".join(
                f"Request: {req}\n{json.dumps(plan)}" for req, plan in examples
            ) + "\n\nNow the real request.\n"
        user = (shots
                + f"The robot is at {world_state.get('current_location', 'reception')}.\n"
                + (f"{hints}\n" if hints else "")
                + f"Request: {user_goal}")
        return self._ask(user, kind="plan", temperature=temperature)

    def correct_plan(self, user_goal, world_state, previous, errors, hints=""):
        """Ask for a corrected plan after PlanValidator rejected one."""
        user = (f"The robot is at {world_state.get('current_location', 'reception')}.\n"
                + (f"{hints}\n" if hints else "")
                + f"Request: {user_goal}\n"
                f"Your previous plan was rejected:\n{_short_json(previous)}\n"
                f"Problems: {'; '.join(errors[:5])}\n"
                "Return a corrected plan that follows the format and rules exactly.")
        return self._ask(user, kind="correction")

    def replan(self, original_goal: str, world_state: dict,
               execution_trace, situation=None, examples=None,
               temperature=None) -> Dict[str, Any]:
        """Generate a revised remaining plan after execution failure/change.

        situation: short fixed-format summary built by planning.py from the
        failure evidence (what was searched, what cannot be reached...).
        examples: [(situation, plan)] few-shot pairs, best match last.
        Without them a plain instruction is used (kept for tests).
        """
        if situation is not None:
            shots = ""
            if examples:
                shots = "Examples of revised plans:\n\n" + "\n\n".join(
                    f"Situation:\n{sit}\n{json.dumps(plan)}" for sit, plan in examples
                ) + "\n\nNow the real situation.\n"
            user = (shots + f"The robot is at "
                    f"{world_state.get('current_location', 'reception')}.\n"
                    f"Situation:\n{situation}\n"
                    "Return a revised plan for the remaining work only.")
            return self._ask(user, kind="replan", temperature=temperature)
        evidence = execution_trace if isinstance(execution_trace, dict) else {}
        failure = evidence.get("failure") or {}
        done = [f"{e['step'].get('action')} {e['step'].get('location', '')}".strip()
                + f" -> {e.get('status')}"
                for e in evidence.get("recent_trace", [])
                if isinstance(e.get("step"), dict)]
        results = world_state.get("search_results") or {}
        blocked = sorted({r["to"] for r in world_state.get("blocked_routes", [])})
        user = (f"The robot is at {world_state.get('current_location', 'reception')}.\n"
                f"Original request: {original_goal}\n"
                f"Already done: {', '.join(done) or 'nothing'}.\n"
                f"Problem: {failure.get('status', 'FAILED')} during "
                f"{_short_json(failure.get('step'))}: {failure.get('reason', '')}\n"
                f"Search results so far: {json.dumps(results)}.\n"
                f"Blocked destinations, do not navigate there: {', '.join(blocked) or 'none'}.\n"
                "Return a plan for only the remaining work, with a different "
                "approach where the old one failed. If nothing sensible is "
                "left, return return_home and a report.")
        return self._ask(user, kind="replan")

    # ------------------------------------------------------------ internals
    def _ask(self, user, kind, temperature=None):
        started = time.perf_counter()
        raw = self._complete(self.system_prompt, user, temperature)
        latency = time.perf_counter() - started
        self.last = {"kind": kind, "user_prompt": user, "raw": raw,
                     "latency_s": round(latency, 2), "repairs": []}
        plan = extract_json_object(raw)
        plan = normalise_plan(plan, self.last["repairs"])
        self.last["plan"] = plan
        return plan

    def _complete(self, system, user, temperature=None):
        if self.client is not None:
            return self.client(system, user)
        backend = self.config.get("backend")
        if backend == "mock":
            return _mock_completion(user)
        if backend != "local_gguf":
            raise LLMError("config/llm_config.json: backend must be local_gguf or mock")
        model = self._load_model()
        request = {
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": float(self.config.get("temperature", 0.0)
                                 if temperature is None else temperature),
            "max_tokens": int(self.config.get("max_tokens", 700)),
        }
        if self.config.get("use_json_mode", True):
            request["response_format"] = {"type": "json_object"}
        try:
            response = model.create_chat_completion(**request)
            return response["choices"][0]["message"]["content"]
        except Exception as error:  # llama.cpp raises plain exceptions
            raise LLMError(f"local model inference failed: {error}") from error

    def _load_model(self):
        if self._model is not None:
            return self._model
        path = Path(self.config.get("model_path", ""))
        if not path.is_absolute():
            path = self.project_root / path
        if not path.is_file():
            raise LLMError(f"model file not found: {path}. Copy the Lab 5 GGUF "
                           "file there (see models/README.txt).")
        try:
            from llama_cpp import Llama
        except ImportError as error:
            raise LLMError("llama-cpp-python is not installed for the Python "
                           "that Webots uses (see README).") from error
        threads = int(self.config.get("n_threads", 0))
        if threads <= 0:
            threads = max(1, (os.cpu_count() or 2) - 1)
        try:
            self._model = Llama(
                model_path=str(path),
                n_ctx=int(self.config.get("n_ctx", 4096)),
                n_threads=threads,
                n_gpu_layers=int(self.config.get("n_gpu_layers", 0)),
                seed=int(self.config.get("seed", 42)),
                chat_format=self.config.get("chat_format") or None,
                verbose=False,
            )
        except Exception as error:
            raise LLMError(f"could not load the model: {error}") from error
        return self._model

    def warm_up(self):
        """Load the model and process the fixed prompt once (call off-thread)."""
        if self.client is None and self.config.get("backend") == "local_gguf":
            self._complete(self.system_prompt, "Request: go to the lab")


def extract_json_object(raw):
    """First complete JSON object in the text (code fences and prose tolerated)."""
    if not isinstance(raw, str) or not raw.strip():
        raise LLMError("the model returned no text")
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        raise LLMError("no JSON object in the model output")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as error:
        raise LLMError(f"invalid JSON from the model: {error.msg}") from error
    if not isinstance(value, dict):
        raise LLMError("the model output is not a JSON object")
    return value


def normalise_plan(plan, repairs):
    """Fix ordinary naming slips in place and record each repair.

    Only spelling of known values is changed (e.g. "lab" -> "laboratory",
    version 1.0 -> "1.0"); structure and actions are left for the
    validator to accept or reject.
    """
    if plan.get("version") in (1, 1.0, "1"):
        plan["version"] = "1.0"
        repairs.append("version number -> \"1.0\"")
    if "version" not in plan and "steps" in plan:
        plan["version"] = "1.0"
        repairs.append("missing version -> \"1.0\"")
    # A list of step objects under a made-up top-level key (seen: "report":
    # [{"action": "return_home"}]) is a split steps list: move it back.
    if isinstance(plan.get("steps"), list):
        for key in [k for k in plan if k not in ("version", "goal", "steps")]:
            value = plan[key]
            if (isinstance(value, list) and value
                    and all(isinstance(s, dict) and "action" in s for s in value)):
                plan["steps"].extend(plan.pop(key))
                repairs.append(f"steps found under top-level {key!r} appended to steps")
    steps = plan.get("steps")
    if isinstance(steps, list):
        _normalise_steps(steps, repairs, "steps")
    return plan


def _normalise_steps(steps, repairs, path):
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        here = f"{path}[{i}]"
        for key in ("action", "location", "object", "colour"):
            value = step.get(key)
            if isinstance(value, str) and value != value.strip().lower():
                step[key] = value.strip().lower()
                repairs.append(f"{here}.{key}: lower-cased")
        if "color" in step and "colour" not in step:
            step["colour"] = step.pop("color")
            repairs.append(f"{here}: color -> colour")
        _normalise_action(step, repairs, here)
        if isinstance(step.get("locations"), list):
            fixed = []
            for name in step["locations"]:
                key = name.strip().lower() if isinstance(name, str) else name
                fixed.append(LOCATION_ALIASES.get(key, key))
            if fixed != step["locations"]:
                repairs.append(f"{here}.locations: {step['locations']} -> {fixed}")
                step["locations"] = fixed
        loc = step.get("location")
        if isinstance(loc, str) and loc in LOCATION_ALIASES:
            step["location"] = LOCATION_ALIASES[loc]
            repairs.append(f"{here}.location: {loc} -> {step['location']}")
        colour = step.get("colour")
        if isinstance(colour, str) and colour in COLOUR_ALIASES:
            step["colour"] = COLOUR_ALIASES[colour]
        if step.get("action") == "search" and step.get("object") == "balls":
            step["object"] = "ball"
            repairs.append(f"{here}.object: balls -> ball")
        if step.get("action") == "conditional":
            if "else" not in step and isinstance(step.get("then"), list):
                step["else"] = []
                repairs.append(f"{here}: missing else -> []")
            _normalise_condition(step.get("condition"), repairs, here)
            for branch in ("then", "else"):
                if isinstance(step.get(branch), list):
                    _normalise_steps(step[branch], repairs, f"{here}.{branch}")


def _normalise_action(step, repairs, here):
    action = step.get("action")
    if not isinstance(action, str):
        return
    if action in ("navigate", "search", "return_home"):
        target = action
    else:
        target = next((real for real, slips in ACTION_ALIASES.items()
                       if action in slips), None)
        if target is None:
            return
    if target == "navigate" and "location" not in step:
        for key in LOCATION_KEYS:
            if isinstance(step.get(key), str):
                step["location"] = step.pop(key)
                repairs.append(f"{here}: {key} -> location")
                break
    needs = {"navigate": "location", "search": "object", "return_home": None}[target]
    if needs and needs not in step:
        return
    if target == "return_home" and len(step) > 1:
        return
    if action != target:
        step["action"] = target
        repairs.append(f"{here}.action: {action} -> {target}")


# Ways the model writes "the last search did not find it".
NOT_FOUND_OPERATORS = {"not_found", "notfound", "not found", "missing", "absent"}


def _normalise_condition(cond, repairs, here):
    if not isinstance(cond, dict):
        return
    op = cond.get("operator")
    if isinstance(op, str) and op.strip().lower() in NOT_FOUND_OPERATORS:
        cond["operator"], cond["value"] = "found", False
        repairs.append(f"{here}.condition: operator {op!r} -> found false")
    elif isinstance(op, str) and op.strip() in ("==", "is", "equal"):
        cond["operator"] = "equals"
        repairs.append(f"{here}.condition: operator {op!r} -> equals")
    value = cond.get("value")
    if cond.get("operator") == "found" and isinstance(value, str) \
            and value.strip().lower() in ("true", "false"):
        cond["value"] = value.strip().lower() == "true"
        repairs.append(f"{here}.condition: value {value!r} -> boolean")


def _short_json(value, limit=600):
    text = json.dumps(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _mock_completion(user):
    """Canned plans for software tests only. This is not an LLM."""
    goal = user.lower()
    if "fly" in goal or "jump" in goal:
        steps = [{"action": "ask",
                  "question": "I cannot do that. What else can I do for you?"}]
    elif "lab" in goal:
        steps = [{"action": "navigate", "location": "laboratory"}]
    else:
        steps = [{"action": "return_home"}]
    return json.dumps({"version": "1.0", "goal": "mock plan", "steps": steps})
