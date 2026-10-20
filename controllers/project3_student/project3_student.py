"""Main Webots controller scaffold for the group project.

Only the main loop may access Webots devices. Slow language-model or speech
operations should be submitted to the worker and returned as ordinary data.
"""
import math
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from controller import Robot

from llm_interface import LLMInterface
from navigation import Navigator
from nlu_dialogue import DialogueManager
from perception import ObjectPerception
from plan_validator import PlanValidator
from robot_skills import RobotSkills
from semantic_memory import SemanticMemory
from speech_io import SpeechIO
from task_executive import (
    AWAITING_ANSWER,
    CANCELLED,
    FAILURE_STATUSES,
    SUCCEEDED,
    TaskExecutive,
)
from world_state import WorldState
from debug_plans import DEBUG_PLANS
from planning import describe_plan, plan_request, replan_request
from world_state import spoken_place
from request_channel import RequestChannel
from search import room_of
from grounding import ground

TIME_STEP = 32
MAX_REPLANS = 2

# Development only: name of a plan in debug_plans.py to run at start-up
# without the LLM. Must be None for the demonstration and submission.
DEBUG_PLAN = None
DEBUG_START_DELAY_S = 1.0
# Development only: write logs/run_*.jsonl and camera frames (see
# trace_logger.py) when the marker file logs/ENABLE_TRACE exists. logs/ is
# not shared or submitted, so a fresh copy runs without tracing.
TRACE_ENABLED = (Path(__file__).resolve().parents[2] / "logs" / "ENABLE_TRACE").exists()


robot = Robot()
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

camera = robot.getDevice("camera")
speech = SpeechIO()
dialogue = DialogueManager()
llm = LLMInterface()
validator = PlanValidator()
navigator = Navigator(robot, left_motor, right_motor, TIME_STEP)
perception = ObjectPerception(camera, TIME_STEP)
memory = SemanticMemory()
world_state = WorldState()
skills = RobotSkills(navigator, perception, memory, speech, world_state)
executive = TaskExecutive(skills, world_state, clock=robot.getTime)

trace = None
if TRACE_ENABLED:
    from trace_logger import TraceLogger
    trace = TraceLogger(robot, navigator, world_state, executive,
                        meta={"debug_plan": DEBUG_PLAN})
    skills.search.frame_saver = trace.save_frame

worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="language")
planning_future = None
MAX_PLAN_RETRIES = int(llm.config.get("max_retries", 1))
requests = RequestChannel()

# Load the model and its fixed prompt in the background at start-up, so the
# first real request is not slowed by it. The robot stays responsive.
warmup_future = worker.submit(llm.warm_up)


def plan_goal_off_thread(user_goal: str, snapshot: dict) -> dict:
    """Language-only work: this function must not access Webots devices.

    Missing information (e.g. which ball) was already handled by
    DialogueManager before the goal was submitted.
    """
    return plan_request(llm, validator, user_goal, snapshot, MAX_PLAN_RETRIES)


def replan_off_thread(goal, snapshot, evidence):
    return replan_request(llm, validator, goal, snapshot, evidence,
                          MAX_PLAN_RETRIES)


def submit_goal(user_goal: str):
    """Begin planning when the controller is idle.

    DialogueManager already refuses new goals while busy; this check is a
    second guard so two plans can never run at once.
    """
    global planning_future, discard_planning
    if planning_future is not None or executive.active:
        speech.speak("I am already working on a task. Say stop to cancel it first.")
        return
    world_state.reset_for_new_goal(user_goal)
    navigator.forget_blocked_edges()
    discard_planning = False
    print("[USER GOAL]", user_goal)
    speech.speak("Okay, let me plan that.")
    snapshot = world_state.as_dict()
    pose = navigator.pose()
    if pose is not None:
        snapshot["position"] = [round(pose[0], 3), round(pose[1], 3)]
    planning_future = worker.submit(plan_goal_off_thread, user_goal, snapshot)


def spoken_failure(failure):
    """Plain sentence for a failed step (the technical reason is logged)."""
    failure = failure or {}
    step = failure.get("step") or {}
    where = step.get("location")
    status = failure.get("status")
    if step.get("action") in ("navigate", "return_home"):
        place = spoken_place(where or "reception")
        if status == "BLOCKED":
            return f"The way to {place} is blocked."
        if status == "TIMEOUT":
            return f"Getting to {place} is taking too long."
        return f"I could not get to {place}."
    if step.get("action") == "search":
        return "The search could not be finished."
    return "That step did not work."


def locate_robot():
    """Where the robot is now, from GPS: a named place if within 0.5 m of
    one, otherwise the room it is in, otherwise the central area. Used after
    a stop or failure, when it may be between places."""
    pose = navigator.pose()
    if pose is None:
        return world_state.current_location
    x, y = pose[0], pose[1]
    for name, (lx, ly) in navigator.locations.items():
        if math.hypot(lx - x, ly - y) < 0.5:
            return name
    return room_of(x, y, skills.search.rooms)


def submit_replan():
    """Ask for a revised remainder after a failure (bounded by MAX_REPLANS)."""
    global planning_future
    world_state.replan_count += 1
    evidence = executive.failure_evidence()
    print(f"[REPLAN {world_state.replan_count}/{MAX_REPLANS}]", evidence["failure"])
    planning_future = worker.submit(replan_off_thread, world_state.current_goal,
                                    world_state.as_dict(), evidence)


def consume_planning_result():
    """Transfer completed language output back to the Webots main thread."""
    global planning_future
    if planning_future is None or not planning_future.done():
        return
    if discard_planning:
        # The user said stop while the model was still thinking.
        planning_future = None
        print("[PLANNING] result discarded after stop")
        return
    try:
        result = planning_future.result()
    except Exception as error:  # LLMError is handled inside planning.py
        print("[PLANNING ERROR]", repr(error))
        speech.speak("Planning failed safely. The robot has not moved.")
        planning_future = None
        return
    planning_future = None
    if trace is not None:
        trace.event("planning_result", **{k: v for k, v in result.items()
                                          if k != "grounding"})

    if result["kind"] == "clarification":
        speech.speak(result.get("question") or "Could you clarify your request?")
    elif result["kind"] == "rejected":
        print("[PLAN REJECTED]", result["errors"])
        unsupported = (result.get("grounding") or {}).get("unsupported")
        if result.get("purpose") == "replan":
            speech.speak("I could not make a safe new plan. " + world_state.report_text())
        elif unsupported:
            speech.speak(f"I cannot {' or '.join(unsupported)}. I can drive to a "
                         "room, look around, or search for an object.")
        else:
            speech.speak("I could not turn that into a safe plan. "
                         "Please say it another way.")
    else:
        tries = len(result["attempts"])
        if result.get("purpose") == "replan":
            print("[REPLAN SITUATION]", result.get("situation", "").replace("\n", " | "))
        print(f"[VALIDATED PLAN] ({tries} attempt{'s' if tries > 1 else ''})",
              result["plan"])
        summary = describe_plan(result["plan"])
        if summary:
            speech.speak(summary)
        executive.start(result["plan"])


def handle_input(text):
    """Route one spoken or typed utterance through DialogueManager."""
    global discard_planning
    busy = planning_future is not None or executive.active
    action = dialogue.handle(text, busy=busy)
    if trace is not None:
        trace.event("dialogue", text=text, action=action)
    kind = action["type"]
    if kind == "ignore":
        print(f"[IGNORED] '{text}' ({action.get('reason', 'not understood')}) - "
              "say 'stop' or 'cancel' to stop, or repeat your request")
    elif kind == "stop":
        was_busy = busy
        executive.cancel("cancelled by user")
        skills.cancel()
        world_state.current_location = locate_robot()
        if planning_future is not None:
            discard_planning = True
        speech.speak("Stopped." if was_busy else "I am not doing anything.")
    elif kind == "status":
        speech.speak(status_answer())
    elif kind == "say":
        speech.speak(action["text"])
    elif kind == "plan":
        g = ground(action["goal"])
        only_return = (g["return_home"] and not g["object"] and not g["observe"]
                       and not g["unsupported"]
                       and all(loc == "reception" for loc in g["locations"]))
        if only_return and locate_robot() == "reception":
            # "home" while already at reception: nothing to plan or drive.
            speech.speak("I am already at reception.")
        else:
            submit_goal(action["goal"])


def status_answer():
    """Answer "what are you doing and why" from stored state only."""
    if planning_future is not None:
        return f"I am working out a plan for: {world_state.current_goal}."
    return world_state.describe()


def start_debug_plan(name):
    """Run a fixed test plan through the same validator and executive."""
    plan = DEBUG_PLANS[name]
    valid, errors = validator.validate(plan)
    if not valid:
        print("[DEBUG PLAN REJECTED]", errors)
        return
    world_state.reset_for_new_goal(plan["goal"])
    print(f"[DEBUG PLAN] {name} (fixture, not LLM output)")
    executive.start(plan)


debug_pending = DEBUG_PLAN is not None
discard_planning = False
greeted = False

while robot.step(TIME_STEP) != -1:
    if debug_pending and robot.getTime() >= DEBUG_START_DELAY_S:
        debug_pending = False
        start_debug_plan(DEBUG_PLAN)

    consume_planning_result()

    if executive.active:
        status = executive.update()
        if status == SUCCEEDED:
            print("[TASK SUCCEEDED]", [
                (e["step"]["action"], e["status"]) for e in executive.trace])
            reported = any(e["step"].get("action") == "report"
                           for e in executive.trace)
            speech.speak("Task complete." if reported else
                         f"Done. I am at {spoken_place(world_state.current_location)}.")
        elif status in FAILURE_STATUSES:
            world_state.current_location = locate_robot()
            print(f"[TASK {status}]", executive.failure_evidence()["failure"])
            if trace is not None:
                trace.event("task_failure", **executive.failure_evidence())
            if DEBUG_PLAN is None and world_state.replan_count < MAX_REPLANS:
                speech.speak(f"{spoken_failure(world_state.last_failure)} "
                             "I will make a new plan.")
                submit_replan()
            else:
                speech.speak("I could not complete the task safely. "
                             + world_state.report_text())
        elif status == CANCELLED:
            print("[TASK CANCELLED]", executive.trace)
            speech.speak("Task stopped.")
        elif status == AWAITING_ANSWER:
            # The question was already spoken by the ask skill; the next
            # utterance is read as its answer.
            dialogue.expect_answer(world_state.pending_question)
            print("[WAITING FOR ANSWER]", world_state.pending_question)

    # Speech (assessed input) and typed input (debugging). Neither blocks:
    # both only read text that other threads/processes produced.
    if not greeted and robot.getTime() > 0.5:
        greeted = True
        speech.speak("Hello, I am ready. Tell me what to do.")
    heard = speech.listen()
    if heard:
        print("[HEARD]", heard)
        handle_input(heard)
    typed = requests.poll()
    if typed:
        print("[TYPED]", typed)
        handle_input(typed)

    if trace is not None:
        trace.update()

if trace is not None:
    trace.close()
speech.close()
worker.shutdown(wait=False, cancel_futures=True)
# A model call may still be running in the worker thread; Python would wait
# for it and Webots would force-kill the controller after 1 s. Nothing is
# left to save, so exit at once.
os._exit(0)
