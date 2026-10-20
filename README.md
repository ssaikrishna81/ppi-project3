# Generative-AI Embodied Robot Assistant (Webots R2025a)

A spoken-command assistant for the E-puck in the supplied office world. The user
says a goal; a local LLM (Qwen2.5-0.5B, Lab 5 model) proposes a symbolic plan;
deterministic code validates it; the robot executes only approved skills using
its motors, GPS, Compass, proximity sensors and camera; and it reports what it
actually observed.

```text
microphone -> SpeechIO (Vosk) -> DialogueManager -> LLMInterface (Qwen) -> PlanValidator
          -> TaskExecutive -> RobotSkills -> Navigator / SearchController -> Webots devices
                                   |                         |
                               WorldState  <------------  sensor evidence  ->  spoken report (pyttsx3)
```

## 1. Software versions

| Item | Version |
|---|---|
| OS | Windows 10/11, 64-bit |
| Webots | R2025a |
| Python | 3.12.10, 64-bit (`C:\Program Files\Python312\python.exe`, the one Webots uses) |
| Python packages | see `requirements.txt` (exact versions) |
| LLM | Qwen2.5-0.5B-Instruct, Q4_K_M GGUF (Lab 5 model), via llama-cpp-python 0.3.36, CPU only |
| Speech recognition | Vosk 0.3.45 with `vosk-model-small-en-us-0.15` |
| Speech output | pyttsx3 2.99 (Windows SAPI voices) |

## 2. Installation

1. Install the packages into the Python that Webots runs (`requirements.txt`
   already points pip at the prebuilt llama-cpp-python CPU wheels used in
   Lab 5, so no compiler is needed):

   ```bash
   "C:\Program Files\Python312\python.exe" -m pip install -r requirements.txt
   ```

   In Webots, *Tools > Preferences > Python command* must point to that interpreter.

2. Models (not submitted, see section 4): place
   - the Lab 5 file at `models/qwen2.5-0.5b-instruct-q4_k_m.gguf`
   - the Lab 4 Vosk model folder at `models/vosk-model-small-en-us-0.15/`
     (download: https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip)

## 3. Running

1. Open `worlds/project_genai_embodied_assistant.wbt` in Webots R2025a. The robot
   controller is `project3_student`; the `scene setup supervisor` places the red
   ball at a random approved spawn point on every reset.
2. Press play. The console shows `[SPEECH] listening on ...` and the robot says
   *"Hello, I am ready."* The model loads in the background; the first plan can
   take a few seconds longer.
3. Speak one request at a time. Headphones avoid the robot hearing itself (the
   microphone is also muted while it speaks).

Example requests (all request types of the specification):

| Type | Say |
|---|---|
| Direct | "go to office B" |
| Sequential | "go to the lab, then return to reception" |
| Search | "find the red ball" / "is the red ball in the meeting room?" |
| Conditional | "look in the lab; if the red ball is not there, check the meeting room" |
| Constrained order | "check office B first; only search office A if the red ball is absent" |
| Ambiguous | "find the ball" -> robot asks which colour -> "the red one" |
| Unsupported | "fly through the wall to office B" -> robot refuses and offers to drive -> "yes" |
| Status | "what are you doing?" (any time, also while moving) |
| Cancel | "stop" or "cancel" (any time) |

Typed input for debugging (does not replace speech): with the simulation
running, from the project folder

```bash
python send_goal.py find the red ball
```

### Audio device

`config/speech_config.json`: `input_device` is `null` (Windows default
microphone). To pick another one, list devices with
`"C:\Program Files\Python312\python.exe" -m sounddevice` and put the index of a
device with input channels there. 16 kHz is used if the device accepts it,
otherwise its native rate. Windows *Settings > Privacy > Microphone* must allow
desktop apps. If no microphone or speech engine is available, the controller
still runs: replies are printed and goals can be typed.

## 4. Model configuration

`config/llm_config.json` (included; it holds no credentials):

| Setting | Value | Why |
|---|---|---|
| backend | `local_gguf` | Lab 5 model through llama-cpp-python |
| model_path | `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` | |
| chat_format | `chatml` | Qwen2.5 chat template |
| n_ctx | 4096 | prompt with examples fits with room to spare |
| n_threads | 0 (= CPU cores - 1) | |
| n_gpu_layers | 0 | CPU only, no GPU build needed |
| temperature | 0.0 (first attempt), 0.4 (retries) | deterministic first try; variety when retrying |
| max_tokens | 550 | the longest example plan is 156 tokens; the limit stops a rambling answer early |
| use_json_mode | false | grammar-constrained JSON was roughly twice as slow in our tests; invalid JSON is caught and retried instead |
| max_retries | 2 | at most 3 model calls per request |

Model weights and the Vosk model are not part of the submission
(`.gitignore` excludes `models/*.gguf` and `models/vosk-model-*/`).

## 5. Tests

From the project folder:

```bash
python -m unittest discover -s tests -v
```

About 180 tests, no Webots or model needed. They cover the validator (valid,
malformed, unsupported, unsafe and extension plans), the executive
(conditionals, timeouts, cancel, blocked routes), navigation and search on a
2-D simulated E-puck built from the world file (`tests/fake_webots.py`),
camera detection on synthetic frames, dialogue, grounding and the planning
pipeline with a scripted model. Scripted model responses are test fixtures,
not evidence of model performance.

## 5a. Blocked-door test worlds — TEST ONLY

> **IMPORTANT: these two worlds are for team testing only. Delete both files
> (and this section) before making the submission ZIP.** The assessed world is
> always `worlds/project_genai_embodied_assistant.wbt`.

Each is a copy of the assessed world with obstacles moved, so the robot's
recovery behaviour can be seen without editing the real world.

| World | What is blocked | Say | Expected behaviour |
|---|---|---|---|
| `worlds/test_blocked_office_a.wbt` | the movable black bar is turned across office A's main doorway | "go to office A" | blocked at the main doorway, tries the second approach (also blocked), then goes **through the lab and the lab-office A door** and arrives (about 4 minutes). Console route contains `door_west`. |
| `worlds/test_blocked_office_a_both.wbt` | the bar across the main doorway **and** a grey block in the lab-office A door | "look in office A for the red ball; if it is not there, check the meeting room" | three blocked attempts (about 3-4 minutes), says *"The way to office A is blocked. I will make a new plan."*, console shows `[REPLAN 1/2]` and `[REPLAN SITUATION] ... Cannot reach: office_a`, then it searches the meeting room, returns and reports including *"The route to office A was blocked."* |

How to run: Webots *File > Open World*, pick the test world, press play, wait
for *"Hello, I am ready"*, then speak (or `python send_goal.py ...`). The red ball
is still placed randomly, so the search result varies; the point of these
tests is the route and replanning behaviour.

## 5b. Before the submission ZIP (checklist)

1. Delete `worlds/test_blocked_office_a.wbt`, `worlds/test_blocked_office_a_both.wbt`
   (and any `worlds/.test_blocked_*` files Webots created) and README section 5a.
2. Delete `logs/` and `runtime/` (they are not in the repository anyway).
3. Do not include `models/*.gguf` or `models/vosk-model-*`.
4. Keep `config/llm_config.json` (no credentials in it).
5. `DEBUG_PLAN = None` in `project3_student.py` (already so).
6. Name the file `CSCI444_944_GroupXX_Project.zip` with the report
   `GroupXX_Project_Report.pdf`.

## 6. Project layout

| File | Responsibility |
|---|---|
| `controllers/project3_student/project3_student.py` | main loop: input, planning thread, execution, replanning |
| `speech_io.py` | `SpeechIO`: Vosk listener thread + pyttsx3 speaker thread |
| `nlu_dialogue.py` | `DialogueManager`: stop/status first, busy policy, clarification, yes/no answers |
| `grounding.py` | maps wording to rooms/objects/conditions; checks a plan matches the request |
| `llm_interface.py` | `LLMInterface`: all model code, prompt assembly, JSON extraction and repairs |
| `planning.py` | few-shot selection, correction loop, replanning, spoken plan summary |
| `plan_schema.py`, `plan_validator.py` | `PlanValidator`: the deterministic safety gate |
| `task_executive.py` | `TaskExecutive`: non-blocking step-by-step execution, timeouts, conditionals |
| `robot_skills.py` | `RobotSkills`: maps actions to navigation, search, speech, memory |
| `navigation.py` | `Navigator`: graph route, GPS/Compass control, obstacle stop, reroute, watchdog; the internal lab-office A and meeting room-office B doors are backup routes, used only when no normal route remains |
| `perception.py` | `ObjectPerception`: HSV colour + shape detection from camera pixels |
| `search.py` | `SearchController`: bounded multi-view search, FOUND / NOT_FOUND / UNCERTAIN |
| `world_state.py` | `WorldState`: goal, step, location, results, blocked routes; status and report text |
| `semantic_memory.py` | where objects were seen |
| `config/` | locations, waypoint graph (extended), objects, spawn points (Supervisor only), search viewpoints, LLM and speech settings |
| `prompts/` | planner instructions and few-shot examples |
| `controllers/scene_randomizer/` | supplied setup Supervisor (unchanged) |

## 7. Plan contract extension

Version 1.0 is unchanged except one optional field (validated and tested in
`tests/test_plan_validator.py`):

```json
{"action": "search", "object": "ball", "colour": "red",
 "locations": ["laboratory", "office_a", "office_b", "meeting_room"]}
```

The robot goes to each listed room in order, searches it and stops at the first
FOUND. Each room counts towards the navigation limit. It replaces a long chain
of conditionals that the 0.5B model could not write reliably.

## 8. Safety boundaries

- The LLM only proposes symbolic steps. Every plan, corrected plan and replan
  passes `PlanValidator` (allowed actions and fields, known values, types, size
  and nesting limits, condition grammar, no code or motor commands, semantic
  checks) before anything moves; conditions are evaluated by table lookup,
  never `eval()`.
- Outcomes come from sensors: arrival from GPS, obstacles from proximity
  sensors, the ball from camera pixels. Spoken reports are built from
  `WorldState`, not from the model's text.
- The robot controller never reads the Supervisor, DEF names, translations or
  `config/spawn_points.json`. The `[SETUP GROUND TRUTH]` console line is for
  evaluation only.
- Speech and model inference run in background threads; only the main loop
  touches Webots devices.

## 9. Development switches

- `DEBUG_PLAN` in `project3_student.py`: name of a fixed test plan from
  `debug_plans.py` to run without the LLM; `None` for normal use.
- Tracing: create an empty file `logs/ENABLE_TRACE` to record
  `logs/run_*.jsonl` traces and annotated camera frames for debugging. Without
  that file (the default in a fresh copy) nothing is logged. `logs/` is never
  shared or submitted.
