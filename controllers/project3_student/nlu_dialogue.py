"""Dialogue state and clarification support.

The generative planner may interpret language, but the controller still needs
explicit dialogue state so that incomplete requests and follow-up answers can
be handled coherently.

handle(text, busy) decides what one utterance means and returns an
instruction for the controller (plain data, no side effects):

  {"type": "stop"}                     stop / cancel / halt (always first)
  {"type": "status"}                   "what are you doing", "why", "where are you"
  {"type": "plan", "goal": text}       send this goal to the planner
  {"type": "say", "text": text}        just reply (clarification, refusal...)
  {"type": "ignore"}                   empty or unrecognised noise

Rules, in order:
1. stop words always win, even while busy;
2. status questions are answered from WorldState (no LLM), even while busy;
3. while a task is running or being planned, a new goal is refused with a
   reply: the user must say "stop" first (one task at a time, nothing
   queued behind the user's back);
4. if the robot asked something, the next utterance is read as the answer:
   - "which colour?": a colour completes the pending goal;
   - a yes/no question from a plan ("Shall I drive to office B
     instead?"): yes plans the suggestion, no drops it;
   - anything else is treated as a new request;
5. a search request whose target is unclear ("find the ball") gets a
   clarification question, unless this conversation already settled the
   colour for that object;
6. otherwise the text goes to the planner.
"""
import re
from dataclasses import dataclass, field
from typing import Optional

from grounding import COLOURS, ground

STOP = r"^\s*(stop|cancel|halt|abort|freeze|stop now|stop moving|emergency stop)\b"
# Loose on purpose: the recogniser often drops small words ("what you doing").
STATUS = (r"\bwhat\b(\s+\w+){0,2}\s+(you|u)\s+(doing|up to)\b|"
          r"\bwhy\b(\s+\w+){0,2}\s+(you|u)\b|\bwhy\b|"
          r"\bwhat('s| is) (your )?status\b|^\s*status\b|\bwhere are you\b|"
          r"\bwhat('s| is) happening\b")
YES = r"^\s*(yes|yeah|yep|sure|ok(ay)?|please do|go ahead|do it|correct)\b"
NO = r"^\s*(no|nope|never ?mind|don'?t|do not|cancel that|forget it)\b"

# Common mis-hearings of room names by the speech recogniser.
HEARING_FIXES = [
    (r"\boffice (be|bee)\b", "office b"),
    (r"\boffice (eh|hey|ay)\b", "office a"),
    (r"\boff a\b", "office a"),
    (r"\bmeeting rooms\b", "meeting room"),
]


@dataclass
class DialogueState:
    pending_goal: Optional[str] = None
    awaiting_clarification: bool = False
    clarification_topic: Optional[str] = None     # "colour" or "plan_question"
    pending_question: Optional[str] = None
    known_colours: dict = field(default_factory=dict)  # object -> colour said


class DialogueManager:
    def __init__(self):
        self.state = DialogueState()

    # ------------------------------------------------------------ main
    def handle(self, text: str, busy: bool = False) -> dict:
        text = self.clean(text)
        if not text:
            return {"type": "ignore", "reason": "nothing recognised"}
        if re.search(STOP, text):
            self._clear()
            return {"type": "stop"}
        if re.search(STATUS, text):
            return {"type": "status"}
        if busy:
            g = ground(text)
            if not (g["locations"] or g["object"] or len(text.split()) >= 3):
                # One or two stray words while working ("head", "help"):
                # probably background speech, so do not interrupt with a reply.
                return {"type": "ignore", "reason": "ignored noise"}
            return {"type": "say",
                    "text": "I am still working on the last request. "
                            "Say stop if you want me to cancel it."}
        if self.state.awaiting_clarification:
            answer = self._answer(text)
            if answer is not None:
                return answer
        if not actionable(text):
            # Nothing the robot could act on (garbled or cut-off speech such
            # as "where if see" or "go back to"): ask again straight away
            # instead of spending the model's time on it.
            return {"type": "say", "text": REPEAT_PROMPT}
        ambiguity = self.analyse_for_missing_information(text)
        if ambiguity["needs_clarification"]:
            self.state.pending_goal = text
            self.state.awaiting_clarification = True
            self.state.clarification_topic = "colour"
            self.state.pending_question = ambiguity["question"]
            return {"type": "say", "text": ambiguity["question"]}
        return self._plan(ambiguity.get("goal", text))

    def expect_answer(self, question: str):
        """The robot asked a question as part of a plan (an 'ask' step)."""
        self.state.awaiting_clarification = True
        self.state.clarification_topic = "plan_question"
        self.state.pending_question = question

    def clean(self, text):
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        for pattern, fix in HEARING_FIXES:
            text = re.sub(pattern, fix, text)
        return text

    # ------------------------------------------------------------ rules
    def analyse_for_missing_information(self, text: str) -> dict:
        """Return explicit ambiguity/missing-information signals."""
        g = ground(text)
        if not g["search"] or g["colour"] or g["unsupported"]:
            if g["object"] and g["colour"]:
                self.state.known_colours[g["object"]] = g["colour"]
            return {"needs_clarification": False, "question": None}
        known = self.state.known_colours.get(g["object"])
        if known:
            # Earlier in this conversation the user named the colour.
            goal = re.sub(rf"\b(the|a)\s+{g['object']}\b",
                          f"the {known} {g['object']}", text, count=1)
            if goal == text:
                goal = f"{text} (the {known} {g['object']})"
            return {"needs_clarification": False, "question": None, "goal": goal}
        noun = g["object"]
        return {"needs_clarification": True,
                "question": f"Which {noun} should I look for: the red, green, "
                            f"blue or yellow {noun}?"}

    def consume_followup(self, text: str) -> str:
        """Combine a follow-up answer with the pending goal."""
        colour = next((c for c in COLOURS if re.search(rf"\b{c}\b", text)), None)
        goal = self.state.pending_goal or text
        g = ground(goal)
        if colour and g["object"]:
            self.state.known_colours[g["object"]] = colour
            combined = re.sub(rf"\b{g['object']}\b", f"{colour} {g['object']}",
                              goal, count=1)
            return combined
        return f"{goal} {text}".strip()

    def _answer(self, text):
        topic = self.state.clarification_topic
        if topic == "colour":
            colour = next((c for c in COLOURS if re.search(rf"\b{c}\b", text)), None)
            if colour:
                goal = self.consume_followup(text)
                self._clear()
                return self._plan(goal)
            if re.search(NO, text):
                self._clear()
                return {"type": "say", "text": "Okay, I will not search."}
            # Not an answer: treat it as a new request.
            self._clear()
            return None
        if topic == "plan_question":
            suggestion = suggestion_from_question(self.state.pending_question)
            if re.search(YES, text) and suggestion:
                self._clear()
                return self._plan(suggestion)
            if re.search(NO, text) or (re.search(YES, text) and not suggestion):
                self._clear()
                return {"type": "say", "text": "Okay. What would you like me to do?"}
            self._clear()
            return None
        self._clear()
        return None

    def _plan(self, goal):
        g = ground(goal)
        if g["object"] and g["colour"]:
            self.state.known_colours[g["object"]] = g["colour"]
        return {"type": "plan", "goal": goal}

    def _clear(self):
        self.state.pending_goal = None
        self.state.awaiting_clarification = False
        self.state.clarification_topic = None
        self.state.pending_question = None


REPEAT_PROMPT = ("Sorry, I did not catch that. Could you say it again? For example: "
                 "go to office B, or find the red ball.")


def actionable(text):
    """Does the request mention anything the robot could plan for?

    A place, an object to find, coming back, looking around, or something
    it cannot do (which still deserves a proper refusal). Everything else
    is asked again.
    """
    # Cut off mid-sentence ("go back to"). "a"/"b" are not in this list:
    # they end valid requests such as "go to office a".
    if re.search(r"\b(to|the|and|then|in|for|of|at|into|with)\s*$", text):
        return False
    g = ground(text)
    return bool(g["locations"] or g["object"] or g["return_home"]
                or g["observe"] or g["unsupported"])


def suggestion_from_question(question):
    """'... Shall I drive to office B instead?' -> 'drive to office B'."""
    if not question:
        return None
    match = re.search(r"\b(shall|should|can|may) i (.+?)(\binstead\b)?\s*\??\s*$",
                      question.strip(), re.IGNORECASE)
    if not match:
        return None
    suggestion = match.group(2).strip().rstrip("?.! ")
    return suggestion or None
