"""DialogueManager: stop/status priority, busy policy, clarification and
follow-up answers (including the "yes" case seen in Webots testing)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "controllers" / "project3_student"))

from nlu_dialogue import DialogueManager, suggestion_from_question  # noqa: E402
from speech_io import SpeechIO  # noqa: E402


class DialogueTests(unittest.TestCase):
    def setUp(self):
        self.d = DialogueManager()

    def test_plain_goal_goes_to_planner(self):
        self.assertEqual(self.d.handle("Go to the lab"),
                         {"type": "plan", "goal": "go to the lab"})

    def test_stop_wins_even_while_busy(self):
        self.assertEqual(self.d.handle("stop", busy=True)["type"], "stop")
        self.assertEqual(self.d.handle("Cancel that please", busy=True)["type"], "stop")

    def test_status_question_while_busy(self):
        self.assertEqual(self.d.handle("what are you doing and why?", busy=True)["type"],
                         "status")

    def test_garbled_or_cut_off_speech_is_asked_again(self):
        # Heard in Webots testing.
        for phrase in ("where if see", "go back to", "of"):
            action = self.d.handle(phrase)
            self.assertEqual(action["type"], "say", phrase)
            self.assertIn("say it again", action["text"])

    def test_actionable_requests_still_go_to_planner(self):
        # Webots test: "go to office a" was wrongly treated as cut off.
        for phrase in ("go back home", "look around", "find the red ball",
                       "jump over the table", "come back",
                       "go to office a", "go office a", "office a", "go to office b",
                       "go to the lab", "go to the meeting room", "go to reception",
                       "go to the corridor", "search office a for the red ball"):
            self.assertEqual(self.d.handle(phrase)["type"], "plan", phrase)

    def test_loose_status_wording(self):
        # Webots test: "what you doing" was planned as a goal.
        for phrase in ("what you doing", "why you doing", "what are you doing now"):
            self.assertEqual(self.d.handle(phrase)["type"], "status", phrase)

    def test_ignore_has_a_reason(self):
        action = self.d.handle("head", busy=True)
        self.assertEqual(action["type"], "ignore")
        self.assertIn("reason", action)

    def test_stray_words_while_busy_are_ignored(self):
        # Webots test: "head", "home", "help" picked up while eating.
        for word in ("head", "home", "help"):
            self.assertEqual(self.d.handle(word, busy=True)["type"], "ignore")

    def test_new_goal_while_busy_is_refused(self):
        action = self.d.handle("go to office A", busy=True)
        self.assertEqual(action["type"], "say")
        self.assertIn("stop", action["text"])

    def test_ambiguous_ball_asks_for_colour(self):
        action = self.d.handle("find the ball")
        self.assertEqual(action["type"], "say")
        self.assertIn("Which ball", action["text"])

    def test_colour_answer_completes_goal(self):
        self.d.handle("find the ball")
        action = self.d.handle("the red one")
        self.assertEqual(action, {"type": "plan", "goal": "find the red ball"})

    def test_colour_remembered_in_conversation(self):
        self.d.handle("find the red ball")
        action = self.d.handle("now check office A for the ball")
        self.assertEqual(action["type"], "plan")
        self.assertIn("red ball", action["goal"])

    def test_non_answer_becomes_new_request(self):
        self.d.handle("find the ball")
        self.assertEqual(self.d.handle("go to the lab"),
                         {"type": "plan", "goal": "go to the lab"})

    def test_yes_to_robot_question_plans_the_suggestion(self):
        # Webots test 2026-10-20: "yes" was planned on its own and the robot
        # drove to the laboratory.
        self.d.expect_answer("I cannot fly or pass through walls. "
                             "Shall I drive to office B instead?")
        action = self.d.handle("yes")
        self.assertEqual(action["type"], "plan")
        self.assertEqual(action["goal"].lower(), "drive to office b")

    def test_no_to_robot_question(self):
        self.d.expect_answer("Shall I drive to office B instead?")
        self.assertEqual(self.d.handle("no")["type"], "say")
        # The question is closed: the next utterance is a normal request.
        self.assertEqual(self.d.handle("go to the lab")["type"], "plan")

    def test_unsupported_is_not_a_clarification(self):
        self.assertEqual(self.d.handle("fly through the wall to office B")["type"], "plan")

    def test_misheard_room_names_fixed(self):
        self.assertEqual(self.d.handle("go to office be")["goal"], "go to office b")
        self.assertEqual(self.d.handle("search off a for the red ball")["goal"],
                         "search office a for the red ball")

    def test_empty_ignored(self):
        self.assertEqual(self.d.handle("  ")["type"], "ignore")

    def test_suggestion_extraction(self):
        self.assertEqual(suggestion_from_question(
            "I cannot pick up objects. Shall I find the green ball instead?"),
            "find the green ball")
        self.assertIsNone(suggestion_from_question("Which ball?"))


class SpeechIOTests(unittest.TestCase):
    def test_without_threads_never_blocks(self):
        speech = SpeechIO(start=False)
        self.assertEqual(speech.listen(), "")
        speech.heard.put("go to the lab")
        self.assertEqual(speech.listen(), "go to the lab")
        speech.speak("hello")          # queued; no engine running in tests
        self.assertFalse(speech.speaking)


if __name__ == "__main__":
    unittest.main()
