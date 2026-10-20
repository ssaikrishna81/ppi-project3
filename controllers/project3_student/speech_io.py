"""Speech input/output interface.

Recognition: Vosk (vosk-model-small-en-us-0.15) fed by a sounddevice
microphone stream. Synthesis: pyttsx3 (Windows SAPI voices).

Both run in their own background threads, so the Webots loop never waits:
  - the microphone callback only copies audio blocks into a queue;
  - the listener thread feeds Vosk and puts finished utterances in
    self.heard (a queue the controller polls with listen());
  - the speaker thread owns the pyttsx3 engine (it must be created and used
    in one thread) and speaks queued sentences one at a time.
While the robot is talking, and for a short time after, microphone audio is
discarded so the robot does not hear and act on its own voice.

If the microphone, Vosk model or TTS engine is unavailable, the controller
still runs: replies are printed and goals can be typed (send_goal.py).
"""
import json
import queue
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "speech_config.json"


class SpeechIO:
    def __init__(self, config_path=CONFIG_PATH, start=True):
        try:
            self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.config = {"enabled": False}
        self.console_fallback = True
        self.heard = queue.Queue()        # finished utterances (text)
        self._audio = queue.Queue()       # raw microphone blocks
        self._say = queue.Queue()         # sentences to speak
        self._speaking = threading.Event()
        self._mute_until = 0.0
        self._stop = threading.Event()
        self.asr_ready = False
        self.tts_ready = False
        self.status = []
        if start and self.config.get("enabled", True):
            threading.Thread(target=self._speaker_loop, name="tts", daemon=True).start()
            threading.Thread(target=self._listener_loop, name="asr", daemon=True).start()

    # ------------------------------------------------------------ public
    def listen(self) -> str:
        """Return one recognised user utterance, or "" (never blocks)."""
        try:
            return self.heard.get_nowait()
        except queue.Empty:
            return ""

    def speak(self, text: str) -> None:
        """Speak the supplied robot response (queued; returns at once)."""
        print(f"[ROBOT] {text}")
        if self.tts_ready or self.config.get("enabled", True):
            self._say.put(str(text))

    @property
    def speaking(self):
        return self._speaking.is_set()

    def close(self):
        self._stop.set()

    # ------------------------------------------------------------ speaker
    def _speaker_loop(self):
        try:
            try:
                import comtypes          # SAPI needs COM set up in this thread
                comtypes.CoInitialize()
            except Exception:
                pass
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", int(self.config.get("tts_rate", 170)))
            self.tts_ready = True
        except Exception as error:   # no TTS: replies stay printed only
            self.status.append(f"speech output unavailable: {error}")
            print(f"[SPEECH] output unavailable ({error}); replies are printed only")
            return
        while not self._stop.is_set():
            try:
                text = self._say.get(timeout=0.2)
            except queue.Empty:
                continue
            self._speaking.set()
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as error:
                print(f"[SPEECH] could not speak: {error}")
            finally:
                self._mute_until = time.monotonic() + float(
                    self.config.get("mute_after_speaking_s", 0.6))
                self._speaking.clear()

    # ------------------------------------------------------------ listener
    def _listener_loop(self):
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model, SetLogLevel
            SetLogLevel(-1)
            model_path = Path(self.config.get("vosk_model", ""))
            if not model_path.is_absolute():
                model_path = PROJECT_ROOT / model_path
            if not model_path.is_dir():
                raise FileNotFoundError(f"Vosk model not found at {model_path}")
            model = Model(str(model_path))
            device = self.config.get("input_device")
            rate = self._pick_rate(sd, device)
            if self.config.get("use_grammar", True):
                words = sorted(set(self.config.get("vocabulary", "").split()))
                recogniser = KaldiRecognizer(model, rate, json.dumps(words + ["[unk]"]))
            else:
                recogniser = KaldiRecognizer(model, rate)
            stream = sd.RawInputStream(
                samplerate=rate, blocksize=int(self.config.get("block_size", 4000)),
                device=device, dtype="int16", channels=1, callback=self._on_audio)
            stream.start()
        except Exception as error:   # no microphone: typed input still works
            self.status.append(f"speech input unavailable: {error}")
            print(f"[SPEECH] input unavailable ({error}); use send_goal.py to type")
            return
        self.asr_ready = True
        name = sd.query_devices(device, "input")["name"]
        print(f"[SPEECH] listening on '{name}' at {rate} Hz")
        try:
            while not self._stop.is_set():
                try:
                    block = self._audio.get(timeout=0.2)
                except queue.Empty:
                    continue
                if self._speaking.is_set() or time.monotonic() < self._mute_until:
                    recogniser.Reset()      # drop our own voice
                    continue
                if recogniser.AcceptWaveform(block):
                    text = json.loads(recogniser.Result()).get("text", "")
                    text = text.replace("[unk]", "").strip()
                    if text:
                        self.heard.put(text)
        finally:
            stream.stop()
            stream.close()

    def _on_audio(self, indata, frames, time_info, status):
        # Runs in the audio driver's thread: keep it short.
        self._audio.put(bytes(indata))

    def _pick_rate(self, sd, device):
        """Use 16 kHz if the device accepts it, else its native rate."""
        preferred = int(self.config.get("sample_rate", 16000))
        try:
            sd.check_input_settings(device=device, channels=1, dtype="int16",
                                    samplerate=preferred)
            return preferred
        except Exception:
            return int(sd.query_devices(device, "input")["default_samplerate"])
