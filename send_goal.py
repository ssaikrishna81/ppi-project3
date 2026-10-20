"""Type a goal to the running Webots controller (debugging only).

    python send_goal.py go to the lab
    python send_goal.py "what are you doing?"
    python send_goal.py stop
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "controllers" / "project3_student"))
from request_channel import write_request  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    request = write_request(" ".join(sys.argv[1:]))
    print(f"sent: {request['text']}")
