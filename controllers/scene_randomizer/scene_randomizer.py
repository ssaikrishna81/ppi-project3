"""Assessment setup only: randomise RED_BALL once when a trial starts.

This Supervisor never sends the selected location to the assessed controller.
Its console line is external evaluation ground truth for the marker/report.
"""
import json
import os
from pathlib import Path
import random
import secrets

from controller import Supervisor


def choose_seed():
    configured = os.environ.get("PROJECT_RANDOM_SEED")
    if configured is not None:
        try:
            return int(configured, 0)
        except ValueError:
            return sum(ord(character) for character in configured)
    return secrets.randbits(32)


supervisor = Supervisor()
project_root = Path(__file__).resolve().parents[2]
config = json.loads((project_root / "config" / "spawn_points.json").read_text(encoding="utf-8"))
target = supervisor.getFromDef(config["target_def"])
if target is None:
    print(f"[SETUP ERROR] DEF {config['target_def']!r} was not found; target was not randomised.")
else:
    seed = choose_seed()
    selected = random.Random(seed).choice(config["approved_spawn_points"])
    target.getField("translation").setSFVec3f([float(value) for value in selected["translation"]])
    target.resetPhysics()
    print(
        "[SETUP GROUND TRUTH] "
        f"seed={seed} spawn={selected['id']} location={selected['location']}"
    )

# Remain alive without performing further world mutations.
while supervisor.step(int(supervisor.getBasicTimeStep())) != -1:
    pass
