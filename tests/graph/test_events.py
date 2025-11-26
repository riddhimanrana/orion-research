import json
from pathlib import Path

from orion.graph import build_events


def test_basic_lifecycle_events():
    # Synthetic memory with two segments (two tracks) -> appeared, disappeared, reappeared, merged, disappeared
    memory = {
        "objects": [
            {
                "memory_id": "mem_001",
                "class": "tv",
                "appearance_history": [
                    {"track_id": 10, "first_frame": 100, "last_frame": 120, "observations": 3},
                    {"track_id": 11, "first_frame": 140, "last_frame": 145, "observations": 2},
                ],
            }
        ]
    }

    events = build_events(memory)
    types = [e["type"] for e in events]
    assert types[0] == "appeared"
    # Order should include disappeared after first seg, then reappeared (+ merged), then final disappeared
    assert "reappeared" in types
    assert "merged" in types
    assert types[-1] == "disappeared"
    # Check gap
    rep = next(e for e in events if e["type"] == "reappeared")
    assert rep["gap_frames"] == 20
