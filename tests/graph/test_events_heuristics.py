import json
from pathlib import Path

from orion.graph import build_split_events, build_state_events


def make_tracks(lines):
    return lines


def test_state_change_held_by_person(tmp_path: Path):
    memory = {
        "objects": [
            {
                "memory_id": "mem_obj",
                "class": "bottle",
                "prototype_embedding": "e_obj",
                "first_seen_frame": 1,
                "last_seen_frame": 3,
                "appearance_history": [
                    {"first_seen_frame": 1, "last_seen_frame": 3, "track_ids": [1]}
                ],
            },
            {
                "memory_id": "mem_person",
                "class": "person",
                "prototype_embedding": "e_person",
                "first_seen_frame": 1,
                "last_seen_frame": 3,
                "appearance_history": [
                    {"first_seen_frame": 1, "last_seen_frame": 3, "track_ids": [2]}
                ],
            },
        ],
        "embeddings": {
            "e_obj": [1.0, 0.0],
            "e_person": [0.0, 1.0],
        },
    }

    # Frames: 1 no overlap, 2 overlap, 3 no overlap => state false->true->false
    tracks = [
        {"frame_id": 1, "track_id": 1, "embedding_id": "e_obj", "category": "bottle", "bbox": [10, 10, 20, 20], "frame_width": 100, "frame_height": 100},
        {"frame_id": 1, "track_id": 2, "embedding_id": "e_person", "category": "person", "bbox": [60, 60, 90, 90], "frame_width": 100, "frame_height": 100},
        {"frame_id": 2, "track_id": 1, "embedding_id": "e_obj", "category": "bottle", "bbox": [60, 60, 80, 80], "frame_width": 100, "frame_height": 100},
        {"frame_id": 2, "track_id": 2, "embedding_id": "e_person", "category": "person", "bbox": [55, 55, 85, 85], "frame_width": 100, "frame_height": 100},
        {"frame_id": 3, "track_id": 1, "embedding_id": "e_obj", "category": "bottle", "bbox": [10, 10, 20, 20], "frame_width": 100, "frame_height": 100},
        {"frame_id": 3, "track_id": 2, "embedding_id": "e_person", "category": "person", "bbox": [60, 60, 90, 90], "frame_width": 100, "frame_height": 100},
    ]

    events = build_state_events(memory, tracks, iou_threshold=0.1)
    # Expect two state_change events for mem_obj at frames 2 and 3
    held_events = [e for e in events if e.get("type") == "state_change" and e.get("memory_id") == "mem_obj" and e.get("state") == "held_by_person"]
    assert len(held_events) == 2
    assert held_events[0]["frame"] == 2 and held_events[0]["value"] is True
    assert held_events[1]["frame"] == 3 and held_events[1]["value"] is False


def test_split_detection(tmp_path: Path):
    memory = {
        "objects": [
            {
                "memory_id": "mem_A",
                "class": "bottle",
                "prototype_embedding": "e_A",
                "first_seen_frame": 10,
                "last_seen_frame": 20,
                "appearance_history": [
                    {"first_seen_frame": 10, "last_seen_frame": 20, "track_ids": [11]}
                ],
            },
            {
                "memory_id": "mem_B",
                "class": "bottle",
                "prototype_embedding": "e_B",
                "first_seen_frame": 21,
                "last_seen_frame": 30,
                "appearance_history": [
                    {"first_seen_frame": 21, "last_seen_frame": 30, "track_ids": [12]}
                ],
            },
        ],
        "embeddings": {
            # Highly similar
            "e_A": [1.0, 0.0, 0.0],
            "e_B": [0.999, 0.001, 0.0],
        },
    }

    # Last bbox of A (frame 20) near first bbox of B (frame 21)
    tracks = [
        {"frame_id": 20, "track_id": 11, "embedding_id": "e_A", "category": "bottle", "bbox": [50, 50, 80, 80], "frame_width": 100, "frame_height": 100},
        {"frame_id": 21, "track_id": 12, "embedding_id": "e_B", "category": "bottle", "bbox": [52, 52, 82, 82], "frame_width": 100, "frame_height": 100},
    ]

    events = build_split_events(memory, tracks, split_sim_threshold=0.95, max_gap_frames=5, spatial_dist_norm=0.2)
    splits = [e for e in events if e.get("type") == "split"]
    assert len(splits) == 1
    s = splits[0]
    assert set(s["memory_ids"]) == {"mem_A", "mem_B"}
    assert s["frame"] == 21
