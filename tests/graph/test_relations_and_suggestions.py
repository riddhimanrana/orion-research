import json
from pathlib import Path

from orion.graph import (
    build_merge_suggestions,
    build_relation_events,
    build_state_events,
)


def _mk_obs(frame, tid, emb, cat, bbox, wh=(100,100)):
    return {
        "frame_id": frame,
        "track_id": tid,
        "embedding_id": emb,
        "category": cat,
        "bbox": bbox,
        "frame_width": wh[0],
        "frame_height": wh[1],
    }


def _mk_memory(objs):
    return {
        "objects": objs,
        "embeddings": {o["prototype_embedding"]: [1.0, 0.0, 0.0] for o in objs},
    }


def test_state_debounce_suppresses_noise():
    memory = _mk_memory([
        {"memory_id": "m_obj", "class": "bottle", "prototype_embedding": "e_obj", "first_seen_frame": 1, "last_seen_frame": 5, "appearance_history": [{"first_seen_frame": 1, "last_seen_frame": 5, "track_ids": [1]}]},
        {"memory_id": "m_p", "class": "person", "prototype_embedding": "e_p", "first_seen_frame": 1, "last_seen_frame": 5, "appearance_history": [{"first_seen_frame": 1, "last_seen_frame": 5, "track_ids": [2]}]},
    ])

    tracks = [
        _mk_obs(1, 1, "e_obj", "bottle", [10, 10, 20, 20]),
        _mk_obs(1, 2, "e_p", "person", [60, 60, 90, 90]),
        # Single-frame overlap (noise)
        _mk_obs(2, 1, "e_obj", "bottle", [60, 60, 80, 80]),
        _mk_obs(2, 2, "e_p", "person", [55, 55, 85, 85]),
        # Back to far
        _mk_obs(3, 1, "e_obj", "bottle", [10, 10, 20, 20]),
        _mk_obs(3, 2, "e_p", "person", [60, 60, 90, 90]),
    ]

    ev = build_state_events(memory, tracks, iou_threshold=0.1, debounce_window=2)
    assert [e for e in ev if e.get("state") == "held_by_person"] == []


def test_relation_near_and_on_debounce():
    memory = _mk_memory([
        {"memory_id": "A", "class": "cup", "prototype_embedding": "eA", "first_seen_frame": 1, "last_seen_frame": 12, "appearance_history": [{"first_seen_frame": 1, "last_seen_frame": 12, "track_ids": [1]}]},
        {"memory_id": "B", "class": "table", "prototype_embedding": "eB", "first_seen_frame": 1, "last_seen_frame": 12, "appearance_history": [{"first_seen_frame": 1, "last_seen_frame": 12, "track_ids": [2]}]},
    ])

    tracks = [
        # Far
        _mk_obs(1, 1, "eA", "cup", [10, 10, 20, 20]),
        _mk_obs(1, 2, "eB", "table", [60, 60, 90, 90]),
        # Near toggles for 2 frames (debounce=2) -> emit True at frame 3
        _mk_obs(2, 1, "eA", "cup", [50, 50, 60, 60]),
        _mk_obs(2, 2, "eB", "table", [62, 50, 82, 80]),
        _mk_obs(3, 1, "eA", "cup", [50, 50, 60, 60]),
        _mk_obs(3, 2, "eB", "table", [62, 50, 82, 80]),
        # Move far again for 2 frames -> emit False at frame 5
        _mk_obs(4, 1, "eA", "cup", [10, 10, 20, 20]),
        _mk_obs(4, 2, "eB", "table", [60, 60, 90, 90]),
        _mk_obs(5, 1, "eA", "cup", [10, 10, 20, 20]),
        _mk_obs(5, 2, "eB", "table", [60, 60, 90, 90]),
        # On: place A immediately above B (frames 10-11) -> emit True at 11
        _mk_obs(10, 1, "eA", "cup", [60, 40, 80, 60]),
        _mk_obs(10, 2, "eB", "table", [55, 60, 85, 90]),
        _mk_obs(11, 1, "eA", "cup", [60, 40, 80, 60]),
        _mk_obs(11, 2, "eB", "table", [55, 60, 85, 90]),
    ]

    rel_events = build_relation_events(
        memory,
        tracks,
        relations=["near", "on"],
        near_dist_norm=0.2,
        on_h_overlap=0.3,
        on_vgap_norm=0.05,
        debounce_window=2,
    )

    near_AB = [e for e in rel_events if e.get("relation") == "near" and e.get("subject") == "A" and e.get("object") == "B"]
    on_AB = [e for e in rel_events if e.get("relation") == "on" and e.get("subject") == "A" and e.get("object") == "B"]

    # Expect near True at frame 3 and False at frame 5
    assert any(e["value"] is True and e["frame"] == 3 for e in near_AB)
    assert any(e["value"] is False and e["frame"] == 5 for e in near_AB)

    # Expect on True at frame 11
    assert any(e["relation"] == "on" and e["value"] is True and e["frame"] == 11 for e in on_AB)


def test_merge_suggestions_contains_pair():
    memory = {
        "objects": [
            {"memory_id": "X", "class": "bottle", "prototype_embedding": "eX", "first_seen_frame": 10, "last_seen_frame": 20, "appearance_history": [{"first_seen_frame": 10, "last_seen_frame": 20, "track_ids": [11]}]},
            {"memory_id": "Y", "class": "bottle", "prototype_embedding": "eY", "first_seen_frame": 21, "last_seen_frame": 40, "appearance_history": [{"first_seen_frame": 21, "last_seen_frame": 40, "track_ids": [12]}]},
        ],
        "embeddings": {
            "eX": [1.0, 0.0, 0.0],
            "eY": [0.998, 0.05, 0.0],
        },
    }
    tracks = [
        _mk_obs(20, 11, "eX", "bottle", [50, 50, 80, 80]),
        _mk_obs(21, 12, "eY", "bottle", [52, 52, 82, 82]),
    ]

    sugg = build_merge_suggestions(memory, tracks, split_sim_threshold=0.95, max_gap_frames=30, spatial_dist_norm=0.3, top_k=5)
    assert any(set(s["memory_ids"]) == {"X", "Y"} for s in sugg)
