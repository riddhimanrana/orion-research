import json
from pathlib import Path

import numpy as np

from orion.perception.reid.reid import (
    cluster_tracks,
    update_tracks_with_embeddings,
)


def make_track_obs(track_id: int, frame_ids, category: str):
    return [
        {
            "track_id": track_id,
            "frame_id": f,
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "category": category,
            "confidence": 0.9,
        }
        for f in frame_ids
    ]


def test_cluster_tracks_merges_with_threshold():
    # Two tracks of 'tv' very similar; one 'chair' dissimilar
    emb_tv_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_tv_b = np.array([0.98, 0.01, 0.0], dtype=np.float32)
    emb_chair = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    track_embeddings = {1: emb_tv_a, 2: emb_tv_b, 3: emb_chair}
    tracks = []
    tracks += make_track_obs(1, [1, 2, 3], "tv")
    tracks += make_track_obs(2, [4, 5], "tv")
    tracks += make_track_obs(3, [1, 2], "chair")

    # Global threshold 0.75 merges tv tracks; chair remains separate
    memory, t2m, emb_map = cluster_tracks(track_embeddings, tracks, cosine_threshold=0.75)
    assert len(memory) == 2  # one for tv cluster, one for chair
    # Ensure both tv track_ids map to the same memory id
    mem_tv_ids = {t2m[1], t2m[2]}
    assert len(mem_tv_ids) == 1


def test_cluster_tracks_respects_class_thresholds():
    # Similar tv tracks, but with higher class threshold so they don't merge
    emb_tv_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_tv_b = np.array([0.98, 0.01, 0.0], dtype=np.float32)
    track_embeddings = {1: emb_tv_a, 2: emb_tv_b}
    tracks = []
    tracks += make_track_obs(1, [1, 2, 3], "tv")
    tracks += make_track_obs(2, [4, 5], "tv")

    # Global threshold 0.75 would merge, but with very high tv threshold it should not
    memory, t2m, _ = cluster_tracks(track_embeddings, tracks, cosine_threshold=0.75, class_thresholds={"tv": 0.99999})
    assert len(memory) == 2
    assert t2m[1] != t2m[2]


def test_update_tracks_with_embeddings(tmp_path: Path):
    # Minimal memory_objects and tracks
    memory_objects = {
        "mem_001": {
            "memory_id": "mem_001",
            "class": "tv",
            "prototype_embedding": "emb_001",
            "appearance_history": [
                {"track_id": 1, "first_frame": 1, "last_frame": 3, "observations": 3}
            ],
        }
    }

    lines = [
        json.dumps({"track_id": 1, "frame_id": 1, "bbox": [0, 0, 1, 1], "category": "tv"}),
        json.dumps({"track_id": 2, "frame_id": 1, "bbox": [0, 0, 1, 1], "category": "chair"}),
    ]
    tracks_path = tmp_path / "tracks.jsonl"
    tracks_path.write_text("\n".join(lines) + "\n")

    updated_path = update_tracks_with_embeddings(tracks_path, memory_objects)
    assert updated_path.exists()
    updated = [json.loads(l) for l in updated_path.read_text().strip().splitlines()]
    # Track 1 should receive embedding_id, track 2 should not
    tid1 = [x for x in updated if x["track_id"] == 1][0]
    tid2 = [x for x in updated if x["track_id"] == 2][0]
    assert tid1["embedding_id"] == "emb_001"
    assert "embedding_id" not in tid2
