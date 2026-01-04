"""orion filter - FastVLM semantic filtering"""

import json
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


def run_filter(args) -> int:
    """Filter tracks using FastVLM and sentence transformer."""
    
    episode_dir = Path("results") / args.episode
    tracks_path = episode_dir / "tracks.jsonl"
    memory_path = episode_dir / "memory.json"
    
    if not tracks_path.exists():
        print(f"No tracks found for episode: {args.episode}")
        return 1
    
    print(f"\n  Stage 3: FILTERING (FastVLM)")
    print(f"  Sentence model: {args.sentence_model}")
    print("  " + "─" * 60)
    
    # Load tracks and memory
    all_observations = []
    with open(tracks_path) as f:
        for line in f:
            if line.strip():
                all_observations.append(json.loads(line))
    
    memory_objects = []
    if memory_path.exists():
        with open(memory_path) as f:
            memory = json.load(f)
            memory_objects = memory.get("objects", [])
    
    print(f"  Loaded {len(all_observations)} observations, {len(memory_objects)} memory objects")
    
    # Load episode meta
    meta_path = episode_dir / "episode_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    video_path = meta["video_path"]
    
    # Initialize sentence transformer
    try:
        from sentence_transformers import SentenceTransformer
        sentence_model = SentenceTransformer(args.sentence_model)
        print(f"  Loaded sentence transformer")
    except ImportError:
        print("  ✗ sentence-transformers not installed")
        print("  Run: pip install sentence-transformers")
        return 1
    except Exception as e:
        print(f"  ✗ Failed to load sentence model: {e}")
        return 1
    
    # Try to load FastVLM
    fastvlm = None
    try:
        import mlx.core as mx
        from mlx_vlm import load, generate
        
        model_path = "models/fastvlm-0.5b-mlx"
        if not Path(model_path).exists():
            model_path = "TIGER-Lab/FastVLM-0.5B"
        
        fastvlm_model, fastvlm_processor = load(model_path)
        fastvlm = (fastvlm_model, fastvlm_processor)
        print(f"  Loaded FastVLM")
    except ImportError:
        print("  ⚠ mlx-vlm not available, using detector labels only")
    except Exception as e:
        print(f"  ⚠ FastVLM load failed: {e}")
    
    # Build track → memory object mapping
    track_to_obj = {}
    for obj in memory_objects:
        for tid in obj.get("track_ids", [obj["id"]]):
            track_to_obj[tid] = obj
    
    # Process each memory object
    import cv2
    cap = cv2.VideoCapture(video_path)
    
    filtered_objects = []
    filtered_observations = []
    
    for obj in memory_objects:
        track_ids = obj.get("track_ids", [obj["id"]])
        
        # Get observations for this object
        obj_obs = [o for o in all_observations if o["track_id"] in track_ids]
        
        if not obj_obs:
            continue
        
        detector_label = obj["canonical_label"]
        
        # Get VLM description if available
        vlm_description = None
        if fastvlm:
            # Get middle observation's crop
            mid_obs = obj_obs[len(obj_obs) // 2]
            frame_id = mid_obs["frame_id"]
            bbox = mid_obs.get("bbox")
            
            if bbox:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                
                if ret:
                    x1, y1, x2, y2 = map(int, bbox)
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crop = frame[y1:y2, x1:x2]
                    
                    # Query VLM
                    try:
                        from mlx_vlm import generate
                        from PIL import Image
                        
                        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        prompt = f"What object is this? Reply with just the object name."
                        
                        output = generate(
                            fastvlm[0], fastvlm[1],
                            pil_crop, prompt,
                            max_tokens=20
                        )
                        vlm_description = output.strip()
                    except Exception as e:
                        logger.warning(f"VLM failed for {obj['id']}: {e}")
        
        # Compute semantic similarity
        if vlm_description:
            emb_detector = sentence_model.encode(detector_label)
            emb_vlm = sentence_model.encode(vlm_description)
            
            similarity = float(np.dot(emb_detector, emb_vlm) / 
                              (np.linalg.norm(emb_detector) * np.linalg.norm(emb_vlm) + 1e-8))
        else:
            similarity = 1.0  # No VLM, keep by default
        
        # Filter decision
        keep = similarity >= args.scene_threshold or vlm_description is None
        
        # Update object
        obj["vlm_description"] = vlm_description
        obj["label_similarity"] = round(similarity, 3)
        obj["keep"] = keep
        
        if keep:
            filtered_objects.append(obj)
            for o in obj_obs:
                o["memory_object_id"] = obj["id"]
                filtered_observations.append(o)
    
    cap.release()
    
    print(f"\n  Results:")
    print(f"    Original objects: {len(memory_objects)}")
    print(f"    Filtered objects: {len(filtered_objects)}")
    print(f"    Removed: {len(memory_objects) - len(filtered_objects)}")
    
    # Show filtered objects with reasons
    removed = [o for o in memory_objects if not o.get("keep", True)]
    if removed:
        print(f"\n  Removed objects:")
        for obj in removed[:5]:
            print(f"    - {obj['canonical_label']} (VLM: {obj.get('vlm_description', '?')}, sim: {obj.get('label_similarity', 0):.2f})")
    
    # Save filtered tracks
    filtered_path = episode_dir / "tracks_filtered.jsonl"
    with open(filtered_path, "w") as f:
        for obs in filtered_observations:
            f.write(json.dumps(obs) + "\n")
    
    print(f"\n  Saved {len(filtered_observations)} filtered observations to {filtered_path}")
    
    # Update memory with filtering info
    memory["objects"] = memory_objects  # Include keep flag
    memory["filtered_objects"] = filtered_objects
    with open(memory_path, "w") as f:
        json.dump(memory, f, indent=2)
    
    # Update status
    meta["status"]["filtered"] = True
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    return 0
