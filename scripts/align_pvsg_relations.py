#!/usr/bin/env python3
"""
PVSG Relation Alignment Tool

Maps our scene graph predictions to PVSG ground truth format for evaluation.
Uses class-based heuristic matching since PVSG lacks trajectory annotations.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set


def load_pvsg_gt(pvsg_path: str, video_id: str) -> Dict:
    """Load PVSG ground truth for a specific video."""
    with open(pvsg_path, 'r') as f:
        data = json.load(f)
    
    # Find the video in the dataset
    for video_data in data['data']:
        if video_data['video_id'] == video_id:
            return video_data
    
    raise ValueError(f"Video {video_id} not found in PVSG dataset")


def load_our_predictions(scene_graph_path: str) -> List[Dict]:
    """Load our scene graph predictions from JSONL."""
    predictions = []
    with open(scene_graph_path, 'r') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


def normalize_class_name(class_name: str) -> str:
    """Normalize class names for matching between detector and PVSG vocabulary."""
    class_name = class_name.lower().strip()
    
    # Map detector classes to PVSG classes
    mapping = {
        # Ball variations
        'sports ball': 'ball',
        
        # Bike variations
        'bicycle': 'bike',
        
        # Chair variations (detector often adds descriptors)
        'kitchen chair': 'chair',
        'dining chair': 'chair',
        'office chair': 'chair',
        'bedroom chair': 'chair',
        'accent chair': 'chair',
        
        # Person variations (PVSG uses adult/child, but we'll try adult first)
        'person': 'adult',
        'man': 'adult',
        'woman': 'adult',
        'boy': 'child',
        'girl': 'child',
        
        # Container variations
        'backpack handbag': 'bag',
        'handbag': 'bag',
        'backpack': 'bag',
        
        # Tableware
        'cup bowl': 'bowl',
        
        # Furniture
        'refrigerator': 'fridge',
        'couch': 'sofa',
        'tv': 'monitor',
        
        # Teddy bear
        'person teddy bear': 'teddy bear',
    }
    
    # First try exact match
    if class_name in mapping:
        return mapping[class_name]
    
    # Then try partial matches (e.g., "kitchen chair" contains "chair")
    for key, value in mapping.items():
        if key in class_name or class_name in key:
            return value
    
    # If no mapping found, return the original (cleaned)
    return class_name


def map_tracks_to_gt_objects(predictions: List[Dict], gt_objects: List[Dict]) -> Dict[str, int]:
    """
    Map our memory_ids to GT object_ids using class-based heuristic.
    
    Strategy:
    1. Group GT objects by category
    2. For each memory_id, find GT objects of same class
    3. Assign based on temporal overlap and order of appearance
    """
    # Build GT object lookup by category
    gt_by_category = defaultdict(list)
    for obj in gt_objects:
        category = normalize_class_name(obj['category'])
        gt_by_category[category].append(obj['object_id'])
    
    # Track which memory_ids appear in which frames
    memory_frames = defaultdict(set)
    memory_classes = {}
    
    for frame_data in predictions:
        frame_id = frame_data['frame']
        for node in frame_data['nodes']:
            # Handle both memory_id (from memory.json) and track_id (from tracks.jsonl)
            memory_id = node.get('memory_id') or node.get('track_id')
            if not memory_id:
                continue
            
            class_name = normalize_class_name(node['class'])
            memory_frames[memory_id].add(frame_id)
            memory_classes[memory_id] = class_name
    
    # Assign memory_ids to GT object_ids
    mapping = {}
    
    # Sort memory_ids by first appearance
    sorted_memories = sorted(memory_frames.items(), key=lambda x: min(x[1]))
    
    # GT object assignment counters (to distribute tracks if multiple GT objects exist)
    gt_counts = {cat: 0 for cat in gt_by_category.keys()}
    
    for memory_id, frames in sorted_memories:
        class_name = memory_classes[memory_id]
        
        gt_ids = gt_by_category.get(class_name, [])
        if gt_ids:
            # Map this track to one of the GT objects of this class
            # We cycle through GT objects if multiple exist for this class
            idx = gt_counts[class_name] % len(gt_ids)
            gt_id = gt_ids[idx]
            mapping[memory_id] = gt_id
            gt_counts[class_name] += 1
        else:
            # No GT object available - skip this track
            if class_name: # Only warn if class is not empty
                print(f"Warning: No GT object found for {memory_id} (class: {class_name})")
    
    return mapping


def aggregate_temporal_relations(predictions: List[Dict], track_mapping: Dict[str, int]) -> Dict[Tuple[int, int, str], List[Tuple[int, int]]]:
    """
    Aggregate frame-level relations into temporal ranges.
    
    Returns:
        Dict mapping (subject_id, object_id, predicate) -> list of frame ranges
    """
    # Track consecutive frames for each relation
    relation_frames = defaultdict(list)
    
    for frame_data in predictions:
        frame_id = frame_data['frame']
        
        for edge in frame_data.get('edges', []):
            # Handle both memory_id and track_id formats
            subject_mem = edge.get('subject')
            object_mem = edge.get('object')
            predicate = edge['relation']
            
            # Map to GT object IDs
            if subject_mem not in track_mapping or object_mem not in track_mapping:
                continue
            
            subject_id = track_mapping[subject_mem]
            object_id = track_mapping[object_mem]
            
            key = (subject_id, object_id, predicate)
            relation_frames[key].append(frame_id)
    
    # Convert frame lists to ranges
    relation_ranges = {}
    for key, frames in relation_frames.items():
        frames = sorted(set(frames))
        ranges = []
        
        if not frames:
            continue
        
        start = frames[0]
        end = frames[0]
        
        for frame in frames[1:]:
            if frame == end + 1:
                # Consecutive frame
                end = frame
            else:
                # Gap - save current range and start new one
                ranges.append([start, end])
                start = frame
                end = frame
        
        # Save final range
        ranges.append([start, end])
        relation_ranges[key] = ranges
    
    return relation_ranges


def convert_to_pvsg_format(relation_ranges: Dict[Tuple[int, int, str], List[Tuple[int, int]]]) -> List[List]:
    """Convert aggregated relations to PVSG format."""
    pvsg_relations = []
    
    for (subject_id, object_id, predicate), ranges in relation_ranges.items():
        pvsg_relations.append([
            subject_id,
            object_id,
            predicate,
            ranges
        ])
    
    return pvsg_relations


def save_pvsg_predictions(video_id: str, predictions: List[List], output_path: str):
    """Save predictions in PVSG format."""
    output_data = {
        'video_id': video_id,
        'relations': predictions
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved {len(predictions)} relations to {output_path}")


def print_alignment_summary(track_mapping: Dict[str, int], relation_ranges: Dict):
    """Print summary of alignment results."""
    print("\n" + "="*80)
    print("ALIGNMENT SUMMARY")
    print("="*80)
    
    print(f"\nTrack Mapping ({len(track_mapping)} tracks):")
    for mem_id, gt_id in sorted(track_mapping.items()):
        print(f"  {mem_id} -> GT object {gt_id}")
    
    print(f"\nRelations ({len(relation_ranges)} unique):")
    for (subj, obj, pred), ranges in sorted(relation_ranges.items()):
        total_frames = sum(end - start + 1 for start, end in ranges)
        print(f"  [{subj}] --{pred}--> [{obj}]: {len(ranges)} segments, {total_frames} frames")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Align scene graph predictions to PVSG format')
    parser.add_argument('--pvsg_gt', required=True, help='Path to pvsg.json')
    parser.add_argument('--video_id', required=True, help='Video ID to process')
    parser.add_argument('--predictions', required=True, help='Path to scene_graph.jsonl')
    parser.add_argument('--output', required=True, help='Output path for aligned predictions')
    parser.add_argument('--verbose', action='store_true', help='Print detailed alignment info')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading PVSG ground truth for video {args.video_id}...")
    gt_data = load_pvsg_gt(args.pvsg_gt, args.video_id)
    
    print(f"Loading predictions from {args.predictions}...")
    predictions = load_our_predictions(args.predictions)
    
    # Map tracks to GT objects
    print("Mapping tracks to GT objects...")
    track_mapping = map_tracks_to_gt_objects(predictions, gt_data['objects'])
    
    # Aggregate relations
    print("Aggregating temporal relations...")
    relation_ranges = aggregate_temporal_relations(predictions, track_mapping)
    
    # Convert to PVSG format
    pvsg_relations = convert_to_pvsg_format(relation_ranges)
    
    # Save results
    save_pvsg_predictions(args.video_id, pvsg_relations, args.output)
    
    # Print summary
    if args.verbose:
        print_alignment_summary(track_mapping, relation_ranges)
    
    print(f"\n✓ Alignment complete!")
    print(f"  Mapped {len(track_mapping)} tracks")
    print(f"  Generated {len(pvsg_relations)} relations")


if __name__ == '__main__':
    main()
