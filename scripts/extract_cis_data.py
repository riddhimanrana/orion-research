#!/usr/bin/env python3
"""
Extract CIS training data from Orion perception logs
"""
import json
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.causal_inference import AgentCandidate, StateChange


def load_perception_log(log_path: Path) -> Dict[str, Any]:
    """Load Orion pipeline results JSON"""
    with open(log_path) as f:
        return json.load(f)


def extract_agent_candidates(log_data: Dict) -> List[AgentCandidate]:
    """Extract agent candidates from perception log"""
    candidates = []
    
    # Get entities from the log
    entities = log_data.get('entities', [])
    
    for entity in entities:
        entity_id = entity.get('entity_id', '')
        class_name = entity.get('class_name', '')
        
        # Get observations (detections per frame)
        observations = entity.get('observations', [])
        
        for obs in observations:
            frame_idx = obs.get('frame_idx', 0)
            bbox = obs.get('bbox', [0, 0, 0, 0])
            
            # Create agent candidate
            candidate = AgentCandidate(
                entity_id=entity_id,
                class_name=class_name,
                frame_idx=frame_idx,
                bbox=bbox,
                confidence=obs.get('confidence', 1.0)
            )
            candidates.append(candidate)
    
    return candidates


def extract_state_changes(log_data: Dict) -> List[StateChange]:
    """Extract state changes from perception log"""
    state_changes = []
    
    # Get temporal events
    events = log_data.get('temporal_analysis', {}).get('events', [])
    
    for event in events:
        entity_id = event.get('entity_id', '')
        
        # Parse state change
        change = StateChange(
            entity_id=entity_id,
            from_state=event.get('from_state', 'unknown'),
            to_state=event.get('to_state', 'unknown'),
            frame_idx=event.get('frame_idx', 0),
            confidence=event.get('confidence', 1.0)
        )
        state_changes.append(change)
    
    return state_changes


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract CIS training data from Orion perception log'
    )
    parser.add_argument(
        'perception_log',
        type=str,
        help='Path to pipeline_results_*.json'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/hpo/extracted_data.pkl',
        help='Output pickle file'
    )
    
    args = parser.parse_args()
    
    # Load perception log
    log_path = Path(args.perception_log)
    if not log_path.exists():
        print(f"Error: {log_path} not found")
        return 1
    
    print(f"Loading perception log: {log_path}")
    log_data = load_perception_log(log_path)
    
    # Extract data
    print("Extracting agent candidates...")
    agent_candidates = extract_agent_candidates(log_data)
    print(f"  Found {len(agent_candidates)} agent candidates")
    
    print("Extracting state changes...")
    state_changes = extract_state_changes(log_data)
    print(f"  Found {len(state_changes)} state changes")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'agent_candidates': agent_candidates,
        'state_changes': state_changes,
        'source': str(log_path)
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n✓ Saved to {output_path}")
    print(f"\nNext: Run CIS optimization:")
    print(f"  python scripts/run_cis_hpo.py \\")
    print(f"    --extracted-data {output_path} \\")
    print(f"    --ground-truth data/benchmarks/ground_truth/vsgr_aspire_train_full.json \\")
    print(f"    --trials 200")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
