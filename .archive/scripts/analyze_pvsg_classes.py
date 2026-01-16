#!/usr/bin/env python3
"""Analyze PVSG dataset to find all object classes used in ground truth."""

import json
from collections import Counter

def main():
    with open('datasets/PVSG/pvsg.json', 'r') as f:
        pvsg = json.load(f)
    
    all_classes = []
    all_predicates = []
    
    for video in pvsg['data']:
        # Collect object classes
        for obj in video.get('objects', []):
            cls = obj.get('category', 'unknown')
            all_classes.append(cls)
        
        # Collect predicates
        for rel in video.get('relations', []):
            _, _, pred, _ = rel
            all_predicates.append(pred)
    
    # Count frequencies
    class_counts = Counter(all_classes)
    pred_counts = Counter(all_predicates)
    
    print("="*80)
    print("PVSG Object Classes (sorted by frequency)")
    print("="*80)
    for cls, count in class_counts.most_common(50):
        print(f"  {cls:<30} {count:>6}")
    
    print(f"\nTotal unique classes: {len(class_counts)}")
    print(f"Total object instances: {sum(class_counts.values())}")
    
    print("\n" + "="*80)
    print("PVSG Predicates (sorted by frequency)")
    print("="*80)
    for pred, count in pred_counts.most_common():
        print(f"  {pred:<30} {count:>6}")
    
    print(f"\nTotal unique predicates: {len(pred_counts)}")
    print(f"Total relation instances: {sum(pred_counts.values())}")
    
    # Generate YOLO-World prompt for top classes
    top_classes = [cls for cls, _ in class_counts.most_common(100)]
    yolo_prompt = " . ".join(top_classes)
    
    print("\n" + "="*80)
    print("Suggested YOLO-World Prompt (top 100 classes):")
    print("="*80)
    print(yolo_prompt[:500] + "...")
    
    # Save to file
    with open('pvsg_yoloworld_prompt.txt', 'w') as f:
        f.write(yolo_prompt)
    
    print(f"\n✓ Full prompt saved to pvsg_yoloworld_prompt.txt")

if __name__ == '__main__':
    main()
