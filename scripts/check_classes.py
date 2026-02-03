#!/usr/bin/env python3
from orion.sga.ag_dataset_v2 import AG_OBJECT_CLASSES

print("AG Object Classes (36):")
print(AG_OBJECT_CLASSES)

# YOLO COCO classes that overlap with AG
yolo_to_ag = {
    'person': 'person',
    'chair': 'chair', 
    'couch': 'sofa',
    'bed': 'bed',
    'dining table': 'table',
    'laptop': 'laptop',
    'tv': 'television',
    'cell phone': 'phone/camera',
    'book': 'book',
    'cup': 'cup/glass/bottle',
    'bottle': 'cup/glass/bottle',
    'bowl': 'dish',
    'refrigerator': 'refrigerator',
    'backpack': 'bag',
    'handbag': 'bag',
    'suitcase': 'bag',
    'toilet': 'toilet',
}

ag_covered = set(yolo_to_ag.values())
ag_missing = [c for c in AG_OBJECT_CLASSES if c not in ag_covered]

print(f"\nYOLO covers {len(ag_covered)}/{len(AG_OBJECT_CLASSES)} AG classes:")
print(f"  Covered: {sorted(ag_covered)}")
print(f"  Missing: {ag_missing}")
