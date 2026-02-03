#!/usr/bin/env python3
"""Debug PVSG data format to understand root cause of 0% recall."""
import json

# Load PVSG GT
with open('datasets/PVSG/pvsg.json', 'r') as f:
    pvsg = json.load(f)

# Check structure
print('Keys in pvsg:', list(pvsg.keys()))
print()

# Check objects vocabulary
if 'objects' in pvsg:
    print('Objects structure:', type(pvsg['objects']))
    if isinstance(pvsg['objects'], dict):
        print('Object keys:', list(pvsg['objects'].keys()))
        if 'thing' in pvsg['objects']:
            print('Thing classes (first 20):', pvsg['objects']['thing'][:20])
        if 'stuff' in pvsg['objects']:
            print('Stuff classes:', pvsg['objects']['stuff'])
    elif isinstance(pvsg['objects'], list):
        print('Objects list (first 5):', pvsg['objects'][:5])

# Check predicates
if 'predicates' in pvsg:
    print()
    print('Predicates (first 20):', pvsg['predicates'][:20])
    print('Total predicates:', len(pvsg['predicates']))

# Check a sample video - 1002_4060588783 (the one with 0%)
if 'data' in pvsg:
    for video in pvsg['data']:
        if video.get('video_id') == '1002_4060588783':
            print()
            print('='*60)
            print('VIDEO: 1002_4060588783')
            print('='*60)
            print('Keys:', list(video.keys()))
            print()
            print('Objects:')
            for obj in video.get('objects', []):
                print(f"  {obj}")
            print()
            print('Relations:')
            for rel in video.get('relations', []):
                print(f"  {rel}")
            break
