import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def analyze_tracks(tracks_file):
    """
    Analyzes the tracking data from a tracks.jsonl file.

    Args:
        tracks_file (str): Path to the tracks.jsonl file.
    """
    tracks = defaultdict(list)
    class_names = defaultdict(list)
    
    with open(tracks_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            track_id = data['track_id']
            frame_id = data['frame_id']
            class_name = data['class_name']
            tracks[track_id].append(frame_id)
            class_names[track_id].append(class_name)

    print(f"Found {len(tracks)} unique tracks.")

    track_lengths = [len(frames) for frames in tracks.values()]
    
    print("\nTrack Lengths:")
    for track_id, frames in tracks.items():
        print(f"  Track {track_id}: {len(frames)} frames")

    print("\nClass Names per Track:")
    for track_id, names in class_names.items():
        unique_names = set(names)
        print(f"  Track {track_id}: {unique_names}")

    # Plot histogram of track lengths
    plt.figure(figsize=(10, 6))
    plt.hist(track_lengths, bins=np.arange(min(track_lengths), max(track_lengths) + 1, 1), edgecolor='black')
    plt.title('Histogram of Track Lengths')
    plt.xlabel('Track Length (number of frames)')
    plt.ylabel('Number of Tracks')
    plt.xticks(np.arange(min(track_lengths), max(track_lengths) + 1, 1))
    plt.grid(axis='y', alpha=0.75)
    plt.show()


if __name__ == '__main__':
    analyze_tracks('results/tracks.jsonl')
