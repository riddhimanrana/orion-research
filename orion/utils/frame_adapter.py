"""
Frame folder adapter for TAO-Amodal dataset.

TAO videos come as folders of frames (frame0001.jpg, frame0002.jpg, ...)
This adapter converts them to a format compatible with Orion's pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
import tempfile


class FrameFolderAdapter:
    """
    Adapter to read videos from frame folders (TAO-Amodal format).
    Makes frame folders behave like video files for the pipeline.
    """
    
    def __init__(self, frame_folder: Union[str, Path]):
        """
        Initialize adapter with a folder of frames.
        
        Args:
            frame_folder: Path to folder containing frames (frame0001.jpg, etc.)
        """
        self.frame_folder = Path(frame_folder)
        
        if not self.frame_folder.exists():
            raise ValueError(f"Frame folder does not exist: {frame_folder}")
        
        # Load all frame paths
        self.frame_paths = sorted(
            self.frame_folder.glob('frame*.jpg'),
            key=lambda x: int(x.stem.replace('frame', ''))
        )
        
        if len(self.frame_paths) == 0:
            raise ValueError(f"No frames found in {frame_folder}")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(self.frame_paths[0]))
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {self.frame_paths[0]}")
        
        self.height, self.width = first_frame.shape[:2]
        self.fps = 30  # Default FPS for TAO videos
        self.total_frames = len(self.frame_paths)
        self.current_frame = 0
    
    def read(self) -> tuple:
        """
        Read next frame.
        
        Returns:
            (success, frame) tuple compatible with cv2.VideoCapture
        """
        if self.current_frame >= self.total_frames:
            return False, None
        
        frame_path = self.frame_paths[self.current_frame]
        frame = cv2.imread(str(frame_path))
        
        if frame is None:
            return False, None
        
        self.current_frame += 1
        return True, frame
    
    def get(self, prop_id: int):
        """
        Get video property (compatible with cv2.VideoCapture).
        
        Args:
            prop_id: Property ID (cv2.CAP_PROP_*)
        """
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        elif prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.total_frames
        elif prop_id == cv2.CAP_PROP_POS_FRAMES:
            return self.current_frame
        else:
            return 0.0
    
    def set(self, prop_id: int, value: float) -> bool:
        """
        Set video property (compatible with cv2.VideoCapture).
        
        Args:
            prop_id: Property ID (cv2.CAP_PROP_*)
            value: Value to set
        """
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            frame_num = int(value)
            if 0 <= frame_num < self.total_frames:
                self.current_frame = frame_num
                return True
        return False
    
    def isOpened(self) -> bool:
        """Check if adapter is ready."""
        return len(self.frame_paths) > 0
    
    def release(self):
        """Release resources (no-op for frame folders)."""
        pass
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.release()


def create_video_capture(video_path: Union[str, Path]):
    """
    Create a video capture object that works with both .mp4 files and frame folders.
    
    Args:
        video_path: Path to video file or frame folder
    
    Returns:
        cv2.VideoCapture or FrameFolderAdapter instance
    """
    path = Path(video_path)
    
    # Check if it's a directory (frame folder)
    if path.is_dir():
        return FrameFolderAdapter(path)
    
    # Check if it's a video file
    elif path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        return cv2.VideoCapture(str(path))
    
    # Try as frame folder anyway
    elif path.exists():
        return FrameFolderAdapter(path)
    
    else:
        raise ValueError(f"Invalid video path: {video_path}")


def convert_frames_to_video(
    frame_folder: Union[str, Path],
    output_video: Union[str, Path],
    fps: int = 30,
    codec: str = 'mp4v'
) -> bool:
    """
    Convert a folder of frames to a video file.
    Useful if you need .mp4 files instead of frame folders.
    
    Args:
        frame_folder: Path to folder containing frames
        output_video: Path to output video file
        fps: Frames per second
        codec: Video codec (e.g., 'mp4v', 'h264')
    
    Returns:
        True if successful
    """
    adapter = FrameFolderAdapter(frame_folder)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (adapter.width, adapter.height)
    )
    
    if not out.isOpened():
        print(f"Failed to create video writer for {output_video}")
        return False
    
    # Write all frames
    print(f"Converting {adapter.total_frames} frames to video...")
    for i in range(adapter.total_frames):
        success, frame = adapter.read()
        if not success:
            print(f"Failed to read frame {i}")
            break
        out.write(frame)
    
    out.release()
    print(f"Saved video to {output_video}")
    return True


def batch_convert_frames_to_videos(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    fps: int = 30,
    codec: str = 'mp4v'
):
    """
    Convert all frame folders in a directory to video files.
    
    Args:
        input_dir: Directory containing frame folders
        output_dir: Directory to save video files
        fps: Frames per second
        codec: Video codec
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all folders with frames
    frame_folders = [
        d for d in input_path.rglob('*')
        if d.is_dir() and len(list(d.glob('frame*.jpg'))) > 0
    ]
    
    print(f"Found {len(frame_folders)} frame folders to convert")
    
    for folder in frame_folders:
        # Preserve directory structure
        rel_path = folder.relative_to(input_path)
        output_video = output_path / rel_path.with_suffix('.mp4')
        output_video.parent.mkdir(parents=True, exist_ok=True)
        
        if output_video.exists():
            print(f"Skipping {folder.name} (already exists)")
            continue
        
        print(f"\nConverting {folder.name}...")
        convert_frames_to_video(folder, output_video, fps, codec)
    
    print(f"\n=== Conversion Complete ===")
    print(f"Saved {len(frame_folders)} videos to {output_dir}")
