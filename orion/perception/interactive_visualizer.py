"""
Interactive OpenCV Visualizer for Orion Spatial Mapping

Provides real-time interactive controls for exploring tracking + spatial data:
- Mouse click to inspect entity details
- Keyboard shortcuts for toggling overlays
- Real-time visualization controls
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class EntityInfo:
    """Information about a tracked entity"""
    entity_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    depth_mm: Optional[float] = None
    zone_id: Optional[int] = None
    world_pos: Optional[Tuple[float, float, float]] = None


class InteractiveVisualizer:
    """
    Interactive OpenCV-based visualizer with mouse + keyboard controls
    
    Features:
    - Mouse click: Show entity details panel
    - Keyboard shortcuts:
        z: Toggle zone visualization
        t: Toggle trajectories
        d: Toggle depth heatmap
        s: Toggle spatial map
        m: Toggle SLAM mini-map
        h: Show/hide help overlay
        space: Pause/resume playback
        q: Quit
    """
    
    def __init__(self, window_name: str = "Orion Interactive Viewer"):
        self.window_name = window_name
        
        # Toggle states
        self.show_zones = True
        self.show_trajectories = True
        self.show_depth = True
        self.show_spatial_map = True
        self.show_slam_minimap = False
        self.show_help = False
        self.paused = False
        
        # Selection state
        self.selected_entity: Optional[int] = None
        self.mouse_pos: Optional[Tuple[int, int]] = None
        self.entities: Dict[int, EntityInfo] = {}
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Help text
        self.help_text = [
            "=== Interactive Controls ===",
            "z: Toggle zones",
            "t: Toggle trajectories",
            "d: Toggle depth heatmap",
            "s: Toggle spatial map",
            "m: Toggle SLAM mini-map",
            "h: Show/hide this help",
            "Space: Pause/resume",
            "q: Quit",
            "Click entity: Show details"
        ]
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicked on an entity
            clicked_entity = self._find_entity_at_point(x, y)
            if clicked_entity is not None:
                self.selected_entity = clicked_entity
            else:
                self.selected_entity = None
    
    def _find_entity_at_point(self, x: int, y: int) -> Optional[int]:
        """Find entity whose bbox contains the click point"""
        for entity_id, info in self.entities.items():
            x1, y1, x2, y2 = info.bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                return entity_id
        return None
    
    def update_entities(self, entities: Dict[int, EntityInfo]):
        """Update current frame's entity information"""
        self.entities = entities
    
    def handle_keyboard(self) -> bool:
        """
        Process keyboard input
        
        Returns:
            True to continue, False to quit
        """
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return False
        elif key == ord('z'):
            self.show_zones = not self.show_zones
        elif key == ord('t'):
            self.show_trajectories = not self.show_trajectories
        elif key == ord('d'):
            self.show_depth = not self.show_depth
        elif key == ord('s'):
            self.show_spatial_map = not self.show_spatial_map
        elif key == ord('m'):
            self.show_slam_minimap = not self.show_slam_minimap
        elif key == ord('h'):
            self.show_help = not self.show_help
        elif key == ord(' '):
            self.paused = not self.paused
        
        return True
    
    def draw_overlays(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw interactive overlays on frame
        
        Args:
            frame: Input frame (already has basic visualization from TrackingVisualizer)
        
        Returns:
            Frame with interactive overlays
        """
        frame = frame.copy()
        
        # Draw selected entity details panel
        if self.selected_entity is not None and self.selected_entity in self.entities:
            frame = self._draw_entity_panel(frame, self.entities[self.selected_entity])
        
        # Draw mouse hover highlight
        if self.mouse_pos is not None:
            hovered_entity = self._find_entity_at_point(*self.mouse_pos)
            if hovered_entity is not None and hovered_entity in self.entities:
                frame = self._draw_hover_highlight(frame, self.entities[hovered_entity])
        
        # Draw help overlay
        if self.show_help:
            frame = self._draw_help_overlay(frame)
        
        # Draw status bar
        frame = self._draw_status_bar(frame)
        
        return frame
    
    def _draw_entity_panel(self, frame: np.ndarray, entity: EntityInfo) -> np.ndarray:
        """Draw detailed information panel for selected entity"""
        h, w = frame.shape[:2]
        
        # Panel dimensions
        panel_width = 300
        panel_height = 200
        panel_x = w - panel_width - 20
        panel_y = 20
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (40, 40, 40), -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 255, 255), 2)
        
        # Title
        cv2.putText(frame, f"Entity #{entity.entity_id}", 
                   (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Details
        y_offset = panel_y + 55
        line_height = 25
        
        details = [
            f"Class: {entity.class_name}",
            f"Confidence: {entity.confidence:.2%}",
        ]
        
        if entity.depth_mm is not None:
            details.append(f"Depth: {entity.depth_mm/1000:.2f}m")
        
        if entity.zone_id is not None:
            details.append(f"Zone: {entity.zone_id}")
        
        if entity.world_pos is not None:
            x, y, z = entity.world_pos
            details.append(f"World Pos:")
            details.append(f"  X: {x:.2f}mm")
            details.append(f"  Y: {y:.2f}mm")
            details.append(f"  Z: {z:.2f}mm")
        
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = entity.bbox
        details.append(f"BBox: ({bbox_x1},{bbox_y1})")
        details.append(f"      ({bbox_x2},{bbox_y2})")
        
        for line in details:
            cv2.putText(frame, line, (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += line_height
        
        # Highlight bbox on frame
        x1, y1, x2, y2 = entity.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        
        return frame
    
    def _draw_hover_highlight(self, frame: np.ndarray, entity: EntityInfo) -> np.ndarray:
        """Draw highlight when hovering over entity"""
        x1, y1, x2, y2 = entity.bbox
        
        # Draw dashed rectangle
        dash_length = 10
        for i in range(x1, x2, dash_length * 2):
            cv2.line(frame, (i, y1), (min(i + dash_length, x2), y1), (255, 255, 0), 2)
            cv2.line(frame, (i, y2), (min(i + dash_length, x2), y2), (255, 255, 0), 2)
        
        for i in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x1, i), (x1, min(i + dash_length, y2)), (255, 255, 0), 2)
            cv2.line(frame, (x2, i), (x2, min(i + dash_length, y2)), (255, 255, 0), 2)
        
        # Show entity ID near cursor
        if self.mouse_pos:
            mx, my = self.mouse_pos
            cv2.putText(frame, f"#{entity.entity_id}: {entity.class_name}",
                       (mx + 10, my - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return frame
    
    def _draw_help_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw help text overlay"""
        h, w = frame.shape[:2]
        
        # Calculate panel size
        panel_width = 280
        panel_height = len(self.help_text) * 25 + 40
        panel_x = (w - panel_width) // 2
        panel_y = (h - panel_height) // 2
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (20, 20, 20), -1)
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (255, 255, 255), 2)
        
        # Help text
        y_offset = panel_y + 30
        for line in self.help_text:
            cv2.putText(frame, line, (panel_x + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_offset += 25
        
        return frame
    
    def _draw_status_bar(self, frame: np.ndarray) -> np.ndarray:
        """Draw status bar showing toggle states"""
        h, w = frame.shape[:2]
        
        # Status bar background
        bar_height = 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (40, 40, 40), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Status indicators
        x_offset = 10
        status_items = [
            (f"Zones: {'ON' if self.show_zones else 'OFF'}", self.show_zones),
            (f"Trajectories: {'ON' if self.show_trajectories else 'OFF'}", self.show_trajectories),
            (f"Depth: {'ON' if self.show_depth else 'OFF'}", self.show_depth),
            (f"Spatial: {'ON' if self.show_spatial_map else 'OFF'}", self.show_spatial_map),
            (f"SLAM: {'ON' if self.show_slam_minimap else 'OFF'}", self.show_slam_minimap),
        ]
        
        for text, is_on in status_items:
            color = (0, 255, 0) if is_on else (100, 100, 100)
            cv2.putText(frame, text, (x_offset, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            x_offset += 150
        
        # Pause indicator
        if self.paused:
            pause_text = "PAUSED"
            text_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = w - text_size[0] - 10
            cv2.putText(frame, pause_text, (text_x, 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Help reminder
        cv2.putText(frame, "Press 'h' for help", (w - 150, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def show(self, frame: np.ndarray) -> bool:
        """
        Display frame in window and handle input
        
        Args:
            frame: Frame to display (with overlays)
        
        Returns:
            True to continue, False to quit
        """
        cv2.imshow(self.window_name, frame)
        return self.handle_keyboard()
    
    def close(self):
        """Clean up and close window"""
        cv2.destroyWindow(self.window_name)
    
    def get_toggle_states(self) -> Dict[str, bool]:
        """Get current toggle states for conditional rendering"""
        return {
            'zones': self.show_zones,
            'trajectories': self.show_trajectories,
            'depth': self.show_depth,
            'spatial_map': self.show_spatial_map,
            'slam_minimap': self.show_slam_minimap,
            'paused': self.paused,
        }


def run_interactive_demo(video_path: str):
    """
    Demo function showing interactive visualizer usage
    
    Args:
        video_path: Path to video file
    """
    visualizer = InteractiveVisualizer(window_name="Orion Interactive Demo")
    cap = cv2.VideoCapture(video_path)
    
    frame_idx = 0
    
    print("Interactive Visualizer Demo")
    print("===========================")
    print("Controls:")
    print("  z: Toggle zones")
    print("  t: Toggle trajectories")
    print("  d: Toggle depth")
    print("  s: Toggle spatial map")
    print("  m: Toggle SLAM mini-map")
    print("  h: Show/hide help")
    print("  Space: Pause/resume")
    print("  q: Quit")
    print("\nClick on entities to inspect details")
    
    frame = None
    try:
        while cap.isOpened():
            if not visualizer.paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Simulate entity data (in real usage, get from tracking system)
                entities = {
                    1: EntityInfo(
                        entity_id=1,
                        class_name="person",
                        bbox=(100, 100, 200, 300),
                        confidence=0.95,
                        depth_mm=2500.0,
                        zone_id=1,
                        world_pos=(1200.0, 800.0, 2500.0)
                    )
                }
                
                visualizer.update_entities(entities)
                frame_idx += 1
            
            # Draw overlays (only if frame exists)
            if frame is not None:
                frame_with_overlays = visualizer.draw_overlays(frame)
                
                # Show and check for quit
                if not visualizer.show(frame_with_overlays):
                    break
        
    finally:
        cap.release()
        visualizer.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        run_interactive_demo(sys.argv[1])
    else:
        print("Usage: python interactive_visualizer.py <video_path>")
        print("Example: python interactive_visualizer.py data/examples/video.mp4")
