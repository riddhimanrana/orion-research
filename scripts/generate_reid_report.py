#!/usr/bin/env python3
"""
Generate HTML report comparing YOLO detections vs Gemini ground truth
with Re-ID visualization frames
"""

import os
import base64
from pathlib import Path


def generate_html_report(
    visualization_dir: str = "results/reid_visualization",
    output_file: str = "results/reid_report.html",
    gemini_results: dict = None,
    reid_summary: dict = None,
):
    """Generate interactive HTML report with Re-ID visualizations"""
    
    # Get all saved frames
    frame_files = sorted(Path(visualization_dir).glob("frame_*.jpg"))
    
    # Sample frames for display (every 10th + Re-ID highlights)
    sample_frames = frame_files[::10][:30]  # First 30 sampled frames
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Orion Re-ID Tracking Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .summary {
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .summary h2 {
            margin-top: 0;
            color: #667eea;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            color: #666;
            margin-top: 5px;
        }
        .frames-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .frame-card {
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .frame-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }
        .frame-card img {
            width: 100%;
            display: block;
        }
        .frame-info {
            padding: 15px;
        }
        .frame-info h3 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .frame-info p {
            margin: 5px 0;
            color: #666;
            font-size: 0.9em;
        }
        .reid-badge {
            display: inline-block;
            background: #ff6b6b;
            color: white;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .gemini-section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .gemini-section h2 {
            color: #667eea;
            margin-top: 0;
        }
        .gemini-result {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #667eea;
        }
        .comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        .comparison-col {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
        }
        .comparison-col h3 {
            margin-top: 0;
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Orion Re-ID Tracking Report</h1>
        <p>Enhanced 3D Tracker with Appearance-Based Re-Identification</p>
    </div>
"""
    
    # Add summary metrics
    if reid_summary:
        html_content += f"""
    <div class="summary">
        <h2>📊 Tracking Summary</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{reid_summary.get('frames_processed', 0)}</div>
                <div class="metric-label">Frames Processed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{reid_summary.get('unique_tracks', 0)}</div>
                <div class="metric-label">Unique Tracks</div>
            </div>
            <div class="metric">
                <div class="metric-value">{reid_summary.get('reid_events', 0)}</div>
                <div class="metric-label">Re-ID Events</div>
            </div>
            <div class="metric">
                <div class="metric-value">{reid_summary.get('frames_saved', 0)}</div>
                <div class="metric-label">Frames Saved</div>
            </div>
        </div>
        <p><strong>Objects Tracked:</strong> {', '.join(reid_summary.get('object_classes', []))}</p>
    </div>
"""
    
    # Add Gemini comparison
    if gemini_results:
        html_content += """
    <div class="gemini-section">
        <h2>🤖 Gemini Ground Truth Comparison</h2>
"""
        for frame_idx, result in sorted(gemini_results.items()):
            if result:
                html_content += f"""
        <div class="gemini-result">
            <h3>Frame {frame_idx}</h3>
            <p>{result.replace('<', '&lt;').replace('>', '&gt;')}</p>
        </div>
"""
        html_content += """
    </div>
"""
    
    # Add frame visualizations
    html_content += """
    <div class="summary">
        <h2>🎬 Re-ID Visualization Frames</h2>
        <p>Frames showing tracked objects with consistent IDs across time. [RE-ID] badges indicate re-identification events.</p>
    </div>
    
    <div class="frames-grid">
"""
    
    for frame_path in sample_frames:
        frame_name = frame_path.stem
        frame_number = frame_name.split('_')[1]
        
        # Read and encode image
        with open(frame_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()
        
        html_content += f"""
        <div class="frame-card">
            <img src="data:image/jpeg;base64,{img_data}" alt="{frame_name}">
            <div class="frame-info">
                <h3>Frame {frame_number}</h3>
                <p>Track IDs persist across frames with colored bounding boxes</p>
                <p>Trajectories shown as colored paths</p>
            </div>
        </div>
"""
    
    html_content += """
    </div>
    
    <div class="summary">
        <p><strong>Legend:</strong></p>
        <ul>
            <li>Each track ID gets a consistent color throughout the video</li>
            <li><span class="reid-badge">RE-ID</span> badges mark re-identification events (object reappearing after occlusion)</li>
            <li>Colored lines show object trajectories over recent frames</li>
            <li>Thicker bounding boxes highlight recent Re-ID events</li>
        </ul>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"\n✓ HTML report generated: {output_file}")
    print(f"  Open in browser to view Re-ID visualizations")
    return output_file


if __name__ == "__main__":
    # Example usage
    reid_summary = {
        'frames_processed': 300,
        'unique_tracks': 20,
        'reid_events': 755,
        'frames_saved': 260,
        'object_classes': ['keyboard', 'mouse', 'tv', 'person', 'bed', 'chair', 'couch', 'laptop', 'book', 'potted plant'],
    }
    
    # Load Gemini results from previous run
    gemini_results = {
        100: "Monitor (top center), Keyboard (bottom-left), Mouse (bottom-right)",
        500: "Blinds (top left), Wall (right/center)",
        1000: "Notebook (center), Hand (right center)",
        1500: "Door (left), Door (center)",
    }
    
    html_file = generate_html_report(
        reid_summary=reid_summary,
        gemini_results=gemini_results,
    )
    
    print(f"\n🌐 View report: open {html_file}")
