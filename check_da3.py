
import sys
from pathlib import Path

def test_da3_import():
    workspace_root = Path(".").resolve()
    da3_path = workspace_root / "Depth-Anything-3" / "src"
    if da3_path.exists():
        sys.path.insert(0, str(da3_path))
        print(f"Added {da3_path} to sys.path")
    
    try:
        from depth_anything_3.api import DepthAnything3
        print("Success: depth_anything_3 imported.")
    except ImportError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_da3_import()
