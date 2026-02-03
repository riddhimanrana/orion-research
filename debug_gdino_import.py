import sys
from pathlib import Path

# Add current dir to sys.path
sys.path.append(str(Path.cwd()))

try:
    from orion.perception.detectors.grounding_dino import GroundingDINOWrapper
    print("SUCCESS: GroundingDINOWrapper imported successfully.")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Exception during import: {e}")
    import traceback
    traceback.print_exc()
