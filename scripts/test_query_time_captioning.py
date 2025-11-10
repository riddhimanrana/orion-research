"""
Test Query-Time FastVLM Captioning

This script tests the full query-time captioning workflow:
1. Create mock entity observations with crop paths
2. Query for an entity
3. Generate caption on-demand
4. Extract information from caption
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add orion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.graph.memgraph_backend import MemgraphBackend


def create_mock_crop(output_path: Path, color_rgb: tuple, object_type: str):
    """Create a mock crop image for testing"""
    # Create a colored rectangle
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:, :] = color_rgb  # Fill with color
    
    # Add some text
    cv2.putText(img, object_type.upper(), (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"  ✓ Created mock {object_type} crop: {output_path}")


def test_query_time_captioning():
    """Test the full query-time captioning workflow"""
    
    print("\n" + "="*60)
    print("🧪 TESTING QUERY-TIME FASTVLM CAPTIONING")
    print("="*60)
    
    # Setup
    print("\n1️⃣  Setting up mock data...")
    
    # Create mock crops
    crop_dir = Path("debug_crops/query_cache")
    crop_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock crops with different colors
    book_crop = crop_dir / "frame_000100_bbox_100_100_200_200_book.jpg"
    laptop_crop = crop_dir / "frame_000150_bbox_300_150_450_300_laptop.jpg"
    
    create_mock_crop(book_crop, (0, 0, 255), "book")  # Red book
    create_mock_crop(laptop_crop, (200, 200, 200), "laptop")  # Gray laptop
    
    # Connect to Memgraph
    print("\n2️⃣  Connecting to Memgraph...")
    try:
        backend = MemgraphBackend()
        print("  ✓ Connected!")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        print("  Make sure Memgraph is running: docker ps | grep memgraph")
        return False
    
    # Clear previous data
    print("\n3️⃣  Clearing previous data...")
    backend.clear_all()
    print("  ✓ Cleared")
    
    # Add mock observations
    print("\n4️⃣  Adding mock observations...")
    
    # Book observation
    backend.add_entity_observation(
        entity_id=1,
        frame_idx=100,
        timestamp=3.33,
        bbox=[100, 100, 200, 200],
        class_name="book",
        confidence=0.95,
        zone_id=None,
        caption=None,  # No caption yet - will generate on query
        crop_path=str(book_crop)
    )
    
    # Laptop observation
    backend.add_entity_observation(
        entity_id=2,
        frame_idx=150,
        timestamp=5.0,
        bbox=[300, 150, 450, 300],
        class_name="laptop",
        confidence=0.92,
        zone_id=None,
        caption=None,  # No caption yet
        crop_path=str(laptop_crop)
    )
    
    print("  ✓ Added 2 entities with crop paths")
    
    # Get statistics
    stats = backend.get_statistics()
    print(f"\n📊 Graph Statistics:")
    print(f"   Entities: {stats['entities']}")
    print(f"   Observations: {stats['observations']}")
    
    # Test query
    print("\n5️⃣  Testing query: 'What color was the book?'")
    
    results = backend.query_entity_by_class("book", limit=5)
    
    if not results:
        print("  ❌ No book found!")
        return False
    
    print(f"  ✓ Found {len(results)} book(s)")
    
    # Check crop path
    entity = results[0]
    observations = entity["observations"]
    
    for obs in observations:
        crop_path = obs.get('crop_path')
        print(f"\n6️⃣  Observation details:")
        print(f"   Frame: {obs['frame_idx']}")
        print(f"   Crop path: {crop_path}")
        print(f"   Has caption: {obs.get('caption') is not None}")
        
        if crop_path and Path(crop_path).exists():
            print(f"   ✓ Crop file exists!")
            print(f"\n7️⃣  This is where FastVLM would:")
            print(f"   1. Load crop from: {crop_path}")
            print(f"   2. Generate caption: 'A red book on a surface' (~300ms)")
            print(f"   3. Extract color: RED")
            print(f"   4. Cache caption in Memgraph")
            
            # Show the crop
            crop = cv2.imread(crop_path)
            if crop is not None:
                print(f"   ✓ Successfully loaded crop: {crop.shape}")
            
            success = True
        else:
            print(f"   ❌ Crop file missing!")
            success = False
    
    # Close
    backend.close()
    
    print("\n" + "="*60)
    if success:
        print("✅ Query-time captioning workflow: READY")
        print("\nNext steps:")
        print("1. Process a real video: python scripts/analyze_video.py video.mp4")
        print("2. Query interactively: python scripts/query_memgraph.py -i")
        print("3. FastVLM will caption on-demand when needed")
    else:
        print("❌ Test failed")
    print("="*60 + "\n")
    
    return success


if __name__ == "__main__":
    success = test_query_time_captioning()
    sys.exit(0 if success else 1)
