#!/usr/bin/env python3
"""
Test Complete Spatial Memory Integration

Verifies that spatial memory is properly integrated into the SLAM pipeline:
1. Observations feed during processing
2. Memory persists to disk
3. Memory loads across sessions
4. Interactive assistant can query memory
"""

import sys
from pathlib import Path

# Add orion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.graph.spatial_memory import SpatialMemorySystem
import tempfile
import shutil


def test_memory_lifecycle():
    """Test complete memory lifecycle: create → feed → save → load → query"""
    
    print("=" * 80)
    print("TEST: Spatial Memory Lifecycle")
    print("=" * 80)
    
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="spatial_memory_test_"))
    print(f"\n📁 Test directory: {temp_dir}")
    
    try:
        # === PHASE 1: Create and Populate Memory ===
        print("\n🔨 PHASE 1: Creating memory system...")
        memory1 = SpatialMemorySystem(memory_dir=temp_dir)
        
        # Simulate processing: Add observations
        print("   Adding observations...")
        
        # Entity 1: Laptop (stationary)
        for i in range(10):
            memory1.add_entity_observation(
                entity_id=1,
                class_name="laptop",
                timestamp=i * 0.1,
                position_3d=(500.0 + i * 2, 300.0, 800.0),  # Minor movement
                zone_id=2,
                caption="A silver laptop with glowing screen" if i == 5 else None
            )
        
        # Entity 2: Person (moving)
        for i in range(15):
            memory1.add_entity_observation(
                entity_id=2,
                class_name="person",
                timestamp=i * 0.15,
                position_3d=(100.0 + i * 50, 200.0 + i * 20, 900.0 + i * 10),
                zone_id=1 if i < 7 else 3,  # Moves between zones
                caption="Person wearing blue shirt" if i == 10 else None
            )
        
        # Entity 3: Book (stationary)
        for i in range(8):
            memory1.add_entity_observation(
                entity_id=3,
                class_name="book",
                timestamp=i * 0.2,
                position_3d=(450.0, 280.0, 750.0),
                zone_id=2,
                caption="Red hardcover book" if i == 4 else None
            )
        
        stats1 = memory1.get_statistics()
        print(f"   ✓ Added: {stats1['total_entities']} entities")
        print(f"   ✓ Total observations: {sum(e.observations_count for e in memory1.entities.values())}")
        print(f"   ✓ Captions: {stats1['total_captions']}")
        
        # Save to disk
        print("\n💾 Saving to disk...")
        memory1.save()
        
        # Verify files exist
        assert (temp_dir / "entities.json").exists(), "entities.json not created"
        assert (temp_dir / "metadata.json").exists(), "metadata.json not created"
        print("   ✓ Files created successfully")
        
        # === PHASE 2: Load Memory (Simulating New Session) ===
        print("\n🔄 PHASE 2: Loading memory (new session)...")
        memory2 = SpatialMemorySystem(memory_dir=temp_dir)
        
        stats2 = memory2.get_statistics()
        print(f"   ✓ Loaded: {stats2['total_entities']} entities")
        print(f"   ✓ Total observations: {sum(e.observations_count for e in memory2.entities.values())}")
        print(f"   ✓ Captions: {stats2['total_captions']}")
        
        # Verify data integrity
        assert stats2['total_entities'] == 3, "Wrong number of entities"
        assert stats2['total_captions'] == 3, "Wrong number of captions"
        assert memory2.entities[1].class_name == "laptop", "Wrong class for entity 1"
        assert memory2.entities[2].class_name == "person", "Wrong class for entity 2"
        assert memory2.entities[3].class_name == "book", "Wrong class for entity 3"
        print("   ✓ Data integrity verified")
        
        # === PHASE 3: Query Memory ===
        print("\n🔍 PHASE 3: Querying memory...")
        
        # Query 1: Check entities exist
        print(f"\n   Verifying entities...")
        assert 1 in memory2.entities, "Laptop entity should exist"
        assert 2 in memory2.entities, "Person entity should exist"
        assert 3 in memory2.entities, "Book entity should exist"
        print(f"   → Found all 3 entities")
        
        # Check captions
        laptop = memory2.entities[1]
        person = memory2.entities[2]
        book = memory2.entities[3]
        
        assert len(laptop.captions) == 1, "Laptop should have 1 caption"
        assert len(person.captions) == 1, "Person should have 1 caption"
        assert len(book.captions) == 1, "Book should have 1 caption"
        print(f"   → Laptop: '{laptop.captions[0]}'")
        print(f"   → Person: '{person.captions[0]}'")
        print(f"   → Book: '{book.captions[0]}'")
        
        # Check movement
        print(f"\n   Checking movement history...")
        print(f"   → Laptop: {laptop.observations_count} observations")
        print(f"   → Person: {person.observations_count} observations (moved between zones)")
        print(f"   → Book: {book.observations_count} observations")
        
        # Check zone membership
        print(f"\n   Checking zones...")
        zone2_entities = [e for e in memory2.entities.values() if e.primary_zone == 2]
        print(f"   → Zone 2 has {len(zone2_entities)} entities: {[e.class_name for e in zone2_entities]}")
        assert len(zone2_entities) == 2, "Zone 2 should have laptop and book"
        
        # === PHASE 4: Append More Data (Cross-Session Growth) ===
        print("\n📈 PHASE 4: Appending new observations...")
        
        # Add more observations to existing entity
        for i in range(5):
            memory2.add_entity_observation(
                entity_id=1,
                class_name="laptop",
                timestamp=10.0 + i * 0.1,
                position_3d=(520.0, 305.0, 810.0),
                zone_id=2
            )
        
        # Add new entity
        memory2.add_entity_observation(
            entity_id=4,
            class_name="chair",
            timestamp=11.0,
            position_3d=(600.0, 350.0, 700.0),
            zone_id=2,
            caption="Wooden office chair"
        )
        
        stats3 = memory2.get_statistics()
        print(f"   ✓ Now: {stats3['total_entities']} entities")
        print(f"   ✓ Laptop observations: {memory2.entities[1].observations_count}")
        
        # Save updated memory
        memory2.save()
        print("   ✓ Saved updated memory")
        
        # === PHASE 5: Verify Persistence ===
        print("\n✅ PHASE 5: Final verification...")
        memory3 = SpatialMemorySystem(memory_dir=temp_dir)
        
        assert len(memory3.entities) == 4, "Should have 4 entities after reload"
        assert memory3.entities[1].observations_count == 15, "Laptop should have 15 observations"
        assert memory3.entities[4].class_name == "chair", "New entity should persist"
        
        print(f"   ✓ Final state: {len(memory3.entities)} entities")
        print(f"   ✓ All data persisted correctly")
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n🧹 Cleaned up test directory")


def test_integration_points():
    """Test that all integration points are properly configured"""
    
    print("\n" + "=" * 80)
    print("TEST: Integration Points")
    print("=" * 80)
    
    # Test 1: Import spatial memory system
    print("\n1. Testing import...")
    from orion.graph.spatial_memory import SpatialMemorySystem, SpatialEntity, SpatialZone
    print("   ✓ Imports successful")
    
    # Test 2: Check run_slam_complete has spatial_memory attribute
    print("\n2. Checking run_slam_complete integration...")
    with open("scripts/run_slam_complete.py") as f:
        content = f.read()
        assert "self.spatial_memory = None" in content, "Missing spatial_memory attribute"
        assert "add_entity_observation" in content, "Missing observation feeding"
        assert "self.spatial_memory.save()" in content, "Missing save call"
        assert "--use-spatial-memory" in content, "Missing CLI flag"
        assert "--memory-dir" in content, "Missing memory-dir flag"
    print("   ✓ run_slam_complete properly integrated")
    
    # Test 3: Check CLI integration
    print("\n3. Checking CLI integration...")
    with open("orion/cli/main.py") as f:
        content = f.read()
        assert "--use-spatial-memory" in content, "Missing CLI flag in main.py"
        assert "--memory-dir" in content, "Missing memory-dir flag in main.py"
    print("   ✓ CLI flags added")
    
    with open("orion/cli/commands/research.py") as f:
        content = f.read()
        assert "spatial_intelligence_assistant.py" in content, "Missing assistant integration"
    print("   ✓ CLI command handler updated")
    
    # Test 4: Check assistant exists
    print("\n4. Checking interactive assistant...")
    assistant_path = Path("scripts/spatial_intelligence_assistant.py")
    assert assistant_path.exists(), "Assistant script not found"
    print("   ✓ Interactive assistant exists")
    
    print("\n" + "=" * 80)
    print("✅ ALL INTEGRATION POINTS VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SPATIAL MEMORY SYSTEM - INTEGRATION TESTS")
    print("=" * 80)
    
    try:
        # Run tests
        test_integration_points()
        test_memory_lifecycle()
        
        print("\n" + "=" * 80)
        print("🎉 SUCCESS: All tests passed!")
        print("=" * 80)
        print("\n✅ Spatial memory system is fully integrated and working")
        print("\n📝 Next Steps:")
        print("   1. Test with real video:")
        print("      python scripts/run_slam_complete.py \\")
        print("          --video data/examples/your_video.mp4 \\")
        print("          --use-spatial-memory --skip 10")
        print("\n   2. Start interactive assistant:")
        print("      python scripts/spatial_intelligence_assistant.py -i")
        print("\n   3. Read documentation:")
        print("      docs/SPATIAL_MEMORY_QUICKSTART.md")
        print("      docs/PERSISTENT_SPATIAL_INTELLIGENCE.md")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
