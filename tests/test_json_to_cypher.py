"""
Test JSON to Cypher translation functionality.

This validates that we're using structured JSON from LLM instead of directly
generating Cypher, as per mentor feedback.
"""

import pytest
from orion.semantic_uplift import EventComposer


class TestJSONToCypher:
    """Test suite for JSON to Cypher translation."""

    @pytest.fixture
    def composer(self):
        """Create event composer for testing."""
        return EventComposer()

    def test_state_change_event(self, composer):
        """Test converting state change event from JSON to Cypher."""
        event_data = {
            "events": [
                {
                    "type": "state_change",
                    "entity_id": "person_1",
                    "entity_label": "person",
                    "description": "person sitting down",
                    "timestamp": 1.5,
                    "location": "living_room",
                    "attributes": {
                        "old_state": "standing",
                        "new_state": "sitting",
                        "confidence": 0.95
                    }
                }
            ]
        }
        
        queries = composer.json_to_cypher(event_data)
        
        # Should generate 2 queries: entity+state_change, location link
        assert len(queries) == 2
        
        # Check entity and state change creation
        assert "MERGE (e:Entity {id: 'person_1'})" in queries[0]
        assert "MERGE (sc:StateChange" in queries[0]
        assert "old_state = 'standing'" in queries[0]
        assert "new_state = 'sitting'" in queries[0]
        assert "confidence = 0.95" in queries[0]
        
        # Check location link
        assert "MERGE (l:Location {id: 'living_room'})" in queries[1]
        assert "OCCURRED_AT" in queries[1]

    def test_causal_link_event(self, composer):
        """Test converting causal link event from JSON to Cypher."""
        event_data = {
            "events": [
                {
                    "type": "causal_link",
                    "entity_id": "event_1",
                    "entity_label": "causal_event",
                    "description": "person picked up cup",
                    "timestamp": 2.0,
                    "location": None,
                    "attributes": {
                        "cause_entity": "person_1",
                        "effect_entity": "cup_1",
                        "confidence": 0.88
                    }
                }
            ]
        }
        
        queries = composer.json_to_cypher(event_data)
        
        # Should generate 1 query for causal link
        assert len(queries) == 1
        
        # Check causal relationship
        assert "MERGE (cause:Entity {id: 'person_1'})" in queries[0]
        assert "MERGE (effect:Entity {id: 'cup_1'})" in queries[0]
        assert "MERGE (cl:CausalLink" in queries[0]
        assert "CAUSES" in queries[0]
        assert "AFFECTS" in queries[0]
        assert "confidence = 0.88" in queries[0]

    def test_entity_movement_event(self, composer):
        """Test converting entity movement event from JSON to Cypher."""
        event_data = {
            "events": [
                {
                    "type": "entity_movement",
                    "entity_id": "person_1",
                    "entity_label": "person",
                    "description": "moved to kitchen",
                    "timestamp": 3.5,
                    "location": "kitchen",
                    "attributes": {}
                }
            ]
        }
        
        queries = composer.json_to_cypher(event_data)
        
        # Should generate 1 query for movement
        assert len(queries) == 1
        
        # Check movement relationship
        assert "MERGE (e:Entity {id: 'person_1'})" in queries[0]
        assert "MERGE (l:Location {id: 'kitchen'})" in queries[0]
        assert "MOVED_TO" in queries[0]
        assert "timestamp: 3.5" in queries[0]

    def test_multiple_events(self, composer):
        """Test converting multiple events in one batch."""
        event_data = {
            "events": [
                {
                    "type": "state_change",
                    "entity_id": "laptop_1",
                    "entity_label": "laptop",
                    "description": "laptop opened",
                    "timestamp": 1.0,
                    "location": "desk",
                    "attributes": {
                        "old_state": "closed",
                        "new_state": "open",
                        "confidence": 0.92
                    }
                },
                {
                    "type": "entity_movement",
                    "entity_id": "person_1",
                    "entity_label": "person",
                    "description": "walked to desk",
                    "timestamp": 0.5,
                    "location": "desk",
                    "attributes": {}
                }
            ]
        }
        
        queries = composer.json_to_cypher(event_data)
        
        # Should generate 3 queries: state_change (2), movement (1)
        assert len(queries) == 3

    def test_sql_injection_protection(self, composer):
        """Test that single quotes are properly escaped."""
        event_data = {
            "events": [
                {
                    "type": "state_change",
                    "entity_id": "test'entity",  # Malicious input
                    "entity_label": "person's laptop",  # Malicious input
                    "description": "it's working",  # Malicious input
                    "timestamp": 1.0,
                    "location": None,
                    "attributes": {
                        "old_state": "state'1",
                        "new_state": "state'2",
                        "confidence": 0.9
                    }
                }
            ]
        }
        
        queries = composer.json_to_cypher(event_data)
        
        # All single quotes should be escaped
        assert len(queries) == 1
        assert "test\\'entity" in queries[0]
        assert "person\\'s laptop" in queries[0]
        assert "it\\'s working" in queries[0]
        assert "state\\'1" in queries[0]
        assert "state\\'2" in queries[0]

    def test_parse_json_from_llm_output(self, composer):
        """Test parsing JSON from LLM output with markdown."""
        llm_output = """```json
{
  "events": [
    {
      "type": "state_change",
      "entity_id": "cup_1",
      "entity_label": "cup",
      "description": "cup moved",
      "timestamp": 2.5,
      "location": "table",
      "attributes": {
        "old_state": "on counter",
        "new_state": "on table",
        "confidence": 0.85
      }
    }
  ]
}
```"""
        
        queries = composer.parse_cypher_queries(llm_output)
        
        # Should extract JSON and convert to Cypher
        assert len(queries) == 2  # state_change + location
        assert "MERGE (e:Entity {id: 'cup_1'})" in queries[0]

    def test_backwards_compatibility_with_direct_cypher(self, composer):
        """Test that old Cypher output still works (backwards compatibility)."""
        old_style_output = """
MERGE (e:Entity {id: 'test_1'});
CREATE (sc:StateChange {timestamp: 1.0});
"""
        
        queries = composer.parse_cypher_queries(old_style_output)
        
        # Should parse old-style Cypher
        assert len(queries) == 2
        assert "MERGE (e:Entity {id: 'test_1'});" in queries[0]
        assert "CREATE (sc:StateChange {timestamp: 1.0});" in queries[1]

    def test_empty_events(self, composer):
        """Test handling empty events list."""
        event_data = {"events": []}
        
        queries = composer.json_to_cypher(event_data)
        
        assert queries == []

    def test_invalid_event_type(self, composer):
        """Test handling unknown event types."""
        event_data = {
            "events": [
                {
                    "type": "unknown_type",
                    "entity_id": "test_1",
                    "entity_label": "test",
                    "description": "test event",
                    "timestamp": 1.0,
                    "location": None,
                    "attributes": {}
                }
            ]
        }
        
        queries = composer.json_to_cypher(event_data)
        
        # Unknown type should be ignored
        assert queries == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
