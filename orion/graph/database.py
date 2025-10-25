"""
Neo4j Database Management Utilities

This module provides utilities for managing the Neo4j database,
including clearing all data before new pipeline runs.

All Neo4j credentials should be managed via ConfigManager, which reads
from environment variables for security.
"""

import logging
from typing import Any, Optional

from neo4j import GraphDatabase

from orion.settings import ConfigManager

logger = logging.getLogger(__name__)


class Neo4jManager:
    """Manager for Neo4j database operations"""

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize Neo4j manager

        Args:
            uri: Neo4j connection URI (e.g., neo4j://127.0.0.1:7687). 
                 If None, uses ConfigManager.
            user: Database username. If None, uses ConfigManager.
            password: Database password. If None, uses ConfigManager.
        """
        # Use centralized config if credentials not provided
        if uri is None or user is None or password is None:
            config = ConfigManager.get_config()
            uri = uri if uri is not None else config.neo4j.uri
            user = user if user is not None else config.neo4j.user
            # For password, try to get from OrionSettings first, then fall back to env var
            if password is None:
                try:
                    from orion.settings import OrionSettings
                    settings = OrionSettings.load()
                    password = settings.get_neo4j_password()
                except Exception:
                    # Fall back to environment variable
                    try:
                        password = config.neo4j.password
                    except Exception:
                        password = "password"  # Ultimate fallback

        self.uri = uri
        self.user = user
        self.password = password
        self.driver: Any = None
        
        # Attempt to connect on initialization
        self.connect()

    def connect(self) -> bool:
        """
        Connect to Neo4j database

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("✓ Connected to Neo4j")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to connect to Neo4j: {e}")
            if self.driver:
                try:
                    self.driver.close()
                except:
                    pass
                self.driver = None
            return False

    def clear_database(self) -> bool:
        """
        Clear all nodes and relationships from the database

        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            logger.error("Not connected to Neo4j")
            return False

        try:
            with self.driver.session() as session:
                # Delete all nodes and relationships
                result = session.run("MATCH (n) DETACH DELETE n")
                logger.info("✓ Cleared Neo4j database")
                return True
        except Exception as e:
            logger.error(f"✗ Failed to clear database: {e}")
            return False

    def get_stats(self) -> dict:
        """
        Get database statistics

        Returns:
            Dictionary with node and relationship counts
        """
        if not self.driver:
            return {"nodes": 0, "relationships": 0}

        try:
            with self.driver.session() as session:
                # Count nodes
                node_result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = node_result.single()["count"]

                # Count relationships
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rel_count = rel_result.single()["count"]

                return {"nodes": node_count, "relationships": rel_count}
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"nodes": 0, "relationships": 0}

    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logger.debug("Neo4j connection closed")

    def create_entity_node(self, entity_id: str, object_class: str, properties: Optional[dict] = None) -> bool:
        """
        Create an entity node in the graph.
        
        Args:
            entity_id: Unique entity identifier
            object_class: Object class (e.g., "person", "car")
            properties: Additional properties
            
        Returns:
            True if successful
        """
        if not self.driver:
            logger.error("Not connected to Neo4j")
            return False
            
        try:
            with self.driver.session() as session:
                props = properties or {}
                props.update({"entity_id": entity_id, "object_class": object_class})
                
                query = """
                MERGE (e:Entity {entity_id: $entity_id})
                SET e += $properties
                """
                session.run(query, entity_id=entity_id, properties=props)
                return True
        except Exception as e:
            logger.error(f"Failed to create entity node: {e}")
            return False
    
    def create_scene_node(self, scene_id: str, properties: Optional[dict] = None) -> bool:
        """
        Create a scene node in the graph.
        
        Args:
            scene_id: Unique scene identifier
            properties: Scene properties
            
        Returns:
            True if successful
        """
        if not self.driver:
            logger.error("Not connected to Neo4j")
            return False
            
        try:
            with self.driver.session() as session:
                props = properties or {}
                props.update({"scene_id": scene_id})
                
                query = """
                MERGE (s:Scene {scene_id: $scene_id})
                SET s += $properties
                """
                session.run(query, scene_id=scene_id, properties=props)
                return True
        except Exception as e:
            logger.error(f"Failed to create scene node: {e}")
            return False
    
    def link_entity_to_scene(self, entity_id: str, scene_id: str) -> bool:
        """
        Create a relationship between an entity and a scene.
        
        Args:
            entity_id: Entity identifier
            scene_id: Scene identifier
            
        Returns:
            True if successful
        """
        if not self.driver:
            logger.error("Not connected to Neo4j")
            return False
            
        try:
            with self.driver.session() as session:
                query = """
                MATCH (e:Entity {entity_id: $entity_id})
                MATCH (s:Scene {scene_id: $scene_id})
                MERGE (e)-[:APPEARS_IN]->(s)
                """
                session.run(query, entity_id=entity_id, scene_id=scene_id)
                return True
        except Exception as e:
            logger.error(f"Failed to link entity to scene: {e}")
            return False


def clear_neo4j_for_new_run(
    uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None
) -> bool:
    """
    Utility function to clear Neo4j database before a new pipeline run

    Args:
        uri: Neo4j URI (optional, uses ConfigManager if not provided)
        user: Database username (optional, uses ConfigManager if not provided)
        password: Database password (optional, uses ConfigManager if not provided)

    Returns:
        True if successful, False otherwise
    """
    manager = Neo4jManager(uri, user, password)

    if not manager.connect():
        return False

    # Get current stats
    stats = manager.get_stats()
    if stats["nodes"] > 0 or stats["relationships"] > 0:
        logger.info(
            f"Found existing data: {stats['nodes']} nodes, {stats['relationships']} relationships"
        )
        success = manager.clear_database()
    else:
        logger.info("Database already empty")
        success = True

    manager.close()
    return success


if __name__ == "__main__":
    # Test the manager
    logging.basicConfig(level=logging.INFO)

    print("Testing Neo4j Manager...")
    success = clear_neo4j_for_new_run()

    if success:
        print("✓ Neo4j manager working correctly")
    else:
        print("✗ Neo4j manager test failed")
