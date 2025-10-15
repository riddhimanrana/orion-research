"""
Neo4j Database Management Utilities

This module provides utilities for managing the Neo4j database,
including clearing all data before new pipeline runs.
"""

import logging
from neo4j import GraphDatabase
from typing import Optional

logger = logging.getLogger(__name__)


class Neo4jManager:
    """Manager for Neo4j database operations"""

    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize Neo4j manager

        Args:
            uri: Neo4j connection URI (e.g., neo4j://127.0.0.1:7687)
            user: Database username
            password: Database password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver: Optional[GraphDatabase.driver] = None

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


def clear_neo4j_for_new_run(
    uri: str = "neo4j://127.0.0.1:7687", user: str = "neo4j", password: str = "orion123"
) -> bool:
    """
    Utility function to clear Neo4j database before a new pipeline run

    Args:
        uri: Neo4j URI
        user: Database username
        password: Database password

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
