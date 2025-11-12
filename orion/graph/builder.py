"""Stub GraphBuilder"""
import logging

logger = logging.getLogger('orion.graph')

class GraphBuilder:
    def __init__(self, config=None):
        self.config = config
    
    def add_entity(self, entity):
        return entity.get('id', 'unknown')
    
    def add_relationship(self, subj, pred, obj):
        pass
