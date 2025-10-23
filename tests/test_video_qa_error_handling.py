"""
Tests for Video QA error handling improvements
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from orion.video_qa.system import VideoQASystem


class TestVideoQAErrorHandling:
    """Test error handling in VideoQASystem"""
    
    def test_ask_question_empty(self):
        """Empty question should return helpful message"""
        qa = VideoQASystem()
        result = qa.ask_question("")
        assert "Please provide a question" in result
        
        result = qa.ask_question("   ")
        assert "Please provide a question" in result
    
    def test_ask_question_no_connection(self):
        """Should handle missing Neo4j connection gracefully"""
        qa = VideoQASystem()
        # Don't connect to Neo4j
        result = qa.ask_question("What happened?")
        assert "No video analysis is available" in result or "No knowledge graph" in result
    
    @patch('orion.video_qa.system.GraphDatabase')
    def test_connect_service_unavailable(self, mock_graphdb):
        """Should handle Neo4j service unavailable"""
        from neo4j.exceptions import ServiceUnavailable
        
        mock_graphdb.driver.side_effect = ServiceUnavailable("Service unavailable")
        
        qa = VideoQASystem()
        result = qa.connect()
        
        assert result is False
    
    @patch('orion.video_qa.system.GraphDatabase')
    def test_connect_success(self, mock_graphdb):
        """Should successfully connect to Neo4j"""
        # Mock successful connection
        mock_driver = Mock()
        mock_session = Mock()
        mock_result = Mock()
        mock_result.single.return_value = 1
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_graphdb.driver.return_value = mock_driver
        
        qa = VideoQASystem()
        result = qa.connect()
        
        assert result is True
        assert qa.driver is not None
    
    def test_close(self):
        """Should close driver cleanly"""
        qa = VideoQASystem()
        qa.driver = Mock()
        qa.close()
        
        qa.driver.close.assert_called_once()
        assert qa.driver is None
    
    @patch('orion.video_qa.system.GraphDatabase')
    def test_check_data_available_with_data(self, mock_graphdb):
        """Should correctly report data availability"""
        # Mock Neo4j with data
        mock_driver = Mock()
        mock_session = Mock()
        
        # Mock count queries
        def run_side_effect(query):
            mock_result = Mock()
            if "Entity" in query:
                mock_result.single.return_value = {"count": 10}
            elif "Scene" in query:
                mock_result.single.return_value = {"count": 5}
            elif "Event" in query:
                mock_result.single.return_value = {"count": 20}
            return mock_result
        
        mock_session.run.side_effect = run_side_effect
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_graphdb.driver.return_value = mock_driver
        
        qa = VideoQASystem()
        qa.connect()
        
        status = qa.check_data_available()
        
        assert status["available"] is True
        assert status["entities"] == 10
        assert status["scenes"] == 5
        assert status["events"] == 20
    
    @patch('orion.video_qa.system.GraphDatabase')
    def test_check_data_available_empty(self, mock_graphdb):
        """Should correctly report no data available"""
        # Mock Neo4j with no data
        mock_driver = Mock()
        mock_session = Mock()
        
        def run_side_effect(query):
            mock_result = Mock()
            mock_result.single.return_value = {"count": 0}
            return mock_result
        
        mock_session.run.side_effect = run_side_effect
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_graphdb.driver.return_value = mock_driver
        
        qa = VideoQASystem()
        qa.connect()
        
        status = qa.check_data_available()
        
        assert status["available"] is False
        assert status["entities"] == 0
    
    @patch('orion.video_qa.system.ollama')
    @patch('orion.video_qa.system.GraphDatabase')
    def test_ask_question_ollama_connection_error(self, mock_graphdb, mock_ollama):
        """Should handle Ollama connection errors gracefully"""
        # Mock Neo4j connection
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.run.return_value.data.return_value = []
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_graphdb.driver.return_value = mock_driver
        
        # Mock Ollama connection error
        mock_ollama.chat.side_effect = ConnectionError("Cannot connect to Ollama")
        
        qa = VideoQASystem()
        qa.connect()
        
        # Force some context (to bypass "no data" check)
        with patch.object(qa, 'get_video_context', return_value="Some context"):
            result = qa.ask_question("What happened?")
        
        assert "Cannot connect to Ollama" in result or "Ollama" in result
    
    @patch('orion.video_qa.system.GraphDatabase')
    def test_get_video_context_partial_failure(self, mock_graphdb):
        """Should handle partial context retrieval failures"""
        # Mock Neo4j with some queries failing
        mock_driver = Mock()
        mock_session = Mock()
        
        call_count = [0]
        
        def run_side_effect(query):
            call_count[0] += 1
            if call_count[0] == 1:
                # First query succeeds
                mock_result = Mock()
                mock_result.data.return_value = [{"class": "person", "appearances": 10}]
                return mock_result
            else:
                # Subsequent queries fail
                raise Exception("Query failed")
        
        mock_session.run.side_effect = run_side_effect
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_graphdb.driver.return_value = mock_driver
        
        qa = VideoQASystem()
        qa.connect()
        
        # Should not crash, should return partial results
        context = qa.get_video_context("What happened?")
        
        # Should return something (not empty string)
        assert isinstance(context, str)
        # Might be partial data or "No video analysis available"
        assert len(context) > 0
    
    def test_format_history_empty(self):
        """Should handle empty conversation history"""
        qa = VideoQASystem()
        result = qa._format_history()
        
        # Should return empty or minimal string
        assert isinstance(result, str)
    
    def test_classify_question_none(self):
        """Should handle None question"""
        qa = VideoQASystem()
        result = qa._classify_question(None)
        
        # Should default to "general"
        assert result == "general"


class TestVideoQAInteractiveSession:
    """Test interactive session error handling"""
    
    @patch('orion.video_qa.system.ollama')
    @patch('orion.video_qa.system.Console')
    def test_interactive_session_ollama_unavailable(self, mock_console, mock_ollama):
        """Should gracefully handle Ollama unavailable"""
        mock_ollama.list.side_effect = ConnectionError("Cannot connect")
        
        qa = VideoQASystem()
        
        # Should not crash
        qa.start_interactive_session()
        
        # Should have printed error message
        mock_console.return_value.print.assert_called()
    
    @patch('orion.video_qa.system.ollama')
    @patch('orion.video_qa.system.Console')
    @patch('orion.video_qa.system.GraphDatabase')
    def test_interactive_session_neo4j_unavailable(self, mock_graphdb, mock_console, mock_ollama):
        """Should gracefully handle Neo4j unavailable"""
        # Ollama is available
        mock_ollama.list.return_value = {"models": [{"name": "gemma3:4b"}]}
        
        # Neo4j is not available
        from neo4j.exceptions import ServiceUnavailable
        mock_graphdb.driver.side_effect = ServiceUnavailable("Service unavailable")
        
        qa = VideoQASystem()
        
        # Should not crash
        qa.start_interactive_session()
        
        # Should have printed error message
        calls = mock_console.return_value.print.call_args_list
        assert any("Neo4j" in str(call) for call in calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
