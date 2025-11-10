"""Tests for the main orchestrator."""

import pytest
import tempfile
import shutil
from src.core.orchestrator import Orchestrator


class TestOrchestrator:
    """Tests for Orchestrator."""

    @pytest.fixture
    def temp_config(self):
        """Create temporary config for testing."""
        config = {
            "models": {
                "supervisor": {
                    "model_id": "test-supervisor",
                    "model_type": "mock",
                },
                "teachers": [
                    {
                        "model_id": "test-teacher-1",
                        "model_type": "mock",
                        "domain": "mathematics",
                    }
                ],
                "students": [
                    {
                        "model_id": "test-student-1",
                        "model_type": "mock",
                        "domain": "mathematics",
                        "teacher_id": "test-teacher-1",
                    }
                ],
            },
            "vector_database": {
                "path": "./data/test_vector_db",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "routing": {
                "similarity_threshold": 0.80,
                "novel_query_threshold": 0.50,
            },
            "evaluation": {
                "weights": {
                    "relevance": 0.3,
                    "correctness": 0.4,
                    "completeness": 0.2,
                    "clarity": 0.1,
                },
            },
            "feedback": {
                "distillation": {
                    "enabled": True,
                    "temperature": 2.0,
                    "alpha": 0.7,
                },
            },
        }

        return config

    def test_initialization(self, temp_config):
        """Test orchestrator initialization."""
        # Note: This test requires config file, so we test with default
        orchestrator = Orchestrator()

        assert orchestrator.supervisor is not None
        assert len(orchestrator.models) > 0

    def test_process_query(self, temp_config):
        """Test query processing."""
        orchestrator = Orchestrator()

        result = orchestrator.process_query("What is mathematics?")

        assert result["query_id"]
        assert result["final_response"]
        assert result["winner_model"]
        assert result["winner_score"] > 0

    def test_add_student_model(self):
        """Test adding a new student model."""
        orchestrator = Orchestrator()

        initial_count = len(orchestrator.models)

        student = orchestrator.add_student_model(
            model_id="new-student",
            domain="science",
            teacher_id="teacher-science",
        )

        assert len(orchestrator.models) == initial_count + 1
        assert "new-student" in orchestrator.models
        assert student.domain == "science"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
