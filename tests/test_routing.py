"""Tests for query routing."""

import pytest
import tempfile
import shutil
from src.routing.query_router import QueryRouter
from src.database.vector_store import VectorStore
from src.database.metrics_store import MetricsStore


class TestQueryRouter:
    """Tests for QueryRouter."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary directory for test databases."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def router(self, temp_db_path):
        """Create a test router."""
        vector_store = VectorStore(db_path=f"{temp_db_path}/vector")
        metrics_store = MetricsStore(db_path=f"{temp_db_path}/metrics")

        return QueryRouter(
            vector_store=vector_store,
            metrics_store=metrics_store,
            similarity_threshold=0.80,
            novel_threshold=0.50,
        )

    def test_process_novel_query(self, router):
        """Test processing a novel query."""
        query, decision = router.process_query("What is quantum physics?")

        assert query.query_id
        assert query.query_text == "What is quantum physics?"
        assert decision.is_novel
        assert decision.routing_strategy == "parallel"

    def test_process_similar_query(self, router):
        """Test processing a similar query."""
        # Add initial query
        query1, _ = router.process_query("What is machine learning?")

        # Update with winner
        router.update_query_result(
            query_id=query1.query_id,
            winning_model="teacher-1",
            confidence=0.9,
            evaluation_metrics={"correctness": 0.9},
        )

        # Process similar query
        query2, decision2 = router.process_query("What is machine learning?")

        # Should find high similarity
        assert decision2.similarity_score > 0.8
        assert not decision2.is_novel

    def test_update_query_result(self, router):
        """Test updating query results."""
        query, _ = router.process_query("Test query")

        router.update_query_result(
            query_id=query.query_id,
            winning_model="model-1",
            confidence=0.85,
            evaluation_metrics={"correctness": 0.9},
        )

        # Retrieve and verify
        stored_query = router.vector_store.get_query(query.query_id)
        assert stored_query["metadata"]["winning_model"] == "model-1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
