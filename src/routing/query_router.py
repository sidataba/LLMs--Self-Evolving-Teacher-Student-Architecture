"""Query routing system for intelligent model selection."""

import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from src.database.vector_store import VectorStore
from src.database.metrics_store import MetricsStore


@dataclass
class Query:
    """Structured query object."""
    query_id: str
    query_text: str
    domain: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RoutingDecision:
    """Routing decision for a query."""
    query_id: str
    is_novel: bool
    similarity_score: float
    similar_queries: List[Dict[str, Any]]
    recommended_models: List[str]
    routing_strategy: str  # "parallel", "targeted", "hybrid"
    reasoning: str


class QueryRouter:
    """
    Intelligent query routing system.
    Determines whether queries are novel or similar to past queries,
    and routes them to appropriate models.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        metrics_store: MetricsStore,
        similarity_threshold: float = 0.80,
        novel_threshold: float = 0.50,
    ):
        """
        Initialize query router.

        Args:
            vector_store: Vector database for similarity search
            metrics_store: Metrics database for performance tracking
            similarity_threshold: Threshold for routing to best performer
            novel_threshold: Threshold below which query is considered novel
        """
        self.vector_store = vector_store
        self.metrics_store = metrics_store
        self.similarity_threshold = similarity_threshold
        self.novel_threshold = novel_threshold

        logger.info(
            f"QueryRouter initialized "
            f"(similarity_threshold: {similarity_threshold}, "
            f"novel_threshold: {novel_threshold})"
        )

    def process_query(
        self,
        query_text: str,
        domain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Query, RoutingDecision]:
        """
        Process a new query and determine routing strategy.

        Args:
            query_text: The query text
            domain: Optional domain hint
            metadata: Optional metadata

        Returns:
            Tuple of (Query object, RoutingDecision)
        """
        # Create query object
        query = Query(
            query_id=str(uuid.uuid4()),
            query_text=query_text,
            domain=domain,
            metadata=metadata or {},
        )

        # Find similar queries
        similar_queries = self.vector_store.find_similar_queries(
            query_text,
            top_k=5,
            min_similarity=0.3,  # Get at least some matches if available
        )

        # Determine similarity score
        similarity_score = (
            similar_queries[0]["similarity"]
            if similar_queries
            else 0.0
        )

        # Determine if query is novel
        is_novel = similarity_score < self.novel_threshold

        # Get routing decision
        decision = self._make_routing_decision(
            query,
            similarity_score,
            similar_queries,
            is_novel,
        )

        # Store query in vector database
        self.vector_store.add_query(
            query_id=query.query_id,
            query_text=query_text,
            metadata={
                "domain": domain,
                "is_novel": is_novel,
                "similarity_score": similarity_score,
                **(metadata or {}),
            },
        )

        logger.info(
            f"Query processed: {query.query_id[:8]}... "
            f"(novel: {is_novel}, similarity: {similarity_score:.3f}, "
            f"strategy: {decision.routing_strategy})"
        )

        return query, decision

    def _make_routing_decision(
        self,
        query: Query,
        similarity_score: float,
        similar_queries: List[Dict[str, Any]],
        is_novel: bool,
    ) -> RoutingDecision:
        """Make routing decision based on query analysis."""
        recommended_models = []
        routing_strategy = "parallel"  # default

        if is_novel:
            # Novel query - request parallel answers from all models
            routing_strategy = "parallel"
            reasoning = (
                f"Query is novel (similarity: {similarity_score:.3f} < {self.novel_threshold:.3f}). "
                f"Requesting parallel answers from all available models for comprehensive evaluation."
            )

        elif similarity_score >= self.similarity_threshold:
            # High similarity - route to best historical performer
            routing_strategy = "targeted"

            # Find best performer from similar queries
            best_performer = self._find_best_performer(similar_queries)

            if best_performer:
                recommended_models = [best_performer]
                reasoning = (
                    f"High similarity to past queries (similarity: {similarity_score:.3f} >= {self.similarity_threshold:.3f}). "
                    f"Routing to best historical performer: {best_performer}."
                )
            else:
                # No clear best performer, use parallel
                routing_strategy = "parallel"
                reasoning = (
                    f"High similarity (similarity: {similarity_score:.3f}) but no clear best performer found. "
                    f"Using parallel strategy."
                )

        else:
            # Medium similarity - hybrid approach
            routing_strategy = "hybrid"

            # Route to top performers and some students for learning
            top_performers = self._find_top_performers(similar_queries, top_k=2)
            recommended_models = top_performers

            reasoning = (
                f"Medium similarity (similarity: {similarity_score:.3f}). "
                f"Using hybrid strategy: route to top performers and selected students."
            )

        decision = RoutingDecision(
            query_id=query.query_id,
            is_novel=is_novel,
            similarity_score=similarity_score,
            similar_queries=similar_queries,
            recommended_models=recommended_models,
            routing_strategy=routing_strategy,
            reasoning=reasoning,
        )

        return decision

    def _find_best_performer(
        self,
        similar_queries: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Find the best performing model from similar queries.

        Args:
            similar_queries: List of similar queries with metadata

        Returns:
            Model ID of best performer or None
        """
        if not similar_queries:
            return None

        # Count wins per model
        model_performance = {}

        for similar_query in similar_queries:
            metadata = similar_query.get("metadata", {})
            winning_model = metadata.get("winning_model")

            if winning_model:
                if winning_model not in model_performance:
                    model_performance[winning_model] = {
                        "wins": 0,
                        "total": 0,
                        "avg_confidence": 0.0,
                    }

                model_performance[winning_model]["wins"] += 1
                model_performance[winning_model]["total"] += 1

        if not model_performance:
            return None

        # Find model with highest win rate
        best_model = max(
            model_performance.items(),
            key=lambda x: (x[1]["wins"], x[1]["total"]),
        )

        return best_model[0]

    def _find_top_performers(
        self,
        similar_queries: List[Dict[str, Any]],
        top_k: int = 2,
    ) -> List[str]:
        """Find top K performing models from similar queries."""
        if not similar_queries:
            return []

        # Count performance per model
        model_performance = {}

        for similar_query in similar_queries:
            metadata = similar_query.get("metadata", {})
            winning_model = metadata.get("winning_model")

            if winning_model:
                model_performance[winning_model] = model_performance.get(winning_model, 0) + 1

        # Sort by performance
        sorted_models = sorted(
            model_performance.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Return top K
        return [model_id for model_id, _ in sorted_models[:top_k]]

    def update_query_result(
        self,
        query_id: str,
        winning_model: str,
        confidence: float,
        evaluation_metrics: Dict[str, float],
    ) -> None:
        """
        Update query metadata with results.

        Args:
            query_id: Query ID
            winning_model: ID of winning model
            confidence: Confidence score
            evaluation_metrics: Evaluation metrics
        """
        metadata_update = {
            "winning_model": winning_model,
            "confidence": confidence,
            "evaluation_metrics": evaluation_metrics,
        }

        self.vector_store.update_query_metadata(query_id, metadata_update)

        logger.debug(
            f"Updated query {query_id[:8]}... with winner: {winning_model}"
        )

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get statistics about routing decisions."""
        # This could be enhanced to track routing decisions over time
        return {
            "total_queries_stored": self.vector_store.count(),
            "similarity_threshold": self.similarity_threshold,
            "novel_threshold": self.novel_threshold,
        }
