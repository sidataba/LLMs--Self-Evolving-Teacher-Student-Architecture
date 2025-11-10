"""Response evaluation system."""

from typing import Dict, Any, List
from dataclasses import dataclass
from loguru import logger

from src.models.base import ModelResponse


@dataclass
class EvaluationResult:
    """Result of response evaluation."""
    model_id: str
    metrics: Dict[str, float]
    weighted_score: float
    final_score: float
    rank: int = 0


class ResponseEvaluator:
    """
    System for evaluating model responses.
    Compares multiple responses and ranks them based on configurable metrics.
    """

    def __init__(
        self,
        metric_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize evaluator.

        Args:
            metric_weights: Weights for different evaluation metrics
        """
        self.metric_weights = metric_weights or {
            "relevance": 0.3,
            "correctness": 0.4,
            "completeness": 0.2,
            "clarity": 0.1,
        }

        # Validate weights sum to 1.0
        total_weight = sum(self.metric_weights.values())
        if not (0.99 <= total_weight <= 1.01):
            logger.warning(
                f"Metric weights sum to {total_weight}, not 1.0. Normalizing..."
            )
            # Normalize
            self.metric_weights = {
                k: v / total_weight
                for k, v in self.metric_weights.items()
            }

        logger.info(
            f"ResponseEvaluator initialized with weights: {self.metric_weights}"
        )

    def evaluate_responses(
        self,
        query: str,
        responses: List[ModelResponse],
        supervisor_evaluations: Optional[List[Dict[str, float]]] = None,
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple responses and rank them.

        Args:
            query: The original query
            responses: List of model responses
            supervisor_evaluations: Optional pre-computed evaluations from supervisor

        Returns:
            List of evaluation results, sorted by final score
        """
        if not responses:
            return []

        evaluation_results = []

        for i, response in enumerate(responses):
            # Use supervisor evaluation if provided, otherwise use model's own
            if supervisor_evaluations and i < len(supervisor_evaluations):
                metrics = supervisor_evaluations[i]
            else:
                # Use placeholder metrics (in real system, would compute these)
                metrics = {
                    "relevance": response.confidence * 0.9,
                    "correctness": response.confidence * 0.95,
                    "completeness": response.confidence * 0.85,
                    "clarity": response.confidence * 0.9,
                }

            # Calculate weighted score
            weighted_score = self._calculate_weighted_score(metrics)

            # Combine weighted score with model confidence
            final_score = 0.7 * weighted_score + 0.3 * response.confidence

            result = EvaluationResult(
                model_id=response.model_id,
                metrics=metrics,
                weighted_score=weighted_score,
                final_score=final_score,
            )

            evaluation_results.append(result)

        # Sort by final score and assign ranks
        evaluation_results.sort(key=lambda x: x.final_score, reverse=True)

        for rank, result in enumerate(evaluation_results, 1):
            result.rank = rank

        logger.info(
            f"Evaluated {len(responses)} responses. "
            f"Winner: {evaluation_results[0].model_id} "
            f"(score: {evaluation_results[0].final_score:.3f})"
        )

        return evaluation_results

    def _calculate_weighted_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted score from metrics."""
        score = 0.0

        for metric, weight in self.metric_weights.items():
            metric_value = metrics.get(metric, 0.0)
            score += metric_value * weight

        return score

    def compare_responses(
        self,
        response1: ModelResponse,
        response2: ModelResponse,
        query: str,
    ) -> Dict[str, Any]:
        """
        Compare two responses head-to-head.

        Args:
            response1: First response
            response2: Second response
            query: Original query

        Returns:
            Dictionary with comparison results
        """
        eval1 = self.evaluate_responses(query, [response1])[0]
        eval2 = self.evaluate_responses(query, [response2])[0]

        winner = eval1 if eval1.final_score > eval2.final_score else eval2
        loser = eval2 if winner == eval1 else eval1

        return {
            "winner": winner.model_id,
            "loser": loser.model_id,
            "winner_score": winner.final_score,
            "loser_score": loser.final_score,
            "score_difference": abs(winner.final_score - loser.final_score),
            "winner_metrics": winner.metrics,
            "loser_metrics": loser.metrics,
        }

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update metric weights."""
        total = sum(new_weights.values())
        self.metric_weights = {
            k: v / total
            for k, v in new_weights.items()
        }

        logger.info(f"Updated metric weights: {self.metric_weights}")
