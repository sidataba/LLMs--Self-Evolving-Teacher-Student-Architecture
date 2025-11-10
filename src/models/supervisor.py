"""Supervisor model implementation - manages routing and evaluation."""

from typing import Dict, Any, Optional, List
from src.models.base import BaseModel, ModelResponse, ModelConfig, ModelRole
from src.models.mock_model import MockLLM
from loguru import logger


class SupervisorModel(MockLLM):
    """
    Supervisor model that manages query routing and response evaluation.
    Coordinates between teachers and students, evaluates responses, and provides feedback.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize supervisor model.

        Args:
            config: Model configuration
        """
        # Ensure supervisor role
        config.role = ModelRole.SUPERVISOR
        super().__init__(config)

        logger.info(f"Initialized Supervisor Model: {self.model_id}")

    def evaluate_multiple_responses(
        self,
        query: str,
        responses: List[ModelResponse],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate multiple responses and select the best one.

        Args:
            query: The original query
            responses: List of model responses to evaluate
            weights: Optional weights for different metrics

        Returns:
            Dictionary with evaluation results and winner selection
        """
        if not responses:
            return {
                "winner": None,
                "evaluations": [],
                "reasoning": "No responses to evaluate",
            }

        # Default weights
        if weights is None:
            weights = {
                "relevance": 0.3,
                "correctness": 0.4,
                "completeness": 0.2,
                "clarity": 0.1,
            }

        evaluations = []

        # Evaluate each response
        for response in responses:
            metrics = self.evaluate_response(query, response.response_text)

            # Calculate weighted score
            weighted_score = sum(
                metrics.get(metric, 0) * weight
                for metric, weight in weights.items()
            )

            # Combine with model's self-reported confidence
            final_score = 0.7 * weighted_score + 0.3 * response.confidence

            evaluation = {
                "model_id": response.model_id,
                "metrics": metrics,
                "weighted_score": weighted_score,
                "final_score": final_score,
                "confidence": response.confidence,
            }

            evaluations.append(evaluation)

        # Select winner (highest final score)
        winner = max(evaluations, key=lambda x: x["final_score"])

        # Generate reasoning
        reasoning = self._generate_evaluation_reasoning(query, responses, evaluations, winner)

        result = {
            "winner": winner,
            "evaluations": evaluations,
            "reasoning": reasoning,
            "query": query,
        }

        logger.info(
            f"Supervisor evaluated {len(responses)} responses. "
            f"Winner: {winner['model_id']} (score: {winner['final_score']:.3f})"
        )

        return result

    def _generate_evaluation_reasoning(
        self,
        query: str,
        responses: List[ModelResponse],
        evaluations: List[Dict[str, Any]],
        winner: Dict[str, Any],
    ) -> str:
        """Generate human-readable reasoning for the evaluation."""
        reasoning_parts = [
            f"Evaluated {len(responses)} responses for query: '{query[:50]}...'",
            f"\nWinner: {winner['model_id']} with score {winner['final_score']:.3f}",
            "\nScores breakdown:",
        ]

        # Sort by score
        sorted_evals = sorted(evaluations, key=lambda x: x["final_score"], reverse=True)

        for i, eval_data in enumerate(sorted_evals[:5], 1):  # Top 5
            reasoning_parts.append(
                f"  {i}. {eval_data['model_id']}: "
                f"Final={eval_data['final_score']:.3f} "
                f"(Weighted={eval_data['weighted_score']:.3f}, "
                f"Confidence={eval_data['confidence']:.3f})"
            )

        return "\n".join(reasoning_parts)

    def synthesize_response(
        self,
        query: str,
        responses: List[ModelResponse],
        evaluation_result: Dict[str, Any],
    ) -> ModelResponse:
        """
        Synthesize a final response, potentially combining multiple responses.

        Args:
            query: The original query
            responses: All model responses
            evaluation_result: Results from evaluate_multiple_responses

        Returns:
            Synthesized final response
        """
        winner = evaluation_result["winner"]

        if not winner:
            # No valid responses, generate supervisor response
            return self.generate_response(query)

        # Find winning response
        winning_response = next(
            (r for r in responses if r.model_id == winner["model_id"]),
            None
        )

        if not winning_response:
            return self.generate_response(query)

        # In a real implementation, might combine multiple good responses
        # For now, return the winning response with supervisor endorsement
        synthesized = ModelResponse(
            model_id=self.model_id,
            response_text=winning_response.response_text,
            confidence=winner["final_score"],
            reasoning=evaluation_result["reasoning"],
            metadata={
                "synthesized_from": winner["model_id"],
                "num_candidates": len(responses),
                "evaluation": evaluation_result,
            },
        )

        return synthesized

    def should_route_to_teacher(
        self,
        query: str,
        similarity_score: float,
        best_performer: Optional[str] = None,
        threshold: float = 0.80,
    ) -> Dict[str, Any]:
        """
        Decide whether to route query to a specific teacher or request all models to answer.

        Args:
            query: The query text
            similarity_score: Similarity to past queries (0-1)
            best_performer: ID of best past performer for similar queries
            threshold: Similarity threshold for routing

        Returns:
            Dictionary with routing decision
        """
        route_to_teacher = similarity_score >= threshold and best_performer is not None

        decision = {
            "route_to_teacher": route_to_teacher,
            "teacher_id": best_performer if route_to_teacher else None,
            "similarity_score": similarity_score,
            "reasoning": self._generate_routing_reasoning(
                similarity_score, threshold, route_to_teacher, best_performer
            ),
            "request_parallel": not route_to_teacher,  # If not routing, request parallel answers
        }

        logger.info(
            f"Routing decision: "
            f"{'Route to ' + str(best_performer) if route_to_teacher else 'Parallel answering'} "
            f"(similarity: {similarity_score:.3f})"
        )

        return decision

    def _generate_routing_reasoning(
        self,
        similarity_score: float,
        threshold: float,
        route_to_teacher: bool,
        teacher_id: Optional[str],
    ) -> str:
        """Generate reasoning for routing decision."""
        if route_to_teacher:
            return (
                f"Query similarity ({similarity_score:.3f}) exceeds threshold ({threshold:.3f}). "
                f"Routing to best performer: {teacher_id} for cost efficiency."
            )
        else:
            return (
                f"Query similarity ({similarity_score:.3f}) below threshold ({threshold:.3f}). "
                f"Novel query detected - requesting parallel answers from all available models."
            )
