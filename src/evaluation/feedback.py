"""Feedback loop for model learning and improvement."""

from typing import Dict, Any, List
from loguru import logger

from src.models.base import BaseModel, ModelResponse
from src.evaluation.evaluator import EvaluationResult


class FeedbackLoop:
    """
    Feedback system that distributes learning signals to all models.
    Enables knowledge distillation and continuous improvement.
    """

    def __init__(
        self,
        enable_distillation: bool = True,
        distillation_temperature: float = 2.0,
        distillation_alpha: float = 0.7,
    ):
        """
        Initialize feedback loop.

        Args:
            enable_distillation: Whether to enable knowledge distillation
            distillation_temperature: Temperature for distillation
            distillation_alpha: Alpha parameter for distillation
        """
        self.enable_distillation = enable_distillation
        self.distillation_temperature = distillation_temperature
        self.distillation_alpha = distillation_alpha

        self.feedback_count = 0

        logger.info(
            f"FeedbackLoop initialized "
            f"(distillation: {enable_distillation}, "
            f"temperature: {distillation_temperature}, "
            f"alpha: {distillation_alpha})"
        )

    def distribute_feedback(
        self,
        query: str,
        models: Dict[str, BaseModel],
        responses: List[ModelResponse],
        evaluations: List[EvaluationResult],
    ) -> Dict[str, Any]:
        """
        Distribute feedback to all participating models.

        Args:
            query: The original query
            models: Dictionary of model_id -> model objects
            responses: List of all responses
            evaluations: Evaluation results for all responses

        Returns:
            Dictionary with feedback statistics
        """
        if not evaluations:
            return {"feedback_sent": 0}

        # Get winning response
        winner = evaluations[0]  # Already sorted by rank
        winning_response = next(
            (r for r in responses if r.model_id == winner.model_id),
            None
        )

        if not winning_response:
            logger.warning("Could not find winning response for feedback")
            return {"feedback_sent": 0}

        feedback_sent = 0

        # Send feedback to each model
        for response in responses:
            model = models.get(response.model_id)

            if not model:
                continue

            # Find this model's evaluation
            model_eval = next(
                (e for e in evaluations if e.model_id == response.model_id),
                None
            )

            if not model_eval:
                continue

            # Prepare feedback
            feedback = self._prepare_feedback(
                query=query,
                model_response=response,
                model_evaluation=model_eval,
                winning_response=winning_response,
                winner_evaluation=winner,
            )

            # Send feedback to model
            model.receive_feedback(
                query=query,
                your_response=response.response_text,
                winning_response=winning_response.response_text,
                evaluation=feedback,
            )

            feedback_sent += 1

        self.feedback_count += feedback_sent

        logger.info(
            f"Distributed feedback to {feedback_sent} models. "
            f"Winner: {winner.model_id}"
        )

        return {
            "feedback_sent": feedback_sent,
            "winner_id": winner.model_id,
            "winner_score": winner.final_score,
            "total_feedback_count": self.feedback_count,
        }

    def _prepare_feedback(
        self,
        query: str,
        model_response: ModelResponse,
        model_evaluation: EvaluationResult,
        winning_response: ModelResponse,
        winner_evaluation: EvaluationResult,
    ) -> Dict[str, Any]:
        """Prepare feedback package for a model."""
        is_winner = model_response.model_id == winning_response.model_id

        feedback = {
            "query": query,
            "is_winner": is_winner,
            "your_metrics": model_evaluation.metrics,
            "your_score": model_evaluation.final_score,
            "your_rank": model_evaluation.rank,
            "winner_id": winning_response.model_id,
            "winner_metrics": winner_evaluation.metrics,
            "winner_score": winner_evaluation.final_score,
            "winner_response": winning_response.response_text,
            "score_gap": winner_evaluation.final_score - model_evaluation.final_score,
        }

        # Add metric-specific gaps for improvement
        metric_gaps = {}
        for metric, your_score in model_evaluation.metrics.items():
            winner_score = winner_evaluation.metrics.get(metric, your_score)
            if winner_score > your_score:
                metric_gaps[metric] = winner_score - your_score

        feedback["improvement_areas"] = metric_gaps

        # Add distillation information if enabled
        if self.enable_distillation and not is_winner:
            feedback["distillation"] = {
                "teacher_response": winning_response.response_text,
                "teacher_confidence": winning_response.confidence,
                "temperature": self.distillation_temperature,
                "alpha": self.distillation_alpha,
            }

        return feedback

    def generate_learning_summary(
        self,
        model_id: str,
        recent_feedback: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate a learning summary for a model based on recent feedback.

        Args:
            model_id: Model ID
            recent_feedback: List of recent feedback events

        Returns:
            Dictionary with learning summary
        """
        if not recent_feedback:
            return {
                "model_id": model_id,
                "total_feedback": 0,
                "win_rate": 0.0,
                "avg_score": 0.0,
                "common_improvement_areas": [],
            }

        # Calculate statistics
        total = len(recent_feedback)
        wins = sum(1 for f in recent_feedback if f.get("is_winner", False))
        win_rate = wins / total if total > 0 else 0.0

        total_score = sum(f.get("your_score", 0) for f in recent_feedback)
        avg_score = total_score / total if total > 0 else 0.0

        # Find common improvement areas
        improvement_counts = {}
        for feedback in recent_feedback:
            for area in feedback.get("improvement_areas", {}).keys():
                improvement_counts[area] = improvement_counts.get(area, 0) + 1

        common_areas = sorted(
            improvement_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]  # Top 3

        summary = {
            "model_id": model_id,
            "total_feedback": total,
            "wins": wins,
            "win_rate": win_rate,
            "avg_score": avg_score,
            "common_improvement_areas": [area for area, _ in common_areas],
        }

        return summary
