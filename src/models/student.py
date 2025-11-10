"""Student model implementation - learning models that can be promoted."""

from typing import Dict, Any, Optional
from src.models.base import ModelConfig, ModelRole
from src.models.mock_model import MockLLM
from loguru import logger


class StudentModel(MockLLM):
    """
    Student model - learning model that improves over time.
    Can be promoted to TA or Teacher based on performance.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize student model.

        Args:
            config: Model configuration
        """
        # Ensure student role
        config.role = ModelRole.STUDENT
        super().__init__(config)

        # Student-specific attributes
        self.teacher_id: Optional[str] = config.metadata.get("teacher_id")
        self.learning_rate: float = config.metadata.get("learning_rate", 0.01)

        # Learning progress tracking
        self.learning_history = []
        self.feedback_received = 0
        self.improvement_trajectory = []

        logger.info(
            f"Initialized Student Model: {self.model_id} "
            f"(domain: {self.domain}, teacher: {self.teacher_id})"
        )

    def receive_feedback(
        self,
        query: str,
        your_response: str,
        winning_response: str,
        evaluation: Dict[str, Any],
    ) -> None:
        """
        Receive feedback and learn from it.

        Args:
            query: The original query
            your_response: This model's response
            winning_response: The selected best response
            evaluation: Evaluation metrics and feedback
        """
        super().receive_feedback(query, your_response, winning_response, evaluation)

        # Track feedback
        self.feedback_received += 1

        # Extract metrics
        your_metrics = evaluation.get("your_metrics", {})
        winner_metrics = evaluation.get("winner_metrics", {})

        # Calculate improvement needed
        improvement_needed = {}
        for metric, your_score in your_metrics.items():
            winner_score = winner_metrics.get(metric, your_score)
            if winner_score > your_score:
                improvement_needed[metric] = winner_score - your_score

        # Record learning event
        learning_event = {
            "query": query,
            "your_metrics": your_metrics,
            "winner_metrics": winner_metrics,
            "improvement_needed": improvement_needed,
            "winning_model": evaluation.get("winner_id"),
        }

        self.learning_history.append(learning_event)

        # Simulate learning - improve confidence gradually
        if improvement_needed:
            avg_gap = sum(improvement_needed.values()) / len(improvement_needed)
            improvement = min(avg_gap * self.learning_rate, 0.05)
            self.base_confidence = min(0.85, self.base_confidence + improvement)

            logger.debug(
                f"Student {self.model_id} learned from feedback. "
                f"Confidence: {self.base_confidence:.3f} (+{improvement:.3f})"
            )

        # Track improvement trajectory
        self.improvement_trajectory.append({
            "feedback_count": self.feedback_received,
            "confidence": self.base_confidence,
            "avg_metrics": self._calculate_recent_avg_metrics(),
        })

    def _calculate_recent_avg_metrics(self, window: int = 10) -> Dict[str, float]:
        """Calculate average metrics over recent history."""
        if not self.learning_history:
            return {}

        recent = self.learning_history[-window:]
        metrics_sum = {}
        count = 0

        for event in recent:
            for metric, value in event["your_metrics"].items():
                metrics_sum[metric] = metrics_sum.get(metric, 0) + value
                count += 1

        if count == 0:
            return {}

        return {
            metric: total / len(recent)
            for metric, total in metrics_sum.items()
        }

    def is_ready_for_promotion(
        self,
        min_queries: int = 30,
        min_confidence: float = 0.75,
        min_win_rate: float = 0.60,
    ) -> Dict[str, Any]:
        """
        Check if student is ready for promotion to TA.

        Args:
            min_queries: Minimum number of queries handled
            min_confidence: Minimum average confidence score
            min_win_rate: Minimum win rate

        Returns:
            Dictionary with promotion eligibility and reasoning
        """
        stats = self.get_statistics()

        criteria_met = {
            "sufficient_queries": stats["total_queries"] >= min_queries,
            "sufficient_confidence": stats["avg_confidence"] >= min_confidence,
            "sufficient_win_rate": stats["win_rate"] >= min_win_rate,
        }

        is_ready = all(criteria_met.values())

        result = {
            "is_ready": is_ready,
            "criteria_met": criteria_met,
            "current_stats": stats,
            "requirements": {
                "min_queries": min_queries,
                "min_confidence": min_confidence,
                "min_win_rate": min_win_rate,
            },
            "reasoning": self._generate_promotion_reasoning(
                stats, criteria_met, min_queries, min_confidence, min_win_rate
            ),
        }

        if is_ready:
            logger.info(f"Student {self.model_id} is ready for promotion to TA!")

        return result

    def _generate_promotion_reasoning(
        self,
        stats: Dict[str, Any],
        criteria_met: Dict[str, bool],
        min_queries: int,
        min_confidence: float,
        min_win_rate: float,
    ) -> str:
        """Generate reasoning for promotion decision."""
        parts = [f"Student {self.model_id} promotion analysis:"]

        parts.append(
            f"  Queries: {stats['total_queries']}/{min_queries} "
            f"({'✓' if criteria_met['sufficient_queries'] else '✗'})"
        )
        parts.append(
            f"  Confidence: {stats['avg_confidence']:.3f}/{min_confidence:.3f} "
            f"({'✓' if criteria_met['sufficient_confidence'] else '✗'})"
        )
        parts.append(
            f"  Win Rate: {stats['win_rate']:.3f}/{min_win_rate:.3f} "
            f"({'✓' if criteria_met['sufficient_win_rate'] else '✗'})"
        )

        if all(criteria_met.values()):
            parts.append("  Result: READY FOR PROMOTION")
        else:
            parts.append("  Result: NOT YET READY")

        return "\n".join(parts)

    def promote_to_ta(self) -> None:
        """Promote this student to TA role."""
        self.role = ModelRole.TA
        self.base_confidence += 0.05  # Small confidence boost from promotion

        logger.info(
            f"Student {self.model_id} promoted to TA "
            f"(queries: {self.total_queries}, "
            f"win_rate: {self.total_wins/self.total_queries if self.total_queries > 0 else 0:.3f})"
        )

    def promote_to_teacher(self) -> None:
        """Promote this TA to Teacher role."""
        if self.role != ModelRole.TA:
            logger.warning(
                f"Cannot promote {self.model_id} to Teacher - must be TA first"
            )
            return

        self.role = ModelRole.TEACHER
        self.base_confidence += 0.05

        logger.info(
            f"TA {self.model_id} promoted to Teacher "
            f"(queries: {self.total_queries}, "
            f"win_rate: {self.total_wins/self.total_queries if self.total_queries > 0 else 0:.3f})"
        )

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get detailed learning statistics."""
        stats = self.get_statistics()

        recent_metrics = self._calculate_recent_avg_metrics()

        return {
            **stats,
            "teacher_id": self.teacher_id,
            "learning_rate": self.learning_rate,
            "feedback_received": self.feedback_received,
            "total_learning_events": len(self.learning_history),
            "recent_avg_metrics": recent_metrics,
            "confidence_trajectory": [
                point["confidence"] for point in self.improvement_trajectory
            ],
        }
