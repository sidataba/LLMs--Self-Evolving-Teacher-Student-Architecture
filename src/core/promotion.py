"""Promotion system for student models."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from loguru import logger

from src.models.student import StudentModel
from src.models.base import ModelRole
from src.database.metrics_store import MetricsStore


@dataclass
class PromotionCriteria:
    """Criteria for model promotion."""
    min_queries: int
    min_confidence: float
    min_win_rate: float


class PromotionSystem:
    """
    System for managing model promotions from Student -> TA -> Teacher.
    Monitors performance and promotes models based on configurable criteria.
    """

    def __init__(
        self,
        metrics_store: MetricsStore,
        student_to_ta_criteria: Optional[PromotionCriteria] = None,
        ta_to_teacher_criteria: Optional[PromotionCriteria] = None,
        enable_demotion: bool = True,
        demotion_window: int = 20,
        demotion_threshold: float = 0.50,
    ):
        """
        Initialize promotion system.

        Args:
            metrics_store: Metrics database
            student_to_ta_criteria: Criteria for Student -> TA promotion
            ta_to_teacher_criteria: Criteria for TA -> Teacher promotion
            enable_demotion: Whether to enable demotion for underperforming models
            demotion_window: Number of recent queries to consider for demotion
            demotion_threshold: Performance threshold for demotion
        """
        self.metrics_store = metrics_store

        # Default promotion criteria
        self.student_to_ta_criteria = student_to_ta_criteria or PromotionCriteria(
            min_queries=30,
            min_confidence=0.75,
            min_win_rate=0.60,
        )

        self.ta_to_teacher_criteria = ta_to_teacher_criteria or PromotionCriteria(
            min_queries=50,
            min_confidence=0.85,
            min_win_rate=0.70,
        )

        self.enable_demotion = enable_demotion
        self.demotion_window = demotion_window
        self.demotion_threshold = demotion_threshold

        self.promotion_history = []

        logger.info(
            f"PromotionSystem initialized "
            f"(S->TA: {self.student_to_ta_criteria.min_queries}q/{self.student_to_ta_criteria.min_win_rate}wr, "
            f"TA->T: {self.ta_to_teacher_criteria.min_queries}q/{self.ta_to_teacher_criteria.min_win_rate}wr)"
        )

    def check_promotions(
        self,
        models: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Check all models for promotion eligibility.

        Args:
            models: Dictionary of model_id -> model objects

        Returns:
            List of promotion events
        """
        promotions = []

        for model_id, model in models.items():
            # Only check students and TAs
            if model.role not in [ModelRole.STUDENT, ModelRole.TA]:
                continue

            # Get model statistics from metrics store
            stats = self.metrics_store.get_model_statistics(model_id)

            if not stats:
                continue

            # Check for promotion
            promotion = self._check_model_promotion(model, stats)

            if promotion:
                promotions.append(promotion)

        return promotions

    def _check_model_promotion(
        self,
        model: Any,
        stats: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Check if a specific model is eligible for promotion."""
        if model.role == ModelRole.STUDENT:
            criteria = self.student_to_ta_criteria
            target_role = ModelRole.TA
        elif model.role == ModelRole.TA:
            criteria = self.ta_to_teacher_criteria
            target_role = ModelRole.TEACHER
        else:
            return None

        # Check criteria
        meets_queries = stats["total_queries"] >= criteria.min_queries
        meets_confidence = stats["avg_confidence"] >= criteria.min_confidence
        meets_win_rate = stats["win_rate"] >= criteria.min_win_rate

        if meets_queries and meets_confidence and meets_win_rate:
            # Eligible for promotion!
            promotion = self._execute_promotion(
                model=model,
                from_role=model.role,
                to_role=target_role,
                stats=stats,
                criteria=criteria,
            )
            return promotion

        return None

    def _execute_promotion(
        self,
        model: Any,
        from_role: ModelRole,
        to_role: ModelRole,
        stats: Dict[str, Any],
        criteria: PromotionCriteria,
    ) -> Dict[str, Any]:
        """Execute a model promotion."""
        reason = (
            f"Met promotion criteria: "
            f"{stats['total_queries']}>={criteria.min_queries} queries, "
            f"{stats['avg_confidence']:.3f}>={criteria.min_confidence} confidence, "
            f"{stats['win_rate']:.3f}>={criteria.min_win_rate} win rate"
        )

        # Update model role
        if to_role == ModelRole.TA:
            model.promote_to_ta()
        elif to_role == ModelRole.TEACHER:
            model.promote_to_teacher()

        # Log promotion
        self.metrics_store.log_promotion(
            model_id=model.model_id,
            from_role=from_role.value,
            to_role=to_role.value,
            reason=reason,
            stats=stats,
        )

        promotion_event = {
            "model_id": model.model_id,
            "from_role": from_role.value,
            "to_role": to_role.value,
            "reason": reason,
            "stats": stats,
        }

        self.promotion_history.append(promotion_event)

        logger.info(
            f"PROMOTION: {model.model_id} from {from_role.value} to {to_role.value} "
            f"({stats['total_queries']} queries, {stats['win_rate']:.3f} win rate)"
        )

        return promotion_event

    def check_demotions(
        self,
        models: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Check for models that should be demoted due to poor performance.

        Args:
            models: Dictionary of model_id -> model objects

        Returns:
            List of demotion events
        """
        if not self.enable_demotion:
            return []

        demotions = []

        for model_id, model in models.items():
            # Only check TAs and Teachers
            if model.role not in [ModelRole.TA, ModelRole.TEACHER]:
                continue

            # Check recent performance
            recent_queries = self.metrics_store.get_query_history(
                model_id=model_id,
                limit=self.demotion_window,
            )

            if len(recent_queries) < self.demotion_window:
                continue  # Not enough data

            # Calculate recent win rate
            wins = sum(1 for q in recent_queries if q.get("is_winner", False))
            recent_win_rate = wins / len(recent_queries)

            if recent_win_rate < self.demotion_threshold:
                # Poor recent performance - consider demotion
                demotion = self._execute_demotion(
                    model=model,
                    recent_win_rate=recent_win_rate,
                )
                demotions.append(demotion)

        return demotions

    def _execute_demotion(
        self,
        model: Any,
        recent_win_rate: float,
    ) -> Dict[str, Any]:
        """Execute a model demotion."""
        from_role = model.role

        # Demote by one level
        if from_role == ModelRole.TEACHER:
            model.role = ModelRole.TA
            to_role = ModelRole.TA
        elif from_role == ModelRole.TA:
            model.role = ModelRole.STUDENT
            to_role = ModelRole.STUDENT
        else:
            # Can't demote students further
            return {}

        reason = (
            f"Poor recent performance: "
            f"{recent_win_rate:.3f} win rate over last {self.demotion_window} queries "
            f"(threshold: {self.demotion_threshold})"
        )

        demotion_event = {
            "model_id": model.model_id,
            "from_role": from_role.value,
            "to_role": to_role.value,
            "reason": reason,
            "recent_win_rate": recent_win_rate,
        }

        logger.warning(
            f"DEMOTION: {model.model_id} from {from_role.value} to {to_role.value} "
            f"(recent win rate: {recent_win_rate:.3f})"
        )

        return demotion_event

    def get_promotion_statistics(self) -> Dict[str, Any]:
        """Get statistics about promotions."""
        total_promotions = len(self.promotion_history)

        # Count by type
        student_to_ta = sum(
            1 for p in self.promotion_history
            if p["from_role"] == "student" and p["to_role"] == "ta"
        )

        ta_to_teacher = sum(
            1 for p in self.promotion_history
            if p["from_role"] == "ta" and p["to_role"] == "teacher"
        )

        return {
            "total_promotions": total_promotions,
            "student_to_ta": student_to_ta,
            "ta_to_teacher": ta_to_teacher,
            "promotion_history": self.promotion_history,
            "criteria": {
                "student_to_ta": {
                    "min_queries": self.student_to_ta_criteria.min_queries,
                    "min_confidence": self.student_to_ta_criteria.min_confidence,
                    "min_win_rate": self.student_to_ta_criteria.min_win_rate,
                },
                "ta_to_teacher": {
                    "min_queries": self.ta_to_teacher_criteria.min_queries,
                    "min_confidence": self.ta_to_teacher_criteria.min_confidence,
                    "min_win_rate": self.ta_to_teacher_criteria.min_win_rate,
                },
            },
        }
