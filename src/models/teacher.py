"""Teacher model implementation - specialized domain experts."""

from typing import Dict, Any, Optional
from src.models.base import ModelConfig, ModelRole
from src.models.mock_model import MockLLM
from loguru import logger


class TeacherModel(MockLLM):
    """
    Teacher model - specialized expert in a specific domain.
    Provides high-quality responses and mentors student models.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize teacher model.

        Args:
            config: Model configuration
        """
        # Ensure teacher role
        config.role = ModelRole.TEACHER
        super().__init__(config)

        # Teacher-specific attributes
        self.students: list = []  # Student models under this teacher
        self.mentoring_stats = {
            "total_feedback_given": 0,
            "students_promoted": 0,
        }

        logger.info(
            f"Initialized Teacher Model: {self.model_id} "
            f"(domain: {self.domain}, specialization: {self.specialization})"
        )

    def add_student(self, student_id: str) -> None:
        """
        Add a student model to this teacher's mentorship.

        Args:
            student_id: ID of the student model
        """
        if student_id not in self.students:
            self.students.append(student_id)
            logger.info(f"Teacher {self.model_id} now mentoring student {student_id}")

    def remove_student(self, student_id: str) -> None:
        """
        Remove a student from this teacher's mentorship.

        Args:
            student_id: ID of the student model
        """
        if student_id in self.students:
            self.students.remove(student_id)
            logger.info(f"Student {student_id} removed from teacher {self.model_id}")

    def provide_mentorship_feedback(
        self,
        student_id: str,
        query: str,
        student_response: str,
        evaluation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Provide detailed feedback to a student model.

        Args:
            student_id: ID of the student receiving feedback
            query: The original query
            student_response: Student's response
            evaluation: Evaluation metrics

        Returns:
            Dictionary with feedback and guidance
        """
        # Generate teacher's own response for comparison
        teacher_response = self.generate_response(query)

        # Analyze student performance
        student_metrics = evaluation.get("metrics", {})

        # Generate feedback
        feedback = {
            "student_id": student_id,
            "teacher_id": self.model_id,
            "query": query,
            "student_response": student_response,
            "teacher_response": teacher_response.response_text,
            "evaluation": evaluation,
            "guidance": self._generate_guidance(student_metrics),
            "areas_for_improvement": self._identify_improvement_areas(student_metrics),
        }

        self.mentoring_stats["total_feedback_given"] += 1

        logger.debug(
            f"Teacher {self.model_id} provided feedback to student {student_id}"
        )

        return feedback

    def _generate_guidance(self, metrics: Dict[str, float]) -> str:
        """Generate guidance based on student metrics."""
        guidance_parts = []

        if metrics.get("correctness", 0) < 0.7:
            guidance_parts.append(
                "Focus on accuracy - verify your reasoning before responding."
            )

        if metrics.get("completeness", 0) < 0.7:
            guidance_parts.append(
                "Provide more comprehensive answers - cover all aspects of the question."
            )

        if metrics.get("clarity", 0) < 0.7:
            guidance_parts.append(
                "Improve clarity - structure your response more clearly."
            )

        if not guidance_parts:
            guidance_parts.append(
                "Good work! Continue refining your approach."
            )

        return " ".join(guidance_parts)

    def _identify_improvement_areas(self, metrics: Dict[str, float]) -> list:
        """Identify specific areas where student needs improvement."""
        areas = []

        threshold = 0.75

        for metric, score in metrics.items():
            if score < threshold:
                areas.append({
                    "metric": metric,
                    "current_score": score,
                    "target_score": threshold,
                    "gap": threshold - score,
                })

        return sorted(areas, key=lambda x: x["gap"], reverse=True)

    def get_mentoring_statistics(self) -> Dict[str, Any]:
        """Get statistics about this teacher's mentoring activities."""
        return {
            "teacher_id": self.model_id,
            "domain": self.domain,
            "specialization": self.specialization,
            "active_students": len(self.students),
            "student_ids": self.students,
            **self.mentoring_stats,
            **self.get_statistics(),
        }
