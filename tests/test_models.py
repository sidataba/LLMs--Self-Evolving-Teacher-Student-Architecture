"""Tests for model implementations."""

import pytest
from src.models.base import ModelConfig, ModelRole
from src.models.supervisor import SupervisorModel
from src.models.teacher import TeacherModel
from src.models.student import StudentModel


class TestSupervisorModel:
    """Tests for SupervisorModel."""

    def test_initialization(self):
        """Test supervisor initialization."""
        config = ModelConfig(
            model_id="test-supervisor",
            model_type="mock",
            role=ModelRole.SUPERVISOR,
        )

        supervisor = SupervisorModel(config)

        assert supervisor.model_id == "test-supervisor"
        assert supervisor.role == ModelRole.SUPERVISOR

    def test_generate_response(self):
        """Test response generation."""
        config = ModelConfig(
            model_id="test-supervisor",
            model_type="mock",
            role=ModelRole.SUPERVISOR,
        )

        supervisor = SupervisorModel(config)
        response = supervisor.generate_response("What is AI?")

        assert response.model_id == "test-supervisor"
        assert response.response_text
        assert 0.0 <= response.confidence <= 1.0

    def test_evaluate_multiple_responses(self):
        """Test evaluation of multiple responses."""
        config = ModelConfig(
            model_id="test-supervisor",
            model_type="mock",
            role=ModelRole.SUPERVISOR,
        )

        supervisor = SupervisorModel(config)

        # Create mock responses
        response1 = supervisor.generate_response("What is math?")
        response2 = supervisor.generate_response("What is math?")

        # Evaluate
        result = supervisor.evaluate_multiple_responses(
            "What is math?",
            [response1, response2],
        )

        assert result["winner"] is not None
        assert len(result["evaluations"]) == 2
        assert result["reasoning"]


class TestTeacherModel:
    """Tests for TeacherModel."""

    def test_initialization(self):
        """Test teacher initialization."""
        config = ModelConfig(
            model_id="test-teacher",
            model_type="mock",
            role=ModelRole.TEACHER,
            domain="mathematics",
        )

        teacher = TeacherModel(config)

        assert teacher.model_id == "test-teacher"
        assert teacher.role == ModelRole.TEACHER
        assert teacher.domain == "mathematics"

    def test_add_student(self):
        """Test adding students."""
        config = ModelConfig(
            model_id="test-teacher",
            model_type="mock",
            role=ModelRole.TEACHER,
        )

        teacher = TeacherModel(config)

        teacher.add_student("student-1")
        teacher.add_student("student-2")

        assert len(teacher.students) == 2
        assert "student-1" in teacher.students


class TestStudentModel:
    """Tests for StudentModel."""

    def test_initialization(self):
        """Test student initialization."""
        config = ModelConfig(
            model_id="test-student",
            model_type="mock",
            role=ModelRole.STUDENT,
            domain="mathematics",
            metadata={"teacher_id": "teacher-1"},
        )

        student = StudentModel(config)

        assert student.model_id == "test-student"
        assert student.role == ModelRole.STUDENT
        assert student.teacher_id == "teacher-1"

    def test_promotion_to_ta(self):
        """Test promotion to TA."""
        config = ModelConfig(
            model_id="test-student",
            model_type="mock",
            role=ModelRole.STUDENT,
        )

        student = StudentModel(config)
        student.promote_to_ta()

        assert student.role == ModelRole.TA

    def test_promotion_to_teacher(self):
        """Test promotion to teacher."""
        config = ModelConfig(
            model_id="test-student",
            model_type="mock",
            role=ModelRole.STUDENT,
        )

        student = StudentModel(config)
        student.promote_to_ta()
        student.promote_to_teacher()

        assert student.role == ModelRole.TEACHER

    def test_receive_feedback(self):
        """Test feedback reception."""
        config = ModelConfig(
            model_id="test-student",
            model_type="mock",
            role=ModelRole.STUDENT,
        )

        student = StudentModel(config)
        initial_confidence = student.base_confidence

        feedback = {
            "your_metrics": {"correctness": 0.5},
            "winner_metrics": {"correctness": 0.9},
        }

        student.receive_feedback(
            query="Test query",
            your_response="Test response",
            winning_response="Better response",
            evaluation=feedback,
        )

        # Confidence should improve after feedback
        assert student.feedback_received == 1
        assert len(student.learning_history) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
