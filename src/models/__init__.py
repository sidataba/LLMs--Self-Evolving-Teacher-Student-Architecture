"""Model implementations for the teacher-student architecture."""

from src.models.base import BaseModel, ModelRole
from src.models.supervisor import SupervisorModel
from src.models.teacher import TeacherModel
from src.models.student import StudentModel
from src.models.mock_model import MockLLM

__all__ = [
    "BaseModel",
    "ModelRole",
    "SupervisorModel",
    "TeacherModel",
    "StudentModel",
    "MockLLM",
]
