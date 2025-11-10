"""
Self-Evolving Teacher-Student Architecture for LLMs
A modular, self-improving architecture for deploying LLMs in a cost-efficient and scalable manner.
"""

__version__ = "0.1.0"
__author__ = "Nguyen Trung Hieu"
__email__ = "hieuhip4444@gmail.com"

from src.core.orchestrator import Orchestrator
from src.models.supervisor import SupervisorModel
from src.models.teacher import TeacherModel
from src.models.student import StudentModel

__all__ = [
    "Orchestrator",
    "SupervisorModel",
    "TeacherModel",
    "StudentModel",
]
