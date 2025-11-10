"""Base model interface for all LLM models in the system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime


class ModelRole(Enum):
    """Enumeration of possible model roles in the system."""
    SUPERVISOR = "supervisor"
    TEACHER = "teacher"
    TA = "ta"  # Teaching Assistant
    STUDENT = "student"


@dataclass
class ModelResponse:
    """Structured response from a model."""
    model_id: str
    response_text: str
    confidence: float  # 0.0 to 1.0
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ModelConfig:
    """Configuration for a model."""
    model_id: str
    model_type: str
    role: ModelRole
    domain: Optional[str] = None
    specialization: List[str] = field(default_factory=list)
    max_tokens: int = 2048
    temperature: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """
    Abstract base class for all models in the teacher-student architecture.
    Defines the interface that all models must implement.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the base model.

        Args:
            config: Model configuration
        """
        self.config = config
        self.model_id = config.model_id
        self.role = config.role
        self.domain = config.domain
        self.specialization = config.specialization

        # Performance tracking
        self.total_queries = 0
        self.total_wins = 0
        self.confidence_scores: List[float] = []

    @abstractmethod
    def generate_response(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        """
        Generate a response to a query.

        Args:
            query: The user query
            context: Optional context information

        Returns:
            ModelResponse with the generated response
        """
        pass

    @abstractmethod
    def evaluate_response(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a response (used primarily by supervisor models).

        Args:
            query: The original query
            response: The response to evaluate
            context: Optional context

        Returns:
            Dictionary of evaluation metrics
        """
        pass

    def update_stats(self, confidence: float, is_winner: bool = False) -> None:
        """
        Update model performance statistics.

        Args:
            confidence: Confidence score for the response
            is_winner: Whether this response was selected as best
        """
        self.total_queries += 1
        self.confidence_scores.append(confidence)

        if is_winner:
            self.total_wins += 1

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        avg_confidence = (
            sum(self.confidence_scores) / len(self.confidence_scores)
            if self.confidence_scores
            else 0.0
        )

        win_rate = (
            self.total_wins / self.total_queries
            if self.total_queries > 0
            else 0.0
        )

        return {
            "model_id": self.model_id,
            "role": self.role.value,
            "domain": self.domain,
            "total_queries": self.total_queries,
            "total_wins": self.total_wins,
            "avg_confidence": avg_confidence,
            "win_rate": win_rate,
        }

    def receive_feedback(
        self,
        query: str,
        your_response: str,
        winning_response: str,
        evaluation: Dict[str, Any],
    ) -> None:
        """
        Receive feedback on a response (used for learning).

        Args:
            query: The original query
            your_response: This model's response
            winning_response: The selected best response
            evaluation: Evaluation metrics and feedback
        """
        # Base implementation does nothing
        # Subclasses can override for learning/fine-tuning
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.model_id}, role={self.role.value})"
