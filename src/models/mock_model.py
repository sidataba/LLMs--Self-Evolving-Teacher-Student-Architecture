"""Mock LLM implementation for demo and testing purposes."""

import random
from typing import Dict, Any, Optional
from src.models.base import BaseModel, ModelResponse, ModelConfig
from loguru import logger


class MockLLM(BaseModel):
    """
    Mock LLM model for demonstration purposes.
    Simulates responses with varying quality based on domain expertise.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize mock model.

        Args:
            config: Model configuration
        """
        super().__init__(config)

        # Domain expertise affects response quality
        self.base_confidence = self._calculate_base_confidence()

        # Knowledge base for different domains
        self.knowledge_base = self._initialize_knowledge_base()

    def _calculate_base_confidence(self) -> float:
        """Calculate base confidence based on role and domain."""
        role_confidence = {
            "supervisor": 0.85,
            "teacher": 0.75,
            "ta": 0.65,
            "student": 0.50,
        }
        return role_confidence.get(self.role.value, 0.5)

    def _initialize_knowledge_base(self) -> Dict[str, Dict[str, str]]:
        """Initialize domain-specific knowledge base."""
        return {
            "mathematics": {
                "algebra": "I can help with algebraic equations, polynomials, and factoring.",
                "calculus": "I can explain derivatives, integrals, and limits.",
                "geometry": "I can help with geometric shapes, angles, and theorems.",
                "default": "I can help with various mathematical concepts.",
            },
            "science": {
                "physics": "I can explain concepts in mechanics, thermodynamics, and electromagnetism.",
                "chemistry": "I can help with chemical reactions, periodic table, and molecular structures.",
                "biology": "I can explain cellular biology, genetics, and ecosystems.",
                "default": "I can help with scientific concepts.",
            },
            "programming": {
                "python": "I can help with Python programming, libraries, and best practices.",
                "javascript": "I can assist with JavaScript, web development, and frameworks.",
                "algorithms": "I can explain data structures and algorithmic concepts.",
                "default": "I can help with programming concepts.",
            },
            "general": {
                "default": "I'll do my best to answer your question.",
            },
        }

    def generate_response(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        """
        Generate a mock response to a query.

        Args:
            query: The user query
            context: Optional context information

        Returns:
            ModelResponse with the generated response
        """
        # Detect query domain
        query_domain = self._detect_domain(query)

        # Calculate confidence based on domain match
        confidence = self._calculate_confidence(query_domain)

        # Generate response
        response_text = self._generate_response_text(query, query_domain)

        # Generate reasoning
        reasoning = self._generate_reasoning(query, query_domain, confidence)

        response = ModelResponse(
            model_id=self.model_id,
            response_text=response_text,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "role": self.role.value,
                "domain": self.domain,
                "query_domain": query_domain,
            },
        )

        self.update_stats(confidence)

        logger.debug(
            f"{self.model_id} ({self.role.value}) generated response "
            f"with confidence {confidence:.2f}"
        )

        return response

    def _detect_domain(self, query: str) -> str:
        """Detect the domain of a query based on keywords."""
        query_lower = query.lower()

        # Math keywords
        math_keywords = ["equation", "solve", "calculate", "math", "number", "algebra",
                        "calculus", "integral", "derivative", "geometry"]
        if any(kw in query_lower for kw in math_keywords):
            return "mathematics"

        # Science keywords
        science_keywords = ["physics", "chemistry", "biology", "atom", "molecule",
                           "force", "energy", "cell", "experiment"]
        if any(kw in query_lower for kw in science_keywords):
            return "science"

        # Programming keywords
        programming_keywords = ["code", "program", "python", "javascript", "function",
                               "algorithm", "debug", "compile", "syntax"]
        if any(kw in query_lower for kw in programming_keywords):
            return "programming"

        return "general"

    def _calculate_confidence(self, query_domain: str) -> float:
        """Calculate confidence based on domain expertise."""
        # Higher confidence if query domain matches model domain
        if query_domain == self.domain:
            # Domain match - higher confidence
            confidence = self.base_confidence + random.uniform(0.05, 0.15)
        elif self.domain == "general" or query_domain == "general":
            # General domain - moderate confidence
            confidence = self.base_confidence + random.uniform(-0.05, 0.05)
        else:
            # Domain mismatch - lower confidence
            confidence = self.base_confidence - random.uniform(0.10, 0.25)

        # Clamp to valid range
        return max(0.1, min(1.0, confidence))

    def _generate_response_text(self, query: str, query_domain: str) -> str:
        """Generate response text based on query and domain."""
        # Get domain knowledge
        domain_knowledge = self.knowledge_base.get(
            self.domain or query_domain,
            self.knowledge_base["general"]
        )

        # Try to find specific knowledge
        for topic, knowledge in domain_knowledge.items():
            if topic.lower() in query.lower():
                response = knowledge
                break
        else:
            response = domain_knowledge.get("default", "I'll try to answer your question.")

        # Add role-specific prefix
        role_prefixes = {
            "supervisor": "[Supervisor Analysis] ",
            "teacher": "[Teacher Guidance] ",
            "ta": "[TA Support] ",
            "student": "[Student Attempt] ",
        }

        prefix = role_prefixes.get(self.role.value, "")

        # Construct full response
        full_response = (
            f"{prefix}{response}\n\n"
            f"Regarding your question: '{query}'\n\n"
        )

        # Add quality variation based on role
        if self.role.value == "supervisor":
            full_response += "After careful analysis, I recommend this approach..."
        elif self.role.value == "teacher":
            full_response += "Based on my expertise, here's the explanation..."
        elif self.role.value == "ta":
            full_response += "Let me help guide you through this..."
        else:  # student
            full_response += "I think the answer might be..."

        return full_response

    def _generate_reasoning(self, query: str, query_domain: str, confidence: float) -> str:
        """Generate reasoning explanation."""
        reasoning_parts = [
            f"Query domain detected: {query_domain}",
            f"My specialization: {self.domain or 'general'}",
            f"Role: {self.role.value}",
        ]

        if query_domain == self.domain:
            reasoning_parts.append("Domain match - high confidence in response")
        else:
            reasoning_parts.append("Domain mismatch - moderate confidence")

        reasoning_parts.append(f"Confidence score: {confidence:.2f}")

        return " | ".join(reasoning_parts)

    def evaluate_response(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a response (primarily for supervisor models).

        Args:
            query: The original query
            response: The response to evaluate
            context: Optional context

        Returns:
            Dictionary of evaluation metrics
        """
        # Simple heuristic-based evaluation
        metrics = {}

        # Relevance (based on length and keyword matching)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words) / len(query_words) if query_words else 0
        metrics["relevance"] = min(1.0, overlap + random.uniform(0.3, 0.5))

        # Correctness (simulated)
        metrics["correctness"] = random.uniform(0.6, 0.95) if len(response) > 50 else random.uniform(0.3, 0.6)

        # Completeness (based on response length)
        metrics["completeness"] = min(1.0, len(response) / 200 + random.uniform(0.2, 0.4))

        # Clarity (simulated)
        metrics["clarity"] = random.uniform(0.7, 0.95)

        return metrics

    def receive_feedback(
        self,
        query: str,
        your_response: str,
        winning_response: str,
        evaluation: Dict[str, Any],
    ) -> None:
        """
        Receive feedback and simulate learning.

        Args:
            query: The original query
            your_response: This model's response
            winning_response: The selected best response
            evaluation: Evaluation metrics and feedback
        """
        # For students, slightly improve base confidence over time
        if self.role.value == "student" and evaluation.get("your_metrics", {}).get("correctness", 0) < 0.8:
            # Small improvement through learning
            self.base_confidence = min(0.75, self.base_confidence + 0.01)

            logger.debug(
                f"{self.model_id} received feedback and improved "
                f"confidence to {self.base_confidence:.3f}"
            )
