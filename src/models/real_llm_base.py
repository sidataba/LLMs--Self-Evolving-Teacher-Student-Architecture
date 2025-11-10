"""
Base class for real LLM provider integrations.

This module provides a common interface for integrating with various LLM APIs
(OpenAI, Anthropic, Cohere, etc.) with proper error handling, rate limiting,
and cost tracking.
"""

import time
import logging
from typing import Optional, Dict, Any, List
from abc import abstractmethod
from dataclasses import dataclass
import os

from src.models.base import BaseModel, ModelResponse

logger = logging.getLogger(__name__)


@dataclass
class LLMProviderConfig:
    """Configuration for LLM provider."""
    provider_name: str  # "openai", "anthropic", "cohere", etc.
    model_name: str  # "gpt-4", "claude-3-opus", etc.
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout: int = 30
    max_retries: int = 3
    cost_per_1k_input: float = 0.0  # USD per 1K input tokens
    cost_per_1k_output: float = 0.0  # USD per 1K output tokens


class RealLLMBase(BaseModel):
    """
    Base class for real LLM provider integrations.

    Provides common functionality:
    - API key management
    - Rate limiting and retries
    - Cost tracking
    - Error handling
    - Token counting
    """

    def __init__(
        self,
        model_id: str,
        config: LLMProviderConfig,
        domain: Optional[str] = None,
        base_confidence: float = 0.85,
    ):
        """
        Initialize real LLM model.

        Args:
            model_id: Unique identifier for this model instance
            config: Provider-specific configuration
            domain: Optional domain specialization
            base_confidence: Base confidence score
        """
        super().__init__(model_id, domain, base_confidence)

        self.config = config
        self.provider_name = config.provider_name
        self.model_name = config.model_name

        # API key handling
        self.api_key = config.api_key or self._get_api_key_from_env()
        if not self.api_key:
            logger.warning(f"No API key found for {self.provider_name}")

        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Minimum seconds between requests

        logger.info(
            f"Initialized {self.provider_name} model: {self.model_name} "
            f"(id={model_id}, domain={domain})"
        )

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables."""
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "cohere": "COHERE_API_KEY",
        }

        env_var = env_var_map.get(self.provider_name.lower())
        if env_var:
            return os.getenv(env_var)
        return None

    def _enforce_rate_limit(self):
        """Enforce minimum time between API requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _update_cost_tracking(self, input_tokens: int, output_tokens: int):
        """Update token usage and cost statistics."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        input_cost = (input_tokens / 1000) * self.config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.config.cost_per_1k_output
        request_cost = input_cost + output_cost

        self.total_cost += request_cost

        logger.debug(
            f"Request cost: ${request_cost:.6f} "
            f"({input_tokens} in, {output_tokens} out)"
        )

        return request_cost

    def get_cost_statistics(self) -> Dict[str, Any]:
        """Get detailed cost statistics."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": self.total_cost,
            "avg_cost_per_query": (
                self.total_cost / self.query_count if self.query_count > 0 else 0
            ),
            "cost_per_1k_input": self.config.cost_per_1k_input,
            "cost_per_1k_output": self.config.cost_per_1k_output,
        }

    @abstractmethod
    def _call_api(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make API call to LLM provider.

        Must be implemented by subclasses for specific providers.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict with keys: 'response', 'input_tokens', 'output_tokens'
        """
        pass

    def _call_api_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call API with exponential backoff retry logic.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            API response dictionary

        Raises:
            Exception: If all retries fail
        """
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                # Enforce rate limiting
                self._enforce_rate_limit()

                # Make API call
                result = self._call_api(prompt, system_prompt, **kwargs)

                # Update cost tracking
                self._update_cost_tracking(
                    result.get('input_tokens', 0),
                    result.get('output_tokens', 0)
                )

                return result

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )

                if attempt < self.config.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s, ...
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)

        # All retries failed
        logger.error(f"All {self.config.max_retries} API call attempts failed")
        raise last_exception

    def generate_response(
        self,
        query_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ModelResponse:
        """
        Generate response using real LLM API.

        Args:
            query_text: User query
            context: Optional context (domain, history, etc.)

        Returns:
            ModelResponse with LLM-generated content
        """
        # Increment query count
        self.query_count += 1

        # Build system prompt based on role and domain
        system_prompt = self._build_system_prompt(context)

        # Build user prompt with context
        full_prompt = self._build_full_prompt(query_text, context)

        try:
            # Call API with retry logic
            result = self._call_api_with_retry(
                prompt=full_prompt,
                system_prompt=system_prompt,
            )

            response_text = result.get('response', '')

            # Extract confidence from response or use base confidence
            confidence = self._extract_confidence(response_text)

            # Generate reasoning explanation
            reasoning = self._generate_reasoning(query_text, response_text, context)

            return ModelResponse(
                model_id=self.model_id,
                response_text=response_text,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    'provider': self.provider_name,
                    'model': self.model_name,
                    'input_tokens': result.get('input_tokens', 0),
                    'output_tokens': result.get('output_tokens', 0),
                    'cost_usd': result.get('cost', 0),
                }
            )

        except Exception as e:
            logger.error(f"Failed to generate response: {e}")

            # Return fallback response
            return ModelResponse(
                model_id=self.model_id,
                response_text=f"Error: Unable to generate response ({str(e)})",
                confidence=0.0,
                reasoning="API call failed",
                metadata={'error': str(e)}
            )

    def _build_system_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Build system prompt based on role and domain."""
        role_prompts = {
            "supervisor": "You are a supervisor AI responsible for high-quality, accurate responses. Provide comprehensive and well-reasoned answers.",
            "teacher": "You are a teacher AI specializing in providing clear, accurate explanations. Focus on educational value and correctness.",
            "ta": "You are a teaching assistant AI. Provide helpful, accurate responses while learning from feedback.",
            "student": "You are a student AI learning to provide quality responses. Do your best to answer accurately.",
        }

        base_prompt = role_prompts.get(self.role.value, "You are a helpful AI assistant.")

        if self.domain:
            base_prompt += f" You specialize in {self.domain}."

        return base_prompt

    def _build_full_prompt(
        self,
        query_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build full prompt with context."""
        prompt_parts = [query_text]

        if context:
            if 'domain' in context:
                prompt_parts.insert(0, f"[Domain: {context['domain']}]")

            if 'history' in context and context['history']:
                history_text = "\n".join([
                    f"Q: {h['query']}\nA: {h['response']}"
                    for h in context['history'][-3:]  # Last 3 exchanges
                ])
                prompt_parts.insert(0, f"Previous context:\n{history_text}\n")

        return "\n".join(prompt_parts)

    def _extract_confidence(self, response_text: str) -> float:
        """
        Extract confidence score from response.

        Can be overridden by subclasses for provider-specific confidence extraction.
        """
        # Default: use base confidence adjusted by response quality indicators
        confidence = self.base_confidence

        # Adjust based on response characteristics
        if len(response_text) < 20:
            confidence *= 0.7  # Very short responses
        elif "I don't know" in response_text or "I'm not sure" in response_text:
            confidence *= 0.6  # Uncertain responses
        elif any(word in response_text.lower() for word in ["error", "unable", "cannot"]):
            confidence *= 0.5  # Error responses

        return min(0.99, max(0.1, confidence))

    def _generate_reasoning(
        self,
        query_text: str,
        response_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate reasoning explanation for the response."""
        reasoning_parts = [
            f"Model: {self.model_name}",
            f"Role: {self.role.value}",
        ]

        if self.domain:
            reasoning_parts.append(f"Domain: {self.domain}")

        reasoning_parts.append(f"Query length: {len(query_text)} chars")
        reasoning_parts.append(f"Response length: {len(response_text)} chars")

        return " | ".join(reasoning_parts)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics including cost data."""
        base_stats = super().get_statistics()
        cost_stats = self.get_cost_statistics()

        return {
            **base_stats,
            **cost_stats,
            'provider': self.provider_name,
            'model': self.model_name,
        }
