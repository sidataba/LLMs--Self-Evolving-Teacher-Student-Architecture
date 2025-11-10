"""
OpenAI LLM integration for GPT models.

Supports GPT-4, GPT-3.5-turbo, and other OpenAI models with proper
token counting, cost tracking, and error handling.
"""

import logging
from typing import Optional, Dict, Any
import tiktoken

from src.models.real_llm_base import RealLLMBase, LLMProviderConfig

logger = logging.getLogger(__name__)

# Try to import OpenAI client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI package not installed. Install with: pip install openai")
    OPENAI_AVAILABLE = False


class OpenAIModel(RealLLMBase):
    """
    OpenAI GPT model integration.

    Supports:
    - GPT-4 (gpt-4, gpt-4-turbo)
    - GPT-3.5 (gpt-3.5-turbo)
    - Accurate token counting with tiktoken
    - Cost tracking with current API pricing
    """

    # Current OpenAI pricing (as of 2024)
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},  # USD per 1K tokens
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
    }

    def __init__(
        self,
        model_id: str,
        model_name: str = "gpt-4",
        domain: Optional[str] = None,
        base_confidence: float = 0.90,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        """
        Initialize OpenAI model.

        Args:
            model_id: Unique identifier for this model instance
            model_name: OpenAI model name (gpt-4, gpt-3.5-turbo, etc.)
            domain: Optional domain specialization
            base_confidence: Base confidence score
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-2)
        """
        # Get pricing for this model
        pricing = self._get_pricing(model_name)

        # Create configuration
        config = LLMProviderConfig(
            provider_name="openai",
            model_name=model_name,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            cost_per_1k_input=pricing["input"],
            cost_per_1k_output=pricing["output"],
        )

        super().__init__(model_id, config, domain, base_confidence)

        # Initialize OpenAI client
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            if not OPENAI_AVAILABLE:
                logger.error("OpenAI package not installed")
            if not self.api_key:
                logger.error("OpenAI API key not provided")

        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.warning(f"Unknown model {model_name}, using cl100k_base tokenizer")

    def _get_pricing(self, model_name: str) -> Dict[str, float]:
        """Get pricing for model."""
        # Try exact match
        if model_name in self.PRICING:
            return self.PRICING[model_name]

        # Try partial match
        for key in self.PRICING:
            if key in model_name:
                return self.PRICING[key]

        # Default to GPT-4 pricing (conservative estimate)
        logger.warning(f"Unknown model {model_name}, using GPT-4 pricing")
        return self.PRICING["gpt-4"]

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback: rough estimate (1 token ≈ 4 chars)
            return len(text) // 4

    def _call_api(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call OpenAI API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Dict with 'response', 'input_tokens', 'output_tokens'
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized (check API key)")

        # Build messages
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        # Count input tokens
        input_tokens = sum(self._count_tokens(msg["content"]) for msg in messages)

        # Make API call
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout,
                **kwargs
            )

            # Extract response
            response_text = response.choices[0].message.content

            # Get token counts from API response (most accurate)
            if hasattr(response, 'usage'):
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
            else:
                # Fallback: count manually
                output_tokens = self._count_tokens(response_text)

            return {
                'response': response_text,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'finish_reason': response.choices[0].finish_reason,
                'model': response.model,
            }

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def _extract_confidence(self, response_text: str) -> float:
        """
        Extract confidence from OpenAI response.

        OpenAI doesn't provide confidence scores, so we use heuristics.
        """
        confidence = super()._extract_confidence(response_text)

        # Adjust based on model capability
        if "gpt-4" in self.model_name:
            confidence = min(0.95, confidence * 1.05)  # GPT-4 is more reliable
        elif "gpt-3.5" in self.model_name:
            confidence = min(0.85, confidence * 0.95)  # GPT-3.5 slightly less reliable

        return confidence


def create_gpt4_model(
    model_id: str,
    domain: Optional[str] = None,
    api_key: Optional[str] = None,
) -> OpenAIModel:
    """
    Convenience function to create GPT-4 model.

    Args:
        model_id: Unique model identifier
        domain: Optional domain specialization
        api_key: Optional API key

    Returns:
        OpenAIModel configured for GPT-4
    """
    return OpenAIModel(
        model_id=model_id,
        model_name="gpt-4",
        domain=domain,
        base_confidence=0.92,
        api_key=api_key,
        temperature=0.7,
    )


def create_gpt4_turbo_model(
    model_id: str,
    domain: Optional[str] = None,
    api_key: Optional[str] = None,
) -> OpenAIModel:
    """
    Convenience function to create GPT-4 Turbo model.

    Args:
        model_id: Unique model identifier
        domain: Optional domain specialization
        api_key: Optional API key

    Returns:
        OpenAIModel configured for GPT-4 Turbo
    """
    return OpenAIModel(
        model_id=model_id,
        model_name="gpt-4-turbo-preview",
        domain=domain,
        base_confidence=0.92,
        api_key=api_key,
        temperature=0.7,
    )


def create_gpt35_turbo_model(
    model_id: str,
    domain: Optional[str] = None,
    api_key: Optional[str] = None,
) -> OpenAIModel:
    """
    Convenience function to create GPT-3.5-turbo model.

    Args:
        model_id: Unique model identifier
        domain: Optional domain specialization
        api_key: Optional API key

    Returns:
        OpenAIModel configured for GPT-3.5-turbo
    """
    return OpenAIModel(
        model_id=model_id,
        model_name="gpt-3.5-turbo",
        domain=domain,
        base_confidence=0.82,
        api_key=api_key,
        temperature=0.7,
    )
