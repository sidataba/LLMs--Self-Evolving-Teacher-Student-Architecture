"""
Anthropic Claude LLM integration.

Supports Claude 3 family (Opus, Sonnet, Haiku) with proper token counting,
cost tracking, and confidence extraction.
"""

import logging
from typing import Optional, Dict, Any

from src.models.real_llm_base import RealLLMBase, LLMProviderConfig

logger = logging.getLogger(__name__)

# Try to import Anthropic client
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    logger.warning("Anthropic package not installed. Install with: pip install anthropic")
    ANTHROPIC_AVAILABLE = False


class AnthropicModel(RealLLMBase):
    """
    Anthropic Claude model integration.

    Supports:
    - Claude 3 Opus (highest capability)
    - Claude 3 Sonnet (balanced)
    - Claude 3 Haiku (fastest, cheapest)
    - Accurate token counting
    - Cost tracking with current API pricing
    """

    # Current Anthropic pricing (as of 2024)
    PRICING = {
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},  # USD per 1K tokens
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        # Aliases
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }

    def __init__(
        self,
        model_id: str,
        model_name: str = "claude-3-sonnet-20240229",
        domain: Optional[str] = None,
        base_confidence: float = 0.90,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        """
        Initialize Anthropic Claude model.

        Args:
            model_id: Unique identifier for this model instance
            model_name: Claude model name (claude-3-opus, claude-3-sonnet, etc.)
            domain: Optional domain specialization
            base_confidence: Base confidence score
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
        """
        # Get pricing for this model
        pricing = self._get_pricing(model_name)

        # Create configuration
        config = LLMProviderConfig(
            provider_name="anthropic",
            model_name=model_name,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=min(1.0, temperature),  # Claude uses 0-1 range
            cost_per_1k_input=pricing["input"],
            cost_per_1k_output=pricing["output"],
        )

        super().__init__(model_id, config, domain, base_confidence)

        # Initialize Anthropic client
        if ANTHROPIC_AVAILABLE and self.api_key:
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None
            if not ANTHROPIC_AVAILABLE:
                logger.error("Anthropic package not installed")
            if not self.api_key:
                logger.error("Anthropic API key not provided")

    def _get_pricing(self, model_name: str) -> Dict[str, float]:
        """Get pricing for model."""
        # Try exact match
        if model_name in self.PRICING:
            return self.PRICING[model_name]

        # Try partial match
        for key in self.PRICING:
            if key in model_name:
                return self.PRICING[key]

        # Default to Sonnet pricing (middle tier)
        logger.warning(f"Unknown model {model_name}, using Claude 3 Sonnet pricing")
        return self.PRICING["claude-3-sonnet"]

    def _count_tokens_approximate(self, text: str) -> int:
        """
        Approximate token count for Claude.

        Claude uses similar tokenization to GPT models.
        Rough estimate: 1 token ≈ 3.5 characters
        """
        return len(text) // 3.5

    def _call_api(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call Anthropic API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Dict with 'response', 'input_tokens', 'output_tokens'
        """
        if not self.client:
            raise RuntimeError("Anthropic client not initialized (check API key)")

        # Build messages (Claude uses messages API)
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Prepare API call parameters
        api_params = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "timeout": self.config.timeout,
        }

        # Add system prompt if provided
        if system_prompt:
            api_params["system"] = system_prompt

        # Add any additional parameters
        api_params.update(kwargs)

        # Make API call
        try:
            response = self.client.messages.create(**api_params)

            # Extract response text
            response_text = ""
            for block in response.content:
                if block.type == "text":
                    response_text += block.text

            # Get token counts from API response
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            return {
                'response': response_text,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'stop_reason': response.stop_reason,
                'model': response.model,
            }

        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise

    def _extract_confidence(self, response_text: str) -> float:
        """
        Extract confidence from Claude response.

        Claude doesn't provide explicit confidence scores, but we can use
        response quality indicators.
        """
        confidence = super()._extract_confidence(response_text)

        # Adjust based on model capability
        if "opus" in self.model_name.lower():
            confidence = min(0.95, confidence * 1.08)  # Opus is most capable
        elif "sonnet" in self.model_name.lower():
            confidence = min(0.92, confidence * 1.03)  # Sonnet is balanced
        elif "haiku" in self.model_name.lower():
            confidence = min(0.88, confidence * 0.98)  # Haiku is fastest but less capable

        # Claude often includes confidence expressions
        confidence_indicators = {
            "I'm confident": 1.1,
            "I'm certain": 1.15,
            "definitely": 1.05,
            "I think": 0.9,
            "I believe": 0.9,
            "possibly": 0.8,
            "might": 0.85,
            "uncertain": 0.7,
        }

        response_lower = response_text.lower()
        for indicator, multiplier in confidence_indicators.items():
            if indicator in response_lower:
                confidence *= multiplier
                break

        return min(0.99, max(0.1, confidence))


def create_claude_opus_model(
    model_id: str,
    domain: Optional[str] = None,
    api_key: Optional[str] = None,
) -> AnthropicModel:
    """
    Convenience function to create Claude 3 Opus model.

    Args:
        model_id: Unique model identifier
        domain: Optional domain specialization
        api_key: Optional API key

    Returns:
        AnthropicModel configured for Claude 3 Opus
    """
    return AnthropicModel(
        model_id=model_id,
        model_name="claude-3-opus-20240229",
        domain=domain,
        base_confidence=0.94,
        api_key=api_key,
        temperature=0.7,
    )


def create_claude_sonnet_model(
    model_id: str,
    domain: Optional[str] = None,
    api_key: Optional[str] = None,
) -> AnthropicModel:
    """
    Convenience function to create Claude 3 Sonnet model.

    Args:
        model_id: Unique model identifier
        domain: Optional domain specialization
        api_key: Optional API key

    Returns:
        AnthropicModel configured for Claude 3 Sonnet
    """
    return AnthropicModel(
        model_id=model_id,
        model_name="claude-3-sonnet-20240229",
        domain=domain,
        base_confidence=0.91,
        api_key=api_key,
        temperature=0.7,
    )


def create_claude_haiku_model(
    model_id: str,
    domain: Optional[str] = None,
    api_key: Optional[str] = None,
) -> AnthropicModel:
    """
    Convenience function to create Claude 3 Haiku model.

    Args:
        model_id: Unique model identifier
        domain: Optional domain specialization
        api_key: Optional API key

    Returns:
        AnthropicModel configured for Claude 3 Haiku
    """
    return AnthropicModel(
        model_id=model_id,
        model_name="claude-3-haiku-20240307",
        domain=domain,
        base_confidence=0.87,
        api_key=api_key,
        temperature=0.7,
    )
