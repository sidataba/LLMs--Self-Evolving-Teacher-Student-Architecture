"""
Real LLM Demo - Using OpenAI and Anthropic APIs

This demo shows how to use the self-evolving teacher-student architecture
with REAL LLM APIs (OpenAI GPT-4, GPT-3.5-turbo, Claude 3 family).

IMPORTANT:
1. Set your API keys in environment variables:
   export OPENAI_API_KEY=sk-...
   export ANTHROPIC_API_KEY=sk-ant-...

2. Or create a .env file (copy from .env.example)

3. This demo will make REAL API calls and incur costs!
   Estimated cost for this demo: ~$0.50-$1.00

WARNING: Real API calls will be made. Press Ctrl+C to cancel.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.models.openai_model import (
    create_gpt4_model,
    create_gpt35_turbo_model,
)
from src.models.anthropic_model import (
    create_claude_opus_model,
    create_claude_sonnet_model,
    create_claude_haiku_model,
)
from src.core.orchestrator import Orchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_api_keys():
    """Check if required API keys are set."""
    load_dotenv()  # Load from .env file

    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key or openai_key == "your_openai_key_here":
        logger.warning("⚠️  OPENAI_API_KEY not set. OpenAI models will not work.")
        logger.warning("   Get your key from: https://platform.openai.com/api-keys")

    if not anthropic_key or anthropic_key == "your_anthropic_key_here":
        logger.warning("⚠️  ANTHROPIC_API_KEY not set. Anthropic models will not work.")
        logger.warning("   Get your key from: https://console.anthropic.com/settings/keys")

    return bool(openai_key or anthropic_key)


def demo_simple_query():
    """Demo 1: Simple query with a single model."""
    print("\n" + "="*80)
    print("DEMO 1: Simple Query with Real LLM")
    print("="*80)

    # Create a GPT-3.5-turbo model (cheapest option)
    model = create_gpt35_turbo_model(
        model_id="demo-gpt35",
        domain="general"
    )

    # Query
    query = "What is the capital of France?"
    print(f"\nQuery: {query}")

    # Generate response
    print("Calling OpenAI API...")
    response = model.generate_response(query)

    print(f"\nResponse: {response.response_text}")
    print(f"Confidence: {response.confidence:.3f}")
    print(f"Model: {response.metadata.get('model')}")
    print(f"Tokens: {response.metadata.get('input_tokens')} in, {response.metadata.get('output_tokens')} out")
    print(f"Cost: ${response.metadata.get('cost_usd', 0):.6f}")

    # Get statistics
    stats = model.get_cost_statistics()
    print(f"\nTotal cost so far: ${stats['total_cost_usd']:.6f}")


def demo_model_comparison():
    """Demo 2: Compare responses from different models."""
    print("\n" + "="*80)
    print("DEMO 2: Model Comparison (GPT-4 vs Claude Sonnet)")
    print("="*80)

    # Create models
    gpt4 = create_gpt4_model(model_id="gpt4-demo")
    claude = create_claude_sonnet_model(model_id="claude-demo")

    # Query
    query = "Explain quantum entanglement in simple terms."
    print(f"\nQuery: {query}\n")

    # Get responses
    models = [
        ("GPT-4", gpt4),
        ("Claude 3 Sonnet", claude)
    ]

    for name, model in models:
        print(f"\n--- {name} ---")
        response = model.generate_response(query)
        print(f"Response: {response.response_text[:200]}...")
        print(f"Confidence: {response.confidence:.3f}")
        print(f"Tokens: {response.metadata.get('input_tokens', 0)} in, {response.metadata.get('output_tokens', 0)} out")
        print(f"Cost: ${response.metadata.get('cost_usd', 0):.6f}")


def demo_teacher_student_architecture():
    """Demo 3: Full teacher-student architecture with real LLMs."""
    print("\n" + "="*80)
    print("DEMO 3: Teacher-Student Architecture with Real LLMs")
    print("="*80)

    # Create orchestrator
    orchestrator = Orchestrator()

    # Add supervisor (GPT-4 - highest quality)
    supervisor = create_gpt4_model(
        model_id="supervisor-gpt4",
        domain=None
    )
    supervisor.role = supervisor.role.SUPERVISOR
    orchestrator.models["supervisor-gpt4"] = supervisor
    orchestrator.supervisor = supervisor

    # Add teachers (GPT-3.5-turbo and Claude Sonnet - mid-tier)
    teacher_math = create_gpt35_turbo_model(
        model_id="teacher-math",
        domain="mathematics"
    )
    teacher_math.role = teacher_math.role.TEACHER
    orchestrator.models["teacher-math"] = teacher_math

    teacher_science = create_claude_sonnet_model(
        model_id="teacher-science",
        domain="science"
    )
    teacher_science.role = teacher_science.role.TEACHER
    orchestrator.models["teacher-science"] = teacher_science

    # Add students (Claude Haiku - fastest/cheapest)
    student_math = create_claude_haiku_model(
        model_id="student-math-1",
        domain="mathematics"
    )
    student_math.role = student_math.role.STUDENT
    student_math.teacher_id = "teacher-math"
    orchestrator.models["student-math-1"] = student_math

    print("\nInitialized system:")
    print(f"  Supervisor: GPT-4 (OpenAI)")
    print(f"  Teachers: GPT-3.5-turbo (math), Claude Sonnet (science)")
    print(f"  Students: Claude Haiku (math)")

    # Process queries
    queries = [
        ("What is the Pythagorean theorem?", "mathematics"),
        ("How does photosynthesis work?", "science"),
        ("Solve: 2x + 5 = 13", "mathematics"),
    ]

    total_cost = 0.0

    for query, domain in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Domain: {domain}")
        print(f"{'='*60}")

        result = orchestrator.process_query(query, domain)

        print(f"\nFinal Answer: {result['final_response'][:150]}...")
        print(f"Routing: {result['routing_strategy']}")
        print(f"Models used: {len(result['candidate_responses'])}")
        print(f"Winner: {result.get('winner_model', 'N/A')}")

        # Calculate cost for this query
        query_cost = 0.0
        for model_id in orchestrator.models:
            model = orchestrator.models[model_id]
            if hasattr(model, 'total_cost'):
                query_cost += model.total_cost

        print(f"Query cost: ${query_cost - total_cost:.6f}")
        total_cost = query_cost

    # Print final statistics
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)

    for model_id, model in orchestrator.models.items():
        if hasattr(model, 'get_cost_statistics'):
            stats = model.get_cost_statistics()
            print(f"\n{model_id}:")
            print(f"  Provider: {stats.get('provider', 'N/A')}")
            print(f"  Model: {stats.get('model', 'N/A')}")
            print(f"  Queries: {stats.get('query_count', 0)}")
            print(f"  Total tokens: {stats.get('total_tokens', 0):,}")
            print(f"  Total cost: ${stats.get('total_cost_usd', 0):.6f}")

    print(f"\n💰 TOTAL COST: ${total_cost:.6f}")


def demo_cost_optimization():
    """Demo 4: Cost optimization through intelligent routing."""
    print("\n" + "="*80)
    print("DEMO 4: Cost Optimization Through Intelligent Routing")
    print("="*80)

    print("\nThis demo shows how the system saves cost by routing queries")
    print("to cheaper models when appropriate, while maintaining quality.")

    # Create models with different costs
    expensive_model = create_gpt4_model(model_id="expensive", domain="general")
    cheap_model = create_claude_haiku_model(model_id="cheap", domain="general")

    queries = [
        ("What is 2+2?", "Should use cheap model"),
        ("Explain the philosophical implications of Gödel's incompleteness theorems", "Should use expensive model"),
        ("What color is the sky?", "Should use cheap model"),
    ]

    total_cost_smart = 0.0
    total_cost_baseline = 0.0

    for query, expected in queries:
        print(f"\n{'-'*60}")
        print(f"Query: {query}")
        print(f"Expected: {expected}")

        # Simulate smart routing (simple heuristic: short queries = simple)
        if len(query) < 30:
            model = cheap_model
            strategy = "Cheap model (Claude Haiku)"
        else:
            model = expensive_model
            strategy = "Expensive model (GPT-4)"

        print(f"Routing: {strategy}")

        response = model.generate_response(query)
        cost = response.metadata.get('cost_usd', 0)
        total_cost_smart += cost

        # Calculate baseline cost (always using expensive model)
        baseline_response = expensive_model.generate_response(query)
        baseline_cost = baseline_response.metadata.get('cost_usd', 0)
        total_cost_baseline += baseline_cost

        print(f"Cost: ${cost:.6f} (baseline: ${baseline_cost:.6f})")

    print(f"\n{'='*60}")
    print("COST COMPARISON:")
    print(f"  Smart routing cost: ${total_cost_smart:.6f}")
    print(f"  Baseline cost (GPT-4 only): ${total_cost_baseline:.6f}")
    print(f"  Savings: ${total_cost_baseline - total_cost_smart:.6f} ({(1 - total_cost_smart/total_cost_baseline)*100:.1f}%)")


def main():
    """Run all demos."""
    print("="*80)
    print("REAL LLM DEMO - Self-Evolving Teacher-Student Architecture")
    print("="*80)

    # Check API keys
    if not check_api_keys():
        print("\n❌ No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print("   See .env.example for instructions")
        sys.exit(1)

    # Warning
    print("\n⚠️  WARNING: This will make REAL API calls and incur costs!")
    print("   Estimated cost: $0.50-$1.00")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(0)

    try:
        # Run demos
        demo_simple_query()
        time.sleep(1)

        demo_model_comparison()
        time.sleep(1)

        demo_teacher_student_architecture()
        time.sleep(1)

        demo_cost_optimization()

        print("\n" + "="*80)
        print("✅ Demo completed successfully!")
        print("="*80)

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("\nCommon issues:")
        print("  - Invalid API key: Check your .env file")
        print("  - Insufficient credits: Add credits to your account")
        print("  - Rate limit: Wait a few seconds and try again")


if __name__ == "__main__":
    main()
