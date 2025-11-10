"""
Basic demo of the Self-Evolving Teacher-Student Architecture.

This example demonstrates:
1. System initialization
2. Processing queries
3. Automatic model promotion
4. System monitoring
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.orchestrator import Orchestrator
from src.monitoring.dashboard import Dashboard
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="<level>{level: <8}</level> | {message}")


def main():
    """Run basic demo."""
    print("\n" + "="*70)
    print("  Self-Evolving Teacher-Student Architecture - Basic Demo")
    print("="*70 + "\n")

    # 1. Initialize the system
    print("📦 Initializing system...")
    orchestrator = Orchestrator(config_path="./config/default_config.yaml")

    # Create dashboard
    dashboard = Dashboard(orchestrator)

    # 2. Show initial system status
    print("\n📊 Initial System Status:")
    dashboard.print_system_status()

    # 3. Process sample queries
    sample_queries = [
        # Mathematics queries
        "How do I solve a quadratic equation?",
        "What is the derivative of x squared?",
        "Explain the Pythagorean theorem",
        "How do I calculate the area of a circle?",
        "What is calculus used for?",

        # Science queries
        "Explain photosynthesis",
        "What is Newton's first law?",
        "How does DNA replication work?",
        "What causes chemical reactions?",
        "Explain the water cycle",

        # Programming queries
        "How do I write a Python function?",
        "What is a JavaScript closure?",
        "Explain bubble sort algorithm",
        "How do I debug my code?",
        "What are design patterns?",

        # More math queries to trigger promotions
        "How to solve linear equations?",
        "Explain integration",
        "What is a matrix?",
        "How to find the slope of a line?",
        "What is probability?",
        "Explain set theory",
        "How to calculate percentages?",
        "What is trigonometry?",
        "Explain geometric series",
        "How to solve inequalities?",
    ]

    print("\n📝 Processing queries...\n")

    # Process each query
    for i, query in enumerate(sample_queries, 1):
        print(f"\n{'─'*70}")
        print(f"Query {i}/{len(sample_queries)}: {query}")
        print(f"{'─'*70}")

        result = orchestrator.process_query(query)

        # Show brief result
        print(f"✓ Winner: {result['winner_model']} (Score: {result['winner_score']:.3f})")
        print(f"  Strategy: {result['routing_strategy']}, Models queried: {result['num_models_queried']}")

        # Show promotions if any
        if result.get('promotions'):
            print(f"\n  🎉 PROMOTIONS:")
            for promo in result['promotions']:
                print(f"     {promo['model_id']}: {promo['from_role']} → {promo['to_role']}")

    # 4. Show final system status
    print("\n\n📊 Final System Status:")
    dashboard.print_system_status()

    # 5. Show detailed model statistics
    print("\n📈 Detailed Model Statistics:")
    dashboard.print_model_details()

    # 6. Export results
    print("\n💾 Exporting results...")
    metrics_files = orchestrator.export_metrics("./data/demo_export")
    dashboard_report = dashboard.export_dashboard_report("./data/demo_dashboard.json")

    print(f"\n✓ Exported metrics to:")
    for name, path in metrics_files.items():
        print(f"  • {name}: {path}")
    print(f"  • dashboard: {dashboard_report}")

    print("\n" + "="*70)
    print("  Demo completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
