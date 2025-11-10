"""
Interactive demo of the Self-Evolving Teacher-Student Architecture.

This example provides an interactive CLI for testing the system.
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
logger.add(sys.stdout, level="WARNING", format="<level>{level: <8}</level> | {message}")


def print_menu():
    """Print interactive menu."""
    print("\n" + "="*70)
    print("  Self-Evolving Teacher-Student System - Interactive Demo")
    print("="*70)
    print("\nOptions:")
    print("  1. Process a query")
    print("  2. Show system status")
    print("  3. Show model details")
    print("  4. Add a new student model")
    print("  5. Export metrics")
    print("  6. Exit")
    print("="*70)


def main():
    """Run interactive demo."""
    # Initialize system
    print("\n📦 Initializing system...")
    orchestrator = Orchestrator(config_path="./config/default_config.yaml")
    dashboard = Dashboard(orchestrator)

    print("✓ System initialized!")

    while True:
        print_menu()
        choice = input("\nEnter your choice (1-6): ").strip()

        if choice == "1":
            # Process query
            query = input("\nEnter your query: ").strip()

            if query:
                print(f"\n🔄 Processing query...")
                result = orchestrator.process_query(query)
                dashboard.print_query_summary(result)

        elif choice == "2":
            # Show system status
            dashboard.print_system_status()

        elif choice == "3":
            # Show model details
            model_id = input("\nEnter model ID (or press Enter for all models): ").strip()

            if model_id:
                dashboard.print_model_details(model_id)
            else:
                dashboard.print_model_details()

        elif choice == "4":
            # Add new student
            model_id = input("\nEnter new student model ID: ").strip()
            domain = input("Enter domain (mathematics/science/programming): ").strip()
            teacher_id = input("Enter teacher ID (optional): ").strip() or None

            if model_id and domain:
                student = orchestrator.add_student_model(model_id, domain, teacher_id)
                print(f"\n✓ Added new student: {student.model_id}")

        elif choice == "5":
            # Export metrics
            print("\n💾 Exporting metrics...")
            metrics_files = orchestrator.export_metrics()
            dashboard_report = dashboard.export_dashboard_report()

            print(f"\n✓ Exported metrics:")
            for name, path in metrics_files.items():
                print(f"  • {name}: {path}")
            print(f"  • dashboard: {dashboard_report}")

        elif choice == "6":
            # Exit
            print("\n👋 Goodbye!")
            break

        else:
            print("\n❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
