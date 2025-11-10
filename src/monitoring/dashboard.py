"""Dashboard for monitoring system performance."""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from loguru import logger


class Dashboard:
    """
    Monitoring dashboard for the self-evolving system.
    Provides real-time statistics and visualizations.
    """

    def __init__(self, orchestrator):
        """
        Initialize dashboard.

        Args:
            orchestrator: Reference to the main orchestrator
        """
        self.orchestrator = orchestrator
        self.start_time = datetime.now()

        logger.info("Dashboard initialized")

    def print_system_status(self) -> None:
        """Print current system status to console."""
        status = self.orchestrator.get_system_status()

        print("\n" + "="*70)
        print("  SELF-EVOLVING TEACHER-STUDENT SYSTEM - STATUS DASHBOARD")
        print("="*70)

        # System overview
        print(f"\n📊 SYSTEM OVERVIEW")
        print(f"  Uptime: {self._get_uptime()}")
        print(f"  Total Queries Processed: {status['query_count']}")
        print(f"  Total Models: {sum(status['role_distribution'].values())}")

        # Model distribution
        print(f"\n🤖 MODEL DISTRIBUTION")
        for role, count in status['role_distribution'].items():
            print(f"  {role.capitalize()}: {count}")

        # Top performers
        print(f"\n🏆 TOP PERFORMING MODELS")
        top_models = self._get_top_models(status['model_stats'], limit=5)
        for i, (model_id, stats) in enumerate(top_models, 1):
            print(
                f"  {i}. {model_id}: "
                f"Win Rate: {stats['win_rate']:.2%} "
                f"({stats['total_wins']}/{stats['total_queries']} wins)"
            )

        # Recent promotions
        promo_stats = status['promotion_stats']
        print(f"\n⬆️  PROMOTIONS")
        print(f"  Total Promotions: {promo_stats['total_promotions']}")
        print(f"  Student → TA: {promo_stats['student_to_ta']}")
        print(f"  TA → Teacher: {promo_stats['ta_to_teacher']}")

        if promo_stats['promotion_history']:
            print(f"\n  Recent Promotions:")
            for promo in promo_stats['promotion_history'][-3:]:
                print(f"    • {promo['model_id']}: {promo['from_role']} → {promo['to_role']}")

        # Vector DB stats
        print(f"\n💾 VECTOR DATABASE")
        vdb_stats = status['vector_db_stats']
        print(f"  Stored Queries: {vdb_stats['total_queries']}")
        print(f"  Embedding Model: {vdb_stats['embedding_model']}")

        print("\n" + "="*70 + "\n")

    def print_model_details(self, model_id: Optional[str] = None) -> None:
        """Print detailed statistics for a specific model or all models."""
        status = self.orchestrator.get_system_status()

        if model_id:
            # Print specific model
            if model_id not in status['model_stats']:
                print(f"❌ Model not found: {model_id}")
                return

            stats = status['model_stats'][model_id]
            self._print_single_model(model_id, stats)

        else:
            # Print all models
            print("\n" + "="*70)
            print("  MODEL STATISTICS")
            print("="*70)

            for model_id, stats in status['model_stats'].items():
                self._print_single_model(model_id, stats)

    def _print_single_model(self, model_id: str, stats: Dict[str, Any]) -> None:
        """Print statistics for a single model."""
        print(f"\n📌 {model_id}")
        print(f"  Role: {stats['role']}")
        print(f"  Domain: {stats.get('domain', 'N/A')}")
        print(f"  Total Queries: {stats['total_queries']}")
        print(f"  Total Wins: {stats['total_wins']}")
        print(f"  Win Rate: {stats['win_rate']:.2%}")
        print(f"  Avg Confidence: {stats['avg_confidence']:.3f}")

    def export_dashboard_report(self, output_path: str = "./data/dashboard_report.json") -> str:
        """Export dashboard data as JSON report."""
        status = self.orchestrator.get_system_status()

        report = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "status": status,
            "top_models": self._get_top_models(status['model_stats'], limit=10),
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Dashboard report exported to {output_path}")

        return output_path

    def _get_top_models(
        self,
        model_stats: Dict[str, Dict[str, Any]],
        limit: int = 5,
    ) -> List[tuple]:
        """Get top performing models."""
        # Sort by win rate, then by total queries
        sorted_models = sorted(
            model_stats.items(),
            key=lambda x: (x[1]['win_rate'], x[1]['total_queries']),
            reverse=True,
        )

        return sorted_models[:limit]

    def _get_uptime(self) -> str:
        """Get formatted uptime string."""
        uptime = datetime.now() - self.start_time
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        return f"{hours}h {minutes}m {seconds}s"

    def print_query_summary(self, query_result: Dict[str, Any]) -> None:
        """Print a summary of a query result."""
        print("\n" + "-"*60)
        print("  QUERY RESULT")
        print("-"*60)

        print(f"\nQuery: {query_result['query_text']}")
        print(f"Query ID: {query_result['query_id']}")
        print(f"\nWinner: {query_result['winner_model']} (Score: {query_result['winner_score']:.3f})")
        print(f"Routing Strategy: {query_result['routing_strategy']}")
        print(f"Models Queried: {query_result['num_models_queried']}")

        if query_result.get('promotions'):
            print(f"\n🎉 Promotions during this query:")
            for promo in query_result['promotions']:
                print(f"  • {promo['model_id']}: {promo['from_role']} → {promo['to_role']}")

        print(f"\nResponse:\n{query_result['final_response'][:200]}...")

        print("\n" + "-"*60 + "\n")
