"""
Unified Benchmark Runner

Runs all standard benchmarks (MMLU, TruthfulQA, GSM8K, etc.) and compares
the self-evolving system against baselines.

Usage:
    python experiments/run_all_benchmarks.py --config=config/real_llm_config.yaml
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.orchestrator import Orchestrator
from experiments.benchmarks.mmlu_benchmark import run_mmlu_benchmark
from experiments.benchmarks.truthfulqa_benchmark import run_truthfulqa_benchmark
from experiments.benchmarks.gsm8k_benchmark import run_gsm8k_benchmark

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for the self-evolving architecture.
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        output_dir: Path = Path("./data/benchmark_results"),
    ):
        """
        Initialize benchmark suite.

        Args:
            orchestrator: System orchestrator to evaluate
            output_dir: Directory for saving results
        """
        self.orchestrator = orchestrator
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {}
        self.start_time = None
        self.end_time = None

    def run_all(
        self,
        max_samples_per_benchmark: int = None,
        benchmarks: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Run all benchmarks.

        Args:
            max_samples_per_benchmark: Limit samples per benchmark (for testing)
            benchmarks: Specific benchmarks to run (None = all)

        Returns:
            Dictionary with all results
        """
        self.start_time = datetime.now()

        logger.info("="*80)
        logger.info("BENCHMARK SUITE - Self-Evolving Teacher-Student Architecture")
        logger.info("="*80)
        logger.info(f"Start time: {self.start_time}")
        logger.info(f"Max samples per benchmark: {max_samples_per_benchmark or 'unlimited'}")

        # Determine which benchmarks to run
        available_benchmarks = {
            'mmlu': self.run_mmlu,
            'truthfulqa': self.run_truthfulqa,
            'gsm8k': self.run_gsm8k,
        }

        benchmarks_to_run = benchmarks or list(available_benchmarks.keys())

        # Run each benchmark
        for name in benchmarks_to_run:
            if name not in available_benchmarks:
                logger.warning(f"Unknown benchmark: {name}")
                continue

            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"Running {name.upper()} benchmark...")
                logger.info(f"{'='*80}")

                result = available_benchmarks[name](max_samples_per_benchmark)
                self.results[name] = result

                logger.info(f"✅ {name.upper()} complete")

            except Exception as e:
                logger.error(f"❌ {name.upper()} failed: {e}", exc_info=True)
                self.results[name] = {'error': str(e)}

        self.end_time = datetime.now()

        # Generate summary report
        summary = self.generate_summary()

        # Save all results
        self.save_results(summary)

        # Print summary
        self.print_summary(summary)

        return summary

    def run_mmlu(self, max_samples: int = None) -> Dict[str, Any]:
        """Run MMLU benchmark."""
        return run_mmlu_benchmark(
            self.orchestrator,
            max_samples=max_samples,
        )

    def run_truthfulqa(self, max_samples: int = None) -> Dict[str, Any]:
        """Run TruthfulQA benchmark."""
        return run_truthfulqa_benchmark(
            self.orchestrator,
            max_samples=max_samples,
        )

    def run_gsm8k(self, max_samples: int = None) -> Dict[str, Any]:
        """Run GSM8K benchmark."""
        return run_gsm8k_benchmark(
            self.orchestrator,
            max_samples=max_samples,
        )

    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary of all benchmark results."""
        summary = {
            'timestamp': self.start_time.isoformat(),
            'duration_seconds': (self.end_time - self.start_time).total_seconds(),
            'benchmarks': {},
            'overall': {},
        }

        # Aggregate results
        total_queries = 0
        total_correct = 0
        total_cost = 0.0

        for name, result in self.results.items():
            if 'error' in result:
                summary['benchmarks'][name] = result
                continue

            stats = result['statistics']

            summary['benchmarks'][name] = {
                'accuracy': stats['accuracy'],
                'accuracy_ci': stats['confidence_interval_95'],
                'total_queries': stats['total_queries'],
                'correct': stats['correct'],
                'avg_confidence': stats['avg_confidence'],
                'total_cost': stats['total_cost_usd'],
                'cost_per_query': stats['cost_per_query'],
            }

            total_queries += stats['total_queries']
            total_correct += stats['correct']
            total_cost += stats['total_cost_usd']

        # Overall statistics
        if total_queries > 0:
            summary['overall'] = {
                'total_queries': total_queries,
                'total_correct': total_correct,
                'overall_accuracy': total_correct / total_queries,
                'total_cost_usd': total_cost,
                'avg_cost_per_query': total_cost / total_queries,
            }

        # Get model statistics
        summary['model_statistics'] = self.get_model_statistics()

        return summary

    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics for all models in the system."""
        model_stats = {}

        for model_id, model in self.orchestrator.models.items():
            stats = model.get_statistics()

            model_stats[model_id] = {
                'role': model.role.value,
                'domain': model.domain,
                'query_count': stats.get('query_count', 0),
                'avg_confidence': stats.get('avg_confidence', 0),
            }

            # Add cost stats if available (real LLMs)
            if 'total_cost_usd' in stats:
                model_stats[model_id].update({
                    'total_cost_usd': stats['total_cost_usd'],
                    'total_tokens': stats.get('total_tokens', 0),
                })

        return model_stats

    def save_results(self, summary: Dict[str, Any]):
        """Save all results to files."""
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')

        # Save summary
        summary_file = self.output_dir / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved summary to {summary_file}")

        # Save detailed results
        results_file = self.output_dir / f"detailed_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Saved detailed results to {results_file}")

    def print_summary(self, summary: Dict[str, Any]):
        """Print human-readable summary."""
        print("\n" + "="*80)
        print("BENCHMARK SUITE SUMMARY")
        print("="*80)

        print(f"\n⏱️  Duration: {summary['duration_seconds']:.1f}s")

        # Per-benchmark results
        print(f"\n📊 Benchmark Results:")
        for name, stats in summary['benchmarks'].items():
            if 'error' in stats:
                print(f"  ❌ {name.upper()}: ERROR - {stats['error']}")
                continue

            print(f"\n  {name.upper()}:")
            print(f"    Accuracy: {stats['accuracy']:.3f} "
                  f"(95% CI: [{stats['accuracy_ci'][0]:.3f}, {stats['accuracy_ci'][1]:.3f}])")
            print(f"    Correct: {stats['correct']}/{stats['total_queries']}")
            print(f"    Avg Confidence: {stats['avg_confidence']:.3f}")
            print(f"    Cost: ${stats['total_cost']:.4f} "
                  f"(${stats['cost_per_query']:.6f}/query)")

        # Overall results
        if 'overall' in summary and summary['overall']:
            overall = summary['overall']
            print(f"\n🎯 Overall Performance:")
            print(f"  Total Queries: {overall['total_queries']}")
            print(f"  Overall Accuracy: {overall['overall_accuracy']:.3f}")
            print(f"  Total Cost: ${overall['total_cost_usd']:.4f}")
            print(f"  Avg Cost/Query: ${overall['avg_cost_per_query']:.6f}")

        # Model usage
        print(f"\n🤖 Model Usage:")
        for model_id, stats in summary['model_statistics'].items():
            print(f"  {model_id} ({stats['role']}):")
            print(f"    Queries: {stats['query_count']}")
            if 'total_cost_usd' in stats:
                print(f"    Cost: ${stats['total_cost_usd']:.6f}")

        print("\n" + "="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run all benchmarks for the self-evolving architecture'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./config/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per benchmark (for testing)'
    )
    parser.add_argument(
        '--benchmarks',
        type=str,
        nargs='+',
        default=None,
        help='Specific benchmarks to run (mmlu, truthfulqa, gsm8k)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/benchmark_results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Initialize orchestrator
    logger.info(f"Initializing orchestrator with config: {args.config}")
    orchestrator = Orchestrator(config_path=args.config)

    # Create benchmark suite
    suite = BenchmarkSuite(
        orchestrator=orchestrator,
        output_dir=Path(args.output_dir)
    )

    # Run all benchmarks
    summary = suite.run_all(
        max_samples_per_benchmark=args.max_samples,
        benchmarks=args.benchmarks,
    )

    logger.info("✅ All benchmarks complete!")

    # Return exit code based on success
    if any('error' in result for result in summary['benchmarks'].values()):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
