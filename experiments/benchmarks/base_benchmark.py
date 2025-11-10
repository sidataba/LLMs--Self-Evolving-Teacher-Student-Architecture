"""
Base benchmark infrastructure for evaluating the self-evolving architecture.

Provides common functionality for running standard benchmarks (MMLU, TruthfulQA, etc.)
with proper result tracking, statistical analysis, and cost monitoring.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    query: str
    domain: Optional[str]
    correct_answer: str
    model_answer: str
    is_correct: bool
    confidence: float
    response_time: float
    cost_usd: float
    metadata: Dict[str, Any]


@dataclass
class BenchmarkStatistics:
    """Aggregated benchmark statistics."""
    benchmark_name: str
    total_queries: int
    correct: int
    accuracy: float
    accuracy_std: float
    avg_confidence: float
    avg_response_time: float
    total_cost_usd: float
    cost_per_query: float

    # Statistical significance
    confidence_interval_95: Tuple[float, float]
    p_value: Optional[float] = None

    # Per-domain breakdown
    domain_accuracy: Dict[str, float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BaseBenchmark(ABC):
    """
    Base class for all benchmarks.

    Subclasses must implement:
    - load_dataset(): Load benchmark dataset
    - evaluate_response(): Check if response is correct
    """

    def __init__(
        self,
        name: str,
        dataset_path: Optional[Path] = None,
        output_dir: Path = Path("./data/benchmark_results"),
        max_samples: Optional[int] = None,
    ):
        """
        Initialize benchmark.

        Args:
            name: Benchmark name
            dataset_path: Path to dataset (if None, will download)
            output_dir: Directory for saving results
            max_samples: Maximum samples to evaluate (for testing)
        """
        self.name = name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.max_samples = max_samples

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[BenchmarkResult] = []
        self.dataset = None

        logger.info(f"Initialized {name} benchmark")

    @abstractmethod
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load benchmark dataset.

        Returns:
            List of examples, each with 'query', 'answer', 'domain', etc.
        """
        pass

    @abstractmethod
    def evaluate_response(
        self,
        response: str,
        correct_answer: str,
        example: Dict[str, Any]
    ) -> bool:
        """
        Evaluate if response is correct.

        Args:
            response: Model's response
            correct_answer: Correct answer
            example: Full example dict

        Returns:
            True if correct, False otherwise
        """
        pass

    def run(
        self,
        orchestrator,
        save_interval: int = 50,
    ) -> BenchmarkStatistics:
        """
        Run benchmark evaluation.

        Args:
            orchestrator: System orchestrator
            save_interval: Save results every N queries

        Returns:
            BenchmarkStatistics with results
        """
        logger.info(f"Starting {self.name} benchmark evaluation")

        # Load dataset
        if self.dataset is None:
            self.dataset = self.load_dataset()

        # Limit samples if specified
        if self.max_samples:
            self.dataset = self.dataset[:self.max_samples]

        logger.info(f"Evaluating {len(self.dataset)} samples")

        # Process each example
        for i, example in enumerate(self.dataset):
            try:
                result = self._evaluate_example(example, orchestrator)
                self.results.append(result)

                # Log progress
                if (i + 1) % 10 == 0:
                    accuracy = np.mean([r.is_correct for r in self.results])
                    logger.info(
                        f"Progress: {i+1}/{len(self.dataset)} | "
                        f"Accuracy: {accuracy:.3f} | "
                        f"Cost: ${sum(r.cost_usd for r in self.results):.4f}"
                    )

                # Save intermediate results
                if (i + 1) % save_interval == 0:
                    self.save_results()

            except Exception as e:
                logger.error(f"Error evaluating example {i}: {e}")
                continue

        # Calculate final statistics
        stats = self.calculate_statistics()

        # Save final results
        self.save_results()
        self.save_statistics(stats)

        logger.info(f"Benchmark complete: {stats.accuracy:.3f} accuracy")

        return stats

    def _evaluate_example(
        self,
        example: Dict[str, Any],
        orchestrator
    ) -> BenchmarkResult:
        """Evaluate a single example."""
        query = example['query']
        correct_answer = example['answer']
        domain = example.get('domain')

        # Time the query
        start_time = time.time()

        # Get model response
        result = orchestrator.process_query(query, domain=domain)

        response_time = time.time() - start_time

        # Extract answer
        model_answer = result['final_response']

        # Evaluate correctness
        is_correct = self.evaluate_response(
            model_answer,
            correct_answer,
            example
        )

        # Calculate cost (sum across all models used)
        cost = 0.0
        for model_id in orchestrator.models:
            model = orchestrator.models[model_id]
            if hasattr(model, 'total_cost'):
                cost += model.total_cost

        # Extract per-query cost (difference from previous)
        if self.results:
            prev_total_cost = sum(r.cost_usd for r in self.results)
            query_cost = cost - prev_total_cost
        else:
            query_cost = cost

        return BenchmarkResult(
            query=query,
            domain=domain,
            correct_answer=correct_answer,
            model_answer=model_answer,
            is_correct=is_correct,
            confidence=result.get('winner_score', 0.0),
            response_time=response_time,
            cost_usd=query_cost,
            metadata={
                'routing_strategy': result.get('routing_strategy'),
                'models_used': len(result.get('candidate_responses', [])),
                'winner_model': result.get('winner_model'),
            }
        )

    def calculate_statistics(
        self,
        baseline_accuracy: Optional[float] = None
    ) -> BenchmarkStatistics:
        """
        Calculate comprehensive statistics.

        Args:
            baseline_accuracy: Baseline accuracy for significance testing

        Returns:
            BenchmarkStatistics
        """
        if not self.results:
            raise ValueError("No results to calculate statistics from")

        # Basic metrics
        correctness = [r.is_correct for r in self.results]
        accuracy = np.mean(correctness)
        accuracy_std = np.std(correctness)

        # Confidence interval (95%)
        n = len(correctness)
        se = accuracy_std / np.sqrt(n)
        ci_95 = (
            accuracy - 1.96 * se,
            accuracy + 1.96 * se
        )

        # Statistical significance (if baseline provided)
        p_value = None
        if baseline_accuracy is not None:
            # One-sample t-test
            t_stat, p_value = stats.ttest_1samp(correctness, baseline_accuracy)

        # Per-domain accuracy
        domain_accuracy = {}
        for result in self.results:
            if result.domain:
                if result.domain not in domain_accuracy:
                    domain_accuracy[result.domain] = []
                domain_accuracy[result.domain].append(result.is_correct)

        domain_accuracy = {
            domain: np.mean(scores)
            for domain, scores in domain_accuracy.items()
        }

        # Cost metrics
        total_cost = sum(r.cost_usd for r in self.results)

        return BenchmarkStatistics(
            benchmark_name=self.name,
            total_queries=len(self.results),
            correct=sum(correctness),
            accuracy=accuracy,
            accuracy_std=accuracy_std,
            avg_confidence=np.mean([r.confidence for r in self.results]),
            avg_response_time=np.mean([r.response_time for r in self.results]),
            total_cost_usd=total_cost,
            cost_per_query=total_cost / len(self.results),
            confidence_interval_95=ci_95,
            p_value=p_value,
            domain_accuracy=domain_accuracy,
        )

    def save_results(self):
        """Save detailed results to JSON."""
        results_file = self.output_dir / f"{self.name}_results.json"

        results_data = [
            {
                'query': r.query,
                'domain': r.domain,
                'correct_answer': r.correct_answer,
                'model_answer': r.model_answer,
                'is_correct': r.is_correct,
                'confidence': r.confidence,
                'response_time': r.response_time,
                'cost_usd': r.cost_usd,
                'metadata': r.metadata,
            }
            for r in self.results
        ]

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Saved {len(self.results)} results to {results_file}")

    def save_statistics(self, stats: BenchmarkStatistics):
        """Save statistics to JSON."""
        stats_file = self.output_dir / f"{self.name}_statistics.json"

        with open(stats_file, 'w') as f:
            json.dump(stats.to_dict(), f, indent=2)

        logger.info(f"Saved statistics to {stats_file}")

    def print_report(self, stats: Optional[BenchmarkStatistics] = None):
        """Print human-readable report."""
        if stats is None:
            stats = self.calculate_statistics()

        print("\n" + "="*80)
        print(f"BENCHMARK RESULTS: {self.name}")
        print("="*80)

        print(f"\n📊 Overall Performance:")
        print(f"  Accuracy: {stats.accuracy:.3f} ± {stats.accuracy_std:.3f}")
        print(f"  95% CI: [{stats.confidence_interval_95[0]:.3f}, {stats.confidence_interval_95[1]:.3f}]")
        print(f"  Correct: {stats.correct}/{stats.total_queries}")
        print(f"  Avg Confidence: {stats.avg_confidence:.3f}")

        if stats.p_value is not None:
            print(f"  P-value vs baseline: {stats.p_value:.4f}")

        print(f"\n⏱️  Performance:")
        print(f"  Avg Response Time: {stats.avg_response_time:.2f}s")

        print(f"\n💰 Cost:")
        print(f"  Total Cost: ${stats.total_cost_usd:.4f}")
        print(f"  Cost per Query: ${stats.cost_per_query:.6f}")

        if stats.domain_accuracy:
            print(f"\n📈 Per-Domain Accuracy:")
            for domain, acc in sorted(stats.domain_accuracy.items()):
                print(f"  {domain}: {acc:.3f}")

        print("\n" + "="*80)
