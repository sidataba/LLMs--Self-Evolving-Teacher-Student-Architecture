"""
Human Evaluation Framework

Provides tools for conducting human evaluations of system outputs,
including Amazon MTurk integration and local evaluation interfaces.

Evaluation criteria:
- Relevance: Does the answer address the question?
- Correctness: Is the answer factually correct?
- Completeness: Does it cover all aspects?
- Clarity: Is it well-written and easy to understand?
"""

import json
import uuid
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np


@dataclass
class EvaluationSample:
    """Single sample for human evaluation."""
    sample_id: str
    query: str
    response: str
    domain: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HumanRating:
    """Human rating for a single sample."""
    sample_id: str
    evaluator_id: str
    relevance: int  # 1-5 scale
    correctness: int  # 1-5 scale
    completeness: int  # 1-5 scale
    clarity: int  # 1-5 scale
    overall: int  # 1-5 scale
    comments: str = ""
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_average_score(self) -> float:
        """Get average across all criteria (excluding overall)."""
        return np.mean([
            self.relevance,
            self.correctness,
            self.completeness,
            self.clarity
        ])


class HumanEvaluationFramework:
    """
    Framework for conducting human evaluations.

    Supports:
    - Sample selection and export
    - Rating collection
    - Statistical analysis
    - Inter-rater reliability
    """

    def __init__(
        self,
        output_dir: Path = Path("./data/human_eval"),
    ):
        """
        Initialize evaluation framework.

        Args:
            output_dir: Directory for storing evaluation data
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.samples: List[EvaluationSample] = []
        self.ratings: List[HumanRating] = []

    def create_evaluation_batch(
        self,
        orchestrator,
        queries: List[str],
        batch_name: str = "eval_batch",
        num_samples: int = None,
    ) -> List[EvaluationSample]:
        """
        Create a batch of samples for human evaluation.

        Args:
            orchestrator: System orchestrator
            queries: List of queries to evaluate
            batch_name: Name for this evaluation batch
            num_samples: Number of samples to create (None = all)

        Returns:
            List of evaluation samples
        """
        # Limit samples if specified
        if num_samples:
            queries = queries[:num_samples]

        samples = []

        for i, query in enumerate(queries):
            # Get response from system
            result = orchestrator.process_query(query)

            sample = EvaluationSample(
                sample_id=f"{batch_name}_{i:04d}",
                query=query,
                response=result['final_response'],
                domain=result.get('domain'),
                metadata={
                    'routing_strategy': result.get('routing_strategy'),
                    'winner_model': result.get('winner_model'),
                    'confidence': result.get('winner_score'),
                }
            )

            samples.append(sample)

        self.samples.extend(samples)

        # Save batch
        self.save_samples(samples, batch_name)

        return samples

    def save_samples(
        self,
        samples: List[EvaluationSample],
        batch_name: str
    ):
        """Save samples to JSON file."""
        output_file = self.output_dir / f"{batch_name}_samples.json"

        data = [s.to_dict() for s in samples]

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(samples)} samples to {output_file}")

    def export_for_mturk(
        self,
        samples: List[EvaluationSample],
        output_file: Path = None,
    ) -> Path:
        """
        Export samples in MTurk-compatible CSV format.

        Args:
            samples: Samples to export
            output_file: Output CSV file path

        Returns:
            Path to created CSV file
        """
        if output_file is None:
            output_file = self.output_dir / "mturk_batch.csv"

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'sample_id',
                'query',
                'response',
                'domain',
            ])

            # Rows
            for sample in samples:
                writer.writerow([
                    sample.sample_id,
                    sample.query,
                    sample.response,
                    sample.domain or '',
                ])

        print(f"Exported {len(samples)} samples to {output_file} for MTurk")
        return output_file

    def create_evaluation_interface_html(
        self,
        samples: List[EvaluationSample],
        output_file: Path = None,
    ) -> Path:
        """
        Create simple HTML interface for local evaluation.

        Args:
            samples: Samples to evaluate
            output_file: Output HTML file path

        Returns:
            Path to created HTML file
        """
        if output_file is None:
            output_file = self.output_dir / "evaluation_interface.html"

        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Human Evaluation Interface</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; }
        .sample { border: 1px solid #ccc; padding: 20px; margin-bottom: 30px; border-radius: 5px; }
        .query { background-color: #f0f0f0; padding: 15px; margin-bottom: 15px; border-radius: 3px; }
        .response { background-color: #e8f4f8; padding: 15px; margin-bottom: 15px; border-radius: 3px; }
        .rating-row { margin: 10px 0; }
        .rating-row label { display: inline-block; width: 150px; font-weight: bold; }
        .scale { margin-left: 10px; }
        .scale input { margin: 0 5px; }
        button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #45a049; }
        .instructions { background-color: #fff3cd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
        .progress { background-color: #d4edda; padding: 10px; margin-bottom: 20px; text-align: center; }
    </style>
</head>
<body>
    <h1>Human Evaluation Interface</h1>

    <div class="instructions">
        <h3>Instructions:</h3>
        <p>Please rate each response on the following criteria (1=Poor, 5=Excellent):</p>
        <ul>
            <li><strong>Relevance:</strong> Does the answer address the question?</li>
            <li><strong>Correctness:</strong> Is the answer factually correct?</li>
            <li><strong>Completeness:</strong> Does it cover all important aspects?</li>
            <li><strong>Clarity:</strong> Is it well-written and easy to understand?</li>
            <li><strong>Overall:</strong> Overall quality of the response</li>
        </ul>
    </div>

    <div class="progress" id="progress">
        Progress: <span id="current">0</span> / <span id="total">""" + str(len(samples)) + """</span>
    </div>

    <form id="evaluationForm">
"""

        # Add samples
        for i, sample in enumerate(samples):
            html_content += f"""
    <div class="sample" id="sample_{i}" style="display: {'block' if i == 0 else 'none'};">
        <h3>Sample {i+1} / {len(samples)}</h3>
        <p><strong>Sample ID:</strong> {sample.sample_id}</p>
        {f'<p><strong>Domain:</strong> {sample.domain}</p>' if sample.domain else ''}

        <div class="query">
            <strong>Query:</strong><br>
            {sample.query}
        </div>

        <div class="response">
            <strong>Response:</strong><br>
            {sample.response}
        </div>

        <input type="hidden" name="sample_id_{i}" value="{sample.sample_id}">

        <div class="rating-row">
            <label>Relevance:</label>
            <div class="scale">
                <input type="radio" name="relevance_{i}" value="1" required> 1
                <input type="radio" name="relevance_{i}" value="2"> 2
                <input type="radio" name="relevance_{i}" value="3"> 3
                <input type="radio" name="relevance_{i}" value="4"> 4
                <input type="radio" name="relevance_{i}" value="5"> 5
            </div>
        </div>

        <div class="rating-row">
            <label>Correctness:</label>
            <div class="scale">
                <input type="radio" name="correctness_{i}" value="1" required> 1
                <input type="radio" name="correctness_{i}" value="2"> 2
                <input type="radio" name="correctness_{i}" value="3"> 3
                <input type="radio" name="correctness_{i}" value="4"> 4
                <input type="radio" name="correctness_{i}" value="5"> 5
            </div>
        </div>

        <div class="rating-row">
            <label>Completeness:</label>
            <div class="scale">
                <input type="radio" name="completeness_{i}" value="1" required> 1
                <input type="radio" name="completeness_{i}" value="2"> 2
                <input type="radio" name="completeness_{i}" value="3"> 3
                <input type="radio" name="completeness_{i}" value="4"> 4
                <input type="radio" name="completeness_{i}" value="5"> 5
            </div>
        </div>

        <div class="rating-row">
            <label>Clarity:</label>
            <div class="scale">
                <input type="radio" name="clarity_{i}" value="1" required> 1
                <input type="radio" name="clarity_{i}" value="2"> 2
                <input type="radio" name="clarity_{i}" value="3"> 3
                <input type="radio" name="clarity_{i}" value="4"> 4
                <input type="radio" name="clarity_{i}" value="5"> 5
            </div>
        </div>

        <div class="rating-row">
            <label>Overall Quality:</label>
            <div class="scale">
                <input type="radio" name="overall_{i}" value="1" required> 1
                <input type="radio" name="overall_{i}" value="2"> 2
                <input type="radio" name="overall_{i}" value="3"> 3
                <input type="radio" name="overall_{i}" value="4"> 4
                <input type="radio" name="overall_{i}" value="5"> 5
            </div>
        </div>

        <div class="rating-row">
            <label>Comments (optional):</label><br>
            <textarea name="comments_{i}" rows="3" style="width: 100%; margin-top: 10px;"></textarea>
        </div>

        <div style="margin-top: 20px;">
            {'<button type="button" onclick="previousSample()">Previous</button>' if i > 0 else ''}
            {'<button type="button" onclick="nextSample()">Next</button>' if i < len(samples) - 1 else '<button type="submit">Submit Evaluations</button>'}
        </div>
    </div>
"""

        html_content += """
    </form>

    <script>
        let currentSample = 0;
        const totalSamples = """ + str(len(samples)) + """;

        function showSample(index) {
            for (let i = 0; i < totalSamples; i++) {
                document.getElementById(`sample_${i}`).style.display = i === index ? 'block' : 'none';
            }
            currentSample = index;
            document.getElementById('current').textContent = index + 1;
        }

        function nextSample() {
            if (currentSample < totalSamples - 1) {
                showSample(currentSample + 1);
            }
        }

        function previousSample() {
            if (currentSample > 0) {
                showSample(currentSample - 1);
            }
        }

        document.getElementById('evaluationForm').addEventListener('submit', function(e) {
            e.preventDefault();

            const formData = new FormData(this);
            const results = [];

            for (let i = 0; i < totalSamples; i++) {
                results.push({
                    sample_id: formData.get(`sample_id_${i}`),
                    relevance: parseInt(formData.get(`relevance_${i}`)),
                    correctness: parseInt(formData.get(`correctness_${i}`)),
                    completeness: parseInt(formData.get(`completeness_${i}`)),
                    clarity: parseInt(formData.get(`clarity_${i}`)),
                    overall: parseInt(formData.get(`overall_${i}`)),
                    comments: formData.get(`comments_${i}`)
                });
            }

            // Download results as JSON
            const blob = new Blob([JSON.stringify(results, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'evaluation_results_' + new Date().toISOString() + '.json';
            a.click();

            alert('Evaluations saved! Thank you.');
        });
    </script>
</body>
</html>
"""

        with open(output_file, 'w') as f:
            f.write(html_content)

        print(f"Created evaluation interface at {output_file}")
        print(f"Open in browser to evaluate: file://{output_file.absolute()}")

        return output_file

    def load_ratings(self, ratings_file: Path) -> List[HumanRating]:
        """Load ratings from JSON file."""
        with open(ratings_file, 'r') as f:
            data = json.load(f)

        ratings = []
        for item in data:
            rating = HumanRating(
                sample_id=item['sample_id'],
                evaluator_id=item.get('evaluator_id', 'default'),
                relevance=item['relevance'],
                correctness=item['correctness'],
                completeness=item['completeness'],
                clarity=item['clarity'],
                overall=item['overall'],
                comments=item.get('comments', ''),
                timestamp=item.get('timestamp')
            )
            ratings.append(rating)

        self.ratings.extend(ratings)
        print(f"Loaded {len(ratings)} ratings from {ratings_file}")

        return ratings

    def calculate_inter_rater_reliability(
        self,
        ratings_by_evaluator: Dict[str, List[HumanRating]]
    ) -> float:
        """
        Calculate inter-rater reliability (Krippendorff's alpha).

        Simplified implementation using average pairwise agreement.
        """
        # Group ratings by sample
        ratings_by_sample = {}
        for evaluator_id, ratings in ratings_by_evaluator.items():
            for rating in ratings:
                if rating.sample_id not in ratings_by_sample:
                    ratings_by_sample[rating.sample_id] = []
                ratings_by_sample[rating.sample_id].append(rating)

        # Calculate agreement for samples with multiple raters
        agreements = []
        for sample_id, sample_ratings in ratings_by_sample.items():
            if len(sample_ratings) < 2:
                continue

            # Calculate pairwise agreement
            scores = [r.get_average_score() for r in sample_ratings]
            for i in range(len(scores)):
                for j in range(i + 1, len(scores)):
                    # Agreement within 0.5 points on 5-point scale
                    agreement = 1.0 if abs(scores[i] - scores[j]) <= 0.5 else 0.0
                    agreements.append(agreement)

        if not agreements:
            return 0.0

        return np.mean(agreements)

    def generate_statistics(self) -> Dict[str, Any]:
        """Generate statistics from collected ratings."""
        if not self.ratings:
            return {}

        # Per-criterion statistics
        relevance_scores = [r.relevance for r in self.ratings]
        correctness_scores = [r.correctness for r in self.ratings]
        completeness_scores = [r.completeness for r in self.ratings]
        clarity_scores = [r.clarity for r in self.ratings]
        overall_scores = [r.overall for r in self.ratings]
        average_scores = [r.get_average_score() for r in self.ratings]

        stats = {
            'num_ratings': len(self.ratings),
            'num_samples': len(set(r.sample_id for r in self.ratings)),
            'relevance': {
                'mean': np.mean(relevance_scores),
                'std': np.std(relevance_scores),
                'median': np.median(relevance_scores),
            },
            'correctness': {
                'mean': np.mean(correctness_scores),
                'std': np.std(correctness_scores),
                'median': np.median(correctness_scores),
            },
            'completeness': {
                'mean': np.mean(completeness_scores),
                'std': np.std(completeness_scores),
                'median': np.median(completeness_scores),
            },
            'clarity': {
                'mean': np.mean(clarity_scores),
                'std': np.std(clarity_scores),
                'median': np.median(clarity_scores),
            },
            'overall': {
                'mean': np.mean(overall_scores),
                'std': np.std(overall_scores),
                'median': np.median(overall_scores),
            },
            'average_across_criteria': {
                'mean': np.mean(average_scores),
                'std': np.std(average_scores),
                'median': np.median(average_scores),
            }
        }

        return stats

    def print_statistics(self, stats: Dict[str, Any] = None):
        """Print human-readable statistics."""
        if stats is None:
            stats = self.generate_statistics()

        if not stats:
            print("No ratings to analyze.")
            return

        print("\n" + "="*60)
        print("HUMAN EVALUATION STATISTICS")
        print("="*60)

        print(f"\nTotal Ratings: {stats['num_ratings']}")
        print(f"Unique Samples: {stats['num_samples']}")

        print(f"\n📊 Scores (on 5-point scale):")
        for criterion in ['relevance', 'correctness', 'completeness', 'clarity', 'overall']:
            criterion_stats = stats[criterion]
            print(f"\n  {criterion.capitalize()}:")
            print(f"    Mean: {criterion_stats['mean']:.2f} ± {criterion_stats['std']:.2f}")
            print(f"    Median: {criterion_stats['median']:.1f}")

        avg_stats = stats['average_across_criteria']
        print(f"\n  Average Across Criteria:")
        print(f"    Mean: {avg_stats['mean']:.2f} ± {avg_stats['std']:.2f}")

        print("\n" + "="*60)


# Example usage
def create_evaluation_batch_example():
    """Example of creating an evaluation batch."""
    from src.core.orchestrator import Orchestrator

    # Initialize framework
    framework = HumanEvaluationFramework()

    # Initialize orchestrator
    orchestrator = Orchestrator()

    # Sample queries
    queries = [
        "What is the capital of France?",
        "Explain how photosynthesis works.",
        "Write a Python function to reverse a string.",
        "What causes climate change?",
        "Explain the theory of relativity in simple terms.",
    ]

    # Create evaluation batch
    samples = framework.create_evaluation_batch(
        orchestrator,
        queries,
        batch_name="pilot_study",
        num_samples=5
    )

    # Export for MTurk
    framework.export_for_mturk(samples)

    # Create local evaluation interface
    framework.create_evaluation_interface_html(samples)

    print("\n✅ Evaluation batch created!")
    print("  - MTurk CSV ready for upload")
    print("  - Local HTML interface ready for evaluation")


if __name__ == "__main__":
    create_evaluation_batch_example()
