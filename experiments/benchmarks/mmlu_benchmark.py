"""
MMLU (Massive Multitask Language Understanding) Benchmark

Evaluates language understanding across 57 tasks covering:
- STEM (mathematics, physics, chemistry, biology, computer science)
- Humanities (history, philosophy, law)
- Social Sciences (economics, psychology, sociology)
- Other (medicine, business, miscellaneous)

Dataset: https://github.com/hendrycks/test
Paper: "Measuring Massive Multitask Language Understanding" (ICLR 2021)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
import logging

from experiments.benchmarks.base_benchmark import BaseBenchmark

logger = logging.getLogger(__name__)


class MMLUBenchmark(BaseBenchmark):
    """
    MMLU benchmark implementation.

    Format: Multiple-choice questions with 4 options (A, B, C, D)
    """

    # Subject categories
    STEM_SUBJECTS = [
        "abstract_algebra", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics", "electrical_engineering",
        "elementary_mathematics", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning"
    ]

    HUMANITIES_SUBJECTS = [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions"
    ]

    SOCIAL_SCIENCES_SUBJECTS = [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "human_sexuality", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy"
    ]

    OTHER_SUBJECTS = [
        "anatomy", "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition", "professional_accounting",
        "professional_medicine", "virology"
    ]

    def __init__(
        self,
        dataset_path: Path = None,
        subjects: List[str] = None,
        **kwargs
    ):
        """
        Initialize MMLU benchmark.

        Args:
            dataset_path: Path to MMLU dataset directory
            subjects: Specific subjects to test (None = all)
            **kwargs: Passed to BaseBenchmark
        """
        super().__init__(name="MMLU", dataset_path=dataset_path, **kwargs)

        self.subjects = subjects or self._get_all_subjects()

        logger.info(f"Initialized MMLU with {len(self.subjects)} subjects")

    def _get_all_subjects(self) -> List[str]:
        """Get all MMLU subjects."""
        return (
            self.STEM_SUBJECTS +
            self.HUMANITIES_SUBJECTS +
            self.SOCIAL_SCIENCES_SUBJECTS +
            self.OTHER_SUBJECTS
        )

    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load MMLU dataset.

        If dataset_path is provided, loads from files.
        Otherwise, creates sample dataset for demonstration.
        """
        if self.dataset_path and self.dataset_path.exists():
            return self._load_from_files()
        else:
            logger.warning("No dataset path provided, using sample data")
            return self._create_sample_dataset()

    def _load_from_files(self) -> List[Dict[str, Any]]:
        """Load MMLU from CSV/JSON files."""
        examples = []

        for subject in self.subjects:
            subject_file = self.dataset_path / f"{subject}_test.csv"

            if not subject_file.exists():
                logger.warning(f"Subject file not found: {subject_file}")
                continue

            # Parse CSV
            with open(subject_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue

                question = parts[0]
                choices = parts[1:5]  # A, B, C, D
                answer = parts[5]  # Correct choice

                examples.append({
                    'query': self._format_question(question, choices),
                    'answer': answer,
                    'subject': subject,
                    'domain': self._get_domain(subject),
                    'choices': choices,
                })

        logger.info(f"Loaded {len(examples)} MMLU examples")
        return examples

    def _create_sample_dataset(self) -> List[Dict[str, Any]]:
        """Create sample MMLU dataset for testing."""
        samples = [
            # Mathematics
            {
                'query': "What is the derivative of x^2?\n(A) x\n(B) 2x\n(C) x^2\n(D) 2x^2",
                'answer': 'B',
                'subject': 'college_mathematics',
                'domain': 'STEM',
                'choices': ['x', '2x', 'x^2', '2x^2'],
            },
            {
                'query': "What is the integral of 1/x?\n(A) x^2\n(B) ln(x)\n(C) e^x\n(D) 1/x^2",
                'answer': 'B',
                'subject': 'college_mathematics',
                'domain': 'STEM',
                'choices': ['x^2', 'ln(x)', 'e^x', '1/x^2'],
            },
            # Physics
            {
                'query': "What is Newton's second law?\n(A) F = ma\n(B) E = mc^2\n(C) P = mv\n(D) W = Fd",
                'answer': 'A',
                'subject': 'college_physics',
                'domain': 'STEM',
                'choices': ['F = ma', 'E = mc^2', 'P = mv', 'W = Fd'],
            },
            # Computer Science
            {
                'query': "What is the time complexity of binary search?\n(A) O(n)\n(B) O(log n)\n(C) O(n log n)\n(D) O(n^2)",
                'answer': 'B',
                'subject': 'computer_science',
                'domain': 'STEM',
                'choices': ['O(n)', 'O(log n)', 'O(n log n)', 'O(n^2)'],
            },
            # History
            {
                'query': "Who was the first president of the United States?\n(A) Thomas Jefferson\n(B) George Washington\n(C) John Adams\n(D) Benjamin Franklin",
                'answer': 'B',
                'subject': 'us_history',
                'domain': 'Humanities',
                'choices': ['Thomas Jefferson', 'George Washington', 'John Adams', 'Benjamin Franklin'],
            },
            # Philosophy
            {
                'query': "Who wrote 'The Republic'?\n(A) Aristotle\n(B) Socrates\n(C) Plato\n(D) Epicurus",
                'answer': 'C',
                'subject': 'philosophy',
                'domain': 'Humanities',
                'choices': ['Aristotle', 'Socrates', 'Plato', 'Epicurus'],
            },
            # Economics
            {
                'query': "What does GDP stand for?\n(A) General Domestic Product\n(B) Gross Domestic Product\n(C) Global Development Plan\n(D) Government Debt Provision",
                'answer': 'B',
                'subject': 'macroeconomics',
                'domain': 'Social Sciences',
                'choices': ['General Domestic Product', 'Gross Domestic Product', 'Global Development Plan', 'Government Debt Provision'],
            },
            # Medicine
            {
                'query': "What is the largest organ in the human body?\n(A) Liver\n(B) Heart\n(C) Brain\n(D) Skin",
                'answer': 'D',
                'subject': 'anatomy',
                'domain': 'Other',
                'choices': ['Liver', 'Heart', 'Brain', 'Skin'],
            },
        ]

        logger.info(f"Created {len(samples)} sample MMLU examples")
        return samples

    def _format_question(
        self,
        question: str,
        choices: List[str]
    ) -> str:
        """Format question with multiple choice options."""
        formatted = question + "\n"
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # A, B, C, D
            formatted += f"({letter}) {choice}\n"
        return formatted.strip()

    def _get_domain(self, subject: str) -> str:
        """Get domain category for subject."""
        if subject in self.STEM_SUBJECTS:
            return "STEM"
        elif subject in self.HUMANITIES_SUBJECTS:
            return "Humanities"
        elif subject in self.SOCIAL_SCIENCES_SUBJECTS:
            return "Social Sciences"
        else:
            return "Other"

    def evaluate_response(
        self,
        response: str,
        correct_answer: str,
        example: Dict[str, Any]
    ) -> bool:
        """
        Evaluate MMLU response.

        Accepts:
        - Single letter: "B"
        - Letter with parentheses: "(B)"
        - Full answer: "The answer is B"
        """
        # Extract letter from response
        response_upper = response.upper()

        # Try to find answer letter (A, B, C, or D)
        match = re.search(r'\b([A-D])\b', response_upper)

        if match:
            extracted_answer = match.group(1)
            return extracted_answer == correct_answer.upper()

        # If no clear letter, check if response contains the correct choice text
        correct_choice_idx = ord(correct_answer.upper()) - ord('A')
        if 0 <= correct_choice_idx < len(example['choices']):
            correct_choice_text = example['choices'][correct_choice_idx]
            return correct_choice_text.lower() in response.lower()

        return False


def run_mmlu_benchmark(
    orchestrator,
    dataset_path: Path = None,
    max_samples: int = None,
    subjects: List[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run MMLU benchmark.

    Args:
        orchestrator: System orchestrator
        dataset_path: Path to MMLU dataset
        max_samples: Maximum samples to evaluate
        subjects: Specific subjects to test

    Returns:
        Dictionary with results and statistics
    """
    benchmark = MMLUBenchmark(
        dataset_path=dataset_path,
        subjects=subjects,
        max_samples=max_samples,
    )

    stats = benchmark.run(orchestrator)
    benchmark.print_report(stats)

    return {
        'statistics': stats.to_dict(),
        'results': benchmark.results,
    }
