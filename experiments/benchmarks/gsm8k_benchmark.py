"""
GSM8K (Grade School Math 8K) Benchmark

Evaluates mathematical reasoning on grade-school level word problems.
Requires multi-step reasoning and arithmetic.

Dataset: https://github.com/openai/grade-school-math
Paper: "Training Verifiers to Solve Math Word Problems" (2021)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
import logging

from experiments.benchmarks.base_benchmark import BaseBenchmark

logger = logging.getLogger(__name__)


class GSM8KBenchmark(BaseBenchmark):
    """
    GSM8K benchmark implementation.

    Tests mathematical reasoning with word problems requiring:
    - Multi-step reasoning
    - Arithmetic operations
    - Problem decomposition
    """

    def __init__(
        self,
        dataset_path: Path = None,
        **kwargs
    ):
        """Initialize GSM8K benchmark."""
        super().__init__(name="GSM8K", dataset_path=dataset_path, **kwargs)

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load GSM8K dataset."""
        if self.dataset_path and self.dataset_path.exists():
            return self._load_from_file()
        else:
            logger.warning("No dataset path provided, using sample data")
            return self._create_sample_dataset()

    def _load_from_file(self) -> List[Dict[str, Any]]:
        """Load from GSM8K JSONL file."""
        examples = []

        with open(self.dataset_path, 'r') as f:
            for line in f:
                item = json.loads(line)

                # Extract numerical answer from solution
                answer = self._extract_answer(item['answer'])

                examples.append({
                    'query': item['question'],
                    'answer': answer,
                    'solution': item['answer'],  # Full solution with reasoning
                    'domain': 'mathematics',
                })

        logger.info(f"Loaded {len(examples)} GSM8K examples")
        return examples

    def _create_sample_dataset(self) -> List[Dict[str, Any]]:
        """Create sample GSM8K dataset."""
        samples = [
            {
                'query': "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                'answer': '18',
                'solution': "Janet sells 16 - 3 - 4 = 9 duck eggs a day. She makes 9 * 2 = $18 every day at the farmer's market. #### 18",
                'domain': 'mathematics',
            },
            {
                'query': "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                'answer': '3',
                'solution': "It takes 2/2=1 bolt of white fiber. So the total amount of fabric is 2+1=3 bolts of fabric. #### 3",
                'domain': 'mathematics',
            },
            {
                'query': "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
                'answer': '70000',
                'solution': "The cost of the house and repairs came out to 80,000+50,000=$130,000. He increased the value of the house by 80,000*1.5=120,000. So the new value of the house is 120,000+80,000=$200,000. So he made a profit of 200,000-130,000=$70,000. #### 70000",
                'domain': 'mathematics',
            },
            {
                'query': "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                'answer': '6',
                'solution': "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted. #### 6",
                'domain': 'mathematics',
            },
            {
                'query': "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                'answer': '5',
                'solution': "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5",
                'domain': 'mathematics',
            },
            {
                'query': "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                'answer': '39',
                'solution': "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39",
                'domain': 'mathematics',
            },
            {
                'query': "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
                'answer': '8',
                'solution': "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8",
                'domain': 'mathematics',
            },
            {
                'query': "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
                'answer': '8',
                'solution': "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. #### 8",
                'domain': 'mathematics',
            },
        ]

        logger.info(f"Created {len(samples)} sample GSM8K examples")
        return samples

    def _extract_answer(self, solution: str) -> str:
        """
        Extract numerical answer from solution.

        GSM8K solutions end with #### followed by the answer.
        Example: "... So the answer is 42. #### 42"
        """
        # Try to find answer after ####
        match = re.search(r'####\s*([0-9,]+)', solution)
        if match:
            # Remove commas from large numbers
            return match.group(1).replace(',', '')

        # Fallback: try to find last number in solution
        numbers = re.findall(r'\b\d+\b', solution)
        if numbers:
            return numbers[-1]

        return ""

    def evaluate_response(
        self,
        response: str,
        correct_answer: str,
        example: Dict[str, Any]
    ) -> bool:
        """
        Evaluate GSM8K response.

        Extracts numerical answer from response and compares.
        """
        # Extract number from response
        response_answer = self._extract_numerical_answer(response)

        if not response_answer:
            return False

        # Compare as numbers (handle different formats)
        try:
            response_num = float(response_answer)
            correct_num = float(correct_answer)

            # Allow small floating point differences
            return abs(response_num - correct_num) < 0.01

        except ValueError:
            # Fallback to string comparison
            return response_answer == correct_answer

    def _extract_numerical_answer(self, response: str) -> str:
        """
        Extract numerical answer from response.

        Looks for:
        - "The answer is X"
        - "= X" or "equals X"
        - Numbers at the end
        """
        # Try common patterns
        patterns = [
            r'(?:answer|result|total)(?:\s+is)?\s*[:\$]?\s*([0-9,]+(?:\.\d+)?)',
            r'=\s*\$?\s*([0-9,]+(?:\.\d+)?)',
            r'\$\s*([0-9,]+(?:\.\d+)?)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).replace(',', '')

        # Fallback: last number in response
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
        if numbers:
            return numbers[-1]

        return ""


def run_gsm8k_benchmark(
    orchestrator,
    dataset_path: Path = None,
    max_samples: int = None,
) -> Dict[str, Any]:
    """
    Convenience function to run GSM8K benchmark.

    Args:
        orchestrator: System orchestrator
        dataset_path: Path to GSM8K dataset
        max_samples: Maximum samples to evaluate

    Returns:
        Dictionary with results and statistics
    """
    benchmark = GSM8KBenchmark(
        dataset_path=dataset_path,
        max_samples=max_samples,
    )

    stats = benchmark.run(orchestrator)
    benchmark.print_report(stats)

    return {
        'statistics': stats.to_dict(),
        'results': benchmark.results,
    }
