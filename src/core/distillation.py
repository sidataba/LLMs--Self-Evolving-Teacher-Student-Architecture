"""Advanced knowledge distillation system for model learning and improvement."""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from loguru import logger
import numpy as np


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    temperature: float = 2.0
    alpha: float = 0.7  # Weight for distillation loss vs hard target loss
    learning_rate: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 3
    max_samples_per_session: int = 1000
    min_teacher_confidence: float = 0.85
    use_soft_targets: bool = True
    distillation_strategy: str = "response"  # response, reasoning, multi-aspect


@dataclass
class DistillationSample:
    """A single training sample for distillation."""
    query: str
    teacher_response: str
    teacher_confidence: float
    teacher_reasoning: Optional[str]
    student_response: Optional[str]
    domain: str
    timestamp: str
    metrics: Dict[str, float]


class KnowledgeDistillation:
    """
    Advanced knowledge distillation system.
    Implements multiple distillation strategies for transferring knowledge
    from teachers to students.
    """

    def __init__(
        self,
        config: Optional[DistillationConfig] = None,
        storage_path: str = "./data/distillation",
    ):
        """
        Initialize knowledge distillation system.

        Args:
            config: Distillation configuration
            storage_path: Path to store distillation data
        """
        self.config = config or DistillationConfig()
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

        # Storage for distillation samples
        self.samples_buffer: Dict[str, List[DistillationSample]] = {}
        self.distillation_history = []

        logger.info(
            f"KnowledgeDistillation initialized "
            f"(strategy: {self.config.distillation_strategy}, "
            f"temperature: {self.config.temperature})"
        )

    def collect_sample(
        self,
        query: str,
        teacher_response: str,
        teacher_confidence: float,
        teacher_reasoning: Optional[str],
        student_response: Optional[str],
        student_id: str,
        domain: str,
        metrics: Dict[str, float],
    ) -> None:
        """
        Collect a training sample for distillation.

        Args:
            query: The input query
            teacher_response: Teacher's winning response
            teacher_confidence: Teacher's confidence score
            teacher_reasoning: Teacher's reasoning process
            student_response: Student's response (for comparison)
            student_id: Student model ID
            domain: Domain of the query
            metrics: Evaluation metrics
        """
        # Only collect high-quality teacher samples
        if teacher_confidence < self.config.min_teacher_confidence:
            return

        sample = DistillationSample(
            query=query,
            teacher_response=teacher_response,
            teacher_confidence=teacher_confidence,
            teacher_reasoning=teacher_reasoning,
            student_response=student_response,
            domain=domain,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
        )

        # Add to student's buffer
        if student_id not in self.samples_buffer:
            self.samples_buffer[student_id] = []

        self.samples_buffer[student_id].append(sample)

        logger.debug(
            f"Collected distillation sample for {student_id} "
            f"(buffer size: {len(self.samples_buffer[student_id])})"
        )

    def should_trigger_distillation(self, student_id: str) -> bool:
        """Check if we should trigger distillation for a student."""
        if student_id not in self.samples_buffer:
            return False

        buffer_size = len(self.samples_buffer[student_id])
        return buffer_size >= self.config.batch_size

    def distill_knowledge(
        self,
        student_model,
        student_id: str,
    ) -> Dict[str, Any]:
        """
        Perform knowledge distillation for a student model.

        Args:
            student_model: The student model to improve
            student_id: Student model ID

        Returns:
            Dictionary with distillation results
        """
        if student_id not in self.samples_buffer:
            return {"status": "no_samples", "samples_used": 0}

        samples = self.samples_buffer[student_id]

        if len(samples) < self.config.batch_size:
            return {"status": "insufficient_samples", "samples_used": 0}

        logger.info(
            f"Starting knowledge distillation for {student_id} "
            f"with {len(samples)} samples"
        )

        # Perform distillation based on strategy
        if self.config.distillation_strategy == "response":
            result = self._distill_response_based(student_model, samples)
        elif self.config.distillation_strategy == "reasoning":
            result = self._distill_reasoning_based(student_model, samples)
        elif self.config.distillation_strategy == "multi-aspect":
            result = self._distill_multi_aspect(student_model, samples)
        else:
            result = self._distill_response_based(student_model, samples)

        # Record distillation event
        distillation_event = {
            "student_id": student_id,
            "timestamp": datetime.now().isoformat(),
            "samples_used": len(samples),
            "strategy": self.config.distillation_strategy,
            "result": result,
        }

        self.distillation_history.append(distillation_event)

        # Save samples for future reference
        self._save_samples(student_id, samples)

        # Clear buffer (keep last 20% for overlap)
        keep_count = len(samples) // 5
        self.samples_buffer[student_id] = samples[-keep_count:]

        logger.info(
            f"Distillation complete for {student_id}. "
            f"Improvement: {result.get('improvement', 0):.3f}"
        )

        return distillation_event

    def _distill_response_based(
        self,
        student_model,
        samples: List[DistillationSample],
    ) -> Dict[str, Any]:
        """
        Response-based distillation: Learn to mimic teacher's responses.

        This is a simplified version. In production, you would:
        1. Fine-tune the actual model weights
        2. Use proper loss functions (KL divergence, etc.)
        3. Implement mini-batch training
        """
        # For mock models, simulate improvement
        initial_confidence = student_model.base_confidence

        # Calculate improvement based on sample quality
        avg_teacher_confidence = np.mean([s.teacher_confidence for s in samples])
        improvement = (avg_teacher_confidence - initial_confidence) * self.config.alpha * 0.1

        # Apply improvement
        student_model.base_confidence = min(0.95, initial_confidence + improvement)

        return {
            "strategy": "response_based",
            "initial_confidence": initial_confidence,
            "final_confidence": student_model.base_confidence,
            "improvement": improvement,
            "samples_processed": len(samples),
            "avg_teacher_confidence": avg_teacher_confidence,
        }

    def _distill_reasoning_based(
        self,
        student_model,
        samples: List[DistillationSample],
    ) -> Dict[str, Any]:
        """
        Reasoning-based distillation: Learn teacher's reasoning process.

        Focuses on the chain of thought and intermediate steps.
        """
        initial_confidence = student_model.base_confidence

        # Extract reasoning patterns
        reasoning_quality = []
        for sample in samples:
            if sample.teacher_reasoning:
                # Analyze reasoning complexity and correctness
                quality = sample.teacher_confidence * len(sample.teacher_reasoning.split())
                reasoning_quality.append(quality)

        if reasoning_quality:
            avg_quality = np.mean(reasoning_quality)
            # Normalize and apply
            improvement = (avg_quality / 1000) * self.config.alpha * 0.15
            student_model.base_confidence = min(0.95, initial_confidence + improvement)

        return {
            "strategy": "reasoning_based",
            "initial_confidence": initial_confidence,
            "final_confidence": student_model.base_confidence,
            "improvement": student_model.base_confidence - initial_confidence,
            "reasoning_samples": len(reasoning_quality),
        }

    def _distill_multi_aspect(
        self,
        student_model,
        samples: List[DistillationSample],
    ) -> Dict[str, Any]:
        """
        Multi-aspect distillation: Learn from responses, reasoning, and metrics.

        Most comprehensive approach combining multiple signals.
        """
        initial_confidence = student_model.base_confidence

        # 1. Response quality
        response_improvement = self._distill_response_based(student_model, samples)

        # 2. Reasoning patterns
        student_model.base_confidence = initial_confidence  # Reset
        reasoning_improvement = self._distill_reasoning_based(student_model, samples)

        # 3. Domain-specific metrics
        domain_metrics = {}
        for sample in samples:
            domain = sample.domain
            if domain not in domain_metrics:
                domain_metrics[domain] = []
            domain_metrics[domain].append(sample.metrics)

        # Combine improvements
        total_improvement = (
            response_improvement["improvement"] * 0.4 +
            reasoning_improvement["improvement"] * 0.4 +
            0.02  # Base learning rate
        )

        student_model.base_confidence = min(0.95, initial_confidence + total_improvement)

        return {
            "strategy": "multi_aspect",
            "initial_confidence": initial_confidence,
            "final_confidence": student_model.base_confidence,
            "improvement": total_improvement,
            "response_component": response_improvement["improvement"],
            "reasoning_component": reasoning_improvement["improvement"],
            "domain_coverage": len(domain_metrics),
        }

    def _save_samples(self, student_id: str, samples: List[DistillationSample]) -> None:
        """Save distillation samples for analysis."""
        filename = os.path.join(
            self.storage_path,
            f"{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        samples_data = [
            {
                "query": s.query,
                "teacher_response": s.teacher_response,
                "teacher_confidence": s.teacher_confidence,
                "student_response": s.student_response,
                "domain": s.domain,
                "timestamp": s.timestamp,
                "metrics": s.metrics,
            }
            for s in samples
        ]

        with open(filename, 'w') as f:
            json.dump(samples_data, f, indent=2)

        logger.debug(f"Saved {len(samples)} samples to {filename}")

    def get_distillation_statistics(self) -> Dict[str, Any]:
        """Get statistics about distillation activities."""
        total_events = len(self.distillation_history)

        if total_events == 0:
            return {
                "total_events": 0,
                "total_samples_processed": 0,
                "students_trained": 0,
            }

        total_samples = sum(
            event["samples_used"] for event in self.distillation_history
        )

        students_trained = len(set(
            event["student_id"] for event in self.distillation_history
        ))

        avg_improvement = np.mean([
            event["result"].get("improvement", 0)
            for event in self.distillation_history
        ])

        return {
            "total_events": total_events,
            "total_samples_processed": total_samples,
            "students_trained": students_trained,
            "avg_improvement": avg_improvement,
            "recent_events": self.distillation_history[-5:],
        }

    def export_training_data(
        self,
        output_path: str,
        format: str = "jsonl",
    ) -> str:
        """
        Export collected samples as training data.

        Args:
            output_path: Path to save training data
            format: Format (jsonl, csv, parquet)

        Returns:
            Path to exported file
        """
        all_samples = []
        for student_id, samples in self.samples_buffer.items():
            for sample in samples:
                all_samples.append({
                    "student_id": student_id,
                    "query": sample.query,
                    "teacher_response": sample.teacher_response,
                    "teacher_confidence": sample.teacher_confidence,
                    "domain": sample.domain,
                    "timestamp": sample.timestamp,
                })

        if format == "jsonl":
            output_file = f"{output_path}/training_data.jsonl"
            with open(output_file, 'w') as f:
                for sample in all_samples:
                    f.write(json.dumps(sample) + '\n')

        logger.info(f"Exported {len(all_samples)} training samples to {output_file}")

        return output_file
