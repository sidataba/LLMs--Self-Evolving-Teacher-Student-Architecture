"""Self-evolution engine for autonomous system improvement."""

import uuid
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from loguru import logger

from src.models.base import ModelRole, ModelConfig
from src.models.student import StudentModel


@dataclass
class EvolutionMetrics:
    """Metrics tracking system evolution."""
    timestamp: str
    total_models: int
    avg_system_confidence: float
    domain_coverage: Dict[str, int]
    knowledge_gaps: List[str]
    evolution_cycle: int
    cost_reduction: float
    quality_improvement: float


@dataclass
class KnowledgeGap:
    """Identified gap in system knowledge."""
    gap_id: str
    domain: str
    query_pattern: str
    failure_count: int
    avg_confidence: float
    priority: float
    suggested_action: str


class SelfEvolutionEngine:
    """
    Autonomous self-evolution engine.

    Capabilities:
    - Autonomous domain discovery
    - Knowledge gap identification
    - Automatic student spawning for new domains
    - Curriculum generation
    - Meta-learning and strategy adaptation
    - Performance-based architecture evolution
    """

    def __init__(
        self,
        orchestrator,
        evolution_interval: int = 100,  # queries between evolution cycles
        gap_detection_threshold: float = 0.6,  # confidence below this indicates gap
        auto_spawn_students: bool = True,
        max_models_per_domain: int = 5,
    ):
        """
        Initialize self-evolution engine.

        Args:
            orchestrator: Reference to main orchestrator
            evolution_interval: Number of queries between evolution cycles
            gap_detection_threshold: Confidence threshold for gap detection
            auto_spawn_students: Whether to automatically create new students
            max_models_per_domain: Maximum students per domain
        """
        self.orchestrator = orchestrator
        self.evolution_interval = evolution_interval
        self.gap_detection_threshold = gap_detection_threshold
        self.auto_spawn_students = auto_spawn_students
        self.max_models_per_domain = max_models_per_domain

        # Evolution state
        self.evolution_cycle = 0
        self.queries_since_evolution = 0
        self.discovered_domains: Set[str] = set()
        self.knowledge_gaps: List[KnowledgeGap] = []
        self.evolution_history: List[EvolutionMetrics] = []

        # Meta-learning state
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self.routing_effectiveness: Dict[str, float] = {}

        logger.info(
            f"SelfEvolutionEngine initialized "
            f"(interval: {evolution_interval}, "
            f"auto_spawn: {auto_spawn_students})"
        )

    def record_query_result(self, query_result: Dict[str, Any]) -> None:
        """
        Record a query result for evolution analysis.

        Args:
            query_result: Result from orchestrator.process_query()
        """
        self.queries_since_evolution += 1

        # Extract metrics
        winner_score = query_result.get("winner_score", 0.0)
        routing_strategy = query_result.get("routing_strategy")

        # Track strategy performance
        self.strategy_performance[routing_strategy].append(winner_score)

        # Detect potential knowledge gap
        if winner_score < self.gap_detection_threshold:
            self._record_potential_gap(query_result)

        # Trigger evolution cycle if interval reached
        if self.queries_since_evolution >= self.evolution_interval:
            self.trigger_evolution_cycle()

    def trigger_evolution_cycle(self) -> Dict[str, Any]:
        """
        Trigger a complete evolution cycle.

        Returns:
            Dictionary with evolution results
        """
        self.evolution_cycle += 1
        logger.info(f"\n{'='*70}")
        logger.info(f"EVOLUTION CYCLE #{self.evolution_cycle} INITIATED")
        logger.info(f"{'='*70}\n")

        actions_taken = []

        # 1. Analyze knowledge gaps
        gaps_analysis = self._analyze_knowledge_gaps()
        actions_taken.append(("gaps_analyzed", len(gaps_analysis)))

        # 2. Discover new domains
        new_domains = self._discover_new_domains()
        if new_domains:
            actions_taken.append(("domains_discovered", new_domains))

        # 3. Spawn new students for gaps
        if self.auto_spawn_students:
            spawned = self._spawn_students_for_gaps(gaps_analysis)
            if spawned:
                actions_taken.append(("students_spawned", spawned))

        # 4. Optimize routing strategies
        routing_optimization = self._optimize_routing_strategies()
        actions_taken.append(("routing_optimized", routing_optimization))

        # 5. Prune underperforming models
        pruned = self._prune_underperforming_models()
        if pruned:
            actions_taken.append(("models_pruned", pruned))

        # 6. Generate curriculum for existing students
        curriculum = self._generate_curriculum()
        actions_taken.append(("curriculum_generated", len(curriculum)))

        # 7. Calculate evolution metrics
        metrics = self._calculate_evolution_metrics()
        self.evolution_history.append(metrics)

        # Reset counter
        self.queries_since_evolution = 0

        evolution_result = {
            "cycle": self.evolution_cycle,
            "actions_taken": actions_taken,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"\n{'='*70}")
        logger.info(f"EVOLUTION CYCLE #{self.evolution_cycle} COMPLETE")
        logger.info(f"Actions: {len(actions_taken)}")
        logger.info(f"{'='*70}\n")

        return evolution_result

    def _record_potential_gap(self, query_result: Dict[str, Any]) -> None:
        """Record a potential knowledge gap."""
        query_text = query_result.get("query_text", "")
        winner_score = query_result.get("winner_score", 0.0)

        # Extract domain from query (simple keyword matching)
        domain = self._infer_domain(query_text)

        # Check if we already have this gap
        existing_gap = next(
            (g for g in self.knowledge_gaps
             if g.domain == domain and self._similar_pattern(g.query_pattern, query_text)),
            None
        )

        if existing_gap:
            # Update existing gap
            existing_gap.failure_count += 1
            existing_gap.avg_confidence = (
                existing_gap.avg_confidence * 0.8 + winner_score * 0.2
            )
            existing_gap.priority = self._calculate_gap_priority(existing_gap)
        else:
            # Create new gap
            gap = KnowledgeGap(
                gap_id=str(uuid.uuid4())[:8],
                domain=domain,
                query_pattern=self._extract_pattern(query_text),
                failure_count=1,
                avg_confidence=winner_score,
                priority=0.5,
                suggested_action="spawn_student" if domain not in self.discovered_domains else "improve_existing"
            )
            gap.priority = self._calculate_gap_priority(gap)
            self.knowledge_gaps.append(gap)

    def _analyze_knowledge_gaps(self) -> List[KnowledgeGap]:
        """Analyze and prioritize knowledge gaps."""
        # Sort gaps by priority
        sorted_gaps = sorted(
            self.knowledge_gaps,
            key=lambda g: g.priority,
            reverse=True
        )

        # Log top gaps
        logger.info("Top Knowledge Gaps:")
        for i, gap in enumerate(sorted_gaps[:5], 1):
            logger.info(
                f"  {i}. {gap.domain}: {gap.query_pattern} "
                f"(failures: {gap.failure_count}, "
                f"confidence: {gap.avg_confidence:.3f}, "
                f"priority: {gap.priority:.3f})"
            )

        return sorted_gaps

    def _discover_new_domains(self) -> List[str]:
        """Discover new domains from query patterns."""
        # Analyze recent queries to find emerging domains
        status = self.orchestrator.get_system_status()

        # This is a simplified version - in production, use clustering or LLM analysis
        new_domains = []

        # Check for gaps in high-priority areas not yet covered
        for gap in self.knowledge_gaps:
            if gap.domain not in self.discovered_domains and gap.priority > 0.7:
                self.discovered_domains.add(gap.domain)
                new_domains.append(gap.domain)
                logger.info(f"✨ Discovered new domain: {gap.domain}")

        return new_domains

    def _spawn_students_for_gaps(
        self,
        gaps: List[KnowledgeGap]
    ) -> List[str]:
        """Spawn new student models to fill knowledge gaps."""
        spawned_students = []

        # Get current domain distribution
        domain_counts = defaultdict(int)
        for model in self.orchestrator.models.values():
            if model.role == ModelRole.STUDENT and model.domain:
                domain_counts[model.domain] += 1

        # Spawn students for high-priority gaps
        for gap in gaps[:5]:  # Top 5 gaps
            if gap.priority < 0.7:
                continue

            # Check if domain is under capacity
            if domain_counts[gap.domain] >= self.max_models_per_domain:
                continue

            # Find appropriate teacher
            teacher_id = self._find_teacher_for_domain(gap.domain)

            # Spawn new student
            student_id = f"student-{gap.domain}-evolved-{str(uuid.uuid4())[:6]}"

            student = self.orchestrator.add_student_model(
                model_id=student_id,
                domain=gap.domain,
                teacher_id=teacher_id,
            )

            spawned_students.append(student_id)
            domain_counts[gap.domain] += 1

            logger.info(
                f"🐣 Spawned new student: {student_id} "
                f"for domain: {gap.domain} (teacher: {teacher_id})"
            )

        return spawned_students

    def _optimize_routing_strategies(self) -> Dict[str, Any]:
        """Optimize routing strategies based on performance."""
        # Calculate average performance per strategy
        strategy_avg = {}
        for strategy, scores in self.strategy_performance.items():
            if scores:
                strategy_avg[strategy] = np.mean(scores)

        if not strategy_avg:
            return {"status": "no_data"}

        # Find best performing strategy
        best_strategy = max(strategy_avg.items(), key=lambda x: x[1])

        logger.info(f"Routing strategy performance:")
        for strategy, avg_score in strategy_avg.items():
            logger.info(f"  {strategy}: {avg_score:.3f}")

        # Adjust routing thresholds based on performance
        # If targeted routing performs well, lower similarity threshold
        if best_strategy[0] == "targeted" and best_strategy[1] > 0.8:
            current_threshold = self.orchestrator.query_router.similarity_threshold
            new_threshold = max(0.70, current_threshold - 0.02)
            self.orchestrator.query_router.similarity_threshold = new_threshold
            logger.info(f"📊 Adjusted similarity threshold: {current_threshold:.2f} → {new_threshold:.2f}")

        return {
            "best_strategy": best_strategy[0],
            "best_score": best_strategy[1],
            "all_strategies": strategy_avg,
        }

    def _prune_underperforming_models(self) -> List[str]:
        """Remove consistently underperforming models."""
        pruned = []

        # Only prune students (not teachers or supervisor)
        students = [
            (model_id, model)
            for model_id, model in self.orchestrator.models.items()
            if model.role == ModelRole.STUDENT
        ]

        for model_id, student in students:
            stats = student.get_statistics()

            # Prune if:
            # 1. Enough queries to judge (>50)
            # 2. Poor win rate (<20%)
            # 3. Low confidence (<50%)
            if (stats["total_queries"] > 50 and
                stats["win_rate"] < 0.20 and
                stats["avg_confidence"] < 0.50):

                # Remove from orchestrator
                del self.orchestrator.models[model_id]
                pruned.append(model_id)

                logger.warning(
                    f"✂️ Pruned underperforming student: {model_id} "
                    f"(win_rate: {stats['win_rate']:.2%}, "
                    f"confidence: {stats['avg_confidence']:.3f})"
                )

        return pruned

    def _generate_curriculum(self) -> List[Dict[str, Any]]:
        """Generate learning curriculum for students."""
        curriculum = []

        # For each student, identify learning priorities
        for model_id, model in self.orchestrator.models.items():
            if model.role != ModelRole.STUDENT:
                continue

            stats = model.get_statistics()

            # Analyze weaknesses
            if hasattr(model, 'learning_history') and model.learning_history:
                recent_history = model.learning_history[-20:]

                # Find areas needing improvement
                weak_areas = []
                for event in recent_history:
                    if event.get('improvement_needed'):
                        for metric, gap in event['improvement_needed'].items():
                            if gap > 0.2:
                                weak_areas.append(metric)

                if weak_areas:
                    curriculum_item = {
                        "student_id": model_id,
                        "focus_areas": list(set(weak_areas)),
                        "current_level": stats["avg_confidence"],
                        "target_level": min(0.85, stats["avg_confidence"] + 0.15),
                        "suggested_samples": 50,
                    }
                    curriculum.append(curriculum_item)

        return curriculum

    def _calculate_evolution_metrics(self) -> EvolutionMetrics:
        """Calculate comprehensive evolution metrics."""
        status = self.orchestrator.get_system_status()

        # Calculate average system confidence
        all_confidences = [
            stats["avg_confidence"]
            for stats in status["model_stats"].values()
        ]
        avg_system_confidence = np.mean(all_confidences) if all_confidences else 0.0

        # Domain coverage
        domain_coverage = defaultdict(int)
        for model_id, model in self.orchestrator.models.items():
            if model.domain:
                domain_coverage[model.domain] += 1

        # Knowledge gaps summary
        knowledge_gaps = [
            f"{gap.domain}:{gap.query_pattern[:20]}"
            for gap in self.knowledge_gaps[:5]
        ]

        # Calculate cost reduction (simplified)
        # In production, track actual API costs
        total_models = len(self.orchestrator.models)
        student_ratio = sum(
            1 for m in self.orchestrator.models.values()
            if m.role in [ModelRole.STUDENT, ModelRole.TA]
        ) / total_models if total_models > 0 else 0
        cost_reduction = student_ratio * 0.7  # Students cost ~70% less

        # Quality improvement
        if len(self.evolution_history) > 0:
            prev_confidence = self.evolution_history[-1].avg_system_confidence
            quality_improvement = avg_system_confidence - prev_confidence
        else:
            quality_improvement = 0.0

        metrics = EvolutionMetrics(
            timestamp=datetime.now().isoformat(),
            total_models=total_models,
            avg_system_confidence=avg_system_confidence,
            domain_coverage=dict(domain_coverage),
            knowledge_gaps=knowledge_gaps,
            evolution_cycle=self.evolution_cycle,
            cost_reduction=cost_reduction,
            quality_improvement=quality_improvement,
        )

        return metrics

    def _infer_domain(self, query: str) -> str:
        """Infer domain from query text."""
        query_lower = query.lower()

        # Math keywords
        if any(kw in query_lower for kw in ["math", "calculate", "equation", "solve", "algebra"]):
            return "mathematics"

        # Science keywords
        if any(kw in query_lower for kw in ["science", "physics", "chemistry", "biology", "experiment"]):
            return "science"

        # Programming keywords
        if any(kw in query_lower for kw in ["code", "program", "python", "javascript", "algorithm", "debug"]):
            return "programming"

        # Business keywords
        if any(kw in query_lower for kw in ["business", "marketing", "sales", "finance", "strategy"]):
            return "business"

        # Medical keywords
        if any(kw in query_lower for kw in ["medical", "health", "disease", "treatment", "diagnosis"]):
            return "medical"

        return "general"

    def _extract_pattern(self, query: str) -> str:
        """Extract query pattern."""
        # Simplified - in production, use NLP techniques
        words = query.lower().split()
        if len(words) > 5:
            return " ".join(words[:5]) + "..."
        return query[:50]

    def _similar_pattern(self, pattern1: str, pattern2: str) -> bool:
        """Check if two patterns are similar."""
        # Simplified similarity check
        words1 = set(pattern1.lower().split())
        words2 = set(pattern2.lower().split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap > 0.6

    def _calculate_gap_priority(self, gap: KnowledgeGap) -> float:
        """Calculate priority score for a knowledge gap."""
        # Priority based on:
        # - Failure frequency (40%)
        # - Confidence gap (40%)
        # - Domain importance (20%)

        frequency_score = min(1.0, gap.failure_count / 10)
        confidence_gap = 1.0 - gap.avg_confidence
        domain_importance = 0.5  # Could be weighted by domain

        priority = (
            frequency_score * 0.4 +
            confidence_gap * 0.4 +
            domain_importance * 0.2
        )

        return priority

    def _find_teacher_for_domain(self, domain: str) -> Optional[str]:
        """Find best teacher for a domain."""
        domain_teachers = [
            model_id for model_id, model in self.orchestrator.models.items()
            if model.role == ModelRole.TEACHER and model.domain == domain
        ]

        if domain_teachers:
            return domain_teachers[0]

        # Fallback to general teacher
        general_teachers = [
            model_id for model_id, model in self.orchestrator.models.items()
            if model.role == ModelRole.TEACHER
        ]

        return general_teachers[0] if general_teachers else None

    def get_evolution_report(self) -> Dict[str, Any]:
        """Generate comprehensive evolution report."""
        if not self.evolution_history:
            return {"status": "no_evolution_cycles"}

        latest_metrics = self.evolution_history[-1]

        return {
            "current_cycle": self.evolution_cycle,
            "total_cycles": len(self.evolution_history),
            "latest_metrics": {
                "total_models": latest_metrics.total_models,
                "avg_confidence": latest_metrics.avg_system_confidence,
                "domain_coverage": latest_metrics.domain_coverage,
                "cost_reduction": latest_metrics.cost_reduction,
                "quality_improvement": latest_metrics.quality_improvement,
            },
            "discovered_domains": list(self.discovered_domains),
            "active_knowledge_gaps": len(self.knowledge_gaps),
            "evolution_trajectory": [
                {
                    "cycle": m.evolution_cycle,
                    "confidence": m.avg_system_confidence,
                    "models": m.total_models,
                }
                for m in self.evolution_history
            ],
        }
