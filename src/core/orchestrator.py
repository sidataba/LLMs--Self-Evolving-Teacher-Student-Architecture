"""Main orchestrator for the self-evolving teacher-student system."""

import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
from loguru import logger

from src.models.base import BaseModel, ModelConfig, ModelRole
from src.models.supervisor import SupervisorModel
from src.models.teacher import TeacherModel
from src.models.student import StudentModel
from src.database.vector_store import VectorStore
from src.database.metrics_store import MetricsStore
from src.routing.query_router import QueryRouter
from src.evaluation.evaluator import ResponseEvaluator
from src.evaluation.feedback import FeedbackLoop
from src.core.promotion import PromotionSystem


class Orchestrator:
    """
    Main orchestrator that coordinates all components of the system.
    Manages query flow, model coordination, evaluation, and learning.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the orchestrator.

        Args:
            config_path: Path to configuration file (YAML)
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        logger.info("Initializing Self-Evolving Teacher-Student System...")

        # Databases
        self.vector_store = VectorStore(
            db_path=self.config["vector_database"]["path"],
            embedding_model=self.config["vector_database"]["embedding_model"],
        )

        self.metrics_store = MetricsStore(
            db_path="./data/metrics"
        )

        # Routing
        self.query_router = QueryRouter(
            vector_store=self.vector_store,
            metrics_store=self.metrics_store,
            similarity_threshold=self.config["routing"]["similarity_threshold"],
            novel_threshold=self.config["routing"]["novel_query_threshold"],
        )

        # Evaluation
        self.evaluator = ResponseEvaluator(
            metric_weights=self.config["evaluation"]["weights"],
        )

        # Feedback
        self.feedback_loop = FeedbackLoop(
            enable_distillation=self.config["feedback"]["distillation"]["enabled"],
            distillation_temperature=self.config["feedback"]["distillation"]["temperature"],
            distillation_alpha=self.config["feedback"]["distillation"]["alpha"],
        )

        # Promotion
        self.promotion_system = PromotionSystem(
            metrics_store=self.metrics_store,
        )

        # Models
        self.models: Dict[str, BaseModel] = {}
        self.supervisor: Optional[SupervisorModel] = None

        self._initialize_models()

        # Statistics
        self.query_count = 0

        logger.info(
            f"Orchestrator initialized with {len(self.models)} models "
            f"(Supervisor: {self.supervisor.model_id if self.supervisor else 'None'})"
        )

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path is None:
            config_path = "./config/default_config.yaml"

        config_file = Path(config_path)

        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._get_default_config()

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "models": {
                "supervisor": {"model_id": "supervisor", "model_type": "mock"},
                "teachers": [],
                "students": [],
            },
            "vector_database": {
                "path": "./data/vector_db",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "routing": {
                "similarity_threshold": 0.80,
                "novel_query_threshold": 0.50,
            },
            "evaluation": {
                "weights": {
                    "relevance": 0.3,
                    "correctness": 0.4,
                    "completeness": 0.2,
                    "clarity": 0.1,
                },
            },
            "feedback": {
                "distillation": {
                    "enabled": True,
                    "temperature": 2.0,
                    "alpha": 0.7,
                },
            },
        }

    def _initialize_models(self) -> None:
        """Initialize all models from configuration."""
        # Initialize supervisor
        supervisor_config = self.config["models"]["supervisor"]
        model_config = ModelConfig(
            model_id=supervisor_config["model_id"],
            model_type=supervisor_config.get("model_type", "mock"),
            role=ModelRole.SUPERVISOR,
            max_tokens=supervisor_config.get("max_tokens", 2048),
            temperature=supervisor_config.get("temperature", 0.3),
        )

        self.supervisor = SupervisorModel(model_config)
        self.models[self.supervisor.model_id] = self.supervisor

        # Initialize teachers
        for teacher_config in self.config["models"].get("teachers", []):
            model_config = ModelConfig(
                model_id=teacher_config["model_id"],
                model_type=teacher_config.get("model_type", "mock"),
                role=ModelRole.TEACHER,
                domain=teacher_config.get("domain"),
                specialization=teacher_config.get("specialization", []),
            )

            teacher = TeacherModel(model_config)
            self.models[teacher.model_id] = teacher

        # Initialize students
        for student_config in self.config["models"].get("students", []):
            model_config = ModelConfig(
                model_id=student_config["model_id"],
                model_type=student_config.get("model_type", "mock"),
                role=ModelRole.STUDENT,
                domain=student_config.get("domain"),
                metadata={
                    "teacher_id": student_config.get("teacher_id"),
                    "learning_rate": student_config.get("learning_rate", 0.01),
                },
            )

            student = StudentModel(model_config)
            self.models[student.model_id] = student

        logger.info(
            f"Initialized {len(self.models)} models: "
            f"1 supervisor, "
            f"{len([m for m in self.models.values() if m.role == ModelRole.TEACHER])} teachers, "
            f"{len([m for m in self.models.values() if m.role == ModelRole.STUDENT])} students"
        )

    def process_query(self, query_text: str, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a query through the entire system.

        Args:
            query_text: The user query
            domain: Optional domain hint

        Returns:
            Dictionary with final response and metadata
        """
        self.query_count += 1

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Query #{self.query_count}: {query_text[:50]}...")
        logger.info(f"{'='*60}")

        # 1. Route the query
        query, routing_decision = self.query_router.process_query(query_text, domain)

        logger.info(f"Routing: {routing_decision.routing_strategy} (similarity: {routing_decision.similarity_score:.3f})")

        # 2. Get responses from models
        responses = self._get_model_responses(query, routing_decision)

        logger.info(f"Received {len(responses)} responses")

        # 3. Evaluate responses
        evaluation_results = self.evaluator.evaluate_responses(
            query=query.query_text,
            responses=responses,
        )

        # 4. Select winner
        winner = evaluation_results[0] if evaluation_results else None

        if winner:
            logger.info(f"Winner: {winner.model_id} (score: {winner.final_score:.3f})")

            # Update query with winner info
            self.query_router.update_query_result(
                query_id=query.query_id,
                winning_model=winner.model_id,
                confidence=winner.final_score,
                evaluation_metrics=winner.metrics,
            )

            # Log to metrics store
            for i, result in enumerate(evaluation_results):
                response = responses[i]
                self.metrics_store.log_query_response(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    model_id=result.model_id,
                    response=response.response_text,
                    confidence=result.final_score,
                    metrics=result.metrics,
                    is_winner=(result.model_id == winner.model_id),
                )

        # 5. Distribute feedback
        feedback_result = self.feedback_loop.distribute_feedback(
            query=query.query_text,
            models=self.models,
            responses=responses,
            evaluations=evaluation_results,
        )

        logger.info(f"Feedback distributed to {feedback_result['feedback_sent']} models")

        # 6. Check for promotions
        promotions = self.promotion_system.check_promotions(self.models)

        if promotions:
            logger.info(f"Promotions: {len(promotions)} model(s) promoted!")
            for promo in promotions:
                logger.info(f"  {promo['model_id']}: {promo['from_role']} -> {promo['to_role']}")

        # 7. Prepare final response
        final_response = self._prepare_final_response(
            query=query,
            routing_decision=routing_decision,
            responses=responses,
            evaluation_results=evaluation_results,
            winner=winner,
            promotions=promotions,
        )

        logger.info(f"Query processed successfully. Total queries: {self.query_count}")

        return final_response

    def _get_model_responses(self, query, routing_decision) -> List:
        """Get responses from appropriate models based on routing decision."""
        responses = []

        if routing_decision.routing_strategy == "parallel":
            # Get responses from all models
            for model in self.models.values():
                response = model.generate_response(query.query_text)
                responses.append(response)

        elif routing_decision.routing_strategy == "targeted":
            # Get response from recommended model(s)
            for model_id in routing_decision.recommended_models:
                model = self.models.get(model_id)
                if model:
                    response = model.generate_response(query.query_text)
                    responses.append(response)

            # Also get supervisor opinion
            if self.supervisor:
                response = self.supervisor.generate_response(query.query_text)
                responses.append(response)

        elif routing_decision.routing_strategy == "hybrid":
            # Get responses from recommended models + some students
            for model_id in routing_decision.recommended_models:
                model = self.models.get(model_id)
                if model:
                    response = model.generate_response(query.query_text)
                    responses.append(response)

            # Add some students for continued learning
            students = [m for m in self.models.values() if m.role == ModelRole.STUDENT]
            for student in students[:2]:  # First 2 students
                response = student.generate_response(query.query_text)
                responses.append(response)

            # Add supervisor
            if self.supervisor:
                response = self.supervisor.generate_response(query.query_text)
                responses.append(response)

        return responses

    def _prepare_final_response(
        self,
        query,
        routing_decision,
        responses,
        evaluation_results,
        winner,
        promotions,
    ) -> Dict[str, Any]:
        """Prepare final response package."""
        winning_response = None
        if winner and responses:
            winning_response = next(
                (r for r in responses if r.model_id == winner.model_id),
                responses[0]
            )

        return {
            "query_id": query.query_id,
            "query_text": query.query_text,
            "final_response": winning_response.response_text if winning_response else "No response available",
            "winner_model": winner.model_id if winner else None,
            "winner_score": winner.final_score if winner else 0.0,
            "routing_strategy": routing_decision.routing_strategy,
            "num_models_queried": len(responses),
            "promotions": promotions,
            "metadata": {
                "similarity_score": routing_decision.similarity_score,
                "is_novel": routing_decision.is_novel,
                "evaluation_results": [
                    {
                        "model_id": r.model_id,
                        "score": r.final_score,
                        "rank": r.rank,
                    }
                    for r in evaluation_results
                ],
            },
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics."""
        # Model statistics
        model_stats = {
            model_id: model.get_statistics()
            for model_id, model in self.models.items()
        }

        # Role distribution
        role_counts = {
            "supervisor": 1 if self.supervisor else 0,
            "teacher": len([m for m in self.models.values() if m.role == ModelRole.TEACHER]),
            "ta": len([m for m in self.models.values() if m.role == ModelRole.TA]),
            "student": len([m for m in self.models.values() if m.role == ModelRole.STUDENT]),
        }

        # System statistics
        system_stats = self.metrics_store.get_system_statistics()
        promotion_stats = self.promotion_system.get_promotion_statistics()

        return {
            "query_count": self.query_count,
            "model_stats": model_stats,
            "role_distribution": role_counts,
            "system_stats": system_stats,
            "promotion_stats": promotion_stats,
            "vector_db_stats": self.vector_store.get_statistics(),
        }

    def add_student_model(
        self,
        model_id: str,
        domain: str,
        teacher_id: Optional[str] = None,
    ) -> StudentModel:
        """
        Add a new student model to the system.

        Args:
            model_id: Unique model identifier
            domain: Domain specialization
            teacher_id: Optional teacher to assign

        Returns:
            Created student model
        """
        model_config = ModelConfig(
            model_id=model_id,
            model_type="mock",
            role=ModelRole.STUDENT,
            domain=domain,
            metadata={"teacher_id": teacher_id},
        )

        student = StudentModel(model_config)
        self.models[model_id] = student

        logger.info(f"Added new student: {model_id} (domain: {domain}, teacher: {teacher_id})")

        return student

    def export_metrics(self, output_path: str = "./data/export") -> Dict[str, str]:
        """Export metrics to files for analysis."""
        import os
        os.makedirs(output_path, exist_ok=True)

        # Export dataframes
        queries_df = self.metrics_store.export_to_dataframe("queries")
        promotions_df = self.metrics_store.export_to_dataframe("promotions")
        model_stats_df = self.metrics_store.export_to_dataframe("model_stats")

        # Save to CSV
        files = {}

        queries_path = os.path.join(output_path, "queries.csv")
        queries_df.to_csv(queries_path, index=False)
        files["queries"] = queries_path

        promotions_path = os.path.join(output_path, "promotions.csv")
        promotions_df.to_csv(promotions_path, index=False)
        files["promotions"] = promotions_path

        model_stats_path = os.path.join(output_path, "model_stats.csv")
        model_stats_df.to_csv(model_stats_path, index=False)
        files["model_stats"] = model_stats_path

        logger.info(f"Exported metrics to {output_path}")

        return files
