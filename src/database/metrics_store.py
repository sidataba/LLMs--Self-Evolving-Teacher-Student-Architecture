"""Metrics storage for tracking model performance and system statistics."""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict
import pandas as pd
from loguru import logger


class MetricsStore:
    """
    Storage and retrieval system for model performance metrics.
    Tracks query responses, confidence scores, promotions, and system statistics.
    """

    def __init__(self, db_path: str = "./data/metrics"):
        """
        Initialize the metrics store.

        Args:
            db_path: Directory to store metrics files
        """
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)

        # File paths
        self.query_log_path = os.path.join(db_path, "query_log.jsonl")
        self.model_stats_path = os.path.join(db_path, "model_stats.json")
        self.promotion_log_path = os.path.join(db_path, "promotions.jsonl")

        # In-memory caches
        self.model_stats = self._load_model_stats()

        logger.info(f"MetricsStore initialized at {db_path}")

    def _load_model_stats(self) -> Dict[str, Any]:
        """Load model statistics from disk."""
        if os.path.exists(self.model_stats_path):
            with open(self.model_stats_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_model_stats(self) -> None:
        """Save model statistics to disk."""
        with open(self.model_stats_path, 'w') as f:
            json.dump(self.model_stats, f, indent=2)

    def log_query_response(
        self,
        query_id: str,
        query_text: str,
        model_id: str,
        response: str,
        confidence: float,
        metrics: Dict[str, float],
        is_winner: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a model's response to a query.

        Args:
            query_id: Unique query identifier
            query_text: The query text
            model_id: Model that generated the response
            response: The model's response
            confidence: Confidence score (0-1)
            metrics: Dictionary of evaluation metrics
            is_winner: Whether this was the selected response
            metadata: Optional additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query_id": query_id,
            "query_text": query_text,
            "model_id": model_id,
            "response": response,
            "confidence": confidence,
            "metrics": metrics,
            "is_winner": is_winner,
            "metadata": metadata or {},
        }

        # Append to query log
        with open(self.query_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Update model statistics
        self._update_model_stats(model_id, confidence, metrics, is_winner)

        logger.debug(f"Logged response from {model_id} for query {query_id}")

    def _update_model_stats(
        self,
        model_id: str,
        confidence: float,
        metrics: Dict[str, float],
        is_winner: bool,
    ) -> None:
        """Update running statistics for a model."""
        if model_id not in self.model_stats:
            self.model_stats[model_id] = {
                "total_queries": 0,
                "total_wins": 0,
                "total_confidence": 0.0,
                "metrics_sum": defaultdict(float),
                "last_updated": None,
            }

        stats = self.model_stats[model_id]
        stats["total_queries"] += 1
        stats["total_confidence"] += confidence
        stats["last_updated"] = datetime.now().isoformat()

        if is_winner:
            stats["total_wins"] += 1

        for metric_name, value in metrics.items():
            stats["metrics_sum"][metric_name] = stats["metrics_sum"].get(metric_name, 0.0) + value

        # Periodically save to disk
        if stats["total_queries"] % 10 == 0:
            self._save_model_stats()

    def get_model_statistics(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get performance statistics for a model.

        Args:
            model_id: The model identifier

        Returns:
            Dictionary with statistics or None if model not found
        """
        if model_id not in self.model_stats:
            return None

        stats = self.model_stats[model_id]
        total_queries = stats["total_queries"]

        if total_queries == 0:
            return {
                "model_id": model_id,
                "total_queries": 0,
                "avg_confidence": 0.0,
                "win_rate": 0.0,
                "avg_metrics": {},
            }

        avg_metrics = {}
        for metric_name, total in stats["metrics_sum"].items():
            avg_metrics[metric_name] = total / total_queries

        return {
            "model_id": model_id,
            "total_queries": total_queries,
            "total_wins": stats["total_wins"],
            "avg_confidence": stats["total_confidence"] / total_queries,
            "win_rate": stats["total_wins"] / total_queries,
            "avg_metrics": avg_metrics,
            "last_updated": stats["last_updated"],
        }

    def get_all_model_statistics(self) -> List[Dict[str, Any]]:
        """Get statistics for all models."""
        all_stats = []
        for model_id in self.model_stats.keys():
            stats = self.get_model_statistics(model_id)
            if stats:
                all_stats.append(stats)
        return all_stats

    def log_promotion(
        self,
        model_id: str,
        from_role: str,
        to_role: str,
        reason: str,
        stats: Dict[str, Any],
    ) -> None:
        """
        Log a model promotion event.

        Args:
            model_id: Model being promoted
            from_role: Previous role (student, ta, teacher)
            to_role: New role
            reason: Reason for promotion
            stats: Performance statistics at time of promotion
        """
        promotion_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id,
            "from_role": from_role,
            "to_role": to_role,
            "reason": reason,
            "stats": stats,
        }

        with open(self.promotion_log_path, 'a') as f:
            f.write(json.dumps(promotion_entry) + '\n')

        logger.info(f"Promotion logged: {model_id} from {from_role} to {to_role}")

    def get_promotion_history(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get promotion history.

        Args:
            model_id: Optional model ID to filter by

        Returns:
            List of promotion events
        """
        if not os.path.exists(self.promotion_log_path):
            return []

        promotions = []
        with open(self.promotion_log_path, 'r') as f:
            for line in f:
                promotion = json.loads(line)
                if model_id is None or promotion["model_id"] == model_id:
                    promotions.append(promotion)

        return promotions

    def get_query_history(
        self,
        query_id: Optional[str] = None,
        model_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get query response history.

        Args:
            query_id: Optional query ID to filter by
            model_id: Optional model ID to filter by
            limit: Maximum number of entries to return

        Returns:
            List of query response entries
        """
        if not os.path.exists(self.query_log_path):
            return []

        queries = []
        with open(self.query_log_path, 'r') as f:
            for line in f:
                entry = json.loads(line)

                # Apply filters
                if query_id and entry["query_id"] != query_id:
                    continue
                if model_id and entry["model_id"] != model_id:
                    continue

                queries.append(entry)

                if len(queries) >= limit:
                    break

        return queries

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        total_queries = 0
        total_models = len(self.model_stats)

        for stats in self.model_stats.values():
            total_queries += stats["total_queries"]

        # Count promotions
        total_promotions = len(self.get_promotion_history())

        return {
            "total_queries": total_queries,
            "total_models": total_models,
            "total_promotions": total_promotions,
            "models": list(self.model_stats.keys()),
        }

    def export_to_dataframe(self, data_type: str = "queries") -> pd.DataFrame:
        """
        Export data to pandas DataFrame for analysis.

        Args:
            data_type: Type of data to export ("queries", "promotions", "model_stats")

        Returns:
            DataFrame with requested data
        """
        if data_type == "queries":
            if not os.path.exists(self.query_log_path):
                return pd.DataFrame()

            queries = []
            with open(self.query_log_path, 'r') as f:
                for line in f:
                    queries.append(json.loads(line))

            return pd.DataFrame(queries)

        elif data_type == "promotions":
            return pd.DataFrame(self.get_promotion_history())

        elif data_type == "model_stats":
            return pd.DataFrame(self.get_all_model_statistics())

        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    def clear(self) -> None:
        """Clear all metrics data."""
        for path in [self.query_log_path, self.model_stats_path, self.promotion_log_path]:
            if os.path.exists(path):
                os.remove(path)

        self.model_stats = {}
        logger.info("Cleared all metrics data")
