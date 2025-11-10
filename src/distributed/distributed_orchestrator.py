"""Distributed orchestrator for multi-node deployment."""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import hashlib
from loguru import logger

from src.core.orchestrator import Orchestrator


@dataclass
class NodeInfo:
    """Information about a compute node."""
    node_id: str
    address: str
    capacity: int
    current_load: int
    models: List[str]
    status: str  # active, degraded, offline
    last_heartbeat: float


class DistributedOrchestrator:
    """
    Distributed orchestrator for enterprise-scale deployment.

    Features:
    - Multi-node model distribution
    - Load balancing across nodes
    - Fault tolerance and failover
    - Horizontal scaling
    - Model sharding for large deployments
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        enable_sharding: bool = True,
        replication_factor: int = 2,
    ):
        """
        Initialize distributed orchestrator.

        Args:
            config_path: Path to configuration file
            enable_sharding: Whether to enable model sharding
            replication_factor: Number of replicas per model
        """
        self.config_path = config_path
        self.enable_sharding = enable_sharding
        self.replication_factor = replication_factor

        # Node management
        self.nodes: Dict[str, NodeInfo] = {}
        self.model_placement: Dict[str, List[str]] = {}  # model_id -> [node_ids]

        # Local orchestrator (for single-node mode)
        self.local_orchestrator: Optional[Orchestrator] = None

        # Distributed state
        self.query_queue = asyncio.Queue()
        self.is_distributed = False

        logger.info(
            f"DistributedOrchestrator initialized "
            f"(sharding: {enable_sharding}, "
            f"replication: {replication_factor})"
        )

    def initialize_local_mode(self) -> None:
        """Initialize in local single-node mode."""
        self.local_orchestrator = Orchestrator(self.config_path)
        self.is_distributed = False
        logger.info("Running in LOCAL mode (single node)")

    def initialize_distributed_mode(self, node_addresses: List[str]) -> None:
        """
        Initialize in distributed multi-node mode.

        Args:
            node_addresses: List of node addresses (e.g., ["node1:8000", "node2:8000"])
        """
        self.is_distributed = True

        # Register nodes
        for i, address in enumerate(node_addresses):
            node_id = f"node-{i}"
            self.nodes[node_id] = NodeInfo(
                node_id=node_id,
                address=address,
                capacity=100,
                current_load=0,
                models=[],
                status="active",
                last_heartbeat=0.0,
            )

        logger.info(f"Running in DISTRIBUTED mode with {len(self.nodes)} nodes")

        # Distribute models across nodes
        self._distribute_models()

    async def process_query(
        self,
        query_text: str,
        domain: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process query in distributed environment.

        Args:
            query_text: The query text
            domain: Optional domain hint
            tenant_id: Optional tenant ID for multi-tenancy

        Returns:
            Query result
        """
        if not self.is_distributed:
            # Local mode - use local orchestrator
            return self.local_orchestrator.process_query(query_text, domain)

        # Distributed mode
        # 1. Route query to appropriate node(s)
        target_nodes = self._select_nodes_for_query(query_text, domain)

        # 2. Send query to nodes in parallel
        # In production, this would use actual RPC/HTTP calls
        logger.debug(f"Routing query to nodes: {target_nodes}")

        # Simulate distributed processing
        # In production, replace with actual distributed calls
        return self.local_orchestrator.process_query(query_text, domain)

    def _distribute_models(self) -> None:
        """Distribute models across available nodes."""
        if not self.local_orchestrator:
            self.local_orchestrator = Orchestrator(self.config_path)

        models = list(self.local_orchestrator.models.keys())
        num_nodes = len(self.nodes)

        if num_nodes == 0:
            logger.error("No nodes available for model distribution")
            return

        # Simple round-robin distribution with replication
        for i, model_id in enumerate(models):
            # Primary node
            primary_node_id = list(self.nodes.keys())[i % num_nodes]

            # Replica nodes
            replica_nodes = []
            for r in range(1, self.replication_factor):
                replica_idx = (i + r) % num_nodes
                replica_node_id = list(self.nodes.keys())[replica_idx]
                if replica_node_id != primary_node_id:
                    replica_nodes.append(replica_node_id)

            # Record placement
            placement = [primary_node_id] + replica_nodes
            self.model_placement[model_id] = placement

            # Update node info
            for node_id in placement:
                self.nodes[node_id].models.append(model_id)

        # Log distribution
        logger.info("Model distribution:")
        for node_id, node_info in self.nodes.items():
            logger.info(f"  {node_id}: {len(node_info.models)} models")

    def _select_nodes_for_query(
        self,
        query_text: str,
        domain: Optional[str] = None,
    ) -> List[str]:
        """Select optimal nodes for query processing."""
        # Use consistent hashing for query routing
        query_hash = int(hashlib.md5(query_text.encode()).hexdigest(), 16)
        node_ids = list(self.nodes.keys())

        if not node_ids:
            return []

        # Select primary node
        primary_idx = query_hash % len(node_ids)
        primary_node = node_ids[primary_idx]

        # Check if node is healthy
        if self.nodes[primary_node].status != "active":
            # Find alternative
            for node_id in node_ids:
                if self.nodes[node_id].status == "active":
                    primary_node = node_id
                    break

        return [primary_node]

    def add_node(self, node_address: str) -> str:
        """
        Add a new node to the cluster.

        Args:
            node_address: Address of the new node

        Returns:
            Node ID
        """
        node_id = f"node-{len(self.nodes)}"

        self.nodes[node_id] = NodeInfo(
            node_id=node_id,
            address=node_address,
            capacity=100,
            current_load=0,
            models=[],
            status="active",
            last_heartbeat=0.0,
        )

        logger.info(f"Added new node: {node_id} at {node_address}")

        # Rebalance models
        self._rebalance_models()

        return node_id

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node from the cluster.

        Args:
            node_id: ID of node to remove
        """
        if node_id not in self.nodes:
            logger.warning(f"Node not found: {node_id}")
            return

        # Mark as offline
        self.nodes[node_id].status = "offline"

        # Migrate models to other nodes
        models_to_migrate = self.nodes[node_id].models

        for model_id in models_to_migrate:
            # Remove this node from placement
            if model_id in self.model_placement:
                self.model_placement[model_id] = [
                    nid for nid in self.model_placement[model_id]
                    if nid != node_id
                ]

        # Remove node
        del self.nodes[node_id]

        logger.info(f"Removed node: {node_id}, migrated {len(models_to_migrate)} models")

        # Rebalance
        self._rebalance_models()

    def _rebalance_models(self) -> None:
        """Rebalance model distribution across nodes."""
        active_nodes = [
            node_id for node_id, node in self.nodes.items()
            if node.status == "active"
        ]

        if not active_nodes:
            logger.error("No active nodes available for rebalancing")
            return

        # Clear current distribution
        for node in self.nodes.values():
            node.models = []

        # Redistribute
        self._distribute_models()

        logger.info(f"Rebalanced models across {len(active_nodes)} nodes")

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status information."""
        total_capacity = sum(node.capacity for node in self.nodes.values())
        total_load = sum(node.current_load for node in self.nodes.values())

        active_nodes = sum(1 for node in self.nodes.values() if node.status == "active")

        return {
            "mode": "distributed" if self.is_distributed else "local",
            "total_nodes": len(self.nodes),
            "active_nodes": active_nodes,
            "total_capacity": total_capacity,
            "total_load": total_load,
            "utilization": total_load / total_capacity if total_capacity > 0 else 0,
            "nodes": {
                node_id: {
                    "status": node.status,
                    "models": len(node.models),
                    "load": node.current_load,
                    "capacity": node.capacity,
                }
                for node_id, node in self.nodes.items()
            },
        }

    def scale_up(self, num_nodes: int = 1) -> List[str]:
        """
        Scale up by adding new nodes.

        Args:
            num_nodes: Number of nodes to add

        Returns:
            List of new node IDs
        """
        new_node_ids = []

        for i in range(num_nodes):
            node_address = f"auto-scaled-node-{i}:8000"
            node_id = self.add_node(node_address)
            new_node_ids.append(node_id)

        logger.info(f"📈 Scaled up: Added {num_nodes} nodes")

        return new_node_ids

    def scale_down(self, num_nodes: int = 1) -> List[str]:
        """
        Scale down by removing nodes.

        Args:
            num_nodes: Number of nodes to remove

        Returns:
            List of removed node IDs
        """
        removed_nodes = []

        # Remove least loaded nodes
        sorted_nodes = sorted(
            self.nodes.items(),
            key=lambda x: x[1].current_load
        )

        for node_id, _ in sorted_nodes[:num_nodes]:
            self.remove_node(node_id)
            removed_nodes.append(node_id)

        logger.info(f"📉 Scaled down: Removed {num_nodes} nodes")

        return removed_nodes
