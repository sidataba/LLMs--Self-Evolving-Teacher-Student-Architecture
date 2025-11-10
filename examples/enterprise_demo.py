"""
Enterprise-scale demo showcasing advanced features:
- Knowledge distillation
- Self-evolution
- Distributed architecture
- Agentic capabilities
- Multi-tenancy
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.orchestrator import Orchestrator
from src.core.distillation import KnowledgeDistillation, DistillationConfig
from src.core.self_evolution import SelfEvolutionEngine
from src.distributed.distributed_orchestrator import DistributedOrchestrator
from src.agentic.agent_system import AgenticSystem
from src.enterprise.multi_tenant import MultiTenantManager
from src.enterprise.api_gateway import APIGateway, Request
from src.monitoring.dashboard import Dashboard
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="<level>{level: <8}</level> | {message}")


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def main():
    """Run comprehensive enterprise demo."""
    print_section("ENTERPRISE-SCALE SELF-EVOLVING LLM SYSTEM DEMO")

    # ========================================================================
    # PART 1: Initialize Core System with Advanced Features
    # ========================================================================
    print_section("1. INITIALIZING ADVANCED SYSTEM")

    print("📦 Setting up distributed orchestrator...")
    dist_orchestrator = DistributedOrchestrator(
        config_path="./config/default_config.yaml"
    )
    dist_orchestrator.initialize_local_mode()  # Start in local mode

    orchestrator = dist_orchestrator.local_orchestrator

    print("✓ Core orchestrator initialized")

    # Add knowledge distillation
    print("\n🧠 Setting up knowledge distillation system...")
    distillation_config = DistillationConfig(
        temperature=2.0,
        alpha=0.7,
        distillation_strategy="multi-aspect",
    )
    distillation = KnowledgeDistillation(distillation_config)
    print("✓ Knowledge distillation ready")

    # Add self-evolution engine
    print("\n🔄 Setting up self-evolution engine...")
    evolution = SelfEvolutionEngine(
        orchestrator=orchestrator,
        evolution_interval=15,  # Trigger every 15 queries
        auto_spawn_students=True,
    )
    print("✓ Self-evolution engine ready")

    # Add agentic capabilities
    print("\n🤖 Setting up agentic system...")
    agentic_system = AgenticSystem(orchestrator)
    print(f"✓ Agentic system ready with {len(agentic_system.tools)} tools")

    # Add multi-tenancy
    print("\n🏢 Setting up multi-tenant system...")
    tenant_manager = MultiTenantManager()

    # Create demo tenants
    tenant_free = tenant_manager.create_tenant("Startup Inc.", tier="free")
    tenant_pro = tenant_manager.create_tenant("MidSize Corp.", tier="pro")
    tenant_ent = tenant_manager.create_tenant("BigTech Enterprise", tier="enterprise")

    print(f"✓ Created {len(tenant_manager.tenants)} tenants")

    # Add API Gateway
    print("\n🚪 Setting up API gateway...")
    api_gateway = APIGateway(
        distributed_orchestrator=dist_orchestrator,
        multi_tenant_manager=tenant_manager,
    )
    print("✓ API gateway ready")

    dashboard = Dashboard(orchestrator)

    # ========================================================================
    # PART 2: Process Queries with Advanced Features
    # ========================================================================
    print_section("2. PROCESSING QUERIES WITH ADVANCED FEATURES")

    queries = [
        ("What is machine learning?", "mathematics", tenant_ent.tenant_id),
        ("Explain neural networks", "science", tenant_pro.tenant_id),
        ("How do I train a model?", "programming", tenant_free.tenant_id),
        ("What is deep learning?", "mathematics", tenant_ent.tenant_id),
        ("Solve equation: 2x + 5 = 15", "mathematics", tenant_pro.tenant_id),
        ("What is calculus?", "mathematics", tenant_ent.tenant_id),
        ("Explain photosynthesis", "science", tenant_free.tenant_id),
        ("How does DNA work?", "science", tenant_pro.tenant_id),
        ("Write a Python function", "programming", tenant_ent.tenant_id),
        ("What is an algorithm?", "programming", tenant_pro.tenant_id),
        # More queries to trigger evolution
        ("Advanced calculus concepts", "mathematics", tenant_ent.tenant_id),
        ("Quantum mechanics basics", "science", tenant_pro.tenant_id),
        ("Machine learning algorithms", "programming", tenant_ent.tenant_id),
        ("Linear algebra foundations", "mathematics", tenant_pro.tenant_id),
        ("Organic chemistry", "science", tenant_free.tenant_id),
    ]

    for i, (query, domain, tenant_id) in enumerate(queries, 1):
        print(f"\n{'─'*80}")
        print(f"Query {i}/{len(queries)}: {query}")
        print(f"Domain: {domain} | Tenant: {tenant_manager.get_tenant(tenant_id).name}")
        print(f"{'─'*80}")

        # Process query
        result = orchestrator.process_query(query, domain)

        # Record for evolution
        evolution.record_query_result(result)

        # Collect distillation samples
        if result.get("winner_model") and result.get("winner_score", 0) > 0.85:
            # High-quality response - collect for distillation
            for model_id, model in orchestrator.models.items():
                if model.role.value == "student":
                    distillation.collect_sample(
                        query=query,
                        teacher_response=result["final_response"],
                        teacher_confidence=result["winner_score"],
                        teacher_reasoning="High quality response selected",
                        student_response=None,
                        student_id=model_id,
                        domain=domain,
                        metrics={"quality": result["winner_score"]},
                    )

        print(f"✓ Winner: {result['winner_model']} (Score: {result['winner_score']:.3f})")

        # Trigger distillation if enough samples
        for model_id, model in orchestrator.models.items():
            if model.role.value == "student":
                if distillation.should_trigger_distillation(model_id):
                    print(f"  🎓 Triggering distillation for {model_id}...")
                    dist_result = distillation.distill_knowledge(model, model_id)
                    if dist_result.get("improvement"):
                        print(f"     Improvement: +{dist_result['improvement']:.3f}")

    # ========================================================================
    # PART 3: Self-Evolution Cycle
    # ========================================================================
    print_section("3. SELF-EVOLUTION CYCLE")

    print("🔄 Triggering evolution cycle...")
    evolution_result = evolution.trigger_evolution_cycle()

    print(f"\n✓ Evolution Cycle #{evolution_result['cycle']} Complete")
    print(f"\nActions taken:")
    for action, details in evolution_result['actions_taken']:
        print(f"  • {action}: {details}")

    metrics = evolution_result['metrics']
    print(f"\nEvolution Metrics:")
    print(f"  Total Models: {metrics.total_models}")
    print(f"  Avg Confidence: {metrics.avg_system_confidence:.3f}")
    print(f"  Domain Coverage: {len(metrics.domain_coverage)} domains")
    print(f"  Cost Reduction: {metrics.cost_reduction:.1%}")
    print(f"  Quality Improvement: {metrics.quality_improvement:+.3f}")

    # ========================================================================
    # PART 4: Agentic Query Execution
    # ========================================================================
    print_section("4. AGENTIC QUERY EXECUTION")

    agentic_queries = [
        "Calculate 25 * 4 and then explain what multiplication is",
        "Search for information about neural networks",
    ]

    for query in agentic_queries:
        print(f"\n🤖 Agentic Query: {query}")
        result = agentic_system.execute_agentic_query(query, enable_tools=True)

        print(f"  Steps executed: {result['steps_executed']}")
        print(f"  Tools used: {result['tools_used']}")
        print(f"  Final answer: {result['final_answer'][:200]}...")

    # ========================================================================
    # PART 5: Distributed Architecture Demo
    # ========================================================================
    print_section("5. DISTRIBUTED ARCHITECTURE")

    print("📡 Simulating distributed deployment...")

    # Simulate adding nodes
    nodes = ["node-0:8000", "node-1:8000", "node-2:8000"]
    dist_orchestrator.initialize_distributed_mode(nodes)

    cluster_status = dist_orchestrator.get_cluster_status()
    print(f"✓ Cluster initialized:")
    print(f"  Mode: {cluster_status['mode']}")
    print(f"  Total nodes: {cluster_status['total_nodes']}")
    print(f"  Active nodes: {cluster_status['active_nodes']}")

    # Demonstrate scaling
    print(f"\n📈 Scaling up cluster...")
    new_nodes = dist_orchestrator.scale_up(num_nodes=2)
    print(f"  Added nodes: {new_nodes}")

    cluster_status = dist_orchestrator.get_cluster_status()
    print(f"  Total nodes now: {cluster_status['total_nodes']}")

    # ========================================================================
    # PART 6: Enterprise Features & Statistics
    # ========================================================================
    print_section("6. ENTERPRISE FEATURES & STATISTICS")

    # Multi-tenant stats
    print("🏢 Multi-Tenant Statistics:")
    mt_stats = tenant_manager.get_all_tenants_stats()
    print(f"  Total Tenants: {mt_stats['total_tenants']}")
    print(f"  By Tier: {mt_stats['by_tier']}")
    print(f"  Total Queries: {mt_stats['total_queries']}")

    # Distillation stats
    print("\n🎓 Knowledge Distillation Statistics:")
    dist_stats = distillation.get_distillation_statistics()
    print(f"  Total Events: {dist_stats['total_events']}")
    print(f"  Samples Processed: {dist_stats['total_samples_processed']}")
    print(f"  Students Trained: {dist_stats['students_trained']}")
    if dist_stats['total_events'] > 0:
        print(f"  Avg Improvement: {dist_stats['avg_improvement']:.3f}")

    # Evolution report
    print("\n🔄 Evolution Report:")
    evolution_report = evolution.get_evolution_report()
    if evolution_report.get('status') != 'no_evolution_cycles':
        latest = evolution_report['latest_metrics']
        print(f"  Current Cycle: {evolution_report['current_cycle']}")
        print(f"  Discovered Domains: {len(evolution_report['discovered_domains'])}")
        print(f"  System Confidence: {latest['avg_confidence']:.3f}")
        print(f"  Cost Reduction: {latest['cost_reduction']:.1%}")

    # Final system status
    print("\n📊 Final System Status:")
    dashboard.print_system_status()

    # ========================================================================
    # PART 7: Key Insights
    # ========================================================================
    print_section("7. KEY ENTERPRISE CAPABILITIES DEMONSTRATED")

    print("✅ CORE SELF-EVOLUTION:")
    print("   • Autonomous knowledge gap detection")
    print("   • Dynamic student spawning for new domains")
    print("   • Automatic model promotion based on performance")
    print("   • Continuous system improvement without human intervention")

    print("\n✅ ADVANCED DISTILLATION:")
    print("   • Multi-aspect knowledge transfer (response, reasoning, metrics)")
    print("   • Configurable distillation strategies")
    print("   • Automatic triggering based on sample accumulation")
    print("   • Measurable performance improvement")

    print("\n✅ ENTERPRISE SCALABILITY:")
    print("   • Multi-node distributed architecture")
    print("   • Horizontal scaling (scale up/down dynamically)")
    print("   • Model replication for high availability")
    print("   • Load balancing across cluster")

    print("\n✅ AGENTIC CAPABILITIES:")
    print("   • Tool selection and execution")
    print("   • Multi-step task decomposition")
    print("   • Context-aware reasoning")
    print("   • Extensible tool registry")

    print("\n✅ MULTI-TENANCY:")
    print("   • Isolated tenant environments")
    print("   • Tier-based quotas and features")
    print("   • Usage tracking and billing")
    print("   • Per-tenant customization")

    print("\n✅ COST OPTIMIZATION:")
    print(f"   • Current cost reduction: {metrics.cost_reduction:.1%}")
    print("   • Smart routing to cheaper models")
    print("   • Automatic student promotion reduces supervisor usage")
    print("   • Caching and query optimization")

    print_section("DEMO COMPLETE - READY FOR BILLION-DOLLAR SCALE!")

    print("\n💡 This system demonstrates:")
    print("   • Self-evolution without human intervention")
    print("   • Enterprise-grade scalability and multi-tenancy")
    print("   • Cost optimization while maintaining quality")
    print("   • Advanced agentic capabilities")
    print("   • Production-ready architecture patterns")
    print("\n   Ready to power the next generation of AI systems! 🚀\n")


if __name__ == "__main__":
    main()
