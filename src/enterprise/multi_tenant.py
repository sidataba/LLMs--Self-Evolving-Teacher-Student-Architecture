"""Multi-tenant support for enterprise deployments."""

import uuid
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


@dataclass
class Tenant:
    """Tenant information and configuration."""
    tenant_id: str
    name: str
    tier: str  # free, pro, enterprise
    created_at: str
    settings: Dict[str, Any] = field(default_factory=dict)
    quota: Dict[str, int] = field(default_factory=dict)
    usage: Dict[str, int] = field(default_factory=dict)
    models: List[str] = field(default_factory=list)
    api_keys: Set[str] = field(default_factory=set)
    status: str = "active"  # active, suspended, cancelled


@dataclass
class TenantQuota:
    """Resource quotas for a tenant."""
    max_queries_per_day: int
    max_models: int
    max_concurrent_requests: int
    max_storage_mb: int
    features: List[str]


class MultiTenantManager:
    """
    Multi-tenant management system.

    Features:
    - Tenant isolation and resource management
    - Per-tenant quotas and billing
    - Custom model configurations per tenant
    - Usage tracking and analytics
    - Tenant-specific data isolation
    """

    def __init__(self):
        """Initialize multi-tenant manager."""
        self.tenants: Dict[str, Tenant] = {}
        self.api_key_mapping: Dict[str, str] = {}  # api_key -> tenant_id

        # Default quota templates
        self.quota_templates = {
            "free": TenantQuota(
                max_queries_per_day=1000,
                max_models=3,
                max_concurrent_requests=5,
                max_storage_mb=100,
                features=["basic_models"],
            ),
            "pro": TenantQuota(
                max_queries_per_day=50000,
                max_models=20,
                max_concurrent_requests=50,
                max_storage_mb=10000,
                features=["basic_models", "advanced_models", "custom_training"],
            ),
            "enterprise": TenantQuota(
                max_queries_per_day=1000000,
                max_models=100,
                max_concurrent_requests=1000,
                max_storage_mb=1000000,
                features=["all_features", "dedicated_resources", "sla_guarantee"],
            ),
        }

        logger.info("MultiTenantManager initialized")

    def create_tenant(
        self,
        name: str,
        tier: str = "free",
        settings: Optional[Dict[str, Any]] = None,
    ) -> Tenant:
        """
        Create a new tenant.

        Args:
            name: Tenant name
            tier: Subscription tier (free, pro, enterprise)
            settings: Optional custom settings

        Returns:
            Created tenant
        """
        tenant_id = str(uuid.uuid4())

        # Get quota template
        quota_template = self.quota_templates.get(tier, self.quota_templates["free"])

        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            created_at=datetime.now().isoformat(),
            settings=settings or {},
            quota={
                "max_queries_per_day": quota_template.max_queries_per_day,
                "max_models": quota_template.max_models,
                "max_concurrent_requests": quota_template.max_concurrent_requests,
            },
            usage={
                "queries_today": 0,
                "total_queries": 0,
                "storage_used_mb": 0,
            },
        )

        # Generate API key
        api_key = self._generate_api_key()
        tenant.api_keys.add(api_key)
        self.api_key_mapping[api_key] = tenant_id

        self.tenants[tenant_id] = tenant

        logger.info(f"Created tenant: {name} ({tenant_id}) - Tier: {tier}")

        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self.tenants.get(tenant_id)

    def get_tenant_by_api_key(self, api_key: str) -> Optional[Tenant]:
        """Get tenant by API key."""
        tenant_id = self.api_key_mapping.get(api_key)
        if tenant_id:
            return self.tenants.get(tenant_id)
        return None

    def check_quota(
        self,
        tenant_id: str,
        resource: str,
        amount: int = 1,
    ) -> bool:
        """
        Check if tenant has quota available for resource.

        Args:
            tenant_id: Tenant ID
            resource: Resource type (queries, models, etc.)
            amount: Amount to check

        Returns:
            True if quota available
        """
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False

        if tenant.status != "active":
            return False

        # Check specific quotas
        if resource == "queries":
            current_usage = tenant.usage.get("queries_today", 0)
            max_quota = tenant.quota.get("max_queries_per_day", 0)
            return current_usage + amount <= max_quota

        elif resource == "models":
            current_models = len(tenant.models)
            max_models = tenant.quota.get("max_models", 0)
            return current_models + amount <= max_models

        return True

    def consume_quota(
        self,
        tenant_id: str,
        resource: str,
        amount: int = 1,
    ) -> bool:
        """
        Consume tenant quota.

        Args:
            tenant_id: Tenant ID
            resource: Resource type
            amount: Amount to consume

        Returns:
            True if consumed successfully
        """
        if not self.check_quota(tenant_id, resource, amount):
            logger.warning(f"Quota exceeded for tenant {tenant_id}: {resource}")
            return False

        tenant = self.tenants[tenant_id]

        # Update usage
        if resource == "queries":
            tenant.usage["queries_today"] = tenant.usage.get("queries_today", 0) + amount
            tenant.usage["total_queries"] = tenant.usage.get("total_queries", 0) + amount

        return True

    def upgrade_tenant(self, tenant_id: str, new_tier: str) -> bool:
        """
        Upgrade tenant tier.

        Args:
            tenant_id: Tenant ID
            new_tier: New subscription tier

        Returns:
            True if upgraded successfully
        """
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False

        if new_tier not in self.quota_templates:
            logger.error(f"Invalid tier: {new_tier}")
            return False

        old_tier = tenant.tier
        tenant.tier = new_tier

        # Update quotas
        new_quota = self.quota_templates[new_tier]
        tenant.quota = {
            "max_queries_per_day": new_quota.max_queries_per_day,
            "max_models": new_quota.max_models,
            "max_concurrent_requests": new_quota.max_concurrent_requests,
        }

        logger.info(f"Upgraded tenant {tenant_id}: {old_tier} -> {new_tier}")

        return True

    def _generate_api_key(self) -> str:
        """Generate unique API key."""
        return f"sk-{uuid.uuid4().hex}"

    def get_tenant_stats(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get tenant usage statistics."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return None

        return {
            "tenant_id": tenant_id,
            "name": tenant.name,
            "tier": tenant.tier,
            "status": tenant.status,
            "quota": tenant.quota,
            "usage": tenant.usage,
            "utilization": {
                "queries": tenant.usage.get("queries_today", 0) /
                           tenant.quota.get("max_queries_per_day", 1),
                "models": len(tenant.models) / tenant.quota.get("max_models", 1),
            },
        }

    def get_all_tenants_stats(self) -> Dict[str, Any]:
        """Get statistics for all tenants."""
        return {
            "total_tenants": len(self.tenants),
            "by_tier": {
                tier: sum(1 for t in self.tenants.values() if t.tier == tier)
                for tier in ["free", "pro", "enterprise"]
            },
            "total_queries": sum(
                t.usage.get("total_queries", 0)
                for t in self.tenants.values()
            ),
            "active_tenants": sum(
                1 for t in self.tenants.values()
                if t.status == "active"
            ),
        }

    def reset_daily_quotas(self) -> None:
        """Reset daily quotas for all tenants."""
        for tenant in self.tenants.values():
            tenant.usage["queries_today"] = 0

        logger.info("Reset daily quotas for all tenants")
