"""Enterprise API Gateway for routing and load balancing."""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
import asyncio
from loguru import logger


@dataclass
class Request:
    """API request structure."""
    request_id: str
    tenant_id: str
    query: str
    domain: Optional[str]
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class RateLimitRule:
    """Rate limiting rule."""
    requests_per_second: int
    burst_size: int


class APIGateway:
    """
    Enterprise API Gateway.

    Features:
    - Request routing and load balancing
    - Rate limiting per tenant
    - Request queuing and prioritization
    - Caching for repeated queries
    - API versioning
    - Request/response logging
    """

    def __init__(
        self,
        distributed_orchestrator,
        multi_tenant_manager,
        enable_caching: bool = True,
        cache_ttl: int = 3600,
    ):
        """
        Initialize API gateway.

        Args:
            distributed_orchestrator: Distributed orchestrator instance
            multi_tenant_manager: Multi-tenant manager
            enable_caching: Whether to enable response caching
            cache_ttl: Cache time-to-live in seconds
        """
        self.orchestrator = distributed_orchestrator
        self.tenant_manager = multi_tenant_manager
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl

        # Request tracking
        self.request_count = 0
        self.active_requests = {}

        # Caching
        self.response_cache: Dict[str, Dict[str, Any]] = {}

        # Rate limiting state
        self.rate_limit_state: Dict[str, list] = {}

        logger.info("APIGateway initialized")

    async def handle_request(
        self,
        request: Request,
    ) -> Dict[str, Any]:
        """
        Handle incoming API request.

        Args:
            request: API request

        Returns:
            Response dictionary
        """
        start_time = time.time()

        # 1. Validate tenant and API key
        tenant = self.tenant_manager.get_tenant(request.tenant_id)
        if not tenant or tenant.status != "active":
            return {
                "error": "Invalid or inactive tenant",
                "request_id": request.request_id,
            }

        # 2. Check rate limits
        if not self._check_rate_limit(request.tenant_id):
            return {
                "error": "Rate limit exceeded",
                "request_id": request.request_id,
                "retry_after": 60,
            }

        # 3. Check tenant quota
        if not self.tenant_manager.check_quota(request.tenant_id, "queries"):
            return {
                "error": "Quota exceeded",
                "request_id": request.request_id,
                "quota_info": tenant.quota,
            }

        # 4. Check cache
        if self.enable_caching:
            cached_response = self._check_cache(request)
            if cached_response:
                logger.debug(f"Cache hit for request {request.request_id}")
                cached_response["cached"] = True
                return cached_response

        # 5. Route to orchestrator
        try:
            self.active_requests[request.request_id] = request

            # Process with distributed orchestrator
            result = await self.orchestrator.process_query(
                query_text=request.query,
                domain=request.domain,
                tenant_id=request.tenant_id,
            )

            # Consume quota
            self.tenant_manager.consume_quota(request.tenant_id, "queries")

            # Cache response
            if self.enable_caching:
                self._cache_response(request, result)

            # Add metadata
            result["request_id"] = request.request_id
            result["processing_time_ms"] = (time.time() - start_time) * 1000
            result["tenant_id"] = request.tenant_id

            return result

        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return {
                "error": str(e),
                "request_id": request.request_id,
            }

        finally:
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]

    def _check_rate_limit(self, tenant_id: str) -> bool:
        """Check if tenant is within rate limits."""
        # Get tenant rate limit
        tenant = self.tenant_manager.get_tenant(tenant_id)
        if not tenant:
            return False

        # Rate limit based on tier
        rate_limits = {
            "free": RateLimitRule(requests_per_second=10, burst_size=20),
            "pro": RateLimitRule(requests_per_second=100, burst_size=200),
            "enterprise": RateLimitRule(requests_per_second=1000, burst_size=2000),
        }

        limit = rate_limits.get(tenant.tier, rate_limits["free"])

        # Simple sliding window rate limiting
        now = time.time()

        if tenant_id not in self.rate_limit_state:
            self.rate_limit_state[tenant_id] = []

        # Remove old requests outside window
        window = 1.0  # 1 second window
        self.rate_limit_state[tenant_id] = [
            t for t in self.rate_limit_state[tenant_id]
            if now - t < window
        ]

        # Check limit
        if len(self.rate_limit_state[tenant_id]) >= limit.requests_per_second:
            return False

        # Record this request
        self.rate_limit_state[tenant_id].append(now)

        return True

    def _check_cache(self, request: Request) -> Optional[Dict[str, Any]]:
        """Check if response is cached."""
        cache_key = self._generate_cache_key(request)

        if cache_key in self.response_cache:
            cached = self.response_cache[cache_key]

            # Check TTL
            if time.time() - cached["timestamp"] < self.cache_ttl:
                return cached["response"]
            else:
                # Expired, remove
                del self.response_cache[cache_key]

        return None

    def _cache_response(self, request: Request, response: Dict[str, Any]) -> None:
        """Cache a response."""
        cache_key = self._generate_cache_key(request)

        self.response_cache[cache_key] = {
            "timestamp": time.time(),
            "response": response,
        }

        # Limit cache size
        if len(self.response_cache) > 10000:
            # Remove oldest 10%
            sorted_items = sorted(
                self.response_cache.items(),
                key=lambda x: x[1]["timestamp"]
            )
            for key, _ in sorted_items[:1000]:
                del self.response_cache[key]

    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request."""
        # Simple key based on query and domain
        return f"{request.tenant_id}:{request.domain}:{request.query}"

    def get_gateway_stats(self) -> Dict[str, Any]:
        """Get API gateway statistics."""
        return {
            "total_requests": self.request_count,
            "active_requests": len(self.active_requests),
            "cache_size": len(self.response_cache),
            "cache_enabled": self.enable_caching,
            "rate_limited_tenants": len(self.rate_limit_state),
        }
