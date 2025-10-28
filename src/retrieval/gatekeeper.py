"""Retrieval gatekeeper for shallow checks before expensive web fetches."""

from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import hashlib

if TYPE_CHECKING:
    from ..dashboard.content_api import ContentAPI

logger = logging.getLogger(__name__)


class GatekeeperStatus(Enum):
    """Gatekeeper decision status."""
    ALLOW = "allow"
    DENY = "deny"
    PENDING = "pending"


@dataclass
class GateToken:
    """Token for gated web content fetch."""
    token_id: str
    query_hash: str
    timestamp: float
    expires_at: float
    used: bool = False


@dataclass
class ShallowCheckResult:
    """Result of shallow checks."""
    can_proceed: bool
    confidence: float
    reasons: List[str]
    suggested_alternatives: List[str]
    timestamp: float = field(default_factory=time.time)


class RetrievalGatekeeper:
    """Gatekeeper that performs shallow checks before allowing expensive web fetches."""

    def __init__(
        self,
        max_tokens_per_hour: int = 1000,
        token_lifetime_seconds: int = 300,  # 5 minutes
        min_confidence_threshold: float = 0.6,
        content_api: Optional['ContentAPI'] = None,
    ):
        """Initialize gatekeeper.

        Args:
            max_tokens_per_hour: Maximum tokens per hour (budget limit)
            token_lifetime_seconds: How long tokens are valid
            min_confidence_threshold: Minimum confidence to proceed
            content_api: Optional ContentAPI for gate bursts
        """
        self.max_tokens_per_hour = max_tokens_per_hour
        self.token_lifetime_seconds = token_lifetime_seconds
        self.min_confidence_threshold = min_confidence_threshold
        self.content_api = content_api

        # Token tracking
        self.active_tokens: Dict[str, GateToken] = {}
        self.hourly_usage: List[float] = []  # Timestamps of token usage

        # Shallow check cache (query_hash -> result)
        self.shallow_cache: Dict[str, ShallowCheckResult] = {}
        self.cache_max_age = 3600  # 1 hour

        logger.info(
            "RetrievalGatekeeper initialized: max_tokens=%d/hour, token_lifetime=%ds, content_api=%s",
            max_tokens_per_hour,
            token_lifetime_seconds,
            "enabled" if content_api else "disabled"
        )

    def check_and_gate(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        force_allow: bool = False,
    ) -> tuple[GatekeeperStatus, Optional[GateToken], Dict[str, Any]]:
        """Perform shallow checks and gate expensive operations.

        Args:
            query: The query to check
            context: Additional context for checks
            force_allow: Bypass checks if True

        Returns:
            Tuple of (status, token_if_allowed, metadata)
        """
        query_hash = self._hash_query(query)

        # Clean up expired tokens and old cache
        self._cleanup()

        # Check if we have an active token for this query
        existing_token = self.active_tokens.get(query_hash)
        if existing_token and not existing_token.used:
            return GatekeeperStatus.ALLOW, existing_token, {"cached": True}

        # Perform shallow checks
        shallow_result = self._perform_shallow_checks(query, context)

        # Cache the result
        self.shallow_cache[query_hash] = shallow_result

        metadata = {
            "shallow_confidence": shallow_result.confidence,
            "shallow_reasons": shallow_result.reasons,
            "alternatives": shallow_result.suggested_alternatives,
        }

        # Force allow bypasses all checks
        if force_allow:
            token = self._create_token(query_hash)
            return GatekeeperStatus.ALLOW, token, {**metadata, "forced": True}

        # Check confidence threshold
        if shallow_result.confidence < self.min_confidence_threshold:
            return GatekeeperStatus.DENY, None, {
                **metadata,
                "reason": "low_confidence"
            }

        # Check budget limits
        if not self._check_budget():
            return GatekeeperStatus.DENY, None, {
                **metadata,
                "reason": "budget_exceeded"
            }

        # All checks passed - create token
        token = self._create_token(query_hash)
        return GatekeeperStatus.ALLOW, token, metadata

    def _perform_shallow_checks(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ShallowCheckResult:
        """Perform shallow checks to determine if query should proceed.

        Args:
            query: Query to check
            context: Additional context

        Returns:
            ShallowCheckResult
        """
        reasons = []
        alternatives = []
        confidence = 0.5  # Base confidence

        # Length check - very short queries might be too vague
        if len(query.strip()) < 10:
            reasons.append("Query too short/vague")
            confidence -= 0.2
            alternatives.append("Provide more specific query")

        # Keyword analysis
        query_lower = query.lower()

        # Check for Pokemon MD specific terms
        pmd_terms = ["pokemon", "mystery dungeon", "dungeon", "pokemon mystery dungeon"]
        has_pmd_context = any(term in query_lower for term in pmd_terms)

        if has_pmd_context:
            confidence += 0.3
            reasons.append("Contains Pokemon MD context")
        else:
            reasons.append("Missing Pokemon MD context")
            confidence -= 0.1

        # Check for specific actionable terms
        action_terms = ["how to", "how do", "strategy", "tactic", "guide", "walkthrough"]
        has_actionable = any(term in query_lower for term in action_terms)

        if has_actionable:
            confidence += 0.2
            reasons.append("Actionable query type")
        else:
            reasons.append("Non-actionable query")
            confidence -= 0.1

        # Context-based checks
        if context:
            # Check if we have recent similar queries
            recent_queries = context.get("recent_queries", [])
            if any(self._similar_queries(q, query) for q in recent_queries):
                confidence -= 0.2
                reasons.append("Similar to recent queries")
                alternatives.append("Check recent retrievals first")

            # Check current game state context
            game_state = context.get("game_state", {})
            floor = game_state.get("floor", 0)
            if floor > 0:
                confidence += 0.1
                reasons.append(f"In-dungeon context (floor {floor})")

        # Time-based checks (avoid spam)
        recent_hour_usage = sum(1 for t in self.hourly_usage if (time.time() - t) < 3600)

        if recent_hour_usage > self.max_tokens_per_hour * 0.8:  # 80% of budget
            confidence -= 0.3
            reasons.append("High recent usage")
            alternatives.append("Wait before additional queries")

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        can_proceed = confidence >= self.min_confidence_threshold

        return ShallowCheckResult(
            can_proceed=can_proceed,
            confidence=confidence,
            reasons=reasons,
            suggested_alternatives=alternatives,
        )

    def _similar_queries(self, q1: str, q2: str) -> bool:
        """Check if two queries are similar."""
        # Simple similarity check - could be enhanced with embeddings
        q1_words = set(q1.lower().split())
        q2_words = set(q2.lower().split())

        intersection = len(q1_words & q2_words)
        union = len(q1_words | q2_words)

        if union == 0:
            return False

        similarity = intersection / union
        return similarity > 0.6  # 60% word overlap

    def _check_budget(self) -> bool:
        """Check if we're within budget limits."""
        current_time = time.time()

        # Remove usage older than 1 hour
        self.hourly_usage = [t for t in self.hourly_usage if current_time - t < 3600]

        # Check if we can issue more tokens
        return len(self.hourly_usage) < self.max_tokens_per_hour

    def _create_token(self, query_hash: str) -> GateToken:
        """Create a new gate token."""
        token_id = f"token_{query_hash}_{int(time.time())}"
        current_time = time.time()

        token = GateToken(
            token_id=token_id,
            query_hash=query_hash,
            timestamp=current_time,
            expires_at=current_time + self.token_lifetime_seconds,
        )

        self.active_tokens[query_hash] = token
        self.hourly_usage.append(current_time)

        logger.info("Created gate token: %s", token_id)
        return token

    def use_token(self, token: GateToken) -> bool:
        """Mark a token as used."""
        if token.used:
            logger.warning("Token already used: %s", token.token_id)
            return False

        if time.time() > token.expires_at:
            logger.warning("Token expired: %s", token.token_id)
            return False

        token.used = True
        logger.info("Used gate token: %s", token.token_id)
        return True

    async def perform_gate_burst(self, query: str) -> Dict[str, Any]:
        """Perform a gate burst with content API calls.

        Returns results from bulk defaults first, then focused page if still needed.
        """
        if not self.content_api:
            return {"error": "No content API configured"}

        results = {
            "bulk_results": [],
            "focused_results": [],
            "total_calls": 0,
            "budget_remaining": self.content_api.get_budget_status()["remaining"]
        }

        try:
            # First call: bulk defaults
            logger.info("Performing gate burst: bulk defaults for query '%s'", query)
            bulk_pages = await self.content_api.fetch_guide()
            results["bulk_results"] = [p.__dict__ if hasattr(p, '__dict__') else p for p in bulk_pages]
            results["total_calls"] += 1

            # Check if we should do focused call
            if self.content_api.check_gate_token(f"burst_{hash(query)}"):
                logger.info("Performing gate burst: focused page for query '%s'", query)
                focused_pages = await self.content_api.search_old_memories(query)
                results["focused_results"] = [p.__dict__ if hasattr(p, '__dict__') else p for p in focused_pages]
                results["total_calls"] += 1
            else:
                logger.info("Skipping focused call - gate token limit reached")

        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error("Gate burst failed: %s", e)
            results["error"] = str(e)

        results["budget_remaining"] = self.content_api.get_budget_status()["remaining"]
        return results

    def _hash_query(self, query: str) -> str:
        """Create a hash of the query for caching."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def _cleanup(self) -> None:
        """Clean up expired tokens and old cache."""
        current_time = time.time()

        # Remove expired tokens
        expired_tokens = [
            query_hash for query_hash, token in self.active_tokens.items()
            if current_time > token.expires_at
        ]
        for query_hash in expired_tokens:
            del self.active_tokens[query_hash]

        # Remove old cache entries
        expired_cache = [
            query_hash for query_hash, result in self.shallow_cache.items()
            if current_time - result.timestamp > self.cache_max_age
        ]
        for query_hash in expired_cache:
            del self.shallow_cache[query_hash]

    def get_stats(self) -> Dict[str, Any]:
        """Get gatekeeper statistics."""
        current_time = time.time()

        return {
            "active_tokens": len(self.active_tokens),
            "hourly_usage": len(self.hourly_usage),
            "cache_size": len(self.shallow_cache),
            "budget_remaining": max(0, self.max_tokens_per_hour - len(self.hourly_usage)),
            "uptime_seconds": current_time - (current_time // 3600 * 3600),  # Since hour start
        }

    def reset_budget(self) -> None:
        """Reset the hourly budget (for testing)."""
        self.hourly_usage.clear()
        logger.info("Reset gatekeeper budget")