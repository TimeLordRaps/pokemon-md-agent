"""
Router glue with uncertainty computation and policy thresholds.

Computes uncertainty from detector/RAG distances, applies policy_v2 thresholds & hysteresis.
Switches to same-size "Thinking" model when uncertainty ∈[0.55,0.7].
Triggers 8B prefetch and hot-swap when stuck>5 or entropy high.
"""

from typing import Dict, Any, Optional, List, Callable, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.router.policy_v2 import PolicyV2, ModelSize, RoutingDecision

if TYPE_CHECKING:
    from .message_packager import CopilotInput, Message
    from src.retrieval.maint.daemon import TemporalSiloMaintenanceDaemon, MaintenanceMetrics

logger = logging.getLogger(__name__)


class ModelSwitchReason(Enum):
    """Reasons for model switching."""
    LOW_CONFIDENCE = "low_confidence"
    STUCK_ESCALATION = "stuck_escalation"
    HIGH_ENTROPY = "high_entropy"
    UNCERTAINTY_RANGE = "uncertainty_range"
    POLICY_THRESHOLD = "policy_threshold"


@dataclass
class UncertaintyResult:
    """Result of uncertainty computation and routing decision."""
    uncertainty_score: float
    should_switch_model: bool
    recommended_model: ModelSize
    reason: List[ModelSwitchReason]

    def __str__(self) -> str:
        reasons_str = ", ".join([r.value for r in self.reason])
        return (f"UncertaintyResult(uncertainty={self.uncertainty_score:.3f}, "
                f"switch={self.should_switch_model}, model={self.recommended_model.value}, "
                f"reason=[{reasons_str}])")


class RouterGlueError(Exception):
    """Exception raised for router glue errors."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause


class RouterGlue:
    """
    Router glue that computes uncertainty and applies policy thresholds.

    Integrates with policy_v2 for hysteresis and secondary triggers.
    Handles thinking variant switching in uncertainty range [0.55, 0.7].
    Manages stuck escalation with 8B prefetch and hot-swap.
    """

    def __init__(
        self,
        policy_v2: Optional[PolicyV2] = None,
        uncertainty_threshold_low: float = 0.55,
        uncertainty_threshold_high: float = 0.7,
        stuck_threshold: int = 5,
        entropy_threshold: float = 0.8,
        prefetch_callback: Optional[Callable[[ModelSize], None]] = None,
        hotswap_callback: Optional[Callable[[ModelSize], None]] = None,
        maintenance_daemon: Optional["TemporalSiloMaintenanceDaemon"] = None,
    ):
        """
        Initialize router glue.

        Args:
            policy_v2: PolicyV2 instance for hysteresis and thresholds
            uncertainty_threshold_low: Lower bound for thinking variant switch
            uncertainty_threshold_high: Upper bound for thinking variant switch
            stuck_threshold: Threshold for stuck escalation
            entropy_threshold: Threshold for entropy-based switching
            prefetch_callback: Callback to prefetch a model (for stuck/entropy cases)
            hotswap_callback: Callback to perform hot-swap to model

        Raises:
            RouterGlueError: If thresholds are invalid
        """
        if not (0.0 <= uncertainty_threshold_low < uncertainty_threshold_high <= 1.0):
            raise RouterGlueError("Invalid uncertainty thresholds")

        self.policy_v2 = policy_v2 or PolicyV2()
        self.uncertainty_threshold_low = uncertainty_threshold_low
        self.uncertainty_threshold_high = uncertainty_threshold_high
        self.stuck_threshold = stuck_threshold
        self.entropy_threshold = entropy_threshold
        self.prefetch_callback = prefetch_callback
        self.hotswap_callback = hotswap_callback
        self._maintenance_daemon = maintenance_daemon

        # Async executor for prefetch operations
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="router_prefetch")

        logger.info(
            "Initialized RouterGlue: uncertainty_range=[%.2f, %.2f], "
            "stuck_threshold=%d, entropy_threshold=%.2f",
            uncertainty_threshold_low,
            uncertainty_threshold_high,
            stuck_threshold,
            entropy_threshold
        )

    def attach_maintenance_daemon(
        self, daemon: "TemporalSiloMaintenanceDaemon"
    ) -> None:
        """Attach or replace the maintenance daemon."""
        self._maintenance_daemon = daemon

    @property
    def maintenance_daemon(self) -> Optional["TemporalSiloMaintenanceDaemon"]:
        """Access the attached maintenance daemon, if any."""
        return self._maintenance_daemon

    def compute_uncertainty(self, perception_data: Dict[str, Any]) -> UncertaintyResult:
        """
        Compute uncertainty from perception data and determine routing.

        Args:
            perception_data: Dictionary with detector/RAG distances, stuckness, entropy

        Returns:
            UncertaintyResult with score, decision, and reasoning

        Raises:
            RouterGlueError: If required data is missing or invalid
        """
        try:
            # Compute uncertainty from multiple sources
            uncertainty_score = self._compute_combined_uncertainty(perception_data)

            # Determine if model switch is needed
            should_switch, recommended_model, reasons = self._determine_switch(
                uncertainty_score, perception_data
            )

            result = UncertaintyResult(
                uncertainty_score=uncertainty_score,
                should_switch_model=should_switch,
                recommended_model=recommended_model,
                reason=reasons
            )

            logger.debug("Computed uncertainty result: %s", result)
            return result

        except Exception as e:
            logger.error("Failed to compute uncertainty: %s", e)
            raise RouterGlueError("Uncertainty computation failed", e) from e

    def _compute_combined_uncertainty(self, perception_data: Dict[str, Any]) -> float:
        """
        Compute combined uncertainty from detector and RAG distances.

        Args:
            perception_data: Perception data dictionary

        Returns:
            Combined uncertainty score [0.0, 1.0]
        """
        uncertainties = []

        # Detector distances uncertainty
        if "detector_distances" in perception_data:
            detector_uncertainty = self.compute_uncertainty_from_distances(
                perception_data["detector_distances"]
            )
            uncertainties.append(detector_uncertainty)

        # RAG distances uncertainty
        if "rag_distances" in perception_data:
            rag_uncertainty = self.compute_uncertainty_from_rag(
                perception_data["rag_distances"]
            )
            uncertainties.append(rag_uncertainty)

        # Stuckness contributes to uncertainty
        if "stuckness_score" in perception_data:
            stuck_score = min(perception_data["stuckness_score"] / 10.0, 1.0)  # Normalize
            uncertainties.append(stuck_score)

        # Entropy contributes to uncertainty
        if "entropy" in perception_data:
            entropy_score = perception_data["entropy"]
            uncertainties.append(entropy_score)

        if not uncertainties:
            logger.warning("No uncertainty sources available, using default 0.5")
            return 0.5

        # Weighted combination
        combined = sum(uncertainties) / len(uncertainties)
        return max(0.0, min(1.0, combined))  # Clamp to [0, 1]

    def compute_uncertainty_from_distances(self, distances: List[float]) -> float:
        """
        Compute uncertainty from detector distances.

        Args:
            distances: List of distance values

        Returns:
            Uncertainty score [0.0, 1.0]

        Raises:
            RouterGlueError: If distances list is empty
        """
        if not distances:
            raise RouterGlueError("Empty detector distances list")

        # Higher distances = higher uncertainty
        avg_distance = sum(distances) / len(distances)
        uncertainty = min(avg_distance * 2.0, 1.0)  # Scale and clamp

        logger.debug("Detector uncertainty: avg_dist=%.3f -> uncertainty=%.3f",
                    avg_distance, uncertainty)
        return uncertainty

    def compute_uncertainty_from_rag(self, distances: List[float]) -> float:
        """
        Compute uncertainty from RAG retrieval distances.

        Args:
            distances: List of RAG distance values

        Returns:
            Uncertainty score [0.0, 1.0]

        Raises:
            RouterGlueError: If distances list is empty
        """
        if not distances:
            raise RouterGlueError("Empty RAG distances list")

        # Higher distances (lower similarity) = higher uncertainty
        avg_distance = sum(distances) / len(distances)
        uncertainty = min(avg_distance, 1.0)  # Direct mapping, clamp to 1.0

        logger.debug("RAG uncertainty: avg_dist=%.3f -> uncertainty=%.3f",
                    avg_distance, uncertainty)
        return uncertainty

    def _determine_switch(
        self,
        uncertainty: float,
        perception_data: Dict[str, Any]
    ) -> tuple[bool, ModelSize, List[ModelSwitchReason]]:
        """
        Determine if model switch is needed and to which model.

        Args:
            uncertainty: Computed uncertainty score
            perception_data: Perception data

        Returns:
            Tuple of (should_switch, recommended_model, reasons)
        """
        reasons = []
        should_switch = False
        recommended_model = ModelSize.SIZE_4B  # Default

        # Stuck escalation with prefetch
        stuck_score = perception_data.get("stuckness_score", 0)
        if stuck_score >= self.stuck_threshold:
            should_switch = True
            recommended_model = ModelSize.SIZE_8B
            reasons.append(ModelSwitchReason.STUCK_ESCALATION)
            logger.info("Stuck escalation triggered: stuck_score=%d >= threshold=%d",
                        stuck_score, self.stuck_threshold)
            # Trigger 8B prefetch
            self._trigger_prefetch(ModelSize.SIZE_8B, "stuck escalation")

        # High entropy with prefetch
        entropy = perception_data.get("entropy", 0.0)
        if entropy >= self.entropy_threshold:
            should_switch = True
            recommended_model = ModelSize.SIZE_8B
            reasons.append(ModelSwitchReason.HIGH_ENTROPY)
            logger.info("High entropy escalation: entropy=%.3f >= threshold=%.3f",
                        entropy, self.entropy_threshold)
            # Trigger 8B prefetch
            self._trigger_prefetch(ModelSize.SIZE_8B, "high entropy")

        # Uncertainty range for thinking variant (no prefetch needed)
        if self.uncertainty_threshold_low <= uncertainty <= self.uncertainty_threshold_high:
            should_switch = True
            recommended_model = ModelSize.SIZE_4B  # Same size, thinking variant
            reasons.append(ModelSwitchReason.UNCERTAINTY_RANGE)
            logger.info("Uncertainty range thinking switch: uncertainty=%.3f in [%.2f, %.2f]",
                        uncertainty, self.uncertainty_threshold_low, self.uncertainty_threshold_high)

        # Low confidence fallback with prefetch
        if uncertainty > self.uncertainty_threshold_high:
            should_switch = True
            recommended_model = ModelSize.SIZE_8B
            reasons.append(ModelSwitchReason.LOW_CONFIDENCE)
            logger.info("Low confidence escalation: uncertainty=%.3f > threshold=%.2f",
                        uncertainty, self.uncertainty_threshold_high)
            # Trigger 8B prefetch
            self._trigger_prefetch(ModelSize.SIZE_8B, "low confidence")

        return should_switch, recommended_model, reasons

    def should_use_thinking_variant(self, uncertainty: float, current_model: ModelSize) -> bool:
        """
        Check if thinking variant should be used.

        Args:
            uncertainty: Current uncertainty score
            current_model: Current model size

        Returns:
            True if thinking variant preferred
        """
        # Use thinking variant in uncertainty range
        in_range = self.uncertainty_threshold_low <= uncertainty <= self.uncertainty_threshold_high
        if in_range:
            return True

        # Always use thinking for 8B
        if current_model == ModelSize.SIZE_8B:
            return True

        return False

    def make_routing_decision(
        self,
        confidence: Optional[float],
        stuck_counter: int,
        perception_data: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        Make integrated routing decision with uncertainty computation.

        Args:
            confidence: Optional confidence score
            stuck_counter: Stuck detection counter
            perception_data: Optional perception data for uncertainty

        Returns:
            RoutingDecision with integrated logic
        """
        # Get base policy decision
        base_decision = self.policy_v2.select_model(confidence, stuck_counter)

        # If perception data available, apply uncertainty logic
        if perception_data:
            uncertainty_result = self.compute_uncertainty(perception_data)

            if uncertainty_result.should_switch_model:
                # Override with uncertainty-based decision
                final_model = uncertainty_result.recommended_model
                reasoning = f"Uncertainty override: {uncertainty_result}"
                logger.info("Router decision: %s (uncertainty-based)", reasoning)
            else:
                final_model = base_decision.selected_model
                reasoning = f"Policy decision: {base_decision.reasoning}"
                logger.debug("Router decision: %s (policy-based)", reasoning)
        else:
            final_model = base_decision.selected_model
            reasoning = base_decision.reasoning
            logger.debug("Router decision: %s (no perception data)", reasoning)

        # Log decision details
        logger.info(
            "Routing decision made: model=%s, confidence=%.3f, stuck_counter=%d, "
            "perception_data_available=%s, reasoning='%s'",
            final_model.value,
            confidence if confidence is not None else -1.0,
            stuck_counter,
            perception_data is not None,
            reasoning
        )

        # Determine use_thinking based on uncertainty
        use_thinking = False
        if perception_data:
            uncertainty_result = self.compute_uncertainty(perception_data)
            use_thinking = uncertainty_result.should_switch_model and uncertainty_result.recommended_model == final_model

        return RoutingDecision(
            selected_model=final_model,
            use_thinking=use_thinking,
            confidence_threshold_met=base_decision.confidence_threshold_met,
            stuck_counter=stuck_counter,
            reasoning=reasoning,
        )

    def _trigger_prefetch(self, model_size: ModelSize, reason: str) -> None:
        """
        Trigger prefetch of a model for future hot-swapping.

        Args:
            model_size: Model size to prefetch
            reason: Reason for prefetch
        """
        if self.prefetch_callback:
            try:
                # Run prefetch asynchronously to avoid blocking
                self.executor.submit(self.prefetch_callback, model_size)
                logger.info("Prefetch triggered for %s model: %s", model_size.value, reason)
            except Exception as e:
                logger.warning("Prefetch failed for %s: %s", model_size.value, e)
        else:
            logger.debug("No prefetch callback configured, skipping prefetch for %s", model_size.value)

    def perform_hotswap(self, model_size: ModelSize, reason: str) -> bool:
        """
        Perform hot-swap to the specified model.

        Args:
            model_size: Model size to hot-swap to
            reason: Reason for hot-swap

        Returns:
            True if hot-swap was successful
        """
        if self.hotswap_callback:
            try:
                self.hotswap_callback(model_size)
                logger.info("Hot-swap completed to %s model: %s", model_size.value, reason)
                return True
            except Exception as e:
                logger.error("Hot-swap failed to %s: %s", model_size.value, e)
                return False
        else:
            logger.warning("No hot-swap callback configured")
            return False

    def execute_turn_loop(
        self,
        copilot_input: "CopilotInput",
        perception_data: Optional[Dict[str, Any]] = None,
        stuck_counter: int = 0
    ) -> str:
        """
        Execute the inference turn loop: retrieve → package → route → generate → act.

        Args:
            copilot_input: CopilotInput with png, meta.json, and retrieved thumbnails.
            perception_data: Optional perception data for uncertainty computation.
            stuck_counter: Current stuck counter value.

        Returns:
            Action string from LLM (only the action, env executes).

        Raises:
            RouterGlueError: If any step in the loop fails.
        """
        try:
            logger.info("Starting turn loop execution")

            # Step 1: Retrieve - already done via copilot_input.retrieved_thumbnails
            retrieved_thumbnails = copilot_input.retrieved_thumbnails
            logger.info("Retrieved %d thumbnails", len(retrieved_thumbnails))

            # Step 2: Package - use message packager to create messages
            from src.orchestrator.message_packager import pack_from_copilot

            # Determine policy hint from perception data or default
            policy_hint = "explore"  # Default
            if perception_data and "policy_hint" in perception_data:
                policy_hint = perception_data["policy_hint"]

            # Get routing decision to determine model size
            uncertainty_result = self.compute_uncertainty(perception_data or {})
            routing_decision = self.make_routing_decision(
                confidence=None,  # Will be determined by uncertainty
                stuck_counter=stuck_counter,
                perception_data=perception_data
            )

            # Map ModelSize to model_size string
            model_size_map = {
                ModelSize.SIZE_2B: "2B",
                ModelSize.SIZE_4B: "4B",
                ModelSize.SIZE_8B: "8B"
            }
            model_size_str = model_size_map.get(routing_decision.selected_model, "4B")

            messages = pack_from_copilot(copilot_input, policy_hint, model_size_str)
            logger.info("Packaged %d messages for %s model", len(messages), model_size_str)

            # Step 3: Route - already handled via routing_decision
            selected_model = routing_decision.selected_model
            logger.info("Routed to model: %s", selected_model.value)

            # Step 4: Generate - call LLM with messages (mock implementation)
            action_string = self._generate_action(messages, selected_model)
            logger.info("Generated action: %s", action_string)

            # Step 5: Act - return action string only (env executes)
            self._run_maintenance_cycle()
            return action_string

        except Exception as e:
            logger.error("Turn loop execution failed: %s", e)
            raise RouterGlueError("Turn loop execution failed", e) from e

    def _generate_action(self, messages: List["Message"], model: ModelSize) -> str:
        """
        Generate action from LLM using packaged messages.

        Args:
            messages: Packaged messages for LLM.
            model: Selected model size.

        Returns:
            Action string from LLM.

        Note:
            This is a placeholder - actual LLM integration would go here.
        """
        # Placeholder implementation - in real system this would call the actual LLM
        logger.info("Mock LLM call with %d messages to %s model", len(messages), model.value)

        # Extract action from last message (MSG[0] now message)
        now_message = messages[-1] if messages else None
        if now_message and "Please provide the next action" in now_message.text:
            # Mock action generation based on policy hint in message
            if "explore" in now_message.text:
                return "move_forward"
            elif "fight" in now_message.text:
                return "attack"
            else:
                return "wait"

        return "wait"  # Default action

    def _run_maintenance_cycle(self) -> None:
        """Invoke temporal silo maintenance if a daemon is attached."""
        if self._maintenance_daemon is None:
            return

        try:
            metrics = self._maintenance_daemon.step()
            if metrics is None:
                return

            total_compact = sum(metrics.total_removed_compaction.values())
            total_expire = sum(metrics.total_removed_retention.values())

            if total_compact or total_expire:
                logger.info(
                    "Temporal maintenance removed entries (compact=%d, expire=%d)",
                    total_compact,
                    total_expire,
                )
            else:
                logger.debug("Temporal maintenance cycle completed with no removals")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Maintenance daemon step failed: %s", exc)


def to_model_payload(blob: dict) -> dict:
    """
    Transform packaged blob into model payload format.

    Takes the dict from package_triplet and transforms it into the format
    expected by the model payload, which means wrapping it or adjusting keys
    as needed for the router. Pure format transformation with no routing logic.

    Args:
        blob: Dict with 'system', 'plan', 'act' keys from package_triplet

    Returns:
        Dict in model payload format
    """
    return dict(blob)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get router glue statistics.

        Returns:
            Dictionary with configuration and thresholds
        """
        return {
            "uncertainty_threshold_low": self.uncertainty_threshold_low,
            "uncertainty_threshold_high": self.uncertainty_threshold_high,
            "stuck_threshold": self.stuck_threshold,
            "entropy_threshold": self.entropy_threshold,
            "policy_v2_configured": self.policy_v2 is not None,
            "prefetch_callback_configured": self.prefetch_callback is not None,
            "hotswap_callback_configured": self.hotswap_callback is not None,
        }
