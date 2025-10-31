"""Enhanced router policy with hysteresis and secondary triggers."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

from ..agent.model_router import ModelRouter, ModelSize, RoutingDecision

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of triggers for model escalation."""
    CONFIDENCE_LOW = "confidence_low"
    STUCK_DETECTED = "stuck_detected"
    IOU_LOW = "iou_low"
    RAG_DISTANCE_HIGH = "rag_distance_high"
    TIME_SINCE_STAIRS = "time_since_stairs"
    HYSTERESIS_RESET = "hysteresis_reset"
    SPRITE_MAP_ENTROPY_HIGH = "sprite_map_entropy_high"
    RETRIEVAL_CONFLICT = "retrieval_conflict"


@dataclass
class HysteresisState:
    """Hysteresis state for smooth model transitions."""
    current_model: ModelSize
    last_transition_time: float = 0.0
    transition_count: int = 0
    confidence_history: List[float] = field(default_factory=list)
    stuck_history: List[bool] = field(default_factory=list)
    
    def should_transition(
        self,
        new_model: ModelSize,
        confidence: Optional[float],
        stuck_counter: int,
        hysteresis_window: int = 3
    ) -> bool:
        """Check if transition should occur with hysteresis."""
        if new_model == self.current_model:
            return False
        
        # Update history
        if confidence is not None:
            self.confidence_history.append(confidence)
            if len(self.confidence_history) > hysteresis_window:
                self.confidence_history.pop(0)
        
        self.stuck_history.append(stuck_counter > 0)
        if len(self.stuck_history) > hysteresis_window:
            self.stuck_history.pop(0)
        
        # Require consistent signal for hysteresis_window steps
        if len(self.confidence_history) < hysteresis_window:
            return False
        
        # Check confidence trend
        if new_model == ModelSize.SIZE_2B:
            # Only go to 2B if consistently high confidence
            return all(c >= 0.85 for c in self.confidence_history[-hysteresis_window:])
        
        elif new_model == ModelSize.SIZE_4B:
            # Go to 4B if medium confidence trend
            recent_conf = self.confidence_history[-hysteresis_window:]
            return all(0.6 <= c < 0.85 for c in recent_conf)
        
        else:  # SIZE_8B
            # Go to 8B if consistently low confidence or stuck
            low_conf = all(c < 0.6 for c in self.confidence_history[-hysteresis_window:])
            stuck_trend = sum(self.stuck_history[-hysteresis_window:]) >= 2
            return low_conf or stuck_trend
    
    def record_transition(self, new_model: ModelSize) -> None:
        """Record a model transition."""
        self.current_model = new_model
        self.last_transition_time = time.time()
        self.transition_count += 1
        logger.info("Model transition: %s (hysteresis transition #%d)", 
                   new_model.value, self.transition_count)


@dataclass
class SecondaryTriggers:
    """Secondary triggers for model escalation."""
    iou_threshold: float = 0.7  # Minimum IoU between frames
    rag_distance_threshold: float = 0.8  # Maximum RAG distance
    max_time_since_stairs: float = 300.0  # 5 minutes max without stairs
    low_iou_window: int = 5  # Frames to check for low IoU
    sprite_map_entropy_threshold: float = 0.85  # High entropy threshold for sprite maps
    retrieval_conflict_threshold: int = 3  # Number of conflicts triggering escalation

    # Runtime state
    recent_iou_scores: List[float] = field(default_factory=list)
    last_stairs_time: float = 0.0
    rag_distances: List[float] = field(default_factory=list)
    recent_sprite_entropies: List[float] = field(default_factory=list)
    retrieval_conflicts: int = 0
    
    def check_triggers(self) -> List[TriggerType]:
        """Check all secondary triggers and return active ones."""
        triggers = []

        # IoU trigger
        if len(self.recent_iou_scores) >= self.low_iou_window:
            recent_iou = self.recent_iou_scores[-self.low_iou_window:]
            avg_iou = sum(recent_iou) / len(recent_iou)
            if avg_iou < self.iou_threshold:
                triggers.append(TriggerType.IOU_LOW)

        # RAG distance trigger
        if self.rag_distances:
            avg_rag_dist = sum(self.rag_distances) / len(self.rag_distances)
            if avg_rag_dist > self.rag_distance_threshold:
                triggers.append(TriggerType.RAG_DISTANCE_HIGH)

        # Time since stairs trigger
        time_since_stairs = time.time() - self.last_stairs_time
        if time_since_stairs > self.max_time_since_stairs:
            triggers.append(TriggerType.TIME_SINCE_STAIRS)

        # Sprite map entropy trigger
        if self.recent_sprite_entropies:
            avg_entropy = sum(self.recent_sprite_entropies) / len(self.recent_sprite_entropies)
            if avg_entropy > self.sprite_map_entropy_threshold:
                triggers.append(TriggerType.SPRITE_MAP_ENTROPY_HIGH)

        # Retrieval conflict trigger
        if self.retrieval_conflicts >= self.retrieval_conflict_threshold:
            triggers.append(TriggerType.RETRIEVAL_CONFLICT)

        return triggers
    
    def update_iou(self, iou_score: float) -> None:
        """Update IoU score history."""
        self.recent_iou_scores.append(iou_score)
        if len(self.recent_iou_scores) > 10:  # Keep last 10
            self.recent_iou_scores.pop(0)
    
    def update_rag_distance(self, distance: float) -> None:
        """Update RAG distance history."""
        self.rag_distances.append(distance)
        if len(self.rag_distances) > 5:  # Keep last 5
            self.rag_distances.pop(0)
    
    def update_stairs_time(self) -> None:
        """Update last stairs detection time."""
        self.last_stairs_time = time.time()


class PolicyV2:
    """Enhanced router policy with hysteresis and secondary triggers."""
    
    def __init__(
        self,
        base_router: Optional[ModelRouter] = None,
        hysteresis_window: int = 3,
        secondary_triggers: Optional[SecondaryTriggers] = None,
    ):
        """Initialize enhanced policy.
        
        Args:
            base_router: Base ModelRouter instance
            hysteresis_window: Number of steps for hysteresis
            secondary_triggers: Secondary trigger configuration
        """
        self.base_router = base_router or ModelRouter()
        self.hysteresis = HysteresisState(current_model=ModelSize.SIZE_4B)  # Start with 4B
        self.secondary = secondary_triggers or SecondaryTriggers()
        self.hysteresis_window = hysteresis_window
        
        # Override base router thresholds for hysteresis
        self.base_router.confidence_2b_threshold = 0.8
        self.base_router.confidence_4b_threshold = 0.6
        self.base_router.stuck_escalation_threshold = 5
        
        logger.info("Initialized PolicyV2 with hysteresis window %d", hysteresis_window)
    
    def select_model(
        self,
        confidence: Optional[float],
        stuck_counter: int,
        perception_data: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Select model with enhanced policy.
        
        Args:
            confidence: Current confidence score
            stuck_counter: Stuck detection counter
            perception_data: Additional perception data
            
        Returns:
            Routing decision
        """
        # Get base routing decision
        base_decision = self.base_router.select_model(confidence, stuck_counter)
        
        # Check secondary triggers
        secondary_triggers = self.secondary.check_triggers()
        
        # Apply hysteresis
        should_transition = self.hysteresis.should_transition(
            base_decision.selected_model,
            confidence,
            stuck_counter,
            self.hysteresis_window
        )
        
        # Force escalation on secondary triggers
        force_escalation = any(trigger in [
            TriggerType.STUCK_DETECTED,
            TriggerType.IOU_LOW,
            TriggerType.RAG_DISTANCE_HIGH,
            TriggerType.TIME_SINCE_STAIRS
        ] for trigger in secondary_triggers)
        
        if force_escalation and base_decision.selected_model != ModelSize.SIZE_8B:
            logger.warning("Secondary triggers forcing 8B escalation: %s", 
                          [t.value for t in secondary_triggers])
            final_model = ModelSize.SIZE_8B
            reasoning = f"Secondary triggers: {[t.value for t in secondary_triggers]}"
        elif should_transition:
            final_model = base_decision.selected_model
            reasoning = f"Hysteresis transition to {final_model.value}"
            self.hysteresis.record_transition(final_model)
        else:
            final_model = self.hysteresis.current_model
            reasoning = f"Hysteresis prevents transition, staying with {final_model.value}"
        
        # Update secondary trigger state
        if perception_data:
            self._update_secondary_state(perception_data)
        
        return RoutingDecision(
            selected_model=final_model,
            confidence_threshold_met=base_decision.confidence_threshold_met,
            stuck_counter=stuck_counter,
            reasoning=reasoning,
        )
    
    def _update_secondary_state(self, perception_data: Dict[str, Any]) -> None:
        """Update secondary trigger state from perception data."""
        # Update IoU if available
        if "iou_score" in perception_data:
            self.secondary.update_iou(perception_data["iou_score"])
        
        # Update RAG distance if available
        if "rag_distance" in perception_data:
            self.secondary.update_rag_distance(perception_data["rag_distance"])
        
        # Update stairs time if stairs detected
        if perception_data.get("stairs_detected", False):
            self.secondary.update_stairs_time()
    
    def prefer_thinking_variant(
        self,
        confidence: float,
        current_model: ModelSize
    ) -> bool:
        """Check if thinking variant should be preferred.
        
        Args:
            confidence: Current confidence
            current_model: Current model size
            
        Returns:
            True if thinking variant preferred
        """
        # Prefer thinking variant in uncertainty range
        if 0.55 <= confidence < 0.7:
            return True
        
        # Always use thinking for 8B
        if current_model == ModelSize.SIZE_8B:
            return True
        
        return False
    
    def get_policy_stats(self) -> Dict[str, Any]:
        """Get policy statistics."""
        return {
            "current_model": self.hysteresis.current_model.value,
            "transition_count": self.hysteresis.transition_count,
            "last_transition_time": self.hysteresis.last_transition_time,
            "confidence_history_len": len(self.hysteresis.confidence_history),
            "active_secondary_triggers": [t.value for t in self.secondary.check_triggers()],
            "hysteresis_window": self.hysteresis_window,
        }