"""Model routing logic for selecting between 2B, 4B, and 8B Qwen3-VL models."""

from typing import Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelSize(Enum):
    """Available model sizes."""
    SIZE_2B = "2B"
    SIZE_4B = "4B"
    SIZE_8B = "8B"


# Model name mappings for Unsloth quantized models
MODEL_NAMES = {
    ModelSize.SIZE_2B: {
        "instruct": "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
        "thinking": "unsloth/Qwen3-VL-2B-Reasoning-unsloth-bnb-4bit",
    },
    ModelSize.SIZE_4B: {
        "instruct": "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
        "thinking": "unsloth/Qwen3-VL-4B-Reasoning-unsloth-bnb-4bit",
    },
    ModelSize.SIZE_8B: {
        "instruct": "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        "thinking": "unsloth/Qwen3-VL-8B-Reasoning-unsloth-bnb-4bit",
    },
}


@dataclass
class RoutingDecision:
    """Result of a model routing decision."""
    selected_model: ModelSize
    confidence_threshold_met: bool
    stuck_counter: int
    reasoning: str


class ModelRouter:
    """Routes between different Qwen3-VL model sizes based on situation complexity."""
    
    def __init__(
        self,
        confidence_2b_threshold: float = 0.8,
        confidence_4b_threshold: float = 0.6,
        stuck_escalation_threshold: int = 5,
    ):
        """Initialize the model router.
        
        Args:
            confidence_2b_threshold: Minimum confidence to use 2B model
            confidence_4b_threshold: Minimum confidence to use 4B model
            stuck_escalation_threshold: Number of stuck detections before escalating to 8B
        """
        self.confidence_2b_threshold = confidence_2b_threshold
        self.confidence_4b_threshold = confidence_4b_threshold
        self.stuck_escalation_threshold = stuck_escalation_threshold
        
    def select_model(
        self,
        confidence: Optional[float],
        stuck_counter: int,
        complexity_score: Optional[float] = None,
    ) -> RoutingDecision:
        """Select appropriate model based on confidence and context.
        
        Args:
            confidence: Model confidence score (0.0 to 1.0)
            stuck_counter: Number of times stuckness has been detected
            complexity_score: Optional complexity assessment (reserved for future use)
            
        Returns:
            RoutingDecision with model choice and reasoning
        """
        # Escalate to 8B if stuck
        if stuck_counter >= self.stuck_escalation_threshold:
            return RoutingDecision(
                selected_model=ModelSize.SIZE_8B,
                confidence_threshold_met=False,
                stuck_counter=stuck_counter,
                reasoning=f"Stuck detected {stuck_counter} times, escalating to 8B",
            )
        
        # Use 2B if high confidence
        if confidence is not None and confidence >= self.confidence_2b_threshold:
            return RoutingDecision(
                selected_model=ModelSize.SIZE_2B,
                confidence_threshold_met=True,
                stuck_counter=stuck_counter,
                reasoning=f"High confidence ({confidence:.2f}), using 2B for speed",
            )
        
        # Use 4B if medium confidence
        if confidence is not None and confidence >= self.confidence_4b_threshold:
            return RoutingDecision(
                selected_model=ModelSize.SIZE_4B,
                confidence_threshold_met=True,
                stuck_counter=stuck_counter,
                reasoning=f"Medium confidence ({confidence:.2f}), using 4B for reasoning",
            )
        
        # Default to 8B for low confidence
        reason = "Low confidence"
        if confidence is not None:
            reason += f" ({confidence:.2f})"
        
        return RoutingDecision(
            selected_model=ModelSize.SIZE_8B,
            confidence_threshold_met=False,
            stuck_counter=stuck_counter,
            reasoning=f"{reason}, using 8B for complex reasoning",
        )
    
    def get_model_name(self, model_size: ModelSize, use_thinking: bool = False) -> str:
        """Get the full model name for a given size and type.
        
        Args:
            model_size: The model size enum
            use_thinking: Whether to use thinking variant (vs instruct)
            
        Returns:
            Full model name string
        """
        variant = "thinking" if use_thinking else "instruct"
        return MODEL_NAMES[model_size][variant]
