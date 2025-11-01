"""Agent configuration classes.

Provides configuration classes for agent behavior and components.
"""


class AgentConfig:
    """Configuration for PokemonMDAgent behavior."""

    def __init__(
        self,
        screenshot_interval: float = 1.0,
        memory_poll_interval: float = 0.1,
        decision_interval: float = 0.5,
        max_runtime_hours: float = 1.0,
        enable_4up_capture: bool = True,
        enable_trajectory_logging: bool = True,
        enable_stuck_detection: bool = True,
        enable_skill_triggers: bool = False,
        skill_belly_threshold: float = 0.3,
        skill_hp_threshold: float = 0.25,
        skill_backoff_seconds: float = 5.0
    ):
        """Initialize agent configuration.

        Args:
            screenshot_interval: Seconds between screenshots
            memory_poll_interval: Seconds between memory polls
            decision_interval: Seconds between decisions
            max_runtime_hours: Maximum runtime in hours
            enable_4up_capture: Enable 4-up screenshot capture
            enable_trajectory_logging: Enable trajectory logging
            enable_stuck_detection: Enable stuckness detection
            enable_skill_triggers: Enable automatic skill triggers
            skill_belly_threshold: Belly threshold for triggers (0-1)
            skill_hp_threshold: HP threshold for triggers (0-1)
            skill_backoff_seconds: Seconds to wait after skill execution
        """
        self.screenshot_interval = screenshot_interval
        self.memory_poll_interval = memory_poll_interval
        self.decision_interval = decision_interval
        self.max_runtime_hours = max_runtime_hours
        self.enable_4up_capture = enable_4up_capture
        self.enable_trajectory_logging = enable_trajectory_logging
        self.enable_stuck_detection = enable_stuck_detection
        self.enable_skill_triggers = enable_skill_triggers
        self.skill_belly_threshold = skill_belly_threshold
        self.skill_hp_threshold = skill_hp_threshold
        self.skill_backoff_seconds = skill_backoff_seconds