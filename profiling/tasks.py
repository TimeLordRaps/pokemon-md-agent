"""
Micro-task suite for Qwen3-VL benchmarking.

Provides small, deterministic probes approximating PMD-Red subtasks:
- Navigation hinting
- Stairs detection via text
- Item choice rationale
- Simple combat heuristic
- Memory recall

Each task returns a score proxy (0.0-1.0) based on model performance.
"""

from typing import Dict, List, Any, Callable
import re
from dataclasses import dataclass


@dataclass
class TaskResult:
    """Result of a micro-task execution."""
    score: float  # 0.0-1.0 performance proxy
    metadata: Dict[str, Any]  # Additional metrics


@dataclass
class MicroTask:
    """A micro-task definition."""
    name: str
    description: str
    prompt: str
    expected_patterns: List[str]  # Regex patterns for correct responses
    scoring_fn: Callable[[str], float]  # Function to score response


# Tiny Woods situational micro-tasks
MICRO_TASKS = [
    MicroTask(
        name="detect_stairs",
        description="Identify when stairs are adjacent and report the direction to move.",
        prompt=(
            "Tiny Woods minimap summary: Player at (5,5). Walkable tiles north/south/east. "
            "Scanner ping: stairs located at (5,4). Are the stairs adjacent and which way should you step?"
        ),
        expected_patterns=[r"(north|up)", r"stairs"],
        scoring_fn=lambda response: min(
            1.0,
            len(re.findall(r"(north|up)", response.lower())) * 0.5
            + len(re.findall(r"stairs", response.lower())) * 0.5,
        ),
    ),
    MicroTask(
        name="enemy_nearby",
        description="Assess enemy proximity from radar hints and recommend action.",
        prompt=(
            "Enemy radar: hostile Pidgey detected two tiles east and one tile south of you. "
            "HP: 70%, partner nearby. Do you prep for combat or reposition first?"
        ),
        expected_patterns=[r"(east|south|approach)", r"(prepare|position|kite|reposition)"],
        scoring_fn=lambda response: min(
            1.0,
            len(re.findall(r"(east|south|approach)", response.lower())) * 0.4
            + len(re.findall(r"(prepare|position|kite|reposition)", response.lower())) * 0.6,
        ),
    ),
    MicroTask(
        name="inventory_parse",
        description="Parse inventory text and select the correct utility item.",
        prompt=(
            "Inventory summary: Slot1 Oran Berry, Slot2 Pecha Seed, Slot3 Blast Seed, Slot4 Max Elixir. "
            "Your leader is poisoned and losing 5 HP per step. Which item do you use and why?"
        ),
        expected_patterns=[r"(pecha)", r"(poison|status|cure)"],
        scoring_fn=lambda response: min(
            1.0,
            len(re.findall(r"(pecha)", response.lower())) * 0.6
            + len(re.findall(r"(poison|status|cure)", response.lower())) * 0.4,
        ),
    ),
    MicroTask(
        name="dialog_open",
        description="Detect whether an in-game dialog is still active.",
        prompt=(
            "HUD overlay reads: 'Press B to close the message log.' The action grid is dimmed. "
            "Is the dialog state still open? Answer yes or no with reasoning."
        ),
        expected_patterns=[r"(yes|dialog|open)", r"(press|message|dim)"],
        scoring_fn=lambda response: min(
            1.0,
            len(re.findall(r"(yes|dialog|open)", response.lower())) * 0.5
            + len(re.findall(r"(press|message|dim)", response.lower())) * 0.5,
        ),
    ),
    MicroTask(
        name="choose_move",
        description="Select the optimal move based on type matchup and positioning.",
        prompt=(
            "You are a Fire-type facing a Bug/Grass Paras in a one-tile corridor. "
            "Moves available: Ember (special, ranged), Scratch (physical, adjacent), Smokescreen (status). "
            "Which move should you choose and why?"
        ),
        expected_patterns=[r"(ember)", r"(super.?effective|bug|grass|type)"],
        scoring_fn=lambda response: min(
            1.0,
            len(re.findall(r"(ember)", response.lower())) * 0.6
            + len(re.findall(r"(super.?effective|bug|grass|type)", response.lower())) * 0.4,
        ),
    ),
    MicroTask(
        name="path_chunk_decide",
        description="Plan the next 3-step movement chunk through branching corridors.",
        prompt=(
            "Floor layout snippet: to the north a hallway curves east after 3 tiles toward unexplored rooms; "
            "to the west a dead-end with an apple is two tiles away. "
            "You need to find stairs quickly. Outline your next three moves."
        ),
        expected_patterns=[r"(north|east|forward)", r"(stairs|explore|progress)"],
        scoring_fn=lambda response: min(
            1.0,
            len(re.findall(r"(north|east|forward)", response.lower())) * 0.5
            + len(re.findall(r"(stairs|explore|progress)", response.lower())) * 0.5,
        ),
    ),
    MicroTask(
        name="trap_spotting",
        description="Recognize visual indicators that imply a trap tile.",
        prompt=(
            "You notice a floor tile with a faint cross pattern and a slight shimmer compared to adjacent tiles. "
            "You're moving fast. What should you assume about that tile?"
        ),
        expected_patterns=[r"(trap|trigger|hazard)", r"(avoid|step around|skip)"],
        scoring_fn=lambda response: min(
            1.0,
            len(re.findall(r"(trap|trigger|hazard)", response.lower())) * 0.6
            + len(re.findall(r"(avoid|step around|skip)", response.lower())) * 0.4,
        ),
    ),
    MicroTask(
        name="status_effect_response",
        description="Recommend immediate action when afflicted by a status condition.",
        prompt=(
            "Leader is Burned after stepping on a flame trap. HP ticking down during movement. "
            "Inventory contains Oran Berry, Rawst Berry, Heal Seed. What do you do next?"
        ),
        expected_patterns=[r"(rawst|heal seed)", r"(burn|status|recover)"],
        scoring_fn=lambda response: min(
            1.0,
            len(re.findall(r"(rawst|heal seed)", response.lower())) * 0.6
            + len(re.findall(r"(burn|status|recover)", response.lower())) * 0.4,
        ),
    ),
    MicroTask(
        name="item_pickup_priority",
        description="Prioritize item pickup when inventory space is scarce.",
        prompt=(
            "Your bag has one free slot. Items spotted: Max Elixir, Oran Berry, Gravelerock. "
            "Team HP averages 35%, moves are mostly full PP. Which item do you pick up?"
        ),
        expected_patterns=[r"(oran)", r"(surviv|heal|hp)"],
        scoring_fn=lambda response: min(
            1.0,
            len(re.findall(r"(oran)", response.lower())) * 0.6
            + len(re.findall(r"(surviv|heal|hp)", response.lower())) * 0.4,
        ),
    ),
    MicroTask(
        name="corridor_branch_choice",
        description="Choose the better branch when balancing loot vs objective.",
        prompt=(
            "Scanner reveals: east corridor leads to stairs after four tiles; west corridor ends with a random item. "
            "Mission objective is urgent rescue. Which branch do you choose?"
        ),
        expected_patterns=[r"(east|stairs)", r"(objective|rescue|priority)"],
        scoring_fn=lambda response: min(
            1.0,
            len(re.findall(r"(east|stairs)", response.lower())) * 0.5
            + len(re.findall(r"(objective|rescue|priority)", response.lower())) * 0.5,
        ),
    ),
    MicroTask(
        name="monster_house_escape",
        description="React to discovering a Monster House while low on resources.",
        prompt=(
            "You open a door into a Monster House on B4F with only one Reviver Seed left and teammates at half HP. "
            "Escape Orb is available. What is the safest plan?"
        ),
        expected_patterns=[r"(escape orb|flee|run)", r"(monster house|too many enemies)"],
        scoring_fn=lambda response: min(
            1.0,
            len(re.findall(r"(escape orb|flee|run)", response.lower())) * 0.6
            + len(re.findall(r"(monster house|too many enemies)", response.lower())) * 0.4,
        ),
    ),
    MicroTask(
        name="boss_seed_choice",
        description="Decide which seed to throw before a mini-boss encounter.",
        prompt=(
            "Before fighting Kecleon, you can throw a Sleep Seed or Blast Seed. "
            "Team HP is full but damage output is average. Which seed do you throw first?"
        ),
        expected_patterns=[r"(sleep seed)", r"(disable|sleep|control)"],
        scoring_fn=lambda response: min(
            1.0,
            len(re.findall(r"(sleep seed)", response.lower())) * 0.6
            + len(re.findall(r"(disable|sleep|control)", response.lower())) * 0.4,
        ),
    ),
]

# Combine all tasks
ALL_TASKS = MICRO_TASKS

def run_task(task: MicroTask, model_response: str) -> TaskResult:
    """Run a single micro-task and return scored result."""
    score = task.scoring_fn(model_response)

    # Additional metadata
    metadata = {
        "response_length": len(model_response),
        "matched_patterns": sum(1 for pattern in task.expected_patterns
                               if re.search(pattern, model_response, re.IGNORECASE)),
        "total_patterns": len(task.expected_patterns)
    }

    return TaskResult(score=score, metadata=metadata)


def get_task_suite() -> List[MicroTask]:
    """Get the complete micro-task suite."""
    return ALL_TASKS.copy()


def get_task_by_name(name: str) -> MicroTask:
    """Get a specific task by name."""
    for task in ALL_TASKS:
        if task.name == name:
            return task
    raise ValueError(f"Task '{name}' not found")


def run_task_suite(model_responses: Dict[str, str]) -> Dict[str, TaskResult]:
    """Run all tasks against model responses (task_name -> response)."""
    results = {}
    for task in ALL_TASKS:
        if task.name in model_responses:
            results[task.name] = run_task(task, model_responses[task.name])
    return results


def calculate_performance_avg(task_results: Dict[str, TaskResult]) -> float:
    """Calculate average performance across tasks."""
    if not task_results:
        return 0.0
    return sum(result.score for result in task_results.values()) / len(task_results)