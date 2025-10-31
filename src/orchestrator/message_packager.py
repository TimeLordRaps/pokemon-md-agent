"""
Message packager for orchestrator with model presets and three-message protocol.

Handles different model sizes (2B, 4B, 8B) with specific visual token budgets
and message structures. Implements pack(step_state, policy_hint) returning list[Message].
Consumes Copilot's {png,meta.json} format and supports multi-image packs with env_plus_grid + retrieved thumbnails.
Images are separate files, not composites.
"""

import logging
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Message structure for orchestrator protocol."""
    role: str
    text: str
    images: List[str]  # Paths to image files


@dataclass
class CopilotInput:
    """Copilot input structure with png and meta.json."""
    png_path: str
    meta_json_path: str
    retrieved_thumbnails: List[str]  # Additional thumbnail image paths


def _create_episodic_map_message(step_state: Dict[str, Any]) -> Message:
    """Create MSG[-2]: episodic_map with dynamic map and text event log."""
    text = ""
    images = []

    # Dynamic map
    if 'dynamic_map' in step_state:
        text += "EPISODIC_MAP: Dynamic exploration map\n"
        images.append(step_state['dynamic_map'])

    # Text event log
    if 'event_log' in step_state:
        events = step_state['event_log'][-10:]  # Last 10 events
        text += f"Recent events ({len(events)}):\n"
        for event in events:
            text += f"- {event}\n"

    return Message(role="user", text=text, images=images)


def _create_retrieval_message(step_state: Dict[str, Any]) -> Message:
    """Create MSG[-1]: retrieval with short trajectories and summaries."""
    text = "RETRIEVAL: Short trajectories with summaries\n"
    images = []

    if 'retrieved_trajs' in step_state:
        trajs = step_state['retrieved_trajs'][:3]  # Max 3 trajectories
        for i, traj in enumerate(trajs):
            if 'summary' in traj:
                text += f"Trajectory {i+1}: {traj['summary']}\n"
            if 'frames' in traj:
                # Short trajectories: 4-8 frames each
                frames = traj['frames'][:8]
                images.extend(frames)

    return Message(role="assistant", text=text, images=images)


def _create_now_message(step_state: Dict[str, Any], policy_hint: str) -> Message:
    """Create MSG[0]: now with current env+grid and action request."""
    text = f"NOW: Current environment state\nPolicy hint: {policy_hint}\n"
    images = []

    # Current env+grid at default 480×320
    if 'now' in step_state:
        now_data = step_state['now']
        if 'env' in now_data:
            text += "Environment view @480×320\n"
            images.append(now_data['env'])
        if 'grid' in now_data:
            text += "Grid overlay @480×320\n"
            images.append(now_data['grid'])

    text += "\nPlease provide the next action."

    return Message(role="user", text=text, images=images)


def _apply_budget_constraints(messages: List[Message], max_images: int) -> List[Message]:
    """Apply budget constraints by truncating images across messages."""
    total_images = sum(len(msg.images) for msg in messages)
    if total_images <= max_images:
        return messages

    # Truncate from least important to most important (reverse order)
    remaining_budget = max_images
    constrained_messages = []

    for msg in reversed(messages):  # Start from MSG[0] (now) as most important
        if len(msg.images) <= remaining_budget:
            constrained_messages.append(msg)
            remaining_budget -= len(msg.images)
        else:
            # Truncate images in this message
            truncated_images = msg.images[:remaining_budget]
            constrained_msg = Message(
                role=msg.role,
                text=msg.text + f"\n[Truncated {len(msg.images) - len(truncated_images)} images due to budget]",
                images=truncated_images
            )
            constrained_messages.append(constrained_msg)
            remaining_budget = 0

    return list(reversed(constrained_messages))


def _validate_image_dimensions(image_paths: List[str]) -> None:
    """
    Validate that all images are exactly 480×320 pixels. Reject any images that are not this size.

    Args:
        image_paths: List of paths to image files to validate.

    Raises:
        ValueError: If any image is not exactly 480×320 pixels.
    """
    required_size = (480, 320)
    for path in image_paths:
        if not path or not Path(path).exists():
            logger.warning("Skipping validation for non-existent or empty path: %s", path)
            continue
        try:
            with Image.open(path) as img:
                if img.size == required_size:
                    logger.info("Image %s validated at %s", path, required_size)
                else:
                    raise ValueError("Image %s has size %s, required exactly %s." % (path, img.size, required_size))
        except ValueError:
            raise  # Re-raise for size mismatch
        except Exception as e:
            logger.warning("Failed to validate image %s: %s. Skipping.", path, e)


# Model presets with visual token budgets
MODEL_PRESETS: Dict[str, Dict[str, int]] = {
    '2B': {
        'visual_budget': 4000,  # Total visual tokens
        'max_images': 20,  # Allow episodic_map + retrieval + now
        'tokens_per_image': 85,  # Estimated tokens per 480×320 image
    },
    '4B': {
        'visual_budget': 2500,  # Total visual tokens
        'max_images': 10,  # Balanced
        'tokens_per_image': 85,
    },
    '8B': {
        'visual_budget': 600,  # Total visual tokens
        'max_images': 2,  # Context-efficient, usually NOW only
        'tokens_per_image': 85,
    },
}

def parse_copilot_input(copilot_input: CopilotInput) -> Dict[str, Any]:
    """
    Parse Copilot's {png,meta.json} input into step_state format.

    Args:
        copilot_input: CopilotInput with png path, meta.json path, and retrieved thumbnails.

    Returns:
        step_state dictionary compatible with existing pack() function.

    Raises:
        FileNotFoundError: If png or meta.json files don't exist.
        ValueError: If meta.json is malformed.
    """
    if not os.path.exists(copilot_input.png_path):
        raise FileNotFoundError(f"Copilot PNG not found: {copilot_input.png_path}")
    if not os.path.exists(copilot_input.meta_json_path):
        raise FileNotFoundError(f"Copilot meta.json not found: {copilot_input.meta_json_path}")

    # Load meta.json
    with open(copilot_input.meta_json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    # Build step_state from meta.json structure
    step_state = {
        'dynamic_map': meta.get('dynamic_map'),
        'event_log': meta.get('event_log', []),
        'retrieved_trajs': meta.get('retrieved_trajectories', []),
        'now': {
            'env': copilot_input.png_path,  # Main env image
            'grid': meta.get('grid_overlay'),  # Grid overlay if present
        },
        'retrieved_thumbnails': copilot_input.retrieved_thumbnails,  # Additional thumbnails
    }

    return step_state


def _create_now_message_with_thumbnails(step_state: Dict[str, Any], policy_hint: str) -> Message:
    """
    Create MSG[0]: now with current env+grid + retrieved thumbnails + action request.

    Supports multi-image packs: env_plus_grid (current) + retrieved thumbnails.
    """
    text = f"NOW: Current environment state\nPolicy hint: {policy_hint}\n"
    images = []

    # Current env+grid at default 480×320
    if 'now' in step_state:
        now_data = step_state['now']
        if 'env' in now_data:
            text += "Environment view @480×320\n"
            images.append(now_data['env'])
        if 'grid' in now_data:
            text += "Grid overlay @480×320\n"
            images.append(now_data['grid'])

    # Add retrieved thumbnails for multi-image pack
    if 'retrieved_thumbnails' in step_state:
        thumbnails = step_state['retrieved_thumbnails'][:5]  # Limit to 5 thumbnails
        if thumbnails:
            text += f"Retrieved thumbnails ({len(thumbnails)}):\n"
            images.extend(thumbnails)

    text += "\nPlease provide the next action."

    return Message(role="user", text=text, images=images)


def pack_from_copilot(copilot_input: CopilotInput, policy_hint: str, model_size: str = '4B') -> List[Message]:
    """
    Package Copilot input into three-message protocol for LLM.

    Args:
        copilot_input: CopilotInput with png, meta.json, and retrieved thumbnails.
        policy_hint: Action policy hint (e.g., "explore", "fight").
        model_size: Model size key ('2B', '4B', '8B').

    Returns:
        List of three Message objects (MSG[-2], MSG[-1], MSG[0]).

    Raises:
        ValueError: If model_size invalid or budget exceeded.
        FileNotFoundError: If input files don't exist.
    """
    step_state = parse_copilot_input(copilot_input)
    return pack(step_state, policy_hint, model_size)


def pack(step_state: Dict[str, Any], policy_hint: str, model_size: str = '4B') -> List[Message]:
    """
    Package step state into three-message protocol for LLM.

    Args:
        step_state: Current game state with image paths.
        policy_hint: Action policy hint (e.g., "explore", "fight").
        model_size: Model size key ('2B', '4B', '8B').

    Returns:
        List of three Message objects (MSG[-2], MSG[-1], MSG[0]).

    Raises:
        ValueError: If model_size invalid or budget exceeded.
    """
    if model_size not in MODEL_PRESETS:
        raise ValueError(f"Invalid model_size: {model_size}. Must be '2B', '4B', or '8B'.")

    preset = MODEL_PRESETS[model_size]
    logger.info("Packing messages for %s model with budget %d", model_size, preset['visual_budget'])

    # MSG[-2]: episodic_map (dynamic map + text event log)
    episodic_map_msg = _create_episodic_map_message(step_state)

    # MSG[-1]: retrieval (short trajectories + summaries)
    retrieval_msg = _create_retrieval_message(step_state)

    # MSG[0]: now (current env+grid + retrieved thumbnails + action request)
    now_msg = _create_now_message_with_thumbnails(step_state, policy_hint)

    messages = [episodic_map_msg, retrieval_msg, now_msg]

    # Validate all images are 480×320 (only for existing files)
    all_image_paths = []
    for msg in messages:
        all_image_paths.extend([path for path in msg.images if path and os.path.exists(path)])
    if all_image_paths:
        _validate_image_dimensions(all_image_paths)

    # Check total budget across all messages
    total_images = sum(len(msg.images) for msg in messages)
    total_tokens = total_images * preset['tokens_per_image']

    if total_images > preset['max_images'] or total_tokens > preset['visual_budget']:
        logger.warning("Budget exceeded: %d images (%d tokens) > %d images (%d tokens), truncating",
                      total_images, total_tokens, preset['max_images'], preset['visual_budget'])
        messages = _apply_budget_constraints(messages, preset['max_images'])

    logger.info("Packed messages: episodic_map=%d images, retrieval=%d images, now=%d images",
                len(episodic_map_msg.images), len(retrieval_msg.images), len(now_msg.images))
    return messages