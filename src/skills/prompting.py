"""Prompt scaffolds and retrieval utilities for skill authoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

from .spec import SKILL_SCHEMA, SkillSpec, SkillMeta

BASE_SYSTEM_PROMPT = """You are the PMD skills architect.
- Skills are authored as JSON objects that follow the provided schema.
- Use only the listed primitives. Compose behaviour using loops, if-blocks,
  and skill calls. Never invent new primitive names.
- Prefer short, safe plans. Add `annotate` steps when something noteworthy
  happens so the agent can self-critique partial failures.
- Always set partial success notes describing common near-miss outcomes.
"""


def build_guidance_schema() -> str:
    """Return a JSON schema string usable by grammar-guided decoding."""

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "SkillSpec",
        **SKILL_SCHEMA,
    }
    return json.dumps(schema, indent=2)


def compose_system_prompt(additional_rules: Optional[str] = None) -> str:
    """Compose the final system prompt for LM skill generation."""
    if additional_rules:
        return BASE_SYSTEM_PROMPT + "\n" + additional_rules.strip() + "\n"
    return BASE_SYSTEM_PROMPT


def serialize_exemplars(skills: Iterable[SkillSpec]) -> str:
    """Serialize exemplar skills to include in the model context."""
    snippets: List[str] = []
    for skill in skills:
        payload = skill.dict()
        snippets.append(json.dumps(payload, indent=2))
    return "\n\n".join(snippets)


def build_skill_header(meta: SkillMeta) -> str:
    """Small helper summarising metadata for prompt conditioning."""
    tags = ", ".join(meta.tags) if meta.tags else "none"
    return f"Skill `{meta.name}` — tags: {tags} — expects: {meta.expects or 'unspecified'}"


def format_retrieval_context(
    exemplars: Iterable[SkillSpec],
    telemetry_summaries: Iterable[Dict[str, Any]],
) -> str:
    """Combine retrieved skills and telemetry hints for the LM context."""
    exemplar_blob = serialize_exemplars(exemplars)
    telemetry_blob = "\n".join(
        f"- {item.get('summary', 'run')} :: {item.get('notes', [])}"
        for item in telemetry_summaries
    )
    return f"/* Retrieved skill exemplars */\n{exemplar_blob}\n\n/* Telemetry */\n{telemetry_blob}\n"


def load_skill_library(path: Path) -> List[SkillSpec]:
    """Load all JSON skill specs from a directory for retrieval."""
    specs: List[SkillSpec] = []
    for json_file in sorted(path.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
            specs.append(SkillSpec.parse_obj(data))
        except Exception as exc:  # pylint: disable=broad-except
            # Skip malformed entries but keep extra context for debugging.
            print(f"[skills] failed to load {json_file}: {exc}")
    return specs
