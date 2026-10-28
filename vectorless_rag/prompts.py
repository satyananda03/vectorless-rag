"""Prompt YAML loading: each prompt file lives next to the stage that uses it."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=16)
def load_prompt_file(file_path: Path) -> dict[str, Any]:
    """Parse a prompt YAML file. Cached — prompt files do not change at runtime."""
    try:
        with open(file_path, encoding="utf-8") as f:
            prompts = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Prompt file not found: {file_path}") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing prompt YAML: {file_path}") from e
    if not isinstance(prompts, dict):
        raise ValueError(f"Prompt file must contain a mapping: {file_path}")
    return prompts


def load_prompt(target_prompt: str, file_path: Path) -> dict[str, str]:
    """Return one prompt's raw ``system_prompt`` / ``user_prompt`` strings.

    Values come back verbatim so the caller can ``str.format`` them with its own
    context. A missing ``user_prompt`` is returned as an empty string.
    """
    cfg = load_prompt_file(Path(file_path)).get(target_prompt)
    if cfg is None:
        raise ValueError(f"Prompt template {target_prompt} not found in {file_path}")
    return {
        "system_prompt": cfg.get("system_prompt", ""),
        "user_prompt": cfg.get("user_prompt", ""),
    }

