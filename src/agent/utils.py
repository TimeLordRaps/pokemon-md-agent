"""Utility functions for the PMD agent."""

import os
from typing import Optional


def sanitize_hf_home() -> Optional[str]:
    """Sanitize HF_HOME environment variable.

    Handles Windows paths with surrounding quotes or escape characters.
    Returns None if HF_HOME is not set.

    Returns:
        Sanitized HF_HOME path with quotes stripped, or None if not set
    """
    hf_home_raw = os.environ.get('HF_HOME', '')
    if not hf_home_raw:
        return None

    # Strip surrounding quotes (single or double)
    hf_home = hf_home_raw.strip().strip('"').strip("'")

    # Expand user path if contains ~
    hf_home = os.path.expanduser(hf_home)

    # Normalize path separators for Windows
    hf_home = os.path.normpath(hf_home)

    return hf_home


def get_hf_cache_dir() -> Optional[str]:
    """Get the HuggingFace cache directory with hub subdirectory.

    Returns sanitized HF_HOME with 'hub' subdirectory appended.
    Returns None if HF_HOME is not set.

    Returns:
        Path to HF cache directory (HF_HOME/hub), or None if not set
    """
    hf_home = sanitize_hf_home()
    if hf_home:
        return os.path.join(hf_home, 'hub')
    return None