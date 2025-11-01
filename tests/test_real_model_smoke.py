import os
import subprocess
import pytest


@pytest.mark.real_model
def test_real_models_list_and_token_present():
    """Smoke test to list HF models and check HF token presence.

    Marked `real_model` so it is excluded by default. Run manually after
    setting MODEL_BACKEND=hf and HF_TOKEN in your environment.
    """
    backend = os.environ.get("MODEL_BACKEND", "").lower()
    if backend != "hf":
        pytest.skip("MODEL_BACKEND!=hf; skipping real model smoke test")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN not set; skipping real model smoke test")

    # Call the provided real_loader --list to verify model list is readable
    runner = ["python", "-m", "src.models.real_loader", "--list"]
    try:
        out = subprocess.check_output(runner, stderr=subprocess.STDOUT, text=True, timeout=30)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"real_loader returned non-zero: {e.output}")
    except subprocess.TimeoutExpired:
        pytest.fail("real_loader --list timed out")

    assert "Qwen" in out or "unsloth" in out, "Expected HF model IDs to be listed"
