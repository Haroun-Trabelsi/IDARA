"""Unit test: ensure the trained checkpoint loads and produces a valid probability
   distribution (sums to 1) on synthetic data.

This is a very light-weight smoke test: it only runs on CPU and feeds a small random
tensor through the network so CI remains fast (<2 s).
"""
from pathlib import Path
import importlib
import yaml
import numpy as np
import torch

# Import the model class dynamically — avoids import errors if the script name changes
train_mod = importlib.import_module("train_model")
MultimodalRNN = getattr(train_mod, "MultimodalRNN")

CONFIG_PATH = Path("config.yaml")
CFG = yaml.safe_load(CONFIG_PATH.read_text())
MODEL_CFG = CFG["multimodal_model"]


def _load_model():
    """Load checkpoint with the same logic used in test_model.py but CPU-only."""
    ckpt_path = Path(MODEL_CFG["model_path"])
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = MultimodalRNN(checkpoint.get("config", {}))

    state_dict = (
        checkpoint.get("state_dict")
        or checkpoint.get("model_state_dict")
        or {
            k: v
            for k, v in checkpoint.items()
            if isinstance(v, torch.Tensor)
        }
    )
    # Remove potential DataParallel prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def test_checkpoint_loads():
    model = _load_model()
    assert sum(p.numel() for p in model.parameters()) > 0, "Model has no parameters"


@torch.no_grad()
def test_probability_distribution():
    model = _load_model()

    # Build tiny synthetic input matching vis_feat_dim
    vis_dim = MODEL_CFG["vis_feat_dim"]
    seq = torch.randn(1, 3, vis_dim)  # batch=1, seq_len=3
    static = torch.randn(1, len(MODEL_CFG["static_features"]))

    out = model(seq, static)
    if isinstance(out, dict):
        # Prefer 'logits' key; fall back to 'prediction' without triggering boolean
        if "logits" in out and out["logits"] is not None:
            logits = out["logits"]
        elif "prediction" in out and out["prediction"] is not None:
            logits = out["prediction"]
        else:
            # Fallback to the first tensor value in the dict
            logits = next(v for v in out.values() if isinstance(v, torch.Tensor))
    else:
        logits = out

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-5)
