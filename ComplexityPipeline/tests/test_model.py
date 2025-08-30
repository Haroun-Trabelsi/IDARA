import tempfile

import torch
import numpy as np

from train_model import MultimodalRNN


def _dummy_config():
    """Return a minimal, self-consistent model + data config."""
    return {
        "model": {
            "vis_feat_dim": 4,
            "static_feat_dim": 2,
            "rnn_hidden_dim": 8,
            "dense_hidden_dim": 16,
            "output_dim": 3,
            "sequence_length": 5,
            "rnn_type": "lstm",
            "num_layers": 1,
            "bidirectional": False,
            "dropout_rate": 0.0,
            "batch_norm": False,
            "activation": "relu",
        },
        "data": {
            "sequence_length": 5,
            "static_features": ["f1", "f2"],
            "class_names": ["easy", "medium", "hard"],
        },
        "training": {},
        "feature_extractor": {"output_dim": 4},
    }


def test_forward_pass():
    cfg = _dummy_config()
    model = MultimodalRNN(cfg)

    batch_size = 2
    seq_len = cfg["model"]["sequence_length"]
    vis_dim = cfg["model"]["vis_feat_dim"]
    static_dim = cfg["model"]["static_feat_dim"]

    vis_seq = torch.randn(batch_size, seq_len, vis_dim)
    static_scores = torch.randn(batch_size, static_dim)

    with torch.no_grad():
        outputs = model(vis_seq, static_scores)

    assert "prediction" in outputs, "Model output must contain 'prediction' key"
    assert outputs["prediction"].shape == (
        batch_size,
        cfg["model"]["output_dim"],
    ), "Prediction tensor has wrong shape"


def test_save_and_load_checkpoint(tmp_path):
    cfg = _dummy_config()
    model = MultimodalRNN(cfg)

    ckpt_file = tmp_path / "model.ckpt"

    # Save
    model.save_checkpoint(str(ckpt_file))
    assert ckpt_file.exists(), "Checkpoint file not created"

    # Load
    loaded_model, checkpoint = MultimodalRNN.load_from_checkpoint(str(ckpt_file), map_location="cpu")
    assert loaded_model is not None, "Failed to load model from checkpoint"
    assert checkpoint is not None and "config" in checkpoint, "Checkpoint missing config"

    # Sanity-run inference (eval mode avoids BatchNorm training constraints)
    loaded_model.eval()
    vis_seq = torch.randn(2, cfg["model"]["sequence_length"], cfg["model"]["vis_feat_dim"])
    static_scores = torch.randn(2, cfg["model"]["static_feat_dim"])
    with torch.no_grad():
        preds = loaded_model(vis_seq, static_scores)["prediction"]
    assert preds.shape == (2, cfg["model"]["output_dim"])
