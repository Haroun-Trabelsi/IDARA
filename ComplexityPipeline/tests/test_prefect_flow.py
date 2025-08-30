"""Integration test: run the Prefect flow on a dummy video file.

We simulate a 3-frame video by creating a temporary .npy feature file directly,
so the heavy CV extraction step is skipped. MongoDB interactions are mocked so
no DB is required during CI.
"""
from pathlib import Path
import asyncio
import numpy as np
import pytest
from unittest import mock

from pipeline.orchestrater import process_video_flow  # Prefect flow definition
from pipeline.utils import ConfigManager, PipelineContext

CFG = ConfigManager("config.yaml")
FEATURE_DIR = Path(CFG.get("multimodal_model.features_dir"))


@pytest.mark.asyncio
async def test_prefect_flow_runs(tmp_path):
    """Run the Prefect flow end-to-end on synthetic data in <300 ms."""
    # 1. Prepare dummy video file & synthetic feature tensor (3×2048)
    video_path = tmp_path / "dummy.mp4"
    video_path.write_bytes(b"fake")

    feat_path = FEATURE_DIR / f"{video_path.stem}.npy"
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feat_path, np.random.randn(3, CFG.get("multimodal_model.vis_feat_dim")))

    # 2. Mock Mongo insert to avoid real DB connection
    with mock.patch("pipeline.utils.DatabaseManager.insert_result", new_callable=mock.AsyncMock):
        ctx = PipelineContext(video_path)
        # 3. Run flow (no retries) – returns final context
        result_context = await process_video_flow(ctx, CFG)

    assert result_context.final_result, "Flow did not set final_result"
    assert result_context.final_result["predicted_class"] in CFG.get("multimodal_model.class_names"), "Invalid class name"
