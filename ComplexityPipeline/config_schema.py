"""Pydantic configuration schema for the VFX pipeline.

This module defines strongly-typed models that validate the YAML configuration
loaded by training, benchmarking and service scripts.  Using a schema provides:
• Early, informative error messages (missing keys, wrong types, invalid values)
• Editor auto-completion & type hints
• Single source-of-truth for defaults and cross-field constraints

Pydantic v2 is used (declared in requirements.txt).  Validation is triggered via
`PipelineConfig.model_validate(raw_dict)` which returns a structured object that
can be accessed with dot-notation **or** converted back to a plain dict via
`model_dump()` for legacy code that still relies on subscripting.
"""

from __future__ import annotations
 
from typing import List, Literal, Optional
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator

# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------

class DataConfig(BaseModel):
    csv_path: Path = Field(..., description="CSV file containing metadata per shot.")
    sequence_features_dir: Path = Field(..., description="Directory with <shot_id>.npy files.")
    static_features: List[str] = Field(..., min_length=1, description="Names of scalar features.")
    target: str = Field(..., description="Column name for ground-truth label.")
    split_column: str = Field(..., description="Column designating train/val/test split.")
    identifier_col: str = Field(..., description="Column with unique shot identifier.")
    class_names: Optional[List[str]] = Field(None, description="Optional pretty names for classes.")


class ModelConfig(BaseModel):
    vis_feat_dim: int = Field(..., gt=0)
    static_feat_dim: int = Field(..., gt=0)
    rnn_hidden_dim: int = Field(..., gt=0)
    dense_hidden_dim: int = Field(..., gt=0)
    sequence_length: int = Field(..., gt=0)
    rnn_type: Literal["lstm", "gru"] = "lstm"
    num_layers: int = Field(1, ge=1)
    bidirectional: bool = True
    dropout_rate: float = Field(0.3, ge=0.0, le=1.0)
    use_gating: bool = True
    dynamic_weighting: bool = False
    output_dim: int = Field(..., gt=1)
    attention: bool = False


class TrainingConfig(BaseModel):
    optimizer: Literal["adamw", "sgd"] = "adamw"
    learning_rate: float = Field(3e-4, gt=0.0)
    batch_size: int = Field(32, gt=0)
    num_epochs: int = Field(10, gt=0)
    grad_accumulation_steps: int = 1
    early_stopping_patience: int = 8
    early_stopping_monitor: str = "val_loss"
    distributed: bool = False
    use_mixed_precision: bool = True
    output_dir: Path = Path("outputs")
    experiment_name: str = Field("experiment")
    best_model_name: str = Field("best.pt")
    random_seed: int = 42


class FeatureExtractorConfig(BaseModel):
    name: str = Field(...)
    output_dim: int = Field(..., gt=0)


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080


# ---------------------------------------------------------------------------
# Root schema with cross-section checks
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    feature_extractor: FeatureExtractorConfig

    # optional sections
    api: Optional[APIConfig] = None
    complexity_model_paths: Optional[dict] = None

    # --- Cross-field validation ------------------------------------------------

    @model_validator(mode="after")
    def _check_consistency(self) -> "PipelineConfig":
        if self.model.static_feat_dim != len(self.data.static_features):
            raise ValueError("model.static_feat_dim must equal len(data.static_features)")
        if self.model.vis_feat_dim != self.feature_extractor.output_dim:
            raise ValueError("model.vis_feat_dim must equal feature_extractor.output_dim")
        return self


# Convenience alias (used by train_model.py)
ConfigSchema = PipelineConfig

__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "FeatureExtractorConfig",
    "APIConfig",
    "PipelineConfig",
    "ConfigSchema",
]
