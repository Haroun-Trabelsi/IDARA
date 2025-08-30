import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                            confusion_matrix, classification_report, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # Make sure this is imported
from sklearn.model_selection import train_test_split  # Stratified splitting
from tqdm import tqdm
import joblib
import json
import pickle
import time
from datetime import datetime
import sys
import importlib.util
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import argparse
import yaml
from config_schema import PipelineConfig
import random
# Ensure deterministic behavior across runs
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import matplotlib.ticker as mtick
from contextlib import contextmanager
import re
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.profiler
from torchvision.ops import sigmoid_focal_loss

# --- Basic Setup ---
CONFIG: Optional[Dict] = None
DEVICE: Optional[torch.device] = None
SHAP_AVAILABLE = importlib.util.find_spec("shap") is not None

if SHAP_AVAILABLE:
    import shap
else:
    shap = None

# --- Logging Setup ---
def setup_logging(output_dir, experiment_name):
    log_dir = Path(output_dir) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]: 
        root_logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(f"Trainer_{experiment_name}")

logger = logging.getLogger("TrainerInit")

# --- Set Random Seeds ---
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

# --- Context Manager ---
@contextmanager
def timer(task_name: str, logger_instance):
    logger_instance.info(f"Starting: {task_name}...")
    start = datetime.now()
    yield
    end = datetime.now()
    logger_instance.info(f"{task_name} completed in {end - start}")

# --- Activation Helper ---
def get_activation_fn(activation: str) -> nn.Module:
    act_map = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "selu": nn.SELU(),
        "tanh": nn.Tanh()
    }
    act_name = activation.lower()
    if act_name not in act_map:
        raise ValueError(f"Unsupported activation: {activation}")
    return act_map[act_name]

# --- Model Component Classes ---
class DynamicWeightGenerator(nn.Module):
    def __init__(self, static_feat_dim: int, feature_categories: List[str], hidden_dim: int = 32,
                 activation: str = "gelu", num_layers: int=2, dropout_rate: float = 0.1):
        super().__init__()
        self.static_feat_dim = static_feat_dim
        self.feature_categories = feature_categories
        assert len(self.feature_categories) == static_feat_dim, "Categories must match static_feat_dim"
        self.activation = get_activation_fn(activation)
        layers = []
        in_dim = static_feat_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(self.activation)
        layers.append(nn.Dropout(dropout_rate))
        in_dim = hidden_dim
        for _ in range(max(0, num_layers - 2)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(in_dim, static_feat_dim))
        layers.append(nn.Softmax(dim=-1))
        self.weight_network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nonlin = 'leaky_relu' if isinstance(self.activation, (nn.LeakyReLU, nn.GELU, nn.ELU, nn.SELU)) else 'relu'
                try:
                    nn.init.kaiming_normal_(m.weight, nonlinearity=nonlin)
                except ValueError:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, static_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        weights = self.weight_network(static_features)
        feature_weights = {category: weights[..., i] for i, category in enumerate(self.feature_categories)}
        return {"weights": weights, "feature_weights": feature_weights}

class AttentionModule(nn.Module):
    def __init__(self, input_dim: int, dropout_rate: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.input_dim = input_dim
        self.activation = get_activation_fn(activation)
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, max(1, input_dim // 2)),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(max(1, input_dim // 2), 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        energy = self.attention_layer(x)
        attn_weights = F.softmax(energy, dim=1)
        context_pooled = torch.sum(attn_weights * x, dim=1)
        return context_pooled, attn_weights.squeeze(-1)

class ResidualMLP(nn.Module):
    def __init__(self, layers: nn.ModuleList, use_residual: bool = True, input_dim: int = 0, output_dim: int = 0):
        super().__init__()
        self.mlp_layers = layers
        self.use_residual = use_residual and (input_dim == output_dim)
        if self.use_residual and input_dim != output_dim:
            logger.warning(f"Residual connection disabled due to dimension mismatch: input {input_dim} ≠ output {output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = x
        for layer in self.mlp_layers:
            out = layer(out)
        if self.use_residual:
            out = out + identity
        return out

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, rnn_type: str = 'lstm',
                 bidirectional: bool = False, dropout_rate: float = 0.0, attention: bool = False):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.attention = attention
        self.hidden_dim = hidden_dim
        rnn_class = nn.LSTM if rnn_type == 'lstm' else nn.GRU if rnn_type == 'gru' else None
        if rnn_class is None:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        rnn_dropout = dropout_rate if num_layers > 1 else 0.0
        self.rnn = rnn_class(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=rnn_dropout, bidirectional=bidirectional)
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        if attention:
            self.attention_module = AttentionModule(self.output_dim, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        if lengths is not None:
            sorted_lengths, sort_idx = torch.sort(lengths, descending=True)
            x_sorted = x[sort_idx]
            packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True
            ).to(x_sorted.device)
            packed_out, _ = self.rnn(packed)
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=x.size(1)
            )
            _, unsort_idx = torch.sort(sort_idx)
            rnn_out = rnn_out[unsort_idx]
        else:
            rnn_out, _ = self.rnn(x)
            
        if self.attention and hasattr(self, 'attention_module'):
            context, attn_weights = self.attention_module(rnn_out)
            result = {"output": context}
            if return_attention:
                result["attention_weights"] = attn_weights
        else:
            if self.bidirectional:
                last_forward = rnn_out[:, -1, :self.hidden_dim]
                first_backward = rnn_out[:, 0, self.hidden_dim:]
                context = torch.cat((last_forward, first_backward), dim=1)
            else:
                context = rnn_out[:, -1, :]
            result = {"output": context}
        return result

class StaticFeatureProcessor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 dropout_rate: float = 0.3, batch_norm: bool = True,
                 activation: str = "gelu", residual: bool = False):
        super().__init__()
        self.activation = get_activation_fn(activation)
        self.mlp = self._build_mlp(input_dim, hidden_dims, output_dim, 
                                 dropout_rate, batch_norm, self.activation, residual)

    def _build_mlp(self, input_dim, hidden_dims, output_dim, dropout_rate, batch_norm, activation, residual):
        layers = []
        current_dim = input_dim
        all_dims = hidden_dims + [output_dim]
        num_layers = len(all_dims)

        for i, h_dim in enumerate(all_dims):
            is_last = (i == num_layers - 1)
            
            block_layers = nn.ModuleList()
            block_layers.append(nn.Linear(current_dim, h_dim))

            if not is_last:
                if batch_norm:
                    block_layers.append(nn.BatchNorm1d(h_dim))
                block_layers.append(activation)
                if dropout_rate > 0:
                    block_layers.append(nn.Dropout(dropout_rate))
                
                can_use_residual = residual and current_dim == h_dim
                if can_use_residual:
                    residual_block = ResidualMLP(
                        block_layers, 
                        use_residual=True,
                        input_dim=current_dim,
                        output_dim=h_dim
                    )
                    layers.append(residual_block)
                else:
                    seq_block = nn.Sequential(*block_layers)
                    layers.append(seq_block)
            else:
                seq_block = nn.Sequential(*block_layers)
                layers.append(seq_block)
            
            current_dim = h_dim
            
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class ModalityFusion(nn.Module):
    def __init__(self, input_dims: List[int], use_gating: bool = True):
        super().__init__()
        self.input_dims = input_dims
        self.use_gating = use_gating
        self.total_input_dim = sum(input_dims)
        if self.use_gating:
            self.gate = nn.Sequential(
                nn.Linear(self.total_input_dim, self.total_input_dim),
                nn.Sigmoid()
            )

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        concatenated = torch.cat(features, dim=1)
        result = {"combined_unprocessed": concatenated}
        if self.use_gating and hasattr(self, 'gate'):
            gate_values = self.gate(concatenated)
            fused = gate_values * concatenated
            result["fused_features"] = fused
            result["gate_values"] = gate_values
            contribs = {}
            start_idx = 0
            for i, dim in enumerate(self.input_dims):
                contribs[f"modality_{i}"] = torch.mean(gate_values[:, start_idx : start_idx + dim], dim=1)
                start_idx += dim
            result["modality_contributions"] = contribs
        else:
            result["fused_features"] = concatenated
        return result 

class MultimodalRNN(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        model_cfg = config.get('model', {})
        data_cfg = config.get('data', {})

        vis_feat_dim = model_cfg.get('vis_feat_dim', 512)
        static_feat_dim = model_cfg.get('static_feat_dim', 3)
        rnn_hidden_dim = model_cfg.get('rnn_hidden_dim', 128)
        dense_hidden_dim = model_cfg.get('dense_hidden_dim', 64)
        dropout_rate = model_cfg.get('dropout_rate', 0.3)
        rnn_type = model_cfg.get('rnn_type', 'lstm')
        num_layers = model_cfg.get('num_layers', 1)
        bidirectional = model_cfg.get('bidirectional', True)
        activation = model_cfg.get('activation', 'gelu')
        output_dim = model_cfg.get('output_dim', 2)
        attention = model_cfg.get('attention', True)
        batch_norm = model_cfg.get('batch_norm', True)
        self.feature_categories = data_cfg.get('static_features', ['feature1', 'feature2', 'feature3'])
        dynamic_weighting = model_cfg.get('dynamic_weighting', True)
        use_gating = model_cfg.get('use_gating', True)
        residual_connections = model_cfg.get('residual_connections', False)

        self.activation_fn = get_activation_fn(activation)
        self.dynamic_weighting = dynamic_weighting

        self.temporal_encoder = TemporalEncoder(
            input_dim=vis_feat_dim,
            hidden_dim=rnn_hidden_dim,
            num_layers=num_layers,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            dropout_rate=dropout_rate,
            attention=attention
        )
        self.rnn_output_dim = self.temporal_encoder.output_dim

        if self.dynamic_weighting:
            self.weight_generator = DynamicWeightGenerator(
                static_feat_dim, self.feature_categories, dense_hidden_dim // 2, 
                activation, num_layers=2, dropout_rate=0.1
            )

        static_processor_hidden_dims = [dense_hidden_dim]
        self.static_processor = StaticFeatureProcessor(
            input_dim=static_feat_dim,
            hidden_dims=static_processor_hidden_dims,
            output_dim=dense_hidden_dim // 2,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            activation=activation,
            residual=residual_connections
        )
        self.static_output_dim = dense_hidden_dim // 2

        self.fusion = ModalityFusion(
            input_dims=[self.rnn_output_dim, self.static_output_dim],
            use_gating=use_gating
        )
        fusion_output_dim = self.fusion.total_input_dim

        pred_head_hidden_dims = [dense_hidden_dim]
        self.prediction_head = StaticFeatureProcessor(
            input_dim=fusion_output_dim,
            hidden_dims=pred_head_hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            activation=activation,
            residual=residual_connections
        )

        self._init_weights()

    def _init_weights(self):
        logger.debug("Initializing MultimodalRNN weights...")
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nonlin = 'leaky_relu' if isinstance(self.activation_fn, (nn.LeakyReLU, nn.GELU, nn.ELU, nn.SELU)) else 'relu'
                try:
                    nn.init.kaiming_normal_(module.weight, nonlinearity=nonlin)
                except ValueError:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                for param in module.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.constant_(param.data, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, vis_seq: torch.Tensor, static_scores: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None,
                return_attention: bool = False, return_interpretability: bool = False
               ) -> Dict[str, torch.Tensor]:
        vis_seq, static_scores = vis_seq.float(), static_scores.float()
        # Optional quantization path
        if hasattr(self, 'quant'):
            vis_seq = self.quant(vis_seq)
            static_scores = self.quant(static_scores)
        result_dict = {}
        if return_interpretability:
            result_dict["original_static_scores"] = static_scores

        if self.dynamic_weighting and hasattr(self, 'weight_generator'):
            weight_info = self.weight_generator(static_scores)
            feature_weights = weight_info["weights"]
            weighted_static = static_scores * feature_weights
            if return_interpretability:
                result_dict["feature_weights_info"] = weight_info["feature_weights"]
        else:
            weighted_static = static_scores

        static_features = self.static_processor(weighted_static)

        temporal_result = self.temporal_encoder(vis_seq, lengths, return_attention=return_attention)
        temporal_features = temporal_result["output"]
        if return_attention and "attention_weights" in temporal_result:
            result_dict["attention_weights"] = temporal_result["attention_weights"]

        fusion_result = self.fusion([temporal_features, static_features])
        fused_features = fusion_result["fused_features"]
        if return_interpretability and 'modality_contributions' in fusion_result:
            result_dict["modality_contribution"] = fusion_result["modality_contributions"]

        logits = self.prediction_head(fused_features)
        if hasattr(self, 'dequant'):
            logits = self.dequant(logits)
        probabilities = F.softmax(logits, dim=-1)

        clean_result = {"prediction": probabilities, "logits": logits}
        if return_interpretability:
            clean_result.update(result_dict)
        return clean_result

    def get_config(self):
        return self.config

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path, optimizer=None, scheduler=None, epoch=None, loss=None, metrics=None, quantize=None):
        config_to_save = self.config.copy()
        checkpoint = {
            'config': config_to_save,
            'state_dict': self.state_dict()
        }
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
        if metrics:
            checkpoint['metrics'] = metrics
            
        if quantize is None:
            quantize = self.config['training'].get('quantize', False)
        if quantize:
            logger.warning("Quantization during training checkpoints is not recommended; skipping.")
            
        try:
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, path_obj)
            logger.debug(f"Checkpoint saved: {path_obj}")
        except Exception as e:
            logger.error(f"Failed save checkpoint {path}: {e}")

    @classmethod
    def load_from_checkpoint(cls, path, map_location=None):
        if not os.path.exists(path):
            logger.error(f"Checkpoint not found: {path}")
            return None, None
        device = map_location or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            config = checkpoint.get('config')
            if config is None:
                raise KeyError("Model config missing.")
            model = cls(config)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info(f"Model loaded from {path}")
            return model, checkpoint
        except Exception as e:
            logger.error(f"Failed load checkpoint {path}: {e}", exc_info=True)
            return None, None

class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0, path='best_model.pt', monitor='loss', mode='min'): # Changed default monitor
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_best = float('inf') if mode == 'min' else float('-inf')
        self.monitor = monitor
        self.best_epoch = -1
        self.mode = mode

        if self.mode not in ['min', 'max']:
            raise ValueError(f"EarlyStopping mode '{self.mode}' is unknown, choose 'min' or 'max'.")
        if self.mode == 'max':
            self.delta *= -1

    def __call__(self, metrics: Dict, model: nn.Module, optimizer=None, scheduler=None, epoch=None):
        current_metric_val = metrics.get(self.monitor)
        if current_metric_val is None:
            logger.warning(f"EarlyStopping monitor '{self.monitor}' not found in metrics. Available keys: {list(metrics.keys())}")
            return

        score = current_metric_val
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self._save_checkpoint(current_metric_val, model, optimizer, scheduler, epoch, metrics)
        elif (self.mode == 'min' and score > self.best_score - self.delta) or \
             (self.mode == 'max' and score < self.best_score + self.delta):
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self._save_checkpoint(current_metric_val, model, optimizer, scheduler, epoch, metrics)
            self.counter = 0

    def _save_checkpoint(self, val_metric, model, optimizer, scheduler, epoch, metrics):
        if self.verbose:
            if self.mode == 'min':
                logger.info(f'Val metric ({self.monitor}) improved ({self.val_metric_best:.6f} --> {val_metric:.6f}). Saving best model...')
            else:
                logger.info(f'Val metric ({self.monitor}) improved ({self.val_metric_best*100:.2f}% --> {val_metric*100:.2f}%). Saving best model...')
        
        model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
        model_to_save.save_checkpoint(self.path, optimizer, scheduler, epoch, val_metric, metrics)
        self.val_metric_best = val_metric

class VFXDataset(Dataset):
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        sequence_features_dir: str,
        static_feature_cols: List[str],
        target_col: str,
        identifier_col: str,
        sequence_length: int,
        vis_feat_dim: int,
        # No scaler/imputation needed, data is pre-processed
        strict: bool = False,
        padding_strategy: str = "zeros",
        truncation_strategy: str = "last",
        augment: bool = False
    ):
        super().__init__()
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.sequence_features_dir = Path(sequence_features_dir)
        self.static_feature_cols = static_feature_cols
        self.target_col = target_col
        self.identifier_col = identifier_col
        self.sequence_length = sequence_length
        self.vis_feat_dim = vis_feat_dim
        self.strict = strict
        self.padding_strategy = padding_strategy.lower()
        self.truncation_strategy = truncation_strategy.lower()
        self.augment = augment
        
        # Static features are already scaled and imputed in main(), just select them
        self.static_features_processed = self.metadata_df[self.static_feature_cols].values.astype(np.float32)

        logger.info(f"Dataset initialized with {len(self.metadata_df)} samples for split.")
        self._check_feature_files_exist()

    def _check_feature_files_exist(self):
        missing_count = 0
        for idx in range(len(self.metadata_df)):
            shot_id = self.metadata_df.iloc[idx][self.identifier_col]
            feature_file = self.sequence_features_dir / f"{shot_id}.npy"
            if not feature_file.exists():
                missing_count += 1
        if missing_count > 0:
            logger.warning(f"{missing_count}/{len(self.metadata_df)} sequence feature files missing")
        else:
            logger.info("All sequence feature files found.")

    def __len__(self):
        return len(self.metadata_df)

    def _augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:  # 50% chance of time warping
            new_length = max(5, int(len(sequence) * random.uniform(0.8, 1.2)))
            tensor = torch.from_numpy(sequence.copy()).float().permute(1, 0).unsqueeze(0)
            tensor = F.interpolate(tensor, size=new_length, mode='linear', align_corners=False)
            sequence = tensor.squeeze(0).permute(1, 0).numpy()

        if random.random() < 0.3:  # 30% chance of noise
            noise = np.random.normal(0, 0.01, sequence.shape)
            sequence = sequence + noise
                
        return sequence

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        row = self.metadata_df.iloc[idx]
        shot_id = row[self.identifier_col]
        target = torch.tensor(int(row[self.target_col]), dtype=torch.long)
        static_features = torch.tensor(self.static_features_processed[idx], dtype=torch.float32)

        feature_file = self.sequence_features_dir / f"{shot_id}.npy"
        try:
            vis_seq_full = np.load(feature_file, mmap_mode='r')
            if vis_seq_full.dtype != np.float32:
                raise ValueError(f"Invalid dtype: {vis_seq_full.dtype}, expected np.float32")
            actual_length = len(vis_seq_full)
            
            if self.augment:
                vis_seq_full = self._augment_sequence(vis_seq_full)
                actual_length = len(vis_seq_full)
            
            if actual_length > self.sequence_length:
                if self.truncation_strategy == "first":
                    vis_seq = vis_seq_full[:self.sequence_length]
                elif self.truncation_strategy == "last":
                    vis_seq = vis_seq_full[-self.sequence_length:]
                elif self.truncation_strategy == "random":
                    start_idx = np.random.randint(0, actual_length - self.sequence_length)
                    vis_seq = vis_seq_full[start_idx:start_idx+self.sequence_length]
                else:
                    vis_seq = vis_seq_full[:self.sequence_length]
                actual_length = self.sequence_length
            else:
                vis_seq = vis_seq_full
                
            vis_seq = torch.from_numpy(vis_seq_full.copy()).float()
            length = torch.tensor(actual_length, dtype=torch.long)
            
            return vis_seq, static_features, target, length

        except Exception as e:
            logger.error(f"Error loading idx {idx} (shot {shot_id}): {e}", exc_info=True)
            if self.strict:
                raise
            return None

def optimized_collate_fn(batch: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return None
    if len(valid_batch) < len(batch):
        logger.warning(f"Skipped {len(batch) - len(valid_batch)} invalid samples in batch.")

    try:
        vis_seqs, static_features, targets, lengths = zip(*valid_batch)
        
        max_len = max(seq.size(0) for seq in vis_seqs)
        padded_vis_seqs = nn.utils.rnn.pad_sequence(
            vis_seqs,
            batch_first=True,
            padding_value=0.0,
            total_length=max_len  # ensure equal length across batch
        )
        
        static_features_tensor = torch.stack(static_features, dim=0)
        targets_tensor = torch.stack(targets, dim=0)
        lengths_tensor = torch.stack(lengths, dim=0)
        
        return padded_vis_seqs, static_features_tensor, targets_tensor, lengths_tensor
    except Exception as e:
        logger.error(f"Error during batch collation: {e}")
        return None

class MultimodalTrainer:
    def __init__(self, model: nn.Module, config: Dict, device: Optional[torch.device] = None, ddp: bool = False):
        self.config = config
        self.training_config = config['training']
        self.model_config = config['model']
        self.data_config = config['data']
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ddp = ddp
        
        if ddp:
            self.model = DistributedDataParallel(model.to(self.device), device_ids=[self.device])
        else:
            self.model = model.to(self.device)
            
        self.output_dir = Path(self.training_config['output_dir'])
        self.experiment_dir = self.output_dir / self.training_config['experiment_name']
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = self.data_config.get('class_names', [f"Class_{i}" for i in range(self.model_config['output_dim'])])
        self.num_classes = self.model_config['output_dim']
        self._setup_logging()
        self._setup_training_components()
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
        self.best_val_loss = float('inf')
        self.epoch = 0
        self._check_plotting_libraries()
        logger.info(f"Trainer initialized. Output Dir: {self.experiment_dir}")

    def _setup_logging(self):
        log_file = self.experiment_dir / "training.log"
        global logger
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file):
                break
            elif isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)
        else:
            logger.addHandler(logging.FileHandler(log_file))
        self.logger = logger

    def _check_plotting_libraries(self):
        self.matplotlib_available = importlib.util.find_spec("matplotlib") is not None
        self.seaborn_available = importlib.util.find_spec("seaborn") is not None
        self.plotting_enabled = self.matplotlib_available and self.seaborn_available
        if not self.plotting_enabled:
            logger.warning("Plotting disabled.")
        self.training_config['plot_history'] = self.plotting_enabled and self.training_config.get('plot_history', False)
        self.training_config['plot_confusion_matrix'] = self.plotting_enabled and self.training_config.get('plot_confusion_matrix', False)

    def _setup_training_components(self):
        # Loss function setup is moved to the main function to handle dynamic class weights
        if self.training_config.get('use_focal_loss', False):
            self.criterion = lambda logits, targets: sigmoid_focal_loss(
                logits, 
                F.one_hot(targets, num_classes=self.num_classes).float(),
                alpha=self.training_config.get('focal_alpha', 0.25),
                gamma=self.training_config.get('focal_gamma', 2.0),
                reduction='mean'
            )
            logger.info("Using Focal Loss for class imbalance")
        else:
            class_wts = self.training_config.get('class_weights')
            if class_wts:
                weight_tensor = torch.tensor(class_wts, dtype=torch.float32, device=self.device)
                self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
                logger.info(f"Using weighted CrossEntropyLoss: {class_wts}")
            else:
                self.criterion = nn.CrossEntropyLoss()
                logger.info("Using CrossEntropyLoss (no class weights).")
            
        opt_name = self.training_config['optimizer'].lower()
        lr = self.training_config['learning_rate']
        wd = self.training_config.get('weight_decay', 0.0)
        momentum = self.training_config.get('momentum', 0.9)
        if opt_name == 'adamw':
            self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'sgd':
            self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        else:
            logger.warning(f"Optimizer '{opt_name}' unknown. Defaulting to AdamW.")
            self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        logger.info(f"Using Optimizer: {type(self.optimizer).__name__}")
        
        sched_name = self.training_config.get('scheduler')
        if sched_name == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.training_config.get('lr_factor', 0.5),
                patience=self.training_config.get('lr_patience', 5),
                min_lr=1e-7,
                verbose=True
            )
            logger.info("Using ReduceLROnPlateau scheduler.")
        else:
            self.scheduler = None
            logger.info("No LR scheduler used.")
            
        self.amp_scaler = torch.amp.GradScaler(
            enabled=self.training_config.get('use_mixed_precision', False) and self.device.type == 'cuda'
        )
        logger.info(f"AMP Enabled: {self.amp_scaler.is_enabled()}")

    def train_epoch(self, epoch: int, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, correct, total_samples = 0.0, 0, 0
        accumulation_steps = self.training_config.get('grad_accumulation_steps', 1)
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False, ncols=100)
        
        for batch_idx, batch_data in enumerate(train_iter):
            if batch_data is None:
                logger.warning(f"Skipping empty batch {batch_idx}.")
                continue
                
            vis_seq, static_features, targets, lengths = batch_data
            vis_seq, static_features, targets = (
                vis_seq.to(self.device),
                static_features.to(self.device),
                targets.to(self.device).long()
            )
            batch_size = targets.size(0)
            
            with torch.amp.autocast('cuda', enabled=self.amp_scaler.is_enabled()):
                outputs = self.model(vis_seq, static_features, lengths=lengths)
                logits = outputs["logits"]
                loss = self.criterion(logits, targets)
                
            scaled_loss = loss / accumulation_steps
            self.amp_scaler.scale(scaled_loss).backward()
            
            if self.training_config.get('grad_clip_value'):
                self.amp_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.training_config['grad_clip_value']
                )
            
            if (batch_idx + 1) % accumulation_steps == 0:
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * batch_size
            _, predicted = torch.max(logits, 1)
            total_samples += batch_size
            correct += (predicted == targets).sum().item()
            
            accuracy = 100. * correct / total_samples if total_samples > 0 else 0
            train_iter.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.2f}%'})
            
            if batch_idx % self.training_config['log_interval'] == 0 and batch_idx > 0:
                self.logger.info(f"E{epoch+1} B{batch_idx}/{len(train_loader)} Loss: {loss.item():.4f} Acc: {accuracy:.2f}%")

        if (batch_idx + 1) % accumulation_steps != 0:
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
            self.optimizer.zero_grad()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = 100. * correct / total_samples if total_samples > 0 else 0
        self.history["train_loss"].append(avg_loss)
        self.history["train_acc"].append(accuracy)
        return {'loss': avg_loss, 'accuracy': accuracy}

    def evaluate(self, data_loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        total_loss, correct, total_samples = 0.0, 0, 0
        all_targets, all_preds_indices, all_probs = [], [], []
        eval_iter = tqdm(data_loader, desc=f"Epoch {self.epoch+1} Val", leave=False, ncols=100)
        
        with torch.no_grad():
            for batch_data in eval_iter:
                if batch_data is None:
                    logger.warning("Skipping empty eval batch.")
                    continue
                    
                vis_seq, static_features, targets, lengths = batch_data
                vis_seq, static_features, targets = (
                    vis_seq.to(self.device),
                    static_features.to(self.device),
                    targets.to(self.device).long()
                )
                batch_size = targets.size(0)
                
                with torch.amp.autocast('cuda', enabled=self.amp_scaler.is_enabled()):
                    outputs = self.model(vis_seq, static_features, lengths=lengths)
                    logits = outputs["logits"]
                    loss = self.criterion(logits, targets)
                    
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item() * batch_size
                    
                probabilities = outputs["prediction"]
                _, predicted = torch.max(logits, 1)
                total_samples += batch_size
                correct += (predicted == targets).sum().item()
                
                all_targets.extend(targets.cpu().numpy())
                all_preds_indices.extend(predicted.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
                
                accuracy = 100. * correct / total_samples if total_samples > 0 else 0
                eval_iter.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.2f}%'})
                
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        accuracy = 100. * correct / total_samples if total_samples > 0 else 0
        self.history.setdefault("val_loss", []).append(avg_loss)
        self.history.setdefault("val_acc", []).append(accuracy)
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.history.setdefault("lr", []).append(current_lr)

        targets_np = np.array(all_targets)
        preds_np = np.array(all_preds_indices)
        probs_np = np.array(all_probs)
        report_dict = {}
        conf_matrix = None
        roc_auc = None
        class_metrics = {}
        calibration_error = None
        
        if len(targets_np) > 0 and len(preds_np) > 0:
            try:
                labels_present = np.unique(targets_np)
                target_names = [self.class_names[i] for i in labels_present]
                report_dict = classification_report(
                    targets_np,
                    preds_np,
                    target_names=target_names,
                    labels=labels_present,
                    output_dict=True,
                    zero_division=0
                )
                conf_matrix = confusion_matrix(targets_np, preds_np, labels=np.arange(self.num_classes))
                
                if probs_np is not None and len(labels_present) > 1:
                    try:
                        roc_auc = roc_auc_score(
                            targets_np,
                            probs_np,
                            multi_class='ovr',
                            labels=np.arange(self.num_classes),
                            average='weighted'
                        )
                    except Exception as roc_e:
                        logger.warning(f"ROC AUC failed: {roc_e}")
                        roc_auc = np.nan
            except Exception as metric_e:
                logger.error(f"Metric calculation error: {metric_e}", exc_info=True)
                
                # ---- Extra metrics (per-class & calibration) ----
                if report_dict:
                    for cname, cm in report_dict.items():
                        if isinstance(cm, dict) and 'precision' in cm:
                            class_metrics[f"prec_{cname}"] = cm['precision'] * 100
                            class_metrics[f"recall_{cname}"] = cm['recall'] * 100
                            class_metrics[f"f1_{cname}"] = cm['f1-score'] * 100
                if self.num_classes == 2 and probs_np.size > 0:
                    try:
                        from sklearn.calibration import calibration_curve
                        prob_true, prob_pred = calibration_curve(targets_np, probs_np[:, 1], n_bins=10, strategy='uniform')
                        calibration_error = float(np.mean(np.abs(prob_true - prob_pred)))
                    except Exception as cal_e:
                        logger.warning(f"Calibration metric failed: {cal_e}")
                
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix,
            'classification_report': report_dict,
            'class_metrics': class_metrics,
            'calibration_error': calibration_error,
            'targets': targets_np,
            'predictions': preds_np,
            'probabilities': probs_np
        }

    def save_artifacts(self, model, preprocessor, feature_cols: List[str]):
        """Save model artifacts with metadata"""
        # Save preprocessor components
        pickle.dump(preprocessor.named_steps["imputer"],
                   open(self.config.artifacts/"imputer.pkl", "wb"))
        pickle.dump(preprocessor.named_steps["scaler"],
                   open(self.config.artifacts/"scaler.pkl", "wb"))
        
        # Save feature list
        json.dump(feature_cols, 
                 open(self.config.artifacts/"feature_list.json", "w"))
        
        # Save main model
        with open(self.config.artifacts/"ensemble_model.pkl", "wb") as f:
            pickle.dump(model, f)
            
        # Save training metadata
        metadata = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_features': len(feature_cols),
            'random_state': self.config.random_state,
            'metrics': self.metrics_history,
            'config': {
                'test_size': self.config.test_size,
                'cv_folds': self.config.cv_folds
            }
        }
        
        with open(self.config.artifacts/"training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
            
        print(f"\n✔ All artifacts saved to {self.config.artifacts.resolve()}")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        num_epochs = self.training_config['num_epochs']
        patience = self.training_config['early_stopping_patience']
        checkpoint_path = self.checkpoints_dir / self.training_config['best_model_name']
        use_profiler = self.training_config.get('enable_profiling', False)

        monitor_metric = self.training_config.get('early_stopping_monitor', 'loss')
        monitor_mode = 'min' if 'loss' in monitor_metric.lower() else 'max'

        early_stopper = EarlyStopping(
            patience=patience, 
            verbose=True, 
            path=str(checkpoint_path),
            monitor=monitor_metric,
            mode=monitor_mode
        )
        # ---- LR Warm-up ----
        self.warmup_epochs = self.training_config.get('lr_warmup_epochs', 0)
        warmup_scheduler = None
        if self.warmup_epochs > 0:
            from torch.optim.lr_scheduler import LinearLR
            warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01, total_iters=self.warmup_epochs)
        self.logger.info(f"Starting training from epoch {self.epoch} for max {num_epochs} epochs...")
        
        profiler = None
        if use_profiler and self.device.type == 'cuda':
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
                schedule=torch.profiler.schedule(skip_first=2, wait=1, warmup=1, active=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.experiment_dir / "profiler")),
                record_shapes=True, profile_memory=True, with_stack=True
            )
            profiler.start()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            if profiler:
                profiler.step()
                
            train_metrics = self.train_epoch(epoch, train_loader)
            val_metrics = self.evaluate(val_loader)
            lr = self.optimizer.param_groups[0]['lr']
            
            self.logger.info(
                f"Epoch {epoch+1}: Train Loss={train_metrics['loss']:.4f}, "
                f"Train Acc={train_metrics['accuracy']:.2f}%, "
                f"Val Loss={val_metrics['loss']:.4f}, "
                f"Val Acc={val_metrics['accuracy']:.2f}%, "
                f"ROC-AUC={val_metrics.get('roc_auc', 0):.4f}, "
                f"LR={lr:.1e}"
            )
            
            if warmup_scheduler and epoch < self.warmup_epochs:
                warmup_scheduler.step()
            if self.scheduler and isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
                
            early_stopper(val_metrics, self.model, self.optimizer, self.scheduler, epoch)

            # Memory management
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            if early_stopper.early_stop:
                logger.info("Early stopping.")
                break
                
        if profiler:
            profiler.stop()
            logger.info("Profiling completed")

        final_model_path = self.experiment_dir / "final_model_state.pt"
        model_to_save = self.model.module if self.ddp else self.model
        model_to_save.save_checkpoint(
            final_model_path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            metrics=val_metrics,
            quantize=False
        )
        logger.info(f"Final model state saved to {final_model_path}")

        if checkpoint_path.exists():
            self.logger.info(f"Loading best model weights (Epoch {early_stopper.best_epoch + 1}) from {checkpoint_path}")
            best_model_loaded, _ = MultimodalRNN.load_from_checkpoint(str(checkpoint_path), map_location=self.device)
            if best_model_loaded:
                self.model = best_model_loaded.to(self.device)
                self.logger.info("Loaded best model.")
            else:
                self.logger.error("Failed to load best model.")
        else:
            self.logger.warning("Best model checkpoint not found.")
            
        hist_path = self.experiment_dir / "training_history.json"
        try:
            serializable_history = {
                k: [float(item) for item in v if isinstance(item, (float, int, np.number))]
                for k, v in self.history.items() if v
            }
            with open(hist_path, 'w') as f:
                json.dump(serializable_history, f, indent=2)
            logger.info(f"History saved to {hist_path}")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
            
        return self.history

    def plot_training_history(self, save_path=None):
        if not self.plotting_enabled or not self.history or not self.history.get('train_loss'):
            logger.warning("No history to plot.")
            return None
        self.logger.info("Plotting training history...")
        try:
            save_path_full = self.experiment_dir / (save_path or "training_history.png")
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            epochs = range(1, len(self.history['train_loss']) + 1)
            axes[0].plot(epochs, self.history['train_loss'], 'b-o', markersize=3, linewidth=1, label='Training Loss')
            axes[0].plot(epochs, self.history['val_loss'], 'r-o', markersize=3, linewidth=1, label='Validation Loss')
            axes[0].set_title('Loss')
            axes[0].set_xlabel('Epochs')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(bottom=0)
            axes[1].plot(epochs, self.history['train_acc'], 'b-o', markersize=3, linewidth=1, label='Training Accuracy')
            axes[1].plot(epochs, self.history['val_acc'], 'r-o', markersize=3, linewidth=1, label='Validation Accuracy')
            axes[1].set_title('Accuracy')
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.5)
            axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
            axes[1].set_ylim(0, 105)
            plt.tight_layout()
            fig.savefig(save_path_full)
            plt.close(fig)
            logger.info(f"History plot saved: {save_path_full}")
            return fig
        except Exception as e:
            logger.warning(f"Plot history failed: {e}", exc_info=True)
            return None

    def export_onnx(self, onnx_path: str = None, quantize: bool = False, opset: int = 17):
        """Export the trained model to ONNX format and optionally quantize it.

        Args:
            onnx_path: Destination path; defaults to <experiment_dir>/model.onnx.
            quantize: If True, produce an 8-bit quantised version using onnxruntime.
            opset: ONNX opset version to target (default 17).
        Returns:
            Path to the saved ONNX (or quantised) model, or None on failure.
        """
        try:
            # Ensure model is on CPU for export to avoid GPU-specific ops
            cpu_model = (self.model.module if self.ddp else self.model).to('cpu').eval()
            onnx_path = onnx_path or str(self.experiment_dir / "model.onnx")

            seq_len = self.data_config.get('sequence_length', self.model_config.get('sequence_length', 32))
            vis_dim = self.model_config['vis_feat_dim']
            static_dim = self.model_config['static_feat_dim']

            dummy_vis = torch.randn(1, seq_len, vis_dim, dtype=torch.float32)
            dummy_static = torch.randn(1, static_dim, dtype=torch.float32)
            dummy_len = torch.tensor([seq_len], dtype=torch.long)

            input_names = ["vis_seq", "static_features", "lengths"]
            output_names = ["logits"]
            dynamic_axes = {
                'vis_seq': {0: 'batch_size', 1: 'sequence_length'},
                'static_features': {0: 'batch_size'},
                'lengths': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }

            torch.onnx.export(
                cpu_model,
                (dummy_vis, dummy_static, dummy_len),
                onnx_path,
                export_params=True,
                opset_version=opset,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True
            )
            self.logger.info(f"ONNX model exported: {onnx_path}")

            if quantize:
                try:
                    from onnxruntime.quantization import quantize_dynamic, QuantType
                    quant_path = onnx_path.replace('.onnx', '_quant.onnx')
                    quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)
                    self.logger.info(f"Quantized ONNX model saved: {quant_path}")
                    return quant_path
                except Exception as q_e:
                    self.logger.warning(f"Quantization failed: {q_e}")
            return onnx_path
        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}", exc_info=True)
            return None

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        if not self.plotting_enabled:
            return None
        self.logger.info("Plotting confusion matrix...")
        try:
            save_path_full = self.experiment_dir / (save_path or "confusion_matrix.png")
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(self.num_classes))
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)
            fig, ax = plt.subplots(figsize=(8, 7))
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt=".2%",
                cmap="Blues",
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=ax,
                annot_kws={"size": 10}
            )
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            ax.set_title('Normalized Confusion Matrix')
            plt.tight_layout()
            fig.savefig(save_path_full)
            plt.close(fig)
            logger.info(f"Confusion matrix saved: {save_path_full}")
            return fig
        except Exception as e:
            logger.warning(f"Plot confusion matrix failed: {e}")
            return None

class ModelInterpretability:
    def __init__(self, model: MultimodalRNN, device: Union[str, torch.device] = "cuda"):
        self.model = model
        self.device = device
        self.shap_available = importlib.util.find_spec("shap") is not None
        if self.shap_available:
            import shap
            self.shap = shap
            logger.info("SHAP library found.")
        else:
            logger.warning("SHAP library not installed.")
        self.feature_categories = model.feature_categories
        self.class_names = model.config.get('data', {}).get('class_names', [f"C_{i}" for i in range(model.config['model']['output_dim'])])

    def integrated_gradients(self, input_seq: torch.Tensor, static_features: torch.Tensor, 
                             target_class: int, steps: int = 50) -> torch.Tensor:
        baseline_seq = torch.zeros_like(input_seq)
        baseline_static = torch.zeros_like(static_features)
        
        gradients = []
        for alpha in torch.linspace(0, 1, steps, device=self.device):
            interpolated_seq = baseline_seq + alpha * (input_seq - baseline_seq)
            interpolated_static = baseline_static + alpha * (static_features - baseline_static)
            interpolated_seq.requires_grad = True
            
            output = self.model(interpolated_seq, interpolated_static)
            output[0, target_class].backward()
            
            gradients.append(interpolated_seq.grad.clone())
            interpolated_seq.grad.zero_()
        
        avg_grads = torch.mean(torch.stack(gradients), dim=0)
        integrated_grads = (input_seq - baseline_seq) * avg_grads
        return integrated_grads

    def _model_predict_wrapper_for_shap(self, inputs: np.ndarray):
        self.model.eval()
        batch_size = inputs.shape[0]
        vis_dim = self.model.config['model']['vis_feat_dim']
        static_dim = self.model.config['model']['static_feat_dim']
        seq_len = self.model.config['data']['sequence_length']
        
        seq_features_flat_size = seq_len * vis_dim
        vis_seq_flat = inputs[:, :seq_features_flat_size]
        static_scores = inputs[:, seq_features_flat_size:]
        
        vis_seq = vis_seq_flat.reshape(batch_size, seq_len, vis_dim)
        
        vis_seq = torch.tensor(vis_seq).float().to(self.device)
        static_scores = torch.tensor(static_scores).float().to(self.device)
        
        with torch.no_grad():
            outputs = self.model(vis_seq, static_scores)
        
        return outputs["prediction"].cpu().numpy()

    def analyze_features(self, data_loader, num_background=50, num_samples=10):
        if not self.shap_available:
            logger.error("SHAP not installed.")
            return None
        logger.info(f"Starting SHAP (background={num_background}, samples={num_samples})...")
        
        background_data = []
        samples_data = []
        samples_targets = []
        
        with timer("SHAP Data Collection", logger):
            for i, batch_data in enumerate(data_loader):
                if batch_data is None:
                    continue
                vis_seq, static, targets, _ = batch_data
                vis_flat = vis_seq.reshape(vis_seq.shape[0], -1)
                combined = torch.cat([vis_flat, static], dim=1)
                background_data.append(combined)
                if i * data_loader.batch_size >= num_background:
                    break
            if not background_data:
                logger.error("No background data.")
                return None
            background = torch.cat(background_data, dim=0)[:num_background].numpy()
            
            count = 0
            for i, batch_data in enumerate(data_loader):
                if batch_data is None:
                    continue
                vis_seq, static, targets, _ = batch_data
                vis_flat = vis_seq.reshape(vis_seq.shape[0], -1)
                combined = torch.cat([vis_flat, static], dim=1)
                samples_data.append(combined)
                samples_targets.append(targets)
                count += combined.shape[0]
                if count >= num_samples:
                    break
            if not samples_data:
                logger.error("No samples data.")
                return None
            samples = torch.cat(samples_data, dim=0)[:num_samples].numpy()
            true_labels = torch.cat(samples_targets, dim=0)[:num_samples].numpy()

        logger.info("Initializing SHAP KernelExplainer...")
        try:
            explainer = self.shap.KernelExplainer(self._model_predict_wrapper_for_shap, background, link="logit")
            logger.info("Calculating SHAP values...")
            with timer("SHAP Value Calculation", logger):
                shap_values = explainer.shap_values(samples, nsamples='auto')
            logger.info("SHAP analysis complete.")
            
            seq_len = self.model.config['data']['sequence_length']  
            vis_dim = self.model.config['model']['vis_feat_dim']
            
            seq_features_flat_size = seq_len * vis_dim
            vis_seq = samples[:, :seq_features_flat_size].reshape(-1, seq_len, vis_dim)
            static_tensor = samples[:, seq_features_flat_size:]
            
            vis_tensor = torch.tensor(vis_seq).float().to(self.device)
            static_tensor = torch.tensor(static_tensor).float().to(self.device)
            
            with torch.no_grad():
                outputs = self.model(vis_tensor, static_tensor)
            predictions = outputs["prediction"].cpu().numpy()
            predicted_classes = np.argmax(predictions, axis=1)
            
            seq_feat_names = []
            for t in range(seq_len):
                for d in range(vis_dim):
                    seq_feat_names.append(f"seq_t{t}_f{d}")
            
            feature_names = seq_feat_names + self.feature_categories
            
            return {
                "shap_values": shap_values,
                "samples": samples,
                "true_labels": true_labels,
                "feature_names": feature_names,
                "class_names": self.class_names,
                "predictions": predictions,
                "predicted_classes": predicted_classes
            }
        except Exception as e:
            logger.error(f"SHAP calculation failed: {e}", exc_info=True)
            return None

    def plot_shap_summary(self, analysis_results, save_path=None):
        if not self.shap_available or not importlib.util.find_spec("matplotlib") or analysis_results is None:
            return None
        logger.info("Plotting SHAP summary...")
        try:
            save_path_full = Path(self.model.config['training']['output_dir']) / (save_path or "shap_summary.png")
            save_path_full.parent.mkdir(parents=True, exist_ok=True)
            plt.figure()
            self.shap.summary_plot(
                analysis_results["shap_values"],
                features=analysis_results["samples"],
                feature_names=analysis_results["feature_names"],
                class_names=analysis_results["class_names"],
                show=False,
                plot_size=(10, 8)
            )
            plt.tight_layout()
            plt.savefig(save_path_full)
            plt.close()
            logger.info(f"SHAP summary plot saved to {save_path_full}")
            return plt.gcf()
        except Exception as e:
            logger.warning(f"Could not plot SHAP summary: {e}")
            return None

def validate_config(config: Dict):
    required_sections = ['data', 'model', 'training', 'feature_extractor']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config missing section: '{section}'")
    req_data = ['csv_path', 'sequence_features_dir', 'static_features', 'target', 'split_column', 'identifier_col']
    req_model = ['vis_feat_dim', 'static_feat_dim', 'rnn_hidden_dim', 'dense_hidden_dim', 'output_dim', 'sequence_length']
    req_train = ['optimizer', 'learning_rate', 'batch_size', 'num_epochs', 'output_dir', 'experiment_name']
    req_fe = ['output_dim']
    for key in req_data:
        if key not in config['data']:
            raise ValueError(f"Config missing 'data.{key}'")
    for key in req_model:
        if key not in config['model']:
            raise ValueError(f"Config missing 'model.{key}'")
    for key in req_train:
        if key not in config['training']:
            raise ValueError(f"Config missing 'training.{key}'")
    if 'output_dim' not in config['feature_extractor']:
        raise ValueError("Missing 'feature_extractor.output_dim'")
    if config['model']['static_feat_dim'] != len(config['data']['static_features']):
        raise ValueError("static_feature_cols length != model.static_feat_dim")
    if config['model']['vis_feat_dim'] != config['feature_extractor']['output_dim']:
        raise ValueError("vis_feat_dim != feature_extractor.output_dim")
    logger.info("Config validation passed.")

def setup_distributed():
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    return device, local_rank

def main(config_path: str):
    global logger, DEVICE
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
            # Validate configuration with Pydantic and enforce reproducibility
            cfg = PipelineConfig.model_validate(config)
            config = cfg.model_dump(mode='python')  # convert back to dict for legacy code
            seed = cfg.training.random_seed
            set_seed(seed)

        # Handle distributed training
        ddp = config['training'].get('distributed', False)
        if ddp:
            device, rank = setup_distributed()
            world_size = dist.get_world_size()
            config['training']['batch_size'] = config['training']['batch_size'] // world_size
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            rank = 0
            
        logger = setup_logging(config['training']['output_dir'], config['training']['experiment_name'])
        logger.info(f"Loaded config from {config_path}")
        validate_config(config)
        output_dir = Path(config['training']['output_dir'])
        exp_dir = output_dir / config['training']['experiment_name']
        if rank == 0:
            exp_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Config Error: {e}", exc_info=True)
        sys.exit(1)

    try:
        set_seed(config.get('seed', 42))
        DEVICE = device

        logger.info("--- Preparing Data ---")
        with timer("Data Loading, Preprocessing & Splitting", logger):
            metadata_df = pd.read_csv(config['data']['csv_path'])
            static_cols = config['data']['static_features']
            
            # Add split column if not present – stratified on target
            if config['data']['split_column'] not in metadata_df.columns:
                logger.info("Split column not found → performing stratified split (70/15/15)")
                train_df, temp_df = train_test_split(
                    metadata_df,
                    test_size=0.30,
                    stratify=metadata_df[config['data']['target']],
                    random_state=config.get('seed', 42)
                )
                val_df, test_df = train_test_split(
                    temp_df,
                    test_size=0.50,
                    stratify=temp_df[config['data']['target']],
                    random_state=config.get('seed', 42)
                )
                metadata_df.loc[train_df.index, config['data']['split_column']] = 'train'
                metadata_df.loc[val_df.index, config['data']['split_column']] = 'val'
                metadata_df.loc[test_df.index, config['data']['split_column']] = 'test'
                logger.info(
                    "Stratified split created – train %d / val %d / test %d",
                    len(train_df), len(val_df), len(test_df)
                )
            
            split_col = config['data']['split_column']
            target_col = config['data']['target']

            imputer = SimpleImputer(strategy=config['data'].get('imputation_strategy', 'mean'))
            static_scaler = StandardScaler()

            train_mask = metadata_df[split_col] == 'train'
            X_static_train_raw = metadata_df.loc[train_mask, static_cols].values.astype(np.float32)
            X_static_train_imputed = imputer.fit_transform(X_static_train_raw)
            X_static_train_scaled = static_scaler.fit_transform(X_static_train_imputed)

            # ---- Compute class weights for imbalance ----
            class_counts = metadata_df.loc[train_mask, target_col].value_counts().sort_index().values
            class_weights = (class_counts.sum() / (len(class_counts) * class_counts)).tolist()
            config['training']['class_weights'] = class_weights
            logger.info(f"Class weights: {class_weights}")
            logger.info("Static Imputer & Scaler fitted on training data.")

            # Create output directory
            if rank == 0:
                exp_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(static_scaler, exp_dir / config['training']["scaler_static_save_name"])
                joblib.dump(imputer, exp_dir / config['training']["imputer_static_save_name"])
                logger.info(f"Static scaler & imputer saved to {exp_dir}")

            # Create features directory if it doesn't exist
            features_dir = Path(config['data']['sequence_features_dir'])
            if rank == 0:
                features_dir.mkdir(parents=True, exist_ok=True)
            
            # Create dummy feature files for testing if they don't exist (dev only)
            if rank == 0 and config['training'].get('create_dummy_features', False):
                dummy_features = np.random.randn(config['data']['sequence_length'], config['model']['vis_feat_dim']).astype(np.float32)
                for idx, row in metadata_df.iterrows():
                    video_path = row[config['data']['identifier_col']]
                    # Fix: Extract just the filename without the directory part
                    video_filename = os.path.basename(video_path)
                    # Remove the extension if present
                    video_basename = os.path.splitext(video_filename)[0]
                    feature_file = features_dir / f"{video_basename}.npy"
                    if not feature_file.exists():
                        logger.warning(f"Creating dummy feature file for {video_basename}")
                        np.save(feature_file, dummy_features)
            
            # IMPORTANT: We need to update the identifier_col in the metadata
            # to just use the filename without path for model training
            metadata_df['original_path'] = metadata_df[config['data']['identifier_col']]
            metadata_df[config['data']['identifier_col']] = metadata_df[config['data']['identifier_col']].apply(
                lambda x: os.path.splitext(os.path.basename(x))[0]
            )
            
            train_dataset = VFXDataset(
                metadata_df=metadata_df[train_mask],
                sequence_features_dir=config['data']['sequence_features_dir'],
                static_feature_cols=config['data']['static_features'],
                target_col=config['data']['target'],
                identifier_col=config['data']['identifier_col'],
                sequence_length=config['model']['sequence_length'],
                vis_feat_dim=config['model']['vis_feat_dim'],
                padding_strategy=config['data']['padding_strategy'],
                truncation_strategy=config['data']['truncation_strategy'],
                augment=True  # Enable augmentation only for training
            )
            val_dataset = VFXDataset(
                metadata_df=metadata_df[metadata_df[split_col] == 'val'],
                sequence_features_dir=config['data']['sequence_features_dir'],
                static_feature_cols=config['data']['static_features'],
                target_col=config['data']['target'],
                identifier_col=config['data']['identifier_col'],
                sequence_length=config['model']['sequence_length'],
                vis_feat_dim=config['model']['vis_feat_dim'],
                padding_strategy=config['data']['padding_strategy'],
                truncation_strategy=config['data']['truncation_strategy']
            )
            test_dataset = VFXDataset(
                metadata_df=metadata_df[metadata_df[split_col] == 'test'],
                sequence_features_dir=config['data']['sequence_features_dir'],
                static_feature_cols=config['data']['static_features'],
                target_col=config['data']['target'],
                identifier_col=config['data']['identifier_col'],
                sequence_length=config['model']['sequence_length'],
                vis_feat_dim=config['model']['vis_feat_dim'],
                padding_strategy=config['data']['padding_strategy'],
                truncation_strategy=config['data']['truncation_strategy']
            )

            # Distributed samplers
            train_sampler = DistributedSampler(train_dataset) if ddp else None
            val_sampler = DistributedSampler(val_dataset) if ddp else None
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=(train_sampler is None),
                num_workers=config['data'].get('num_workers', 2),
                pin_memory=True,
                persistent_workers=config['data'].get('persistent_workers', False),
                prefetch_factor=config['data'].get('prefetch_factor', 2),
                collate_fn=optimized_collate_fn,
                sampler=train_sampler
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['evaluation']['batch_size'],
                shuffle=False,
                num_workers=config['data'].get('num_workers', 2),
                pin_memory=True,
                persistent_workers=config['data'].get('persistent_workers', False),
                prefetch_factor=config['data'].get('prefetch_factor', 2),
                collate_fn=optimized_collate_fn,
                sampler=val_sampler
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=config['evaluation']['batch_size'],
                shuffle=False,
                num_workers=config['data'].get('num_workers', 2),
                pin_memory=True,
                persistent_workers=config['data'].get('persistent_workers', False),
                prefetch_factor=config['data'].get('prefetch_factor', 2),
                collate_fn=optimized_collate_fn
            )
            logger.info("DataLoaders created.")

    except Exception as e:
        logger.error(f"Data preparation error: {e}", exc_info=True)
        sys.exit(1)

    try:
        model = MultimodalRNN(config=config)
        logger.info(f"Model parameters: {model.count_parameters():,}")
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1 and not ddp:
            logger.info(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
            
        trainer = MultimodalTrainer(model=model, config=config, device=DEVICE, ddp=ddp)
    except Exception as e:
        logger.error(f"Model/Trainer init failed: {e}", exc_info=True)
        sys.exit(1)

    if rank == 0:
        logger.info("\nStarting training...")
    try:
        history = trainer.fit(train_loader, val_loader)
    except Exception as e:
        logger.error(f"Training loop error: {e}", exc_info=True)
        sys.exit(1)

    if rank == 0:
        if config['training'].get('plot_history', True) and trainer.plotting_enabled:
            trainer.plot_training_history(save_path="training_history.png")

        logger.info("\nEvaluating final best model on test set...")
        if trainer.model:
            test_results = trainer.evaluate(test_loader)
            if test_results:
                accuracy = test_results['accuracy']
                precision = test_results.get('precision', 0) * 100
                recall = test_results.get('recall', 0) * 100
                f1 = test_results.get('f1', 0) * 100
                roc_auc = test_results.get('roc_auc')
                targets_np = test_results['targets']
                preds_np = test_results['predictions']
                probs_np = test_results.get('probabilities')
                
                logger.info(f"Test Metrics:")
                logger.info(f"  Accuracy: {accuracy:.2f}%")
                logger.info(f"  Precision: {precision:.2f}%")
                logger.info(f"  Recall: {recall:.2f}%")
                logger.info(f"  F1-Score: {f1:.2f}%")
                logger.info(f"  ROC-AUC: {roc_auc:.4f}")

                logger.info("\nTest Set Classification Report:")
                logger.info("\n" + classification_report(
                    targets_np,
                    preds_np,
                    target_names=trainer.class_names,
                    labels=np.arange(len(trainer.class_names)),
                    zero_division=0
                ))
                if config['training'].get('plot_confusion_matrix', True) and trainer.plotting_enabled:
                    trainer.plot_confusion_matrix(targets_np, preds_np, save_path="confusion_matrix_test.png")
                results_dir = trainer.experiment_dir
                try:
                    np.savez(results_dir / 'test_predictions.npz', targets=targets_np, predictions=preds_np, probabilities=probs_np)
                    report_dict = test_results['classification_report']
                    report_dict['overall_accuracy'] = accuracy / 100.0
                    report_dict['roc_auc_ovr'] = roc_auc
                    report_dict['precision'] = precision / 100.0
                    report_dict['recall'] = recall / 100.0
                    report_dict['f1'] = f1 / 100.0
                    with open(results_dir / 'test_classification_report.json', 'w') as f:
                        json.dump(report_dict, f, indent=2)
                    logger.info(f"Test results and report saved to {results_dir}")
                except Exception as e:
                    logger.error(f"Failed to save test results: {e}")
            else:
                logger.error("Evaluation step failed on test set.")
        else:
            logger.error("Model not available for final evaluation.")

        if config.get('interpretability', {}).get('run_shap_analysis', False):
            logger.info("\n--- Running SHAP Analysis ---")
            if trainer.model and SHAP_AVAILABLE:
                try:
                    interpreter = ModelInterpretability(trainer.model, device=DEVICE)
                    analysis = interpreter.analyze_features(
                        test_loader,
                        num_background=config['interpretability'].get('num_shap_background', 50),
                        num_samples=config['interpretability'].get('num_shap_samples', 10)
                    )
                    if analysis:
                        logger.info("SHAP analysis executed.")
                        if trainer.plotting_enabled:
                            shap_dir = results_dir / "shap_analysis"
                            shap_dir.mkdir(parents=True, exist_ok=True)
                            interpreter.plot_shap_summary(analysis, save_path=str(shap_dir / "shap_summary.png"))
                            logger.info(f"SHAP plots saved to {shap_dir}")
                    else:
                        logger.error("SHAP analysis failed.")
                except Exception as e:
                    logger.error(f"Error during SHAP analysis: {e}", exc_info=True)
            else:
                logger.warning("SHAP disabled/unavailable or model missing.")

    if rank == 0:
        logger.info("\n--- Script Finished Successfully ---")
    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MultimodalRNN model for VFX classification")
    parser.add_argument('--config', type=str, default='D:\\Data Eng\\test multimodale\\test1\\config.yaml', help='Path to YAML config file')
    args = parser.parse_args()
    main(args.config)