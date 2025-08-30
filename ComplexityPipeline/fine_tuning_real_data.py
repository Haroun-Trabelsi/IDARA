import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import joblib
from tqdm import tqdm
import yaml
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import warnings
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import wandb
from contextlib import contextmanager

warnings.filterwarnings('ignore', category=UserWarning)

# --- Enhanced Logging Configuration ---
def setup_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    """Enhanced logging with colored output and better formatting."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, level.upper()))
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Enhanced formatter with more context
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with rotation
    file_handler = logging.FileHandler(
        log_dir / f'finetune_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler.setFormatter(formatter)
    
    # Console handler with colors (if available)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# --- Enhanced Configuration Management ---
class Config:
    """Enhanced configuration with validation and environment variable support."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_default_config()
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                self._deep_update(self.config, file_config)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        self._substitute_env_vars()
        self._validate_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        return {
            'paths': {
                'training_artifacts_base': "D:\\Data Eng\\test multimodale\\final_results\\VFX_Difficulty_Classifier",
                'real_data_csv': "D:\\Data Eng\\test multimodale\\test1\\data\\final_real_data_production.csv",
                'real_npy_folder': "D:\\Data Eng\\test multimodale\\test1\\real_npy_features",
                'output_dir': "D:\\Data Eng\\test multimodale\\test1\\fine_tuning_results"
            },
            'data': {
                'target': 'difficulty_level'
            },
            'fine_tuning': {
                'strategy': 'progressive',  # 'aggressive', 'progressive', 'full'
                'num_epochs': 100,
                'batch_size': 16,
                'learning_rate': 1e-4,
                'val_size': 0.2,
                'early_stopping_patience': 15,
                'use_scheduler': True,
                'scheduler_type': 'cosine',  # 'cosine', 'plateau'
                'warmup_epochs': 5,
                'weight_decay': 1e-4,
                'gradient_clipping': 1.0,
                'use_augmentation': True,
                'augmentation_strength': 0.3,
                'use_class_balancing': True,
                'cross_validation': False,
                'cv_folds': 5,
                'save_intermediate': True,
                'wandb_logging': False,
                'accumulation_steps': 1,
                'label_smoothing': 0.1
            },
            'model': {
                'freeze_backbone_epochs': 10,  # For progressive unfreezing
                'unfreeze_layers': ['temporal_encoder', 'fusion', 'prediction_head'],
                'dropout_increase': 0.1  # Increase dropout during fine-tuning
            },
            'logging': {
                'level': 'INFO',
                'save_plots': True,
                'plot_attention': True
            },
            'training': {
                'label_smoothing': 0.2,
                'class_weights': [1.1, 1.3, 2.0],  
                'learning_rate': 5e-4,  
                'gradient_clipping': 0.8,
                'max_lr_factor': 3,
                'num_epochs': 150,  
                'use_focal_loss': True,
                'focal_gamma': 2.0,
                'augmentation_strength': 0.15  
            }
        }
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _substitute_env_vars(self):
        """Replace environment variables in config values."""
        def substitute_recursive(obj):
            if isinstance(obj, dict):
                return {k: substitute_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                env_var = obj[2:-1]
                return os.getenv(env_var, obj)
            return obj
        
        self.config = substitute_recursive(self.config)
    
    def _validate_config(self):
        """Validate critical configuration parameters."""
        required_paths = ['training_artifacts_base', 'real_data_csv', 'real_npy_folder', 'output_dir']
        for path_key in required_paths:
            path_value = self.get(f'paths.{path_key}')
            if not path_value:
                raise ValueError(f"Required path '{path_key}' not specified in config")
    
    def get(self, key_path: str, default=None):
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def save(self, path: Path):
        """Save current configuration to file."""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

# --- Model Architecture (keeping original but with enhancements) ---
def get_activation_fn(activation: str) -> nn.Module:
    act_map = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "selu": nn.SELU(),
        "tanh": nn.Tanh(),
        "swish": nn.SiLU(),
        "mish": nn.Mish()
    }
    return act_map.get(activation.lower(), nn.GELU())

class AttentionModule(nn.Module):
    def __init__(self, input_dim: int, dropout_rate: float = 0.1, activation: str = "gelu", num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        if num_heads > 1:
            assert input_dim % num_heads == 0, "Input dimension must be divisible by number of heads"
            self.multihead_attn = nn.MultiheadAttention(
                input_dim, num_heads, dropout=dropout_rate, batch_first=True
            )
        else:
            self.attention_layer = nn.Sequential(
                nn.Linear(input_dim, max(1, input_dim // 2)),
                get_activation_fn(activation),
                nn.Dropout(dropout_rate),
                nn.Linear(max(1, input_dim // 2), 1)
            )
    
    def forward(self, x):
        if hasattr(self, 'multihead_attn'):
            attn_out, attn_weights = self.multihead_attn(x, x, x)
            return attn_out.mean(dim=1), attn_weights.mean(dim=1)
        else:
            energy = self.attention_layer(x)
            attn_weights = F.softmax(energy, dim=1)
            return torch.sum(attn_weights * x, dim=1), attn_weights.squeeze(-1)

class EnhancedTemporalEncoder(nn.Module):
    """Enhanced temporal encoder with deeper architecture."""
    
    def __init__(self, input_dim, hidden_dim, num_layers=3, rnn_type='lstm', 
                 bidirectional=True, dropout_rate=0.3, attention_heads=8, num_classes=3):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Deeper RNN with residual connections
        self.rnns = nn.ModuleList()
        current_input_dim = input_dim
        
        for i in range(num_layers):
            rnn_class = nn.LSTM if rnn_type.lower() == 'lstm' else nn.GRU
            rnn = rnn_class(
                current_input_dim, hidden_dim, 1,
                batch_first=True, dropout=0, bidirectional=bidirectional
            )
            self.rnns.append(rnn)
            current_input_dim = hidden_dim * (2 if bidirectional else 1)
        
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Layer normalization for each RNN layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.output_dim) for _ in range(num_layers)
        ])
        
        # Enhanced attention
        self.attention = nn.MultiheadAttention(
            self.output_dim, attention_heads, dropout=dropout_rate, batch_first=True
        )
        
        # Class-specific attention
        self.class_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, attention_heads)
            for _ in range(num_classes)
        ])
        
    def forward(self, x, lengths=None, class_weights=None):
        # Process through multiple RNN layers
        for i, (rnn, layer_norm) in enumerate(zip(self.rnns, self.layer_norms)):
            if lengths is not None and lengths.min() < x.size(1):
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                packed_out, _ = rnn(packed)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_out, batch_first=True
                )
            else:
                x, _ = rnn(x)
            
            x = layer_norm(x)
            
        # Enhanced attention with class-specific focus
        if class_weights is not None:
            # Blend attention from all classes based on weights
            all_attentions = [attn(x, x, x)[0] for attn in self.class_attention]
            class_probs = F.softmax(class_weights, dim=0)
            context = sum(att * prob for att, prob in zip(all_attentions, class_probs))
        else:
            # Default attention
            context, _ = self.attention(x, x, x)
        
        return {"output": context.mean(dim=1)}

class StaticFeatureProcessor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.3, 
                 batch_norm=True, activation="gelu", residual=False):
        super().__init__()
        self.residual = residual and input_dim == output_dim
        
        layers = []
        current_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.BatchNorm1d(h_dim) if batch_norm else nn.Identity(),
                get_activation_fn(activation),
                nn.Dropout(dropout_rate)
            ])
            current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
        if self.residual:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        out = self.mlp(x)
        return out + self.residual_weight * x if self.residual else out

class ModalityFusion(nn.Module):
    def __init__(self, input_dims, fusion_type="gated", use_attention=False):
        super().__init__()
        self.fusion_type = fusion_type
        self.use_attention = use_attention
        self.total_input_dim = sum(input_dims)
        
        if fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(self.total_input_dim, self.total_input_dim),
                nn.Sigmoid()
            )
        elif fusion_type == "attention" or use_attention:
            self.attention = nn.MultiheadAttention(
                self.total_input_dim, max(1, self.total_input_dim // 64), batch_first=True
            )
    
    def forward(self, features):
        concatenated = torch.cat(features, dim=1)
        
        if self.fusion_type == "gated":
            return self.gate(concatenated) * concatenated
        elif self.fusion_type == "attention" or self.use_attention:
            attn_out, _ = self.attention(
                concatenated.unsqueeze(1), concatenated.unsqueeze(1), concatenated.unsqueeze(1)
            )
            return attn_out.squeeze(1)
        return concatenated

class MultimodalRNN(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        model_cfg = config.get('model', {})
        data_cfg = config.get('data', {})
        
        vis_feat_dim = model_cfg.get('vis_feat_dim', 2048)
        static_feat_dim = model_cfg.get('static_feat_dim', 10)
        rnn_hidden_dim = model_cfg.get('rnn_hidden_dim', 128)
        dense_hidden_dim = model_cfg.get('dense_hidden_dim', 64)
        output_dim = model_cfg.get('output_dim', 3)
        
        self.temporal_encoder = EnhancedTemporalEncoder(
            input_dim=vis_feat_dim,
            hidden_dim=rnn_hidden_dim,
            num_layers=model_cfg.get('num_layers', 3),
            rnn_type=model_cfg.get('rnn_type', 'lstm'),
            bidirectional=model_cfg.get('bidirectional', True),
            dropout_rate=model_cfg.get('dropout_rate', 0.3),
            attention_heads=model_cfg.get('attention_heads', 8),
            num_classes=output_dim
        )
        
        self.static_processor = StaticFeatureProcessor(
            input_dim=static_feat_dim,
            hidden_dims=[dense_hidden_dim],
            output_dim=dense_hidden_dim // 2,
            dropout_rate=model_cfg.get('dropout_rate', 0.3),
            batch_norm=model_cfg.get('batch_norm', True),
            residual=model_cfg.get('residual', False)
        )
        
        self.fusion = ModalityFusion(
            input_dims=[self.temporal_encoder.output_dim, dense_hidden_dim // 2],
            fusion_type=model_cfg.get('fusion_type', 'gated'),
            use_attention=model_cfg.get('fusion_attention', False)
        )
        
        self.prediction_head = StaticFeatureProcessor(
            input_dim=self.fusion.total_input_dim,
            hidden_dims=[dense_hidden_dim],
            output_dim=output_dim,
            dropout_rate=model_cfg.get('dropout_rate', 0.3),
            batch_norm=model_cfg.get('batch_norm', True)
        )
    
    def forward(self, vis_seq: torch.Tensor, static_scores: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None, return_attention: bool = False):
        vis_seq = vis_seq.float()
        static_scores = static_scores.float()
        
        temporal_result = self.temporal_encoder(vis_seq, lengths)
        temporal_features = temporal_result["output"]
        
        static_features = self.static_processor(static_scores)
        fused_features = self.fusion([temporal_features, static_features])
        logits = self.prediction_head(fused_features)
        probabilities = F.softmax(logits, dim=-1)
        
        result = {
            "prediction": probabilities,
            "logits": logits
        }
        
        if return_attention and temporal_result.get("attention_weights") is not None:
            result["attention_weights"] = temporal_result["attention_weights"]
        
        return result

# --- Enhanced Dataset with Better Augmentation ---
class AdvancedAugmentations:
    """Advanced data augmentation techniques for multimodal data."""
    
    @staticmethod
    def temporal_jitter(sequence: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """Add temporal jitter to sequence."""
        if np.random.random() < 0.3:
            jitter = np.random.normal(0, strength, sequence.shape)
            return sequence + jitter.astype(np.float32)
        return sequence
    
    @staticmethod
    def feature_dropout(sequence: np.ndarray, drop_prob: float = 0.1) -> np.ndarray:
        """Randomly drop features."""
        if np.random.random() < 0.2:
            mask = np.random.random(sequence.shape) > drop_prob
            return sequence * mask.astype(np.float32)
        return sequence
    
    @staticmethod
    def temporal_masking(sequence: np.ndarray, mask_prob: float = 0.1) -> np.ndarray:
        """Mask random temporal segments."""
        if np.random.random() < 0.15:
            seq_len = sequence.shape[0]
            mask_len = int(seq_len * mask_prob)
            if mask_len > 0:
                start_idx = np.random.randint(0, max(1, seq_len - mask_len))
                sequence[start_idx:start_idx + mask_len] = 0
        return sequence
    
    @staticmethod
    def mixup_sequences(seq1: np.ndarray, seq2: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        """Mixup between two sequences."""
        if np.random.random() < 0.1:
            lam = np.random.beta(alpha, alpha)
            return lam * seq1 + (1 - lam) * seq2
        return seq1

class EnhancedVFXDataset(Dataset):
    """Enhanced dataset with better error handling and augmentation."""
    
    def __init__(self, df: pd.DataFrame, npy_folder: Path, static_features_list: List[str],
                 target_col: str, identifier_col: str, sequence_length: int,
                 vis_feat_dim: int, use_augmentation: bool = False, 
                 augmentation_strength: float = 0.3, cache_features: bool = True):
        self.df = df.reset_index(drop=True)
        self.npy_folder = Path(npy_folder)
        self.static_features_list = static_features_list
        self.target_col = target_col
        self.identifier_col = identifier_col
        self.sequence_length = sequence_length
        self.vis_feat_dim = vis_feat_dim
        self.use_augmentation = use_augmentation
        self.augmentation_strength = augmentation_strength
        self.cache_features = cache_features
        
        self._validate_dataset()
        self.valid_indices = self._find_valid_samples()
        self.augmenter = AdvancedAugmentations()
        
        # Feature caching for faster training
        self.feature_cache = {} if cache_features else None
        
        logger.info(f"Dataset initialized with {len(self.valid_indices)}/{len(self.df)} valid samples.")
        if len(self.valid_indices) < len(self.df):
            missing_samples = len(self.df) - len(self.valid_indices)
            logger.warning(f"WARNING: {missing_samples} samples are being skipped!")
    
    def _validate_dataset(self):
        """Validate dataset structure."""
        required_cols = self.static_features_list + [self.target_col, self.identifier_col]
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")
        
        # Check for NaN values in critical columns
        critical_cols = [self.target_col, self.identifier_col]
        for col in critical_cols:
            if self.df[col].isna().any():
                logger.warning(f"Found NaN values in critical column '{col}'")
    
    def _find_valid_samples(self) -> List[int]:
        """Find samples with valid feature files."""
        valid = []
        missing_files = []
        
        for idx in range(len(self.df)):
            try:
                video_id = self.df.iloc[idx][self.identifier_col]
                if pd.isna(video_id):
                    continue
                    
                npy_path = self.npy_folder / f"{video_id}.npy"
                if npy_path.exists():
                    # Quick validation of file content
                    try:
                        test_load = np.load(npy_path)
                        if test_load.shape[1] == self.vis_feat_dim:
                            valid.append(idx)
                        else:
                            logger.warning(f"Feature dimension mismatch for {video_id}: expected {self.vis_feat_dim}, got {test_load.shape[1]}")
                    except Exception as e:
                        logger.warning(f"Could not load features for {video_id}: {e}")
                        missing_files.append(str(npy_path))
                else:
                    missing_files.append(str(npy_path))
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
        
        if missing_files:
            logger.warning(f"Missing feature files: {len(missing_files)} samples")
            if len(missing_files) <= 10:
                for f in missing_files:
                    logger.debug(f"Missing: {f}")
        
        return valid
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        
        try:
            video_id = row[self.identifier_col]
            
            # Load or retrieve from cache
            cache_key = f"{video_id}_{self.use_augmentation}"
            if self.feature_cache is not None and cache_key in self.feature_cache:
                vis_seq_tensor, static_feat_tensor, target = self.feature_cache[cache_key]
            else:
                # Load visual features
                npy_path = self.npy_folder / f"{video_id}.npy"
                vis_seq_np = np.load(npy_path).astype(np.float32)
                vis_seq_np = self._normalize_sequence_length(vis_seq_np)
                
                # Apply augmentations
                if self.use_augmentation:
                    vis_seq_np = self._apply_augmentation(vis_seq_np)
                
                vis_seq_tensor = torch.tensor(vis_seq_np, dtype=torch.float32)
                
                # Load static features with better error handling
                static_feat_values = []
                for feat in self.static_features_list:
                    val = row[feat]
                    if pd.isna(val):
                        static_feat_values.append(0.0)  # Default value for missing
                        logger.debug(f"Missing static feature '{feat}' for {video_id}, using 0.0")
                    else:
                        static_feat_values.append(float(val))
                
                static_feat_tensor = torch.tensor(static_feat_values, dtype=torch.float32)
                target = torch.tensor(int(row[self.target_col]), dtype=torch.long)
                
                # Cache if enabled
                if self.feature_cache is not None:
                    self.feature_cache[cache_key] = (vis_seq_tensor, static_feat_tensor, target)
            
            return vis_seq_tensor, static_feat_tensor, target
            
        except Exception as e:
            logger.error(f"Error loading sample {row.get(self.identifier_col, 'N/A')}: {e}")
            # Return dummy data to prevent crashes
            dummy_vis = torch.zeros(self.sequence_length, self.vis_feat_dim, dtype=torch.float32)
            dummy_static = torch.zeros(len(self.static_features_list), dtype=torch.float32)
            dummy_target = torch.tensor(0, dtype=torch.long)
            return dummy_vis, dummy_static, dummy_target
    
    def _normalize_sequence_length(self, vis_seq: np.ndarray) -> np.ndarray:
        """Normalize sequence to fixed length with better handling."""
        current_length, feat_dim = vis_seq.shape
        
        if current_length < self.sequence_length:
            # Pad with zeros or repeat last frame
            if current_length > 0:
                padding = np.tile(vis_seq[-1:], (self.sequence_length - current_length, 1))
            else:
                padding = np.zeros((self.sequence_length, feat_dim), dtype=np.float32)
            return np.vstack([vis_seq, padding])
        elif current_length > self.sequence_length:
            # Sample uniformly or take last frames
            if current_length > self.sequence_length * 2:
                # Uniform sampling for very long sequences
                indices = np.linspace(0, current_length - 1, self.sequence_length, dtype=int)
                return vis_seq[indices]
            else:
                # Take last frames
                return vis_seq[-self.sequence_length:]
        
        return vis_seq
    
    def _apply_augmentation(self, vis_seq: np.ndarray) -> np.ndarray:
        """Apply sophisticated augmentation."""
        # Temporal jitter
        vis_seq = self.augmenter.temporal_jitter(vis_seq, self.augmentation_strength * 0.1)
        
        # Feature dropout
        vis_seq = self.augmenter.feature_dropout(vis_seq, self.augmentation_strength * 0.1)
        
        # Temporal masking
        vis_seq = self.augmenter.temporal_masking(vis_seq, self.augmentation_strength * 0.1)
        
        return vis_seq

def enhanced_collate_fn(batch):
    """Enhanced collate function with better error handling."""
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        logger.warning("Empty batch encountered!")
        return None
    
    try:
        vis_seqs, static_features, targets = zip(*valid_batch)
        return (
            torch.stack(list(vis_seqs)),
            torch.stack(list(static_features)),
            torch.stack(list(targets))
        )
    except Exception as e:
        logger.error(f"Error in collate function: {e}")
        return None

# --- Training Utilities ---
class EarlyStopping:
    """Enhanced early stopping with better tracking."""
    
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = 'max', 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = -float('inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, score: float, model: Optional[nn.Module] = None) -> bool:
        improved = False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            if model is not None and self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best weights after early stopping")
        
        return self.early_stop

class MetricsTracker:
    """Enhanced metrics tracking with more detailed statistics."""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        self.preds = []
        self.targets = []
        self.losses = []
        self.probabilities = []
    
    def update(self, preds: np.ndarray, targets: np.ndarray, loss: Optional[float] = None, 
               probabilities: Optional[np.ndarray] = None):
        self.preds.extend(preds.tolist())
        self.targets.extend(targets.tolist())
        if loss is not None:
            self.losses.append(loss)
        if probabilities is not None:
            self.probabilities.extend(probabilities.tolist())
    
    def compute_metrics(self) -> Dict[str, float]:
        if not self.preds:
            return {}
        
        preds_arr = np.array(self.preds)
        targets_arr = np.array(self.targets)
        
        # Basic metrics
        accuracy = np.mean(preds_arr == targets_arr)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets_arr, preds_arr, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        f1_macro = f1_score(targets_arr, preds_arr, average='macro', zero_division=0)
        f1_weighted = f1_score(targets_arr, preds_arr, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'loss': np.mean(self.losses) if self.losses else 0.0
        }
        
        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            if i < len(precision):
                metrics[f'precision_{class_name}'] = precision[i]
                metrics[f'recall_{class_name}'] = recall[i]
                metrics[f'f1_{class_name}'] = f1[i]
                metrics[f'support_{class_name}'] = support[i]
        
        return metrics
    
    def get_report(self) -> str:
        if not self.preds:
            return "No predictions available"
        return classification_report(
            self.targets, self.preds, target_names=self.class_names, zero_division=0
        )
    
    def get_confusion_matrix(self) -> np.ndarray:
        if not self.preds:
            return np.array([])
        return confusion_matrix(self.targets, self.preds)

class FreezingStrategy:
    """Intelligent layer freezing strategies for fine-tuning."""
    
    @staticmethod
    def aggressive_freeze(model: nn.Module) -> int:
        """Freeze everything except prediction head."""
        for param in model.parameters():
            param.requires_grad = False
        
        for param in model.prediction_head.parameters():
            param.requires_grad = True
        
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def progressive_unfreeze(model: nn.Module, epoch: int, unfreeze_schedule: Dict[int, List[str]]) -> int:
        """Progressively unfreeze layers based on epoch."""
        for unfreeze_epoch, layer_names in unfreeze_schedule.items():
            if epoch >= unfreeze_epoch:
                for layer_name in layer_names:
                    if hasattr(model, layer_name):
                        layer = getattr(model, layer_name)
                        for param in layer.parameters():
                            param.requires_grad = True
        
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def full_unfreeze(model: nn.Module) -> int:
        """Unfreeze all parameters."""
        for param in model.parameters():
            param.requires_grad = True
        
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

class VisualizationUtils:
    """Utilities for creating training visualizations."""
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], save_path: Path):
        """Plot training and validation metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.8)
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', alpha=0.8)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_accuracy'], label='Train Accuracy', alpha=0.8)
        axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy', alpha=0.8)
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score plot
        axes[1, 0].plot(history['train_f1_macro'], label='Train F1-Macro', alpha=0.8)
        axes[1, 0].plot(history['val_f1_macro'], label='Validation F1-Macro', alpha=0.8)
        axes[1, 0].set_title('F1-Macro Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1-Macro')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        if 'learning_rate' in history:
            axes[1, 1].plot(history['learning_rate'], label='Learning Rate', alpha=0.8)
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
        """Plot confusion matrix with validation."""
        if cm.size == 0 or len(class_names) == 0:
            logger.warning("Cannot plot empty confusion matrix")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

@contextmanager
def optional_wandb_logging(config: Config):
    """Context manager for optional Weights & Biases logging."""
    if config.get('fine_tuning.wandb_logging', False):
        try:
            wandb.init(
                project="vfx-difficulty-finetuning",
                config=config.config,
                tags=['fine-tuning', 'multimodal']
            )
            yield wandb
        except Exception as e:
            logger.warning(f"Could not initialize wandb: {e}")
            yield None
        finally:
            if wandb.run is not None:
                wandb.finish()
    else:
        yield None

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha=1, gamma=2, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.class_weights is not None:
            focal_loss = focal_loss * self.class_weights[targets]
            
        return focal_loss.mean()

class LabelSmoothingCrossEntropy(nn.Module):
    """Enhanced with class-specific weighting and smoothing"""
    def __init__(self, smoothing=0.2, class_weights=None):
        super().__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights
        
    def forward(self, x, target):
        log_probs = F.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        
        if self.class_weights is not None:
            nll_loss = nll_loss * self.class_weights[target]
            
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean() if self.class_weights is None else loss.sum()

class FinetuningTrainer:
    """Enhanced fine-tuning trainer with multiple strategies."""
    
    def __init__(self, config: Config, model: nn.Module, device: torch.device):
        self.config = config
        self.model = model
        self.device = device
        self.history = defaultdict(list)
        
        # Initialize components
        self.freezing_strategy = FreezingStrategy()
        self.visualizer = VisualizationUtils()
        class_weights = torch.tensor(
            config.get('training.class_weights'), 
            device=device, 
            dtype=torch.float32
        )
        
        # Use focal loss if specified, otherwise label smoothing
        if config.get('training.use_focal_loss', False):
            self.criterion = FocalLoss(
                gamma=config.get('training.focal_gamma', 2.0),
                class_weights=class_weights
            )
        else:
            self.criterion = LabelSmoothingCrossEntropy(
                smoothing=config.get('training.label_smoothing', 0.2),
                class_weights=class_weights
            )
        self.optimizer = AdamW(
            model.parameters(), 
            lr=config.get('training.learning_rate', 2e-4),
            weight_decay=config.get('fine_tuning.weight_decay', 1e-3)
        )
        self.scheduler = OneCycleLR(
            self.optimizer, 
            max_lr=config.get('training.learning_rate') * config.get('training.max_lr_factor', 2),
            epochs=config.get('training.num_epochs'),
            steps_per_epoch=1
        )
    
    def apply_freezing_strategy(self, epoch: int) -> int:
        """Apply the configured freezing strategy."""
        strategy = self.config.get('fine_tuning.strategy', 'progressive')
        
        if strategy == 'aggressive':
            trainable_params = self.freezing_strategy.aggressive_freeze(self.model)
        elif strategy == 'progressive':
            # Define progressive unfreezing schedule
            freeze_epochs = self.config.get('model.freeze_backbone_epochs', 10)
            unfreeze_schedule = {
                0: ['prediction_head'],
                freeze_epochs // 2: ['fusion', 'prediction_head'],
                freeze_epochs: ['temporal_encoder', 'fusion', 'prediction_head']
            }
            trainable_params = self.freezing_strategy.progressive_unfreeze(
                self.model, epoch, unfreeze_schedule
            )
        else:  # full
            trainable_params = self.freezing_strategy.full_unfreeze(self.model)
        
        return trainable_params
    
    def train_epoch(self, train_loader: DataLoader, accumulation_steps: int = 1) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        train_metrics = MetricsTracker(self.config.get('data.class_names', ['Low', 'Medium', 'High']))
        
        # Enable gradient clipping
        max_grad_norm = self.config.get('fine_tuning.gradient_clipping', 1.0)
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc="Training")):
            if batch_data is None:
                continue
                
            vis_seq, static_feat, targets = batch_data
            vis_seq = vis_seq.to(self.device)
            static_feat = static_feat.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(vis_seq, static_feat)
            loss = self.criterion(outputs['logits'], targets)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            _, predicted = torch.max(outputs['logits'], 1)
            train_metrics.update(
                predicted.cpu().numpy(),
                targets.cpu().numpy(),
                loss.item() * accumulation_steps,
                outputs['prediction'].detach().cpu().numpy()
            )
        
        # Handle remaining gradients
        if len(train_loader) % accumulation_steps != 0:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return train_metrics.compute_metrics()
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        val_metrics = MetricsTracker(self.config.get('data.class_names', ['Low', 'Medium', 'High']))
        
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="Validation"):
                if batch_data is None:
                    continue
                    
                vis_seq, static_feat, targets = batch_data
                vis_seq = vis_seq.to(self.device)
                static_feat = static_feat.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(vis_seq, static_feat)
                loss = self.criterion(outputs['logits'], targets)
                
                _, predicted = torch.max(outputs['logits'], 1)
                val_metrics.update(
                    predicted.cpu().numpy(),
                    targets.cpu().numpy(),
                    loss.item(),
                    outputs['prediction'].cpu().numpy()
                )
        
        return val_metrics.compute_metrics()
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              output_dir: Path, wandb_logger=None) -> Dict[str, Any]:
        """Main training loop."""
        num_epochs = self.config.get('fine_tuning.num_epochs')
        accumulation_steps = self.config.get('fine_tuning.accumulation_steps', 1)
        
        # Setup early stopping
        early_stopping = EarlyStopping(
            patience=self.config.get('fine_tuning.early_stopping_patience', 15),
            mode='max',
            restore_best_weights=True
        )
        
        best_val_score = -1.0
        best_epoch = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*50}")
            
            # Apply freezing strategy
            trainable_params_count = self.apply_freezing_strategy(epoch)
            logger.info(f"Trainable parameters: {trainable_params_count:,}")
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, accumulation_steps)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.4f}, "
                       f"F1-Macro: {train_metrics['f1_macro']:.4f}")
            logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.4f}, "
                       f"F1-Macro: {val_metrics['f1_macro']:.4f}")
            logger.info(f"Learning Rate: {current_lr:.2e}")
            
            # Update history
            for key, value in train_metrics.items():
                self.history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.history[f'val_{key}'].append(value)
            self.history['learning_rate'].append(current_lr)
            
            # Log to wandb if available
            if wandb_logger is not None:
                log_dict = {f'train_{k}': v for k, v in train_metrics.items()}
                log_dict.update({f'val_{k}': v for k, v in val_metrics.items()})
                log_dict['learning_rate'] = current_lr
                log_dict['epoch'] = epoch + 1
                log_dict['trainable_params'] = trainable_params_count
                wandb_logger.log(log_dict)
            
            # Save best model
            current_val_score = val_metrics['f1_macro']
            if current_val_score > best_val_score:
                best_val_score = current_val_score
                best_epoch = epoch
                
                # Save model checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'val_f1_macro': current_val_score,
                    'config': self.config.config,
                    'history': dict(self.history)
                }
                torch.save(checkpoint, output_dir / "best_fine_tuned_model.pt")
                logger.info(f"✅ New best model saved! Val F1-Macro: {best_val_score:.4f}")
            
            # Save intermediate checkpoints
            if self.config.get('fine_tuning.save_intermediate', True) and (epoch + 1) % 10 == 0:
                checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'val_f1_macro': current_val_score,
                    'config': self.config.config
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Intermediate checkpoint saved: {checkpoint_path}")
            
            # Early stopping check
            if early_stopping(current_val_score, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Create visualizations
        if self.config.get('logging.save_plots', True):
            plots_dir = output_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Training history plot
            self.visualizer.plot_training_history(dict(self.history), plots_dir / 'training_history.png')
            logger.info("Training history plot saved")
        
        return {
            'best_val_f1_macro': best_val_score,
            'best_epoch': best_epoch,
            'total_epochs': epoch + 1,
            'history': dict(self.history)
        }

def prepare_data_splits(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare train/validation splits with proper stratification."""
    target_col = config.get('data.target')
    val_size = config.get('fine_tuning.val_size', 0.2)
    
    # Check class distribution
    class_counts = df[target_col].value_counts().sort_index()
    logger.info("Class distribution:")
    for class_val, count in class_counts.items():
        logger.info(f"  Class {class_val}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Stratified split
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=42,
        stratify=df[target_col]
    )
    
    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Validation set: {len(val_df)} samples")
    
    return train_df, val_df

def create_balanced_sampler(dataset: EnhancedVFXDataset, target_col: str) -> WeightedRandomSampler:
    """Create a balanced sampler for handling class imbalance."""
    # Use only samples that passed dataset validation
    targets = dataset.df.loc[dataset.valid_indices, target_col].values.astype(int)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / np.maximum(class_counts, 1)  # avoid division by zero
    sample_weights = class_weights[targets]
    
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )

def clean_filename_column(df: pd.DataFrame, identifier_col: str) -> pd.DataFrame:
    """Clean filename column to handle various file extensions and formats."""
    logger.info(f"Cleaning filename column: '{identifier_col}'")
    
    df = df.copy()
    df['original_filename'] = df[identifier_col]
    
    # Remove common video file extensions
    df[identifier_col] = df[identifier_col].str.replace(
        r'\.(mp4|mov|avi|mkv|wmv|flv|webm|m4v|3gp|exr)$', '', regex=True, case=False
    )
    
    # Handle bracket notation (e.g., "filename.[00001-00100]")
    df[identifier_col] = df[identifier_col].str.split(r'\.\[', expand=True)[0]
    
    # Remove extra whitespace
    df[identifier_col] = df[identifier_col].str.strip()
    
    # Log changes
    changed_df = df[df['original_filename'] != df[identifier_col]]
    if not changed_df.empty:
        logger.info(f"Cleaned {len(changed_df)} filenames. Examples:")
        for _, row in changed_df.head(5).iterrows():
            logger.info(f"  '{row['original_filename']}' -> '{row[identifier_col]}'")
    
    return df

def calculate_comprehensive_metrics(targets, predictions, class_names=None):
    """Calculate comprehensive classification metrics."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        cohen_kappa_score, matthews_corrcoef,
        precision_recall_fscore_support, confusion_matrix
    )
    
    results = {}
    
    # Basic metrics
    results['accuracy'] = accuracy_score(targets, predictions)
    results['balanced_accuracy'] = balanced_accuracy_score(targets, predictions)
    results['cohen_kappa'] = cohen_kappa_score(targets, predictions)
    results['mcc'] = matthews_corrcoef(targets, predictions)
    
    # Class-wise metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    
    if class_names is None:
        class_names = [f'class_{i}' for i in range(len(precision))]
    
    for i, name in enumerate(class_names):
        results[f'{name}_precision'] = precision[i]
        results[f'{name}_recall'] = recall[i]
        results[f'{name}_f1'] = f1[i]
        results[f'{name}_support'] = support[i]
    
    # Macro and weighted averages
    for avg in ['macro', 'weighted']:
        results[f'precision_{avg}'] = precision.mean() if avg == 'macro' else \
            np.average(precision, weights=support)
        results[f'recall_{avg}'] = recall.mean() if avg == 'macro' else \
            np.average(recall, weights=support)
        results[f'f1_{avg}'] = f1.mean() if avg == 'macro' else \
            np.average(f1, weights=support)
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    results['confusion_matrix'] = cm
    
    return results

def main():
    """Enhanced main function with comprehensive error handling."""
    try:
        # Initialize configuration
        config = Config()
        
        # Setup output directory and logging
        output_dir = Path(config.get('paths.output_dir'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        global logger
        logger = setup_logging(output_dir / 'logs', config.get('logging.level', 'INFO'))
        
        # Save configuration
        config.save(output_dir / 'fine_tuning_config.yaml')
        
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load pre-trained model and artifacts
        artifacts_path = Path(config.get('paths.training_artifacts_base'))
        pretrained_ckpt_path = artifacts_path / "checkpoints" / "best_vfx_model.pt"
        
        logger.info(f"Loading pretrained checkpoint from: {pretrained_ckpt_path}")
        
        if not pretrained_ckpt_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_ckpt_path}")
        
        # Load checkpoint
        model_data = torch.load(pretrained_ckpt_path, map_location=device, weights_only=False)
        
        if 'config' not in model_data:
            raise ValueError("Checkpoint does not contain model configuration")
        
        model_config = model_data['config']
        logger.info("Successfully loaded model configuration from checkpoint")
        
        # Initialize model
        model = MultimodalRNN(config=model_config).to(device)
        incompat_keys = model.load_state_dict(model_data['state_dict'], strict=False)
        if hasattr(incompat_keys, 'missing_keys') and incompat_keys.missing_keys:
            logger.warning(f"Missing keys when loading checkpoint: {incompat_keys.missing_keys}")
        if hasattr(incompat_keys, 'unexpected_keys') and incompat_keys.unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {incompat_keys.unexpected_keys}")
        logger.info("✅ Model weights loaded (non-strict)")
        
        # Load preprocessing artifacts
        scaler_path = artifacts_path / "static_scaler.pkl"
        imputer_path = artifacts_path / "static_imputer.pkl"
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        scaler = joblib.load(scaler_path)
        imputer = joblib.load(imputer_path) if imputer_path.exists() else None
        logger.info("Preprocessing artifacts loaded")
        
        # Load and validate data
        data_path = Path(config.get('data.csv_path', 'data/final_real_data_cleaned.csv'))
        if not data_path.exists():
            raise FileNotFoundError(f"Real data CSV not found: {data_path}")
        
        df_real = pd.read_csv(data_path)
        logger.info(f"Loaded real data: {len(df_real)} samples")
        
        # Clean filename column
        identifier_col = model_config['data']['identifier_col']
        df_real = clean_filename_column(df_real, identifier_col)
        
        # Preprocess static features
        static_features_list = model_config['data']['static_features']
        target_col = model_config['data']['target']
        
        logger.info("Preprocessing static features...")
        X_static_raw = df_real[static_features_list].values.astype(np.float32)
        
        # Handle missing values
        if imputer is not None:
            X_static_imputed = imputer.transform(X_static_raw)
        else:
            X_static_imputed = np.nan_to_num(X_static_raw, nan=0.0)
        
        # Scale features
        df_real[static_features_list] = scaler.transform(X_static_imputed)
        
        # Prepare data splits
        train_df, val_df = prepare_data_splits(df_real, config)
        
        # Create datasets
        npy_folder = Path(config.get('paths.real_npy_folder'))
        sequence_length = model_config['data']['sequence_length']
        vis_feat_dim = model_config['model']['vis_feat_dim']
        
        logger.info("Creating datasets...")
        train_dataset = EnhancedVFXDataset(
            df=train_df,
            npy_folder=npy_folder,
            static_features_list=static_features_list,
            target_col=target_col,
            identifier_col=identifier_col,
            sequence_length=sequence_length,
            vis_feat_dim=vis_feat_dim,
            use_augmentation=config.get('fine_tuning.use_augmentation', True),
            augmentation_strength=config.get('fine_tuning.augmentation_strength', 0.3)
        )
        
        val_dataset = EnhancedVFXDataset(
            df=val_df,
            npy_folder=npy_folder,
            static_features_list=static_features_list,
            target_col=target_col,
            identifier_col=identifier_col,
            sequence_length=sequence_length,
            vis_feat_dim=vis_feat_dim,
            use_augmentation=False  # No augmentation for validation
        )
        
        # Create data loaders
        batch_size = config.get('fine_tuning.batch_size', 16)
        num_workers = 0 if os.name == 'nt' else min(4, os.cpu_count())
        
        # Optional balanced sampling
        train_sampler = None
        if config.get('fine_tuning.use_class_balancing', True):
            train_sampler = create_balanced_sampler(train_dataset, target_col)
            logger.info("Using balanced sampling for training")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=enhanced_collate_fn,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=enhanced_collate_fn,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Initialize trainer
        trainer = FinetuningTrainer(config, model, device)
        
        # Start training with optional wandb logging
        with optional_wandb_logging(config) as wandb_logger:
            logger.info("🚀 Starting fine-tuning...")
            
            results = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=output_dir,
                wandb_logger=wandb_logger
            )
        
        # Final evaluation
        logger.info("\n" + "="*60)
        logger.info("FINAL EVALUATION")
        logger.info("="*60)
        
        best_model_path = output_dir / "best_fine_tuned_model.pt"
        if best_model_path.exists():
            # Load best checkpoint
            best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(best_checkpoint['state_dict'])
            model.eval()
            
            # Final validation
            final_metrics = MetricsTracker(model_config['data']['class_names'])
            with torch.no_grad():
                all_targets = []
                all_preds = []
                for batch_data in tqdm(val_loader, desc="Final Evaluation"):
                    if batch_data is None:
                        continue
                        
                    vis_seq, static_feat, targets = batch_data
                    vis_seq = vis_seq.to(device)
                    static_feat = static_feat.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(vis_seq, static_feat)
                    _, predicted = torch.max(outputs['logits'], 1)
                    
                    all_targets.extend(targets.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
                
                # Calculate comprehensive metrics
                metrics = calculate_comprehensive_metrics(
                    all_targets, all_preds, 
                    model_config['data']['class_names']
                )
                
                # Print detailed results
                logger.info("\nFinal Validation Results:")
                for metric, value in metrics.items():
                    if isinstance(value, (np.ndarray, list)):
                        logger.info(f"  {metric}:\n{value}")
                    else:
                        logger.info(f"  {metric}: {value:.4f}")
                
                logger.info("\nDetailed Classification Report:")
                logger.info("\n" + final_metrics.get_report())
                
                # Create and save confusion matrix
                if config.get('logging.save_plots', True):
                    cm = final_metrics.get_confusion_matrix()
                    plots_dir = output_dir / 'plots'
                    plots_dir.mkdir(exist_ok=True)
                    
                    trainer.visualizer.plot_confusion_matrix(
                        cm, model_config['data']['class_names'], 
                        plots_dir / 'confusion_matrix.png'
                    )
                    logger.info("Confusion matrix saved")
                
                # Save final results
                results_summary = {
                    'training_results': results,
                    'final_validation_metrics': metrics,
                    'model_config': model_config,
                    'fine_tuning_config': config.config,
                    'class_names': model_config['data']['class_names']
                }
                
                with open(output_dir / 'fine_tuning_results.json', 'w') as f:
                    json.dump(results_summary, f, indent=2, default=str)
                
                logger.info(f"✅ Fine-tuning completed successfully!")
                logger.info(f"Best validation F1-macro: {results['best_val_f1_macro']:.4f}")
                logger.info(f"Best epoch: {results['best_epoch'] + 1}")
                logger.info(f"Total epochs: {results['total_epochs']}")
                logger.info(f"Results saved to: {output_dir}")
                
        else:
            logger.error("❌ No best model checkpoint found!")
            
    except Exception as e:
        logger.error(f"❌ Error during fine-tuning: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Fine-tuning process completed")

def run_cross_validation(config: Config, model_class, model_config, df_real, 
                        npy_folder, static_features_list, target_col, identifier_col):
    """Run k-fold cross-validation for more robust evaluation."""
    logger.info("Starting cross-validation...")
    
    cv_folds = config.get('fine_tuning.cv_folds', 5)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_real, df_real[target_col])):
        logger.info(f"\n{'='*20} Fold {fold + 1}/{cv_folds} {'='*20}")
        
        # Split data
        train_df = df_real.iloc[train_idx].reset_index(drop=True)
        val_df = df_real.iloc[val_idx].reset_index(drop=True)
        
        # Create datasets
        train_dataset = EnhancedVFXDataset(
            df=train_df, npy_folder=npy_folder,
            static_features_list=static_features_list,
            target_col=target_col, identifier_col=identifier_col,
            sequence_length=model_config['data']['sequence_length'],
            vis_feat_dim=model_config['model']['vis_feat_dim'],
            use_augmentation=True
        )
        
        val_dataset = EnhancedVFXDataset(
            df=val_df, npy_folder=npy_folder,
            static_features_list=static_features_list,
            target_col=target_col, identifier_col=identifier_col,
            sequence_length=model_config['data']['sequence_length'],
            vis_feat_dim=model_config['model']['vis_feat_dim'],
            use_augmentation=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=config.get('fine_tuning.batch_size'),
            shuffle=True, collate_fn=enhanced_collate_fn, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.get('fine_tuning.batch_size'),
            shuffle=False, collate_fn=enhanced_collate_fn, num_workers=0
        )
        
        # Initialize fresh model for each fold
        model = model_class(config=model_config).to(device)
        
        # Setup criterion
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(train_df[target_col].values),
            y=train_df[target_col].values
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
        )
        
        # Create temporary output directory for this fold
        fold_output_dir = Path(config.get('paths.output_dir')) / f'fold_{fold + 1}'
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Train on this fold
        trainer = FinetuningTrainer(config, model, device)
        fold_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=fold_output_dir
        )
        
        cv_results.append({
            'fold': fold + 1,
            'best_val_f1_macro': fold_results['best_val_f1_macro'],
            'best_epoch': fold_results['best_epoch'],
            'total_epochs': fold_results['total_epochs']
        })
        
        logger.info(f"Fold {fold + 1} completed - Best F1-Macro: {fold_results['best_val_f1_macro']:.4f}")
    
    # Summarize cross-validation results
    cv_scores = [result['best_val_f1_macro'] for result in cv_results]
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    logger.info(f"\n{'='*50}")
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Mean F1-Macro: {mean_score:.4f} ± {std_score:.4f}")
    logger.info(f"Individual fold scores: {[f'{score:.4f}' for score in cv_scores]}")
    
    # Save CV results
    cv_summary = {
        'cv_results': cv_results,
        'mean_f1_macro': mean_score,
        'std_f1_macro': std_score,
        'individual_scores': cv_scores
    }
    
    cv_results_path = Path(config.get('paths.output_dir')) / 'cross_validation_results.json'
    with open(cv_results_path, 'w') as f:
        json.dump(cv_summary, f, indent=2, default=str)
    
    return cv_summary

def validate_environment():
    """Validate the environment and dependencies."""
    logger.info("Validating environment...")
    
    # Check PyTorch version
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.warning("CUDA not available - using CPU")
    
    # Check memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 4:
            logger.warning("Low GPU memory detected - consider reducing batch size")
    
    # Check required packages
    required_packages = ['pandas', 'numpy', 'sklearn', 'tqdm', 'yaml', 'joblib', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        raise ImportError(f"Please install missing packages: {missing_packages}")
    
    logger.info("✅ Environment validation passed")

if __name__ == "__main__":
    # Validate environment first
    try:
        validate_environment()
    except Exception as e:
        print(f"Environment validation failed: {e}")
        sys.exit(1)
    
    # Parse command line arguments (optional)
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced VFX Difficulty Fine-tuning')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, help='Output directory override')
    parser.add_argument('--batch-size', type=int, help='Batch size override')
    parser.add_argument('--learning-rate', type=float, help='Learning rate override')
    parser.add_argument('--epochs', type=int, help='Number of epochs override')
    parser.add_argument('--strategy', choices=['aggressive', 'progressive', 'full'], 
                       help='Freezing strategy override')
    parser.add_argument('--cross-validation', action='store_true', 
                       help='Run cross-validation instead of single train/val split')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Load configuration with overrides
    config_path = Path(args.config) if args.config else None
    config = Config(config_path)
    
    # Apply command line overrides
    if args.output_dir:
        config.config['paths']['output_dir'] = args.output_dir
    if args.batch_size:
        config.config['fine_tuning']['batch_size'] = args.batch_size
    if args.learning_rate:
        config.config['fine_tuning']['learning_rate'] = args.learning_rate
    if args.epochs:
        config.config['fine_tuning']['num_epochs'] = args.epochs
    if args.strategy:
        config.config['fine_tuning']['strategy'] = args.strategy
    if args.wandb:
        config.config['fine_tuning']['wandb_logging'] = True
    if args.debug:
        config.config['logging']['level'] = 'DEBUG'
    
    # Run cross-validation or regular training
    if args.cross_validation:
        # This would require additional setup - placeholder for now
        logger.info("Cross-validation mode selected")
        config.config['fine_tuning']['cross_validation'] = True
    
    # Run main training
    main()