import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import logging
import argparse
import json
import yaml
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import sys
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# --- Configure Logging ---
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration with optional file output."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)

# --- Configuration Classes ---
@dataclass
class ExtractorConfig:
    """Configuration for the feature extractor."""
    model_type: str = "resnet50"
    pretrained: bool = True
    sequence_length: int = 32
    target_vis_feat_dim: int = 2048
    frame_sample_strategy: str = 'uniform'
    padding_strategy: str = 'zeros'
    truncation_strategy: str = 'last'
    batch_size: int = 8  # Process frames in batches
    device: Optional[str] = None
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'ExtractorConfig':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_path: Union[str, Path]):
        """Save configuration to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

# --- Global Constants ---
SUPPORTED_VIDEO_FORMATS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
MODEL_FEATURE_DIMS = {
    'resnet50': 2048,
    'resnet18': 512,
    'resnet34': 512,
    'resnet101': 2048,
    'resnet152': 2048,
    'efficientnet_b0': 1280,
    'efficientnet_b1': 1280,
    'efficientnet_b2': 1408,
    'efficientnet_b3': 1536,
    'efficientnet_b4': 1792,
    'mobilenet_v3_large': 960,
    'mobilenet_v3_small': 576,
    'vit_b_16': 768,
    'vit_b_32': 768,
}

# --- Enhanced Video Feature Extractor Model ---
class VideoFeatureExtractor(nn.Module):
    """
    Enhanced CNN-based feature extractor for video frames.
    Supports multiple architectures and includes batch processing.
    """
    def __init__(self, model_type: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.model_type = model_type.lower()
        
        # Get model and feature dimension
        if self.model_type not in MODEL_FEATURE_DIMS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported models: {list(MODEL_FEATURE_DIMS.keys())}")
        
        self.feature_dim = MODEL_FEATURE_DIMS[self.model_type]
        base_model = self._get_model(pretrained)
        
        # Extract features based on model type
        if self.model_type.startswith("resnet"):
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.model_type.startswith("efficientnet"):
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.pool = nn.Identity()
        elif self.model_type.startswith("mobilenet"):
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.model_type.startswith("vit"):
            # For Vision Transformers, we need special handling
            self.features = base_model
            self.pool = nn.Identity()
        else:
            # Generic approach
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            
        logging.info(f"Initialized {model_type} feature extractor (pretrained={pretrained}) "
                    f"with output dim {self.feature_dim}")

    def _get_model(self, pretrained: bool):
        """Get the appropriate model based on model_type."""
        if self.model_type == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            return models.resnet18(weights=weights)
        elif self.model_type == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            return models.resnet34(weights=weights)
        elif self.model_type == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            return models.resnet50(weights=weights)
        elif self.model_type == "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            return models.resnet101(weights=weights)
        elif self.model_type == "resnet152":
            weights = models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
            return models.resnet152(weights=weights)
        elif self.model_type == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            return models.efficientnet_b0(weights=weights)
        elif self.model_type == "efficientnet_b1":
            weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            return models.efficientnet_b1(weights=weights)
        elif self.model_type == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            return models.mobilenet_v3_large(weights=weights)
        elif self.model_type == "vit_b_16":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            return models.vit_b_16(weights=weights)
        else:
            raise ValueError(f"Model {self.model_type} not implemented in _get_model")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type.startswith("vit"):
            # Special handling for Vision Transformers
            features = self.features(x)
        else:
            features_map = self.features(x)
            pooled_features = self.pool(features_map)
            features = pooled_features.view(pooled_features.size(0), -1)
        return features

# --- Enhanced Video Processing Logic ---
class VideoProcessor:
    """
    Enhanced video processor with batch processing and better error handling.
    """
    def __init__(self, config: ExtractorConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Setup device
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize extractor
        self.extractor = VideoFeatureExtractor(
            model_type=config.model_type, 
            pretrained=config.pretrained
        ).to(self.device)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.logger.info(f"VideoProcessor initialized on {self.device}. "
                        f"Sequence Length: {config.sequence_length}, "
                        f"Target Feature Dim: {config.target_vis_feat_dim}")

    def _get_frame_indices(self, total_frames: int) -> np.ndarray:
        """Get frame indices based on sampling strategy."""
        if total_frames <= self.config.sequence_length:
            return np.arange(total_frames)
        
        strategy = self.config.frame_sample_strategy
        if strategy == 'uniform':
            return np.linspace(0, total_frames - 1, self.config.sequence_length, dtype=int)
        elif strategy == 'first':
            return np.arange(self.config.sequence_length)
        elif strategy == 'last':
            return np.arange(total_frames - self.config.sequence_length, total_frames)
        elif strategy == "random":
            indices = np.random.choice(total_frames, self.config.sequence_length, replace=False)
            return np.sort(indices)
        else:
            self.logger.warning(f"Unknown frame sampling strategy: {strategy}. Using 'uniform'.")
            return np.linspace(0, total_frames - 1, self.config.sequence_length, dtype=int)

    def _process_sequence(self, frames_list: List[torch.Tensor]) -> torch.Tensor:
        """Process frame sequence with padding/truncation."""
        n_frames = len(frames_list)
        
        if n_frames == self.config.sequence_length:
            return torch.stack(frames_list)
        
        if n_frames < self.config.sequence_length:
            # Padding
            pad_width = self.config.sequence_length - n_frames
            padding_tensor = torch.zeros(3, 224, 224)
            
            if self.config.padding_strategy == "zeros":
                padding_frames = [padding_tensor] * pad_width
            elif self.config.padding_strategy == "repeat" and n_frames > 0:
                padding_frames = [frames_list[-1]] * pad_width
            else:
                padding_frames = [padding_tensor] * pad_width
            
            return torch.stack(frames_list + padding_frames)
        else:
            # Truncation
            if self.config.truncation_strategy == "first":
                start_idx = 0
            elif self.config.truncation_strategy == "last":
                start_idx = n_frames - self.config.sequence_length
            elif self.config.truncation_strategy == "random":
                start_idx = np.random.randint(0, n_frames - self.config.sequence_length + 1)
            else:
                start_idx = n_frames - self.config.sequence_length
            
            return torch.stack(frames_list[start_idx:start_idx + self.config.sequence_length])

    def _extract_frames_batch(self, frames_tensor: torch.Tensor) -> np.ndarray:
        """Extract features from frames using batch processing."""
        self.extractor.eval()
        all_features = []
        
        with torch.no_grad():
            for i in range(0, frames_tensor.size(0), self.config.batch_size):
                batch = frames_tensor[i:i + self.config.batch_size].to(self.device)
                features = self.extractor(batch)
                all_features.append(features.cpu())
        
        return torch.cat(all_features, dim=0).numpy()

    def extract_features_from_video(self, video_path: Path) -> Optional[np.ndarray]:
        """Extract features from a single video file."""
        self.logger.info(f"Processing video: {video_path.name}")
        
        # Validate video file
        if not video_path.exists():
            self.logger.error(f"Video file not found: {video_path}")
            return None
            
        if video_path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
            self.logger.error(f"Unsupported video format: {video_path.suffix}")
            return None
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.error(f"Failed to open video file: {video_path}")
            return None

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            self.logger.info(f"Video info - Frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
            
            if total_frames == 0:
                self.logger.warning(f"Video {video_path.name} has 0 frames.")
                return None

            frame_indices = self._get_frame_indices(total_frames)
            frames_to_process = []
            
            # Extract frames
            for idx in tqdm(frame_indices, desc="Extracting frames", leave=False):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        processed_frame = self.transform(frame_rgb)
                        frames_to_process.append(processed_frame)
                    except Exception as e:
                        self.logger.warning(f"Failed to process frame {idx}: {e}")
                else:
                    self.logger.warning(f"Failed to read frame {idx}")
            
            if not frames_to_process:
                self.logger.error(f"No frames could be processed from {video_path.name}")
                return None
            
            # Process sequence
            frames_tensor = self._process_sequence(frames_to_process)
            
            # Extract features
            features = self._extract_frames_batch(frames_tensor)
            
            # Validate output dimensions
            if features.shape[1] != self.config.target_vis_feat_dim:
                self.logger.error(f"Feature dimension mismatch: expected {self.config.target_vis_feat_dim}, "
                                f"got {features.shape[1]}")
                return None

            self.logger.info(f"Extracted features shape: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {e}", exc_info=True)
            return None
        finally:
            cap.release()

    def process_multiple_videos(self, video_paths: List[Path], output_dir: Path, 
                              max_workers: int = 2) -> Dict[str, bool]:
        """Process multiple videos with optional parallel processing."""
        results = {}
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if max_workers == 1:
            # Sequential processing
            for video_path in tqdm(video_paths, desc="Processing videos"):
                success = self._process_single_video(video_path, output_dir)
                results[video_path.name] = success
        else:
            # Parallel processing (limited due to GPU memory)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_video = {
                    executor.submit(self._process_single_video, video_path, output_dir): video_path 
                    for video_path in video_paths
                }
                
                for future in tqdm(as_completed(future_to_video), 
                                 total=len(video_paths), desc="Processing videos"):
                    video_path = future_to_video[future]
                    try:
                        success = future.result()
                        results[video_path.name] = success
                    except Exception as e:
                        self.logger.error(f"Error processing {video_path.name}: {e}")
                        results[video_path.name] = False
        
        return results

    def _process_single_video(self, video_path: Path, output_dir: Path) -> bool:
        """Process a single video and save features."""
        features = self.extract_features_from_video(video_path)
        
        if features is not None:
            output_filename = video_path.stem + ".npy"
            output_filepath = output_dir / output_filename
            
            try:
                np.save(output_filepath, features)
                self.logger.info(f"Features saved: {output_filepath}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to save features for {video_path.name}: {e}")
                return False
        
        return False

# --- CLI Interface ---
def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description="Extract features from videos and save as NPY files")
    
    # Input/Output
    parser.add_argument("--input", "-i", type=str, 
                       help="Input video file or directory containing videos (optional if using hardcoded paths)")
    parser.add_argument("--output", "-o", type=str,
                       help="Output directory for NPY files (optional if using hardcoded paths)")
    parser.add_argument("--config", "-c", type=str,
                       help="Configuration JSON file")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="resnet50",
                       choices=list(MODEL_FEATURE_DIMS.keys()),
                       help="Model architecture to use")
    parser.add_argument("--no-pretrained", action="store_true",
                       help="Don't use pretrained weights")
    parser.add_argument("--sequence-length", type=int, default=32,
                       help="Number of frames in sequence")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for processing frames")
    
    # Sampling strategies
    parser.add_argument("--frame-strategy", type=str, default="uniform",
                       choices=["uniform", "first", "last", "random"],
                       help="Frame sampling strategy")
    parser.add_argument("--padding-strategy", type=str, default="zeros",
                       choices=["zeros", "repeat"],
                       help="Padding strategy for short videos")
    parser.add_argument("--truncation-strategy", type=str, default="last",
                       choices=["first", "last", "random"],
                       help="Truncation strategy for long videos")
    
    # Processing options
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of worker threads (use 1 for single GPU)")
    parser.add_argument("--device", type=str,
                       help="Device to use (cuda/cpu), auto-detect if not specified")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file", type=str,
                       help="Log file path")
    
    # Utility
    parser.add_argument("--save-config", type=str,
                       help="Save current configuration to JSON file")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be processed without actually processing")
    
    return parser

def main():
    # Setup logging
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description='Extract features from a single video or a directory.')
    
    # --- NEW: Add a --video argument for single file processing ---
    parser.add_argument('--video', type=str, help='Path to a single video file to process.')
    
    # --- Keep old arguments for batch processing, but make them not required ---
    parser.add_argument('--metadata', type=str, help='(Optional) Path to metadata CSV file.')
    parser.add_argument('--videos_dir', type=str, help='(Optional) Directory containing videos. Used if --video is not specified.')
    
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save features')
    parser.add_argument('--model', type=str, default='resnet50', 
                        choices=['resnet50', 'resnet101', 'efficientnet_b3'],
                        help='Model to use for feature extraction')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing frames')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of concurrent workers')
    parser.add_argument('--frame_limit', type=int, default=32, help='Maximum frames to extract per video')
    parser.add_argument('--sample_strategy', type=str, default='uniform', 
                        choices=['uniform', 'first', 'last'],
                        help='Strategy for sampling frames')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                if hasattr(args, key) and getattr(args, key) is None:
                    setattr(args, key, value)
    
    # --- NEW LOGIC: Decide whether to process a single video or a batch ---
    start_time = time.time()
    
    if args.video:
        # --- Single Video Mode ---
        logger.info(f"Processing single video: {args.video}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        config = ExtractorConfig(
            model_type=args.model,
            batch_size=args.batch_size,
            sequence_length=args.frame_limit,
            frame_sample_strategy=args.sample_strategy
        )
        extractor = VideoProcessor(config=config, logger=logger)
        result_path = extractor._process_single_video(Path(args.video), Path(args.output_dir))
        if result_path:
            logger.info(f"Successfully extracted features to {result_path}")
        else:
            logger.error(f"Failed to extract features for {args.video}")
            sys.exit(1) # Exit with an error code if it fails
            
    elif args.videos_dir:
        # --- Batch Mode ---
        logger.info(f"Processing batch of videos from directory: {args.videos_dir}")
        config = ExtractorConfig(
            model_type=args.model,
            batch_size=args.batch_size,
            sequence_length=args.frame_limit,
            frame_sample_strategy=args.sample_strategy
        )
        processor = VideoProcessor(config=config, logger=logger)
        
        video_paths = [Path(args.videos_dir) / f for f in os.listdir(args.videos_dir) 
                      if f.lower().endswith(tuple(SUPPORTED_VIDEO_FORMATS))]
        
        results = processor.process_multiple_videos(
            video_paths=video_paths,
            output_dir=Path(args.output_dir),
            max_workers=args.max_workers
        )
        
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Processed {len(results)} videos: {success_count} successful, {len(results) - success_count} failed")
    else:
        logger.error("Error: You must specify either --video (for a single file) or --videos_dir (for a batch).")
        parser.print_help()
        sys.exit(1)

    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()