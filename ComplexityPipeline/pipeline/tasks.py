# pipeline/tasks.py

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
from prefect import get_run_logger, task

# Import the helper classes we'll need
from .utils import ConfigManager, PipelineContext

# Import the core analysis functions from complexity models
from complexity_models.blur import analyze_camera_motion_blur
from complexity_models.camera_motion_test import analyze_camera_motion
from complexity_models.distortion_test import calculate_distortion_score
from complexity_models.focus_pull_test import analyze_focus_pull
from complexity_models.lightscore_test import analyze_light_score
from complexity_models.noise_test import analyze_video_noise
from complexity_models.overlap_test import analyze_overlap_complexity
from complexity_models.parallax_test import calculate_parallax_score
from complexity_models.zoom import analyze_zoom

logger = logging.getLogger(__name__)


class ComplexityMetric(Enum):
    """Enum for complexity metrics to ensure type safety and consistency."""
    ZOOM = "zoom_score"
    BLUR = "blur_score"
    DISTORTION = "distortion_score"
    MOTION = "motion_score"
    LIGHT = "light_score"
    NOISE = "noise_score"
    OVERLAP = "overlap_score"
    PARALLAX = "parallax_score"
    FOCUS_PULL = "focus_pull_score"


@dataclass
class AnalysisTask:
    """Configuration for a complexity analysis task."""
    name: str
    func: Callable
    args: List[Any]
    kwargs: Dict[str, Any]
    metric: ComplexityMetric
    default_value: float
    result_extractor: Callable[[Any], float]


class ComplexityAnalyzer:
    """Handles the orchestration of complexity analysis tasks."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.model_paths = config.get('complexity_model_paths', {})
        self.logger = get_run_logger()
    
    def _create_analysis_tasks(self, video_path: str, context: PipelineContext) -> List[AnalysisTask]:
        """Create the list of analysis tasks to run."""
        temp_outputs_dir = Path("temp_outputs")
        temp_outputs_dir.mkdir(exist_ok=True)
        
        return [
            AnalysisTask(
                name="zoom_analysis",
                func=analyze_zoom,
                args=[video_path],
                kwargs={},
                metric=ComplexityMetric.ZOOM,
                default_value=1.0,
                result_extractor=lambda result: result.get('avg_zoom', 1.0) if result else 1.0
            ),
            AnalysisTask(
                name="blur_analysis",
                func=analyze_camera_motion_blur,
                args=[video_path],
                kwargs={'model_path': self.model_paths.get('blur')},
                metric=ComplexityMetric.BLUR,
                default_value=0.0,
                result_extractor=lambda result: result[0] if result else 0.0
            ),
            AnalysisTask(
                name="distortion_analysis",
                func=calculate_distortion_score,
                args=[video_path],
                kwargs={},
                metric=ComplexityMetric.DISTORTION,
                default_value=0.0,
                result_extractor=lambda result: result.get('norm_distortion', 0.0) if result else 0.0
            ),
            AnalysisTask(
                name="motion_analysis",
                func=analyze_camera_motion,
                args=[video_path],
                kwargs={},
                metric=ComplexityMetric.MOTION,
                default_value=0.0,
                result_extractor=lambda result: result.get('avg_motion', 0.0) if result else 0.0
            ),
            AnalysisTask(
                name="light_analysis",
                func=analyze_light_score,
                args=[video_path],
                kwargs={},
                metric=ComplexityMetric.LIGHT,
                default_value=0.5,
                result_extractor=lambda result: result.get('avg_light_score', 0.5) if result else 0.5
            ),
            AnalysisTask(
                name="noise_analysis",
                func=analyze_video_noise,
                args=[video_path],
                kwargs={'model_path': self.model_paths.get('noise')},
                metric=ComplexityMetric.NOISE,
                default_value=0.0,
                result_extractor=lambda result: result[0] if result else 0.0
            ),
            AnalysisTask(
                name="overlap_analysis",
                func=analyze_overlap_complexity,
                args=[video_path],
                kwargs={'output_path': temp_outputs_dir / f"{context.file_path.stem}_overlap.mp4"},
                metric=ComplexityMetric.OVERLAP,
                default_value=0.0,
                result_extractor=lambda result: result.get('overlap_complexity_score', 0.0) if result else 0.0
            ),
            AnalysisTask(
                name="parallax_analysis",
                func=calculate_parallax_score,
                args=[video_path],
                kwargs={},
                metric=ComplexityMetric.PARALLAX,
                default_value=0.0,
                result_extractor=lambda result: result if result is not None else 0.0
            ),
            AnalysisTask(
                name="focus_pull_analysis",
                func=analyze_focus_pull,
                args=[video_path],
                kwargs={'model_path': self.model_paths.get('focus_pull')},
                metric=ComplexityMetric.FOCUS_PULL,
                default_value=0.0,
                result_extractor=lambda result: result[0] if result else 0.0
            )
        ]
    
    async def _run_single_analysis(self, task: AnalysisTask) -> Tuple[ComplexityMetric, float]:
        """Run a single analysis task with error handling."""
        self.logger.info(f"Starting analysis: {task.name}")
        try:
            # Filter out None values from kwargs
            clean_kwargs = {k: v for k, v in task.kwargs.items() if v is not None}
            
            result = await asyncio.to_thread(task.func, *task.args, **clean_kwargs)
            score = task.result_extractor(result)
            
            # Validate the result
            if not isinstance(score, (int, float)) or np.isnan(score) or np.isinf(score):
                self.logger.warning(f"Invalid score from {task.name}: {score}, using default {task.default_value}")
                score = task.default_value
            
            self.logger.info(f"Completed analysis: {task.name} -> {score}")
            return task.metric, float(score)
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {task.name} - {e}", exc_info=True)
            return task.metric, task.default_value
    
    async def run_all_analyses(self, video_path: str, context: PipelineContext) -> Dict[str, float]:
        """Run all complexity analyses concurrently."""
        self.logger.info(f"Starting all complexity analyses for {context.file_path.name}")
        
        tasks = self._create_analysis_tasks(video_path, context)
        
        # Run all tasks concurrently
        results = await asyncio.gather(*[
            self._run_single_analysis(task) for task in tasks
        ])
        
        # Convert results to dictionary
        complexity_scores = {metric.value: score for metric, score in results}
        
        self.logger.info(f"Completed complexity analysis. Scores: {complexity_scores}")
        return complexity_scores


class FeatureExtractor:
    """Handles feature extraction from videos."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = get_run_logger()
    
    async def extract_features(self, video_path: Path) -> Path:
        """Extract features from video and return the path to the .npy file."""
        features_dir = Path(self.config.get("multimodal_model.features_dir"))
        features_dir.mkdir(exist_ok=True, parents=True)
        
        cmd = [
            "python", "extract_features.py",
            "--video", str(video_path),
            "--output_dir", str(features_dir)
        ]
        
        self.logger.info(f"Starting feature extraction: {' '.join(cmd)}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Feature extraction failed: {stderr.decode()}")
            
            feature_path = features_dir / f"{video_path.stem}.npy"
            if not feature_path.exists():
                raise FileNotFoundError(f"Expected feature file not created: {feature_path}")
            
            self.logger.info(f"Feature extraction completed: {feature_path}")
            return feature_path
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {video_path.name}: {e}")
            raise


class ModelPredictor:
    """Handles model prediction and result formatting."""
    
    def __init__(self, config: ConfigManager, model, scaler, imputer):
        self.config = config
        self.model = model
        self.scaler = scaler
        self.imputer = imputer
        self.model_config = config.get("multimodal_model")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = get_run_logger()
    
    def _prepare_static_features(self, complexity_scores: Dict[str, float], sequence_features: np.ndarray) -> np.ndarray:
        """Prepare static features for model input."""
        static_feature_names = self.model_config['static_features']
        
        # Get the 9 complexity scores
        static_data = [complexity_scores.get(name, 0.0) for name in static_feature_names[:-1]]
        
        # Handle sequence feature dimensionality adaptation
        expected_feat_dim = self.model_config.get('vis_feat_dim', sequence_features.shape[-1])
        if sequence_features.shape[-1] != expected_feat_dim:
            sequence_features = self._adapt_sequence_features(sequence_features, expected_feat_dim)
        
        # Add the sequence feature (average of all sequence features)
        sequence_feature = np.mean(sequence_features)
        static_data.append(sequence_feature)
        
        static_features = np.array(static_data).reshape(1, -1).astype(np.float32)
        self.logger.info(f"Static features (raw): {static_features.tolist()}")
        
        return static_features
    
    def _adapt_sequence_features(self, sequence_features: np.ndarray, expected_dim: int) -> np.ndarray:
        """Adapt sequence features to expected dimensionality."""
        current_dim = sequence_features.shape[-1]
        
        if expected_dim % current_dim == 0:
            tile_factor = expected_dim // current_dim
            self.logger.warning(
                f"Adapting sequence feature dim {current_dim} to {expected_dim} by tiling {tile_factor}x"
            )
            return np.tile(sequence_features, tile_factor)
        else:
            raise ValueError(
                f"Cannot adapt sequence feature dim {current_dim} to expected {expected_dim}"
            )
    
    def _validate_features(self, features: np.ndarray, feature_type: str) -> None:
        """Validate feature arrays for common issues."""
        if np.isnan(features).any() or np.isinf(features).any():
            raise ValueError(f"{feature_type} features contain NaN or infinity values")
        
        if features.size == 0:
            raise ValueError(f"{feature_type} features are empty")
    
    def _preprocess_features(self, static_features: np.ndarray) -> np.ndarray:
        """Apply imputation and scaling to static features."""
        self._validate_features(static_features, "Raw static")
        
        if self.imputer:
            static_features = self.imputer.transform(static_features)
            self.logger.info(f"Static features after imputation: {static_features.tolist()}")
            self._validate_features(static_features, "Imputed static")
        
        if self.scaler:
            static_features = self.scaler.transform(static_features)
            self.logger.info(f"Static features after scaling: {static_features.tolist()}")
            self._validate_features(static_features, "Scaled static")
        
        return static_features
    
    def _apply_temperature_scaling(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        temperature = float(self.model_config.get('temperature', 5.0))
        return torch.softmax(logits / temperature, dim=1)
    
    def _apply_class_balancing(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply class-prior balancing if configured."""
        priors = self.model_config.get('class_priors')
        if priors and len(priors) == len(probabilities):
            balanced = probabilities / np.array(priors, dtype=np.float32)
            probabilities = balanced / balanced.sum()
            self.logger.info(f"Applied class-prior balancing with priors: {priors}")
        return probabilities
    
    def _apply_easy_tiebreaker(self, probabilities: np.ndarray) -> np.ndarray:
        """Gentle tie-breaker: slightly boost Easy when it's very close to Medium.
        Only activates when Easy and Medium probabilities are within 8 percentage points.
        Configured via 'easy_boost_threshold' in config (default: 0.08).
        """
        boost_threshold = self.model_config.get('easy_boost_threshold', 0.08)
        class_names = self.model_config['class_names']
        
        # Find Easy and Medium indices
        try:
            easy_idx = class_names.index('Easy')
            medium_idx = class_names.index('Medium')
        except ValueError:
            return probabilities  # Class names don't match expected format
        
        easy_prob = probabilities[easy_idx]
        medium_prob = probabilities[medium_idx]
        
        # Only boost if Easy and Medium are close (within threshold)
        if abs(easy_prob - medium_prob) <= boost_threshold:
            # Gentle boost: increase Easy by 15% of the difference
            boost_factor = 0.15
            if medium_prob > easy_prob:
                diff = medium_prob - easy_prob
                probabilities[easy_idx] += diff * boost_factor
                probabilities[medium_idx] -= diff * boost_factor
                
                # Renormalize to ensure probabilities sum to 1
        
        return probabilities
    
    def _apply_rule_based_easy_detection(self, probabilities: np.ndarray, static_features: np.ndarray) -> np.ndarray:
        """Rule-based Easy detection: Override to Easy if static features strongly indicate Easy.
        
        Easy conditions (most must be met):
        - zoom_score < 0.3 (low zoom)
        - blur_score < 0.3 (low blur)
        - motion_score < 0.3 (low motion)
        - noise_score < 0.3 (low noise)
        - distortion_score < 0.3 (low distortion)
        - overlap_score < 0.4 (low overlap)
        - light_score < 0.4 (good lighting)
        
        Only overrides if model's Easy prediction is < 0.4 (model is uncertain about Easy).
        """
        # Get feature names from config
        feature_names = self.model_config.get('static_features', [])
        if len(feature_names) != len(static_features[0]):  # static_features is 2D
            return probabilities  # Feature mismatch, skip rule-based detection
        
        # Create feature dict for easy access (flatten static_features)
        features = {name: static_features[0][i] for i, name in enumerate(feature_names)}
        
        # Define Easy conditions - more aggressive thresholds
        easy_conditions = {
            'zoom_score': 0.4,      # Increased from 0.3
            'blur_score': 0.4,      # Increased from 0.3
            'motion_score': 0.4,    # Increased from 0.3
            'noise_score': 0.4,     # Increased from 0.3
            'distortion_score': 0.4, # Increased from 0.3
            'overlap_score': 0.5,   # Increased from 0.4
            'light_score': 0.5      # Increased from 0.4
        }
        
        # Check if all Easy conditions are met
        conditions_met = []
        self.logger.info(f"🔍 EASY DETECTION DEBUG: Checking {len(easy_conditions)} conditions")
        for feature, threshold in easy_conditions.items():
            if feature in features:
                value = features[feature]
                is_low = value < threshold
                conditions_met.append(is_low)
                status = "✓ PASS" if is_low else "✗ FAIL"
                self.logger.info(f"  {feature}: {value:.4f} < {threshold} → {status}")
            else:
                self.logger.warning(f"  {feature}: NOT FOUND in features")
                
        # Only override if:
        # 1. Most Easy conditions are met (at least 5 out of 7)
        # 2. Model's Easy prediction is low (< 0.4)
        class_names = self.model_config['class_names']
        try:
            easy_idx = class_names.index('Easy')
        except ValueError:
            return probabilities
            
        easy_conditions_count = sum(conditions_met)
        total_conditions = len(conditions_met)
        model_easy_prob = probabilities[easy_idx]
        
        # Enhanced override conditions - more aggressive:
        # Option 1: Strong Easy signal (4/7 conditions + low model confidence)
        # Option 2: Moderate Easy signal (6/7 conditions + higher model confidence)
        # Option 3: All conditions met (override regardless of model confidence)
        
        strong_easy_signal = (easy_conditions_count >= 4 and model_easy_prob < 0.3)
        moderate_easy_signal = (easy_conditions_count >= 6 and model_easy_prob < 0.5)
        overwhelming_easy_signal = (easy_conditions_count >= 7)
        
        self.logger.info(
            f"🔍 EASY DETECTION SUMMARY: {easy_conditions_count}/{total_conditions} conditions met "
            f"({easy_conditions_count/total_conditions*100:.1f}%), model_easy_prob={model_easy_prob:.3f}"
        )
        self.logger.info(
            f"  Signals: strong={strong_easy_signal}, moderate={moderate_easy_signal}, overwhelming={overwhelming_easy_signal}"
        )
        
        if (total_conditions >= 5 and 
            (strong_easy_signal or moderate_easy_signal or overwhelming_easy_signal)):
            
            # Strong override: Set Easy to 0.7, distribute rest between Medium/Hard
            new_probabilities = probabilities.copy()
            new_probabilities[easy_idx] = 0.7
            
            # Distribute remaining 0.3 between other classes proportionally
            remaining_prob = 0.3
            other_indices = [i for i in range(len(probabilities)) if i != easy_idx]
            other_total = sum(probabilities[i] for i in other_indices)
            
            if other_total > 0:
                for i in other_indices:
                    new_probabilities[i] = remaining_prob * (probabilities[i] / other_total)
            else:
                # Fallback: equal distribution
                for i in other_indices:
                    new_probabilities[i] = remaining_prob / len(other_indices)
            
            # Determine override reason
            override_reason = "unknown"
            if overwhelming_easy_signal:
                override_reason = "overwhelming_signal"
            elif moderate_easy_signal:
                override_reason = "moderate_signal"
            elif strong_easy_signal:
                override_reason = "strong_signal"
            
            self.logger.info(
                f"Rule-based Easy override ({override_reason}): {easy_conditions_count}/{total_conditions} conditions met. "
                f"Easy: {model_easy_prob:.3f}→{new_probabilities[easy_idx]:.3f}"
            )
            
            # Log which conditions were met
            met_conditions = [feat for feat, thresh in easy_conditions.items() 
                            if feat in features and features[feat] < thresh]
            self.logger.info(f"Easy conditions met: {met_conditions}")
            
            return new_probabilities
        
        return probabilities
    
    def predict(self, complexity_scores: Dict[str, float], sequence_features: np.ndarray) -> Dict[str, Any]:
        """Make prediction using the multimodal model."""
        # Prepare features
        static_features = self._prepare_static_features(complexity_scores, sequence_features)
        static_features = self._preprocess_features(static_features)
        
        # Run inference
        with torch.no_grad():
            seq_tensor = torch.tensor(sequence_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            static_tensor = torch.tensor(static_features, dtype=torch.float32).to(self.device)
            
            model_output = self.model(seq_tensor, static_tensor)
            
            # Extract logits from model output
            if isinstance(model_output, dict):
                logits = model_output.get('logits', model_output.get('prediction'))
                if logits is None:
                    logits = next(iter(model_output.values()))
            else:
                logits = model_output
            
            # Apply temperature scaling and class balancing
            probabilities = self._apply_temperature_scaling(logits).cpu().numpy()[0]
            probabilities = self._apply_class_balancing(probabilities)
            probabilities = self._apply_easy_tiebreaker(probabilities)
            probabilities = self._apply_rule_based_easy_detection(probabilities, static_features)
            
            self.logger.info(f"Final probabilities: {probabilities.tolist()}")
        
        # Format result
        class_names = self.model_config['class_names']
        pred_index = np.argmax(probabilities)
        
        return {
            'predicted_index': int(pred_index),
            'predicted_class': class_names[pred_index],
            'confidence': f"{probabilities[pred_index]:.2%}",
            'probabilities': {name: f"{prob:.2%}" for name, prob in zip(class_names, probabilities)}
        }


# --- Prefect Tasks ---

@task(name="Analyze Complexity Scores", retries=2)
async def analyze_complexity_task(context: PipelineContext, config: ConfigManager) -> PipelineContext:
    """Orchestrate the execution of all complexity analysis scripts."""
    analyzer = ComplexityAnalyzer(config)
    video_path_str = str(context.file_path)
    
    context.complexity_scores = await analyzer.run_all_analyses(video_path_str, context)
    return context


@task(name="Extract Sequence Features", retries=2)
async def extract_sequence_features_task(context: PipelineContext, config: ConfigManager) -> PipelineContext:
    """Extract sequence features from the video."""
    extractor = FeatureExtractor(config)
    feature_path = await extractor.extract_features(context.file_path)
    
    # Store the feature path for the next task
    context.feature_path = feature_path
    return context


@task(name="Classify Difficulty", retries=1)
async def classify_difficulty_task(
    context: PipelineContext, 
    config: ConfigManager, 
    model, 
    scaler, 
    imputer
) -> PipelineContext:
    """Perform final classification using the multimodal model."""
    logger = get_run_logger()
    
    # Load sequence features
    feature_path = getattr(context, 'feature_path', None)
    if not feature_path:
        # Fallback to constructing path from config
        model_config = config.get("multimodal_model")
        feature_path = Path(model_config['features_dir']) / f"{context.file_path.stem}.npy"
    
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    
    sequence_features = np.load(feature_path).astype(np.float32)
    logger.info(f"Loaded sequence features from {feature_path}, shape: {sequence_features.shape}")
    
    # Make prediction
    predictor = ModelPredictor(config, model, scaler, imputer)
    context.final_result = predictor.predict(context.complexity_scores, sequence_features)
    
    logger.info(f"Classification result: {context.final_result}")
    return context