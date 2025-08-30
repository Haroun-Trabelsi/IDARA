# pipeline/validation_tasks.py

from prefect import task, get_run_logger
from typing import Dict, Any, List
from pathlib import Path
import cv2
import numpy as np
from security.validation import VideoFileValidator, ComplexityScoreValidator, PredictionResultValidator
from monitoring.metrics import MetricsCollector

@task(name="validate_input_video", retries=1)
def validate_input_video(video_path: str, metrics_collector: MetricsCollector) -> Dict[str, Any]:
    """Validate input video file before processing."""
    logger = get_run_logger()
    
    try:
        # Basic file validation
        video_file = Path(video_path)
        if not video_file.exists():
            raise ValueError(f"Video file not found: {video_path}")
        
        # Get video metadata
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Create validation object
        video_metadata = {
            'filename': video_file.name,
            'file_size_mb': video_file.stat().st_size / (1024 * 1024),
            'duration_seconds': duration,
            'frame_count': frame_count,
            'fps': fps,
            'resolution': f"{width}x{height}",
            'format': video_file.suffix.lower()
        }
        
        # Validate using Pydantic
        validator = VideoFileValidator(**video_metadata)
        
        # Record metrics
        metrics_collector.record_validation_result("video_input", True)
        
        logger.info(f"Video validation passed: {video_file.name}")
        
        return {
            'valid': True,
            'metadata': video_metadata,
            'validation_errors': []
        }
        
    except Exception as e:
        logger.error(f"Video validation failed: {str(e)}")
        metrics_collector.record_validation_result("video_input", False)
        
        return {
            'valid': False,
            'metadata': {},
            'validation_errors': [str(e)]
        }

@task(name="validate_complexity_scores", retries=1)
def validate_complexity_scores(complexity_scores: Dict[str, float], 
                             metrics_collector: MetricsCollector) -> Dict[str, Any]:
    """Validate complexity analysis results."""
    logger = get_run_logger()
    
    try:
        # Validate using Pydantic
        validator = ComplexityScoreValidator(**complexity_scores)
        
        # Additional business logic validation
        validation_errors = []
        
        # Check for suspicious patterns
        if all(score == 0.0 for score in complexity_scores.values()):
            validation_errors.append("All complexity scores are zero - possible analysis failure")
        
        if any(score < 0 or score > 1 for score in complexity_scores.values()):
            validation_errors.append("Complexity scores outside valid range [0, 1]")
        
        # Check for missing required metrics
        required_metrics = ['blur', 'motion', 'distortion', 'light', 'noise', 
                          'overlap', 'parallax', 'focus_pull', 'zoom']
        missing_metrics = [metric for metric in required_metrics if metric not in complexity_scores]
        
        if missing_metrics:
            validation_errors.append(f"Missing complexity metrics: {missing_metrics}")
        
        is_valid = len(validation_errors) == 0
        
        # Record metrics
        metrics_collector.record_validation_result("complexity_scores", is_valid)
        
        if is_valid:
            logger.info("Complexity scores validation passed")
        else:
            logger.warning(f"Complexity scores validation issues: {validation_errors}")
        
        return {
            'valid': is_valid,
            'scores': complexity_scores,
            'validation_errors': validation_errors
        }
        
    except Exception as e:
        logger.error(f"Complexity scores validation failed: {str(e)}")
        metrics_collector.record_validation_result("complexity_scores", False)
        
        return {
            'valid': False,
            'scores': {},
            'validation_errors': [str(e)]
        }

@task(name="validate_prediction_result", retries=1)
def validate_prediction_result(prediction_result: Dict[str, Any], 
                             metrics_collector: MetricsCollector) -> Dict[str, Any]:
    """Validate final prediction results."""
    logger = get_run_logger()
    
    try:
        # Validate using Pydantic
        validator = PredictionResultValidator(**prediction_result)
        
        # Additional validation
        validation_errors = []
        
        # Check confidence scores
        if 'confidence_scores' in prediction_result:
            confidence_scores = prediction_result['confidence_scores']
            
            # Should sum to approximately 1.0
            total_confidence = sum(confidence_scores.values())
            if abs(total_confidence - 1.0) > 0.1:
                validation_errors.append(f"Confidence scores don't sum to 1.0: {total_confidence}")
            
            # All should be non-negative
            if any(score < 0 for score in confidence_scores.values()):
                validation_errors.append("Negative confidence scores detected")
        
        # Check predicted class
        if 'predicted_class' in prediction_result:
            valid_classes = ['easy', 'medium', 'hard']
            if prediction_result['predicted_class'] not in valid_classes:
                validation_errors.append(f"Invalid predicted class: {prediction_result['predicted_class']}")
        
        # Check processing metadata
        if 'processing_time_seconds' in prediction_result:
            processing_time = prediction_result['processing_time_seconds']
            if processing_time < 0 or processing_time > 3600:  # Max 1 hour
                validation_errors.append(f"Suspicious processing time: {processing_time}s")
        
        is_valid = len(validation_errors) == 0
        
        # Record metrics
        metrics_collector.record_validation_result("prediction_result", is_valid)
        
        if is_valid:
            logger.info("Prediction result validation passed")
        else:
            logger.warning(f"Prediction result validation issues: {validation_errors}")
        
        return {
            'valid': is_valid,
            'result': prediction_result,
            'validation_errors': validation_errors
        }
        
    except Exception as e:
        logger.error(f"Prediction result validation failed: {str(e)}")
        metrics_collector.record_validation_result("prediction_result", False)
        
        return {
            'valid': False,
            'result': {},
            'validation_errors': [str(e)]
        }

@task(name="validate_data_quality", retries=1)
def validate_data_quality(video_features: np.ndarray, 
                         static_features: np.ndarray,
                         metrics_collector: MetricsCollector) -> Dict[str, Any]:
    """Validate extracted features for data quality issues."""
    logger = get_run_logger()
    
    try:
        validation_errors = []
        
        # Check video features
        if video_features is not None:
            # Check for NaN or infinite values
            if np.any(np.isnan(video_features)):
                validation_errors.append("NaN values detected in video features")
            
            if np.any(np.isinf(video_features)):
                validation_errors.append("Infinite values detected in video features")
            
            # Check feature dimensions
            if video_features.ndim != 2:
                validation_errors.append(f"Invalid video features shape: {video_features.shape}")
            
            # Check for all-zero features (possible extraction failure)
            if np.all(video_features == 0):
                validation_errors.append("All video features are zero - possible extraction failure")
            
            # Check feature variance (low variance might indicate issues)
            feature_variance = np.var(video_features, axis=0)
            low_variance_features = np.sum(feature_variance < 1e-6)
            if low_variance_features > video_features.shape[1] * 0.5:
                validation_errors.append(f"Many features have low variance: {low_variance_features}")
        
        # Check static features
        if static_features is not None:
            if np.any(np.isnan(static_features)):
                validation_errors.append("NaN values detected in static features")
            
            if np.any(np.isinf(static_features)):
                validation_errors.append("Infinite values detected in static features")
            
            # Check expected range for complexity scores (0-1)
            if np.any(static_features < 0) or np.any(static_features > 1):
                validation_errors.append("Static features outside expected range [0, 1]")
        
        is_valid = len(validation_errors) == 0
        
        # Record metrics
        metrics_collector.record_validation_result("data_quality", is_valid)
        
        # Calculate quality metrics
        quality_metrics = {}
        if video_features is not None:
            quality_metrics['video_features_shape'] = video_features.shape
            quality_metrics['video_features_mean'] = float(np.mean(video_features))
            quality_metrics['video_features_std'] = float(np.std(video_features))
        
        if static_features is not None:
            quality_metrics['static_features_shape'] = static_features.shape
            quality_metrics['static_features_mean'] = float(np.mean(static_features))
            quality_metrics['static_features_std'] = float(np.std(static_features))
        
        if is_valid:
            logger.info("Data quality validation passed")
        else:
            logger.warning(f"Data quality validation issues: {validation_errors}")
        
        return {
            'valid': is_valid,
            'quality_metrics': quality_metrics,
            'validation_errors': validation_errors
        }
        
    except Exception as e:
        logger.error(f"Data quality validation failed: {str(e)}")
        metrics_collector.record_validation_result("data_quality", False)
        
        return {
            'valid': False,
            'quality_metrics': {},
            'validation_errors': [str(e)]
        }
