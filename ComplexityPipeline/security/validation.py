# security/validation.py

import os
import magic
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from pydantic import BaseModel, validator, Field
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class VideoFileMetadata(BaseModel):
    """Validated video file metadata."""
    filename: str = Field(..., min_length=1, max_length=255)
    size_bytes: int = Field(..., gt=0, le=5_000_000_000)  # Max 5GB
    duration_seconds: Optional[float] = Field(None, gt=0, le=7200)  # Max 2 hours
    width: Optional[int] = Field(None, gt=0, le=7680)  # Max 8K width
    height: Optional[int] = Field(None, gt=0, le=4320)  # Max 8K height
    fps: Optional[float] = Field(None, gt=0, le=120)
    codec: Optional[str] = None
    
    @validator('filename')
    def validate_filename(cls, v):
        """Validate filename for security."""
        # Check for path traversal attempts
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError("Invalid filename: path traversal detected")
        
        # Check for valid extensions
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        if not any(v.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Invalid file extension. Allowed: {valid_extensions}")
        
        return v

class ComplexityScores(BaseModel):
    """Validated complexity scores."""
    zoom_score: float = Field(..., ge=0.0, le=10.0)
    blur_score: float = Field(..., ge=0.0, le=1.0)
    distortion_score: float = Field(..., ge=0.0, le=1.0)
    motion_score: float = Field(..., ge=0.0, le=1.0)
    light_score: float = Field(..., ge=0.0, le=1.0)
    noise_score: float = Field(..., ge=0.0, le=1.0)
    overlap_score: float = Field(..., ge=0.0, le=1.0)
    parallax_score: float = Field(..., ge=0.0, le=1.0)
    focus_pull_score: float = Field(..., ge=0.0, le=1.0)
    sequence_mean: float = Field(..., ge=-10.0, le=10.0)

class PredictionResult(BaseModel):
    """Validated prediction result."""
    predicted_class: str = Field(..., regex=r'^(Easy|Medium|Hard)$')
    confidence: str = Field(..., regex=r'^\d{1,3}\.\d{2}%$')
    probabilities: Dict[str, str] = Field(...)
    processing_time_ms: float = Field(..., gt=0)
    model_version: str = Field(...)
    
    @validator('probabilities')
    def validate_probabilities(cls, v):
        """Validate probability format and sum."""
        expected_classes = {'Easy', 'Medium', 'Hard'}
        if set(v.keys()) != expected_classes:
            raise ValueError(f"Invalid classes. Expected: {expected_classes}")
        
        # Extract numeric values and check they sum to ~100%
        total = 0.0
        for class_name, prob_str in v.items():
            if not prob_str.endswith('%'):
                raise ValueError(f"Invalid probability format for {class_name}")
            prob_val = float(prob_str[:-1])
            if not (0.0 <= prob_val <= 100.0):
                raise ValueError(f"Invalid probability value for {class_name}: {prob_val}")
            total += prob_val
        
        if not (99.0 <= total <= 101.0):  # Allow small floating point errors
            raise ValueError(f"Probabilities don't sum to 100%: {total}")
        
        return v

class InputValidator:
    """Comprehensive input validation for the VFX pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_file_size = config.get('validation', {}).get('max_file_size_mb', 1000) * 1024 * 1024
        self.allowed_mime_types = [
            'video/mp4', 'video/avi', 'video/quicktime', 
            'video/x-msvideo', 'video/x-matroska'
        ]
    
    def validate_video_file(self, file_path: Path) -> VideoFileMetadata:
        """Validate video file and extract metadata."""
        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
        
        # Check MIME type
        try:
            mime_type = magic.from_file(str(file_path), mime=True)
            if mime_type not in self.allowed_mime_types:
                raise ValueError(f"Invalid MIME type: {mime_type}")
        except Exception as e:
            logger.warning(f"Could not determine MIME type: {e}")
        
        # Extract video metadata using OpenCV
        metadata = self._extract_video_metadata(file_path)
        
        return VideoFileMetadata(
            filename=file_path.name,
            size_bytes=file_size,
            **metadata
        )
    
    def _extract_video_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract video metadata using OpenCV."""
        try:
            cap = cv2.VideoCapture(str(file_path))
            
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            duration = frame_count / fps if fps > 0 else None
            
            # Get codec information
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            cap.release()
            
            return {
                'width': width,
                'height': height,
                'fps': fps,
                'duration_seconds': duration,
                'codec': codec
            }
            
        except Exception as e:
            logger.error(f"Error extracting video metadata: {e}")
            return {}
    
    def validate_complexity_scores(self, scores: Dict[str, float]) -> ComplexityScores:
        """Validate complexity scores."""
        return ComplexityScores(**scores)
    
    def validate_prediction_result(self, result: Dict[str, Any]) -> PredictionResult:
        """Validate prediction result."""
        return PredictionResult(**result)

class FileValidator:
    """File system security validator."""
    
    def __init__(self, allowed_directories: List[str]):
        self.allowed_directories = [Path(d).resolve() for d in allowed_directories]
    
    def is_path_safe(self, file_path: Path) -> bool:
        """Check if file path is within allowed directories."""
        try:
            resolved_path = file_path.resolve()
            return any(
                str(resolved_path).startswith(str(allowed_dir))
                for allowed_dir in self.allowed_directories
            )
        except Exception:
            return False
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe filesystem operations."""
        # Remove dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        sanitized = filename
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:255-len(ext)] + ext
        
        return sanitized
