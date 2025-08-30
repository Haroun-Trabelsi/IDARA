# monitoring/health.py

import asyncio
import time
import psutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import os
import torch
from pathlib import Path
import pymongo

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float
    metadata: Dict[str, Any]

class HealthChecker:
    """Comprehensive health checking for the VFX pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.health_config = config.get('health_checks', {})
        self.checks_history: List[HealthCheck] = []
        
        # Health check intervals
        self.check_interval = self.health_config.get('interval_seconds', 30)
        self.detailed_check_interval = self.health_config.get('detailed_interval_seconds', 300)
        
        # Thresholds
        self.thresholds = self.health_config.get('thresholds', {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'response_time_ms': 5000.0
        })
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive status."""
        start_time = time.time()
        
        checks = await asyncio.gather(
            self._check_system_resources(),
            self._check_database_connection(),
            self._check_model_loading(),
            self._check_file_system(),
            self._check_external_dependencies(),
            return_exceptions=True
        )
        
        # Process results
        health_checks = []
        overall_status = HealthStatus.HEALTHY
        
        for check in checks:
            if isinstance(check, Exception):
                health_check = HealthCheck(
                    name="unknown",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(check)}",
                    timestamp=datetime.now(),
                    duration_ms=0.0,
                    metadata={}
                )
                overall_status = HealthStatus.UNHEALTHY
            else:
                health_check = check
                if check.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif check.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
            
            health_checks.append(health_check)
            self.checks_history.append(health_check)
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.checks_history = [
            check for check in self.checks_history
            if check.timestamp > cutoff_time
        ]
        
        total_duration = (time.time() - start_time) * 1000
        
        return {
            'overall_status': overall_status.value,
            'timestamp': datetime.now().isoformat(),
            'total_duration_ms': total_duration,
            'checks': [
                {
                    'name': check.name,
                    'status': check.status.value,
                    'message': check.message,
                    'duration_ms': check.duration_ms,
                    'metadata': check.metadata
                }
                for check in health_checks
            ]
        }
    
    async def _check_system_resources(self) -> HealthCheck:
        """Check system resource usage."""
        start_time = time.time()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Determine status
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_percent > self.thresholds['cpu_percent']:
                status = HealthStatus.DEGRADED
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > self.thresholds['memory_percent']:
                status = HealthStatus.DEGRADED
                issues.append(f"High memory usage: {memory_percent:.1f}%")
            
            if disk_percent > self.thresholds['disk_percent']:
                status = HealthStatus.UNHEALTHY
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            
            message = "System resources normal" if not issues else "; ".join(issues)
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent,
                    'available_memory_gb': memory.available / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check system resources: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={}
            )
    
    async def _check_database_connection(self) -> HealthCheck:
        """Check database connectivity."""
        start_time = time.time()
        
        try:
            mongodb_config = self.config.get('mongodb', {})
            uri = mongodb_config.get('uri', 'mongodb://localhost:27017')
            database = mongodb_config.get('database', 'vfx_classification_demo')
            
            # Test connection
            client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            
            # Test database access
            db = client[database]
            collections = db.list_collection_names()
            
            client.close()
            
            return HealthCheck(
                name="database_connection",
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={
                    'database': database,
                    'collections_count': len(collections)
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="database_connection",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={}
            )
    
    async def _check_model_loading(self) -> HealthCheck:
        """Check if ML model can be loaded."""
        start_time = time.time()
        
        try:
            model_config = self.config.get('multimodal_model', {})
            model_path = model_config.get('model_path', 'inference_model/best_vfx_model.pt')
            
            # Check if model file exists
            if not Path(model_path).exists():
                return HealthCheck(
                    name="model_loading",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Model file not found: {model_path}",
                    timestamp=datetime.now(),
                    duration_ms=(time.time() - start_time) * 1000,
                    metadata={'model_path': model_path}
                )
            
            # Try to load model metadata
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(model_path, map_location=device)
            
            model_info = {
                'device': str(device),
                'model_size_mb': Path(model_path).stat().st_size / (1024 * 1024),
                'has_config': 'config' in checkpoint,
                'has_state_dict': 'model_state_dict' in checkpoint or 'state_dict' in checkpoint
            }
            
            # Check GPU availability if CUDA model
            gpu_status = "available" if torch.cuda.is_available() else "not_available"
            if torch.cuda.is_available():
                model_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            return HealthCheck(
                name="model_loading",
                status=HealthStatus.HEALTHY,
                message="Model loading check successful",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                metadata=model_info
            )
            
        except Exception as e:
            return HealthCheck(
                name="model_loading",
                status=HealthStatus.UNHEALTHY,
                message=f"Model loading check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={}
            )
    
    async def _check_file_system(self) -> HealthCheck:
        """Check file system access and permissions."""
        start_time = time.time()
        
        try:
            # Check critical directories
            directories_to_check = [
                self.config.get('watchdog', {}).get('directory', 'input_files'),
                self.config.get('multimodal_model', {}).get('features_dir', 'features'),
                'temp_outputs',
                'inference_model'
            ]
            
            issues = []
            for directory in directories_to_check:
                dir_path = Path(directory)
                
                if not dir_path.exists():
                    issues.append(f"Directory missing: {directory}")
                    continue
                
                # Check read/write permissions
                if not os.access(dir_path, os.R_OK):
                    issues.append(f"No read access: {directory}")
                
                if not os.access(dir_path, os.W_OK):
                    issues.append(f"No write access: {directory}")
            
            status = HealthStatus.HEALTHY if not issues else HealthStatus.UNHEALTHY
            message = "File system access normal" if not issues else "; ".join(issues)
            
            return HealthCheck(
                name="file_system",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={'checked_directories': directories_to_check}
            )
            
        except Exception as e:
            return HealthCheck(
                name="file_system",
                status=HealthStatus.UNHEALTHY,
                message=f"File system check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={}
            )
    
    async def _check_external_dependencies(self) -> HealthCheck:
        """Check external dependencies and services."""
        start_time = time.time()
        
        try:
            import cv2
            import numpy as np
            import torch
            import joblib
            import prefect
            
            # Check OpenCV
            cv2_version = cv2.__version__
            
            # Check if we can create a simple tensor
            test_tensor = torch.randn(2, 3)
            
            # Check if we can use numpy
            test_array = np.array([1, 2, 3])
            
            return HealthCheck(
                name="external_dependencies",
                status=HealthStatus.HEALTHY,
                message="All external dependencies available",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={
                    'opencv_version': cv2_version,
                    'torch_version': torch.__version__,
                    'numpy_version': np.__version__,
                    'cuda_available': torch.cuda.is_available()
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="external_dependencies",
                status=HealthStatus.UNHEALTHY,
                message=f"External dependency check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={}
            )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary from recent checks."""
        if not self.checks_history:
            return {
                'status': 'unknown',
                'message': 'No health checks performed yet',
                'last_check': None
            }
        
        # Get most recent check of each type
        recent_checks = {}
        for check in reversed(self.checks_history):
            if check.name not in recent_checks:
                recent_checks[check.name] = check
        
        # Determine overall status
        statuses = [check.status for check in recent_checks.values()]
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            'status': overall_status.value,
            'last_check': max(check.timestamp for check in recent_checks.values()).isoformat(),
            'checks_summary': {
                name: {
                    'status': check.status.value,
                    'message': check.message,
                    'timestamp': check.timestamp.isoformat()
                }
                for name, check in recent_checks.items()
            }
        }
