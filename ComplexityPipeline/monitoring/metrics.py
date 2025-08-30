# monitoring/metrics.py

import time
import psutil
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
import json

logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """Metrics for a single processing job."""
    job_id: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"  # running, completed, failed
    file_size_mb: float = 0.0
    complexity_scores: Dict[str, float] = field(default_factory=dict)
    prediction_result: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.end_time:
            return self.end_time - self.start_time
        return None

class MetricsCollector:
    """Advanced metrics collection for the VFX pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        self._setup_system_monitoring()
        
        # Job tracking
        self.active_jobs: Dict[str, ProcessingMetrics] = {}
        self.completed_jobs: deque = deque(maxlen=1000)  # Keep last 1000 jobs
        self.job_lock = threading.Lock()
        
        # Performance tracking
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        # Processing metrics
        self.jobs_total = Counter(
            'vfx_jobs_total',
            'Total number of VFX processing jobs',
            ['status', 'predicted_class'],
            registry=self.registry
        )
        
        self.processing_duration = Histogram(
            'vfx_processing_duration_seconds',
            'Time spent processing VFX shots',
            ['stage'],
            registry=self.registry
        )
        
        self.file_size_processed = Histogram(
            'vfx_file_size_mb',
            'Size of processed video files in MB',
            buckets=[1, 10, 50, 100, 500, 1000, 5000],
            registry=self.registry
        )
        
        # Model performance metrics
        self.model_confidence = Histogram(
            'vfx_model_confidence',
            'Model prediction confidence scores',
            ['predicted_class'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        self.complexity_scores = Histogram(
            'vfx_complexity_scores',
            'Individual complexity scores',
            ['metric_type'],
            registry=self.registry
        )
        
        # System metrics
        self.active_jobs_gauge = Gauge(
            'vfx_active_jobs',
            'Number of currently active processing jobs',
            registry=self.registry
        )
        
        self.queue_size = Gauge(
            'vfx_queue_size',
            'Number of jobs waiting in queue',
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'vfx_errors_total',
            'Total number of processing errors',
            ['error_type', 'stage'],
            registry=self.registry
        )
        
        # System resource metrics
        self.cpu_usage = Gauge(
            'vfx_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'vfx_memory_usage_mb',
            'Memory usage in MB',
            registry=self.registry
        )
        
        self.gpu_usage = Gauge(
            'vfx_gpu_usage_percent',
            'GPU usage percentage',
            registry=self.registry
        )
        
        # Model info
        self.model_info = Info(
            'vfx_model_info',
            'Information about the loaded model',
            registry=self.registry
        )
    
    def _setup_system_monitoring(self):
        """Setup system resource monitoring."""
        self.system_monitor_interval = self.config.get('monitoring', {}).get('system_interval', 30)
        self.gpu_available = self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import GPUtil
            return len(GPUtil.getGPUs()) > 0
        except ImportError:
            logger.warning("GPUtil not available, GPU monitoring disabled")
            return False
    
    def _start_background_monitoring(self):
        """Start background system monitoring thread."""
        def monitor_system():
            while True:
                try:
                    # CPU and Memory
                    self.cpu_usage.set(psutil.cpu_percent())
                    memory = psutil.virtual_memory()
                    self.memory_usage.set(memory.used / (1024 * 1024))  # MB
                    
                    # GPU monitoring
                    if self.gpu_available:
                        try:
                            import GPUtil
                            gpus = GPUtil.getGPUs()
                            if gpus:
                                self.gpu_usage.set(gpus[0].load * 100)
                        except Exception as e:
                            logger.debug(f"GPU monitoring error: {e}")
                    
                    # Update active jobs count
                    with self.job_lock:
                        self.active_jobs_gauge.set(len(self.active_jobs))
                    
                    time.sleep(self.system_monitor_interval)
                except Exception as e:
                    logger.error(f"System monitoring error: {e}")
                    time.sleep(self.system_monitor_interval)
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    def start_job(self, job_id: str, file_size_mb: float = 0.0) -> ProcessingMetrics:
        """Start tracking a new processing job."""
        with self.job_lock:
            metrics = ProcessingMetrics(
                job_id=job_id,
                start_time=time.time(),
                file_size_mb=file_size_mb
            )
            self.active_jobs[job_id] = metrics
            
            # Update Prometheus metrics
            self.file_size_processed.observe(file_size_mb)
            
            logger.info(f"Started tracking job {job_id}")
            return metrics
    
    def update_job_complexity(self, job_id: str, complexity_scores: Dict[str, float]):
        """Update job with complexity scores."""
        with self.job_lock:
            if job_id in self.active_jobs:
                self.active_jobs[job_id].complexity_scores = complexity_scores
                
                # Update Prometheus metrics
                for metric_name, score in complexity_scores.items():
                    if metric_name != 'sequence_mean':  # Skip derived features
                        self.complexity_scores.labels(metric_type=metric_name).observe(score)
    
    def complete_job(self, job_id: str, prediction_result: Dict[str, Any], status: str = "completed"):
        """Mark job as completed and record final metrics."""
        with self.job_lock:
            if job_id not in self.active_jobs:
                logger.warning(f"Job {job_id} not found in active jobs")
                return
            
            job = self.active_jobs[job_id]
            job.end_time = time.time()
            job.status = status
            job.prediction_result = prediction_result
            
            # Update Prometheus metrics
            predicted_class = prediction_result.get('predicted_class', 'unknown')
            self.jobs_total.labels(status=status, predicted_class=predicted_class).inc()
            
            if job.duration_seconds:
                self.processing_duration.labels(stage='total').observe(job.duration_seconds)
            
            # Record model confidence
            confidence_str = prediction_result.get('confidence', '0.00%')
            try:
                confidence_val = float(confidence_str.replace('%', '')) / 100.0
                self.model_confidence.labels(predicted_class=predicted_class).observe(confidence_val)
            except ValueError:
                logger.warning(f"Could not parse confidence: {confidence_str}")
            
            # Move to completed jobs
            self.completed_jobs.append(job)
            del self.active_jobs[job_id]
            
            logger.info(f"Completed job {job_id} in {job.duration_seconds:.2f}s")
    
    def record_error(self, job_id: str, error_type: str, stage: str, error_message: str):
        """Record an error for a job."""
        with self.job_lock:
            if job_id in self.active_jobs:
                self.active_jobs[job_id].error_message = error_message
                self.complete_job(job_id, {}, status="failed")
            
            self.errors_total.labels(error_type=error_type, stage=stage).inc()
            logger.error(f"Job {job_id} failed at {stage}: {error_message}")
    
    def record_stage_duration(self, stage: str, duration: float):
        """Record duration for a specific processing stage."""
        self.processing_duration.labels(stage=stage).observe(duration)
        
        # Also track in performance history
        self.performance_history[stage].append({
            'timestamp': time.time(),
            'duration': duration
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the last period."""
        with self.job_lock:
            now = time.time()
            hour_ago = now - 3600
            
            # Recent jobs
            recent_jobs = [
                job for job in self.completed_jobs 
                if job.end_time and job.end_time > hour_ago
            ]
            
            # Calculate statistics
            if recent_jobs:
                durations = [job.duration_seconds for job in recent_jobs if job.duration_seconds]
                success_rate = len([j for j in recent_jobs if j.status == "completed"]) / len(recent_jobs)
                
                avg_duration = sum(durations) / len(durations) if durations else 0
                
                # Class distribution
                class_counts = defaultdict(int)
                for job in recent_jobs:
                    pred_class = job.prediction_result.get('predicted_class', 'unknown')
                    class_counts[pred_class] += 1
            else:
                avg_duration = 0
                success_rate = 0
                class_counts = {}
            
            return {
                'period_hours': 1,
                'total_jobs': len(recent_jobs),
                'active_jobs': len(self.active_jobs),
                'success_rate': success_rate,
                'avg_processing_time': avg_duration,
                'class_distribution': dict(class_counts),
                'system_resources': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent
                }
            }
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry).decode('utf-8')
    
    def set_model_info(self, model_info: Dict[str, str]):
        """Set model information."""
        self.model_info.info(model_info)
