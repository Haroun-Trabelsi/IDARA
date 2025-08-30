# optimization/performance.py

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import logging
import psutil
import torch
import numpy as np
from functools import wraps
import cProfile
import pstats
from io import StringIO

logger = logging.getLogger(__name__)

@dataclass
class PerformanceProfile:
    """Performance profiling result."""
    function_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_percent: float
    profile_data: Optional[str] = None

class PerformanceOptimizer:
    """Performance optimization utilities for the VFX pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.perf_config = config.get('performance', {})
        
        # Thread pool configuration
        self.max_workers = self.perf_config.get('max_workers', min(32, psutil.cpu_count() + 4))
        self.complexity_workers = self.perf_config.get('complexity_workers', min(9, psutil.cpu_count()))
        
        # Memory optimization
        self.enable_memory_optimization = self.perf_config.get('enable_memory_optimization', True)
        self.batch_size = self.perf_config.get('batch_size', 32)
        
        # GPU optimization
        self.enable_gpu_optimization = self.perf_config.get('enable_gpu_optimization', True)
        self.mixed_precision = self.perf_config.get('mixed_precision', True)
        
        # Caching
        self.enable_caching = self.perf_config.get('enable_caching', True)
        self.cache_size = self.perf_config.get('cache_size', 1000)
        
        # Initialize thread pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.complexity_workers)
        
        # Performance tracking
        self.performance_history: List[PerformanceProfile] = []
        
        # Feature cache
        self.feature_cache: Dict[str, np.ndarray] = {}
        self.cache_lock = threading.Lock()
        
        logger.info(f"Performance optimizer initialized with {self.max_workers} workers")
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function performance."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            # CPU profiling
            profiler = cProfile.Profile()
            profiler.enable()
            
            try:
                result = func(*args, **kwargs)
            finally:
                profiler.disable()
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            # Generate profile report
            profile_stream = StringIO()
            stats = pstats.Stats(profiler, stream=profile_stream)
            stats.sort_stats('cumulative')
            stats.print_stats(10)  # Top 10 functions
            
            profile = PerformanceProfile(
                function_name=func.__name__,
                execution_time=end_time - start_time,
                memory_usage_mb=end_memory - start_memory,
                cpu_percent=psutil.cpu_percent(),
                profile_data=profile_stream.getvalue()
            )
            
            self.performance_history.append(profile)
            logger.debug(f"Function {func.__name__} took {profile.execution_time:.3f}s")
            
            return result
        
        return wrapper
    
    async def optimize_complexity_analysis(self, video_path: str, analysis_tasks: List[Any]) -> Dict[str, float]:
        """Optimize complexity analysis execution using parallel processing."""
        start_time = time.time()
        
        # Group tasks by computational intensity
        cpu_intensive_tasks = []
        io_intensive_tasks = []
        
        for task in analysis_tasks:
            if task.name in ['blur_analysis', 'motion_analysis', 'distortion_analysis']:
                cpu_intensive_tasks.append(task)
            else:
                io_intensive_tasks.append(task)
        
        # Execute tasks in parallel
        loop = asyncio.get_event_loop()
        
        # CPU-intensive tasks in process pool
        cpu_futures = [
            loop.run_in_executor(self.process_pool, self._execute_analysis_task, task)
            for task in cpu_intensive_tasks
        ]
        
        # I/O-intensive tasks in thread pool
        io_futures = [
            loop.run_in_executor(self.thread_pool, self._execute_analysis_task, task)
            for task in io_intensive_tasks
        ]
        
        # Wait for all tasks to complete
        all_results = await asyncio.gather(*cpu_futures, *io_futures, return_exceptions=True)
        
        # Process results
        complexity_scores = {}
        for i, result in enumerate(all_results):
            task = (cpu_intensive_tasks + io_intensive_tasks)[i]
            if isinstance(result, Exception):
                logger.error(f"Task {task.name} failed: {result}")
                complexity_scores[task.metric.value] = task.default_value
            else:
                complexity_scores[task.metric.value] = task.result_extractor(result)
        
        execution_time = time.time() - start_time
        logger.info(f"Complexity analysis completed in {execution_time:.2f}s")
        
        return complexity_scores
    
    def _execute_analysis_task(self, task) -> Any:
        """Execute a single analysis task."""
        try:
            return task.func(*task.args, **task.kwargs)
        except Exception as e:
            logger.error(f"Analysis task {task.name} failed: {e}")
            raise
    
    def optimize_feature_extraction(self, video_path: Path) -> np.ndarray:
        """Optimize feature extraction with caching and memory management."""
        cache_key = f"{video_path.name}_{video_path.stat().st_mtime}"
        
        # Check cache first
        if self.enable_caching:
            with self.cache_lock:
                if cache_key in self.feature_cache:
                    logger.debug(f"Cache hit for {video_path.name}")
                    return self.feature_cache[cache_key]
        
        # Extract features with optimization
        features = self._extract_features_optimized(video_path)
        
        # Cache result
        if self.enable_caching:
            with self.cache_lock:
                # Implement LRU cache
                if len(self.feature_cache) >= self.cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.feature_cache))
                    del self.feature_cache[oldest_key]
                
                self.feature_cache[cache_key] = features
        
        return features
    
    def _extract_features_optimized(self, video_path: Path) -> np.ndarray:
        """Extract features with memory and GPU optimization."""
        import cv2
        
        # Optimized video reading
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        features = []
        frame_count = 0
        
        # Memory-efficient frame processing
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for memory efficiency
            if frame.shape[0] > 720:  # If height > 720p
                scale_factor = 720 / frame.shape[0]
                new_width = int(frame.shape[1] * scale_factor)
                frame = cv2.resize(frame, (new_width, 720))
            
            # Process frame (placeholder for actual feature extraction)
            # This would call your actual feature extraction model
            frame_features = self._extract_frame_features(frame)
            features.append(frame_features)
            
            frame_count += 1
            
            # Memory management - process in batches
            if frame_count % self.batch_size == 0:
                # Force garbage collection periodically
                import gc
                gc.collect()
        
        cap.release()
        
        if not features:
            raise ValueError("No features extracted from video")
        
        return np.array(features)
    
    def _extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract features from a single frame (optimized)."""
        # Placeholder for actual feature extraction
        # This would use your ResNet model with optimizations
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame).float()
        
        # GPU optimization
        if self.enable_gpu_optimization and torch.cuda.is_available():
            frame_tensor = frame_tensor.cuda()
            
            # Mixed precision if enabled
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    # Your model inference here
                    features = torch.randn(2048)  # Placeholder
            else:
                features = torch.randn(2048)  # Placeholder
            
            features = features.cpu()
        else:
            features = torch.randn(2048)  # Placeholder
        
        return features.numpy()
    
    def optimize_model_inference(self, model: torch.nn.Module, 
                                sequence_features: np.ndarray, 
                                static_features: np.ndarray) -> Dict[str, Any]:
        """Optimize model inference with various techniques."""
        start_time = time.time()
        
        # Convert to tensors
        seq_tensor = torch.from_numpy(sequence_features).float().unsqueeze(0)
        static_tensor = torch.from_numpy(static_features).float()
        
        # GPU optimization
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.enable_gpu_optimization and torch.cuda.is_available():
            seq_tensor = seq_tensor.to(device)
            static_tensor = static_tensor.to(device)
            model = model.to(device)
        
        # Inference optimization
        with torch.no_grad():
            if self.mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    output = model(seq_tensor, static_tensor)
            else:
                output = model(seq_tensor, static_tensor)
        
        # Convert back to CPU for post-processing
        if torch.cuda.is_available():
            output = output.cpu()
        
        inference_time = time.time() - start_time
        logger.debug(f"Model inference took {inference_time:.3f}s")
        
        return {
            'output': output,
            'inference_time': inference_time
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_history:
            return {'message': 'No performance data available'}
        
        recent_profiles = self.performance_history[-100:]  # Last 100 executions
        
        # Calculate statistics
        avg_execution_time = sum(p.execution_time for p in recent_profiles) / len(recent_profiles)
        avg_memory_usage = sum(p.memory_usage_mb for p in recent_profiles) / len(recent_profiles)
        
        # Function-wise statistics
        function_stats = {}
        for profile in recent_profiles:
            func_name = profile.function_name
            if func_name not in function_stats:
                function_stats[func_name] = {
                    'count': 0,
                    'total_time': 0.0,
                    'total_memory': 0.0
                }
            
            function_stats[func_name]['count'] += 1
            function_stats[func_name]['total_time'] += profile.execution_time
            function_stats[func_name]['total_memory'] += profile.memory_usage_mb
        
        # Calculate averages
        for func_name, stats in function_stats.items():
            stats['avg_time'] = stats['total_time'] / stats['count']
            stats['avg_memory'] = stats['total_memory'] / stats['count']
        
        return {
            'overall_stats': {
                'avg_execution_time': avg_execution_time,
                'avg_memory_usage_mb': avg_memory_usage,
                'total_executions': len(recent_profiles)
            },
            'function_stats': function_stats,
            'cache_stats': {
                'cache_size': len(self.feature_cache),
                'cache_enabled': self.enable_caching
            },
            'optimization_settings': {
                'max_workers': self.max_workers,
                'gpu_optimization': self.enable_gpu_optimization,
                'mixed_precision': self.mixed_precision,
                'batch_size': self.batch_size
            }
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        with self.cache_lock:
            self.feature_cache.clear()
        
        logger.info("Performance optimizer cleaned up")
