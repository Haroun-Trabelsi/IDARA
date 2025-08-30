# optimization/scaling.py

import asyncio
import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import psutil
import redis
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)

class ScalingAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"

@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    queue_size: int
    active_jobs: int
    avg_processing_time: float
    error_rate: float

@dataclass
class ScalingDecision:
    """Scaling decision with reasoning."""
    action: ScalingAction
    target_workers: int
    current_workers: int
    reasoning: str
    confidence: float
    timestamp: datetime

class AutoScaler:
    """Automatic scaling system for the VFX pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaling_config = config.get('scaling', {})
        
        # Scaling parameters
        self.min_workers = self.scaling_config.get('min_workers', 2)
        self.max_workers = self.scaling_config.get('max_workers', 20)
        self.target_cpu_usage = self.scaling_config.get('target_cpu_usage', 70.0)
        self.target_memory_usage = self.scaling_config.get('target_memory_usage', 80.0)
        self.scale_up_threshold = self.scaling_config.get('scale_up_threshold', 85.0)
        self.scale_down_threshold = self.scaling_config.get('scale_down_threshold', 50.0)
        
        # Queue management
        self.max_queue_size = self.scaling_config.get('max_queue_size', 100)
        self.queue_scale_factor = self.scaling_config.get('queue_scale_factor', 0.1)
        
        # Timing parameters
        self.evaluation_interval = self.scaling_config.get('evaluation_interval', 60)
        self.cooldown_period = self.scaling_config.get('cooldown_period', 300)
        self.metrics_window = self.scaling_config.get('metrics_window', 600)
        
        # State tracking
        self.current_workers = self.min_workers
        self.last_scaling_action = None
        self.last_scaling_time = None
        self.metrics_history: List[ScalingMetrics] = []
        self.scaling_history: List[ScalingDecision] = []
        
        # Redis for distributed coordination (optional)
        self.redis_client = None
        if self.scaling_config.get('redis_enabled', False):
            self._setup_redis()
        
        # Worker pool for scaling operations
        self.scaling_executor = ThreadPoolExecutor(max_workers=2)
        
        # Start auto-scaling loop
        self._start_scaling_loop()
        
        logger.info(f"AutoScaler initialized: {self.min_workers}-{self.max_workers} workers")
    
    def _setup_redis(self):
        """Setup Redis for distributed coordination."""
        try:
            redis_config = self.scaling_config.get('redis', {})
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection established for scaling coordination")
        except Exception as e:
            logger.warning(f"Redis setup failed: {e}")
            self.redis_client = None
    
    def _start_scaling_loop(self):
        """Start the auto-scaling evaluation loop."""
        def scaling_loop():
            while True:
                try:
                    asyncio.run(self._evaluate_scaling())
                    time.sleep(self.evaluation_interval)
                except Exception as e:
                    logger.error(f"Scaling evaluation error: {e}")
                    time.sleep(self.evaluation_interval)
        
        scaling_thread = threading.Thread(target=scaling_loop, daemon=True)
        scaling_thread.start()
    
    async def _evaluate_scaling(self):
        """Evaluate current metrics and make scaling decisions."""
        # Collect current metrics
        current_metrics = await self._collect_metrics()
        self.metrics_history.append(current_metrics)
        
        # Keep only recent metrics
        cutoff_time = datetime.now() - timedelta(seconds=self.metrics_window)
        self.metrics_history = [
            m for m in self.metrics_history if m.timestamp > cutoff_time
        ]
        
        # Make scaling decision
        decision = self._make_scaling_decision(current_metrics)
        
        if decision.action != ScalingAction.MAINTAIN:
            await self._execute_scaling_decision(decision)
        
        self.scaling_history.append(decision)
        
        # Log decision
        logger.info(f"Scaling decision: {decision.action.value} "
                   f"(current: {decision.current_workers}, target: {decision.target_workers}) "
                   f"- {decision.reasoning}")
    
    async def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Application metrics (would be provided by your metrics collector)
        queue_size = await self._get_queue_size()
        active_jobs = await self._get_active_jobs_count()
        avg_processing_time = await self._get_avg_processing_time()
        error_rate = await self._get_error_rate()
        
        return ScalingMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            queue_size=queue_size,
            active_jobs=active_jobs,
            avg_processing_time=avg_processing_time,
            error_rate=error_rate
        )
    
    async def _get_queue_size(self) -> int:
        """Get current queue size."""
        # This would integrate with your actual queue system
        # For now, return a placeholder
        return 0
    
    async def _get_active_jobs_count(self) -> int:
        """Get number of active jobs."""
        # This would integrate with your job tracking system
        return 0
    
    async def _get_avg_processing_time(self) -> float:
        """Get average processing time."""
        if len(self.metrics_history) < 2:
            return 0.0
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        if not recent_metrics:
            return 0.0
        
        # Calculate average from recent metrics
        # This is a placeholder - you'd use actual processing times
        return sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
    
    async def _get_error_rate(self) -> float:
        """Get current error rate."""
        # This would integrate with your error tracking system
        return 0.0
    
    def _make_scaling_decision(self, current_metrics: ScalingMetrics) -> ScalingDecision:
        """Make scaling decision based on current metrics."""
        # Check cooldown period
        if (self.last_scaling_time and 
            datetime.now() - self.last_scaling_time < timedelta(seconds=self.cooldown_period)):
            return ScalingDecision(
                action=ScalingAction.MAINTAIN,
                target_workers=self.current_workers,
                current_workers=self.current_workers,
                reasoning="In cooldown period",
                confidence=1.0,
                timestamp=datetime.now()
            )
        
        # Calculate scaling factors
        cpu_factor = current_metrics.cpu_usage / self.target_cpu_usage
        memory_factor = current_metrics.memory_usage / self.target_memory_usage
        queue_factor = min(current_metrics.queue_size / self.max_queue_size, 1.0)
        
        # Weighted scaling score
        scaling_score = (cpu_factor * 0.4 + memory_factor * 0.3 + queue_factor * 0.3)
        
        # Determine action
        if scaling_score > 1.2 and self.current_workers < self.max_workers:
            # Scale up
            target_workers = min(
                self.max_workers,
                self.current_workers + max(1, int(scaling_score - 1.0))
            )
            action = ScalingAction.SCALE_UP
            reasoning = f"High load detected (score: {scaling_score:.2f})"
            confidence = min(1.0, scaling_score - 1.0)
            
        elif scaling_score < 0.7 and self.current_workers > self.min_workers:
            # Scale down
            target_workers = max(
                self.min_workers,
                self.current_workers - 1
            )
            action = ScalingAction.SCALE_DOWN
            reasoning = f"Low load detected (score: {scaling_score:.2f})"
            confidence = min(1.0, 1.0 - scaling_score)
            
        else:
            # Maintain current level
            target_workers = self.current_workers
            action = ScalingAction.MAINTAIN
            reasoning = f"Load within acceptable range (score: {scaling_score:.2f})"
            confidence = 1.0 - abs(scaling_score - 1.0)
        
        return ScalingDecision(
            action=action,
            target_workers=target_workers,
            current_workers=self.current_workers,
            reasoning=reasoning,
            confidence=confidence,
            timestamp=datetime.now()
        )
    
    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute a scaling decision."""
        try:
            if decision.action == ScalingAction.SCALE_UP:
                await self._scale_up(decision.target_workers)
            elif decision.action == ScalingAction.SCALE_DOWN:
                await self._scale_down(decision.target_workers)
            
            self.current_workers = decision.target_workers
            self.last_scaling_action = decision.action
            self.last_scaling_time = datetime.now()
            
            # Notify distributed systems if Redis is available
            if self.redis_client:
                await self._notify_scaling_change(decision)
            
        except Exception as e:
            logger.error(f"Scaling execution failed: {e}")
    
    async def _scale_up(self, target_workers: int):
        """Scale up to target number of workers."""
        workers_to_add = target_workers - self.current_workers
        
        # This would integrate with your actual worker management system
        # For example, with Kubernetes, Docker Swarm, or process management
        logger.info(f"Scaling up: adding {workers_to_add} workers")
        
        # Placeholder for actual scaling implementation
        # await self._start_new_workers(workers_to_add)
    
    async def _scale_down(self, target_workers: int):
        """Scale down to target number of workers."""
        workers_to_remove = self.current_workers - target_workers
        
        # This would integrate with your actual worker management system
        logger.info(f"Scaling down: removing {workers_to_remove} workers")
        
        # Placeholder for actual scaling implementation
        # await self._stop_workers(workers_to_remove)
    
    async def _notify_scaling_change(self, decision: ScalingDecision):
        """Notify other instances about scaling changes via Redis."""
        if not self.redis_client:
            return
        
        try:
            scaling_event = {
                'action': decision.action.value,
                'target_workers': decision.target_workers,
                'timestamp': decision.timestamp.isoformat(),
                'reasoning': decision.reasoning
            }
            
            self.redis_client.publish('vfx_scaling_events', json.dumps(scaling_event))
            
        except Exception as e:
            logger.error(f"Failed to notify scaling change: {e}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        recent_metrics = self.metrics_history[-10:] if self.metrics_history else []
        recent_decisions = self.scaling_history[-10:] if self.scaling_history else []
        
        return {
            'current_workers': self.current_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'last_scaling_action': self.last_scaling_action.value if self.last_scaling_action else None,
            'last_scaling_time': self.last_scaling_time.isoformat() if self.last_scaling_time else None,
            'recent_metrics': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'cpu_usage': m.cpu_usage,
                    'memory_usage': m.memory_usage,
                    'queue_size': m.queue_size,
                    'active_jobs': m.active_jobs
                }
                for m in recent_metrics
            ],
            'recent_decisions': [
                {
                    'action': d.action.value,
                    'target_workers': d.target_workers,
                    'reasoning': d.reasoning,
                    'confidence': d.confidence,
                    'timestamp': d.timestamp.isoformat()
                }
                for d in recent_decisions
            ]
        }
    
    def manual_scale(self, target_workers: int, reason: str = "Manual scaling"):
        """Manually trigger scaling to a specific number of workers."""
        if not (self.min_workers <= target_workers <= self.max_workers):
            raise ValueError(f"Target workers must be between {self.min_workers} and {self.max_workers}")
        
        decision = ScalingDecision(
            action=ScalingAction.SCALE_UP if target_workers > self.current_workers else ScalingAction.SCALE_DOWN,
            target_workers=target_workers,
            current_workers=self.current_workers,
            reasoning=reason,
            confidence=1.0,
            timestamp=datetime.now()
        )
        
        asyncio.create_task(self._execute_scaling_decision(decision))
        
        logger.info(f"Manual scaling triggered: {self.current_workers} -> {target_workers}")
    
    def cleanup(self):
        """Clean up resources."""
        self.scaling_executor.shutdown(wait=True)
        if self.redis_client:
            self.redis_client.close()
        logger.info("AutoScaler cleaned up")
