# optimization/caching.py

import asyncio
import time
import threading
from typing import Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import hashlib
import pickle
import json
from pathlib import Path
import redis
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: datetime
    ttl: Optional[int]
    access_count: int
    size_bytes: int

class CacheManager:
    """Advanced caching system for the VFX pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_config = config.get('caching', {})
        
        # Cache configuration
        self.max_memory_mb = self.cache_config.get('max_memory_mb', 1024)
        self.default_ttl = self.cache_config.get('default_ttl', 3600)
        self.cleanup_interval = self.cache_config.get('cleanup_interval', 300)
        
        # Cache types
        self.enable_memory_cache = self.cache_config.get('enable_memory_cache', True)
        self.enable_disk_cache = self.cache_config.get('enable_disk_cache', True)
        self.enable_redis_cache = self.cache_config.get('enable_redis_cache', False)
        
        # Storage paths
        self.disk_cache_dir = Path(self.cache_config.get('disk_cache_dir', 'cache'))
        self.disk_cache_dir.mkdir(exist_ok=True)
        
        # Initialize cache stores
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.cache_lock = threading.RLock()
        
        # Redis cache (optional)
        self.redis_client = None
        if self.enable_redis_cache:
            self._setup_redis()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage_mb': 0.0
        }
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.info(f"CacheManager initialized with {self.max_memory_mb}MB memory limit")
    
    def _setup_redis(self):
        """Setup Redis for distributed caching."""
        try:
            redis_config = self.cache_config.get('redis', {})
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 1),
                decode_responses=False  # Keep binary for numpy arrays
            )
            self.redis_client.ping()
            logger.info("Redis cache connection established")
        except Exception as e:
            logger.warning(f"Redis cache setup failed: {e}")
            self.redis_client = None
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_loop():
            while True:
                try:
                    self._cleanup_expired_entries()
                    self._enforce_memory_limit()
                    time.sleep(self.cleanup_interval)
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
                    time.sleep(self.cleanup_interval)
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        # Create a deterministic key from arguments
        key_data = {
            'prefix': prefix,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        # Try memory cache first
        if self.enable_memory_cache:
            with self.cache_lock:
                if key in self.memory_cache:
                    entry = self.memory_cache[key]
                    
                    # Check TTL
                    if entry.ttl and datetime.now() > entry.timestamp + timedelta(seconds=entry.ttl):
                        del self.memory_cache[key]
                        self.stats['misses'] += 1
                        return default
                    
                    # Update access count
                    entry.access_count += 1
                    self.stats['hits'] += 1
                    return entry.value
        
        # Try Redis cache
        if self.enable_redis_cache and self.redis_client:
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    value = pickle.loads(cached_data)
                    
                    # Store in memory cache for faster access
                    if self.enable_memory_cache:
                        self._store_in_memory(key, value)
                    
                    self.stats['hits'] += 1
                    return value
            except Exception as e:
                logger.warning(f"Redis cache get error: {e}")
        
        # Try disk cache
        if self.enable_disk_cache:
            try:
                cache_file = self.disk_cache_dir / f"{key}.cache"
                if cache_file.exists():
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    # Check TTL
                    if cache_data['ttl'] and datetime.now() > cache_data['timestamp'] + timedelta(seconds=cache_data['ttl']):
                        cache_file.unlink()
                        self.stats['misses'] += 1
                        return default
                    
                    value = cache_data['value']
                    
                    # Store in memory cache for faster access
                    if self.enable_memory_cache:
                        self._store_in_memory(key, value)
                    
                    self.stats['hits'] += 1
                    return value
            except Exception as e:
                logger.warning(f"Disk cache get error: {e}")
        
        self.stats['misses'] += 1
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if ttl is None:
            ttl = self.default_ttl
        
        success = False
        
        # Store in memory cache
        if self.enable_memory_cache:
            success = self._store_in_memory(key, value, ttl)
        
        # Store in Redis cache
        if self.enable_redis_cache and self.redis_client:
            try:
                serialized_value = pickle.dumps(value)
                self.redis_client.setex(key, ttl, serialized_value)
                success = True
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}")
        
        # Store in disk cache
        if self.enable_disk_cache:
            try:
                cache_data = {
                    'value': value,
                    'timestamp': datetime.now(),
                    'ttl': ttl
                }
                
                cache_file = self.disk_cache_dir / f"{key}.cache"
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                
                success = True
            except Exception as e:
                logger.warning(f"Disk cache set error: {e}")
        
        return success
    
    def _store_in_memory(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in memory cache."""
        try:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            with self.cache_lock:
                # Check if we need to make space
                if size_bytes > self.max_memory_mb * 1024 * 1024:
                    logger.warning(f"Value too large for memory cache: {size_bytes} bytes")
                    return False
                
                # Evict entries if necessary
                self._make_space(size_bytes)
                
                # Store entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=datetime.now(),
                    ttl=ttl,
                    access_count=1,
                    size_bytes=size_bytes
                )
                
                self.memory_cache[key] = entry
                self._update_memory_stats()
                
                return True
                
        except Exception as e:
            logger.error(f"Memory cache store error: {e}")
            return False
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, (str, bytes)):
                return len(value)
            else:
                return len(pickle.dumps(value))
        except:
            return 1024  # Default estimate
    
    def _make_space(self, required_bytes: int):
        """Make space in memory cache using LRU eviction."""
        current_size = sum(entry.size_bytes for entry in self.memory_cache.values())
        max_size = self.max_memory_mb * 1024 * 1024
        
        if current_size + required_bytes <= max_size:
            return
        
        # Sort by access count and timestamp (LRU)
        entries_by_lru = sorted(
            self.memory_cache.items(),
            key=lambda x: (x[1].access_count, x[1].timestamp)
        )
        
        # Evict entries until we have enough space
        for key, entry in entries_by_lru:
            if current_size + required_bytes <= max_size:
                break
            
            del self.memory_cache[key]
            current_size -= entry.size_bytes
            self.stats['evictions'] += 1
    
    def _cleanup_expired_entries(self):
        """Clean up expired entries from memory cache."""
        with self.cache_lock:
            current_time = datetime.now()
            expired_keys = []
            
            for key, entry in self.memory_cache.items():
                if entry.ttl and current_time > entry.timestamp + timedelta(seconds=entry.ttl):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                self._update_memory_stats()
    
    def _enforce_memory_limit(self):
        """Enforce memory limit by evicting entries."""
        with self.cache_lock:
            current_size = sum(entry.size_bytes for entry in self.memory_cache.values())
            max_size = self.max_memory_mb * 1024 * 1024
            
            if current_size > max_size:
                self._make_space(0)  # Force cleanup
                self._update_memory_stats()
    
    def _update_memory_stats(self):
        """Update memory usage statistics."""
        total_size = sum(entry.size_bytes for entry in self.memory_cache.values())
        self.stats['memory_usage_mb'] = total_size / (1024 * 1024)
    
    def delete(self, key: str) -> bool:
        """Delete key from all cache stores."""
        success = False
        
        # Delete from memory cache
        if self.enable_memory_cache:
            with self.cache_lock:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    self._update_memory_stats()
                    success = True
        
        # Delete from Redis cache
        if self.enable_redis_cache and self.redis_client:
            try:
                self.redis_client.delete(key)
                success = True
            except Exception as e:
                logger.warning(f"Redis cache delete error: {e}")
        
        # Delete from disk cache
        if self.enable_disk_cache:
            try:
                cache_file = self.disk_cache_dir / f"{key}.cache"
                if cache_file.exists():
                    cache_file.unlink()
                    success = True
            except Exception as e:
                logger.warning(f"Disk cache delete error: {e}")
        
        return success
    
    def clear(self):
        """Clear all cache stores."""
        # Clear memory cache
        if self.enable_memory_cache:
            with self.cache_lock:
                self.memory_cache.clear()
                self._update_memory_stats()
        
        # Clear Redis cache
        if self.enable_redis_cache and self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Redis cache clear error: {e}")
        
        # Clear disk cache
        if self.enable_disk_cache:
            try:
                for cache_file in self.disk_cache_dir.glob("*.cache"):
                    cache_file.unlink()
            except Exception as e:
                logger.warning(f"Disk cache clear error: {e}")
        
        logger.info("All caches cleared")
    
    def cached(self, ttl: Optional[int] = None, key_prefix: str = "func"):
        """Decorator for caching function results."""
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(key_prefix, func.__name__, *args, **kwargs)
                
                # Try to get from cache
                result = self.get(cache_key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        with self.cache_lock:
            memory_entries = len(self.memory_cache)
        
        disk_entries = len(list(self.disk_cache_dir.glob("*.cache"))) if self.enable_disk_cache else 0
        
        return {
            'hit_rate_percent': hit_rate,
            'total_hits': self.stats['hits'],
            'total_misses': self.stats['misses'],
            'total_evictions': self.stats['evictions'],
            'memory_usage_mb': self.stats['memory_usage_mb'],
            'memory_entries': memory_entries,
            'disk_entries': disk_entries,
            'redis_connected': self.redis_client is not None,
            'cache_stores': {
                'memory': self.enable_memory_cache,
                'disk': self.enable_disk_cache,
                'redis': self.enable_redis_cache
            }
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.redis_client:
            self.redis_client.close()
        logger.info("CacheManager cleaned up")
