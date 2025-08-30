# pipeline/utils.py

import os
import yaml
import logging
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextlib import asynccontextmanager
import time

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, PyMongoError

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when there's an issue with configuration loading or validation."""
    pass


class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""
    pass


class ConfigManager:
    """
    Loads and manages configuration from YAML files with validation and type safety.
    
    Supports nested configuration access via dot notation (e.g., 'database.host').
    """
    
    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file
            
        Raises:
            ConfigurationError: If the configuration file is not found or invalid
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from the YAML file."""
        if not self.config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found at {self.config_path}. "
                f"Please create it or specify a valid path."
            )
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML syntax in {self.config_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {self.config_path}: {e}")
        
        logger.info(f"Configuration loaded from {self.config_path}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration key (e.g., 'database.host')
            default: Default value to return if key is not found
            
        Returns:
            The configuration value or default if not found
            
        Examples:
            >>> config.get('database.host', 'localhost')
            'mongodb://localhost:27017'
            >>> config.get('nonexistent.key', 'default')
            'default'
        """
        if not isinstance(key_path, str):
            raise ValueError("Key path must be a string")
        
        parts = key_path.split('.')
        value = self.config
        
        try:
            for part in parts:
                if not isinstance(value, dict) or part not in value:
                    return default
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_required(self, key_path: str) -> Any:
        """
        Get a required configuration value.
        
        Args:
            key_path: Dot-separated path to the configuration key
            
        Returns:
            The configuration value
            
        Raises:
            ConfigurationError: If the required key is not found
        """
        value = self.get(key_path)
        if value is None:
            raise ConfigurationError(f"Required configuration key '{key_path}' not found")
        return value
    
    def validate_required_keys(self, required_keys: list[str]) -> None:
        """
        Validate that all required configuration keys are present.
        
        Args:
            required_keys: List of required configuration key paths
            
        Raises:
            ConfigurationError: If any required key is missing
        """
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            raise ConfigurationError(
                f"Missing required configuration keys: {', '.join(missing_keys)}"
            )
    
    def reload(self) -> None:
        """Reload configuration from the file."""
        self._load_config()
    
    def __contains__(self, key_path: str) -> bool:
        """Check if a configuration key exists."""
        return self.get(key_path) is not None
    
    def __repr__(self) -> str:
        return f"ConfigManager(config_path={self.config_path})"


class DatabaseManager:
    """
    Handles MongoDB connection and operations with proper error handling and connection pooling.
    
    Provides async-compatible methods for database operations and graceful handling
    of connection failures.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the database manager.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.client: Optional[MongoClient] = None
        self.db = None
        self.collection = None
        self._connection_params = self._get_connection_params()
        self._connect()
    
    def _get_connection_params(self) -> Dict[str, Any]:
        """Extract and validate database connection parameters."""
        mongodb_config = self.config.get("mongodb", {})
        
        if not mongodb_config:
            raise ConfigurationError("MongoDB configuration section is missing")
        
        required_keys = ["uri", "database", "collection"]
        for key in required_keys:
            if not mongodb_config.get(key):
                raise ConfigurationError(f"MongoDB configuration missing required key: {key}")
        
        return {
            'uri': mongodb_config['uri'],
            'database': mongodb_config['database'],
            'collection': mongodb_config['collection'],
            'server_selection_timeout_ms': mongodb_config.get('server_selection_timeout_ms', 5000),
            'connect_timeout_ms': mongodb_config.get('connect_timeout_ms', 5000),
            'max_pool_size': mongodb_config.get('max_pool_size', 10)
        }
    
    def _connect(self) -> None:
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(
                self._connection_params['uri'],
                serverSelectionTimeoutMS=self._connection_params['server_selection_timeout_ms'],
                connectTimeoutMS=self._connection_params['connect_timeout_ms'],
                maxPoolSize=self._connection_params['max_pool_size']
            )
            
            # Test the connection
            self.client.admin.command('ping')
            
            # Set up database and collection references
            self.db = self.client[self._connection_params['database']]
            self.collection = self.db[self._connection_params['collection']]
            
            logger.info(f"Connected to MongoDB at {self._connection_params['uri']}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self._handle_connection_failure(e)
        except Exception as e:
            self._handle_connection_failure(e)
    
    def _handle_connection_failure(self, error: Exception) -> None:
        """Handle database connection failures gracefully."""
        self.client = None
        self.db = None
        self.collection = None
        
        logger.warning(
            "Could not connect to MongoDB at %s (%s). Database operations will be skipped.",
            self._connection_params['uri'],
            error
        )
    
    @property
    def is_connected(self) -> bool:
        """Check if database connection is active."""
        if not self.client:
            return False
        
        try:
            self.client.admin.command('ping')
            return True
        except PyMongoError:
            return False
    
    async def insert_result(self, result_data: Dict[str, Any]) -> bool:
        """
        Insert result data into MongoDB.
        
        Args:
            result_data: Dictionary containing the result data to insert
            
        Returns:
            True if insertion was successful, False otherwise
        """
        if not self.is_connected:
            logger.info(
                "MongoDB unavailable; skipping DB insert for %s", 
                result_data.get('filename', 'unknown')
            )
            return False
        
        try:
            # Validate result_data
            if not isinstance(result_data, dict):
                raise ValueError("Result data must be a dictionary")
            
            if not result_data:
                raise ValueError("Result data cannot be empty")
            
            # Add timestamp if not present
            if 'timestamp' not in result_data:
                result_data['timestamp'] = time.time()
            
            # Insert data asynchronously
            await asyncio.to_thread(self.collection.insert_one, result_data)
            
            logger.info("Stored result for %s in MongoDB", result_data.get('filename', 'unknown'))
            return True
            
        except PyMongoError as e:
            logger.error("Database insertion failed: %s", e)
            return False
        except Exception as e:
            logger.error("Unexpected error during database insertion: %s", e)
            return False
    
    async def find_result(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find a single result in the database.
        
        Args:
            query: MongoDB query dictionary
            
        Returns:
            Found document or None
        """
        if not self.is_connected:
            return None
        
        try:
            result = await asyncio.to_thread(self.collection.find_one, query)
            return result
        except PyMongoError as e:
            logger.error("Database query failed: %s", e)
            return None
    
    async def update_result(self, query: Dict[str, Any], update: Dict[str, Any]) -> bool:
        """
        Update a result in the database.
        
        Args:
            query: MongoDB query to find the document
            update: Update operations to apply
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self.is_connected:
            return False
        
        try:
            result = await asyncio.to_thread(self.collection.update_one, query, update)
            return result.modified_count > 0
        except PyMongoError as e:
            logger.error("Database update failed: %s", e)
            return False
    
    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for database transactions.
        
        Usage:
            async with db_manager.transaction() as session:
                await db_manager.insert_result(data, session=session)
        """
        if not self.is_connected:
            yield None
            return
        
        async with await asyncio.to_thread(self.client.start_session) as session:
            async with session.start_transaction():
                yield session
    
    def close(self) -> None:
        """Close the database connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.collection = None
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


@dataclass
class PipelineContext:
    """
    Data carrier for pipeline stages with type safety and validation.
    
    Contains all the data that flows through the pipeline stages,
    including input file information, intermediate results, and final output.
    """
    file_path: Path
    job_id: Optional[str] = None 
    start_time: float = field(default_factory=time.time)
    complexity_scores: Dict[str, float] = field(default_factory=dict)
    final_result: Dict[str, Any] = field(default_factory=dict)
    feature_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_log: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and normalize the file path."""
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)
        
        if not isinstance(self.file_path, Path):
            raise ValueError("file_path must be a Path object or string")
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.file_path}")
        
        # Initialize metadata with basic file information
        self.metadata.update({
            'filename': self.file_path.name,
            'file_size': self.file_path.stat().st_size,
            'file_stem': self.file_path.stem,
            'file_suffix': self.file_path.suffix
        })
    
    @property
    def elapsed_time(self) -> float:
        """Get the elapsed time since pipeline start."""
        return time.time() - self.start_time
    
    @property
    def is_complete(self) -> bool:
        """Check if the pipeline has completed successfully."""
        return bool(self.final_result and not self.error_log)
    
    def add_error(self, error_message: str) -> None:
        """Add an error message to the error log."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        self.error_log.append(f"[{timestamp}] {error_message}")
        logger.error(f"Pipeline error for {self.file_path.name}: {error_message}")
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the context."""
        self.metadata[key] = value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline execution."""
        return {
            'filename': self.file_path.name,
            'elapsed_time': self.elapsed_time,
            'complexity_scores': self.complexity_scores,
            'final_result': self.final_result,
            'errors': self.error_log,
            'metadata': self.metadata,
            'is_complete': self.is_complete
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            'file_path': str(self.file_path),
            'start_time': self.start_time,
            'complexity_scores': self.complexity_scores,
            'final_result': self.final_result,
            'feature_path': str(self.feature_path) if self.feature_path else None,
            'metadata': self.metadata,
            'error_log': self.error_log,
            'elapsed_time': self.elapsed_time,
            'is_complete': self.is_complete
        }
    
    def __repr__(self) -> str:
        return (
            f"PipelineContext(file_path={self.file_path.name}, "
            f"elapsed_time={self.elapsed_time:.2f}s, "
            f"is_complete={self.is_complete})"
        )