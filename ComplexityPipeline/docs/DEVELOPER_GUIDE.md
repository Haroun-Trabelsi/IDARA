# VFX Pipeline Developer Guide

This guide provides comprehensive information for developers working on the VFX Shot Complexity Prediction Pipeline.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Architecture Overview](#architecture-overview)
3. [Code Structure](#code-structure)
4. [Development Workflow](#development-workflow)
5. [Testing](#testing)
6. [Performance Optimization](#performance-optimization)
7. [Security Guidelines](#security-guidelines)
8. [Contributing](#contributing)

## Development Environment Setup

### Prerequisites

- **Python**: 3.9+
- **CUDA**: 11.8+ (for GPU support)
- **MongoDB**: 7.0+
- **Redis**: 7.0+
- **Docker**: 20.10+
- **Git**: 2.30+

### Local Setup

```bash
# Clone repository
git clone https://github.com/your-org/vfx-pipeline.git
cd vfx-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Environment Configuration

```bash
# Copy configuration template
cp config.yaml.example config.yaml

# Set environment variables
export MONGODB_URI="mongodb://localhost:27017"
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
```

### Database Setup

```bash
# Start MongoDB and Redis with Docker
docker-compose up -d mongodb redis

# Verify connections
python -c "
import pymongo
import redis
mongo = pymongo.MongoClient('mongodb://localhost:27017')
r = redis.Redis(host='localhost', port=6379)
print('MongoDB:', mongo.admin.command('ismaster')['ok'])
print('Redis:', r.ping())
"
```

## Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File Watcher  │───▶│   Prefect Flow  │───▶│   FastAPI App   │
│   (Watchdog)    │    │  (Orchestrator) │    │   (REST API)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Files   │    │   ML Pipeline   │    │   Monitoring    │
│   (Videos)      │    │   (Analysis)    │    │  (Prometheus)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    MongoDB      │    │     Redis       │    │     Logs        │
│  (Persistence)  │    │   (Caching)     │    │  (Structured)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Input**: Video files detected by file watcher
2. **Analysis**: 9 complexity metrics calculated in parallel
3. **Feature Extraction**: CNN features extracted from video frames
4. **Classification**: Multimodal Bi-LSTM processes features + complexity scores
5. **Output**: Complexity prediction with confidence scores
6. **Storage**: Results stored in MongoDB with caching in Redis

### Key Technologies

- **ML Framework**: PyTorch with CUDA support
- **Orchestration**: Prefect for workflow management
- **API**: FastAPI with async support
- **Database**: MongoDB for persistence, Redis for caching
- **Monitoring**: Prometheus metrics, structured logging
- **Containerization**: Docker with multi-stage builds

## Code Structure

```
vfx-pipeline/
├── api/                    # FastAPI application
│   ├── __init__.py
│   ├── main.py            # API endpoints
│   ├── auth.py            # Authentication
│   └── middleware.py      # Custom middleware
├── models/                 # ML models and inference
│   ├── __init__.py
│   ├── complexity/        # Complexity analysis models
│   ├── multimodal/        # Multimodal classifier
│   └── predictor.py       # Model inference
├── pipeline/              # Prefect workflows
│   ├── __init__.py
│   ├── tasks.py           # Prefect tasks
│   ├── flows.py           # Prefect flows
│   └── validation_tasks.py # Data validation
├── optimization/          # Performance optimizations
│   ├── __init__.py
│   ├── performance.py     # Performance utilities
│   ├── caching.py         # Caching system
│   └── scaling.py         # Auto-scaling
├── monitoring/            # Monitoring and observability
│   ├── __init__.py
│   ├── metrics.py         # Prometheus metrics
│   ├── alerts.py          # Alert definitions
│   └── health.py          # Health checks
├── security/              # Security components
│   ├── __init__.py
│   ├── auth.py            # Authentication
│   ├── validation.py      # Input validation
│   └── middleware.py      # Security middleware
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── logging.py         # Logging configuration
│   ├── config.py          # Configuration management
│   └── database.py        # Database utilities
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── performance/       # Performance tests
├── deployment/            # Deployment configurations
│   ├── helm/              # Helm charts
│   ├── scripts/           # Deployment scripts
│   └── docker/            # Docker configurations
├── docs/                  # Documentation
├── config.yaml           # Configuration file
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container definition
└── main.py               # Application entry point
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/new-complexity-metric

# Make changes
# ... code changes ...

# Run tests
pytest tests/

# Run linting
ruff check .
mypy --strict .

# Commit changes
git add .
git commit -m "feat: add new complexity metric"

# Push and create PR
git push origin feature/new-complexity-metric
```

### 2. Code Quality Standards

#### Linting and Formatting

```bash
# Run ruff for linting and formatting
ruff check .
ruff format .

# Type checking with mypy
mypy --strict .

# Security scanning
bandit -r .
safety check
```

#### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.0.291
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        args: [--strict]
```

### 3. Adding New Complexity Metrics

```python
# models/complexity/new_metric.py
from typing import Dict, Any
import cv2
import numpy as np
from .base import ComplexityAnalyzer

class NewMetricAnalyzer(ComplexityAnalyzer):
    """Analyzer for new complexity metric."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.threshold = config.get('threshold', 0.5)
    
    def analyze_frame(self, frame: np.ndarray) -> float:
        """Analyze single frame for new metric."""
        # Implement your analysis logic
        complexity_score = self._calculate_complexity(frame)
        return complexity_score
    
    def _calculate_complexity(self, frame: np.ndarray) -> float:
        """Calculate complexity score."""
        # Your implementation here
        pass
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Analyze entire video."""
        cap = cv2.VideoCapture(video_path)
        scores = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            score = self.analyze_frame(frame)
            scores.append(score)
        
        cap.release()
        
        return {
            'metric_name': 'new_metric',
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'max_score': np.max(scores),
            'frame_scores': scores
        }
```

### 4. Adding New API Endpoints

```python
# api/main.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

router = APIRouter()

class NewEndpointRequest(BaseModel):
    parameter: str
    options: Dict[str, Any] = {}

class NewEndpointResponse(BaseModel):
    result: str
    metadata: Dict[str, Any]

@router.post("/new-endpoint", response_model=NewEndpointResponse)
async def new_endpoint(
    request: NewEndpointRequest,
    current_user: str = Depends(get_current_user)
):
    """New API endpoint."""
    try:
        # Process request
        result = process_request(request)
        
        # Record metrics
        endpoint_requests.labels(endpoint="new-endpoint").inc()
        
        return NewEndpointResponse(
            result=result,
            metadata={"user": current_user}
        )
    except Exception as e:
        endpoint_errors.labels(endpoint="new-endpoint").inc()
        raise HTTPException(status_code=500, detail=str(e))
```

## Testing

### Test Structure

```
tests/
├── unit/                  # Unit tests
│   ├── test_complexity.py
│   ├── test_models.py
│   └── test_api.py
├── integration/           # Integration tests
│   ├── test_pipeline.py
│   └── test_database.py
├── performance/           # Performance tests
│   ├── test_throughput.py
│   └── test_latency.py
└── fixtures/              # Test fixtures
    ├── sample_videos/
    └── mock_data.py
```

### Writing Tests

```python
# tests/unit/test_complexity.py
import pytest
import numpy as np
from models.complexity.blur import BlurAnalyzer

class TestBlurAnalyzer:
    @pytest.fixture
    def analyzer(self):
        config = {'threshold': 0.5}
        return BlurAnalyzer(config)
    
    @pytest.fixture
    def sample_frame(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_analyze_frame(self, analyzer, sample_frame):
        """Test frame analysis."""
        score = analyzer.analyze_frame(sample_frame)
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_analyze_video(self, analyzer, tmp_path):
        """Test video analysis."""
        # Create test video
        video_path = create_test_video(tmp_path)
        
        result = analyzer.analyze_video(str(video_path))
        
        assert 'metric_name' in result
        assert result['metric_name'] == 'blur'
        assert 'mean_score' in result
        assert isinstance(result['mean_score'], float)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_complexity.py

# Run with verbose output
pytest -v

# Run performance tests
pytest tests/performance/ --benchmark-only
```

### Integration Testing

```python
# tests/integration/test_pipeline.py
import pytest
from pipeline.flows import complexity_prediction_flow
from utils.database import get_database

@pytest.mark.integration
async def test_full_pipeline(sample_video_path):
    """Test complete pipeline flow."""
    # Run pipeline
    result = await complexity_prediction_flow(sample_video_path)
    
    # Verify result structure
    assert 'complexity_scores' in result
    assert 'prediction' in result
    assert 'confidence' in result
    
    # Verify database storage
    db = get_database()
    stored_result = db.predictions.find_one({'video_path': sample_video_path})
    assert stored_result is not None
```

## Performance Optimization

### 1. Profiling

```python
# optimization/performance.py
import cProfile
import pstats
from functools import wraps

def profile_function(func):
    """Decorator to profile function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        
        return result
    return wrapper

# Usage
@profile_function
def analyze_complexity(video_path: str):
    # Your analysis code
    pass
```

### 2. Caching Strategies

```python
# optimization/caching.py
from functools import lru_cache
import redis
import pickle

class CacheManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def cache_result(self, key: str, value: Any, ttl: int = 3600):
        """Cache result with TTL."""
        serialized = pickle.dumps(value)
        self.redis.setex(key, ttl, serialized)
    
    def get_cached_result(self, key: str) -> Any:
        """Get cached result."""
        cached = self.redis.get(key)
        if cached:
            return pickle.loads(cached)
        return None

# Usage
cache = CacheManager(redis_client)

def get_complexity_score(video_path: str) -> float:
    cache_key = f"complexity:{hash(video_path)}"
    
    # Check cache first
    cached_result = cache.get_cached_result(cache_key)
    if cached_result:
        return cached_result
    
    # Calculate if not cached
    result = calculate_complexity(video_path)
    cache.cache_result(cache_key, result)
    
    return result
```

### 3. GPU Optimization

```python
# models/multimodal/model.py
import torch
from torch.cuda.amp import autocast, GradScaler

class OptimizedModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_mixed_precision = config.get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_mixed_precision else None
    
    @autocast()
    def forward(self, x):
        """Forward pass with mixed precision."""
        # Your model forward pass
        return output
    
    def predict_batch(self, batch):
        """Optimized batch prediction."""
        self.eval()
        with torch.no_grad():
            if self.use_mixed_precision:
                with autocast():
                    return self(batch)
            else:
                return self(batch)
```

## Security Guidelines

### 1. Input Validation

```python
# security/validation.py
from pydantic import BaseModel, validator
from typing import Optional
import re

class VideoUploadRequest(BaseModel):
    filename: str
    file_size: int
    content_type: str
    
    @validator('filename')
    def validate_filename(cls, v):
        if not re.match(r'^[a-zA-Z0-9._-]+$', v):
            raise ValueError('Invalid filename format')
        if not v.lower().endswith(('.mp4', '.avi', '.mov')):
            raise ValueError('Unsupported file format')
        return v
    
    @validator('file_size')
    def validate_file_size(cls, v):
        max_size = 100 * 1024 * 1024  # 100MB
        if v > max_size:
            raise ValueError('File too large')
        return v
```

### 2. Authentication

```python
# security/auth.py
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer
import jwt
from datetime import datetime, timedelta

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(security)):
    """Get current authenticated user."""
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### 3. Rate Limiting

```python
# security/middleware.py
from fastapi import Request, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

@limiter.limit("10/minute")
async def analyze_video(request: Request, video_data: VideoUploadRequest):
    """Rate-limited video analysis endpoint."""
    # Your analysis logic
    pass
```

## Contributing

### 1. Code Review Process

1. **Create Feature Branch**: Always work on feature branches
2. **Write Tests**: Ensure >80% test coverage
3. **Documentation**: Update relevant documentation
4. **Code Review**: At least one reviewer approval required
5. **CI/CD**: All checks must pass
6. **Merge**: Squash and merge to main

### 2. Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(complexity): add new texture complexity metric

Implement texture analysis using Local Binary Patterns (LBP)
to measure surface texture complexity in video frames.

- Add TextureAnalyzer class
- Integrate with existing pipeline
- Add comprehensive tests
- Update documentation

Closes #123
```

### 3. Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Performance tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added for new functionality
```

### 4. Development Best Practices

- **Type Hints**: Use type hints for all functions
- **Docstrings**: Document all public functions and classes
- **Error Handling**: Implement proper error handling
- **Logging**: Use structured logging
- **Configuration**: Use configuration files, not hardcoded values
- **Testing**: Write tests before implementing features (TDD)
- **Performance**: Profile critical paths
- **Security**: Validate all inputs, use secure defaults

## Debugging

### 1. Local Debugging

```python
# utils/debugging.py
import logging
import pdb
from functools import wraps

def debug_on_error(func):
    """Decorator to start debugger on error."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            pdb.post_mortem()
            raise
    return wrapper

# Usage
@debug_on_error
def problematic_function():
    # Your code here
    pass
```

### 2. Remote Debugging

```python
# For remote debugging in containers
import debugpy

# Enable remote debugging
debugpy.listen(("0.0.0.0", 5678))
debugpy.wait_for_client()  # Optional: wait for debugger to attach
```

### 3. Performance Debugging

```python
# utils/profiling.py
import time
import psutil
import torch
from contextlib import contextmanager

@contextmanager
def performance_monitor():
    """Monitor performance metrics."""
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_gpu_memory = torch.cuda.memory_allocated()
    
    yield
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used
    
    print(f"Execution time: {end_time - start_time:.2f}s")
    print(f"Memory usage: {(end_memory - start_memory) / 1024**2:.2f}MB")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        end_gpu_memory = torch.cuda.memory_allocated()
        print(f"GPU memory: {(end_gpu_memory - start_gpu_memory) / 1024**2:.2f}MB")

# Usage
with performance_monitor():
    # Your code here
    pass
```

## Resources

- **Documentation**: `/docs` directory
- **API Reference**: http://localhost:8000/docs (when running locally)
- **Monitoring**: http://localhost:9090 (Prometheus)
- **Code Style**: Follow PEP 8 and use ruff for formatting
- **Testing**: pytest for all testing needs
- **CI/CD**: GitHub Actions workflows in `.github/workflows/`

---

*Last updated: 2024-07-17*
