# VFX Pipeline Quick Start Guide

This guide gets you up and running with the VFX Shot Complexity Prediction Pipeline in minutes.

## 🚀 Choose Your Setup

### Option 1: Local Development (Easiest)

```bash
# 1. Clone and setup
git clone <your-repo>
cd test_app

# 2. Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start local services (MongoDB + Redis)
docker-compose up -d mongodb redis

# 5. Run the pipeline
python main.py
```

**That's it!** Drop a video file into `input_files/` and watch the magic happen.

### Option 2: Full Docker Stack

```bash
# 1. Build and run everything
docker-compose up --build

# 2. Check it's working
curl http://localhost:8000/health
```

### Option 3: Production Kubernetes

```bash
# 1. Make scripts executable (Linux/WSL)
chmod +x deployment/scripts/*.sh

# 2. Deploy to staging
./deployment/scripts/deploy.sh staging v1.0.0

# 3. Monitor deployment
./deployment/scripts/monitor.sh staging
```

## 🔧 What Each Command Does

### Local Development Commands

```bash
# Start the main pipeline (file watcher + processing)
python main.py

# Start just the API server
uvicorn api.main:app --reload --port 8000

# Run tests
pytest

# Check code quality
ruff check .
mypy .
```

### Docker Commands

```bash
# Build the container
docker build -t vfx-pipeline .

# Run with volume mounting
docker run -v "$PWD/input_files:/app/input_files" vfx-pipeline

# Full stack with database
docker-compose up --build
```

### Kubernetes Commands

```bash
# Deploy to staging
./deployment/scripts/deploy.sh staging v1.0.0

# Deploy to production
./deployment/scripts/deploy.sh production v1.0.0

# Monitor deployment
./deployment/scripts/monitor.sh production

# Rollback if needed
./deployment/scripts/rollback.sh production
```

## 📁 What to Expect

### File Structure After Setup
```
test_app/
├── input_files/           # 👈 DROP YOUR VIDEOS HERE
├── temp_outputs/          # Processing artifacts
├── logs/                  # Application logs
└── models/               # ML models and weights
```

### Processing Flow
1. **Drop video** → `input_files/sample.mp4`
2. **Watchdog detects** → Triggers Prefect flow
3. **Analysis runs** → 9 complexity metrics calculated
4. **Features extracted** → CNN processes video frames
5. **Classification** → Multimodal RNN predicts difficulty
6. **Results stored** → MongoDB + logs

### Expected Output
```json
{
  "video_path": "input_files/sample.mp4",
  "complexity_scores": {
    "blur": 0.23,
    "motion": 0.67,
    "zoom": 0.45,
    // ... 6 more metrics
  },
  "prediction": {
    "class": "Medium",
    "confidence": 0.87,
    "probabilities": {
      "Easy": 0.05,
      "Medium": 0.87,
      "Hard": 0.08
    }
  }
}
```

## 🌐 Access Points

Once running, you can access:

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics
- **MongoDB**: mongodb://localhost:27017
- **Redis**: redis://localhost:6379

## 🔍 Troubleshooting

### Common Issues

**"ModuleNotFoundError"**
```bash
# Make sure virtual environment is activated
pip install -r requirements.txt
```

**"Connection refused" (Database)**
```bash
# Start database services
docker-compose up -d mongodb redis
```

**"Permission denied" (Scripts)**
```bash
# On Windows, run:
powershell -ExecutionPolicy Bypass -File deployment/scripts/make-executable.ps1

# On Linux/WSL:
chmod +x deployment/scripts/*.sh
```

**"CUDA not available"**
```bash
# CPU-only mode (add to environment)
export CUDA_VISIBLE_DEVICES=""
```

### Check Everything is Working

```bash
# 1. Check services
docker-compose ps

# 2. Check API
curl http://localhost:8000/health

# 3. Check database
docker-compose exec mongodb mongosh --eval "db.adminCommand('ping')"

# 4. Check logs
docker-compose logs -f vfx-pipeline
```

## 🎯 Quick Test

Want to test everything works? Here's a 30-second test:

```bash
# 1. Start everything
docker-compose up -d

# 2. Wait for services to start
sleep 10

# 3. Check health
curl http://localhost:8000/health

# 4. Process a test video (if you have one)
cp /path/to/your/video.mp4 input_files/

# 5. Watch the logs
docker-compose logs -f vfx-pipeline
```

## 📚 Next Steps

- **Development**: Read [Developer Guide](docs/DEVELOPER_GUIDE.md)
- **Production**: Follow [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- **Operations**: Use [Operational Runbook](docs/OPERATIONAL_RUNBOOK.md)

## 🆘 Need Help?

1. **Check logs**: `docker-compose logs vfx-pipeline`
2. **Check health**: `curl http://localhost:8000/health`
3. **Read docs**: Start with [Developer Guide](docs/DEVELOPER_GUIDE.md)
4. **Debug mode**: Set `LOG_LEVEL=DEBUG` in environment

---

**🎬 Ready to analyze some VFX shots? Drop a video in `input_files/` and watch the pipeline work!**
