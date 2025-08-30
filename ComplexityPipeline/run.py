#!/usr/bin/env python3
"""
VFX Pipeline Runner Script
Simple script to start the VFX pipeline with different configurations.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """Run a command and handle errors."""
    print(f"🔧 Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=check)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    # Check Python
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        return False
    
    # Check Docker
    if not run_command("docker --version", check=False):
        print("❌ Docker not found. Install Docker to use containerized mode.")
        return False
    
    # Check if virtual environment exists
    venv_path = Path("venv") / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    if not venv_path.exists():
        print("⚠️  Virtual environment not found. Creating one...")
        if not run_command(f"{sys.executable} -m venv venv"):
            return False
    
    print("✅ Dependencies check passed")
    return True

def setup_environment():
    """Set up the development environment."""
    print("🛠️  Setting up environment...")
    
    # Activate virtual environment and install dependencies
    if os.name == "nt":  # Windows
        activate_cmd = "venv\\Scripts\\activate && pip install -r requirements.txt"
    else:  # Linux/Mac
        activate_cmd = "source venv/bin/activate && pip install -r requirements.txt"
    
    if not run_command(activate_cmd):
        print("❌ Failed to install dependencies")
        return False
    
    print("✅ Environment setup complete")
    return True

def start_services():
    """Start required services (MongoDB, Redis)."""
    print("🚀 Starting services...")
    
    if not run_command("docker-compose up -d mongodb redis"):
        print("❌ Failed to start services")
        return False
    
    # Wait for services to be ready
    print("⏳ Waiting for services to start...")
    time.sleep(10)
    
    # Check if services are running
    if run_command("docker-compose ps", check=False):
        print("✅ Services started successfully")
        return True
    else:
        print("❌ Services failed to start")
        return False

def run_local():
    """Run the pipeline locally."""
    print("🎬 Starting VFX Pipeline (Local Mode)")
    
    if not check_dependencies():
        return False
    
    if not setup_environment():
        return False
    
    if not start_services():
        return False
    
    print("🎯 Pipeline ready! Drop videos in 'input_files/' directory")
    print("📊 API available at: http://localhost:8000")
    print("📚 Documentation: http://localhost:8000/docs")
    print("🔍 Health check: http://localhost:8000/health")
    print("\n🔄 Starting main pipeline...")
    
    # Start the main pipeline
    if os.name == "nt":  # Windows
        run_command("venv\\Scripts\\activate && python main.py")
    else:  # Linux/Mac
        run_command("source venv/bin/activate && python main.py")

def run_docker():
    """Run the pipeline with Docker."""
    print("🐳 Starting VFX Pipeline (Docker Mode)")
    
    if not run_command("docker --version", check=False):
        print("❌ Docker not available")
        return False
    
    print("🔨 Building Docker image...")
    if not run_command("docker-compose build"):
        return False
    
    print("🚀 Starting full stack...")
    if not run_command("docker-compose up"):
        return False

def run_api_only():
    """Run only the API server."""
    print("🌐 Starting API Server Only")
    
    if not check_dependencies():
        return False
    
    if not setup_environment():
        return False
    
    if not start_services():
        return False
    
    print("🎯 API Server starting...")
    print("📊 Available at: http://localhost:8000")
    print("📚 Documentation: http://localhost:8000/docs")
    
    # Start API server
    if os.name == "nt":  # Windows
        run_command("venv\\Scripts\\activate && uvicorn api.main:app --reload --port 8000")
    else:  # Linux/Mac
        run_command("source venv/bin/activate && uvicorn api.main:app --reload --port 8000")

def run_tests():
    """Run the test suite."""
    print("🧪 Running Tests")
    
    if not setup_environment():
        return False
    
    if not start_services():
        return False
    
    # Run tests
    if os.name == "nt":  # Windows
        run_command("venv\\Scripts\\activate && pytest -v")
    else:  # Linux/Mac
        run_command("source venv/bin/activate && pytest -v")

def deploy_staging():
    """Deploy to staging environment."""
    print("🚀 Deploying to Staging")
    
    # Make scripts executable on Linux/WSL
    if os.name != "nt":
        run_command("chmod +x deployment/scripts/*.sh")
    
    # Deploy
    if not run_command("./deployment/scripts/deploy.sh staging v1.0.0"):
        print("❌ Staging deployment failed")
        return False
    
    print("✅ Staging deployment successful")
    print("🔍 Monitor with: ./deployment/scripts/monitor.sh staging")

def deploy_production():
    """Deploy to production environment."""
    print("🚀 Deploying to Production")
    
    # Confirmation
    response = input("⚠️  Are you sure you want to deploy to PRODUCTION? (yes/no): ")
    if response.lower() != "yes":
        print("❌ Production deployment cancelled")
        return False
    
    # Make scripts executable on Linux/WSL
    if os.name != "nt":
        run_command("chmod +x deployment/scripts/*.sh")
    
    # Deploy
    if not run_command("./deployment/scripts/deploy.sh production v1.0.0"):
        print("❌ Production deployment failed")
        return False
    
    print("✅ Production deployment successful")
    print("🔍 Monitor with: ./deployment/scripts/monitor.sh production")

def show_status():
    """Show current system status."""
    print("📊 VFX Pipeline Status")
    print("=" * 50)
    
    # Check services
    print("\n🔧 Services:")
    run_command("docker-compose ps", check=False)
    
    # Check API
    print("\n🌐 API Health:")
    run_command("curl -s http://localhost:8000/health || echo 'API not available'", check=False)
    
    # Check disk space
    print("\n💾 Disk Usage:")
    if os.name == "nt":
        run_command("dir input_files", check=False)
    else:
        run_command("ls -la input_files/", check=False)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="VFX Pipeline Runner")
    parser.add_argument("mode", choices=[
        "local", "docker", "api", "test", 
        "deploy-staging", "deploy-production", "status"
    ], help="Run mode")
    
    args = parser.parse_args()
    
    print("🎬 VFX Shot Complexity Prediction Pipeline")
    print("=" * 50)
    
    if args.mode == "local":
        run_local()
    elif args.mode == "docker":
        run_docker()
    elif args.mode == "api":
        run_api_only()
    elif args.mode == "test":
        run_tests()
    elif args.mode == "deploy-staging":
        deploy_staging()
    elif args.mode == "deploy-production":
        deploy_production()
    elif args.mode == "status":
        show_status()

if __name__ == "__main__":
    main()
