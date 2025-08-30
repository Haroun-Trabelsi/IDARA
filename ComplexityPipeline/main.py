# main.py

import logging
import os
import uuid
from pathlib import Path
import sys
from typing import Optional
from estimation import VFXBidPredictor
from estimation import VFXTemplateAnalyzer
import joblib
import torch
import uvicorn
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile , Form , Request
from prometheus_fastapi_instrumentator import Instrumentator
from watchdog.observers import Observer
from pipeline.job_tracker  import job_store

# --- Step 1: Import from our new, clean 'pipeline' package ---
from pipeline import ConfigManager, DatabaseManager, VideoFileHandler

# We also need the model class definition
from model_architecture import MultimodalRNN

# --- Enhanced Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)




def initialize_configuration() -> Optional[ConfigManager]:
    """
    Initialize configuration from config.yaml file.
    
    Returns:
        ConfigManager: Configuration manager instance if successful, None otherwise.
    """
    logger.info("Initializing configuration manager...")
    try:
        config = ConfigManager()
        logger.info("Configuration loaded successfully from config.yaml")
        return config
    except FileNotFoundError as e:
        logger.critical(
            "Configuration file not found. Please ensure 'config.yaml' exists in the project root.",
            exc_info=True
        )
        return None
    except Exception as e:
        logger.critical(
            "Unexpected error occurred while loading configuration: %s",
            str(e),
            exc_info=True
        )
        return None


def initialize_database(config: ConfigManager) -> Optional[DatabaseManager]:
    """
    Initialize database connection using configuration.
    
    Args:
        config: Configuration manager instance.
        
    Returns:
        DatabaseManager: Database manager instance if successful, None otherwise.
    """
    logger.info("Initializing database connection...")
    try:
        db_manager = DatabaseManager(config)
        logger.info("Database connection established successfully")
        return db_manager
    except Exception as e:
        logger.critical(
            "Failed to establish database connection: %s",
            str(e),
            exc_info=True
        )
        return None


def load_ml_components(config: ConfigManager) -> Optional[tuple]:
    """
    Load machine learning components including model, scaler, and imputer.
    
    Args:
        config: Configuration manager instance.
        
    Returns:
        tuple: (model, scaler, imputer, device) if successful, None otherwise.
    """
    logger.info("Loading machine learning components...")
    
    model_config = config.get("multimodal_model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)
    bid_model_path = "vfx_bid_predictor_your_data.joblib" 
    try:
        # Load the entire checkpoint dictionary first
        checkpoint_path = model_config['model_path']
        logger.debug("Loading checkpoint from: %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Create an instance of the model architecture
        if 'config' in checkpoint:
            inference_config = checkpoint['config']
            logger.info("Using model configuration saved within the checkpoint file")
        else:
            inference_config = config.get('multimodal_model')
            logger.warning("Checkpoint does not contain a 'config'. Using config from yaml")
            
        model = MultimodalRNN(inference_config)
        
        # Extract checkpoint state_dict (try common keys)
        raw_state_dict = (
            checkpoint.get('model_state_dict')
            or checkpoint.get('state_dict')
            or checkpoint
        )
        
        # Load compatible weights, ignoring shape mismatches
        model.load_state_dict(raw_state_dict, strict=False)
        model.eval()
        model.to(device)
        logger.info("Model loaded and prepared for inference")

        # --- Optional: Load scaler & imputer if provided ---
        scaler_path = model_config.get('scaler_path')
        imputer_path = model_config.get('imputer_path')

        class _Identity:
            """Fallback transformer that returns data unchanged."""
            def fit(self, X):
                return self
            def transform(self, X):
                return X
            def fit_transform(self, X):
                return X

        # Load scaler if path exists, else use identity
        if scaler_path and Path(scaler_path).exists():
            logger.debug("Loading scaler from: %s", scaler_path)
            scaler = joblib.load(scaler_path)
        else:
            logger.warning("Scaler not found or not provided. Using identity transformer.")
            scaler = _Identity()

        # Load imputer if path exists, else use identity
        if imputer_path and Path(imputer_path).exists():
            logger.debug("Loading imputer from: %s", imputer_path)
            imputer = joblib.load(imputer_path)
        else:
            logger.warning("Imputer not found or not provided. Using identity transformer.")
            imputer = _Identity()
        # --- NEW: Load bid predictor ---
        bid_predictor = None
        if bid_model_path and Path(bid_model_path).exists():
            logger.info(f"Loading bid predictor model from {bid_model_path}")
            bid_predictor = VFXBidPredictor()
            bid_predictor.load_model(bid_model_path)
        else:
            logger.warning("Bid predictor path missing or file not found")
        logger.info("All ML components loaded successfully (model + optional transformers)")
        return model, scaler, imputer, device, bid_predictor

    except FileNotFoundError as e:
        logger.critical(
            "Required model file not found: %s. Please ensure all model/scaler/imputer "
            "paths in 'config.yaml' are correct and the files exist in 'inference_model/'",
            str(e)
        )
        return None
    except Exception as e:
        logger.critical(
            "Failed to load ML components: %s",
            str(e),
            exc_info=True
        )
        return None


def setup_file_watcher(config: ConfigManager, db_manager: DatabaseManager, 
                      model: torch.nn.Module, scaler, imputer,bid) -> Optional[Observer]:
    """
    Set up and start the file watcher for monitoring video files.
    
    Args:
        config: Configuration manager instance.
        db_manager: Database manager instance.
        model: Loaded ML model.
        scaler: Loaded scaler.
        imputer: Loaded imputer.
        
    Returns:
        Observer: File watcher observer if successful, None otherwise.
    """
    logger.info("Setting up file watcher...")
    
    try:
        event_handler = VideoFileHandler(config, db_manager, model, scaler, imputer)
        observer = Observer()
        
        watch_dir = Path(config.get("watchdog.directory"))
        logger.debug("Watch directory (raw config): %s", watch_dir)
        logger.debug("Watch directory (absolute): %s", watch_dir.resolve())
        logger.debug("Watch directory exists: %s", watch_dir.exists())
        
        # Create directory if it doesn't exist
        watch_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Watch directory after creation: %s", watch_dir.exists())
        
        observer.schedule(event_handler, str(watch_dir), recursive=True)
        observer.start()
        
        logger.info("File watcher started. Monitoring directory: %s", watch_dir.resolve())
        return observer
        
    except Exception as e:
        logger.critical(
            "Failed to set up file watcher: %s",
            str(e),
            exc_info=True
        )
        return None


def create_fastapi_app(config: ConfigManager, watch_dir: Path, bid_predictor) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        config: Configuration manager instance.
        watch_dir: Directory being watched for files.
        
    Returns:
        FastAPI: Configured FastAPI application.
    """
    logger.info("Creating FastAPI application...")

    app = FastAPI(title="VFX Processing Pipeline API")

    # Prometheus instrumentation
    Instrumentator().instrument(app).expose(app, include_in_schema=False)
    logger.info("Prometheus instrumentation configured")

    # --- CORS middleware for local frontend development ---
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS middleware enabled for http://localhost:3000")

    # Optional API-key security (disabled if API_KEY env var is empty)
    API_KEY = os.getenv("API_KEY", "")
    if API_KEY:
        logger.info("API key authentication enabled")
    else:
        logger.warning("API key authentication disabled - no API_KEY environment variable set")

    @app.middleware("http")
    async def verify_api_key(request: Request, call_next):
        if API_KEY:
            provided = request.headers.get("X-API-Key")
            if provided != API_KEY:
                logger.warning(
                    "Unauthorized API access attempt from %s",
                    request.client.host if request.client else "unknown"
                )
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)

    @app.get("/health")
    def health_check():
        """A simple endpoint to confirm the service is running."""
        logger.debug("Health check endpoint accessed")
        return {"status": "ok", "watching": str(watch_dir.resolve())}

    # --- Upload endpoint ---
    from pipeline.orchestrater import parse_filename  # add at the top
    @app.post("/upload_video")
    async def upload_video(
        file: UploadFile = File(...),
        original_filename: str = Form(...),
        description: str = Form(...),
        notes_count: int = Form(...)
    ):
        job_id = str(uuid.uuid4())
        filename = f"{job_id}_{original_filename}"
        save_path = watch_dir / filename

        try:
            with open(save_path, "wb") as buffer:
                buffer.write(await file.read())
            logger.info(f"original filename: {original_filename}")
            logger.info(f"Uploaded video saved to watch folder: {save_path}")

            # Parse filename once here
            sequence, task, project = parse_filename(filename)

            # Store metadata in job_store (no predictor run here)
            job_store[job_id] = {
                "status": "processing",
                "description": description,
                "notes_count": notes_count,
                "sequence": sequence,
                "task": task,
                "project": project,
                "predicted_vfx_hours": None,  # will be filled after processing
                "result": None,
                "filename": filename
            }

            return {"status": "success", "job_id": job_id}

        except Exception as e:
            logger.error(f"Failed to save uploaded video: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Processing failed."}
            )

    @app.get("/result/{job_id}")
    async def get_result(job_id: str):
        job = job_store.get(job_id)
        if not job:
            return JSONResponse(status_code=404, content={"error": "Job ID not found"})

        if job["status"] == "processing":
            return {"status": "processing"}
        else:
            # If result is None or missing, return empty dict instead of error
            return {
                "status": "done",
                "result": job.get("result") or {}
            }
    return app
     

def start_server(app: FastAPI, config: ConfigManager) -> None:
    """
    Start the FastAPI server with uvicorn.
    
    Args:
        app: Configured FastAPI application.
        config: Configuration manager instance.
    """
    api_config = config.get("api")
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8089)
    
    logger.info("Starting FastAPI server on %s:%s", host, port)
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    except Exception as e:
        logger.critical(
            "Server failed to start: %s",
            str(e),
            exc_info=True
        )


def main() -> None:
    """
    Initialize and run the entire VFX processing service.
    
    This is the main entry point for the application that orchestrates:
    1. Configuration loading
    2. Database initialization
    3. ML component loading
    4. File watcher setup
    5. FastAPI server startup
    
    The function handles all initialization steps with proper error handling
    and logging. If any critical component fails to initialize, the application
    will exit gracefully with appropriate error messages.
    
    Returns:
        None
        
    Raises:
        SystemExit: If any critical component fails to initialize.
    """
    logger.info("=" * 50)
    logger.info("Starting the VFX Processing Pipeline...")
    logger.info("=" * 50)
    
    # Step 1: Initialize configuration
    config = initialize_configuration()
    if config is None:
        logger.critical("Failed to initialize configuration. Shutting down.")
        sys.exit(1)
    
    # Step 2: Initialize database
    db_manager = initialize_database(config)
    if db_manager is None:
        logger.critical("Failed to initialize database. Shutting down.")
        sys.exit(1)
    
    # Step 3: Load ML components
    ml_components = load_ml_components(config)
    if ml_components is None:
        logger.critical("Failed to load ML components. Shutting down.")
        sys.exit(1)
    
    model, scaler, imputer, device, bid_predictor = ml_components
    
    # Step 4: Set up file watcher
    observer = setup_file_watcher(config, db_manager, model, scaler, imputer, bid_predictor)
    if observer is None:
        logger.critical("Failed to set up file watcher. Shutting down.")
        sys.exit(1)
    
    # Step 5: Create and start FastAPI application
    watch_dir = Path(config.get("watchdog.directory"))
    app = create_fastapi_app(config, watch_dir, bid_predictor)
    
    try:
        start_server(app, config)
    except Exception as e:
        logger.critical(
            "Unexpected error during server startup: %s",
            str(e),
            exc_info=True
        )
    finally:
        # Graceful shutdown
        logger.info("Shutting down file watcher...")
        observer.stop()
        observer.join()
        logger.info("Pipeline shut down gracefully")


if __name__ == "__main__":
    # Create a dummy temp folder if it doesn't exist for the overlap script
    # This is a small workaround to ensure one of the complexity scripts doesn't fail.
    os.makedirs("temp_outputs", exist_ok=True)
    main()