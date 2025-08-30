# pipeline/orchestrater.py

import logging
import time
import uuid
from pathlib import Path
from pipeline.job_tracker import job_store
from prefect import flow, get_run_logger
from estimation import VFXBidPredictor

# Import your tasks
from .tasks import (
    analyze_complexity_task,
    classify_difficulty_task,
    extract_sequence_features_task,
)

from .utils import ConfigManager, DatabaseManager, PipelineContext

import re

logger = logging.getLogger(__name__)

# Initialize and load your predictor **once** here
predictor = VFXBidPredictor()
predictor.load_model("vfx_bid_predictor_your_data.joblib")  # update with your path

def parse_filename(filename: str):
    # Your existing parse_filename function
    name = filename.rsplit('.', 1)[0]
    pattern = re.compile(
        r"^([a-f0-9\-]+)_"
        r"Sequence\s*-\s*([\w]+)\s+"
        r"Task\s*-\s*([\w_]+)\s+"
        r"Project Name\s*-\s*([\w]+)$"
    )
    match = pattern.match(name)
    if match:
        uuid_, sequence, task, project = match.groups()
        return sequence, task, project
    return None, None, None

@flow(name="VFX Video Processing Flow")
async def process_video_flow(
    file_path: Path,
    config: ConfigManager,
    db: DatabaseManager,
    model,
    scaler,
    imputer,
    bid_predictor=None  # you can override if you want, else defaults to global predictor
):
    logger = get_run_logger()
    context = PipelineContext(file_path)

    if bid_predictor is None:
        bid_predictor = predictor  # use preloaded global predictor

    try:
        context = await analyze_complexity_task(context, config)
        context = await extract_sequence_features_task(context, config)
        context = await classify_difficulty_task(context, config, model, scaler, imputer)
        job_id = context.file_path.stem.split("_", 1)[0]
        job_data = job_store.get(job_id, {})
        sequence = job_data.get("sequence")
        shot = job_data.get("task")
        project = job_data.get("project")
        description = job_data.get("description")
        notes_count = job_data.get("notes_count", len(context.complexity_scores) or 0)

        print("context final _result : ", context.final_result)

        final_data_to_save = {
            "id": str(uuid.uuid4()),
            "filename": context.file_path.name,
            "filepath": str(context.file_path),
            "timestamp": time.time(),
            "complexity_scores": context.complexity_scores,
            "classification_result": context.final_result,
            "processing_time_seconds": round(time.time() - context.start_time, 2),
            "sequence": sequence,
            "shot": shot,
            "notes_count": notes_count,
            "description": description,
            "project": project,
            "version": 5,
            "company": "Vision Age",
        }

        ## Adapt predicted_class string to your model's expected categories
        mapping = {
            "Easy": "Low",
            "Medium": "Medium",
            "Hard": "High"
        }

        predicted_class_raw = context.final_result.get("predicted_class", "Easy")
        predicted_class = mapping.get(predicted_class_raw, "Low")

        if bid_predictor:
            try:
                predicted_hours = bid_predictor.predict_new_descriptions(
                    [f"{sequence} {shot} {project}"],
                    {
                        "complexity_task": [predicted_class],
                        "task_name": [shot or "UnknownTask"],
                        "project_name": [project or "UnknownProject"],
                        "notes_count": [notes_count]
                    }
                )
                predicted_vfx_hours = round(predicted_hours[0], 1)
                final_data_to_save["predicted_vfx_hours"] = predicted_vfx_hours
                job_data["predicted_vfx_hours"] = predicted_vfx_hours
            except Exception as e:
                logger.warning(f"Bid prediction failed: {e}", exc_info=True)
                final_data_to_save["predicted_vfx_hours"] = None
        else:
            final_data_to_save["predicted_vfx_hours"] = None


        # Save to DB
        await db.insert_result(final_data_to_save)

        # Convert Mongo ObjectId to string to avoid JSON serialization error
        if "_id" in final_data_to_save:
            final_data_to_save["_id"] = str(final_data_to_save["_id"])

        job_data["status"] = "done"
        job_data["result"] = final_data_to_save
        job_store[job_id] = job_data

        print(project, sequence, shot, description, notes_count)
        print(final_data_to_save)
        logger.info(f"SUCCESS: Pipeline completed for {context.file_path.name}")

    except Exception as e:
        logger.error(f"FAIL: The pipeline failed for {file_path.name}. Error: {e}", exc_info=True)
        job_data["status"] = "failed"
        job_data["error"] = str(e)