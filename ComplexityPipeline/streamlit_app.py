"""Streamlit front-end for the Video Difficulty Classification pipeline.

It lets users: 
  • Upload a video (saved into the directory watched by the pipeline)
  • See the predicted difficulty class and confidence once the backend inserts a result into MongoDB.
  • Browse recent processed videos.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import streamlit as st
from pymongo import MongoClient

# Internal utilities
from pipeline.utils import ConfigManager

st.set_page_config(page_title="Video Difficulty Classifier", page_icon="🎬")
st.title("🎬 Video Difficulty Classifier")

# -----------------------------------------------------------------------------
# Initialize configuration and DB connection
# -----------------------------------------------------------------------------

cfg = ConfigManager()

mongo_cfg = cfg.get("mongodb", {}) or {}
collection = None
if mongo_cfg.get("uri"):
    try:
        _client = MongoClient(mongo_cfg["uri"], serverSelectionTimeoutMS=3000)
        _client.admin.command("ping")
        _db = _client[mongo_cfg.get("database")]
        collection = _db[mongo_cfg.get("collection")]
    except Exception as exc:
        st.warning(f"MongoDB unreachable ({exc}). Live result tracking disabled.")

# Determine the folder watched by watchdog (default: input_files/)
watch_dir = Path(cfg.get("watch.input_dir", "input_files"))
watch_dir.mkdir(exist_ok=True, parents=True)

# -----------------------------------------------------------------------------
# Sidebar – recent results
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("Recent Results")
    if collection is not None:
        docs = list(
            collection.find().sort("timestamp", -1).limit(10)
        )
        if docs:
            for d in docs:
                st.markdown(
                    f"**{d['filename']}** → {d['classification_result']['predicted_class']} "
                    f"({d['classification_result']['confidence']})"
                )
        else:
            st.write("No results yet.")
    else:
        st.write("MongoDB not connected.")

# -----------------------------------------------------------------------------
# Main – upload widget
# -----------------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "avi"])

if uploaded_file is not None:
    unique_name = f"{uuid.uuid4().hex[:8]}_{uploaded_file.name}"
    target_path = watch_dir / unique_name

    # Save uploaded bytes
    with open(target_path, "wb") as out_file:
        out_file.write(uploaded_file.read())

    st.success(f"✅ File saved to `{target_path}`. The backend pipeline will process it shortly.")

    # If DB available, poll for result
    if collection is not None:
        with st.spinner("⏳ Waiting for classification result…"):
            result_doc = None
            timeout_secs = 300  # wait up to 5 minutes
            poll_interval = 3
            for _ in range(0, timeout_secs, poll_interval):
                result_doc = collection.find_one({"filename": unique_name})
                if result_doc:
                    break
                time.sleep(poll_interval)

        if result_doc:
            cls = result_doc["classification_result"]
            st.subheader("Result")
            st.success(f"**{cls['predicted_class']}** – confidence: {cls['confidence']}")
            st.json(cls["probabilities"])
        else:
            st.warning("Timeout waiting for a result. Check backend logs.")
    else:
        st.info("Upload succeeded, but live result display requires MongoDB connection.")
