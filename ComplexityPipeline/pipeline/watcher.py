# pipeline/watcher.py

import asyncio
import logging
from pathlib import Path
from threading import Thread

from watchdog.events import FileSystemEventHandler
import zipfile
import shutil
import tempfile

# Import the main flow from the orchestrator.
# This is the crucial link: the watcher calls the orchestrator.
from .orchestrater import process_video_flow

logger = logging.getLogger(__name__)

class VideoFileHandler(FileSystemEventHandler):
    """
    Watches a directory for new video files and triggers the
    Prefect processing flow for each one.
    """
    def __init__(self, config, db, model, scaler, imputer):
        self.config = config
        self.db = db
        self.model = model
        self.scaler = scaler
        self.imputer = imputer

        # The file watcher runs in its own thread, but the tasks it triggers
        # need to run in an asyncio event loop. We create and manage that
        # loop here, running it in a separate background thread.
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self.run_loop, daemon=True)
        self.thread.start()
        logger.info("Asyncio event loop for pipeline tasks started.")
        # Track already-submitted paths to avoid duplicates
        self._processed: set[str] = set()

    def run_loop(self):
        """The target for the background thread, which just runs the event loop."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _submit_video(self, file_path: Path):
        """Submit a single video file to the Prefect flow if not done already."""
        if str(file_path) in self._processed:
            return
        # double-check extension
        if file_path.suffix.lower() not in ('.mp4', '.mov', '.avi'):
            return
        logger.info(f"Submitting video for processing: {file_path}")
        self._processed.add(str(file_path))
        from .orchestrater import process_video_flow  # noqa: E402
        asyncio.run_coroutine_threadsafe(
            process_video_flow(
                file_path,
                self.config,
                self.db,
                self.model,
                self.scaler,
                self.imputer,
            ),
            self.loop,
        )

    def _handle_directory(self, dir_path: Path):
        for child in dir_path.rglob('*'):
            if child.is_file():
                self._submit_video(child)

    def _handle_zip(self, zip_path: Path):
        with zipfile.ZipFile(zip_path, 'r') as zf:
            temp_dir = Path(tempfile.mkdtemp(prefix='vfx_zip_'))
            zf.extractall(path=temp_dir)
            logger.info("Extracted zip %s to %s", zip_path.name, temp_dir)
            self._handle_directory(temp_dir)
            # Optionally clean up temp_dir later – leave for now

    def on_created(self, event):
        """
        This method is called by the watchdog observer when a new file is created.
        """
        file_path = Path(event.src_path)

        if event.is_directory:
            # A new directory appeared – scan it recursively
            self._handle_directory(file_path)
            return

        # If it's a zip archive, extract and process
        if file_path.suffix.lower() == '.zip':
            self._handle_zip(file_path)
            return

        # If it's a single video file, submit directly
        self._submit_video(file_path)