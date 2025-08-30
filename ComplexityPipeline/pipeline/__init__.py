# pipeline/__init__.py

# This file makes the 'pipeline' directory a Python package.

# By importing key classes and functions here, we make them directly
# accessible from the package level, cleaning up imports in other files.
# For example, instead of 'from pipeline.utils import ConfigManager',
# we can now do 'from pipeline import ConfigManager'.

from .utils import ConfigManager, DatabaseManager, PipelineContext
from .watcher import VideoFileHandler

# You can optionally define __all__ to specify what is exported
# when someone does 'from pipeline import *'
__all__ = [
    "ConfigManager",
    "DatabaseManager",
    "PipelineContext",
    "VideoFileHandler"
]