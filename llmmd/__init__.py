import os
import tempfile
os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", os.path.join(tempfile.gettempdir(), "unsloth_compiled_cache"))

__version__ = "0.1.0"
