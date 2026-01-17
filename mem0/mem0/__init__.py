import importlib.metadata

# Try to get version from package metadata, fallback to local version if not installed
try:
    __version__ = importlib.metadata.version("mem0ai")
except importlib.metadata.PackageNotFoundError:
    # Using local copy - set version manually
    __version__ = "1.0.1"

from mem0.client.main import AsyncMemoryClient, MemoryClient  # noqa
from mem0.memory.main import AsyncMemory, Memory  # noqa
