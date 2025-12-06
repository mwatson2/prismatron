"""
Prismatron Path Configuration.

Centralized path management for runtime data storage.
All runtime data is stored outside the source tree.

Directory structure with PRISMATRON_ROOT=/mnt/prismatron:
    /mnt/prismatron/config/       - Configuration files
    /mnt/prismatron/logs/         - Log files
    /mnt/prismatron/media/        - Media files
    /mnt/prismatron/uploads/      - User uploads
    /mnt/prismatron/playlists/    - Playlist definitions
    /mnt/prismatron/patterns/     - Diffusion pattern files
    /mnt/prismatron/conversions/  - Temporary video conversions (cache)

Environment variable:
    PRISMATRON_ROOT - Base directory for all data (default: ~/.local/share/prismatron)
"""

import os
from pathlib import Path

APP_NAME = "prismatron"

# Get root directory from environment or use default
_root_override = os.environ.get("PRISMATRON_ROOT")
if _root_override:
    ROOT_DIR = Path(_root_override)
else:
    _xdg_data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    ROOT_DIR = _xdg_data_home / APP_NAME

# All directories directly under root
CONFIG_DIR = ROOT_DIR / "config"
LOGS_DIR = ROOT_DIR / "logs"
MEDIA_DIR = ROOT_DIR / "media"
UPLOADS_DIR = ROOT_DIR / "uploads"
PLAYLISTS_DIR = ROOT_DIR / "playlists"
PATTERNS_DIR = ROOT_DIR / "patterns"
TEMP_CONVERSIONS_DIR = ROOT_DIR / "conversions"

# Config files
AUDIO_CONFIG_FILE = CONFIG_DIR / "audio_config.json"

# All directories that should be auto-created
_ALL_DIRS = [
    CONFIG_DIR,
    LOGS_DIR,
    MEDIA_DIR,
    UPLOADS_DIR,
    PLAYLISTS_DIR,
    PATTERNS_DIR,
    TEMP_CONVERSIONS_DIR,
]


def ensure_directories() -> None:
    """Create all required directories if they don't exist."""
    for dir_path in _ALL_DIRS:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_log_file_path(filename: str = "prismatron.log") -> Path:
    """Get the full path for a log file."""
    return LOGS_DIR / filename


def get_pattern_file_path(filename: str) -> Path:
    """Get the full path for a diffusion pattern file."""
    return PATTERNS_DIR / filename


# Auto-create directories on module import
ensure_directories()
