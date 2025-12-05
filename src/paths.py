"""
Prismatron Path Configuration.

Centralized path management following XDG Base Directory Specification.
All runtime data is stored outside the source tree in standard locations.

Directory structure (default):
    ~/.config/prismatron/           - Configuration files
    ~/.local/share/prismatron/      - Runtime data (media, uploads, playlists, logs, patterns)
    ~/.cache/prismatron/            - Temporary/cache files (video conversions)

Environment variable overrides:
    PRISMATRON_ROOT         - Override all directories (sets config/, data/, cache/ under this path)
    PRISMATRON_CONFIG_DIR   - Override config directory only
    PRISMATRON_DATA_DIR     - Override data directory only
    PRISMATRON_CACHE_DIR    - Override cache directory only

Example for SSD storage at /mnt/prismatron:
    Environment="PRISMATRON_ROOT=/mnt/prismatron"

This creates:
    /mnt/prismatron/config/
    /mnt/prismatron/data/
    /mnt/prismatron/cache/
"""

import os
from pathlib import Path

APP_NAME = "prismatron"

# Check for unified root override first
_root_override = os.environ.get("PRISMATRON_ROOT")

if _root_override:
    # Single root path for all data (e.g., /mnt/prismatron)
    _root = Path(_root_override)
    _default_config = _root / "config"
    _default_data = _root / "data"
    _default_cache = _root / "cache"
else:
    # XDG Base Directory defaults
    _xdg_config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    _xdg_data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    _xdg_cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    _default_config = _xdg_config_home / APP_NAME
    _default_data = _xdg_data_home / APP_NAME
    _default_cache = _xdg_cache_home / APP_NAME

# Individual directory overrides take precedence
CONFIG_DIR = Path(os.environ.get("PRISMATRON_CONFIG_DIR", _default_config))
DATA_DIR = Path(os.environ.get("PRISMATRON_DATA_DIR", _default_data))
CACHE_DIR = Path(os.environ.get("PRISMATRON_CACHE_DIR", _default_cache))

# Config subdirectories and files
AUDIO_CONFIG_FILE = CONFIG_DIR / "audio_config.json"

# Data subdirectories
MEDIA_DIR = DATA_DIR / "media"
UPLOADS_DIR = DATA_DIR / "uploads"
PLAYLISTS_DIR = DATA_DIR / "playlists"
LOGS_DIR = DATA_DIR / "logs"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"
PATTERNS_DIR = DATA_DIR / "patterns"

# Cache subdirectories
TEMP_CONVERSIONS_DIR = CACHE_DIR / "conversions"

# All directories that should be auto-created
_ALL_DIRS = [
    CONFIG_DIR,
    DATA_DIR,
    CACHE_DIR,
    MEDIA_DIR,
    UPLOADS_DIR,
    PLAYLISTS_DIR,
    LOGS_DIR,
    THUMBNAILS_DIR,
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
