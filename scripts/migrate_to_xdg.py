#!/usr/bin/env python3
"""
Migrate Prismatron data from in-repo directories to /mnt/prismatron.

This script moves runtime data from the old in-repo locations to the SSD
storage location at /mnt/prismatron (avoiding the small SD card at ~/).

Old locations (in repo):          New locations (/mnt/prismatron):
    config/                   ->  /mnt/prismatron/config/
    logs/                     ->  /mnt/prismatron/data/logs/
    media/                    ->  /mnt/prismatron/data/media/
    uploads/                  ->  /mnt/prismatron/data/uploads/
    playlists/                ->  /mnt/prismatron/data/playlists/
    diffusion_patterns/       ->  /mnt/prismatron/data/patterns/
    temp_conversions/         ->  /mnt/prismatron/cache/conversions/
    thumbnails/               ->  /mnt/prismatron/data/thumbnails/

The service file (scripts/prismatron-user.service) sets PRISMATRON_ROOT=/mnt/prismatron
which configures src/paths.py to use these locations.

Usage:
    python scripts/migrate_to_xdg.py [--dry-run]

Options:
    --dry-run   Show what would be migrated without actually moving files
"""

import argparse
import shutil
from pathlib import Path

# Project root (where this script's parent directory is)
PROJECT_ROOT = Path(__file__).parent.parent

# Target location on SSD
PRISMATRON_ROOT = Path("/mnt/prismatron")
CONFIG_DIR = PRISMATRON_ROOT / "config"
DATA_DIR = PRISMATRON_ROOT / "data"
CACHE_DIR = PRISMATRON_ROOT / "cache"

# Data subdirectories
MEDIA_DIR = DATA_DIR / "media"
UPLOADS_DIR = DATA_DIR / "uploads"
PLAYLISTS_DIR = DATA_DIR / "playlists"
LOGS_DIR = DATA_DIR / "logs"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"
PATTERNS_DIR = DATA_DIR / "patterns"

# Cache subdirectories
TEMP_CONVERSIONS_DIR = CACHE_DIR / "conversions"

# Config files
AUDIO_CONFIG_FILE = CONFIG_DIR / "audio_config.json"

# All directories that need to be created
ALL_DIRS = [
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

# Mapping of old paths to new paths
MIGRATIONS = [
    (PROJECT_ROOT / "config" / "audio_config.json", AUDIO_CONFIG_FILE),
    (PROJECT_ROOT / "logs", LOGS_DIR),
    (PROJECT_ROOT / "media", MEDIA_DIR),
    (PROJECT_ROOT / "uploads", UPLOADS_DIR),
    (PROJECT_ROOT / "playlists", PLAYLISTS_DIR),
    (PROJECT_ROOT / "diffusion_patterns", PATTERNS_DIR),
    (PROJECT_ROOT / "temp_conversions", TEMP_CONVERSIONS_DIR),
    (PROJECT_ROOT / "thumbnails", THUMBNAILS_DIR),
]


def migrate_path(old_path: Path, new_path: Path, dry_run: bool = False) -> bool:
    """
    Migrate a single file or directory from old to new location.

    Returns True if migration was performed or would be performed.
    """
    if not old_path.exists():
        return False

    if old_path.is_file():
        if dry_run:
            print(f"  Would copy: {old_path} -> {new_path}")
        else:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            if new_path.exists():
                print(f"  Skipping (already exists): {new_path}")
            else:
                shutil.copy2(old_path, new_path)
                print(f"  Copied: {old_path} -> {new_path}")
        return True

    elif old_path.is_dir():
        # For directories, copy contents
        items = list(old_path.iterdir())
        if not items:
            return False

        if dry_run:
            print(f"  Would migrate directory: {old_path} -> {new_path}")
            for item in items:
                print(f"    - {item.name}")
        else:
            new_path.mkdir(parents=True, exist_ok=True)
            migrated = 0
            skipped = 0
            for item in items:
                dest = new_path / item.name
                if dest.exists():
                    skipped += 1
                    continue
                if item.is_file():
                    shutil.copy2(item, dest)
                else:
                    shutil.copytree(item, dest)
                migrated += 1
            print(f"  Migrated {migrated} items from {old_path} to {new_path}")
            if skipped:
                print(f"  Skipped {skipped} items (already exist)")
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Migrate Prismatron data to SSD at /mnt/prismatron")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated")
    args = parser.parse_args()

    print("Prismatron Data Migration to SSD")
    print("=" * 50)
    print()
    print(f"Target root: {PRISMATRON_ROOT}")
    print(f"  Config:  {CONFIG_DIR}")
    print(f"  Data:    {DATA_DIR}")
    print(f"  Cache:   {CACHE_DIR}")
    print()

    if args.dry_run:
        print("DRY RUN - no files will be moved")
        print()
    else:
        # Create all directories
        print("Creating directories...")
        for dir_path in ALL_DIRS:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {dir_path}")
        print()

    any_migrated = False
    for old_path, new_path in MIGRATIONS:
        if migrate_path(old_path, new_path, args.dry_run):
            any_migrated = True

    print()
    if not any_migrated:
        print("Nothing to migrate - all data is already in XDG locations or old directories are empty.")
    elif args.dry_run:
        print("Run without --dry-run to perform the migration.")
    else:
        print("Migration complete!")
        print()
        print("You can now safely delete the old directories from the repo:")
        print("  rm -rf config/ logs/ media/ uploads/ playlists/ diffusion_patterns/ temp_conversions/ thumbnails/")


if __name__ == "__main__":
    main()
