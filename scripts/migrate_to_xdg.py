#!/usr/bin/env python3
"""
Migrate Prismatron data from in-repo directories to XDG-compliant locations.

This script moves runtime data from the old in-repo locations to the new
XDG Base Directory Specification compliant locations:

Old locations (in repo):          New locations (XDG):
    config/                   ->  ~/.config/prismatron/
    logs/                     ->  ~/.local/share/prismatron/logs/
    media/                    ->  ~/.local/share/prismatron/media/
    uploads/                  ->  ~/.local/share/prismatron/uploads/
    playlists/                ->  ~/.local/share/prismatron/playlists/
    diffusion_patterns/       ->  ~/.local/share/prismatron/diffusion_patterns/
    temp_conversions/         ->  ~/.cache/prismatron/conversions/
    thumbnails/               ->  ~/.local/share/prismatron/thumbnails/

Usage:
    python scripts/migrate_to_xdg.py [--dry-run]

Options:
    --dry-run   Show what would be migrated without actually moving files
"""

import argparse
import shutil
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.paths import (
    AUDIO_CONFIG_FILE,
    CACHE_DIR,
    CONFIG_DIR,
    DATA_DIR,
    LOGS_DIR,
    MEDIA_DIR,
    PATTERNS_DIR,
    PLAYLISTS_DIR,
    TEMP_CONVERSIONS_DIR,
    THUMBNAILS_DIR,
    UPLOADS_DIR,
)

# Project root (where this script's parent directory is)
PROJECT_ROOT = Path(__file__).parent.parent

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
    parser = argparse.ArgumentParser(description="Migrate Prismatron data to XDG directories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated")
    args = parser.parse_args()

    print("Prismatron XDG Migration")
    print("=" * 50)
    print()
    print("New directory locations:")
    print(f"  Config:  {CONFIG_DIR}")
    print(f"  Data:    {DATA_DIR}")
    print(f"  Cache:   {CACHE_DIR}")
    print()

    if args.dry_run:
        print("DRY RUN - no files will be moved")
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
