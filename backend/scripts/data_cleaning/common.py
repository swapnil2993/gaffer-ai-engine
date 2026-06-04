"""Shared utilities for data cleaning pipeline."""

import os
import re
import shutil


DATA_DIR = "data/epl_seasons"
RAW_DIR = os.path.join(DATA_DIR, "raw")


def backup_file(path: str) -> None:
    """Create backup of original file in raw/ directory (first copy only)."""
    os.makedirs(RAW_DIR, exist_ok=True)
    dest = os.path.join(RAW_DIR, os.path.basename(path))
    if not os.path.exists(dest):
        shutil.copy2(path, dest)


def num(value, default: float = 0.0) -> float:
    """Coerce a CSV cell to float, tolerating blanks/strings/NaN."""
    try:
        f = float(value)
        return default if f != f else f  # NaN (f != f) -> default
    except (TypeError, ValueError):
        return default


def pct(value, default: float = 0.0) -> float:
    """Parse a percentage cell like '66%' -> 0.66 (NaN/blank -> default)."""
    try:
        f = float(str(value).replace("%", "").strip()) / 100.0
        return default if f != f else f
    except (TypeError, ValueError):
        return default


def per90(value, minutes: float) -> float:
    """Calculate per-90 rate for comparison."""
    return (value / minutes * 90.0) if minutes else 0.0


def digits(s: str) -> str:
    """Extract only digits from string (for parsing wage values)."""
    return re.sub(r"[^\d]", "", s or "")


def position_phrase(position) -> str:
    """'MF,FW' -> 'midfielder / forward'."""
    pos_word = {"GK": "goalkeeper", "DF": "defender", "MF": "midfielder", "FW": "forward"}
    words = [pos_word.get(p.strip(), p.strip()) for p in str(position).split(",") if p.strip()]
    return " / ".join(words) if words else "player"
