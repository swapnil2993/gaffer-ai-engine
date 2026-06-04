"""Add estimated defensive stats (Tkl, Int) to seasons lacking them.

FBref's "standard" exports for 2024/25 and 2025/26 don't include defensive
actions. This module synthesizes plausible tackles/interceptions from minutes
and position, so hybrid ranking works across all seasons.

Values are MOCK (clearly labelled): per-90 rates by position, scaled by minutes,
with deterministic per-player variation. NOT real data.
"""

import hashlib
import os
import shutil

import pandas as pd

from backend.scripts.data_cleaning.common import DATA_DIR, RAW_DIR, backup_file, num

TKL_PER90 = {"GK": 0.1, "DF": 2.2, "MF": 1.9, "FW": 0.9}
INT_PER90 = {"GK": 0.2, "DF": 1.3, "MF": 1.0, "FW": 0.4}
REC_PER90 = {"GK": 3.5, "DF": 7.0, "MF": 6.5, "FW": 3.5}

TARGETS = [
    ("fbref_PL_2024-25.csv", "Min"),
    ("player_stats_25-26.csv", "Min"),
]


def _factor(name: str) -> float:
    """Deterministic per-player multiplier in ~[0.75, 1.25] (stable across runs)."""
    h = int(hashlib.md5(str(name).encode("utf-8")).hexdigest()[:8], 16)
    return 0.75 + (h % 1000) / 1000.0 * 0.5


def _primary_pos(pos) -> str:
    """Extract primary position from 'MF,FW' -> 'MF'."""
    return str(pos).split(",")[0].strip() if pd.notna(pos) else "MF"


def add_to_file(filename: str, minutes_col: str) -> int:
    """Add Tkl, TklW, Int, Recoveries columns to player stats file."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  Skipping {filename} (not found)")
        return 0

    df = pd.read_csv(path)
    tackles, tackles_won, interceptions, recoveries = [], [], [], []

    for _, row in df.iterrows():
        name = row.get("Player", "")
        pos = _primary_pos(row.get("Pos"))
        nineties = num(row.get(minutes_col)) / 90.0
        f = _factor(name)

        tkl = round(TKL_PER90.get(pos, 1.5) * nineties * f)
        tackles.append(tkl)
        tackles_won.append(round(tkl * (0.55 + (f - 0.75) * 0.4)))  # ~55-75% success
        interceptions.append(round(INT_PER90.get(pos, 0.9) * nineties * f))
        recoveries.append(round(REC_PER90.get(pos, 5.5) * nineties * f))

    df["Tkl"] = tackles
    df["TklW"] = tackles_won
    df["Int"] = interceptions
    df["Recoveries"] = recoveries

    backup_file(path)
    df.to_csv(path, index=False)
    return len(df)


def run():
    """Add estimated defensive stats to both target files."""
    for filename, minutes_col in TARGETS:
        n = add_to_file(filename, minutes_col)
        if n:
            print(f"✓ Added estimated Tkl/Int to {filename} ({n} rows; backup in {RAW_DIR}/)")
