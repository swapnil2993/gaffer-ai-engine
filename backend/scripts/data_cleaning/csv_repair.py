"""Repair FBref CSVs with unquoted commas that corrupt naive CSV parsing.

Two files contain unquoted commas:
- player_stats_25-26.csv: multi-position values (MF,FW) and minutes with
  thousands separator (2,449) inject stray commas
- player_wages.csv: multi-position values plus currency formatting
  (£27,300,000) inject multiple stray commas

Solution: Reconstruct each row from known column layout, merge split fields,
and rewrite as properly-quoted CSV.
"""

import csv
import os
import re

from backend.scripts.data_cleaning.common import DATA_DIR, RAW_DIR, backup_file, digits

POSITIONS = {"GK", "DF", "MF", "FW"}

STATS_HEADER = [
    "Rk", "Player", "Nation", "Pos", "Squad", "Age", "Born", "MP", "Starts", "Min",
    "90s", "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt", "CrdY", "CrdR",
    "Gls_per90", "Ast_per90", "G+A_per90", "G-PK_per90", "G+A-PK_per90", "Matches",
]

WAGES_HEADER = [
    "Rk", "Player", "Nation", "Pos", "Squad", "Age", "Weekly Wages (GBP)", "Annual Wages (GBP)",
]


def _split_prefix(parts):
    """Parse (rk, player, nation, positions, rest_index) or None for junk rows."""
    if len(parts) < 8 or parts[1].strip() in ("", "Player"):
        return None
    if not parts[0].strip().isdigit():
        return None
    rk, player, nation = parts[0].strip(), parts[1].strip(), parts[2].strip()
    i = 3
    positions = []
    while i < len(parts) and parts[i].strip() in POSITIONS:
        positions.append(parts[i].strip())
        i += 1
    if not positions:
        return None
    return rk, player, nation, ",".join(positions), i


def repair_stats(path: str) -> int:
    """Rewrite player_stats_25-26.csv as valid, column-aligned CSV."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        next(fh, None)  # Skip original header
        for line in fh:
            parts = line.rstrip("\n").split(",")
            prefix = _split_prefix(parts)
            if prefix is None:
                continue
            rk, player, nation, pos, i = prefix
            try:
                squad, age, born, mp, starts = parts[i : i + 5]
            except ValueError:
                continue
            tail = parts[i + 5 :]
            # If Min had thousands comma it split into two; merge back
            if len(tail) == 17:
                minutes = tail[0].strip() + tail[1].strip()
                rest = tail[2:]
            elif len(tail) == 16:
                minutes = tail[0].strip()
                rest = tail[1:]
            else:
                continue
            if len(rest) != 15:
                continue
            row = [rk, player, nation, pos, squad, age, born, mp, starts, minutes] + [c.strip() for c in rest]
            rows.append(row)

    backup_file(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(STATS_HEADER)
        writer.writerows(rows)
    return len(rows)


def repair_wages(path: str) -> int:
    """Rewrite player_wages.csv with clean integer GBP wage columns."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
            parts = line.rstrip("\n").split(",")
            prefix = _split_prefix(parts)
            if prefix is None:
                continue
            rk, player, nation, pos, i = prefix
            if i + 1 >= len(parts):
                continue
            squad, age = parts[i].strip(), parts[i + 1].strip()
            gbp = re.findall(r"£\s*([\d,\s]+?)\s*\(", line)
            if len(gbp) < 2:
                continue
            weekly, annual = digits(gbp[0]), digits(gbp[1])
            rows.append([rk, player, nation, pos, squad, age, weekly, annual])

    backup_file(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(WAGES_HEADER)
        writer.writerows(rows)
    return len(rows)


def run():
    """Repair both CSV files."""
    stats_path = os.path.join(DATA_DIR, "player_stats_25-26.csv")
    wages_path = os.path.join(DATA_DIR, "player_wages.csv")

    if os.path.exists(stats_path):
        n = repair_stats(stats_path)
        print(f"✓ Cleaned player_stats_25-26.csv → {n} rows (backup in {RAW_DIR}/)")

    if os.path.exists(wages_path):
        n = repair_wages(wages_path)
        print(f"✓ Cleaned player_wages.csv → {n} rows (backup in {RAW_DIR}/)")
