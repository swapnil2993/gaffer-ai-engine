#!/usr/bin/env python3
"""Clean and standardize 2025-26 player stats to match 2023-24 schema."""

import pandas as pd
import sys
from pathlib import Path

# Paths
RAW_FILE = Path("data/epl_seasons/raw/player_stats_25-26.csv")
CLEANED_FILE = Path("data/epl_seasons/player_stats_25-26.csv")

# Column mapping: raw -> expected clean name
COLUMN_MAPPING = {
    "player_name": "player_name",
    "Pos": "Pos",
    "Squad": "Squad",
    "Appearances": "Appearances",
    "Min": "Minutes",
    "Goals": "Goals",
    "Assists": "Assists",
    "Int": "Interceptions",
    "TklW": "Tackles_Won",
    "position": "position",
}

def clean_25_26_data():
    """Clean 2025-26 player stats to match 2023-24 schema."""

    if not RAW_FILE.exists():
        print(f"❌ Raw file not found: {RAW_FILE}")
        return False

    print(f"Loading raw file: {RAW_FILE}")
    df = pd.read_csv(RAW_FILE)

    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    # Select and rename columns
    selected_cols = {}
    for raw_col, clean_col in COLUMN_MAPPING.items():
        if raw_col in df.columns:
            selected_cols[raw_col] = clean_col
        else:
            print(f"⚠️  Missing column: {raw_col}")

    # Rename columns
    df_clean = df[list(selected_cols.keys())].rename(columns=selected_cols)

    # Standardize column names to match 2023-24 format
    # 2023-24 uses: player_name, Squad, Appearances, Minutes, Goals, Assists, Tackles, Tackles_Won, Interceptions, etc.
    df_clean.columns = [
        "player_name", "Pos", "Squad", "Appearances", "Minutes",
        "Goals", "Assists", "Interceptions", "Tackles_Won", "position"
    ]

    # Add missing columns with default values
    df_clean["Tackles"] = 0  # Not available in 2025-26 raw data
    df_clean["Recoveries"] = 0  # Not available in 2025-26 raw data
    df_clean["xG"] = 0  # Not available
    df_clean["xAG"] = 0  # Not available
    df_clean["PrgC"] = 0  # Progressive carries
    df_clean["PrgP"] = 0  # Progressive passes
    df_clean["BigChancesCreated"] = 0  # Not available

    # Reorder and select final columns
    final_cols = [
        "player_name", "Pos", "Squad", "position",
        "Appearances", "Minutes", "Goals", "Assists",
        "Tackles", "Tackles_Won", "Interceptions", "Recoveries",
        "xG", "xAG", "PrgC", "PrgP", "BigChancesCreated"
    ]

    # Convert to lowercase column names for consistency
    df_final = df_clean[final_cols].copy()
    df_final.columns = [c.lower() for c in df_final.columns]

    # Handle missing values
    df_final = df_final.fillna(0)

    print(f"\nCleaned shape: {df_final.shape}")
    print(f"Cleaned columns: {list(df_final.columns)}")

    # Save
    df_final.to_csv(CLEANED_FILE, index=False)
    print(f"\n✅ Saved to: {CLEANED_FILE}")

    return True

if __name__ == "__main__":
    success = clean_25_26_data()
    sys.exit(0 if success else 1)
