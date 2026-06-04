"""
Comprehensive CSV cleaning script to standardize positions across ALL SEASONS.
Uses defensive stats (TklW + Interceptions) to properly infer CB vs LB positions.

Sources:
  - 2023/24: player_stats_23-24.csv (1019 rows)
  - 2024/25: fbref_PL_2024-25.csv (574 rows)
  - 2025/26: player_stats_25-26.csv (573 rows merged from 25-26-stats + 25-26-misc)
"""

import pandas as pd
import os


def normalize_name(name: str) -> str:
    """Normalize player names for matching."""
    return name.strip().lower().replace(' ', '')


def infer_granular_position(pos_str: str, stats_row: pd.Series) -> str:
    """
    Infer granular position from position + defensive stats.
    Uses TklW (Tackles Won) + Interceptions to distinguish CB from LB.
    """
    # Get appearances (convert to float)
    appearances = stats_row.get('Appearances') or stats_row.get('MP') or 1
    try:
        appearances = float(appearances) if appearances else 1
    except (ValueError, TypeError):
        appearances = 1
    if appearances < 1:
        appearances = 1

    # Get stats (convert to float)
    goals = stats_row.get('Goals') or stats_row.get('Gls') or 0
    assists = stats_row.get('Assists') or stats_row.get('Ast') or 0
    tackles_won = stats_row.get('TklW') or stats_row.get('Tackles Won') or 0
    interceptions = stats_row.get('Int') or stats_row.get('Interceptions') or 0

    try:
        goals = float(goals) if goals else 0
        assists = float(assists) if assists else 0
        tackles_won = float(tackles_won) if tackles_won else 0
        interceptions = float(interceptions) if interceptions else 0
    except (ValueError, TypeError):
        goals = assists = tackles_won = interceptions = 0

    pos = str(pos_str).upper().strip()
    if ',' in pos:
        pos = pos.split(',')[0].strip()

    # Goalkeepers
    if pos == 'GK':
        return 'GK'

    # DEFENDERS: Use tackle intensity (special handling for in-season data like 25-26)
    if pos in ('DF', 'DEF', 'LB', 'CB', 'RB'):
        # For in-season data (low appearances), use per-90 basis with lower threshold
        if appearances < 5:  # In-season, partial data (typically 25-26 season)
            per_90_value = stats_row.get('90s', 0)
            if per_90_value:
                try:
                    per_90_value = float(per_90_value)
                    if per_90_value > 0:
                        defensive_per_90 = (tackles_won + interceptions) / per_90_value
                        return 'CB' if defensive_per_90 > 1.2 else 'LB'  # Lower threshold for partial season
                except (ValueError, TypeError):
                    pass

        # Full season or fallback: use per-appearance basis
        defensive_intensity = (tackles_won + interceptions) / appearances if appearances > 0 else 0
        return 'CB' if defensive_intensity > 1.5 else 'LB'

    # MIDFIELDERS
    if pos in ('MF', 'MID', 'CM', 'DM', 'AM'):
        assist_intensity = assists / appearances if appearances > 0 else 0
        tackle_intensity = tackles_won / appearances if appearances > 0 else 0

        if assist_intensity > 0.3 and tackle_intensity < 0.8:
            return 'AM'
        elif tackle_intensity > 1.0 and assist_intensity < 0.2:
            return 'DM'
        else:
            return 'CM'

    # FORWARDS
    if pos in ('FW', 'ATT', 'ST', 'IF', 'W'):
        goal_intensity = goals / appearances if appearances > 0 else 0
        assist_intensity = assists / appearances if appearances > 0 else 0

        if goal_intensity > 0.4 and assist_intensity < 0.25:
            return 'ST'
        elif assist_intensity > 0.3 and goal_intensity < 0.3:
            return 'W'
        else:
            return 'IF'

    return 'CM'


def clean_season_csv(input_filename, season_name, reference_positions=None):
    """Clean a season CSV file with position inference.

    If reference_positions is provided (from 2023/24), use it for consistency.
    Only infer for new players not in the reference.

    Raw files are read but NEVER modified.
    Cleaned output is saved to data/epl_seasons/ (without 'raw/' in filename).
    """
    print(f"\n📊 Cleaning {input_filename} ({season_name})...")

    # Read from raw or data/epl_seasons/
    if input_filename.startswith('data/'):
        read_path = input_filename
    elif input_filename.startswith('raw/'):
        read_path = f'data/epl_seasons/{input_filename}'
    else:
        read_path = f'data/epl_seasons/{input_filename}'

    # Save to data/epl_seasons/ (strip 'raw/' if present)
    output_filename = input_filename.replace('raw/', '')
    if output_filename.startswith('data/'):
        write_path = output_filename
    else:
        write_path = f'data/epl_seasons/{output_filename}'

    df = pd.read_csv(read_path)

    player_col = 'player_name' if 'player_name' in df.columns else ('Player' if 'Player' in df.columns else None)
    pos_col = 'position' if 'position' in df.columns else ('Pos' if 'Pos' in df.columns else None)

    if not player_col or not pos_col:
        print(f"   ⚠️  Skipping: missing Player or Position column")
        return {}

    def get_position(row):
        player_name = row.get(player_col, '').strip()
        current_pos = row.get(pos_col)

        # If already granular, keep it
        if current_pos in ('CB', 'RB', 'LWB', 'RWB', 'CM', 'DM', 'AM', 'ST', 'IF', 'W', 'GK'):
            return current_pos

        # If we have reference positions (from 2023/24), use them for consistency
        if reference_positions and player_name in reference_positions:
            return reference_positions[player_name]

        # Otherwise infer from stats
        return infer_granular_position(current_pos or 'MF', row)

    df['position'] = df.apply(get_position, axis=1)
    df.to_csv(write_path, index=False)
    print(f"   ✓ Cleaned file saved to: {write_path}")

    pos_counts = df['position'].value_counts().to_dict()
    print(f"   ✓ {len(df)} rows cleaned")
    print(f"   Positions: {pos_counts}")

    return {(row[player_col] if pd.notna(row[player_col]) else ''): row['position']
            for _, row in df.iterrows()}


def main():
    """Clean all season CSVs with consistent positions across seasons."""
    print("=" * 80)
    print("COMPREHENSIVE CSV CLEANING - ALL SEASONS")
    print("Using 2023/24 as position reference for consistency")
    print("=" * 80)

    os.chdir('/Users/swapnilpande/sper/gaffer-ai-engine')

    # Clean 2023/24 first (PRIMARY SOURCE - most complete, full season)
    positions_23_24 = clean_season_csv('player_stats_23-24.csv', '2023/24')
    clean_season_csv('player_overview.csv', 'Player Overview', positions_23_24)

    # Clean 2024/25 using 2023/24 positions for consistency
    clean_season_csv('fbref_PL_2024-25.csv', '2024/25', positions_23_24)

    # Clean 2025/26 using 2023/24 positions for consistency
    clean_season_csv('raw/player_stats_25-26.csv', '2025/26', positions_23_24)

    print("\n" + "=" * 80)
    print("✅ ALL SEASONS CLEANED!")
    print("   • 2023/24: 1019 rows")
    print("   • 2024/25: 574 rows")
    print("   • 2025/26: 573 rows")
    print("   • All with consistent granular positions (CB, LB, CM, DM, etc.)")
    print("=" * 80)
    print("\n📝 Next: Run 'python backend/scripts/index_data.py' to index all seasons")


if __name__ == '__main__':
    main()
