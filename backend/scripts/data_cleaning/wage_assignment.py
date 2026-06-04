"""Identify missing players in wages data and assign realistic wage values.

Strategy: Assign wages based on position and performance metrics to match
real-world salary ranges for EPL players.
"""

import os
import shutil

import pandas as pd
import numpy as np

from backend.scripts.data_cleaning.common import DATA_DIR, RAW_DIR, backup_file, num


def load_player_data():
    """Load all player stats from CSV files (3 seasons)."""
    stats = []
    for filename, season in [
        ("player_stats_23-24.csv", "2023/2024"),
        ("fbref_PL_2024-25.csv", "2024/2025"),
        ("player_stats_25-26.csv", "2025/2026"),
    ]:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["Season"] = season
            stats.append(df)
            print(f"  Loaded {filename}: {len(df)} players")
    return pd.concat(stats, ignore_index=True) if stats else pd.DataFrame()


def load_wages():
    """Load existing wages data."""
    path = os.path.join(DATA_DIR, "player_wages.csv")
    if os.path.exists(path):
        wages = pd.read_csv(path)
        print(f"  Loaded wages: {len(wages)} players")
        return wages
    return pd.DataFrame()


def get_position_code(player_name, stats_df):
    """Get position code for a player from stats."""
    player_stats = stats_df[stats_df.get("Player", stats_df.get("Name")) == player_name]
    if len(player_stats) > 0:
        pos = player_stats.iloc[0].get("Pos", "Unknown")
        return str(pos).split(",")[0].strip() if pd.notna(pos) else "Unknown"
    return "Unknown"


def estimate_wage(player_name, position, stats_df):
    """Estimate realistic wage based on position and performance."""
    player_stats = stats_df[(stats_df.get("Player", stats_df.get("Name")) == player_name)]

    base_ranges = {
        "GK": (50_000, 150_000),
        "DF": (60_000, 200_000),
        "MF": (70_000, 250_000),
        "FW": (80_000, 300_000),
        "Unknown": (70_000, 180_000),
    }

    if len(player_stats) == 0:
        base_min, base_max = base_ranges.get(position, (70_000, 180_000))
        annual = np.random.uniform(base_min, base_max) * 52
        return _format_wage(annual)

    # Performance-based adjustment
    row = player_stats.iloc[0]
    goals = float(row.get("Gls", row.get("Goals", 0)) or 0)
    assists = float(row.get("Ast", row.get("Assists", 0)) or 0)
    minutes = float(row.get("Min", row.get("minutes", 0)) or 0)

    base_min, base_max = base_ranges.get(position, (70_000, 180_000))
    performance_score = goals + assists

    if performance_score > 20:
        salary = np.random.uniform(base_max * 0.8, base_max) * 52
    elif performance_score > 10:
        salary = np.random.uniform(base_max * 0.5, base_max * 0.8) * 52
    elif performance_score > 5:
        salary = np.random.uniform(base_min + 100_000, base_max * 0.5) * 52
    elif minutes > 1800:
        salary = np.random.uniform(base_min + 50_000, base_min + 150_000) * 52
    else:
        salary = np.random.uniform(base_min, base_min + 50_000) * 52

    return _format_wage(salary)


def _format_wage(annual_pounds):
    """Format wage in £X,XXX,XXX format."""
    annual = int(annual_pounds)
    if annual >= 1_000_000:
        return f"£{annual / 1_000_000:.1f}M"
    else:
        return f"£{annual:,}"


def run():
    """Find missing players and assign realistic wages."""
    print("Finding missing players and assigning wages...")
    print()

    stats_df = load_player_data()
    wages_df = load_wages()

    if stats_df.empty:
        print("❌ No player stats found")
        return

    print()

    player_col = "Player" if "Player" in stats_df.columns else "Name"
    all_players = set(stats_df[player_col].dropna().unique())

    wage_col = "Player" if "Player" in wages_df.columns else "Name"
    players_with_wages = set(wages_df[wage_col].dropna().unique())

    missing_players = all_players - players_with_wages
    missing_players = {p for p in missing_players if isinstance(p, str) and p.strip()}

    print(f"📊 Analysis:")
    print(f"  Total unique players in stats: {len(all_players)}")
    print(f"  Players with wages: {len(players_with_wages)}")
    print(f"  Missing players: {len(missing_players)}")
    print()

    if not missing_players:
        print("✓ All players have wages assigned!")
        return

    print(f"🔄 Assigning wages to {len(missing_players)} missing players...")
    print()

    new_wages = []
    for player_name in sorted(missing_players):
        position = get_position_code(player_name, stats_df)
        annual = estimate_wage(player_name, position, stats_df)

        if "M" in annual:
            millions = float(annual.replace("£", "").replace("M", ""))
            annual_num = int(millions * 1_000_000)
        else:
            annual_num = int(annual.replace("£", "").replace(",", ""))
        weekly = annual_num / 52

        new_wages.append({
            "Name": player_name,
            "Position": position,
            "Annual Wages": annual,
            "Weekly Wages": f"£{weekly:,.0f}",
        })

    new_wages_df = pd.DataFrame(new_wages)
    combined = pd.concat([wages_df, new_wages_df], ignore_index=True).drop_duplicates(
        subset=["Name"], keep="first"
    )

    backup_file(os.path.join(DATA_DIR, "player_wages.csv"))
    output_path = os.path.join(DATA_DIR, "player_wages.csv")
    combined.to_csv(output_path, index=False)
    print(f"✓ Updated wages saved to {output_path}")
    print()

    print(f"📈 Summary:")
    print(f"  Original wages file: {len(wages_df)} players")
    print(f"  Added: {len(new_wages)} new players")
    print(f"  Total: {len(combined)} players")
    print()

    print("Sample of assigned wages:")
    print()
    sample = new_wages_df.sort_values("Annual Wages", ascending=False).head(10)
    for _, row in sample.iterrows():
        print(f"  {row['Name']:30s} ({row['Position']:2s}) {row['Annual Wages']:>12s} {row['Weekly Wages']:>15s}")

    print()
    print("✓ Done! All missing players now have wage assignments.")
