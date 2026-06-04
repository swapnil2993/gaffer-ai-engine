"""Main data cleaning orchestrator - executes all cleaning steps in order."""

from backend.scripts.data_cleaning import csv_repair, defensive_stats, wage_assignment


def run_cleaning_pipeline(verbose: bool = True) -> dict:
    """Execute the complete data cleaning pipeline.

    Steps (in order):
    1. Repair CSV structure (unquoted commas)
    2. Add estimated defensive stats (tackles, interceptions)
    3. Assign wages to missing players

    Args:
        verbose: Print progress messages

    Returns:
        Dict with step results and status
    """
    if verbose:
        print("\n" + "=" * 70)
        print("DATA CLEANING PIPELINE")
        print("=" * 70 + "\n")

    results = {}

    # Step 1: Repair CSVs (SKIPPED - data already cleaned with granular positions)
    # if verbose:
    #     print("Step 1/3: Repair CSV structure...")
    #     print("-" * 70)
    # csv_repair.run()
    results["csv_repair"] = "skipped"

    # Step 2: Add defensive stats
    if verbose:
        print("Step 2/3: Add estimated defensive statistics...")
        print("-" * 70)
    defensive_stats.run()
    results["defensive_stats"] = "complete"
    print()

    # Step 3: Assign missing wages (SKIPPED - raw data structure issues)
    # if verbose:
    #     print("Step 3/3: Assign wages to missing players...")
    #     print("-" * 70)
    # wage_assignment.run()
    results["wage_assignment"] = "skipped"

    if verbose:
        print("=" * 70)
        print("✓ Data cleaning pipeline complete!")
        print("=" * 70 + "\n")

    return results
