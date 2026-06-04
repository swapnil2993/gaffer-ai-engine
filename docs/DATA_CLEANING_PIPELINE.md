# Data Cleaning Pipeline — Complete Guide

Comprehensive guide to the modular data cleaning pipeline used to prepare scouting data for indexing and retrieval.

---

## Quick Start

```bash
# Run the complete cleaning pipeline
uv run python -m backend.scripts.index_data
```

All cleaning happens automatically as the first step, then data is indexed.

---

## Table of Contents

1. [What Gets Cleaned](#what-gets-cleaned)
2. [Why We Clean](#why-we-clean)
3. [Modular Architecture](#modular-architecture)
4. [Using the Pipeline](#using-the-pipeline)
5. [Module Reference](#module-reference)
6. [Extending the Pipeline](#extending-the-pipeline)
7. [Troubleshooting](#troubleshooting)

---

## What Gets Cleaned

The pipeline performs **3 major cleaning steps**:

### 1. CSV Structure Repair
**Files**: `player_stats_25-26.csv`, `player_wages.csv`

**Problem**: Unquoted commas corrupt naive CSV parsing
- Multi-position values: "MF,FW" (position)
- Currency formatting: "£27,300,000" (wages)
- Numbers with thousands separators: "2,449" (minutes)

**Solution**: Parse row prefixes, reconstruct, rewrite with proper quoting

**Result**: Valid CSV readable by pandas

---

### 2. Defensive Statistics (Estimated)
**Files**: `fbref_PL_2024-25.csv`, `player_stats_25-26.csv`

**Problem**: FBref exports for 2024/25 and 2025/26 lack defensive actions
- No Tackles (Tkl)
- No Interceptions (Int)
- No Recoveries

**Solution**: Synthesize plausible values using:
- Per-position baseline rates (TKL_PER90, INT_PER90)
- Player's minutes played
- Deterministic per-player factor (stable across runs)

**Result**: All seasons have defensive data for hybrid ranking

**Note**: Clearly marked as MOCK (demo) data, not real FBref values

---

### 3. Missing Player Wages
**File**: `player_wages.csv`

**Problem**: 270 players missing from wages (34% gap breaks ranking UI)

**Solution**: Infer realistic wages based on:
- Position (GK, DF, MF, FW have different ranges)
- Performance (goals + assists)
- Playing time (minutes)

**Result**: 797 players with complete wage assignments

---

## Why We Clean

### Semantic Matching
Prose ("prolific goal threat") embeds better than raw stats ("Gls: 12"). Users type natural language queries; indexed data should read the same way.

### Complete Dataset
Missing wages break the UI ranking system. Filling gaps ensures the system works end-to-end.

### Defensive Parity
Without defensive stats in recent seasons, multi-year queries ("improving defender") can't work. Mock values enable feature parity.

### Data Consistency
Type coercion (CSV strings → floats) prevents calculation errors. Backup originals before overwriting to never lose data.

---

## Modular Architecture

The pipeline is organized into **7 reusable modules**:

```
backend/scripts/data_cleaning/
├── __init__.py              # Public API
├── common.py                # Shared utilities
├── csv_repair.py            # Fix CSV structure
├── defensive_stats.py       # Add estimated stats
├── wage_assignment.py       # Infer missing wages
├── data_transformer.py      # Convert stats → prose
├── privacy_handler.py       # Scrub PII
└── pipeline.py              # Orchestrator
```

### Module Interdependencies

```
common.py (num, per90, pct, position_phrase, backup_file)
   ↑
   ├── csv_repair.py
   ├── defensive_stats.py
   └── wage_assignment.py

data_transformer.py (describe_outfield_player, describe_2023_24_player)
   ↑
   └── index_data.py

privacy_handler.py (scrub_text)
   ↑
   └── parser.py

pipeline.py (orchestrator)
   ↑
   ├── csv_repair.py
   ├── defensive_stats.py
   └── wage_assignment.py
```

---

## Using the Pipeline

### Run Full Pipeline

```python
from backend.scripts.data_cleaning import run_cleaning_pipeline

result = run_cleaning_pipeline(verbose=True)
# Returns: {"csv_repair": "complete", "defensive_stats": "complete", "wage_assignment": "complete"}
```

### Run Individual Steps

```python
from backend.scripts.data_cleaning import csv_repair, defensive_stats, wage_assignment

csv_repair.run()
defensive_stats.run()
wage_assignment.run()
```

### In index_data.py

```bash
uv run python -m backend.scripts.index_data
```

This runs:
1. Data cleaning pipeline (3 steps)
2. Initialize Milvus collections
3. Index tactical reference & books
4. Index player stats (3 seasons)
5. Generate career aggregate profiles

---

## Module Reference

### `common.py` — Shared Utilities

**Purpose**: Eliminate code duplication across modules

**Functions**:

| Function | Purpose |
|----------|---------|
| `num(value, default=0.0)` | Parse CSV cell to float; handle NaN/blanks |
| `per90(value, minutes)` | Calculate per-90 rate for comparison |
| `pct(value, default=0.0)` | Parse "66%" → 0.66 |
| `digits(s)` | Extract digits from string (for wage parsing) |
| `position_phrase(position)` | "MF,FW" → "midfielder / forward" |
| `backup_file(path)` | Copy original to `raw/` before overwriting |

**Constants**:
- `DATA_DIR = "data/epl_seasons"`
- `RAW_DIR = os.path.join(DATA_DIR, "raw")`

---

### `csv_repair.py` — Fix Unquoted Commas

**Problem**: 
- `player_stats_25-26.csv`: Multi-position "MF,FW" + minutes "2,449" inject stray commas
- `player_wages.csv`: Multi-position + currency "£27,300,000" corrupt row alignment

**Solution**:
1. Parse row prefix (Rk | Player | Nation | Position(s))
2. Detect and merge split fields
3. Reconstruct row with correct alignment
4. Rewrite with proper CSV quoting

**API**:

```python
repair_stats(path: str) -> int       # Fix player_stats_25-26.csv
repair_wages(path: str) -> int       # Fix player_wages.csv
run()                                 # Run both
```

**Example**:
```
Before: ["Rk","Smith","ENG","MF","FW","MAN U",...]  # 27 fields (stray commas)
After:  ["Rk","Smith","ENG","MF,FW","MAN U",...]    # 25 fields (correct)
```

---

### `defensive_stats.py` — Add Estimated Stats

**Problem**: 2024/25 and 2025/26 exports lack defensive actions

**Solution**: Synthesize per-position baselines scaled by minutes

**Baseline Rates (per-90)**:
```python
Tackles:       GK=0.1, DF=2.2, MF=1.9, FW=0.9
Interceptions: GK=0.2, DF=1.3, MF=1.0, FW=0.4
Recoveries:    GK=3.5, DF=7.0, MF=6.5, FW=3.5
```

**Calculation**:
```python
nineties = minutes / 90.0
factor = _factor(player_name)  # Deterministic [0.75, 1.25]
tackles = round(BASELINE[pos] * nineties * factor)
```

**Determinism**: Seeded by player name (MD5 hash) → stable across runs

**API**:

```python
add_to_file(filename: str, minutes_col: str) -> int  # Process one file
run()                                                   # Process both targets
```

**Targets**: 
- `fbref_PL_2024-25.csv`
- `player_stats_25-26.csv`

**Skips**: 2023/24 (already has real defensive data from FBref)

---

### `wage_assignment.py` — Infer Missing Wages

**Problem**: 270 missing players (34% gap)

**Solution**: Infer from position + performance + minutes

**Position Ranges (weekly)**:
```python
GK: £50k–£150k
DF: £60k–£200k
MF: £70k–£250k
FW: £80k–£300k
```

**Performance Tiers** (goals + assists):
```python
High (>20):      0.8–1.0x of max range
Medium (>10):    0.5–0.8x
Regular (>5):    base_min+100k to 0.5x max
Starter (>1800 min): base_min+50k to base_min+150k
Fringe:          base_min to base_min+50k
```

**API**:

```python
load_player_data() -> DataFrame     # Load 3 seasons
load_wages() -> DataFrame            # Load existing wages
estimate_wage(name, pos, stats) -> str  # Infer single wage
run()                                 # Load, infer, save
```

**Output Format**: 
- Annual: "£3.4M" or "£127,500"
- Weekly: derived from annual / 52

---

### `data_transformer.py` — Convert Stats to Prose

**Purpose**: Transform numbers into natural language for semantic embedding

**Functions**:

```python
describe_outfield_player(name, squad, position, season, stats) -> str
describe_2023_24_player(name, squad, position, season, stats) -> str
```

**Example Output**:
```
"Marcus Smith is a midfielder for Manchester United in the 2024/2025 season.
Played 1,847 minutes across 28 appearances. A regular goal threat with 12 goals.
Contributes creatively with 6 assists. Underlying numbers: 9.2 xG and 4.1 xAG.
Ball progression: 156 progressive carries, 89 progressive passes.
Defensive actions: 45 tackles, 32 won, 28 interceptions, 156 recoveries 
(wins the ball back)."
```

**Why Prose**: User queries ("find prolific wingers") match prose better than numbers

---

### `privacy_handler.py` — Scrub PII

**Purpose**: Mask sensitive data before indexing books

**Scrubs**:
- Phone numbers
- Email addresses
- Passport/IBAN codes
- Credit card numbers

**Preserves** (needed for context):
- Person names (players, managers)
- Location names (clubs, countries)

**API**:

```python
scrub_text(text: str) -> str  # Mask PII, preserve context
```

---

### `pipeline.py` — Main Orchestrator

**Purpose**: Execute all steps in order with progress reporting

**API**:

```python
run_cleaning_pipeline(verbose: bool = True) -> dict
```

**Execution Order**:
1. Repair CSVs
2. Add defensive stats
3. Assign missing wages

**Output**:
```python
{
    "csv_repair": "complete",
    "defensive_stats": "complete",
    "wage_assignment": "complete",
}
```

**Progress Display**:
```
======================================================================
DATA CLEANING PIPELINE
======================================================================

Step 1/3: Repair CSV structure...
----------------------------------------------------------------------
✓ Cleaned player_stats_25-26.csv → 280 rows
✓ Cleaned player_wages.csv → 270 rows

Step 2/3: Add estimated defensive statistics...
----------------------------------------------------------------------
✓ Added estimated Tkl/Int to fbref_PL_2024-25.csv (850 rows)
✓ Added estimated Tkl/Int to player_stats_25-26.csv (280 rows)

Step 3/3: Assign wages to missing players...
----------------------------------------------------------------------
✓ Updated wages saved to data/epl_seasons/player_wages.csv

📊 Analysis:
  Total unique players in stats: 797
  Players with wages: 527
  Missing players: 270
  Added: 270 new players
  Total: 797 players

======================================================================
✓ Data cleaning pipeline complete!
======================================================================
```

---

## Extending the Pipeline

### Add a New Cleaning Step

**Step 1**: Create module `backend/scripts/data_cleaning/my_step.py`

```python
"""My new cleaning step."""

from backend.scripts.data_cleaning.common import backup_file, num

def run():
    """Execute my cleaning step."""
    # Your logic here
    print("✓ My step complete")
```

**Step 2**: Import and call in `pipeline.py`

```python
from backend.scripts.data_cleaning import my_step

def run_cleaning_pipeline(verbose=True):
    # ... existing steps ...
    
    if verbose:
        print("Step N/X: My cleaning step...")
        print("-" * 70)
    my_step.run()
    results["my_step"] = "complete"
    print()
    
    # ... rest of pipeline ...
```

**Step 3**: Update `__init__.py` to export if needed

```python
from backend.scripts.data_cleaning.my_step import run as run_my_step

__all__ = ["run_cleaning_pipeline", "run_my_step"]
```

**Step 4**: Update documentation (this file)

---

## Troubleshooting

### Issue: "File not found" errors

**Cause**: CSV files don't exist in `data/epl_seasons/`

**Solution**: 
```bash
# Check what files exist
ls data/epl_seasons/

# Update file paths in the module if names changed
```

### Issue: "CSV has wrong number of fields"

**Cause**: CSV structure changed, repair logic is outdated

**Solution**:
1. Read the error message for which file/row failed
2. Update column headers in `csv_repair.py` if they changed
3. Update the prefix parsing logic if position extraction needs adjustment

### Issue: Wage values look unrealistic

**Cause**: Performance tier thresholds need adjustment

**Solution**: Edit `wage_assignment.py` lines 45-65 to change thresholds

```python
if performance_score > 20:  # Change from 20 to 15 for example
    salary = np.random.uniform(base_max * 0.8, base_max) * 52
```

### Issue: Defensive stats seem too high/low

**Cause**: Baseline rates (per-90) are off

**Solution**: Edit `defensive_stats.py` lines 10-12

```python
TKL_PER90 = {"GK": 0.1, "DF": 2.2, "MF": 1.9, "FW": 0.9}  # Adjust these
```

### Issue: Backups aren't being created

**Cause**: RAW_DIR doesn't exist and creation failed

**Solution**: Create directory manually
```bash
mkdir -p data/epl_seasons/raw
```

---

## Data Flow Diagram

```
┌─────────────────────────┐
│   Raw CSV Files         │
│ (broken structure)      │
└────────────┬────────────┘
             │
             ↓
┌─────────────────────────┐
│ [csv_repair.run()]      │
│ Fix unquoted commas     │
└────────────┬────────────┘
             │
             ↓
┌─────────────────────────┐
│  Valid CSV Files        │
│ (structure correct)     │
└────────────┬────────────┘
             │
             ↓
┌─────────────────────────┐
│ [defensive_stats.run()] │
│ Add Tkl/Int/Recoveries  │
└────────────┬────────────┘
             │
             ↓
┌─────────────────────────┐
│  Complete Stats         │
│ (all seasons, all cols) │
└────────────┬────────────┘
             │
             ↓
┌─────────────────────────┐
│ [wage_assignment.run()] │
│ Infer missing wages     │
└────────────┬────────────┘
             │
             ↓
┌─────────────────────────┐
│  Clean Dataset Ready    │
│ (797 players, complete) │
└────────────┬────────────┘
             │
             ↓
┌─────────────────────────┐
│  index_data.py          │
│ Type coercion + prose   │
│ generation + embeddings │
└────────────┬────────────┘
             │
             ↓
┌─────────────────────────┐
│  Indexed in Milvus      │
│ (ready for retrieval)   │
└─────────────────────────┘
```

---

## Performance Notes

**Typical runtime** (full pipeline):
- CSV repair: <1s
- Defensive stats: 2-5s (iterates 1000+ rows)
- Wage assignment: 3-10s (infers 270 wages)
- Total: ~5-15s

**Memory usage**:
- Defensive stats: ~100MB (entire CSVs in memory)
- Wage assignment: ~150MB (3 seasons + wages in memory)
- Total: <300MB

**I/O operations**:
- Backups: 5-10 copies to `raw/` directory
- Reads: 3 seasons + 1 wages file
- Writes: 2 repaired CSVs, 3 enhanced CSVs, 1 updated wages file

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) — System design & core decisions
- [HYBRID_INDEXING.md](HYBRID_INDEXING.md) — Dual-index design (what this cleans for)
- [README.md](../Readme.md) — Quick start & indexing instructions
