# Gaffer AI Engine — Complete Architecture Guide

Building an Explainable RAG Scouting System for Football Recruitment

---

## Table of Contents

1. [Overview](#overview)
2. [System Flow](#system-flow)
3. [Technology Stack & Rationale](#technology-stack--rationale)
4. [Core Architectural Decisions](#core-architectural-decisions)
5. [Data Pipeline Architecture](#data-pipeline-architecture)
6. [Query Processing Pipeline](#query-processing-pipeline)
7. [Quality & Evaluation](#quality--evaluation)
8. [Module Organization](#module-organization)
9. [Runtime Architecture](#runtime-architecture)
10. [Hard-Won Lessons](#hard-won-lessons)
11. [Design Philosophy](#design-philosophy)

---

## Overview

Gaffer AI Engine is a **Retrieval-Augmented Generation (RAG) system** for football scouting. It answers natural language queries about player recruitment by:

1. **Retrieving** real player data and tactical theory
2. **Ranking** candidates via hybrid (vector + scalar) search
3. **Enriching** results with manager, wage, and career context
4. **Generating** a grounded scouting brief with visible reasoning

**Key differentiator**: Explainability is the product. Every decision is traced, every number is sourced, every output is auditable. Users see not just "who to scout" but **why** and **how confident** we are.

---

## System Flow

### End-to-End Pipeline

```
User Query (plain English)
    ↓
[Phase 1] Embed query (384-d)
    ↓
[Phase 2] Understand query → hard filters + ranking signals
    ↓
[Phase 3] Retrieve tactical theory (Index A: "The Inverted Pyramid" chunks)
    ↓
[Phase 4] Retrieve candidate pool (Index B: hybrid scalar + cosine)
    ↓
[Phase 5] Rank by stats (blend cosine 50/50 with per-90 metrics)
    ↓
[Phase 6] Enrich candidates (join manager, style, wages from reference CSVs)
    ↓
[Phase 7] Project similarity (PCA 384-d → 2-d for visualization)
    ↓
[Phase 8] Generate brief (DSPy ChainOfThought with metrics checklist grounding)
    ↓
Response (brief + reasoning + candidates + impact analysis + scatter plot)
    ↓
UI Timeline (show phases trace, display scatter, render impact drivers)
```

### Data Preparation Pipeline

```
Raw CSVs
    ↓
[Step 1] Repair CSV structure (fix unquoted commas in wages, positions)
    ↓
[Step 2] Add defensive stats (synthesize Tkl/Int for 2024/25 & 2025/26)
    ↓
[Step 3] Assign missing wages (infer for 270 players based on position + performance)
    ↓
[Step 4] Transform stats → prose (convert numbers to natural language descriptions)
    ↓
[Step 5] Aggregate careers (3-year profiles with consistency, momentum, trends)
    ↓
[Step 6] Embed & index (build 3 Milvus collections: theory, season-stats, career)
    ↓
Indexed Data (ready for queries)
```

---

## Technology Stack & Rationale

| Component | Technology | Problem Solved | Why This Choice | Key Integration |
|-----------|------------|-----------------|-----------------|-----------------|
| **Vector Store** | **Milvus Lite** | Need embedded, hybrid-capable store without external infra | Runs in-process from single `.db` file; supports hybrid (vector + scalar pre-filter); zero DevOps | 3 collections: theory, season-stats, career. Pre-filters by season/position/name, ranks by cosine |
| **Embeddings** | **sentence-transformers** (all-MiniLM-L6-v2, 384-d) | Query and docs must share semantic space; need CPU efficiency | Fast, lightweight, sufficient for prose matching; trade-off: cosine scores 0.4-0.6 are normal (relative ranking matters, not absolute magnitude) | Embeds all docs once; 384-d PCA'd to 2-d for UI scatter plot; ~10ms query embedding |
| **LLM Orchestration** | **DSPy** | Avoid brittle hand-written prompts; make reasoning auditable | Declarative signatures enforce input/output structure; ChainOfThought reasoning is machine-readable | `ScoutingReportSignature` defines inputs (context, stats, metrics checklist) and outputs (reasoning bullets + brief); grounding rules ("ONLY cite from checklist") are declarative, not free-form |
| **LLM Provider** | **OpenRouter** (fallback: OpenAI via DSPy/LiteLLM) | Flexible model selection without vendor lock-in; cost control | OpenRouter gives vendor flexibility; code auto-selects (OpenRouter > OpenAI) based on env vars; can swap `openrouter/gpt-4o` ↔ `openai/gpt-4` without code change | DSPy routes via LiteLLM; judge model for DeepEval configurable |
| **Document Parsing** | **Docling + HybridChunker** | Parse EPUB/PDF to structured Markdown preserving layout (tables, headings) | Layout-aware; respects semantic boundaries (heading hierarchy); HybridChunker produces 320-token chunks fitting MiniLM context window | Parses "The Inverted Pyramid" EPUB → Markdown; prepends heading path to chunks for context; privacy-scrubbed before storage |
| **PII Scrubbing** | **Presidio + spaCy** | Protect sensitive data in ingested dossiers | Industry-standard NER-based detection; spaCy NLP backend | Lazy-loaded in parser.py; scrubs contact info from book excerpts (phone, email, passport, credit cards); preserves player/club names |
| **Query Intent Parsing** | **Regex + Curated Concept→Column Map** | Parse natural English into structured intent (hard filters + ranking signals) | Deterministic (no LLM latency/cost); fully transparent; football vocabulary is finite; every mapping reported in UI | `query_understanding.py`: regex → position/numbers → hard filters; concept map (e.g., "high-volume passing" → `PrgP` column); maps shown in phase 2 |
| **Hybrid Ranking** | **Manual Blending (50/50 cosine + per-90 stats)** | Concepts should steer ranking without overriding semantic relevance | 50/50 cosine ↔ stat score lets "midfielder who wins the ball back" surface João Palhinha (152 tackles) even if cosine score lower | Min-max normalize stat columns to [0,1], average, blend with normalized cosine, re-sort top 3; per-90 floor prevents rotation players from dominating raw totals |
| **Web API** | **FastAPI** | Type-safe HTTP routing with automatic OpenAPI docs | Native Pydantic integration; streaming support; built-in validation; zero setup | `/query` returns full 8-phase trace; `/system` shows live index status; `/theory/document` renders HTML |
| **Frontend UI** | **React (Create React App)** | Explainability requires visual, interactive timeline + scatter plot | Proven ecosystem; component-based rendering; real-time reactivity | Phases timeline (INPUT→OUTPUT→WHY), scatter plot (query vs candidates vs corpus), impact drivers per candidate, theory viewer link |
| **Evaluation & Regression Testing** | **DeepEval (custom OpenRouter judge)** | Automated quality assurance without external eval service | Reuses OPENROUTER_API_KEY; judge model swappable; threshold-based pass/fail | FaithfulnessMetric (brief cites only context), AnswerRelevancyMetric (brief answers query); thresholds: 0.85+ |
| **Data Transformation** | **Pydantic** | Strongly-typed data handling and validation at system boundaries | Automatic validation at request boundaries; clear schemas (PlayerStats, TacticalTheory); enables IDE autocomplete | Request/response schemas; player record validation during indexing |
| **Tactical Theory Enrichment** | **Custom Concept→Metrics Map** (tactical_scorer.py) | Translate user's tactical intent to measurable player attributes | Maps concepts ("high press") to FBref metrics ("Succ_Press > 0.75"); includes position-specific thresholds | Metrics checklist ground truth for DSPy; compared against candidate stats |

---

## Core Architectural Decisions

### 1. Hybrid Indexing: Season-Specific + Career Aggregates

**Decision**: Maintain two separate player indices instead of one.

```
Index B-1: player_stats_collection
  - One document per player-season
  - All stats for that season
  - Query: "Best midfielder in 2024/25?" ✓

Index B-2: player_career_collection (NEW)
  - One document per player (all seasons)
  - Aggregated stats, consistency score, momentum, trend, best season
  - Query: "Which midfielders are improving?" ✓
```

**Why?**
- Single-index queries fail: "improving midfielder" doesn't match 2024/25 snapshot (doesn't show trend)
- Query routing automates selection: `is_career_query()` detects keywords ("improving", "declining", "trajectory", "3-year", "momentum") → routes to career index
- Prevents mis-ranking: Career data with "best season" peak would always surface peak year, not current momentum

**Career Profile Structure** (career_aggregation.py):
- **Aggregated stats**: Total, average, best season, worst season per metric
- **Consistency score**: Coefficient of variation (very consistent → highly variable)
- **Trend & momentum**: YoY % change in last season vs prior
- **Best season marker**: Which season peaked
- **Prose profile**: Natural language embedding of all above

---

### 2. Query Routing: Career vs Form Queries

**Decision**: Automatically route queries to appropriate index based on detected keywords.

**Career keywords** (query_understanding.py:22-39):
```python
"career", "improving", "declining", "progression", "trajectory",
"three year", "3 year", "across seasons", "consistent", "consistency",
"trend", "momentum", "long-term", "multi-year", "over time"
```

**When detected** → Query `player_career_collection` (3-year trends)  
**Default** → Query `player_stats_collection` (current season or specified season)

**Why?** Multi-year trends are meaningless on a single season snapshot; form queries on career data would always surface "best season" instead of "current momentum".

---

### 3. Prose-First Embedding ⭐ Core Design

**Decision**: Index players as natural language prose, not raw numbers.

**Indexed prose** (data_transformer.py):
```
"Declan Rice is a midfielder for Arsenal in the 2024/2025 season.
Played 2825 minutes across 28 appearances. A regular goal threat with
12 goals (0.38 per 90). Contributes creatively with 6 assists. Underlying
numbers: 9.2 xG and 4.1 xAG. Ball progression: 156 progressive carries,
89 progressive passes. Defensive actions: 45 tackles, 32 won, 28 interceptions,
156 recoveries (wins the ball back)."
```

**NOT raw stats**:
```
Gls: 12, Ast: 6, xG: 9.2, xAG: 4.1, PrgC: 156, PrgP: 89,
Tkl: 45, TklW: 32, Int: 28, Rec: 156, Min: 2825, Appearances: 28
```

**Why?**
- Queries are prose ("creative midfielder who presses high")
- Embeddings match poorly across asymmetric shapes (prose query ↔ stats dump)
- Prose indexes read like the queries users type → better semantic alignment
- Generated deterministically from stats → reproducible, auditable

**Downside**: Loses numeric precision (prose is lossy), but semantic matching wins over raw numeric matching.

---

### 4. Dual Indices: Tactical Theory + Player Stats

**Decision**: Separate indices for concepts and facts.

```
Index A: tactical_theory_collection
  Chunks of "The Inverted Pyramid" (+ Football Hackers)
  → Conceptual grounding ("gegenpressing requires press success > 75%")

Index B: player_stats_collection + player_career_collection
  Player profiles (stats + career arcs)
  → Factual evidence ("João Palhinha has 78% press success")
```

**Why?** Mixing would blur retrieval. Brief can cite **theory** for "why" (what does the tactic require?) and **stats** for "who" (which real players fit?). Separation ensures each serves its purpose.

---

### 5. Hybrid Retrieval: Scalar Pre-Filter + Vector Rank

**Decision**: Use Milvus hybrid search: apply scalar filters first, rank remaining rows by cosine.

```
Query: "midfielder with >10 goals who plays for top-6 clubs"

Step 1 (Scalar pre-filter): Filter by position="MF" AND goals >= 10
       Result: ~150 candidates

Step 2 (Vector rank): Embed query, compute cosine vs ~150 embeddings
       Result: Top 25 candidates
```

**Why?**
- **Hard requirements** (goals >= 10) must be precise → filtering is cheap + guaranteed
- **Fuzzy concepts** ("midfielder who wins the ball back") are ranking signals, not filters
- Pre-filtering avoids computing cosine on irrelevant rows (cost savings)
- **Graceful degradation**: If 0 candidates pass filters, relaxation ladder drops constraints in order: stats thresholds → min-minutes → player name → position → season

---

### 6. Query Understanding: Deterministic Intent Parsing

**Decision**: Parse queries into hard filters and ranking signals via regex + curated maps, not LLM.

```python
detect_position("attacking midfielder") → "MF"
detect_stats_threshold("more than 10 goals") → "goals >= 10"
detect_ranking_concept("high-volume passing") → rank_by_column="PrgP"
```

**Why?**
- No LLM latency or cost
- Fully transparent and auditable (every mapping shown in phase 2)
- Football vocabulary is finite and well-known
- Error cases are predictable (show "could not detect position" vs silent failures)
- Can be gradually improved without LLM retraining

**Trade-off**: Loses generality (only understands pre-mapped concepts), but gain in reliability and cost.

---

### 7. Blended Re-Ranking: The "Even Better Way"

**Decision**: After hybrid retrieval returns ~25 candidates, blend cosine + stat scores 50/50.

```
For each candidate:
  1. Per-90 normalize each rank_by column to [0,1]
     (rate stats / 90, volume stats as-is)
  2. Average normalized columns → stat_score
  3. stat_score = 0.5 * stat_score + 0.5 * cosine_score
  4. Re-sort by blended score, take top 3
```

**Example**: "Midfielder who wins the ball back"
- Query cosine might rank player A higher (0.72 vs 0.68)
- But João Palhinha has 152 tackles vs player A's 95
- Blended score surfaces Palhinha (#1) because stat signal overrides marginal cosine diff

**Why?**
- Concepts correctly break ties when cosine scores are close
- Degrades gracefully if no stat signal (falls back to pure cosine)
- Per-90 floor (`_MIN_NINETIES = 3.0`, ~270 mins) prevents rotation players from dominating raw totals

---

### 8. Grounding & Honesty

**Decision**: No LLM calls if data is missing; return explicit "no data" instead of invented briefs.

```python
# From engine.py:784-815
if not candidates:
    return {"scouting_brief": "No data available for these constraints",
            "candidates": []}
```

**Also**:
- **Reference data joined at query time** (not embedded): Manager, wages come from CSVs, so they're grounded facts
- **Season-aware manager lookup** (reference.py): Picks manager who finished that season; non-sacked rows preferred
- **Metrics checklist as ground truth**: LLM instructed "ONLY cite from checklist, do NOT invent thresholds"

**Why?** Prevents confident-sounding invented briefs when data doesn't support them.

---

### 9. Scouting for Your Club (Exclusion Filter)

**Decision**: Allow users to exclude players from their own club.

```python
# Detect club from explicit field OR extract from query
club = detect_club(query) or scouting_for
# Drop players from that club from results
```

**Why?** You want external recruitment candidates, not existing squad members.

---

### 10. Explainability is the Product

**Decision**: Make every step of the pipeline visible and auditable.

- **Phases trace**: 8-phase timeline in UI showing INPUT, OUTPUT, WHY, timing
- **System endpoint** (`/system`): Live index row counts, embedding model, LLM status
- **Similarity scatter**: Query + candidates + background corpus projected to 2-d PCA with cosine scores
- **Theory viewer**: Renders Docling-structured Markdown as HTML, linked from phase 3
- **Bulleted reasoning**: DSPy signature instructed to produce bullets (UI renders as scannable list)
- **Impact drivers**: Why each candidate was chosen (top stats, career context, risk factors)

**Why?** Users don't trust black boxes. By showing work, we build confidence and enable debugging.

---

## Data Pipeline Architecture

### 3 Cleaning Steps + 2 Enhancement Steps

**Step 1: CSV Repair** (csv_repair.py)
- **Problem**: FBref CSVs have unquoted commas ("£27,300,000", "MF,FW") that break CSV parsing
- **Solution**: Reconstruct rows from known schema, rewrite with proper quoting
- **Files affected**: player_stats_25-26.csv, player_wages.csv
- **Backup**: Originals copied to data/epl_seasons/raw/

**Step 2: Defensive Stats** (defensive_stats.py)
- **Problem**: 2024/25 & 2025/26 exports lack tackles/interceptions (only 2023/24 has them)
- **Solution**: Synthesize deterministically using per-position baselines (TKL_PER90, INT_PER90, REC_PER90) scaled by minutes + deterministic per-player factor
- **Important**: Values are MOCK (marked clearly); swap in real FBref data if available
- **Result**: 3-season parity for hybrid ranking

**Step 3: Wage Assignment** (wage_assignment.py)
- **Problem**: 270 players missing from wages.csv (34% gap breaks ranking UI)
- **Solution**: Infer from position + performance (goals+assists) + minutes
- **Performance tiers**:
  - High (>20 goals+assists): 0.8–1.0× position max
  - Medium (>10): 0.5–0.8×
  - Regular (>5): base_min+100k to 0.5×
  - Starter (>1800 mins): base_min+50k to base_min+150k
  - Fringe: base_min to base_min+50k
- **Result**: Complete 797-player dataset

**Step 4: Data Transformer** (data_transformer.py, called by index_data.py)
- **Input**: Raw stats dictionaries
- **Output**: Prose descriptions (see "Prose-First Embedding" section)
- **Why separate**: Reusable for all indexing; can be enhanced (e.g., add injury context)

**Step 5: Career Aggregation** (career_aggregation.py, called by index_data.py)
- **Input**: Player's 3-season stat history
- **Output**: Career profile with aggregates, consistency score, momentum, trend, best season
- **Consistency metric**: Coefficient of variation
  - `< 0.2`: very consistent
  - `0.2–0.4`: consistent
  - `0.4–0.6`: variable
  - `> 0.6`: highly variable
- **Why separate**: Can be independently improved (e.g., add injury resilience tracking)

### Modular Structure

All 5 steps live in `backend/scripts/data_cleaning/` with shared utilities:

```
common.py
  └── num(), per90(), pct(), position_phrase(), backup_file(), DATA_DIR, RAW_DIR

pipeline.py
  └── run_cleaning_pipeline() orchestrates all steps in order
```

**Why modular?**
- Each step is independently testable
- Can run individual steps (e.g., just wage assignment)
- New steps bolt in without touching others
- Shared utilities avoid duplication

---

## Query Processing Pipeline

### Phase-by-Phase Breakdown

**Phase 1: Embed Query**
- Use pre-trained `all-MiniLM-L6-v2` (384-d)
- Output: `query_vector` (384-d float array)
- Timing: ~10ms

**Phase 2: Understand Query**
- **Parse intent**: Regex → detect position, numbers, club
- **Curated concept→column map**: "high-volume passing" → rank_by="PrgP"
- **Output**: `{hard_filters: {season, position, player_name, ...}, rank_by_columns: [...]}`
- **Transparency**: All mappings shown to user

**Phase 3: Retrieve Tactical Theory**
- Query `tactical_theory_collection` by cosine similarity
- Return: Top ~3 chunks with highest similarity scores
- **Purpose**: Conceptual grounding ("gegenpressing requires…")
- **UI link**: Theory viewer (renders Docling HTML)

**Phase 4: Retrieve Candidate Pool**
- **Route decision**: Is career query? (from phase 2 keywords)
  - Yes → Query `player_career_collection`
  - No → Query `player_stats_collection`
- **Hybrid search**: Apply scalar filters (season, position, goals >= 10, etc.), rank remaining ~25 by cosine
- **Relaxation ladder** (if 0 candidates):
  1. Drop stat thresholds
  2. Drop min-minutes
  3. Drop player name
  4. Drop position
  5. Drop season
- **Output**: ~25 candidates with stats, squad, season, cosine score

**Phase 5: Rank by Stats**
- **Per-90 normalize** each rank_by column
- **Blend** 50/50 cosine + stat score
- **Re-sort**, take top 3
- **Output**: Top 3 candidates with blended scores

**Phase 6: Enrich Candidates**
- **Join manager** (managers.csv, season-aware lookup)
- **Join wages** (player_wages.csv)
- **Calculate progression** (3-season history if available)
- **Build metrics checklist** (tactical scorer)
- **Output**: Fully enriched candidate profiles

**Phase 7: Project Similarity**
- **PCA** 384-d → 2-d across [query + candidates + corpus]
- **Compute cosine** similarity of all points to query
- **Output**: Scatter plot points (label, x, y, similarity, type, club, position, season)

**Phase 8: Generate Brief**
- **Input to DSPy**:
  - Query intent (from phase 2)
  - Candidate stats (from phase 6)
  - **Metrics checklist** (filtered to query-relevant metrics)
  - Tactical theory (from phase 3)
- **DSPy ChainOfThought**:
  - Produces reasoning bullets
  - Generates scouting brief
  - **Grounding rule**: "ONLY cite from metrics checklist, do NOT invent thresholds"
- **Output**: Reasoning + brief + impact drivers

---

## Quality & Evaluation

### DeepEval Integration

**Metrics**:
- **FaithfulnessMetric**: Is the brief grounded only in retrieved context? (no hallucinations)
- **AnswerRelevancyMetric**: Does the brief actually answer the query?
- **Threshold**: 0.85+ (85%) on both; test fails if below

**Judge model**: Custom OpenRouterJudge (reuses OPENROUTER_API_KEY); swappable for OpenAI default

**Integration** (engine.py → main.py `/evaluate` endpoint):
```python
@app.post("/evaluate")
def evaluate_brief(req: EvalRequest):
    return evaluate_scouting_brief(
        req.query,
        req.scouting_brief,
        req.context
    )
```

### DeepEval Improvements (3 Changes)

**1. Enhanced DSPy Signature** (engine.py:ScoutingReportSignature)
- Added verification-first rules: "DO NOT invent metrics", "ONLY cite from checklist"
- Example format: `"Tackles 160 (target > 150) ✓"`
- Why: Reduces hallucination risk; LLM knows exact format expected

**2. Checklist Filtering** (engine.py:_filter_checklist_by_concept)
- Show only query-relevant metrics to LLM
- E.g., if query is "improving midfielders", only MOMENTUM section shown
- Why: Reduces cognitive load; fewer off-topic divergences

**3. Updated forward() Method** (engine.py)
- Filter checklist before passing to DSPy
- Focused context yields better grounding

**Expected improvement**: 84% → 90–95% (Faithfulness +8–12%, Answer Relevancy +5–12%)

### Metrics Checklist Approach

Ground truth for LLM; prevent hallucination via structure:

```
CONCEPT: High-Volume Passing (PrgP > 100)
  Player: João Moutinho (134) ✓

CONCEPT: Defensive Intensity (Tkl > 100)
  Player: João Palhinha (152) ✓

CONCEPT: Goal Contribution (Gls + Ast > 15)
  Player: Bruno Fernandes (32) ✓
```

**LLM instructions**: "ONLY cite from this checklist. Do NOT invent metrics or thresholds."

### Impact Analysis

Surfaces why each player was chosen (enrichment/impact_analysis.py):

- **Impact drivers**: Top stats (goals, tackles)
- **Career context**: Trend, momentum, consistency (if 3+ seasons available)
- **Risk factors**: Declining trajectory, small sample size
- **Confidence level**: High (improving momentum), medium, or low (declining)

---

## Module Organization

### backend/app/ (Core System)

```
main.py                    FastAPI app, CORS, lifespan, endpoints
engine.py                  ScoutIntelRAG (8-phase pipeline, blended ranking)
query_understanding.py     Intent parsing (regex + concept maps)
query_mapper.py            Tactical concept → FBref metric mapping
database.py                Milvus client, embeddings, hybrid queries
reference.py               Manager + wage data, club detection/normalization
parser.py                  Docling EPUB/PDF parsing (lazy-loaded)
privacy.py                 Presidio/spaCy PII scrubbing (lazy-loaded)
schemas.py                 Pydantic models (PlayerStats, TacticalTheory, etc.)
llm.py                     OpenRouter judge for DeepEval
tactical_scorer.py         Metrics checklist builder
config.py                  CORS middleware setup
services/
  evaluation.py            DeepEval integration
  theory.py                Theory document fetching
  markdown.py              Markdown → HTML conversion
enrichment/                Response enrichment
  impact_analysis.py       Why was this player recommended?
  visualization.py         2D PCA scatter + cosine visualization
  progression.py           Career trajectory, momentum, consistency
```

### backend/scripts/ (Data & Indexing)

```
index_data.py              Build indices (prose descriptions + embeddings)
reindex_theory.py          Utility: re-index tactical theory only
data_cleaning/             Modular cleaning pipeline
  __init__.py              Public API (run_cleaning_pipeline)
  common.py                Shared utilities (num, per90, position_phrase, etc.)
  csv_repair.py            Fix unquoted commas
  defensive_stats.py       Synthesize Tkl/Int/Recoveries
  wage_assignment.py       Infer missing player wages
  data_transformer.py      Stats → prose descriptions
  career_aggregation.py    3-year profile consolidation
  privacy_handler.py       PII scrubbing wrapper
  pipeline.py              Orchestrate all steps
```

### frontend/src/

```
App.js                     Main React component (phases timeline, scatter, candidates)
ImpactAnalysis.jsx         Impact drivers + career context display
... (supporting components)
```

---

## Runtime Architecture

### Startup Ordering (main.py:lifespan)

```python
async def lifespan(app: FastAPI):
    init_collections()  # Fork Milvus Lite FIRST
    init_dspy()         # Spawn dspy/torch threads SECOND
    yield
```

**Critical**: Milvus fork fails if torch has already spawned threads on macOS. Must init collections before any deep-learning libs.

### Environment Guards (database.py:10-15)

```
GRPC_ENABLE_FORK_SUPPORT=1      # Milvus fork-awareness
GRPC_POLL_STRATEGY=poll          # gRPC polling (not epoll, which breaks fork)
TOKENIZERS_PARALLELISM=false     # Prevent transformers warnings
KMP_DUPLICATE_LIB_OK=TRUE        # Allow duplicate OpenMP (torch, spaCy, Milvus each bundle it)
```

**Why?** Milvus (gRPC), transformers, spaCy, torch each bundle OpenMP; multiple copies abort unless duplicates allowed.

### Single-Process Constraint

Milvus Lite allows only 1 process to open `.db` at a time:

- `get_client()` retries briefly, then fails with clear message
- `start.sh` kills prior instances + clears lock file before startup
- `--reload` handoff covered by brief retry window (covers uvicorn hot-reloading)

---

## Hard-Won Lessons

### 1. Segfault on Startup (SIGSEGV on macOS)

**Problem**: Milvus forks its embedded server after threads exist → native crash

**Root cause**: sentence_transformers imports torch at module level; uvicorn starts threads during import

**Fixes**:
- Lazy imports of torch/Docling (load only on first use, not at module import)
- Explicit lifespan ordering: init_collections BEFORE init_dspy
- gRPC fork-awareness flags (GRPC_ENABLE_FORK_SUPPORT=1)

---

### 2. Single-Process Lock Hangs

**Problem**: Second process trying to open `.db` hangs indefinitely

**Root cause**: Milvus Lite enforces exclusive lock; no timeout

**Fix**:
- `get_client()` retries 8×, then fails with actionable message
- `start.sh` kills prior instances before startup
- Reloader brief retry window covers uvicorn --reload handoff

---

### 3. Eager DB Open Starves Worker

**Problem**: Earlier code opened Milvus at import time; uvicorn reloader grabbed lock, starved the worker process

**Fix**: Open DB only inside lifespan (not at module import)

---

### 4. Result Concatenation Corruption

**Problem**: Concatenating pymilvus results with `+=` corrupted internal bitmap

**Fix**: Materialize results to plain dicts before combining

---

## Design Philosophy

### 1. Explainability Over Hidden Power

- Every step of the pipeline is visible
- User sees INPUT, OUTPUT, WHY for each phase
- Confidence scores are explicit

### 2. Grounding Over Hallucination

- Metrics checklist as ground truth
- No LLM calls if data is missing
- Reference data joined at query time, not embedded

### 3. Transparency Over Accuracy

- Regex + curated maps (not LLM) for intent parsing
- Every mapping shown in phase 2
- User can see what was understood/misunderstood

### 4. Modularity Over Monoliths

- 9 independent data cleaning steps
- 3 separate indices (theory + season-stats + career)
- Reusable components (common.py utilities)

### 5. Local Over Cloud

- Milvus Lite (embedded) instead of managed vector DB
- No external services required for core scouting
- Single `.db` file for portability

### 6. Semantic Quality Over Raw Metrics

- Prose indexing (not number dumps)
- Per-90 normalization (prevents sample-size bias)
- Career consistency tracking (identifies reliable performers)

---

## Summary: Why Each Technology

| Priority | Technology | Benefit |
|----------|-----------|---------|
| **Explainability** | DSPy, phases trace, UI timeline | Users understand why each decision was made |
| **Grounding** | Metrics checklist, reference joins, no-bluff | Prevents confident-sounding hallucinations |
| **Transparency** | Regex parsing, curated concept maps, shown in UI | Every intent mapping is auditable |
| **Quality** | Prose indexing, per-90 norm, career consistency | Better semantic match + bias awareness |
| **Reliability** | Milvus Lite (embedded), no external deps | Zero infrastructure; portable single-file DB |
| **Cost** | MiniLM (384-d), local parsing, deterministic intent | Cheap embeddings, no LLM latency on intent parsing |


---

## Next Steps for Implementation

1. **Data preparation**: Run the cleaning pipeline (csv_repair → defensive_stats → wage_assignment → transform → aggregate)
2. **Indexing**: Build the 3 Milvus collections (theory, season-stats, career)
3. **Query loop**: Test the 8-phase pipeline with realistic scouting queries
4. **Evaluation**: Run DeepEval regression tests; aim for 90%+ on Faithfulness + Answer Relevancy

---
