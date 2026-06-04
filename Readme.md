# Gaffer AI Engine - Scouting Intelligence Platform

**Explainable RAG-based scouting system for football recruitment with unified blended indexing, grounded LLM generation, and quality-assured outputs.**

---

## 📚 Documentation (Quick Links)

### 🎯 **Start Here**
- **[COMPLETE_GUIDE_FOR_BEGINNERS.md](COMPLETE_GUIDE_FOR_BEGINNERS.md)** — Everything explained for beginners
  - What is RAG? (Retrieval-Augmented Generation)
  - Complete 8-phase query walkthrough
  - All AI terminology explained simply
  - Integration of Docling (book parsing), DSPy (brief generation), DeepEval (quality validation)
  - Blended collection architecture

- **[complete-flow-client-server.excalidraw](complete-flow-client-server.excalidraw)** — Visual architecture diagram
  - Client-server flow with arrows
  - All 8 phases of query processing
  - Tool integration points
  - Database layer (Milvus + CSVs)

---

## 🚀 Quick Start

```bash
./start.sh
```

- Backend: http://localhost:8000
- Frontend: http://localhost:3000

---

## 📦 Indexing Data

The system requires indexed data to function. Data indexing is **not automatic** — it must be run explicitly.

### Initial Setup (First Time)

Run the complete indexing pipeline:

```bash
uv run python -m backend.scripts.index_data
```

This executes **6 steps** in order:

**Step 1: Data Cleaning** (modular, in `backend/scripts/data_cleaning/`)
- CSV repair: Fix unquoted commas in position/wage fields
- Defensive stats: Fill estimated tackles/interceptions
- Wage assignment: Infer wages for 270 missing players
- Data transformation: Convert stats to prose descriptions
- Career aggregation: Create 3-year player profiles

**Step 2-6: Indexing & Milvus Setup** (from `backend/scripts/index_data.py`)
- Initialize Milvus collections (Blended Index A + B)
- Index tactical reference from books (theory chunks)
- Index player stats for 2024/25 season
- Index player stats for 2023/24 season
- Generate and index career aggregate profiles

**Result:** 2,391 indexed documents (797 players × 3 seasons) + ~100 theory chunks

### After Data Updates

If you update CSV files in `data/epl_seasons/`, rerun the full pipeline:

```bash
uv run python -m backend.scripts.index_data
```

This re-cleans and re-indexes all data from scratch.

### Environment Variables

Control indexing behavior with:

```bash
# Limit indexing to first N players per season (0 = no limit)
MAX_PLAYERS_PER_SEASON=50 uv run python -m backend.scripts.index_data
```

Useful for testing/debugging.

---

## ✨ Key Features

✅ **Blended Collection** — Unified season + career data in single Milvus index  
✅ **8-Phase Pipeline** — Embed → Parse → Theory → Route → Retrieve → Blend → Enrich → Generate  
✅ **Three Tools** — Docling (parsing) + DSPy (grounded generation) + DeepEval (quality validation)  
✅ **Hybrid Search** — Combines semantic vectors (50%) with statistical ranking (50%)  
✅ **Grounded Briefs** — No hallucinations, every fact is sourced and traceable  
✅ **Complete Data** — 2,391 player records (797 players × 3 seasons) + tactical theory  

---

## 🏗️ Architecture Highlights

**Blended Vector Indexing** ⭐ *Unified approach*
- Single Milvus collection stores BOTH season stats AND career metrics per player
- Season-specific: goals, tackles, press success, progressive passes, etc.
- Career metrics: momentum (+8%), consistency (0.85), trend (improving), best season
- Simplifies queries: no routing needed, one collection handles both form and career queries
- 2,391 documents (797 players × 3 seasons)

**8-Phase RAG Pipeline**
1. **Embed** — Convert query to 384-d vector (sentence-transformers)
2. **Parse** — Extract intent using regex + concept mapping
3. **Retrieve Theory** — Get tactical knowledge (Docling-parsed books)
4. **Route** — Detect career vs form queries
5. **Retrieve Candidates** — Hybrid search (filters + cosine similarity)
6. **Blend Scores** — 50% vector similarity + 50% statistical ranking
7. **Enrich** — Join manager, wages, club data
8. **Generate Brief** — DSPy structured generation with grounding constraints

**Quality-First with Tools**
- 🔴 **Docling**: Parse books into semantic chunks
- 🟢 **DSPy**: Structured brief generation (no hallucinations)
- 🔵 **DeepEval**: Validate with 3 metrics (Faithfulness, Relevancy, Contextual)
- Result: 90-95% quality outputs, fully auditable

---

## 🎯 Getting Started (By Role)

**Product Managers & Scouts:**
- Read [COMPLETE_GUIDE_FOR_BEGINNERS.md](COMPLETE_GUIDE_FOR_BEGINNERS.md) Part 1 (What is RAG?)
- Open [complete-flow-client-server.excalidraw](complete-flow-client-server.excalidraw) for visual overview
- Understand: RAG retrieves real data, AI writes grounded briefs, DeepEval validates quality

**Backend & Full-Stack Engineers:**
- Read [COMPLETE_GUIDE_FOR_BEGINNERS.md](COMPLETE_GUIDE_FOR_BEGINNERS.md) Parts 4-6 (8 phases, blended collection)
- Check `backend/app/engine.py` for phase orchestration
- Review `backend/app/database.py` for Milvus operations

**AI/ML & LLM Engineers:**
- Study [COMPLETE_GUIDE_FOR_BEGINNERS.md](COMPLETE_GUIDE_FOR_BEGINNERS.md) Part 5 (Blended scoring, 50/50 weighting)
- Check `backend/generation/dspy_framework.py` for DSPy signature + constraints
- Review `backend/evaluation/deepeval_metrics.py` for quality metrics (Faithfulness, Relevancy, Contextual)

---

## 📂 Project Structure

```
gaffer-ai-engine/
├── docs/                       # Complete documentation
├── backend/
│   ├── app/                   # Core system modules
│   │   ├── ...core modules (RAG engine, database, LLM, query understanding)
│   │   └── enrichment/        # Response enrichment (explain & visualize)
│   │       ├── impact_analysis.py    # Why was player recommended?
│   │       ├── visualization.py      # PCA scatter plot + similarity
│   │       └── progression.py        # Career trajectory & momentum
│   ├── scripts/
│   │   ├── index_data.py      # Main indexing orchestrator
│   │   ├── reindex_theory.py  # Utility: re-index tactical theory
│   │   └── data_cleaning/     # Data cleaning & enhancements
│   │       ├── common.py           # Shared utilities
│   │       ├── csv_repair.py       # Fix CSV structure
│   │       ├── defensive_stats.py  # Fill defensive stats
│   │       ├── wage_assignment.py  # Infer wages
│   │       ├── data_transformer.py # Stats → prose conversion
│   │       ├── career_aggregation.py # 3-year career profiles
│   │       ├── privacy_handler.py  # PII scrubbing
│   │       └── pipeline.py         # Orchestrator
│   └── tests/                 # Quality assurance
├── frontend/                   # React UI components
├── data/                       # EPL season data (3 years)
├── Readme.md                  # This file
└── start.sh                   # Quick start script
```

---

## 📊 System Diagram

See [complete-flow-client-server.excalidraw](complete-flow-client-server.excalidraw) for the complete visual architecture showing:
- Client-server communication
- All 8 phases of query processing
- Tool integration (Docling, DSPy, DeepEval)
- Milvus blended collection
- Reference data (CSVs)
