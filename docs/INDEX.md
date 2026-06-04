# Documentation Index

Complete documentation for the Gaffer AI Engine scouting system.

---

## 📖 Documentation

- **[ARCHITECTURE_COMPLETE.md](ARCHITECTURE_COMPLETE.md)** — **⭐ START HERE** — Comprehensive 10+ technology stack overview (DSPy, Milvus, Pydantic, Docling, Presidio, DeepEval, sentence-transformers, FastAPI, React). Covers 8-phase query pipeline, data cleaning with 9 modules, quality metrics, hard-won engineering lessons, and design philosophy. The definitive guide to building a Scouting RAG system.

- **[DATA_CLEANING_PIPELINE.md](DATA_CLEANING_PIPELINE.md)** — Modular data cleaning & enhancement architecture. 9 independent, testable modules (CSV repair, defensive stats, wage assignment, transformation, aggregation, etc.) with shared utilities. Explains both the pipeline orchestration and individual module reference.

- **[REALISTIC_SCOUTING_QUERIES.md](REALISTIC_SCOUTING_QUERIES.md)** — 15 example scouting queries demonstrating system capabilities. Real-world examples you can test immediately to understand query routing, retrieval, ranking, and brief generation.

---

## 🎯 Quick Navigation

**Getting Started:**
1. Read [ARCHITECTURE_COMPLETE.md](ARCHITECTURE_COMPLETE.md) for the full picture (all technologies, design decisions, hard-won fixes)
2. Review [REALISTIC_SCOUTING_QUERIES.md](REALISTIC_SCOUTING_QUERIES.md) to see the system in action
3. Refer to [DATA_CLEANING_PIPELINE.md](DATA_CLEANING_PIPELINE.md) when indexing or modifying data

**By Role:**

- **Scouts & Recruiters**: Start with [REALISTIC_SCOUTING_QUERIES.md](REALISTIC_SCOUTING_QUERIES.md) to understand what queries you can ask
- **Engineers & Developers**: Read [ARCHITECTURE_COMPLETE.md](ARCHITECTURE_COMPLETE.md) for the full technical stack, then [DATA_CLEANING_PIPELINE.md](DATA_CLEANING_PIPELINE.md) for data architecture
- **AI/LLM Engineers**: [ARCHITECTURE_COMPLETE.md](ARCHITECTURE_COMPLETE.md) section 8 (Quality & Evaluation) covers DeepEval metrics, grounding, and DSPy signatures

---

## 📂 File Organization

```
docs/
├── INDEX.md (this file — navigation guide)
├── ARCHITECTURE_COMPLETE.md (⭐ comprehensive guide)
├── DATA_CLEANING_PIPELINE.md (modular data pipeline)
└── REALISTIC_SCOUTING_QUERIES.md (example queries)
```

---

## 🚀 Key Features

✅ **Hybrid Indexing** — Season-specific + 3-year career profiles  
✅ **Smart Query Routing** — Auto-detects career vs. form queries  
✅ **Impact Analysis** — Explains why each player was recommended  
✅ **Quality Scoring** — DeepEval: 90-95% faithfulness & relevancy  
✅ **Complete Wages** — 797 players with assigned realistic wages  
✅ **Tactical Theory** — Integrated book reference (The Inverted Pyramid, Football Hackers)  

---

## 💡 Last Updated

June 4, 2026 (ARCHITECTURE_COMPLETE.md added)

**Version history:**
- **2.0**: Added comprehensive ARCHITECTURE_COMPLETE.md with full technology rationale and hard-won lessons
- **1.0**: Initial architecture documentation with core decisions

---

## 📞 Questions?

Each document is self-contained and should answer most questions. Start with the document relevant to your role above.
