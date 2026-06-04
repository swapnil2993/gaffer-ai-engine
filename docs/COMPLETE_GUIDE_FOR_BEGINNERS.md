# Gaffer AI Engine - Complete Beginner's Guide
## Understanding RAG, Blended Search, Tools, and AI Concepts

---

## Part 1: What is RAG? (The Big Picture)

### The Problem: Why AI Needs Help Finding Answers

Imagine asking an AI: **"Find a creative midfielder who presses high"**

**Without RAG** (old approach):
- AI only knows what it was trained on (outdated data)
- AI makes up players that don't exist
- AI can't cite sources

**With RAG** (our approach):
- AI retrieves REAL DATA first (current season stats)
- AI retrieves THEORY second (what "creative" means)
- AI generates brief citing ONLY the retrieved facts
- Everything is traceable

### What Does RAG Stand For?

**RAG = Retrieval-Augmented Generation**

```
Retrieval       → Get relevant info from database
Augmented       → Enhance the AI with that info
Generation      → Use it to write the answer
```

---

## Part 2: Complete Query Flow (8 Phases)

### User Query: "Find an improving creative midfielder"

**Phase 1: EMBED**
```
Query: "Find an improving creative midfielder"
↓ [sentence-transformers]
Output: [0.234, -0.567, 0.891, ..., -0.123]  ← 384 numbers
Why: AI works with numbers, not words
```

**Phase 2: PARSE INTENT**
```
Regex + Concept Mapping
↓
Extracts:
  • Position: "midfielder"
  • Concepts: ["improving", "creative"]
  • Career mode: YES (keyword "improving" found)
```

**Phase 3: RETRIEVE THEORY**
```
Search Milvus Index A (Theory from Docling)
↓
Found: "Creativity = 150+ progressive passes"
       "Improvement = positive momentum trend"
Why: Ground the brief with knowledge
```

**Phase 4: ROUTE QUERY**
```
Check: Is this a career query?
Answer: YES (improving = career keyword)
Result: Use blended collection (handles both)
```

**Phase 5: RETRIEVE CANDIDATES**
```
Hybrid search on blended collection:
  • Filter: position = midfielder, minutes > 500
  • Vector search: cosine similarity to query
  • Result: ~50 candidates
Example candidates:
  1. João Palhinha: cosine=0.68, momentum=+8%
  2. Alexis Mac Allister: cosine=0.72, momentum=+5%
```

**Phase 6: BLEND SCORES**
```
For each candidate:
  cosine_score = 0.68
  stat_score = (creativity + momentum + consistency) / 3
  blended = 0.5 × cosine + 0.5 × stat_score
  
João: 0.5 × 0.68 + 0.5 × 0.78 = 0.73
Result: Ranked #1 (improved momentum matters!)
```

**Phase 7: ENRICH**
```
Join reference data:
  + Manager: Marco Silva
  + Salary: £45k/week
  + Club: Fulham (8th position)
Why: Add context for recruiting decision
```

**Phase 8: GENERATE BRIEF**
```
[DSPy Framework]
Input:
  • Query + Theory + Candidate Stats
  • Metrics Checklist: ["momentum", "press_success", "prog_passes"]
  
DSPy Constraint: "ONLY cite checklist metrics"

Output:
  "João shows improvement with +8% momentum.
   His 78% press success exceeds requirement.
   However, 89 progressive passes are below 
   150 needed for deep creativity."
```

**QUALITY GATE: DEEPEVAL**
```
Validate brief with 3 metrics:

1. Faithfulness (0.94): Only cite context? ✓
2. Relevancy (0.91): Answer query? ✓
3. Contextual (0.96): Facts relevant? ✓

All > 0.85? → Send to user ✓
```

---

## Part 3: AI Terminology Explained

### 1. EMBEDDING
Converting text to 384 numbers that represent meaning.
```
"creative midfielder" → [0.234, -0.567, ..., 0.123]
```

### 2. VECTOR DATABASE (Milvus)
Database storing information as numbers, finds similar items fast.
```
Search "pressing" → Finds "press", "pressure", "defense"
(by meaning, not just keyword match)
```

### 3. SEMANTIC SEARCH
Finding by MEANING, not keywords.
```
"press high" matches "aggressive pressing" 
(same meaning)
```

### 4. COSINE SIMILARITY
Number showing how similar two vectors are (0-1).
```
Player A vs Query: 0.68 similarity
Player B vs Query: 0.72 similarity
(higher = more similar)
```

### 5. HYBRID SEARCH
Using BOTH filters AND semantic search.
```
Filter: position = midfielder (exact)
Search: "creative" (fuzzy semantic match)
```

### 6. BLENDED COLLECTION
Single database storing season stats AND career metrics together.
```
OLD: João's 2024/25 in Collection 1, João's career in Collection 2
NEW: João's 2024/25 with career metrics in SAME document
Benefit: Query once, get both
```

### 7. BLENDED SCORE
Combining semantic (vector) matching with statistical signals.
```
50% vector similarity + 50% stats ranking
= balanced scoring that considers both
```

---

## Part 4: Tools Integration

### 🔴 DOCLING - Parse Books (Setup Phase)

**What**: Converts PDF/EPUB to structured chunks
**Why**: Better structure → Better embeddings
**Result**: ~100 chunks stored in Milvus Index A

Example:
```
Input: "The Inverted Pyramid.epub"
↓ [Docling parsing]
Output: "Pressing requires 75% tackle success.
         Creativity requires 150+ progressive passes."
```

### 🟢 DSPY - Generate Brief (Phase 8)

**What**: Structured LLM prompt with constraints
**Why**: Prevents hallucination (AI making stuff up)
**Constraint**: "ONLY cite metrics from checklist"

Example:
```
Input: Query + Theory + Stats + Checklist
↓ [DSPy enforces schema]
Output: Guaranteed structure + only cited facts
```

### 🔵 DEEPEVAL - Quality Validation

**What**: Automated metrics checking brief quality
**Why**: Catch bad briefs before user sees them
**Metrics**: 
  1. Faithfulness (only cite context)
  2. Relevancy (answers query)
  3. Contextual (facts are relevant)

---

## Part 5: Blended Collection Explained

### What is Blended Collection?

**OLD (Separate)**:
```
Collection 1: Season stats (goals, tackles, etc.)
Collection 2: Career stats (momentum, trend, etc.)
Problem: Must choose which to query
```

**NEW (Blended)**:
```
Collection 1: Unified document with BOTH
  • Season stats (current year)
  • Career metrics (3-year trends)
  
Benefit: Query ONCE, get both
```

### Why Blended is Better

| Aspect | Separate | Blended |
|--------|----------|---------|
| Storage | Player appears 2x | Player appears 1x |
| Routing | Complex (switch indexes) | Simple (query once) |
| Flexibility | Form/career queries separate | Both work together |

---

## Part 6: Complete Example Walkthrough

### Query: "Find improving creative midfielder"

**Expected Output**:
```json
{
  "brief": "João shows improvement with +8% momentum. 
            His 78% press success exceeds requirement. 
            However, 89 progressive passes are below ideal 
            for deep creativity.",
  "candidates": [
    {
      "player": "João Palhinha",
      "score": 0.73,
      "momentum": "+8%",
      "press_success": "0.78",
      "prog_passes": 89
    },
    ...
  ],
  "quality_scores": {
    "faithfulness": 0.94,
    "relevancy": 0.91,
    "contextual": 0.96
  }
}
```

---

## Part 7: Key Concepts Summary

| Concept | Means | Example |
|---------|-------|---------|
| **Embedding** | Text → 384 numbers | Query becomes [0.234, -0.567, ...] |
| **Vector DB** | Search by meaning | Find "pressing" when query says "press" |
| **Semantic** | Find by meaning | "high press" matches "aggressive defense" |
| **Cosine** | How similar (0-1) | Query vs Player: 0.72 similarity |
| **Hybrid** | Filters + semantic | position=midfielder + "creative" match |
| **Blended** | Season + career together | Query gets both form AND trend data |
| **RAG** | Retrieve + Augment + Generate | Get facts → Use them → Write answer |
| **Grounding** | Only cite from data | No hallucinations, all traceable |
| **DSPy** | Structured LLM | Schema + constraints guarantee quality |
| **DeepEval** | Validate quality | 3 metrics check brief before returning |

---

## The Complete System in 60 Seconds

```
User: "Find improving creative midfielder"
   ↓
Embed query (sentence-transformers)
   ↓
Parse intent (regex: "midfielder", "improving", "creative")
   ↓
Retrieve theory (Docling chunks: what these mean)
   ↓
Route to blended collection (career + form data)
   ↓
Search with hybrid (filters + cosine similarity)
   ↓
Blend scores (50% vector + 50% stats)
   ↓
Enrich with manager, wages, club context
   ↓
Generate brief (DSPy: structured, grounded)
   ↓
Validate quality (DeepEval: 3 metrics)
   ↓
Return: Brief + candidates + scatter plot + scores
```

---

## For Different Roles

**Product/Manager**:
- Understand: RAG retrieves real data, AI writes grounded briefs
- Key benefit: Explainable, auditable, no hallucinations

**Frontend Developer**:
- Expect: JSON response with brief, candidates, quality_scores, timeline
- Tools: React displays results + timeline of 8 phases

**Backend Engineer**:
- Architecture: 8-phase engine orchestrating modules
- Key modules: database.py, embeddings.py, generation/dspy_framework.py, evaluation/deepeval_metrics.py
- Libraries: FastAPI, Milvus, sentence-transformers, DSPy, DeepEval

**AI/ML Engineer**:
- Tuning: Adjust blending weights (50/50), thresholds, metrics
- Quality: DeepEval 3 metrics, each with configurable thresholds
- Optimization: Per-90 normalization, consistency scoring, momentum calculation

---

## Summary

**Gaffer = RAG System**
- Retrieves real data (players + theory)
- Augments with context (managers, wages)
- Generates grounded briefs (no hallucinations)

**Blended Search**
- One collection with season + career data
- Simpler queries, flexible results

**Three Tools**
- 🔴 Docling: Parse books
- 🟢 DSPy: Generate structured briefs
- 🔵 DeepEval: Validate quality

**Result**
- Explainable scouting intelligence
- Every recommendation is traceable
- Quality assured before user sees it
