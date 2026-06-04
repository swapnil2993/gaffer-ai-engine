# DeepEval Contextual Relevancy - Comprehensive Fixes

## ROOT CAUSES

### 1. **Data Pipeline - Missing Indexed Metrics**
- **Problem**: New fields (avg_tackles_won, avg_interceptions, improvement_score, stability_score, consistency_pct) added to code but NOT YET INDEXED in Milvus
- **Impact**: Retrieved players lack defensive stats, brief can't cite them, DeepEval gets "statistics unavailable"
- **Fix**: Run `python -m backend.scripts.index_data` to reindex with new fields

### 2. **Enrichment - Incomplete Stats Dict**
- **Problem**: stats_dict in _build_candidate only captures specific fields; career aggregates have avg_* prefix that doesn't match
- **Impact**: avg_tackles_won from career not mapped to stats dict, focused_profile can't find them
- **Fix**: Add fallback mappings for avg_* fields in _build_candidate

### 3. **Context Profile - Missing Query-Relevant Metrics**
- **Problem**: focused_profile only shows metrics in rank_by columns; if ranking by improvement_score but it's missing from stats dict, it won't be shown
- **Impact**: Context lacks the metrics the query asked for, DeepEval sees irrelevant context
- **Fix**: Add explicit mapping of career aggregate fields to stats dict

### 4. **Checklist Filtering - Removing Important Data**
- **Problem**: _filter_checklist_by_concept might be too aggressive, removing metrics relevant to the query
- **Impact**: Brief generation doesn't cite key stats, context not fully used
- **Fix**: Review and relax checklist filtering logic

### 5. **Position Fallback - Wrong Players Returned**
- **Problem**: Fallback ladder could still return wrong positions for specific position queries
- **Impact**: Centre-back query gets fullbacks, context irrelevant to query
- **Fix**: (Already fixed with granular fallback + guarded fully-unfiltered)

### 6. **Candidate Selection - Top 5 Not Most Relevant**
- **Problem**: Blended score might not properly weight query-relevant metrics
- **Impact**: Retrieved candidates aren't best fit, context irrelevant to query
- **Fix**: Ensure improvement_score/stability_score are in blended ranking

### 7. **Brief Generation - Not Citing Context**
- **Problem**: ScoutingReportSignature tells LLM to cite metrics, but LLM might miss averaging or paraphrasing
- **Impact**: DeepEval sees brief doesn't properly cite the context metrics
- **Fix**: Make metrics checklist format more explicit and structured

### 8. **Career Aggregates - Missing Fields in Output**
- **Problem**: CAREER_OUTPUT_FIELDS might not include all metrics
- **Impact**: Career queries lack defensive stats even after indexing
- **Fix**: Verify all required metrics in CAREER_OUTPUT_FIELDS

## FIXES (Priority Order)

### PRIORITY 1: Enable Data
```bash
# Reindex with all new fields captured
python -m backend.scripts.index_data
```

### PRIORITY 2: Fix Enrichment Mappings (code changes below)
- Map avg_tackles_won → stats["tackles_won"] if missing
- Map avg_interceptions → stats["interceptions"] if missing  
- Map improvement_score, stability_score, consistency_pct if missing

### PRIORITY 3: Ensure Checklist Format
- Make checklist show metrics with exact values: "Tackles 156 ✓"
- Don't filter checklist too aggressively

### PRIORITY 4: Verify Rankings
- Ensure improvement_score/stability_score are blended into ranking when requested
- Check that top 5 includes query-relevant players

### PRIORITY 5: Context Passing
- Ensure all candidate fields are available in brief generation
- Make stats_summary show ALL relevant metrics, not just selected ones

## FILES TO MODIFY
1. backend/app/engine.py - _build_candidate, _enrich_candidate_defaults
2. backend/app/database.py - verify CAREER_OUTPUT_FIELDS
3. backend/app/scripts/index_data.py - verify metadata extraction
4. backend/app/engine.py - _filter_checklist_by_concept (review logic)
