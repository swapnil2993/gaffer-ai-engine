# Realistic Scouting Queries (15 Examples)

These queries reflect **actual scouting use cases** — recruitment teams looking for specific player types to fill roles in their squads.

---

## Group 1: Tactical Role Requirements (Season-Specific)

### Query 1: Pressing Midfielder
```
Ball-winning midfielder for our high press system in 2024/25. Need someone with 
high tackle success rate and recovery runs in the final third. What we're running now.
```
**Scouting context:** Need immediate replacement for injured midfielder  
**Expected:** Routes to `STATS_COLLECTION`, filters for 2024/25 stats  
**DeepEval:** Should cite tackles, interceptions, recoveries with thresholds

---

### Query 2: Creative Playmaker
```
Looking for a 10 or inside forward who can unlock defenses with key passes and 
through balls. For Liverpool's current system. 2024/25 form only.
```
**Scouting context:** Upgrade to attacking midfield depth  
**Expected:** Routes to `STATS_COLLECTION`, season filter applied  
**DeepEval:** Only mentions assists, key passes, not defensive stats

---

### Query 3: Defensive Full-Back
```
Right-back for Chelsea's back four. Need someone good at 1v1 defending with 
high tackle success. Current season performance — must be sharp right now.
```
**Scouting context:** Squad rotation/competition for RB position  
**Expected:** Routes to `STATS_COLLECTION`, position filter: RB  
**DeepEval:** Cites tackles, tackle success rate specific to current form

---

### Query 4: Ball Progression Center-Back
```
We're rebuilding our defense to play out from the back. Center-back for Man City's 
system who can pass progressively and carry the ball forward. This season form.
```
**Scouting context:** System-specific recruitment (build-up play)  
**Expected:** Routes to `STATS_COLLECTION`, filters for progressive passes  
**DeepEval:** Only cites passing metrics, not defensive metrics

---

### Query 5: False 9 Striker
```
Wanted: Forward who drops deep to receive in the 9 space, can link play, not 
necessarily the highest goal scorer. For our 2024/25 squad planning.
```
**Scouting context:** Specific tactical role rather than pure number 9  
**Expected:** Routes to `STATS_COLLECTION`, creative metrics > goal metrics  
**DeepEval:** Cites assists, chances created, not just goals

---

## Group 2: Youth/Emerging Talent Assessment (Career-Focused)

### Query 6: Breakout Young Midfielder
```
Which young midfielders have shown consistent improvement over the last 2-3 seasons? 
We want emerging talent with genuine upward trajectory, not one-season wonders.
```
**Scouting context:** Youth development/academy graduation tracking  
**Expected:** Routes to `CAREER_COLLECTION`, shows momentum + consistency  
**DeepEval:** Cites momentum percentage and trend, not single-season peaks

---

### Query 7: Improving Defender
```
Center-backs who are improving year-on-year. We need defenders getting better, 
not in decline. 3-year progression matters — shows they're learning.
```
**Scouting context:** Long-term defensive recruitment strategy  
**Expected:** Routes to `CAREER_COLLECTION`, momentum filter  
**Impact:** HIGH confidence for improving players, LOW for declining

---

### Query 8: Consistent Performer
```
Midfielder we can rely on. Need someone very stable across multiple seasons — 
not hot/cold. Consistency over volatility.
```
**Scouting context:** Reduce risk — avoid flaky players  
**Expected:** Routes to `CAREER_COLLECTION`, filters by consistency metric  
**DeepEval:** Only mentions consistency, variance, not individual season peaks

---

## Group 3: Market Opportunity / Transfer Window (Mixed Signals)

### Query 9: Rising Star for Resale
```
Young attacker improving over time with 5+ goals this season. Good investment — 
growing player with current output. Resale value in 2-3 years.
```
**Scouting context:** Financial planning / academy profit  
**Expected:** Hybrid — checks both career momentum AND current season goals  
**Impact:** Shows both "+15% improvement" AND "8 goals this season"

---

### Query 10: Bargain Buy
```
Defender who was solid in 2023/24 but seems to have dipped recently. If he's 
been consistent overall, could be undervalued in 2024/25. Bounce-back candidate.
```
**Scouting context:** Post-injury recovery / form dip opportunity  
**Expected:** Compares career average to current season (below average = opportunity)  
**DeepEval:** Shows season-vs-career comparison explicitly

---

### Query 11: Veteran Leader for Depth
```
Midfielder over 30 with very stable, consistent performance over 3+ years. 
Not exciting but reliable. Squad depth, changing room presence.
```
**Scouting context:** Experience signing for squad balance  
**Expected:** Routes to `CAREER_COLLECTION`, shows "very consistent" badge  
**Impact:** Stability valued over growth/momentum

---

## Group 4: Position-Specific Replacement (Tactical Needs)

### Query 12: Defensive Transition Partner
```
Our new 4-2-3-1 formation needs a defensive-minded midfielder who's strong in 
recovery and tackle. Partner for the press. Give me options for 2024/25.
```
**Scouting context:** Formation change requires specific type  
**Expected:** Routes to `STATS_COLLECTION`, filters for tackles/interceptions  
**DeepEval:** Only mentions defensive stats, not creativity

---

### Query 13: Left-Back for Overlap Attacks
```
Left-back who gets forward well — need high assists/key passes and progressive 
carries to support our wide attacking system. This season form.
```
**Scouting context:** Attacking full-back in transition-heavy team  
**Expected:** Routes to `STATS_COLLECTION`, filters for attacking metrics on fullback  
**DeepEval:** Cites assists and progressive carries, not just defensive work

---

### Query 14: Pressing Forward Upgrade
```
Striker for our high press. Need someone who works hard off the ball — tackles, 
pressing success in the final third. Not just about goals. Current season.
```
**Scouting context:** Modern pressing system demands (not traditional 9)  
**Expected:** Routes to `STATS_COLLECTION`, filters for defensive activity  
**Impact:** Shows tackles/pressing before goals in importance

---

### Query 15: Rebuild Center-Back Pairing
```
Two center-backs to rebuild the defense. Both must be very consistent over time — 
one experienced (stable career), one improving (upward trajectory). 3-year view.
```
**Scouting context:** Major defensive overhaul  
**Expected:** Routes to `CAREER_COLLECTION`, shows two different types  
**Impact:** Shows contrasting profiles (stable vs. improving)

---

## Testing Script for Realistic Scenarios

Save as `scout_test.sh`:

```bash
#!/bin/bash

echo "Starting scouting query tests..."
./start.sh &
sleep 5

declare -a queries=(
  "Ball-winning midfielder for our high press system in 2024/25. Need someone with high tackle success rate and recovery runs in the final third. What we're running now."
  "Looking for a 10 or inside forward who can unlock defenses with key passes and through balls. For Liverpool's current system. 2024/25 form only."
  "Right-back for Chelsea's back four. Need someone good at 1v1 defending with high tackle success. Current season performance — must be sharp right now."
  "We're rebuilding our defense to play out from the back. Center-back for Man City's system who can pass progressively and carry the ball forward. This season form."
  "Wanted: Forward who drops deep to receive in the 9 space, can link play, not necessarily the highest goal scorer. For our 2024/25 squad planning."
  "Which young midfielders have shown consistent improvement over the last 2-3 seasons? We want emerging talent with genuine upward trajectory, not one-season wonders."
  "Center-backs who are improving year-on-year. We need defenders getting better, not in decline. 3-year progression matters — shows they're learning."
  "Midfielder we can rely on. Need someone very stable across multiple seasons — not hot/cold. Consistency over volatility."
  "Young attacker improving over time with 5+ goals this season. Good investment — growing player with current output. Resale value in 2-3 years."
  "Defender who was solid in 2023/24 but seems to have dipped recently. If he's been consistent overall, could be undervalued in 2024/25. Bounce-back candidate."
  "Midfielder over 30 with very stable, consistent performance over 3+ years. Not exciting but reliable. Squad depth, changing room presence."
  "Our new 4-2-3-1 formation needs a defensive-minded midfielder who's strong in recovery and tackle. Partner for the press. Give me options for 2024/25."
  "Left-back who gets forward well — need high assists/key passes and progressive carries to support our wide attacking system. This season form."
  "Striker for our high press. Need someone who works hard off the ball — tackles, pressing success in the final third. Not just about goals. Current season."
  "Two center-backs to rebuild the defense. Both must be very consistent over time — one experienced (stable career), one improving (upward trajectory). 3-year view."
)

echo "Running 15 realistic scouting queries..."
echo ""

for i in "${!queries[@]}"; do
  query_num=$((i+1))
  echo "=========================================="
  echo "Query $query_num of ${#queries[@]}"
  echo "=========================================="
  echo "Prompt: ${queries[$i]}"
  echo ""
  
  response=$(curl -s -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"${queries[$i]}\"}")
  
  # Extract key fields
  collection=$(echo $response | jq -r '.phases[3].collection // "unknown"' 2>/dev/null)
  query_type=$(echo $response | jq -r '.impact_analysis.query_type // "unknown"' 2>/dev/null)
  confidence=$(echo $response | jq -r '.impact_analysis.primary_driver.confidence_level // "unknown"' 2>/dev/null)
  player=$(echo $response | jq -r '.candidates[0].player_name // "unknown"' 2>/dev/null)
  
  echo "Results:"
  echo "  Collection: $collection"
  echo "  Query Type: $query_type"
  echo "  Top Pick: $player"
  echo "  Confidence: $confidence"
  echo ""
  
  # Full response (uncomment for debugging)
  # echo "Full response:"
  # echo $response | jq '.' 
  
  sleep 1
done

echo "=========================================="
echo "Testing complete!"
echo "=========================================="
```

Run with:
```bash
chmod +x scout_test.sh
./scout_test.sh 2>&1 | tee scout_results.log
```

---

## What to Verify for Each Query

### Query 1: High Press Midfielder
- ✅ Routes to: `STATS_COLLECTION` (2024/25 specific)
- ✅ Mentions: tackles, tackle success, recoveries
- ✅ Does NOT mention: assists, creativity
- ✅ Confidence: Based on defensive metrics quality

### Query 6: Emerging Talent
- ✅ Routes to: `CAREER_COLLECTION` (3-year improvement)
- ✅ Shows: Momentum %, upward trend
- ✅ Mentions: "Consistent improvement" or "improving YoY"
- ✅ Confidence: HIGH if momentum > +5%

### Query 9: Rising Star
- ✅ Routes to: Either (mixed signals)
- ✅ Shows: Both momentum + this-season goals
- ✅ Example brief: "8 goals this season (good form), +12% YoY (improving trajectory)"
- ✅ Impact: Highlights both current output AND growth rate

### Query 15: Rebuild Pairing
- ✅ Routes to: `CAREER_COLLECTION` (3-year context)
- ✅ Primary: Improving defender (momentum positive)
- ✅ Runner-up: Stable defender (consistency high, low variance)
- ✅ Shows: Contrasting profiles for different roles

---

## Expected DeepEval Performance

After improvements, these queries should show:

| Query Type | Faithfulness | Answer Relevancy |
|-----------|--------------|------------------|
| Season-specific (1-5, 12-14) | 92-95% | 88-92% |
| Career-focused (6-8, 11, 15) | 90-94% | 86-90% |
| Mixed/Complex (9-10) | 88-92% | 84-88% |

**Overall average target: 90-93%** (up from baseline 84%)

---




