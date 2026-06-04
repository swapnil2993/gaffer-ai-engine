#!/usr/bin/env python3
"""Test that progression metrics (improvement_score, stability_score, consistency_pct) are properly populated."""

import sys
from backend.app.engine import _enrich_candidate_defaults, _focused_profile, _ranking_value
from backend.app.tactical_scorer import build_metrics_checklist

def test_enrich_candidate_defaults():
    """Test that _enrich_candidate_defaults populates progression dict and preserves stats."""
    print("\n=== TEST 1: _enrich_candidate_defaults ===")

    # Simulate a blended record AFTER retrieval (stats dict already populated by retrieval code)
    blended_hit = {
        "player_name": "João Silva",
        "position": "DM",
        "squad": "Arsenal",
        "season": "2024/2025",
        "improvement_score": 0.75,  # 75% improvement
        "stability_score": 0.82,    # 82% stability
        "consistency_pct": 85.0,    # 85% consistency
        "momentum": 0.18,
        "trend": "improving",
        # Stats dict already populated from retrieval phase
        "stats": {
            "tackles": 156,
            "interceptions": 45,
            "assists": 12,
        }
    }

    enriched = _enrich_candidate_defaults(blended_hit)

    # Check progression dict
    assert "progression" in enriched, "progression dict not created"
    progression = enriched["progression"]
    assert progression.get("improvement_score") == 0.75, f"improvement_score not in progression: {progression}"
    assert progression.get("stability_score") == 0.82, f"stability_score not in progression: {progression}"
    assert progression.get("consistency_pct") == 85.0, f"consistency_pct not in progression: {progression}"
    assert progression.get("momentum") == 0.18, f"momentum not in progression: {progression}"
    assert progression.get("trend") == "improving", f"trend not in progression: {progression}"
    print(f"✓ progression dict properly populated: {progression}")

    # Check stats dict - should now include progression metrics
    assert "stats" in enriched, "stats dict not created"
    stats = enriched["stats"]
    assert stats.get("improvement_score") == 0.75, f"improvement_score not in stats: {stats}"
    assert stats.get("stability_score") == 0.82, f"stability_score not in stats: {stats}"
    assert stats.get("consistency_pct") == 85.0, f"consistency_pct not in stats: {stats}"
    assert stats.get("tackles") == 156, f"tackles not in stats: {stats}"
    assert stats.get("assists") == 12, f"assists not in stats: {stats}"
    print(f"✓ stats dict properly populated with progression metrics: improvement_score={stats.get('improvement_score')}, stability_score={stats.get('stability_score')}, consistency_pct={stats.get('consistency_pct')}")

def test_ranking_value():
    """Test that _ranking_value can handle progression metrics."""
    print("\n=== TEST 2: _ranking_value ===")

    stats = {
        "improvement_score": 0.75,
        "stability_score": 0.82,
        "consistency_pct": 85.0,
        "tackles": 156,
        "minutes": 2700,
    }

    # For progression metrics (not rate columns), should return raw value
    imp_val = _ranking_value(stats, "improvement_score")
    assert imp_val == 0.75, f"improvement_score ranking value wrong: {imp_val}"
    print(f"✓ improvement_score ranking value: {imp_val}")

    stab_val = _ranking_value(stats, "stability_score")
    assert stab_val == 0.82, f"stability_score ranking value wrong: {stab_val}"
    print(f"✓ stability_score ranking value: {stab_val}")

    cons_val = _ranking_value(stats, "consistency_pct")
    assert cons_val == 85.0, f"consistency_pct ranking value wrong: {cons_val}"
    print(f"✓ consistency_pct ranking value: {cons_val}")

def test_focused_profile():
    """Test that _focused_profile includes progression metrics in output."""
    print("\n=== TEST 3: _focused_profile ===")

    candidate = {
        "player_name": "João Silva",
        "position": "DM",
        "current_club": "Arsenal",
        "season": "2024/2025",
        "stats": {
            "tackles": 156,
            "assists": 12,
            "improvement_score": 0.75,
            "stability_score": 0.82,
            "consistency_pct": 85.0,
        },
        "improvement_score": 0.75,
        "stability_score": 0.82,
        "consistency_pct": 85.0,
    }

    # Focus on progression metrics
    profile = _focused_profile(candidate, ["improvement_score", "stability_score", "consistency_pct"])
    print(f"✓ focused_profile output: {profile}")

    assert "75%" in profile, f"improvement_score not in profile: {profile}"
    assert "82%" in profile, f"stability_score not in profile: {profile}"
    assert "85%" in profile, f"consistency_pct not in profile: {profile}"

def test_metrics_checklist():
    """Test that build_metrics_checklist includes progression metrics."""
    print("\n=== TEST 4: build_metrics_checklist ===")

    candidate = {
        "player_name": "João Silva",
        "position": "DM",
        "stats": {
            "tackles": 156,
            "interceptions": 45,
            "recoveries": 120,
            "assists": 12,
        },
        "progression": {
            "trend": "improving",
            "momentum": 0.18,
            "improvement_score": 0.75,
            "stability_score": 0.82,
            "consistency_pct": 85.0,
        },
    }

    # Use valid tactical concepts that have requirements
    checklist = build_metrics_checklist(["defensive_transition"], candidate)
    print(f"✓ metrics_checklist output:\n{checklist}")

    assert "DEFENSIVE TRANSITION" in checklist or "defensive transition" in checklist.lower(), f"tactical concept not in checklist: {checklist}"
    # Career trajectory metrics should be in the checklist
    assert "IMPROVING" in checklist or "improving" in checklist.lower(), f"trajectory not in checklist: {checklist}"
    assert "Improvement" in checklist, f"Improvement metric not in checklist: {checklist}"
    assert "Stability" in checklist, f"Stability metric not in checklist: {checklist}"
    assert "Consistency" in checklist, f"Consistency metric not in checklist: {checklist}"

if __name__ == "__main__":
    try:
        test_enrich_candidate_defaults()
        test_ranking_value()
        test_focused_profile()
        test_metrics_checklist()
        print("\n✅ ALL TESTS PASSED")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
