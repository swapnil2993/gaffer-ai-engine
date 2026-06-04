from backend.app.schemas import PlayerStats, TacticalTheoryPayload


def test_player_stats_alias_mapping():
    """Verify that FBref column aliases map correctly to PlayerStats fields."""
    stats_data = {
        "Gls": 10.0,
        "Ast": 5.0,
        "xG": 8.5,
        "xAG": 4.2,
        "PrgC": 25.0,
        "PrgP": 30.0,
        "PrgR": 15.0,
        "Min": 1800.0,
    }
    stats = PlayerStats(**stats_data)
    assert stats.goals == 10.0
    assert stats.assists == 5.0
    assert stats.expected_goals == 8.5
    assert stats.progressive_passes == 30.0
    assert stats.minutes_played == 1800.0


def test_tactical_theory_payload_creation():
    """Verify TacticalTheoryPayload validation and defaults."""
    content = "The W-M formation revolutionized tactical flexibility in the 1930s."
    theory = TacticalTheoryPayload(
        era="Pre-War",
        formation="W-M (3-2-2-3)",
        key_figures=["Herbert Chapman"],
        content=content,
    )
    assert theory.title == "The Inverted Pyramid"
    assert theory.era == "Pre-War"
    assert "Herbert Chapman" in theory.key_figures
    assert theory.content == content
