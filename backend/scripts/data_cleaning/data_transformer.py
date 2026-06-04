"""Data transformation utilities for converting stats to semantic prose."""

from backend.scripts.data_cleaning.common import num, per90, position_phrase, pct


def describe_outfield_player(name: str, squad: str, position: str, season: str, stats: dict) -> str:
    """Build natural-language scouting description from stats.

    Embedding prose (rather than raw stats) improves semantic matching quality
    since user queries ("prolific goal threat") read like natural language.
    """
    g = stats.get("goals", 0.0)
    a = stats.get("assists", 0.0)
    mins = stats.get("minutes", 0.0)
    apps = stats.get("appearances", 0.0)
    xg = stats.get("xg")
    xag = stats.get("xag")
    prgc = stats.get("prgc")
    prgp = stats.get("prgp")

    parts = [f"{name} is a {position_phrase(position)} for {squad} in the {season} season."]

    if mins:
        played = f"Played {int(mins)} minutes"
        played += f" across {int(apps)} appearances." if apps else "."
        parts.append(played)
    elif apps:
        parts.append(f"Made {int(apps)} appearances.")

    g90 = per90(g, mins)
    if g >= 15 or g90 >= 0.5:
        parts.append(f"A prolific goalscorer with {int(g)} goals" + (f" ({g90:.2f} per 90)." if mins else "."))
    elif g >= 5:
        parts.append(f"A regular goal threat with {int(g)} goals.")
    elif g > 0:
        parts.append(f"Chips in occasionally with {int(g)} goals.")

    if a >= 8:
        parts.append(f"A high-volume creator providing {int(a)} assists.")
    elif a >= 3:
        parts.append(f"Contributes creatively with {int(a)} assists.")
    elif a > 0:
        parts.append(f"{int(a)} assists.")

    if xg is not None or xag is not None:
        parts.append(f"Underlying numbers: {num(xg):.1f} xG and {num(xag):.1f} xAG.")

    prog = []
    if prgc:
        prog.append(f"{int(prgc)} progressive carries")
    if prgp:
        prog.append(f"{int(prgp)} progressive passes")
    if prog:
        parts.append("Ball progression: " + ", ".join(prog) + ".")

    tk = stats.get("tackles", 0.0)
    interc = stats.get("interceptions", 0.0)
    rec = stats.get("recoveries", 0.0)
    tklw = stats.get("tackles_won", 0.0)
    if tk or interc or rec:
        bits = [f"{int(tk)} tackles"]
        if tklw:
            bits.append(f"{int(tklw)} won")
        bits.append(f"{int(interc)} interceptions")
        if rec:
            bits.append(f"{int(rec)} recoveries")
        parts.append("Defensive actions: " + ", ".join(bits) + " (wins the ball back).")

    return " ".join(parts)


def describe_2023_24_player(name: str, squad: str, position: str, season: str, stats: dict) -> str:
    """Prose for 2023/24 profile dataset (counting + defensive stats)."""
    parts = [f"{name} is a {position_phrase(position)} for {squad} in the {season} season."]

    apps = stats.get("appearances", 0.0)
    if apps:
        parts.append(f"Made {int(apps)} appearances.")

    g = stats.get("goals", 0.0)
    a = stats.get("assists", 0.0)
    if g >= 10:
        parts.append(f"A strong goal threat with {int(g)} goals.")
    elif g > 0:
        parts.append(f"{int(g)} goals.")
    if a >= 5:
        parts.append(f"A creative outlet with {int(a)} assists.")
    elif a > 0:
        parts.append(f"{int(a)} assists.")

    tk = stats.get("tackles", 0.0)
    interc = stats.get("interceptions", 0.0)
    rec = stats.get("recoveries", 0.0)
    if tk or interc or rec:
        bits = [f"{int(tk)} tackles", f"{int(interc)} interceptions"]
        if rec:
            bits.append(f"{int(rec)} recoveries")
        parts.append("Defensive profile: " + ", ".join(bits) + ".")

    return " ".join(parts)
