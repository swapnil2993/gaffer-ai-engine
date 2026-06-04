from typing import Optional, List

from pydantic import BaseModel, Field


class PlayerStats(BaseModel):
    """Refined based on fbref_PL_2024-25.csv"""

    goals: float = Field(default=0.0, alias="Gls")
    assists: float = Field(default=0.0, alias="Ast")
    expected_goals: float = Field(default=0.0, alias="xG")
    expected_assists: float = Field(default=0.0, alias="xAG")
    progressive_carries: float = Field(default=0.0, alias="PrgC")
    progressive_passes: float = Field(default=0.0, alias="PrgP")
    progressive_receptions: float = Field(default=0.0, alias="PrgR")
    minutes_played: float = Field(default=0.0, alias="Min")


class PlayerWages(BaseModel):
    """Refined based on player_wages.csv"""

    weekly_wages: str = Field(..., alias="Weekly Wages")
    annual_wages: str = Field(..., alias="Annual Wages")



class TacticalTheoryPayload(BaseModel):
    """Lightweight schema for Inverted Pyramid book chunks."""

    title: str = Field(default="The Inverted Pyramid")
    chapter: Optional[str] = None
    content: str = Field(..., description="The extracted tactical theory text.")


class EvalRequest(BaseModel):
    """Request schema for scouting brief evaluation."""

    query: str
    scouting_brief: str
    context: List[str] = []
