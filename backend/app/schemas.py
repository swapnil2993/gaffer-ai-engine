from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any
import re

class ScoutingPayload(BaseModel):
    """
    Data model for football scouting reports, ensuring strict typing and validation.
    """
    player_name: str = Field(
        ..., 
        description="The full name of the scouted football player."
    )
    position: str = Field(
        ..., 
        description="The tactical position of the player (e.g., 'CF', 'LW', 'DM')."
    )
    season: str = Field(
        ..., 
        description="The scouting season in 'YYYY/YYYY' format (e.g., '2024/2025').",
        pattern=r"^\d{4}/\d{4}$"
    )
    raw_text: str = Field(
        ..., 
        description="The layout-aware markdown text extracted from the scouting PDF."
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict, 
        description="A collection of quantitative performance metrics extracted from the report."
    )

    @field_validator("season")
    @classmethod
    def validate_season_format(cls, v: str) -> str:
        if not re.match(r"^\d{4}/\d{4}$", v):
            raise ValueError("Season must be in 'YYYY/YYYY' format.")
        return v
