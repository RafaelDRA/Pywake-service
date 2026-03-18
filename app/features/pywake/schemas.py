from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class SimParams(BaseModel):
    spacing_downwind_d: float = Field(default=15.0, ge=5.0, le=30.0)
    spacing_crosswind_d: float = Field(default=3.0, ge=1.5, le=10.0)


class GeoJSONQuery(BaseModel):
    properties: Dict[str, Any]
    geometry: Dict[str, Any]
    id: str
    sim_params: Optional[SimParams] = None