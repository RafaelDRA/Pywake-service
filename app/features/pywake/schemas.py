from typing import Any, Dict
from pydantic import BaseModel


class GeoJSONQuery(BaseModel):
    properties: Dict[str, Any]
    geometry: Dict[str, Any]