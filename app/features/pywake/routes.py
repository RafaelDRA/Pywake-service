from fastapi import APIRouter

from app.features.pywake import schemas
from app.features.pywake.services import generate_geojson, run_simulation

router = APIRouter(prefix="/pywake", tags=["GeoJSONs"])

@router.post("/wind-farm/{geojson_name}")
async def wind_farm(geojson_name: str,polygon: schemas.GeoJSONQuery):
  simulation_result = generate_geojson(polygon)
  return simulation_result