from fastapi import APIRouter

from app.features.pywake import schemas
from app.features.pywake.services import all_data_area, generate_geojson, run_simulation

router = APIRouter(prefix="/pywake", tags=["GeoJSONs"])

@router.post("/wind-farm/{geojson_name}")
async def wind_farm(geojson_name: str,polygon: schemas.GeoJSONQuery):
  simulation_result = await generate_geojson(geojson_name, polygon)
  return simulation_result

@router.post("/wind-farm/dash-data/{geojson_name}")
async def dashdata(geojson_name: str,polygon: schemas.GeoJSONQuery):
  dados_aep_gwh = await all_data_area(geojson_name, polygon)
  return dados_aep_gwh