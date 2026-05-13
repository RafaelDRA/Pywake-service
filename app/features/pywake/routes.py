from fastapi import APIRouter

from app.features.pywake import schemas

router = APIRouter(tags=["GeoJSONs"])


def _services():
    from app.features.pywake import services as pywake_services

    return pywake_services


@router.post("/wind-farm/{geojson_name}")
async def wind_farm(geojson_name: str,polygon: schemas.GeoJSONQuery):
  simulation_result = await _services().generate_geojson(geojson_name, polygon)
  return simulation_result