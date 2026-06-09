from fastapi import APIRouter, HTTPException, Request

from app.core.audit import safe_emit_request_audit_event
from app.features.pywake import schemas

router = APIRouter(tags=["GeoJSONs"])


def _services():
    from app.features.pywake import services as pywake_services

    return pywake_services


@router.post("/wind-farm/{geojson_name}")
async def wind_farm(request: Request, geojson_name: str, polygon: schemas.GeoJSONQuery):
  payload_meta = {
    "geometry_type": polygon.geometry.get("type") if isinstance(polygon.geometry, dict) else None,
    "has_sim_params": polygon.sim_params is not None,
  }
  await safe_emit_request_audit_event(
    request,
    event_type="wind_farm_simulation_started",
    action="compute",
    status="started",
    resource_type="wind_farm_simulation",
    resource_id=geojson_name,
    resource_label=geojson_name,
    payload_meta=payload_meta,
  )
  try:
    simulation_result = await _services().generate_geojson(geojson_name, polygon)
    await safe_emit_request_audit_event(
      request,
      event_type="wind_farm_simulation_completed",
      action="compute",
      status="success",
      resource_type="wind_farm_simulation",
      resource_id=geojson_name,
      resource_label=geojson_name,
      payload_meta=payload_meta,
    )
    return simulation_result
  except Exception as exc:
    await safe_emit_request_audit_event(
      request,
      event_type="wind_farm_simulation_failed",
      action="compute",
      status="failed",
      severity="warning",
      resource_type="wind_farm_simulation",
      resource_id=geojson_name,
      resource_label=geojson_name,
      payload_meta=payload_meta,
      error_code="PYWAKE_SIMULATION_ERROR",
      error_message_redacted=str(exc),
    )
    raise HTTPException(status_code=500, detail=str(exc)) from exc