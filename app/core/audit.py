from typing import Any

import httpx
from fastapi import Request

from app.core.request_context import build_internal_request_headers
from app.core.settings import settings


async def emit_request_audit_event(
    request: Request,
    *,
    event_type: str,
    action: str,
    status: str,
    severity: str = "info",
    resource_type: str,
    resource_id: str,
    resource_label: str,
    payload_meta: dict[str, Any] | None = None,
    error_code: str | None = None,
    error_message_redacted: str | None = None,
) -> None:
    if not settings.MAIN_SERVICE or not settings.AUDIT_INTERNAL_SECRET:
        return

    request_context = getattr(request.state, "request_context", None)
    payload = {
        "request_id": getattr(request.state, "request_id", None),
        "correlation_id": getattr(request.state, "correlation_id", None),
        "service_name": settings.SERVICE_NAME,
        "environment": settings.APP_ENVIRONMENT,
        "actor_type": request_context.actor_type if request_context else None,
        "actor_user_id": request_context.actor_user_id if request_context else None,
        "actor_email": request_context.actor_email if request_context else None,
        "actor_role": request_context.actor_role if request_context else None,
        "source_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "request_method": request.method,
        "request_path": request.url.path,
        "event_category": "computation",
        "event_type": event_type,
        "action": action,
        "status": status,
        "severity": severity,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "resource_label": resource_label,
        "payload_meta": payload_meta or {},
        "error_code": error_code,
        "error_message_redacted": error_message_redacted,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            f"{settings.MAIN_SERVICE}audit/internal/events",
            json=payload,
            headers=build_internal_request_headers(),
        )
        response.raise_for_status()


async def safe_emit_request_audit_event(*args, **kwargs) -> None:
    try:
        await emit_request_audit_event(*args, **kwargs)
    except Exception as exc:
        print(f"[WARN] Failed to emit pywake audit event: {exc}")
