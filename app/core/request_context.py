from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.settings import settings


CORRELATION_ID_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"
INTERNAL_SECRET_HEADER = "X-Internal-Service-Secret"
SERVICE_NAME_HEADER = "X-Service-Name"
FORWARDED_ACTOR_TYPE_HEADER = "X-Forwarded-Actor-Type"
FORWARDED_ACTOR_USER_ID_HEADER = "X-Forwarded-Actor-User-Id"
FORWARDED_ACTOR_EMAIL_HEADER = "X-Forwarded-Actor-Email"
FORWARDED_ACTOR_ROLE_HEADER = "X-Forwarded-Actor-Role"

_request_context: ContextVar["RequestContext | None"] = ContextVar("request_context", default=None)


def _generate_identifier() -> str:
    uuid7 = getattr(uuid, "uuid7", None)
    if uuid7 is not None:
        return str(uuid7())
    return str(uuid.uuid4())


@dataclass
class RequestContext:
    request_id: str
    correlation_id: str
    service_name: str
    started_at: datetime
    started_perf: float
    caller_service_name: str | None = None
    actor_type: str | None = None
    actor_user_id: str | None = None
    actor_email: str | None = None
    actor_role: str | None = None
    trusted_internal: bool = False

    @property
    def duration_ms(self) -> int:
        return int((perf_counter() - self.started_perf) * 1000)


def get_request_context() -> RequestContext | None:
    return _request_context.get()


def build_internal_request_headers() -> dict[str, str]:
    context = get_request_context()
    headers: dict[str, str] = {
        SERVICE_NAME_HEADER: settings.SERVICE_NAME,
    }

    if context is not None:
        headers[CORRELATION_ID_HEADER] = context.correlation_id

    if settings.AUDIT_INTERNAL_SECRET:
        headers[INTERNAL_SECRET_HEADER] = settings.AUDIT_INTERNAL_SECRET
        if context is not None:
            if context.actor_type:
                headers[FORWARDED_ACTOR_TYPE_HEADER] = context.actor_type
            if context.actor_user_id:
                headers[FORWARDED_ACTOR_USER_ID_HEADER] = context.actor_user_id
            if context.actor_email:
                headers[FORWARDED_ACTOR_EMAIL_HEADER] = context.actor_email
            if context.actor_role:
                headers[FORWARDED_ACTOR_ROLE_HEADER] = context.actor_role

    return headers


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = _generate_identifier()
        correlation_id = request.headers.get(CORRELATION_ID_HEADER) or request_id
        caller_secret = request.headers.get(INTERNAL_SECRET_HEADER)
        trusted_internal = bool(
            settings.AUDIT_INTERNAL_SECRET
            and caller_secret
            and caller_secret == settings.AUDIT_INTERNAL_SECRET
        )

        context = RequestContext(
            request_id=request_id,
            correlation_id=correlation_id,
            service_name=settings.SERVICE_NAME,
            started_at=datetime.now(timezone.utc),
            started_perf=perf_counter(),
            caller_service_name=request.headers.get(SERVICE_NAME_HEADER) if trusted_internal else None,
            actor_type=request.headers.get(FORWARDED_ACTOR_TYPE_HEADER) if trusted_internal else None,
            actor_user_id=request.headers.get(FORWARDED_ACTOR_USER_ID_HEADER) if trusted_internal else None,
            actor_email=request.headers.get(FORWARDED_ACTOR_EMAIL_HEADER) if trusted_internal else None,
            actor_role=request.headers.get(FORWARDED_ACTOR_ROLE_HEADER) if trusted_internal else None,
            trusted_internal=trusted_internal,
        )

        request.state.request_context = context
        request.state.request_id = request_id
        request.state.correlation_id = correlation_id

        token = _request_context.set(context)
        try:
            response = await call_next(request)
        finally:
            _request_context.reset(token)

        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        return response