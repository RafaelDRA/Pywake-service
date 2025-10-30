from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, List
from jose import jwt, JWTError
from fastapi import HTTPException, Request, status
from app.core.settings import settings

# Dicionário central de permissões
ROUTE_PERMISSIONS: dict = {
    "/docs": {"POST": [], "GET":[]},
    "/openapi.json": {"POST": [], "GET":[]},
    "/geojsons/": {"POST": ["ADMIN", "EDITOR"]},
    "/geojsons/names": {"GET": []},  # pública
    "/geojsons/query/stats/": {"GET": [], "POST": ["ADMIN", "EDITOR"]},
    "/geojsons/layer-group": {"POST": ["ADMIN", "EDITOR"], "GET": [], "DELETE":["ADMIN"]},
    "/geojsons/layer": {"DELETE":["ADMIN"]},
    "/geojsons/create-layer": {"POST": ["ADMIN", "EDITOR"]},
    "/geojsons/side-bar": {"GET": []},  # pública
    "/geojsons/style/": {"GET": [], "POST": ["ADMIN", "EDITOR"]},
    "/geojsons/popup/": {"GET": [], "POST": ["ADMIN", "EDITOR"]},
}


def get_route_permissions(path: str, method: str) -> List[str] | None:
    if not path.endswith('/'):
        path += '/'

    for route, methods in sorted(ROUTE_PERMISSIONS.items(), key=lambda item: len(item[0]), reverse=True):
        if path.startswith(route):
            return methods.get(method)
        
    return None

def decode_token(token: str, scope: str = "access_token"):
		try:
			payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
			if payload.get("scope") != scope and scope == "refresh_token":
				raise JWTError("Token inválido para refresh")
			return payload
		except JWTError:    
			raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido ou expirado"
        )

def authorize_request(request: Request, token: str | None):
    route_roles = get_route_permissions(request.url.path, request.method)

    # if route_roles is None:
    #     raise Exception("Acesso não permitido: rota não registrada.")

    if route_roles == [] or route_roles is None:
        return {"public": True}

    if not token:
        raise Exception("Token de autenticação ausente.")

    payload = decode_token(token)
    user_role = payload.get("user_role")

    if not user_role:
        raise Exception("Papel (role) não encontrado no token.")

    if user_role not in route_roles:
        raise Exception("Permissão negada: acesso não autorizado para este papel.")

    return payload


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        auth_header = request.headers.get("Authorization")
        token = None

        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")

        try:
            user_payload = authorize_request(request, token)
            if not user_payload.get("public"):
                request.state.user = user_payload
        except Exception as e:
            return JSONResponse(status_code=401, content={"detail": str(e)})

        return await call_next(request)