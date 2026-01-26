from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.features.pywake.routes import router as pywake_router
from fastapi.openapi.utils import get_openapi
from app.security.encryption_middleware import EncryptionMiddleware

app = FastAPI()

app.add_middleware(EncryptionMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Pywake API",
        version="1.0.0",
        description="API para simulações de parques eólicos usando Pywake",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    openapi_schema["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

app.include_router(pywake_router)
