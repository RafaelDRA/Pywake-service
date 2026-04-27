import os
from pydantic import field_validator
from pydantic_settings import BaseSettings
from passlib.context import CryptContext


def _normalize_service_url(value: str) -> str:
    return f"{value.rstrip('/')}/"


class Settings(BaseSettings):
    MAIN_SERVICE: str = os.getenv("MAIN_SERVICE", "http://localhost:5000")
    ENCRYPTION_KEY_HEX: str = os.getenv("ENCRYPTION_KEY_HEX") or os.getenv("ENCRYPTION_KEY") or "00e6c33aa1a2d3da5fa7766aae8b1dfc5293341f7104a92097a5e26b09640059"

    @field_validator("MAIN_SERVICE", mode="before")
    @classmethod
    def normalize_main_service(cls, value: str) -> str:
        return _normalize_service_url(value or "http://localhost:5000")

settings = Settings()