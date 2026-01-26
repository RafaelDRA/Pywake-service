import os
from pydantic_settings import BaseSettings
from passlib.context import CryptContext

class Settings(BaseSettings):
    MAIN_SERVICE: str = os.getenv("MAIN_SERVICE", "http://localhost:5000/")
    ENCRYPTION_KEY_HEX: str = os.getenv("ENCRYPTION_KEY_HEX", "00e6c33aa1a2d3da5fa7766aae8b1dfc5293341f7104a92097a5e26b09640059")

settings = Settings()