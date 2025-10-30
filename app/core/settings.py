import os
from pydantic_settings import BaseSettings
from passlib.context import CryptContext

class Settings(BaseSettings):
    MAIN_SERVICE: str = os.getenv("MAIN_SERVICE", "http://localhost:5000/")

settings = Settings()