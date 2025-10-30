import os
from pydantic_settings import BaseSettings
from passlib.context import CryptContext

class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv('DATABASE_URL', "postgresql+asyncpg://postgres:postgres@localhost:5432/ute")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "99a3e4b11b348bdd559badb89674ac7eb0b20d5c0bbfd1d89845ff1bcc397fce")
    ALGORITHM: str = "HS256"
    REFRESH_TOKEN_EXPIRE_DAYS: int = os.getenv('REFRESH_TOKEN_EXPIRE_DAYS', 7)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', 30)
    bcrypt_context: CryptContext = CryptContext(schemes=["bcrypt"], deprecated="auto")

settings = Settings()