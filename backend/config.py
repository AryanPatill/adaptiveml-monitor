from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DATABASE_URL: str = "sqlite:///./adaptiveml.db"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:5173"]
    MODEL_STORE_PATH: Path = Path("./model_store")
    CMAPSS_DATA_PATH: Path = Path("./data/cmapss")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()