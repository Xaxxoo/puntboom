"""Application configuration via pydantic-settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    database_url: str = "sqlite+aiosqlite:///./puntboom.db"

    # API Keys
    football_data_api_key: str = ""
    balldontlie_api_key: str = ""
    odds_api_key: str = ""
    rapidapi_key: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # ML
    model_dir: Path = Path("./models")
    retrain_interval_hours: int = 24

    # Bankroll
    default_bankroll: float = 1000.0
    kelly_fraction: float = 0.25
    min_value_threshold: float = 0.05


settings = Settings()
