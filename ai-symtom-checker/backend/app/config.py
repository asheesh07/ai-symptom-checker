from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Configuration
    app_name: str = "AI Symptom Checker"
    version: str = "1.0.0"
    app_debug: bool = False  # Changed from debug to app_debug to avoid conflicts
    environment: str = os.getenv("ENVIRONMENT", "PRODUCTION")
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-3.5-turbo"
    openai_temperature: float = 0.2
    openai_max_tokens: int = 1000
    
    # Admin Configuration
    admin_api_key: str = os.getenv("ADMIN_API_KEY", "admin-secret-key-change-in-production")
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    
    # Caching
    redis_url: Optional[str] = os.getenv("REDIS_URL", None)
    cache_ttl: int = 3600  # 1 hour
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security
    cors_origins: list = ["*"]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

settings = Settings() 