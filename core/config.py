from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import json
import os

class Settings(BaseSettings):
    """Application settings"""
    # Server settings
    PORT: int = 8000
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]  # For development, restrict in production
    
    # Rate limiting settings
    RATE_LIMIT_WEBHOOK: str = "100/minute"
    RATE_LIMIT_API: str = "60/minute"
    RATE_LIMIT_DASHBOARD: str = "30/minute"

    # Chatwoot settings
    CHATWOOT_API_KEY: str
    CHATWOOT_ACCOUNT_ID: str
    CHATWOOT_BASE_URL: str
    CHATWOOT_API_TOKEN: str
    CHATWOOT_WEBHOOK_SECRET: str = ""  # Optional, but recommended for production

    # LangChain settings
    LANGCHAIN_ENDPOINT: str
    LANGCHAIN_API_KEY: str
    LANGSMITH_TAGS: List[str]

    # DeepSeek settings
    DEEPSEEK_API_KEY: str
    DEEPSEEK_MODEL_NAME: str = "deepseek-reasoner"

    # Redis settings
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""

    model_config = SettingsConfigDict(
        env_file=".env.test" if os.path.exists(".env.test") else ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow",  # Allow extra fields
        json_schema_extra={
            "example": {
                "PORT": 8000,
                "ENVIRONMENT": "development",
                "LOG_LEVEL": "INFO",
            }
        }
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Parse LANGSMITH_TAGS from string to list if it's a string
        if isinstance(self.LANGSMITH_TAGS, str):
            try:
                self.LANGSMITH_TAGS = json.loads(self.LANGSMITH_TAGS)
            except json.JSONDecodeError:
                self.LANGSMITH_TAGS = []
        
        # In production, ensure CORS_ORIGINS is properly restricted
        if self.ENVIRONMENT == "production" and "*" in self.CORS_ORIGINS:
            raise ValueError("CORS_ORIGINS must be explicitly set in production")

    def validate_settings(self):
        """Validate required settings"""
        required_vars = [
            "CHATWOOT_API_KEY",
            "CHATWOOT_ACCOUNT_ID",
            "CHATWOOT_BASE_URL",
            "REDIS_HOST",
            "REDIS_PORT"
        ]
        
        missing = []
        for var in required_vars:
            if not getattr(self, var):
                missing.append(var)
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

settings = Settings()
settings.validate_settings()
