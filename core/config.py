from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Optional, Dict, Any
import json
import os
import logging
from pydantic import SecretStr

logger = logging.getLogger(__name__)

# Define model costs per 1M tokens (input/output)
DEFAULT_MODEL_COSTS = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
    "deepseek-chat": {"input": 0.14, "output": 0.28}, # Example costs for DeepSeek
    "deepseek-coder": {"input": 0.14, "output": 0.28}
}

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    # Server settings
    PORT: int = Field(8000, description="Port to run the server on")
    ENVIRONMENT: str = "development"
    HTTP_TIMEOUT: int = 30 # Default timeout for HTTP requests in seconds
    LOG_LEVEL: str = "INFO"

    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]  # For development, restrict in production

    # Rate limiting settings
    RATE_LIMIT_WEBHOOK: str = "100/minute"
    RATE_LIMIT_API: str = "60/minute"
    RATE_LIMIT_DASHBOARD: str = "30/minute"

    # Chatwoot settings
    CHATWOOT_ACCOUNT_ID: int = Field(..., description="Chatwoot Account ID")
    CHATWOOT_BASE_URL: str = Field(..., description="Chatwoot Base URL")
    CHATWOOT_API_TOKEN: SecretStr = Field(..., description="Chatwoot API Access Token")
    CHATWOOT_WEBHOOK_SECRET: Optional[SecretStr] = Field(None, description="Chatwoot Webhook Secret (optional)")
    # CHATWOOT_BOT_AGENT_ID is no longer needed, it's fetched dynamically
    # CHATWOOT_BOT_AGENT_ID: int = Field(..., description="Chatwoot Agent ID used by this bot") 

    # LangSmith settings
    LANGSMITH_TRACING_ENABLED: bool = True # Master switch for LangSmith tracing
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: Optional[str] = None # Mark as Optional, validation will check if needed
    LANGCHAIN_PROJECT: str = "chatwoot-automation"
    LANGSMITH_TAGS: List[str] = [] # Default to empty list

    # LangSmith Cost Monitoring Settings (add defaults)
    DAILY_TOKEN_LIMIT: int = 100000
    MONTHLY_BUDGET: float = 50.0  # USD

    # OpenAI / General LLM Settings
    OPENAI_API_KEY: Optional[str] = None
    DEFAULT_MODEL: str = "gpt-4o-mini" # Default LLM model
    DEFAULT_LLM_TEMPERATURE: float = 0.7 # Default temperature
    DEFAULT_LLM_MAX_TOKENS: Optional[int] = None # Default max tokens (None means use model default)
    DEFAULT_LLM_TOP_P: float = 1.0 # Default top_p (1.0 means no nucleus sampling)
    DEFAULT_LLM_N: int = 1 # Default number of completions
    DEFAULT_LLM_STOP: Optional[List[str]] = None # Default stop sequences
    DEFAULT_LLM_STREAM: bool = False # Default streaming behavior
    MODEL_COSTS: Dict[str, Dict[str, float]] = Field(default_factory=lambda: DEFAULT_MODEL_COSTS.copy())

    # DeepSeek settings (Commented out as we switch to OpenAI)
    # DEEPSEEK_API_KEY: Optional[str] = None # Mark as Optional
    # DEEPSEEK_API_BASE: str = "https://api.deepseek.com" # Default DeepSeek API base URL
    # DEEPSEEK_MODEL_NAME: str = "deepseek-reasoner"

    # Redis settings
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None # Allow empty/None

    model_config = SettingsConfigDict(
        # Load .env.test first if it exists, otherwise .env
        env_file=(".env.test", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False, # Env vars usually case-insensitive
        extra="ignore",  # Ignore extra fields from env
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Manually parse complex types that BaseSettings might struggle with
        self._parse_complex_types()
        # Run validation after initial load
        self.validate_settings()

    def _parse_complex_types(self):
        """Parse complex types like JSON strings from environment variables."""
        # Parse LANGSMITH_TAGS
        tags_env = os.getenv("LANGSMITH_TAGS", "[]")
        try:
            parsed_tags = json.loads(tags_env)
            if isinstance(parsed_tags, list):
                self.LANGSMITH_TAGS = parsed_tags
            else:
                 logger.warning("LANGSMITH_TAGS env var is not a valid JSON list, using default.")
                 self.LANGSMITH_TAGS = []
        except json.JSONDecodeError:
            logger.warning("LANGSMITH_TAGS env var is not valid JSON, using default.")
            self.LANGSMITH_TAGS = []

        # Parse MODEL_COSTS
        model_costs_env = os.getenv("MODEL_COSTS")
        if model_costs_env:
            logger.info(f"Attempting to load MODEL_COSTS from env var: {model_costs_env[:100]}...")
            try:
                parsed_costs = json.loads(model_costs_env)
                # Basic validation: Check if it's a dictionary and keys are strings
                if isinstance(parsed_costs, dict):
                    valid_costs = True
                    for model, costs in parsed_costs.items():
                        if not (isinstance(costs, dict) and 'input' in costs and 'output' in costs):
                            valid_costs = False
                            break
                    if valid_costs:
                        self.MODEL_COSTS = parsed_costs
                    else:
                        logger.warning("MODEL_COSTS env var structure is invalid, using default.")
                        self.MODEL_COSTS = DEFAULT_MODEL_COSTS
            except json.JSONDecodeError:
                logger.warning("MODEL_COSTS env var is not valid JSON, using default.")
                self.MODEL_COSTS = DEFAULT_MODEL_COSTS
        else:
            # If not set in env, use the default factory value
            pass # Already handled by pydantic Field default_factory

        # Parse CORS_ORIGINS
        origins_env = os.getenv("CORS_ORIGINS", '["*"]') # Default to allow all as string
        try:
            parsed_origins = json.loads(origins_env)
            if isinstance(parsed_origins, list) and all(isinstance(i, str) for i in parsed_origins):
                self.CORS_ORIGINS = parsed_origins
            else:
                 logger.warning("CORS_ORIGINS env var is not a valid JSON list of strings, using default '["*"]'.")
                 self.CORS_ORIGINS = ["*"]
        except json.JSONDecodeError:
             logger.warning("CORS_ORIGINS env var is not valid JSON, using default '["*"]'.")
             self.CORS_ORIGINS = ["*"]


    def validate_settings(self):
        """Validate required settings and environment-specific rules."""
        required_vars = [
            "CHATWOOT_ACCOUNT_ID",
            "CHATWOOT_BASE_URL",
            "CHATWOOT_API_TOKEN", # Added as it's essential
            "REDIS_HOST",
            "REDIS_PORT",
            # Add others that are truly mandatory for core function
        ]

        # Conditionally required based on features used
        # For example, if LangSmith is intended to be used:
        # if self.LANGCHAIN_API_KEY is None: # Or some other flag indicating LS is needed
        #    required_vars.append("LANGCHAIN_API_KEY")
        # If DeepSeek is intended:
        # if self.DEEPSEEK_API_KEY is None: # Or flag
        #    required_vars.append("DEEPSEEK_API_KEY")


        missing = [var for var in required_vars if not getattr(self, var, None)]
        if missing:
            logger.critical(f"Missing required environment variables: {', '.join(missing)}")
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        # In production, ensure CORS_ORIGINS is properly restricted
        if self.ENVIRONMENT == "production" and self.CORS_ORIGINS == ["*"]:
             logger.critical("CORS_ORIGINS must be explicitly set to trusted domains in production")
             raise ValueError("CORS_ORIGINS must be explicitly set in production, cannot be '*'")

        # Ensure webhook secret is set in production if not explicitly empty
        if self.ENVIRONMENT == "production" and not self.CHATWOOT_WEBHOOK_SECRET:
            logger.warning("CHATWOOT_WEBHOOK_SECRET is not set in production. This is insecure.")

        logger.info("Configuration validated successfully.")


# Initialize settings instance - this will load .env and run validation
try:
    settings = Settings()
except ValueError as e:
    logger.critical(f"Configuration error: {e}")
    # Exit if configuration is invalid
    import sys
    sys.exit(1)
