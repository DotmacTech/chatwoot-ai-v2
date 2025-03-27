# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from main import app
import os
from dotenv import load_dotenv

# Load test environment variables
load_dotenv(".env.test", override=True)

@pytest.fixture(scope="module")
def test_client():
    return TestClient(app)

@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables for testing"""
    # Use environment variables from CI/CD secrets if available
    monkeypatch.setenv("DEEPSEEK_API_KEY", os.getenv("TEST_DEEPSEEK_API_KEY", "test_key"))
    monkeypatch.setenv("CHATWOOT_API_TOKEN", os.getenv("TEST_CHATWOOT_API_TOKEN", "test_token"))
    monkeypatch.setenv("LANGCHAIN_API_KEY", os.getenv("TEST_LANGCHAIN_API_KEY", "test_langsmith"))
    monkeypatch.setenv("LANGCHAIN_PROJECT", "test-project")
    monkeypatch.setenv("CHATWOOT_WEBHOOK_SECRET", os.getenv("TEST_CHATWOOT_WEBHOOK_SECRET", "test_secret"))