# tests/integration/test_server.py
import pytest
from fastapi import HTTPException
import json

def test_health_check(test_client):
    """Test health check endpoint"""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_webhook_processing(test_client):
    """Test webhook processing without authentication"""
    # Test with a simple message payload
    payload = {"event": "message_created", "data": {"content": "Hello"}}
    
    response = test_client.post(
        "/chatwoot-webhook", 
        json=payload
    )
    
    # The webhook should now process without authentication
    assert response.status_code == 200