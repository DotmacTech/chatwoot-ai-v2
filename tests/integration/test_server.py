# tests/integration/test_server.py
import pytest
from fastapi import HTTPException
import hmac
import hashlib
import os

def test_health_check(test_client):
    """Test health check endpoint"""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_webhook_authentication(test_client, mock_env):
    """Test webhook authentication"""
    # Test without signature
    response = test_client.post("/chatwoot-webhook")
    assert response.status_code == 401
    assert "Missing signature header" in response.json()["detail"]
    
    # Test with invalid signature
    headers = {"X-Chatwoot-Signature": "invalid_signature"}
    response = test_client.post("/chatwoot-webhook", headers=headers)
    assert response.status_code == 401
    assert "Invalid signature" in response.json()["detail"]
    
    # Test with valid signature
    payload = b'{"event": "message_created"}'
    secret = os.getenv("TEST_CHATWOOT_WEBHOOK_SECRET", "test_secret").encode()
    signature = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    
    headers = {"X-Chatwoot-Signature": signature}
    response = test_client.post("/chatwoot-webhook", headers=headers, data=payload)
    assert response.status_code == 200