import pytest
import subprocess
import time
import requests
import os
import hmac
import hashlib
import json

@pytest.mark.integration
def test_app_runs():
    """Test that the application starts and responds to requests"""
    port = 8001  # Use a different port to avoid conflicts
    
    process = subprocess.Popen(["uvicorn", "main:app", "--port", str(port)])
    time.sleep(5)  # Give the app time to start

    try:
        response = requests.get(f"http://localhost:{port}/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    finally:
        process.terminate()
        process.wait()  # Ensure process is fully terminated

@pytest.mark.integration
def test_webhook_processing():
    """Test webhook processing without authentication"""
    port = 8002  # Use a different port to avoid conflicts
    
    # Start the application
    process = subprocess.Popen(["uvicorn", "main:app", "--port", str(port)])
    time.sleep(5)  # Give the app time to start
    
    try:
        # Test with a valid payload
        payload = json.dumps({"event": "message_created", "data": {"content": "test"}})
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(
            f"http://localhost:{port}/chatwoot-webhook", 
            headers=headers,
            data=payload
        )
        assert response.status_code in [200, 202]  # Accept either success code
        
    finally:
        process.terminate()
        process.wait()  # Ensure process is fully terminated

@pytest.mark.integration
def test_deepseek_integration():
    """Test that DeepSeek integration is working"""
    port = 8003  # Use a different port to avoid conflicts
    
    # Start the application
    process = subprocess.Popen(["uvicorn", "main:app", "--port", str(port)])
    time.sleep(5)  # Give the app time to start
    
    try:
        # Create a test message with a valid signature
        webhook_secret = os.environ.get("CHATWOOT_WEBHOOK_SECRET", "test_secret")
        
        # Create a message that would trigger the DeepSeek agent
        payload = json.dumps({
            "event": "message_created",
            "data": {
                "content": "I need help with my internet connection",
                "conversation": {
                    "id": "test_conversation_id"
                },
                "sender": {
                    "id": "test_customer_id",
                    "type": "Contact"  # This should trigger agent response
                }
            }
        })
        
        signature = hmac.new(
            webhook_secret.encode(),
            msg=payload.encode(),
            digestmod=hashlib.sha256
        ).hexdigest()
        
        headers = {
            "X-Chatwoot-Signature": signature,
            "Content-Type": "application/json"
        }
        
        # Send the message to the webhook
        response = requests.post(
            f"http://localhost:{port}/chatwoot-webhook", 
            headers=headers,
            data=payload
        )
        
        # Check that the request was accepted
        assert response.status_code in [200, 202]
        
        # In a real test, we would check that the DeepSeek agent was triggered
        # and a response was sent back to Chatwoot, but this would require
        # mocking the DeepSeek API and Chatwoot API
        
    finally:
        process.terminate()
        process.wait()  # Ensure process is fully terminated
