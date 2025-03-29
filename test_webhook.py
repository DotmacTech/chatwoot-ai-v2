#!/usr/bin/env python3
import httpx
import json
import hmac
import hashlib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.test")

# Configuration
BASE_URL = "http://149.102.135.97:8000"
WEBHOOK_SECRET = os.getenv("CHATWOOT_WEBHOOK_SECRET", "")

# Test payload that mimics a Chatwoot webhook for a new message
payload = {
    "event": "message_created",
    "message": {
        "id": 12345,
        "content": "Hello, I need help with my order",
        "message_type": "incoming",
        "private": False,
        "content_type": "text",
        "created_at": "2025-03-28T15:00:00.000Z"
    },
    "conversation": {
        "id": 6789,
        "inbox_id": 1,
        "status": "open",
        "assignee_id": None,
        "contact_id": 1001,
        "contact_inbox": {
            "contact_id": 1001,
            "contact": {
                "id": 1001,
                "name": "Test User",
                "email": "test@example.com"
            }
        }
    },
    "account": {
        "id": 1
    },
    "contact": {
        "id": 1001,
        "name": "Test User",
        "email": "test@example.com"
    }
}

def generate_signature(payload, secret):
    """Generate HMAC signature for the webhook payload"""
    if not secret:
        return None
    
    payload_json = json.dumps(payload, separators=(',', ':'))
    signature = hmac.new(
        secret.encode(),
        payload_json.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return signature

async def test_webhook():
    """Send a test webhook to the local server"""
    async with httpx.AsyncClient() as client:
        # Generate signature if webhook secret is available
        signature = generate_signature(payload, WEBHOOK_SECRET)
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json"
        }
        
        if signature:
            headers["X-Chatwoot-Signature"] = signature
            print(f"Using signature: {signature}")
        else:
            print("Warning: No webhook secret provided. Signature verification will be skipped.")
        
        # Send request
        url = f"{BASE_URL}/webhook"
        print(f"Sending webhook to: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=30.0
            )
            
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")
            
            return response
        except Exception as e:
            print(f"Error sending webhook: {str(e)}")
            return None

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_webhook())
