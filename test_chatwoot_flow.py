#!/usr/bin/env python3
import httpx
import json
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.test")

# Configuration
CHATWOOT_BASE_URL = os.getenv("CHATWOOT_BASE_URL", "https://chat.dotmac.ng")
CHATWOOT_API_TOKEN = os.getenv("CHATWOOT_API_TOKEN", "")
CHATWOOT_ACCOUNT_ID = os.getenv("CHATWOOT_ACCOUNT_ID", "1")

async def create_test_conversation():
    """Create a test conversation in Chatwoot"""
    if not CHATWOOT_API_TOKEN:
        print("Error: CHATWOOT_API_TOKEN not set")
        return None
    
    headers = {
        "Content-Type": "application/json",
        "api_access_token": CHATWOOT_API_TOKEN
    }
    
    # 1. Get the first inbox
    async with httpx.AsyncClient() as client:
        print(f"Getting inboxes from {CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/inboxes")
        try:
            response = await client.get(
                f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/inboxes",
                headers=headers
            )
            response.raise_for_status()
            inboxes = response.json()
            
            if not inboxes or len(inboxes) == 0:
                print("No inboxes found")
                return None
            
            inbox_id = inboxes[0]["id"]
            print(f"Using inbox ID: {inbox_id}")
            
            # 2. Create a contact
            contact_data = {
                "inbox_id": inbox_id,
                "name": "Test Customer",
                "email": f"test_{int(asyncio.get_event_loop().time())}@example.com",
                "phone_number": "+1234567890"
            }
            
            print(f"Creating contact with data: {json.dumps(contact_data, indent=2)}")
            response = await client.post(
                f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/contacts",
                headers=headers,
                json=contact_data
            )
            response.raise_for_status()
            contact = response.json()
            contact_id = contact["id"]
            print(f"Created contact with ID: {contact_id}")
            
            # 3. Create a conversation
            conversation_data = {
                "source_id": "test_conversation",
                "inbox_id": inbox_id,
                "contact_id": contact_id,
                "status": "open",
                "assignee_id": None
            }
            
            print(f"Creating conversation with data: {json.dumps(conversation_data, indent=2)}")
            response = await client.post(
                f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations",
                headers=headers,
                json=conversation_data
            )
            response.raise_for_status()
            conversation = response.json()
            conversation_id = conversation["id"]
            print(f"Created conversation with ID: {conversation_id}")
            
            # 4. Send a message in the conversation
            message_data = {
                "content": "Hello, I need help with my order #12345",
                "message_type": "incoming",
                "private": False
            }
            
            print(f"Sending message to conversation {conversation_id}: {json.dumps(message_data, indent=2)}")
            response = await client.post(
                f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages",
                headers=headers,
                json=message_data
            )
            response.raise_for_status()
            message = response.json()
            print(f"Sent message with ID: {message['id']}")
            
            # 5. Wait for a response
            print("Waiting for AI response (10 seconds)...")
            await asyncio.sleep(10)
            
            # 6. Check for responses
            response = await client.get(
                f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages",
                headers=headers
            )
            response.raise_for_status()
            messages = response.json()
            
            print("\nConversation messages:")
            for msg in messages:
                sender = "Customer" if msg["message_type"] == "incoming" else "AI/Agent"
                print(f"{sender}: {msg['content']}")
            
            return conversation_id
            
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

if __name__ == "__main__":
    asyncio.run(create_test_conversation())
