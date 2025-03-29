#!/usr/bin/env python3
"""
Test script for Chatwoot webhook with real payload structure
"""
import json
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Real Chatwoot payload from n8n test
REAL_PAYLOAD = {
    "account": {
        "id": 2,
        "name": "Dotmac Technologies"
    },
    "additional_attributes": {},
    "content_attributes": {},
    "content_type": "text",
    "content": "Hey",
    "conversation": {
        "additional_attributes": {
            "chat_id": 670097216
        },
        "can_reply": True,
        "channel": "Channel::Telegram",
        "contact_inbox": {
            "id": 7887,
            "contact_id": 9337,
            "inbox_id": 88,
            "source_id": "670097216",
            "created_at": "2025-03-24T10:18:20.743Z",
            "updated_at": "2025-03-24T10:18:20.743Z",
            "hmac_verified": False,
            "pubsub_token": "1oRhtqHbibFfH3mCWjeqiPzH"
        },
        "id": 1,
        "inbox_id": 88,
        "messages": [
            {
                "id": 227051,
                "content": "Hey",
                "account_id": 2,
                "inbox_id": 88,
                "conversation_id": 1,
                "message_type": 0,
                "created_at": 1743174783,
                "updated_at": "2025-03-28T15:13:03.922Z",
                "private": False,
                "status": "sent",
                "source_id": "1699",
                "content_type": "text",
                "content_attributes": {},
                "sender_type": "Contact",
                "sender_id": 9337,
                "external_source_ids": {},
                "additional_attributes": {},
                "processed_message_content": "Hey",
                "sentiment": {},
                "conversation": {
                    "assignee_id": None,
                    "unread_count": 1,
                    "last_activity_at": 1743174783,
                    "contact_inbox": {
                        "source_id": "670097216"
                    }
                },
                "sender": {
                    "additional_attributes": {
                        "username": "Princekid",
                        "language_code": "en",
                        "social_telegram_user_id": 670097216,
                        "social_telegram_user_name": "Princekid"
                    },
                    "custom_attributes": {},
                    "email": None,
                    "id": 9337,
                    "identifier": None,
                    "name": "Oracle ",
                    "phone_number": None,
                    "thumbnail": "https://chat.dotmac.ng/rails/active_storage/representations/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBbUF6IiwiZXhwIjpudWxsLCJwdXIiOiJibG9iX2lkIn19--328f2339ce69c6843732c4d8b5fc2d856d79d882/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaDdCem9MWm05eWJXRjBTU0lJYW5CbkJqb0dSVlE2RTNKbGMybDZaVjkwYjE5bWFXeHNXd2RwQWZvdyIsImV4cCI6bnVsbCwicHVyIjoidmFyaWF0aW9uIn19--03dca47078ca089c2695bafe74bd3664b021a9d8/file_7.jpg",
                    "type": "contact"
                }
            }
        ],
        "labels": [],
        "meta": {
            "sender": {
                "additional_attributes": {
                    "username": "Princekid",
                    "language_code": "en",
                    "social_telegram_user_id": 670097216,
                    "social_telegram_user_name": "Princekid"
                },
                "custom_attributes": {},
                "email": None,
                "id": 9337,
                "identifier": None,
                "name": "Oracle ",
                "phone_number": None,
                "thumbnail": "https://chat.dotmac.ng/rails/active_storage/representations/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBbUF6IiwiZXhwIjpudWxsLCJwdXIiOiJibG9iX2lkIn19--328f2339ce69c6843732c4d8b5fc2d856d79d882/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaDdCem9MWm05eWJXRjBTU0lJYW5CbkJqb0dSVlE2RTNKbGMybDZaVjkwYjE5bWFXeHNXd2RwQWZvdyIsImV4cCI6bnVsbCwicHVyIjoidmFyaWF0aW9uIn19--03dca47078ca089c2695bafe74bd3664b021a9d8/file_7.jpg",
                "type": "contact"
            },
            "assignee": None,
            "team": None,
            "hmac_verified": False
        },
        "status": "open",
        "custom_attributes": {},
        "snoozed_until": None,
        "unread_count": 1,
        "first_reply_created_at": "2025-03-25T10:21:18.407Z",
        "priority": None,
        "waiting_since": 0,
        "agent_last_seen_at": 1743144538,
        "contact_last_seen_at": 0,
        "timestamp": 1743174783,
        "created_at": 1742811501
    },
    "created_at": "2025-03-28T15:13:03.922Z",
    "id": 227051,
    "inbox": {
        "id": 88,
        "name": "DappairdropBot"
    },
    "message_type": "incoming",
    "private": False,
    "sender": {
        "account": {
            "id": 2,
            "name": "Dotmac Technologies"
        },
        "additional_attributes": {
            "username": "Princekid",
            "language_code": "en",
            "social_telegram_user_id": 670097216,
            "social_telegram_user_name": "Princekid"
        },
        "avatar": "https://chat.dotmac.ng/rails/active_storage/representations/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBbUF6IiwiZXhwIjpudWxsLCJwdXIiOiJibG9iX2lkIn19--328f2339ce69c6843732c4d8b5fc2d856d79d882/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaDdCem9MWm05eWJXRjBTU0lJYW5CbkJqb0dSVlE2RTNKbGMybDZaVjkwYjE5bWFXeHNXd2RwQWZvdyIsImV4cCI6bnVsbCwicHVyIjoidmFyaWF0aW9uIn19--03dca47078ca089c2695bafe74bd3664b021a9d8/file_7.jpg",
        "custom_attributes": {},
        "email": None,
        "id": 9337,
        "identifier": None,
        "name": "Oracle ",
        "phone_number": None,
        "thumbnail": "https://chat.dotmac.ng/rails/active_storage/representations/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBbUF6IiwiZXhwIjpudWxsLCJwdXIiOiJibG9iX2lkIn19--328f2339ce69c6843732c4d8b5fc2d856d79d882/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaDdCem9MWm05eWJXRjBTU0lJYW5CbkJqb0dSVlE2RTNKbGMybDZaVjkwYjE5bWFXeHNXd2RwQWZvdyIsImV4cCI6bnVsbCwicHVyIjoidmFyaWF0aW9uIn19--03dca47078ca089c2695bafe74bd3664b021a9d8/file_7.jpg"
    },
    "source_id": "1699",
    "event": "message_created"
}

def test_webhook():
    """Test the webhook endpoint with a real Chatwoot payload"""
    url = "http://localhost:8000/webhook"
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        logger.info("Sending webhook test with real Chatwoot payload")
        response = requests.post(url, headers=headers, json=REAL_PAYLOAD)
        
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response body: {response.text}")
        
        return response.json()
    except Exception as e:
        logger.error(f"Error testing webhook: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    result = test_webhook()
    print(json.dumps(result, indent=2))
