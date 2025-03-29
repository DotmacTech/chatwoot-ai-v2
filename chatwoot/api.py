import os
from typing import Dict, Any, Optional
import requests
import logging

logger = logging.getLogger(__name__)

class ChatwootAPI:
    """
    Client for interacting with the Chatwoot API
    Handles sending messages and retrieving conversation data
    """
    
    def __init__(self, base_url: Optional[str] = None, api_token: Optional[str] = None):
        """
        Initialize the Chatwoot API client
        
        Args:
            base_url: Chatwoot instance URL (defaults to env var)
            api_token: API access token (defaults to env var)
        """
        self.base_url = base_url or os.getenv("CHATWOOT_BASE_URL")
        self.api_token = api_token or os.getenv("CHATWOOT_API_TOKEN")
        
        if not self.base_url or not self.api_token:
            raise ValueError("CHATWOOT_BASE_URL and CHATWOOT_API_TOKEN must be set")
        
        # Remove trailing slash if present
        self.base_url = self.base_url.rstrip("/")
        
        # Default headers for all requests
        self.headers = {
            "api_access_token": self.api_token,
            "Content-Type": "application/json"
        }

    async def send_message(self, account_id: int, conversation_id: int, message: str) -> Dict[str, Any]:
        """
        Send a message in a conversation
        
        Args:
            account_id: Account ID
            conversation_id: Conversation ID
            message: Message content to send
            
        Returns:
            API response
        """
        url = f"{self.base_url}/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"
        
        data = {
            "content": message,
            "message_type": "outgoing",
            "private": False
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            raise
