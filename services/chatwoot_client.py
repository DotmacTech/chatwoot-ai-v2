"""
Chatwoot Client Service Module.

Provides a dedicated service for interacting with the Chatwoot Client API,
encapsulating all API calls and configurations.
"""

import httpx
from typing import Dict, Any, Optional, List, Union

from core.config import settings # Centralized configuration
from utils.logging import AppLogger # Consistent logging
from langchain_core.messages import AIMessage, HumanMessage

logger = AppLogger(__name__)

class ChatwootClient:
    """Dedicated service for Chatwoot API interactions."""

    def __init__(self):
        """
        Initialize the ChatwootClient using settings.
        """
        self.base_url = settings.CHATWOOT_BASE_URL
        self.api_token = settings.CHATWOOT_API_TOKEN.get_secret_value() if settings.CHATWOOT_API_TOKEN else None
        self.account_id = settings.CHATWOOT_ACCOUNT_ID
        self.client = None # Initialize client as None
        self.api_base = None
        self.agent_id: Optional[int] = None # Store the agent ID for loop prevention

        if not self.base_url or not self.api_token or not self.account_id:
            logger.error("Chatwoot configuration (BASE_URL, API_TOKEN, ACCOUNT_ID) incomplete. Client cannot function.")
        else:
            self.api_base = f"{self.base_url}/api/v1"
            # Note: Base URL for httpx client includes the /accounts/{id} part for most operations
            client_base_url = f"{self.api_base}/accounts/{self.account_id}"
            self.client = httpx.AsyncClient(
                base_url=client_base_url,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "api_access_token": self.api_token
                },
                timeout=settings.HTTP_TIMEOUT # Use configurable timeout
            )
            token_preview = f"{self.api_token[:4]}...{self.api_token[-4:]}" if self.api_token else "Not Set"
            logger.info(f"ChatwootClient initialized. Account: {self.account_id}, Base URL: {self.base_url}, Token: {token_preview}")

    async def _fetch_self_profile(self) -> None:
        """Fetch the profile associated with the API token to get the agent ID."""
        # This endpoint is outside the /accounts/{id} scope, but likely still under /api/v1
        profile_endpoint = "/api/v1/profile"
        url = f"{self.base_url}{profile_endpoint}" # Use the root base URL
        if not self.api_token:
             logger.error("Cannot fetch profile, API token is missing.")
             return
        try:
            # Need a temporary client or manual request as the base_url is different
            async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
                 response = await client.get(
                      url,
                      headers={
                           "Content-Type": "application/json; charset=utf-8",
                           "api_access_token": self.api_token
                      }
                 )
                 response.raise_for_status()
                 profile_data = response.json()
                 self.agent_id = profile_data.get('id')
                 if self.agent_id:
                      logger.info(f"Successfully fetched agent profile. Bot Agent ID: {self.agent_id} (Name: {profile_data.get('name')})")
                 else:
                      logger.error("Failed to extract agent ID from profile data.", extra={"profile_data": profile_data})

        except httpx.HTTPStatusError as e:
            # Specific handling for 401 Unauthorized
            if e.response.status_code == 401:
                 logger.critical(f"Authentication failed when fetching Chatwoot profile. Check API Token. Status: {e.response.status_code}", exc_info=True)
            else:
                 logger.error(f"HTTP error fetching Chatwoot profile: Status {e.response.status_code}", exc_info=True)
            self.agent_id = None # Ensure agent_id is None on error
        except httpx.RequestError as e:
            logger.error(f"Network error fetching Chatwoot profile: {e}", exc_info=True)
            self.agent_id = None
        except Exception as e:
            logger.error(f"Unexpected error fetching Chatwoot profile: {e}", exc_info=True)
            self.agent_id = None

    async def _request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Internal helper to make API requests relative to the account base."""
        if not self.client:
            logger.error(f"Chatwoot client not initialized. Cannot make {method} request to {endpoint}.")
            return None

        url = f"{self.api_base}{endpoint}" # Construct full URL
        try:
            response = await self.client.request(method, endpoint, **kwargs)
            response.raise_for_status() # Raise HTTPStatusError for 4xx/5xx
            # Check if response content is empty or not JSON
            if response.status_code == 204: # No Content
                 return {}
            if not response.content:
                 logger.warning(f"{method} {endpoint} returned status {response.status_code} with empty body.")
                 return None # Or return {} depending on expected behavior
            try:
                 return response.json()
            except ValueError: # Includes JSONDecodeError
                 logger.error(f"Failed to decode JSON response from {method} {endpoint}. Status: {response.status_code}, Content: {response.text[:100]}...", exc_info=True)
                 return None # Indicate JSON failure

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling Chatwoot API: {e.request.method} {e.request.url} - Status {e.response.status_code}", exc_info=True)
            # Log response body for debugging if available
            if e.response and e.response.text:
                 logger.error(f"Chatwoot Error Response Body: {e.response.text[:500]}") # Log first 500 chars
            return None # Indicate API error
        except httpx.RequestError as e:
            logger.error(f"Network error calling Chatwoot API: {e.request.method} {e.request.url} - {e}", exc_info=True)
            return None # Indicate network/request error
        except Exception as e:
            logger.error(f"Unexpected error calling Chatwoot API: {method} {endpoint} - {e}", exc_info=True)
            return None # Indicate unexpected error

    async def send_message(self, conversation_id: int, message: str, private: bool = False, content_type: str = "text") -> Optional[Dict[str, Any]]:
        """Send a message to a conversation."""
        endpoint = f"/conversations/{conversation_id}/messages"
        payload = {
            "content": message,
            "message_type": "outgoing", # Always outgoing from the bot's perspective
            "private": private,
            "content_type": content_type,
        }
        logger.debug(f"Sending message to Chatwoot conv {conversation_id}: {message[:50]}...", extra={"payload": payload})
        response_data = await self._request("POST", endpoint, json=payload)
        if response_data is not None:
             logger.info(f"Successfully sent message to conversation {conversation_id}. Message ID: {response_data.get('id')}")
        # Return the full response dict or None if error occurred
        return response_data

    async def get_conversation(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve details of a specific conversation."""
        endpoint = f"/conversations/{conversation_id}"
        logger.debug(f"Fetching conversation {conversation_id} from Chatwoot.")
        return await self._request("GET", endpoint)

    async def get_messages(self, conversation_id: int) -> Optional[List[Dict[str, Any]]]:
        """Retrieve all messages for a specific conversation."""
        # Note: Chatwoot API for messages might return payload.messages directly
        endpoint = f"/conversations/{conversation_id}/messages"
        logger.debug(f"Fetching messages for conversation {conversation_id} from Chatwoot.")
        response = await self._request("GET", endpoint)
        # The API might return {"payload": [messages]} or just [messages]
        if isinstance(response, dict) and "payload" in response and isinstance(response["payload"], list):
            return response["payload"]
        elif isinstance(response, list):
             return response
        elif response is not None: # Log unexpected format
             logger.warning(f"Unexpected format received for messages of conv {conversation_id}: {type(response)}")
        return None # Return None if error or unexpected format

    async def update_conversation_labels(self, conversation_id: int, labels: List[str]) -> Optional[Dict[str, Any]]:
        """Update the labels for a conversation."""
        endpoint = f"/conversations/{conversation_id}/labels"
        payload = {"labels": labels}
        logger.debug(f"Updating labels for conv {conversation_id} to: {labels}")
        return await self._request("POST", endpoint, json=payload)

    async def toggle_conversation_status(self, conversation_id: int, status: str = "open") -> Optional[Dict[str, Any]]:
        """Toggle the status of a conversation (e.g., 'open', 'resolved')."""
        endpoint = f"/conversations/{conversation_id}/toggle_status"
        payload = {"status": status}
        logger.debug(f"Setting status for conv {conversation_id} to: {status}")
        return await self._request("POST", endpoint, json=payload)

    async def get_conversation_history(
        self, 
        conversation_id: int, 
        limit: int = 20 # Optional limit
    ) -> List[Union[AIMessage, HumanMessage]]:
        """Fetches message history for a conversation from Chatwoot.

        Args:
            conversation_id: The ID of the conversation.
            limit: Maximum number of messages to fetch.

        Returns:
            A list of AIMessage or HumanMessage objects representing the history,
            or an empty list if an error occurs or no history is found.
        """
        if not self.client:
            logger.error("Chatwoot client not initialized. Cannot fetch history.")
            return []

        url = f"{self.base_url}/api/v1/accounts/{self.account_id}/conversations/{conversation_id}/messages?limit={limit}"
        logger.info(f"Fetching conversation history from: {url}")
       
        try:
            response = await self.client.get(url, headers={"api_access_token": self.api_token})
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
           
            messages_data = response.json()
            logger.debug(f"Received raw messages data: {messages_data}")

            # Ensure payload is a list (Chatwoot API might return {'payload': [...]})
            if isinstance(messages_data, dict) and 'payload' in messages_data:
                messages_list = messages_data['payload']
            elif isinstance(messages_data, list):
                messages_list = messages_data
            else:
                logger.error(f"Unexpected response format for conversation history: {type(messages_data)}")
                return []

            if not isinstance(messages_list, list):
                logger.error(f"Expected a list of messages in payload, but got: {type(messages_list)}")
                return []

            history: List[Union[AIMessage, HumanMessage]] = []
            # Process messages in reverse chronological order (API returns newest first)
            # and then reverse the final list to get chronological order for the LLM.
            for message in reversed(messages_list):
                content = message.get("content")
                message_type = message.get("message_type") # outgoing (agent/bot) or incoming (user)
                sender_type = message.get("sender", {}).get("type") # agent, bot, user
               
                if not content: # Skip messages without content (e.g., activity logs)
                    logger.debug(f"Skipping message ID {message.get('id')} with no content.")
                    continue
               
                # Classify as AIMessage (bot/agent) or HumanMessage (user)
                if message_type == "outgoing" or sender_type in ["agent", "bot"]:
                    history.append(AIMessage(content=content))
                elif message_type == "incoming" or sender_type == "user" or sender_type == "contact" or message_type == 0:
                    # Type 0 messages from contacts are user messages
                    logger.debug(f"Processing user message (type={message_type}, sender={sender_type}): {content[:50]}...")
                    history.append(HumanMessage(content=content))
                else:
                    logger.warning(f"Unknown message type/sender combination: type={message_type}, sender={sender_type}. Skipping message ID {message.get('id')}.")
            logger.info(f"Successfully fetched and processed {len(history)} messages for conversation {conversation_id}.")
            # Reverse the history to maintain chronological order (oldest first)
            return history # No need to reverse again if we iterate `reversed(messages_list)`

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching history for conv {conversation_id}: {e.response.status_code} - {e.response.text}", exc_info=True)
            return []
        except httpx.RequestError as e:
            logger.error(f"Request error fetching history for conv {conversation_id}: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching history for conv {conversation_id}: {e}", exc_info=True)
            return []

    # --- Add other useful Chatwoot API methods as needed ---
    # async def get_contact(self, contact_id: int) -> Optional[Dict[str, Any]]: ...
    # async def update_contact(self, contact_id: int, data: Dict) -> Optional[Dict[str, Any]]: ...
    # async def create_contact(self, data: Dict) -> Optional[Dict[str, Any]]: ...
    # async def assign_agent(self, conversation_id: int, agent_id: int) -> Optional[Dict[str, Any]]: ...

    async def close(self):
        """Close the underlying HTTP client gracefully."""
        if self.client:
            await self.client.aclose()
            logger.info("ChatwootClient HTTP client closed.")
