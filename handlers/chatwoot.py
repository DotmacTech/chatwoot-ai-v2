from typing import Dict, Any, Optional
from core.config import settings
import httpx
from models.webhook import WebhookPayload
from chatwoot_langgraph.workflow import WorkflowManager # Import the class for type hinting
import logging
import traceback
from langchain_core.messages import HumanMessage, AIMessage
from services.langsmith_service import LangSmithService # Import the service
from services.chatwoot_client import ChatwootClient # Import the new client
from services.llm_service import LLMService # Import the LLM service

logger = logging.getLogger(__name__)

class ChatwootHandler:
    def __init__(self, 
                 chatwoot_client: ChatwootClient, 
                 workflow_manager: WorkflowManager,
                 langsmith_service: Optional[LangSmithService],
                 llm_service: LLMService): # Make optional
        self.base_url = settings.CHATWOOT_BASE_URL
        self.account_id = settings.CHATWOOT_ACCOUNT_ID
        self.api_token = settings.CHATWOOT_API_TOKEN
        # Use chatwoot_client.agent_id for loop prevention check
        self.chatwoot_client = chatwoot_client # Use the new client service
        self.workflow_manager = workflow_manager # Restore this assignment
        self.langsmith_service = langsmith_service # Store service instance
        self.llm_service = llm_service # Store LLM service
        
        # Correctly access the secret value for logging
        token_preview = "Not Set" # Default if token is None
        if self.api_token:
            secret_value = self.api_token.get_secret_value()
            token_preview = f"{secret_value[:4]}...{secret_value[-4:]}"
            
        logger.info(f"ChatwootHandler initialized with Account ID: {self.account_id}, Base URL: {self.base_url}, Token Preview: {token_preview}")
        if not self.base_url or not self.account_id or not self.api_token:
            logger.error("Critical Chatwoot configuration (Base URL, Account ID, API Token) missing in ChatwootHandler. Handler may fail.")
            # Consider if this should raise an error or prevent initialization

    async def process(self, payload: WebhookPayload) -> Dict[str, Any]:
        """Process incoming webhook from Chatwoot"""
        try:
            logger.info(f"--- ChatwootHandler.process entered for event: {getattr(payload, 'event', 'UNKNOWN')} ---")
            
            # Extract conversation object for use in error handling
            conversation = getattr(payload, 'conversation', None)
            
            # Extract basic message info if available (can be None)
            message = getattr(payload, 'message', None)
            sender = getattr(payload, 'sender', None)
            
            # Log the message object structure for debugging
            if message:
                logger.info(f"Message object attributes: {dir(message)}")
                logger.info(f"Message object dict: {message.dict() if hasattr(message, 'dict') else str(message)}")
            
            # Extract message details with better fallbacks
            message_id = getattr(message, 'id', None) if message else None
            message_content = None
            message_type = None
            
            event = getattr(payload, 'event', None)
            
            # For message_created, prioritize top-level fields
            if event == "message_created":
                logger.info("Prioritizing top-level payload fields for message_created event.")
                message_content = getattr(payload, 'content', None)
                message_id = getattr(payload, 'id', message_id) # Use top-level ID if available
                
                # Determine message_type from top-level payload or fallback
                raw_type = getattr(payload, 'message_type', None)
                if isinstance(raw_type, str):
                     # Handle cases like "incoming", "outgoing"
                    message_type = raw_type.lower()
                    logger.info(f"Found top-level payload.message_type (string): {message_type}")
                elif isinstance(raw_type, int):
                     # Handle cases like 0, 1 based on Chatwoot convention (0=incoming, 1=outgoing)
                    message_type = "incoming" if raw_type == 0 else "outgoing"
                    logger.info(f"Inferred message_type '{message_type}' from top-level integer: {raw_type}")
                else:
                    logger.warning(f"Unexpected type for top-level message_type: {type(raw_type)}, value: {raw_type}")

                if not message_content:
                     logger.warning("Top-level 'content' missing in message_created payload.")
                if not message_type:
                     logger.warning("Could not determine message_type from top-level payload in message_created.")

            # Fallback or other event types: Try extracting from the message object
            if not message_content and message:
                logger.info("Falling back to message object for content.")
                message_content = getattr(message, 'content', None)
                if not message_content:
                    logger.warning("Content not found in message object either.")
            
            if not message_type and message:
                 logger.info("Falling back to message object for message_type.")
                 raw_type = getattr(message, 'message_type', None)
                 if isinstance(raw_type, str):
                     message_type = raw_type.lower()
                     logger.info(f"Found message.message_type (string): {message_type}")
                 elif isinstance(raw_type, int):
                     message_type = "incoming" if raw_type == 0 else "outgoing"
                     logger.info(f"Inferred message_type '{message_type}' from message object integer: {raw_type}")
                 else:
                     logger.warning(f"Unexpected type for message.message_type: {type(raw_type)}, value: {raw_type}")
                 if not message_type:
                     logger.warning("message_type not found in message object either.")
                     
            logger.info(f"Extracted message_type: {message_type}")
            
            # Log values just before the check
            logger.info(f"Checking required data: message_content='{bool(message_content)}', conversation='{bool(conversation)}'")
            
            if not message_content or not conversation:
                # Determine the exact reason for failure
                reason = "Missing message content" if not message_content else "Missing conversation object"
                logger.error(f"Missing required data: {reason}. Payload event: {event}")
                # Log available details for debugging
                if conversation:
                    logger.info(f"Conversation details (present but maybe content missing): id={conversation.id}")
                if payload:
                    try:
                        # Be careful logging the whole payload, could be large/sensitive
                        payload_dict = payload.dict(exclude_unset=True)
                        logger.info(f"Payload structure (partial): { {k: type(v).__name__ for k, v in payload_dict.items()} }")
                    except Exception as log_err:
                        logger.error(f"Error logging payload details: {log_err}")
                        
                return {"status": "error", "reason": reason}

            conversation_id = payload.conversation.id
            logger.info(f"Processing message for conversation ID: {conversation_id}")
            logger.info(f"Account ID: {self.account_id}")
            logger.info(f"Message type: {message_type}") 
            
            # Get sender information for filtering (Log the raw sender object too)
            sender_name = getattr(sender, 'name', None) if sender else None
            sender_type = getattr(sender, 'type', None) if sender else None
            logger.info(f"Handler extracted sender details: name='{sender_name}', type='{sender_type}'")
            if sender and hasattr(sender, 'dict'):
                logger.info(f"Sender object dict: {sender.dict()}")
            elif sender:
                logger.info(f"Sender object raw: {str(sender)}")
                
            logger.info(f"Sender name: {sender_name}, Sender type: {sender_type}")

            # Log values *before* the check
            logger.debug(f"Loop Check Values: message_type='{message_type}', sender_type='{sender_type}', sender_id='{getattr(sender, 'id', None)}', self.agent_id='{self.chatwoot_client.agent_id}'")

            # Enhanced loop prevention logic
            is_bot_message = False
            
            # Check 1: Standard check - outgoing message from our bot agent
            if message_type == "outgoing" and sender_type == "user" and getattr(sender, 'id', None) == self.chatwoot_client.agent_id:
                logger.info(f"Skipping outgoing message from our bot (Agent ID: {self.chatwoot_client.agent_id}) to prevent loops")
                is_bot_message = True
            
            # Check 2: Check for messages with our agent name (in case ID doesn't match)
            elif message_type == "outgoing" and sender_type == "user" and sender_name and "Test" in sender_name:
                logger.info(f"Skipping outgoing message from sender with our bot name '{sender_name}' to prevent loops")
                is_bot_message = True
                
            # Check 3: Check for recent duplicate messages (within last 5 seconds)
            # This helps catch cases where the bot sends multiple identical messages
            elif message_type == "outgoing" and message_content:
                # Log the detection of an outgoing message for debugging
                logger.info(f"Detected outgoing message: '{message_content[:50]}...' from {sender_type}/{sender_name}")
                
                # Additional check for any outgoing message from a user (likely the bot)
                if sender_type == "user":
                    logger.info(f"Skipping outgoing message from user sender '{sender_name}' (ID: {getattr(sender, 'id', None)}) to prevent loops")
                    is_bot_message = True
            
            if is_bot_message:
                return {"status": "skipped", "reason": "outgoing_bot_agent_message"}

            # Fetch conversation history using the ChatwootClient service
            history = await self.chatwoot_client.get_conversation_history(conversation_id)
             
            # Prepare input for the LangGraph workflow
            logger.info(f"Calling workflow_manager.process_message for conv {conversation_id}")
            
            # Start a LangSmith trace for the workflow call
            if self.langsmith_service:
                with self.langsmith_service.trace(
                    name="Workflow Processing",
                    run_type="chain",
                    inputs={"conversation_id": conversation_id, "message_content": message_content},
                    tags=["workflow", "langgraph"],
                    metadata={
                        "account_id": self.account_id,
                        "conversation_id": conversation_id,
                        "message_type": message_type,
                    }
                ) as trace_id:
                    # Create a simplified chatwoot message dict
                    chatwoot_message = {
                        "id": message_id,
                        "content": message_content,
                        "message_type": message_type,
                        "sender": {
                            "name": sender_name,
                            "type": sender_type,
                            "raw": sender
                        }
                    }
                    
                    # Create a simplified chatwoot conversation dict
                    chatwoot_conversation = {
                        "id": conversation_id,
                        "account_id": self.account_id,
                        "messages": history
                    }
                    
                    response_state = await self.workflow_manager.process_message(
                        conversation_id=conversation_id,
                        current_message=message_content,
                        chatwoot_conversation=chatwoot_conversation,
                        chatwoot_message=chatwoot_message,
                        langsmith_service=self.langsmith_service,
                        llm_service=self.llm_service,
                        parent_trace_id=trace_id
                    )
                    logger.info(f"Workflow manager returned state: {response_state}")
            else:
                # Create a simplified chatwoot message dict
                chatwoot_message = {
                    "id": message_id,
                    "content": message_content,
                    "message_type": message_type,
                    "sender": {
                        "name": sender_name,
                        "type": sender_type,
                        "raw": sender
                    }
                }
                
                # Create a simplified chatwoot conversation dict
                chatwoot_conversation = {
                    "id": conversation_id,
                    "account_id": self.account_id,
                    "messages": history
                }
                
                response_state = await self.workflow_manager.process_message(
                    conversation_id=conversation_id,
                    current_message=message_content,
                    chatwoot_conversation=chatwoot_conversation,
                    chatwoot_message=chatwoot_message,
                    langsmith_service=None,
                    llm_service=self.llm_service,
                    parent_trace_id=None
                )
                logger.info(f"Workflow manager returned state: {response_state}")
 
            # Extract final response from the workflow result state
            final_response = response_state.get("final_response")
 
            if final_response:
                logger.info(f"Workflow generated response: '{final_response[:50]}...'")
 
            # Send the response back to Chatwoot if needed
            response_content = final_response
            if response_content:
                logger.info(f"Attempting to send response to Chatwoot: '{response_content[:100]}...'")
                send_result = await self.chatwoot_client.send_message(
                    conversation_id=conversation_id,
                    message=response_content
                )
                if send_result:
                    logger.info(f"Sent response to conversation {conversation_id}. Message ID: {send_result.get('id')}")
                    return {"status": "success", "response_sent": True}
                else:
                    logger.error(f"Failed to send response to conversation {conversation_id}")
                    # Even if sending failed, mark as processed to avoid retries?
                    # Return success status but indicate send failure, or raise an error?
                    # For now, return processed but without response_sent flag
                    return {"status": "success", "response_sent": False, "reason": "failed_to_send"}
            else:
                logger.info("Workflow did not produce a response to send.")
                return {"status": "success", "response_sent": False}
                    
        except Exception as e:
            # Log any exception that occurs within the process method
            conv_id_for_log = getattr(conversation, 'id', 'UNKNOWN')
            logger.error(f"Exception in ChatwootHandler.process for conv {conv_id_for_log}: {e}", exc_info=True)
            # Return an error status
            return {"status": "error", "reason": f"Internal handler error: {type(e).__name__}"}

    async def _send_response(self, conversation_id: int, message: str) -> None:
        """Send response back to Chatwoot"""
        if not all([self.base_url, self.api_token, self.account_id, conversation_id]):
            logger.error("Missing configuration or IDs for sending response.", extra={
                "base_url_set": bool(self.base_url),
                "api_token_set": bool(self.api_token),
                "account_id": self.account_id,
                "conversation_id": conversation_id
            })
            return
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/api/v1/accounts/{self.account_id}/conversations/{conversation_id}/messages"
                headers = {
                    "api_access_token": self.api_token,
                    "Content-Type": "application/json"
                }
                
                data = {
                    "content": message,
                    "message_type": "outgoing"
                }
                
                logger.info(f"---> Sending response to Chatwoot Conv ID {conversation_id}: '{message[:50]}...'")
                logger.info(f"Request URL: {url}")
                logger.info(f"Request Headers: {headers}")
                logger.info(f"Request Data: {data}")
                
                response = await client.post(url, headers=headers, json=data)
                logger.info(f"<--- Chatwoot API response status: {response.status_code} for Conv ID {conversation_id}")
                
                # Log the response body for debugging
                try:
                    response_body = response.json()
                    logger.info(f"Response body: {response_body}")
                except Exception as json_err:
                    logger.warning(f"Could not parse response as JSON: {str(json_err)}")
                    logger.info(f"Response text: {response.text[:500]}")
                
                # Check for specific error status codes
                if response.status_code == 401:
                    logger.error("Authentication failed with Chatwoot API. Check your API token.")
                    return None
                elif response.status_code == 403:
                    logger.error("Permission denied by Chatwoot API. Check account and conversation IDs.")
                    return None
                elif response.status_code == 404:
                    logger.error(f"Resource not found. Check if conversation {conversation_id} exists in account {self.account_id}.")
                    return None
                elif response.status_code >= 400:
                    logger.error(f"Chatwoot API error: {response.status_code} - {response.text[:500]}")
                    return None
                
                response.raise_for_status()
                
                return response.json()
        except httpx.RequestError as e:
            logger.error(f"Network error sending response to Chatwoot: {str(e)}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error sending response to Chatwoot: {str(e)}", exc_info=True)
            return None
