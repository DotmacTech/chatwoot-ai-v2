from typing import Dict, Any, Optional
from core.config import settings
import httpx
from models.intent import IntentClassification
from models.webhook import WebhookPayload
from chatwoot_langchain.intent_classifier import IntentClassifier
from chatwoot_langsmith.integration import langsmith_manager
from chatwoot_langgraph.workflow import workflow_manager
import logging
import time
import json
import traceback
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

class ChatwootHandler:
    def __init__(self):
        self.base_url = settings.CHATWOOT_BASE_URL
        self.account_id = settings.CHATWOOT_ACCOUNT_ID
        self.api_token = settings.CHATWOOT_API_TOKEN
        self._last_message_id = None
        self._last_message_time = 0
        self._dedup_window = 10  # Increased to 10 seconds
        self._processed_messages = set()  # Track processed message IDs
        self.intent_classifier = IntentClassifier()

    async def process(self, payload: WebhookPayload) -> Dict[str, Any]:
        """Process incoming webhook from Chatwoot"""
        try:
            # Extract relevant information from payload
            event = payload.event
            
            # Log the full payload for debugging
            logger.info("Processing webhook with full payload", extra={
                "payload": payload.dict() if hasattr(payload, "dict") else str(payload)
            })
            
            # Handle both top-level message fields and message object
            message_id = getattr(payload, 'id', None)
            message_content = getattr(payload, 'content', None)
            message_type = getattr(payload, 'message_type', None)
            
            # Use the message object if available, otherwise use top-level fields
            message = getattr(payload, 'message', None)
            if message:
                message_id = getattr(message, 'id', message_id)
                message_content = getattr(message, 'content', message_content)
                message_type = getattr(message, 'message_type', message_type)
            
            conversation = getattr(payload, 'conversation', None)
            contact = getattr(payload, 'contact', None)
            sender = getattr(payload, 'sender', None)
            
            # Debug logging
            logger.info("Processing webhook with payload details", extra={
                "event_type": event,
                "message_present": message is not None,
                "conversation_present": conversation is not None,
                "contact_present": contact is not None,
                "sender_present": sender is not None,
                "message_id": message_id,
                "message_content": message_content,
                "message_type": message_type,
                "conversation_id": getattr(conversation, 'id', None) if conversation else None
            })
            
            # Check for required data
            if not message_content or not conversation:
                logger.error("Missing required data", extra={
                    "message_content": message_content is not None,
                    "conversation": conversation is not None
                })
                return {
                    "status": "skipped",
                    "reason": "missing_required_data"
                }
            
            # Create a LangSmith run for this webhook processing
            with langsmith_manager.create_run(
                name="process_webhook",
                inputs={"event": event, "message_id": message_id},
                tags=["webhook", "chatwoot"]
            ) as run:
                try:
                    logger.info("Processing webhook payload", 
                               extra={
                                   "event": event,
                                   "message_id": message_id,
                                   "conversation_id": getattr(conversation, 'id', None) if conversation else None
                               })

                    # Skip if not a new message event or if it's not an incoming message
                    if event != "message_created" or message_type != "incoming":
                        logger.info("Skipping non-incoming message", extra={
                            "event": event,
                            "message_type": message_type
                        })
                        return {
                            "status": "skipped",
                            "reason": "not_new_message"
                        }

                    # Deduplicate messages
                    current_time = time.time()
                    if (message_id == self._last_message_id and 
                        current_time - self._last_message_time < self._dedup_window):
                        return {
                            "status": "skipped",
                            "reason": "duplicate_message"
                        }

                    # Update dedup tracking
                    self._last_message_id = message_id
                    self._last_message_time = current_time

                    # Get conversation history
                    try:
                        conversation_id = getattr(conversation, 'id', 0)
                        prev_messages = await self._get_conversation_history(conversation_id)
                        logger.info("Retrieved conversation history", extra={"message_count": len(prev_messages)})
                    except Exception as e:
                        logger.error(f"Error getting conversation history: {str(e)}", exc_info=True)
                        prev_messages = []

                    # Check if contact information is available
                    # Use either contact or sender information
                    contact_info = contact or sender
                    if not contact_info:
                        logger.error("Contact information is missing", extra={
                            "conversation_id": getattr(conversation, 'id', None) if conversation else None
                        })
                        return {
                            "status": "error",
                            "error": "Contact information is missing"
                        }
                    
                    contact_id = getattr(contact_info, 'id', 0)
                    contact_name = getattr(contact_info, 'name', '')
                    contact_email = getattr(contact_info, 'email', '')
                    
                    logger.info("Contact details", extra={
                        "contact_id": contact_id,
                        "contact_name": contact_name,
                        "contact_email": contact_email
                    })

                    # Process through LangGraph workflow
                    try:
                        result = await workflow_manager.process_message(
                            message=message_content,
                            customer_id=str(contact_id),
                            conversation_id=str(getattr(conversation, 'id', 0)),
                            contact_info=contact_email if contact_email else "",
                            prev_messages=prev_messages
                        )
                        logger.info("Message processed successfully", extra={"result": result})
                    except Exception as e:
                        logger.error(f"Error in workflow processing: {str(e)}", exc_info=True)
                        return {
                            "status": "error",
                            "error": f"Workflow processing error: {str(e)}"
                        }

                    # Send response back to Chatwoot
                    if result and "response" in result and result["response"]:
                        await self._send_response(
                            conversation_id=getattr(conversation, 'id', 0),
                            message=result["response"]
                        )
                        logger.info("Response sent to Chatwoot")
                    
                    # Update run with output if it exists
                    if run:
                        run.end(outputs={"status": "success", "response_sent": bool(result.get("response", ""))})
                    
                    return {
                        "status": "success",
                        "response_sent": bool(result.get("response", ""))
                    }
                except Exception as e:
                    error_trace = traceback.format_exc()
                    logger.error(f"Error processing webhook: {str(e)}\n{error_trace}")
                    if run:
                        run.end(error=str(e))
                    return {
                        "status": "error",
                        "error": str(e)
                    }
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Error in webhook processing: {str(e)}\n{error_trace}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _get_conversation_history(self, conversation_id: int) -> list:
        """Get conversation history from Chatwoot API"""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/api/v1/accounts/{self.account_id}/conversations/{conversation_id}/messages"
                headers = {
                    "api_access_token": self.api_token,
                    "Content-Type": "application/json"
                }
                
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                # The API response structure may have a 'payload' or other wrapper
                response_data = response.json()
                
                # Check if the response is a dictionary with a 'payload' key
                if isinstance(response_data, dict) and 'payload' in response_data:
                    messages = response_data.get('payload', [])
                # Check if the response is a dictionary with a 'meta' and 'data' keys
                elif isinstance(response_data, dict) and 'meta' in response_data and 'data' in response_data:
                    messages = response_data.get('data', [])
                # If it's already a list, use it directly
                elif isinstance(response_data, list):
                    messages = response_data
                else:
                    logger.warning(f"Unexpected response structure: {type(response_data)}")
                    messages = []
                
                logger.info(f"Retrieved {len(messages)} messages from conversation history")
                
                # Format messages for the workflow - convert to LangChain message format
                formatted_messages = []
                
                for msg in messages:
                    # Skip if msg is not a dictionary
                    if not isinstance(msg, dict):
                        logger.warning(f"Skipping non-dict message: {type(msg)}")
                        continue
                    
                    # Get content and message type
                    content = msg.get("content", "")
                    if not content:
                        logger.debug("Skipping message with empty content")
                        continue
                    
                    # Determine message type (0 = incoming/user, 1 = outgoing/assistant)
                    message_type = msg.get("message_type")
                    
                    # Convert numeric message_type to string if needed
                    if isinstance(message_type, int):
                        message_type = "incoming" if message_type == 0 else "outgoing"
                    
                    # Create appropriate LangChain message
                    if message_type == "incoming" or message_type == 0:
                        formatted_messages.append(HumanMessage(content=content))
                    else:
                        formatted_messages.append(AIMessage(content=content))
                
                logger.info(f"Formatted {len(formatted_messages)} messages for the workflow")
                return formatted_messages
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Error fetching conversation history: {str(e)}\n{error_trace}")
            # Return empty list instead of raising to avoid breaking the flow
            return []
    
    async def _send_response(self, conversation_id: int, message: str) -> None:
        """Send response back to Chatwoot"""
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
                
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                
                return response.json()
        except Exception as e:
            logger.error(f"Error sending response to Chatwoot: {str(e)}", exc_info=True)
            raise
