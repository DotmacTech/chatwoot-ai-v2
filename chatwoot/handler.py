import os
import requests
import logging
import time
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from chatwoot.channels import get_channel_handler
from chatwoot.routing import create_default_router, AgentType
from chatwoot_langsmith import tracing_manager, feedback_manager, cost_monitor
from chatwoot_langchain import intent_classifier
from chatwoot_langgraph import workflow_manager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Initialize the router
router = create_default_router()

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
    
    def send_message(self, 
                     account_id: int, 
                     conversation_id: int, 
                     message: str, 
                     message_type: str = "outgoing", 
                     private: bool = False) -> Dict[str, Any]:
        """
        Send a message to a Chatwoot conversation
        
        Args:
            account_id: Chatwoot account ID
            conversation_id: Conversation ID to send message to
            message: Message content
            message_type: Type of message (outgoing, template)
            private: Whether message is private (internal note)
            
        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"
        
        payload = {
            "content": message,
            "message_type": message_type,
            "private": private
        }
        
        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending message to Chatwoot: {e}")
            raise
    
    def get_conversation(self, account_id: int, conversation_id: int) -> Dict[str, Any]:
        """
        Get details of a specific conversation
        
        Args:
            account_id: Chatwoot account ID
            conversation_id: Conversation ID to retrieve
            
        Returns:
            Conversation details as dictionary
        """
        url = f"{self.base_url}/api/v1/accounts/{account_id}/conversations/{conversation_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving conversation from Chatwoot: {e}")
            raise
    
    def get_messages(self, account_id: int, conversation_id: int) -> List[Dict[str, Any]]:
        """
        Get all messages in a conversation
        
        Args:
            account_id: Chatwoot account ID
            conversation_id: Conversation ID to retrieve messages from
            
        Returns:
            List of messages
        """
        url = f"{self.base_url}/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving messages from Chatwoot: {e}")
            raise
    
    def assign_conversation(self, 
                           account_id: int, 
                           conversation_id: int, 
                           assignee_id: int) -> Dict[str, Any]:
        """
        Assign a conversation to an agent
        
        Args:
            account_id: Chatwoot account ID
            conversation_id: Conversation ID to assign
            assignee_id: User ID of the agent to assign
            
        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}/api/v1/accounts/{account_id}/conversations/{conversation_id}/assignments"
        
        payload = {
            "assignee_id": assignee_id
        }
        
        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error assigning conversation in Chatwoot: {e}")
            raise

class ChatwootHandler:
    """
    Handler for Chatwoot webhook events.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the handler with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.chatwoot_client = ChatwootAPI(
            base_url=os.getenv("CHATWOOT_BASE_URL"),
            api_token=os.getenv("CHATWOOT_API_TOKEN")
        )
        self.feedback_manager = feedback_manager
        self.cost_monitor = cost_monitor
        
        # Create default router
        self.router = create_default_router()
    
    @tracing_manager.traceable(run_type="chain")
    async def process_message(self, conversation_id: int, message_id: int, account_id: int, inbox_id: int) -> Dict[str, Any]:
        """
        Process a new message in a conversation.
        
        Args:
            conversation_id: ID of the conversation
            message_id: ID of the message
            account_id: ID of the account
            inbox_id: ID of the inbox
            
        Returns:
            Processing result
        """
        # Check usage limits before processing
        within_limits, limit_reason = self.cost_monitor.check_limits()
        if not within_limits:
            logger.warning(f"Usage limits exceeded: {limit_reason}")
            return {
                "status": "error", 
                "message": f"Usage limits exceeded: {limit_reason}",
                "conversation_id": conversation_id
            }
        
        try:
            # Get conversation details
            conversation = await self.chatwoot_client.get_conversation(account_id, conversation_id)
            
            # Get the latest message
            message = await self.chatwoot_client.get_message(account_id, message_id)
            
            if not message or not conversation:
                return {"status": "error", "message": "Failed to retrieve message or conversation"}
            
            # Skip if the message is not from a contact
            if message.get("message_type") != "incoming" or message.get("private"):
                return {"status": "skipped", "reason": "Not a customer message"}
            
            # Get the message content
            message_content = message.get("content", "")
            if not message_content:
                return {"status": "skipped", "reason": "Empty message content"}
            
            # Estimate cost for processing this message
            input_tokens = self.cost_monitor.estimate_tokens(message_content)
            estimated_cost = self.cost_monitor.estimate_cost(input_tokens=input_tokens, output_tokens=100)
            
            # Classify the intent of the message
            classification = intent_classifier.classify_intent(message_content)
            
            # Update conversation custom attributes with intent classification
            custom_attributes = conversation.get("custom_attributes", {})
            custom_attributes["intent"] = classification["intent"]
            custom_attributes["intent_confidence"] = classification["confidence"]
            custom_attributes["intent_timestamp"] = classification["timestamp"]
            
            await self.chatwoot_client.update_conversation(
                account_id, 
                conversation_id, 
                custom_attributes=custom_attributes
            )
            
            # Determine if we need to assign to a human agent based on intent and confidence
            intent_team = classification["intent"]
            
            # Get available human agents for this inbox
            available_agents = await self.chatwoot_client.get_agents(account_id, inbox_id)
            
            # Filter agents based on intent if possible (using teams as a proxy for intent specialization)
            filtered_agents = []
            for agent in available_agents:
                # Check if agent has teams attribute and if any team matches the intent
                agent_teams = agent.get("teams", [])
                if any(team.lower() == intent_team.lower() for team in agent_teams):
                    filtered_agents.append(agent)
            
            # If no intent-specific agents are available, use any available agent
            target_agents = filtered_agents if filtered_agents else available_agents
            
            if target_agents:
                # Select an agent (simple round-robin for now)
                agent_id = target_agents[conversation_id % len(target_agents)].get("id")
                
                # Assign the conversation to the selected agent
                await self.chatwoot_client.assign_conversation(
                    account_id, conversation_id, agent_id
                )
                
                # Add a private note with the intent classification
                private_note = (
                    f"Intent Classification:\n"
                    f"- Intent: {classification['intent']}\n"
                    f"- Confidence: {classification['confidence']:.2f}\n"
                    f"- Reasoning: {classification['reasoning']}\n\n"
                    f"Suggested response: {classification['suggested_response']}"
                )
                
                await self.chatwoot_client.create_note(
                    account_id, conversation_id, private_note, is_private=True
                )
            else:
                # No agents available, add to unassigned
                private_note = (
                    f"No agents available for {intent_team} intent.\n"
                    f"Intent Classification:\n"
                    f"- Intent: {classification['intent']}\n"
                    f"- Confidence: {classification['confidence']:.2f}\n"
                    f"- Reasoning: {classification['reasoning']}"
                )
                
                await self.chatwoot_client.create_note(
                    account_id, conversation_id, private_note, is_private=True
                )
            
            # Track token usage and cost
            self.cost_monitor.track_usage(
                input_tokens=input_tokens,
                output_tokens=0,  # No AI response generated yet
                metadata={
                    "conversation_id": conversation_id,
                    "message_id": message_id,
                    "intent": classification["intent"]
                }
            )
            
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "message_id": message_id,
                "intent": classification["intent"],
                "confidence": classification["confidence"],
                "assigned_agent_id": agent_id if target_agents else None
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {"status": "error", "message": str(e), "conversation_id": conversation_id}

# Webhook handler functions
async def process_webhook(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a webhook event from Chatwoot.
    
    Args:
        payload: The webhook payload
        
    Returns:
        Processing result
    """
    try:
        event_type = payload.get("event")
        if not event_type:
            return {"status": "error", "message": "Missing event type"}
        
        # Initialize handler
        handler = ChatwootHandler()
        
        # Process based on event type
        if event_type == "message_created":
            message = payload.get("message", {})
            conversation = payload.get("conversation", {})
            
            # Skip if not a customer message
            if message.get("message_type") != "incoming" or message.get("private"):
                return {"status": "skipped", "reason": "Not a customer message"}
            
            # Process the message
            return await handler.process_message(
                conversation_id=conversation.get("id"),
                message_id=message.get("id"),
                account_id=payload.get("account", {}).get("id"),
                inbox_id=conversation.get("inbox_id")
            )
        
        # Other event types can be handled here
        return {"status": "skipped", "reason": f"Event type {event_type} not handled"}
    
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return {"status": "error", "message": str(e)}

async def handle_message_created(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle message_created event from Chatwoot
    
    Args:
        payload: Webhook payload
        
    Returns:
        Response dictionary
    """
    # Create trace for message handling
    trace_metadata = {
        "message_type": payload.get("message", {}).get("message_type"),
        "conversation_id": payload.get("conversation", {}).get("id"),
        "source": "message_created"
    }
    
    # Use tracing manager to trace the function execution
    return await tracing_manager.trace_function(
        name="handle_message_created",
        func=_handle_message_created,
        payload=payload,
        metadata=trace_metadata,
        tags=["message", "processing"]
    )

async def _handle_message_created(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Internal implementation of message created handling
    """
    message = payload.get("message", {})
    conversation = payload.get("conversation", {})
    
    # Skip messages created by the bot itself
    if message.get("sender_type") == "user" and message.get("sender", {}).get("type") == "bot":
        return {"status": "ignored", "reason": "bot_message"}
    
    # Only process incoming messages from contacts
    if message.get("message_type") != "incoming":
        return {"status": "ignored", "reason": "not_incoming"}
    
    # Extract relevant information
    account_id = payload.get("account", {}).get("id")
    conversation_id = conversation.get("id")
    
    # Get conversation details if needed
    api_client = ChatwootAPI()
    conversation_details = api_client.get_conversation(account_id, conversation_id)
    
    # Get channel handler for this conversation
    channel_handler = get_channel_handler(conversation)
    
    # Process message through channel handler
    channel_result = await channel_handler.process_message(
        conversation_id=conversation_id, 
        message_id=message.get("id"), 
        account_id=account_id, 
        inbox_id=conversation.get("inbox_id")
    )
    
    # Process message through LangGraph workflow
    workflow_result = workflow_manager.process_message(
        message=message.get("content", ""),
        customer_id=conversation.get("contact_id"),
        conversation_id=str(conversation_id)
    )
    
    # Extract intent and agent type from workflow result
    intent = workflow_result.get("intent", "general")
    agent_used = workflow_result.get("agent_used", "general")
    
    # Map LangGraph agent to router agent type
    agent_mapping = {
        "sales": AgentType.SALES,
        "support": AgentType.SUPPORT,
        "account": AgentType.ACCOUNT,
        "general": AgentType.GENERAL
    }
    
    agent_type = agent_mapping.get(agent_used, AgentType.GENERAL)
    reason = f"Intent classified as '{intent}' and handled by {agent_used} agent"
    
    # Check usage limits before processing
    limits_ok, limit_reason = cost_monitor.check_limits()
    if not limits_ok:
        logger.warning(f"Usage limits exceeded: {limit_reason}")
        # Send a message to the customer about service limitations
        api_client.send_message(
            account_id=account_id,
            conversation_id=conversation_id,
            message="We're currently experiencing high demand. Please try again later or contact support for assistance.",
            message_type="outgoing"
        )
        return {"status": "limited", "reason": limit_reason}
    
    # Estimate cost for this message
    message_content = message.get("content", "")
    estimated_cost = cost_monitor.estimate_cost(
        input_text=message_content,
        expected_output_length=100,  # Estimate for response length
        model=os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
    )
    
    # Create a trace run ID for this message processing
    trace_id = None
    if tracing_manager.enabled:
        trace_id = tracing_manager.create_run(
            name="message_processing",
            inputs={"message": message_content},
            metadata={
                "conversation_id": conversation_id,
                "account_id": account_id,
                "estimated_cost": estimated_cost,
                "intent": intent,
                "agent_type": str(agent_type)
            }
        )
    
    try:
        # Use the response from the LangGraph workflow
        ai_response = workflow_result.get("response", "I'm sorry, I couldn't process your request at this time.")
        
        # Send the response back to the customer
        api_client.send_message(
            account_id=account_id,
            conversation_id=conversation_id,
            message=ai_response,
            message_type="outgoing"
        )
        
        # Record the completion of the trace
        if trace_id and tracing_manager.enabled:
            tracing_manager.update_run(
                run_id=trace_id,
                outputs={"response": ai_response},
                end=True
            )
        
        # Record cost for this interaction
        cost_monitor.record_usage(
            input_tokens=len(message_content) // 4,  # Rough estimate
            output_tokens=len(ai_response) // 4,     # Rough estimate
            model=os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
        )
        
        # Collect feedback if enabled
        if feedback_manager.enabled:
            feedback_manager.create_feedback(
                run_id=trace_id,
                key="response_quality",
                feedback_source_type="api",
                extra={
                    "conversation_id": conversation_id,
                    "message_id": message.get("id"),
                    "intent": intent,
                    "agent_type": str(agent_type)
                }
            )
        
        # Determine if human handoff is needed
        if agent_type in [AgentType.SALES, AgentType.SUPPORT]:
            # Assign to appropriate human agent
            assign_to_human_agent(account_id, conversation_id, agent_type)
            
            # Send handoff message
            api_client.send_message(
                account_id=account_id,
                conversation_id=conversation_id,
                message="I've connected you with a specialist who will continue this conversation shortly.",
                message_type="outgoing"
            )
        
        return {
            "status": "success",
            "response": ai_response,
            "intent": intent,
            "agent_type": str(agent_type),
            "trace_id": trace_id
        }
    
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return {"status": "error", "message": str(e), "conversation_id": conversation_id}

async def handle_conversation_created(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle conversation_created event from Chatwoot
    
    Args:
        payload: Webhook payload
        
    Returns:
        Response dictionary
    """
    conversation = payload.get("conversation", {})
    account_id = payload.get("account", {}).get("id")
    conversation_id = conversation.get("id")
    
    logger.info(f"New conversation created: {conversation_id}")
    
    # TODO: Implement conversation routing logic
    return {"status": "processed", "conversation_id": conversation_id}

async def handle_conversation_status_changed(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle conversation_status_changed event from Chatwoot
    
    Args:
        payload: Webhook payload
        
    Returns:
        Response dictionary
    """
    conversation = payload.get("conversation", {})
    status = conversation.get("status")
    
    logger.info(f"Conversation status changed to: {status}")
    
    # TODO: Implement status-specific handling
    return {"status": "processed", "conversation_status": status}

async def assign_to_human_agent(account_id: int, conversation_id: int, agent_type: AgentType) -> None:
    """
    Assign conversation to appropriate human agent based on type
    
    Args:
        account_id: Chatwoot account ID
        conversation_id: Conversation ID to assign
        agent_type: Type of human agent needed
    """
    # Map agent types to agent IDs (these would come from configuration)
    agent_id_map = {
        AgentType.HUMAN_SALES: int(os.getenv("CHATWOOT_SALES_AGENT_ID", "0")),
        AgentType.HUMAN_SUPPORT: int(os.getenv("CHATWOOT_SUPPORT_AGENT_ID", "0")),
        AgentType.HUMAN_MANAGER: int(os.getenv("CHATWOOT_MANAGER_AGENT_ID", "0")),
    }
    
    # Get the appropriate agent ID
    agent_id = agent_id_map.get(agent_type, 0)
    
    if agent_id > 0:
        try:
            api = ChatwootAPI()
            logger.info(f"Assigning conversation {conversation_id} to human agent ID {agent_id}")
            api.assign_conversation(
                account_id=account_id,
                conversation_id=conversation_id,
                assignee_id=agent_id
            )
        except Exception as e:
            logger.error(f"Failed to assign conversation: {e}")
    else:
        logger.warning(f"No agent ID configured for agent type {agent_type.value}")
