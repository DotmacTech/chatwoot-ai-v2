"""
LangGraph workflow module for Chatwoot Automation.
Defines the complete workflow with empathetic specialized agents.
"""
import os
import time
import logging
from typing import Dict, List, Any, Optional, Union, TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field
from chatwoot_langsmith import tracing_manager
from chatwoot_langchain.intent_classifier import IntentClassifier

logger = logging.getLogger(__name__)

# Define the agent state
class AgentState(TypedDict):
    """Full state for the agent workflow"""
    # Conversation data - using Annotated to handle multiple values correctly
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage]], "append_only"]
    full_conversation_history: List[Union[HumanMessage, AIMessage, SystemMessage]]
    
    # Customer and context info
    customer_id: Optional[str]
    customer_info: Optional[Dict[str, Any]]
    verified: bool
    verification_attempts: int
    contact_info: Optional[str]
    
    # Analysis and routing
    intent: Optional[str]
    sub_intent: Optional[str]
    intent_confidence: Optional[float]
    sentiment: Optional[Dict[str, Any]]
    entities: Optional[Dict[str, Any]]
    
    # Business data
    service_info: Optional[Dict[str, Any]]
    account_status: Optional[Dict[str, Any]]
    billing_data: Optional[Dict[str, Any]]
    
    # Flow control
    current_step: str
    next_step: Optional[str]
    needs_human_review: bool
    human_review_reason: Optional[str]
    final_response: Optional[str]
    completion_status: Optional[str]
    
    # Analytics and metrics
    processing_time: Optional[float]
    token_usage: Optional[Dict[str, int]]
    
    # New field to store the latest message for processing
    current_message: Optional[str]
    
    # Temporary field to store a new message before appending to messages
    new_message: Optional[Union[HumanMessage, AIMessage, SystemMessage]]

# Define workflow nodes
async def initial_analysis(state: AgentState) -> AgentState:
    """Process incoming message with sentiment analysis and entity extraction"""
    # Create trace for this operation
    trace_id = tracing_manager.create_trace(
        name="initial_analysis",
        inputs={"state": state},
        tags=["workflow", "analysis"]
    )
    
    try:
        logger.info("Starting initial analysis of customer message")
        
        # Get the current message from state
        message = state["current_message"]
        if not message:
            logger.warning("No current message found in state")
            return state
            
        # Create a copy of the state to avoid modifying the original
        updated_state = state.copy()
        
        # Create the new message but don't append it to messages yet
        # Store it in a temporary field for the next node to handle
        new_message = HumanMessage(content=message)
        updated_state["new_message"] = new_message
        
        # Do sentiment analysis and entity extraction if needed
        # For now, just setting placeholder values
        updated_state["sentiment"] = {"sentiment": "neutral", "intensity": 0.5}
        updated_state["entities"] = {}
        
        # Set next step
        updated_state["current_step"] = "append_message"
        
        # End trace with success
        tracing_manager.end_trace(trace_id, outputs={"state": updated_state})
        
        return updated_state
    except Exception as e:
        logger.error(f"Error in initial analysis: {str(e)}", exc_info=True)
        tracing_manager.end_trace(trace_id, error=str(e))
        
        # Return original state with error flag
        return {
            **state,
            "needs_human_review": True,
            "human_review_reason": f"Error in initial analysis: {str(e)}"
        }

async def append_message(state: AgentState) -> AgentState:
    """Append the new message to the messages list"""
    # Create trace for this operation
    trace_id = tracing_manager.create_trace(
        name="append_message",
        inputs={"state": state},
        tags=["workflow", "append"]
    )
    
    try:
        # Create a copy of the state to avoid modifying the original
        updated_state = state.copy()
        
        # Get the new message from the temporary field
        new_message = updated_state.pop("new_message", None)
        if not new_message:
            logger.warning("No new message found in state")
            return state
        
        # Append the new message to the messages list
        # Since messages is annotated as "append_only", we need to create a new list
        # and then assign it to updated_state["messages"]
        messages_copy = list(updated_state["messages"])
        messages_copy.append(new_message)
        updated_state["messages"] = messages_copy
        
        # Also update the full conversation history
        history_copy = list(updated_state["full_conversation_history"])
        history_copy.append(new_message)
        updated_state["full_conversation_history"] = history_copy
        
        # Set next step
        updated_state["current_step"] = "intent_classification"
        
        # End trace with success
        tracing_manager.end_trace(trace_id, outputs={"state": updated_state})
        
        return updated_state
    except Exception as e:
        logger.error(f"Error in append_message: {str(e)}", exc_info=True)
        tracing_manager.end_trace(trace_id, error=str(e))
        
        # Return original state with error flag
        return {
            **state,
            "needs_human_review": True,
            "human_review_reason": f"Error in append_message: {str(e)}"
        }

async def intent_classification(state: AgentState) -> AgentState:
    """Classify the customer's intent and sub-intent"""
    # Create trace for this operation
    trace_id = tracing_manager.create_trace(
        name="intent_classification",
        inputs={"state": state},
        tags=["workflow", "intent"]
    )
    
    try:
        logger.info("Classifying customer intent")
        
        # Create a copy of the state to avoid modifying the original
        updated_state = state.copy()
        
        # Get the current message
        message = state["current_message"]
        if not message:
            logger.warning("No current message found in state")
            return state
        
        # Use the intent classifier to determine intent
        # This is a placeholder for now
        intent = "general_inquiry"
        sub_intent = "information_request"
        confidence = 0.85
        
        # Update state with intent information
        updated_state["intent"] = intent
        updated_state["sub_intent"] = sub_intent
        updated_state["intent_confidence"] = confidence
        
        # Determine if human review is needed based on confidence
        if confidence < 0.6:
            updated_state["needs_human_review"] = True
            updated_state["human_review_reason"] = "Low confidence in intent classification"
        
        # End trace with success
        tracing_manager.end_trace(trace_id, outputs={"state": updated_state})
        
        return updated_state
    except Exception as e:
        logger.error(f"Error in intent classification: {str(e)}", exc_info=True)
        tracing_manager.end_trace(trace_id, error=str(e))
        
        # Return original state with error flag
        return {
            **state,
            "needs_human_review": True,
            "human_review_reason": f"Error in intent classification: {str(e)}"
        }

async def specialized_agent(state: AgentState) -> AgentState:
    """Process the customer request with a specialized agent based on intent"""
    # Create trace for this operation
    trace_id = tracing_manager.create_trace(
        name="specialized_agent",
        inputs={"state": state},
        tags=["workflow", "agent"]
    )
    
    try:
        logger.info("Processing with specialized agent")
        
        # Create a copy of the state to avoid modifying the original
        updated_state = state.copy()
        
        # Get conversation history for context
        history = state.get("full_conversation_history", [])
        
        # Generate response
        try:
            # Create a system prompt based on intent
            intent = state.get("intent", "general_inquiry")
            
            system_prompt = "You are a helpful AI assistant for our company."
            if intent == "technical_support":
                system_prompt = "You are a technical support specialist. Be precise and helpful."
            elif intent == "billing_inquiry":
                system_prompt = "You are a billing specialist. Be clear about policies and helpful."
            elif intent == "complaint":
                system_prompt = "You are a customer service specialist. Be empathetic and solution-oriented."
            
            # Set up the model - using DeepSeek instead of OpenAI
            try:
                model = ChatDeepSeek(
                    model=os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat"),
                    temperature=0.7,
                    openai_api_key=os.getenv("DEEPSEEK_API_KEY")
                )
                logger.info(f"Using DeepSeek model: {os.getenv('DEEPSEEK_MODEL_NAME', 'deepseek-chat')}")
            except Exception as model_error:
                logger.error(f"Error initializing DeepSeek model: {str(model_error)}")
                # Create a simple fallback response without using an external model
                updated_state["final_response"] = "I'm sorry, I'm having trouble accessing our AI service right now. Let me connect you with a human agent who can help."
                updated_state["needs_human_review"] = True
                updated_state["human_review_reason"] = f"Error initializing model: {str(model_error)}"
                
                # Create a new AI message but don't append it yet
                ai_message = AIMessage(content=updated_state["final_response"])
                updated_state["new_message"] = ai_message
                updated_state["current_step"] = "append_response"
                
                # End trace and return early
                tracing_manager.end_trace(trace_id, outputs={"state": updated_state})
                return updated_state
            
            # Format history for the prompt - ensure proper interleaving for DeepSeek
            # DeepSeek requires alternating human/AI messages
            
            # First, categorize messages by type
            human_messages = []
            ai_messages = []
            system_messages = []
            
            for msg in history:
                if isinstance(msg, HumanMessage):
                    human_messages.append(msg)
                elif isinstance(msg, AIMessage):
                    ai_messages.append(msg)
                elif isinstance(msg, SystemMessage):
                    system_messages.append(msg)
            
            # Create a properly interleaved history
            formatted_history = []
            
            # Add system messages first (if any)
            formatted_history.extend(system_messages)
            
            # Interleave human and AI messages
            # If we have more of one type than the other, we'll use only the most recent ones
            # to ensure proper alternation
            max_pairs = min(len(human_messages), len(ai_messages))
            
            # If we have human messages but no AI messages, we'll just use the human messages
            if max_pairs == 0 and human_messages:
                # Only include the most recent human message to avoid consecutive human messages
                formatted_history.append(human_messages[-1])
            else:
                # Interleave the messages, starting with human (as per DeepSeek's requirements)
                for i in range(max_pairs):
                    # Use negative indices to get the most recent messages first
                    h_idx = -(max_pairs - i)
                    a_idx = -(max_pairs - i)
                    
                    formatted_history.append(human_messages[h_idx])
                    formatted_history.append(ai_messages[a_idx])
            
            # Log the formatted history for debugging
            logger.debug(f"Formatted history for DeepSeek: {len(formatted_history)} messages")
            for i, msg in enumerate(formatted_history):
                logger.debug(f"Message {i}: {type(msg).__name__}")
            
            # Create a custom prompt for DeepSeek
            messages = []
            
            # Add the system message
            messages.append(SystemMessage(content=system_prompt))
            
            # Add the formatted history
            messages.extend(formatted_history)
            
            # Add the current message
            messages.append(HumanMessage(content=state["current_message"]))
            
            # Generate the response directly with the model
            response = await model.ainvoke(messages)
            
            # Extract the content from the response
            response_content = response.content
            
            # Create a new AI message but don't append it yet
            ai_message = AIMessage(content=response_content)
            updated_state["new_message"] = ai_message
            updated_state["final_response"] = response_content
            
            # Set next step
            updated_state["current_step"] = "append_response"
            
        except Exception as agent_error:
            logger.error(f"Error generating response: {str(agent_error)}", exc_info=True)
            updated_state["needs_human_review"] = True
            updated_state["human_review_reason"] = f"Error generating response: {str(agent_error)}"
            updated_state["final_response"] = "I'm sorry, I encountered an issue processing your request. Let me connect you with a human agent who can help."
            
            # Create a new AI message but don't append it yet
            ai_message = AIMessage(content=updated_state["final_response"])
            updated_state["new_message"] = ai_message
        
        # End trace with success
        tracing_manager.end_trace(trace_id, outputs={"state": updated_state})
        
        return updated_state
    except Exception as e:
        logger.error(f"Error in specialized agent: {str(e)}", exc_info=True)
        tracing_manager.end_trace(trace_id, error=str(e))
        
        # Return original state with error flag
        return {
            **state,
            "needs_human_review": True,
            "human_review_reason": f"Error in specialized agent: {str(e)}",
            "final_response": "I'm sorry, I encountered an issue processing your request. Let me connect you with a human agent who can help."
        }

async def append_response(state: AgentState) -> AgentState:
    """Append the AI response to the messages list"""
    # Create trace for this operation
    trace_id = tracing_manager.create_trace(
        name="append_response",
        inputs={"state": state},
        tags=["workflow", "append"]
    )
    
    try:
        # Create a copy of the state to avoid modifying the original
        updated_state = state.copy()
        
        # Get the new message from the temporary field
        new_message = updated_state.pop("new_message", None)
        if not new_message:
            logger.warning("No new AI message found in state")
            return state
        
        # Append the new message to the messages list
        # Since messages is annotated as "append_only", we need to create a new list
        # and then assign it to updated_state["messages"]
        messages_copy = list(updated_state["messages"])
        messages_copy.append(new_message)
        updated_state["messages"] = messages_copy
        
        # Also update the full conversation history
        history_copy = list(updated_state["full_conversation_history"])
        history_copy.append(new_message)
        updated_state["full_conversation_history"] = history_copy
        
        # Set completion status
        updated_state["completion_status"] = "completed"
        
        # End trace with success
        tracing_manager.end_trace(trace_id, outputs={"state": updated_state})
        
        return updated_state
    except Exception as e:
        logger.error(f"Error in append_response: {str(e)}", exc_info=True)
        tracing_manager.end_trace(trace_id, error=str(e))
        
        # Return original state with error flag
        return {
            **state,
            "needs_human_review": True,
            "human_review_reason": f"Error in append_response: {str(e)}"
        }

async def human_handoff(state: AgentState) -> AgentState:
    """Prepare for handoff to human agent"""
    # Create trace for this operation
    trace_id = tracing_manager.create_trace(
        name="human_handoff",
        inputs={"state": state},
        tags=["workflow", "handoff"]
    )
    
    try:
        logger.info("Preparing for human handoff")
        
        # Create a copy of the state to avoid modifying the original
        updated_state = state.copy()
        
        # Generate a handoff message
        handoff_reason = state.get("human_review_reason", "Automated handoff to human agent")
        handoff_message = f"I'll connect you with a human agent who can better assist you. Reason: {handoff_reason}"
        
        # Set the final response
        updated_state["final_response"] = handoff_message
        
        # Create a new AI message but don't append it yet
        ai_message = AIMessage(content=handoff_message)
        updated_state["new_message"] = ai_message
        
        # Set next step
        updated_state["current_step"] = "append_response"
        
        # End trace with success
        tracing_manager.end_trace(trace_id, outputs={"state": updated_state})
        
        return updated_state
    except Exception as e:
        logger.error(f"Error in human handoff: {str(e)}", exc_info=True)
        tracing_manager.end_trace(trace_id, error=str(e))
        
        # Return original state with error flag
        return {
            **state,
            "final_response": "I'm connecting you with a human agent who can help you further.",
            "needs_human_review": True,
            "human_review_reason": f"Error in human handoff: {str(e)}"
        }

def build_workflow_graph() -> StateGraph:
    """Build the workflow graph with all nodes and edges"""
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("initial_analysis", initial_analysis)
    workflow.add_node("append_message", append_message)
    workflow.add_node("intent_classification", intent_classification)
    workflow.add_node("specialized_agent", specialized_agent)
    workflow.add_node("append_response", append_response)
    workflow.add_node("human_handoff", human_handoff)
    
    # Define conditional routing based on state
    def should_route_to_human(state: AgentState) -> str:
        """Determine if we should route to human agent"""
        if state.get("needs_human_review", False):
            return "human_handoff"
        return "specialized_agent"
    
    # Add START edge to the initial node
    workflow.add_edge(START, "initial_analysis")
    
    # Add edges between nodes
    workflow.add_edge("initial_analysis", "append_message")
    workflow.add_edge("append_message", "intent_classification")
    workflow.add_conditional_edges(
        "intent_classification",
        should_route_to_human
    )
    
    # Add edge from specialized agent to append_response
    workflow.add_edge("specialized_agent", "append_response")
    
    # Add edge from human handoff to append_response
    workflow.add_edge("human_handoff", "append_response")
    
    # Add edge from append_response to END
    workflow.add_edge("append_response", END)
    
    return workflow

class WorkflowManager:
    """Manages the LangGraph workflow for customer interactions"""
    
    def __init__(self):
        """Initialize the workflow manager"""
        try:
            self.workflow = build_workflow_graph()
            # Compile the workflow - this is required for newer LangGraph versions
            if hasattr(self.workflow, 'compile'):
                logger.info("Compiling LangGraph workflow...")
                self.workflow = self.workflow.compile()
            logger.info("LangGraph workflow initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LangGraph workflow: {str(e)}", exc_info=True)
            # Still create the instance but mark it as failed
            self.workflow = None
            self.initialization_error = str(e)
    
    async def process_message(self, 
                            message: str, 
                            customer_id: str = None, 
                            conversation_id: str = None,
                            contact_info: str = None, 
                            prev_messages: List = None) -> Dict[str, Any]:
        """
        Process a customer message through the workflow
        
        Args:
            message: The customer message
            customer_id: ID of the customer (if known)
            conversation_id: ID of the conversation
            contact_info: Contact information (email/phone)
            prev_messages: Previous messages in the conversation
            
        Returns:
            Dictionary with the processed response and metadata
        """
        # Create trace for this operation
        trace_id = tracing_manager.create_trace(
            name="process_message",
            inputs={
                "message": message,
                "customer_id": customer_id,
                "conversation_id": conversation_id
            },
            tags=["workflow", "process"]
        )
        
        try:
            # Check if workflow was properly initialized
            if self.workflow is None:
                error_msg = f"Workflow not initialized properly: {self.initialization_error}"
                logger.error(error_msg)
                return {
                    "response": "I'm sorry, I couldn't process your request due to a technical issue with the workflow engine.",
                    "error": error_msg,
                    "conversation_id": conversation_id
                }
            
            start_time = time.time()
            
            # Convert previous messages to the correct format if needed
            formatted_prev_messages = []
            if prev_messages:
                for msg in prev_messages:
                    if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
                        formatted_prev_messages.append(msg)
                    elif isinstance(msg, dict) and "content" in msg:
                        if msg.get("type") == "human":
                            formatted_prev_messages.append(HumanMessage(content=msg["content"]))
                        elif msg.get("type") == "ai":
                            formatted_prev_messages.append(AIMessage(content=msg["content"]))
                        elif msg.get("type") == "system":
                            formatted_prev_messages.append(SystemMessage(content=msg["content"]))
                    else:
                        logger.warning(f"Skipping invalid message format: {msg}")
            
            # Prepare the initial state
            initial_state = AgentState(
                messages=formatted_prev_messages,
                full_conversation_history=formatted_prev_messages.copy(),
                current_message=message,  
                customer_id=customer_id,
                contact_info=contact_info,
                current_step="initial",
                verified=False,
                verification_attempts=0,
                needs_human_review=False,
                human_review_reason=None,
                new_message=None  # Initialize the temporary field
            )
            
            # Process the message through the workflow
            logger.debug(f"Invoking workflow with message: {message[:50]}...")
            
            # Detect available methods on the workflow object
            has_ainvoke = hasattr(self.workflow, 'ainvoke')
            has_invoke = hasattr(self.workflow, 'invoke')
            
            logger.debug(f"Available workflow methods - ainvoke: {has_ainvoke}, invoke: {has_invoke}")
            
            # Try different methods based on availability
            try:
                # First try the async method (preferred for newer LangGraph versions)
                if has_ainvoke:
                    logger.info("Using ainvoke method")
                    result = await self.workflow.ainvoke(initial_state)
                # If that fails, try the sync method
                elif has_invoke:
                    logger.info("Using synchronous invoke method")
                    result = self.workflow.invoke(initial_state)
                else:
                    error_msg = "No compatible workflow invocation method found"
                    logger.error(error_msg)
                    return {
                        "response": "I'm sorry, I couldn't process your request due to a technical issue with the workflow engine.",
                        "error": error_msg,
                        "conversation_id": conversation_id
                    }
            except Exception as e:
                error_msg = f"Error invoking workflow: {str(e)}"
                logger.error(error_msg, exc_info=True)
                tracing_manager.end_trace(trace_id, error=str(e))
                return {
                    "response": "I'm sorry, I couldn't process your request due to a technical issue with the workflow engine.",
                    "error": error_msg,
                    "conversation_id": conversation_id
                }
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Extract the final response
            final_response = result.get("final_response", "I'm sorry, I couldn't generate a response.")
            
            # Prepare the result
            response_data = {
                "response": final_response,
                "conversation_id": conversation_id,
                "processing_time": processing_time,
                "needs_human_review": result.get("needs_human_review", False),
                "intent": result.get("intent"),
                "sentiment": result.get("sentiment")
            }
            
            # End trace with success
            tracing_manager.end_trace(trace_id, outputs=response_data)
            
            return response_data
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg, exc_info=True)
            tracing_manager.end_trace(trace_id, error=str(e))
            
            return {
                "response": "I'm sorry, I couldn't process your request due to a technical issue.",
                "error": error_msg,
                "conversation_id": conversation_id
            }

# Create a singleton instance
workflow_manager = WorkflowManager()