"""
LangGraph workflow module for Chatwoot Automation.
Defines the complete workflow with empathetic specialized agents.
"""
import logging
from typing import List, Dict, Any, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field
from services.langsmith_service import LangSmithService
from services.llm_service import LLMService # Import the LLM Service
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.messages import SystemMessage
import os
from itertools import zip_longest

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """Represents the state of our workflow."""
    messages: List[BaseMessage] 
    current_message: Optional[str] 
    conversation_id: int
    account_id: int
    intent: Optional[str] 
    final_response: Optional[str] 
    current_step: str 
    needs_human_review: bool
    human_review_reason: Optional[str]
    langsmith_service: Optional[LangSmithService] # Add service to state
    llm_service: LLMService # Add LLM service to state
    parent_trace_id: Optional[str] # Add parent trace ID to state

# Node functions 
async def initial_analysis(state: AgentState) -> dict:
    """Perform initial analysis, now only focusing on the current message if needed.
       History is assumed to be already in the state.
    """
    logger.info(f"[Conv ID: {state.get('conversation_id')}] Running initial analysis...")
    analysis_result = {"current_step": "initial_analysis"}
    return analysis_result

async def append_message(state: AgentState) -> AgentState:
    """Append the new message to the messages list"""
    try:
        current_msg_content = state.get("current_message")
        if not current_msg_content:
            logger.warning("No current message content found to append.")
            return {"messages": state['messages']}
        
        new_message = HumanMessage(content=current_msg_content)
        updated_messages = state['messages'] + [new_message]
        logger.info(f"[Conv ID: {state.get('conversation_id')}] Appended HumanMessage: '{current_msg_content[:50]}...'")
        return {"messages": updated_messages}
    except Exception as e:
        logger.error(f"Error in append_message: {e}", exc_info=True)
        return {"messages": state.get("messages", [])}

async def classify_intent(state: AgentState) -> Dict[str, Any]:
    """
    Classifies the user's intent based on the latest message and conversation history.
    Uses a dedicated prompt and the LLM client.
    """
    logger.info(f"Classifying intent for conv {state['conversation_id']}...")
    llm_service = state["llm_service"]
    langsmith_service = state.get("langsmith_service") # Optional
    conversation_id = state["conversation_id"]
    
    # Combine history and the latest user message
    history = state.get("messages", [])
    latest_message = state["current_message"] # Assuming input holds the latest message content string
    
    # Ensure input is a HumanMessage if it's just a string
    # Note: This assumes 'input' is the raw text. The history should already be BaseMessages.
    if isinstance(latest_message, str):
        # Need to decide if we create a new HumanMessage or use the last one in history
        # Let's assume history already includes the latest message from the trigger.
        # If not, we might need to add: history.append(HumanMessage(content=latest_message))
        # For now, assume 'history' passed to this node is complete.
        pass # Assuming history is correct
        
    history_plus_current = history # Use the history as is, assuming it includes the latest message

    trace_id = None
    langsmith_context = None
    callbacks = []

    if not llm_service or not llm_service.is_available():
         logger.error(f"LLMService not available for intent classification in conv {conversation_id}.")
         return {"intent": "error", "error_message": "LLMService not available"}

    # Prepare tracing if LangSmith is enabled
    if langsmith_service:
        # Simplified trace setup for classification
        trace_inputs = {"messages": [msg.to_json() for msg in history_plus_current]}
        trace_tags = ["intent_classification", conversation_id]
        trace_metadata = {"conversation_id": conversation_id}

        langsmith_context = langsmith_service.trace(
            name="Intent Classification",
            run_type="llm", # It's an LLM call
            inputs=trace_inputs,
            tags=trace_tags,
            metadata=trace_metadata,
            trace_id_param_name='trace_id' 
        )
        trace_id = langsmith_context.__enter__()
        callbacks = [langsmith_service.get_callback_handler(trace_id)] if trace_id else []

    intent_str = "error" # Default
    error_message = None
    try:
        # Define the classification prompt
        # TODO: Define intents more formally (e.g., general_query, request_human, feedback, etc.)
        classification_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an intent classification system. Based on the conversation history, classify the latest user message into one of the following intents: general_query, request_human, irrelevant. Respond with ONLY the intent name."),
            MessagesPlaceholder(variable_name="history"),
            # ("human", "{latest_message_content}") # Assuming history includes the last message
        ])
        
        # Create the chain for classification
        classification_chain = classification_prompt_template | llm_service.client | StrOutputParser()
        
        logger.debug(f"Invoking classification chain for conv {conversation_id}", extra={"trace_id": trace_id})
        
        # Invoke the chain
        # Pass the history which should include the latest message
        intent_str = await classification_chain.ainvoke(
             {"history": history_plus_current}, 
             config={"callbacks": callbacks}
        )
        intent_str = intent_str.strip().lower() # Normalize
        logger.info(f"Classified intent for conv {conversation_id}: {intent_str}", extra={"trace_id": trace_id})

        # Validate intent (optional but good practice)
        valid_intents = ["general_query", "request_human", "irrelevant"]
        if intent_str not in valid_intents:
            logger.warning(f"LLM returned unexpected intent '{intent_str}' for conv {conversation_id}. Defaulting to 'general_query'.", extra={"trace_id": trace_id})
            intent_str = "general_query" # Fallback

        if langsmith_service and langsmith_context:
             outputs = {"intent": intent_str}
             langsmith_context.__exit__(None, None, None) # Exit trace successfully
             # langsmith_service.end_trace(trace_id, outputs=outputs) # Assuming exit handles output logging

    except Exception as e:
        error_type = type(e).__name__
        error_message = f"Error during intent classification: {error_type}: {e}"
        logger.error(f"{error_message} for conv {conversation_id}", exc_info=True, extra={"trace_id": trace_id})
        intent_str = "error"
        if langsmith_service and langsmith_context:
            langsmith_context.__exit__(type(e), e, e.__traceback__) # Exit trace with error
            # langsmith_service.end_trace(trace_id, error=str(e)) # Assuming exit handles error logging
    finally:
        # Ensure context manager is exited
        if langsmith_service and langsmith_context and not langsmith_context._is_exited():
             try:
                 langsmith_context.__exit__(None, None, None)
             except Exception as exit_e:
                 logger.error(f"Error exiting Langsmith trace context for classification: {exit_e}", extra={"trace_id": trace_id})


    return {"intent": intent_str, "error_message": error_message if intent_str == "error" else None}

async def generate_response(state: AgentState) -> Dict[str, Any]:
    """
    Generates a response based on the conversation history using the LLM.
    """
    logger.info(f"Generating response for conv {state['conversation_id']}...")
    llm_service = state["llm_service"]
    langsmith_service = state.get("langsmith_service") # Optional
    conversation_id = state["conversation_id"]
    history = state.get("messages", [])

    trace_id = None
    langsmith_context = None
    callbacks = []

    if not llm_service or not llm_service.is_available():
         logger.error(f"LLMService not available for response generation in conv {conversation_id}.")
         # Decide how to handle this - maybe return a generic error message?
         return {
             "response": "Apologies, I'm currently unable to generate a response. Please try again later.", 
             "error_message": "LLMService not available",
             "final_response": "Apologies, I'm currently unable to generate a response. Please try again later."
         }

    # Prepare tracing if LangSmith is enabled
    if langsmith_service:
        trace_inputs = {"messages": [msg.to_json() for msg in history]}
        trace_tags = ["response_generation", conversation_id]
        trace_metadata = {"conversation_id": conversation_id}

        langsmith_context = langsmith_service.trace(
            name="Response Generation",
            run_type="llm", # It's an LLM call
            inputs=trace_inputs,
            tags=trace_tags,
            metadata=trace_metadata,
            trace_id_param_name='trace_id' 
        )
        trace_id = langsmith_context.__enter__()
        callbacks = [langsmith_service.get_callback_handler(trace_id)] if trace_id else []

    response_content = "I encountered an internal error. Please try again later." # Default error
    error_message = None
    try:
        # Define the response generation prompt
        # You might want a more sophisticated prompt or load it from elsewhere
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant interacting with a user."),
            MessagesPlaceholder(variable_name="history")
        ])

        # Create the generation chain - use llm_service.client here
        chain = prompt_template | llm_service.client | StrOutputParser()

        logger.debug(f"Invoking generation chain for conv {conversation_id}", extra={"trace_id": trace_id})
        
        # Invoke the chain
        response_content = await chain.ainvoke(
            {"history": history}, 
            config={"callbacks": callbacks}
        )
        
        logger.info(f"Generated response for conv {conversation_id}: {response_content[:50]}...", extra={"trace_id": trace_id})

        if langsmith_service and langsmith_context:
             outputs = {"response": response_content}
             langsmith_context.__exit__(None, None, None) # Exit trace successfully
             # langsmith_service.end_trace(trace_id, outputs=outputs) # Assuming exit handles output

    except Exception as e:
        error_type = type(e).__name__
        error_message = f"Error generating response: {error_type}: {e}"
        logger.error(f"{error_message} for conv {conversation_id}", exc_info=True, extra={"trace_id": trace_id})
        # Keep default error message in response_content
        if langsmith_service and langsmith_context:
            langsmith_context.__exit__(type(e), e, e.__traceback__) # Exit trace with error
            # langsmith_service.end_trace(trace_id, error=str(e)) # Assuming exit handles error
    finally:
         # Ensure context manager is exited
        if langsmith_service and langsmith_context and not langsmith_context._is_exited():
             try:
                 langsmith_context.__exit__(None, None, None)
             except Exception as exit_e:
                 logger.error(f"Error exiting Langsmith trace context for generation: {exit_e}", extra={"trace_id": trace_id})


    return {
        "response": response_content, 
        "error_message": error_message,
        "final_response": response_content  # Add the final_response key with the same content
    }

async def specialized_agent(state: AgentState) -> dict:
    """Process the customer request with a specialized agent based on intent"""
    intent = state.get('intent')
    conversation_id = state.get('conversation_id')
    logger.info(f"[Conv ID: {conversation_id}] Running specialized agent for intent: {intent}...")

    try:
        if intent == "order_status":
            response_text = "To check your order status, please provide your order number."
        elif intent == "refund_request":
            response_text = "I can help with refund requests. Please provide your order number and reason for the refund."
        elif intent == "general_support":
            response_text = "How can I assist you further with general support?"
        elif intent == "human_handoff_request":
            response_text = "Connecting you to a human agent..." 
            output = {
                "final_response": response_text, 
                "current_step": "specialized_agent",
                "needs_human_review": True, 
                "human_review_reason": "User requested handoff"
            }
            return output
        elif intent == "general_inquiry":
            response_text = "Thank you for your inquiry. How can I help you specifically?"
        elif intent == "unknown":
             response_text = "I'm not sure how to help with that. Could you please rephrase?"
        elif intent == "error":
             response_text = "I encountered an error trying to understand your request. Please try again or ask for a human agent."
             state['needs_human_review'] = True 
             state['human_review_reason'] = "Intent classification error"
        else:
            response_text = "I'm sorry, I didn't understand that. Can you please clarify?"
            intent = "fallback"
            state['intent'] = intent 

        logger.info(f"[Conv ID: {conversation_id}] Generated response: '{response_text[:50]}...'")
        
        output = {
            "final_response": response_text, 
            "current_step": "specialized_agent",
            "needs_human_review": state.get('needs_human_review', False), 
            "human_review_reason": state.get('human_review_reason') 
        }
        return output

    except Exception as e:
        logger.error(f"Error in specialized agent for intent {intent}: {e}", exc_info=True)
        return {
            "final_response": "I encountered an internal error. A human agent will assist you shortly.", 
            "current_step": "specialized_agent", 
            "needs_human_review": True, 
            "human_review_reason": f"Agent error: {e}"
        }

async def append_response(state: AgentState) -> AgentState:
    """Append the AI response to the messages list"""
    try:
        final_response_content = state.get("final_response")
        if not final_response_content:
            logger.warning("No final response content found to append.")
            return {"messages": state['messages']}
        
        ai_message = AIMessage(content=final_response_content)
        updated_messages = state["messages"] + [ai_message]
        logger.info(f"[Conv ID: {state.get('conversation_id')}] Appended AIMessage: '{final_response_content[:50]}...'")
        return {"messages": updated_messages}
    except Exception as e:
        logger.error(f"Error in append_response: {e}", exc_info=True)
        return {"messages": state.get("messages", [])}

# --- Conditional Edge Logic --- 
def should_route_to_human(state: AgentState) -> str:
    """Determines if the conversation needs human review."""
    intent = state.get('intent')
    needs_review = state.get('needs_human_review', False)
    
    logger.info(f"[Conv ID: {state.get('conversation_id')}] Checking routing: Intent='{intent}', NeedsReview={needs_review}")
    
    if intent == "human_handoff_request" or needs_review:
        logger.info("Routing to human_handoff.")
        return "human_handoff"
    else:
        logger.info("Routing to append_response.")
        return "append_response"
# --------------------------

async def human_handoff(state: AgentState) -> dict:
    """Prepare for handoff to human agent"""
    conversation_id = state.get('conversation_id')
    reason = state.get('human_review_reason', 'Needs human assistance')
    logger.info(f"[Conv ID: {conversation_id}] Preparing for human handoff. Reason: {reason}")
    
    handoff_message = f"Handoff triggered for conversation {conversation_id}. Reason: {reason}"
    logger.info(handoff_message)
    
    final_response = state.get("final_response")
    if not final_response:
        final_response = "Connecting you to a human agent now."
 
    output = {
        "final_response": final_response,
        "current_step": "human_handoff",
        "needs_human_review": True
    }
    # End trace here if tracing added
    return output


# --- Build Workflow Graph --- 
def build_workflow_graph(): 
    """Build the workflow graph with all nodes and edges"""
    workflow = StateGraph(AgentState)

    workflow.add_node("initial_analysis", initial_analysis)
    workflow.add_node("append_message", append_message)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("specialized_agent", specialized_agent)
    workflow.add_node("append_response", append_response)
    workflow.add_node("human_handoff", human_handoff)

    workflow.add_edge(START, "append_message") 
    workflow.add_edge("append_message", "classify_intent")
    workflow.add_edge("classify_intent", "generate_response")

    workflow.add_conditional_edges(
        "generate_response",
        should_route_to_human,
        {
            "human_handoff": "human_handoff",
            "append_response": "append_response",
        }
    )
    
    workflow.add_edge("human_handoff", "append_response") 
    workflow.add_edge("append_response", END) 
    
    compiled_workflow = workflow.compile()
    
    logger.info("Workflow graph built and compiled.")
    return compiled_workflow
# --------------------------


class WorkflowManager:
    """Manages the LangGraph workflow for processing messages."""

    def __init__(self, llm_service: LLMService, langsmith_service: Optional[LangSmithService] = None): 
        """Initializes the WorkflowManager."""
        logger.info("Initializing WorkflowManager...")
        self.llm_service = llm_service
        self.langsmith_service = langsmith_service
        self.workflow = build_workflow_graph() 
        logger.info("Workflow graph built successfully.")

    async def process_message(
        self, 
        conversation_id: int,
        current_message: str,
        chatwoot_conversation: Dict, 
        chatwoot_message: Dict, 
        langsmith_service: Optional[LangSmithService], 
        llm_service: LLMService, 
        parent_trace_id: Optional[str] = None 
    ) -> Dict:
        """Process an incoming message through the LangGraph workflow."""
        initial_state: AgentState = {
            "messages": [],
            "current_message": current_message,
            "conversation_id": conversation_id,
            "account_id": chatwoot_conversation.get('account_id'), 
            "intent": None,
            "final_response": None,
            "current_step": START,
            "needs_human_review": False,
            "human_review_reason": None,
            "langsmith_service": langsmith_service, 
            "llm_service": llm_service, 
            "parent_trace_id": parent_trace_id 
        }
        
        try:
            formatted_history = []
            if chatwoot_conversation.get('messages'):
                for msg in chatwoot_conversation['messages']:
                    if isinstance(msg, dict):
                        if msg.get('message_type') == 'incoming':
                            formatted_history.append(HumanMessage(content=msg.get('content', '')))
                        elif msg.get('message_type') == 'outgoing':
                            sender = msg.get('sender')
                            sender_type = getattr(sender, 'type', None) if sender else None
                            if sender_type == 'agent_bot' or msg.get('sender_id') is None: 
                                 formatted_history.append(AIMessage(content=msg.get('content', '')))
                            else: 
                                formatted_history.append(HumanMessage(content=msg.get('content', '')))
                    elif isinstance(msg, HumanMessage):
                        formatted_history.append(msg)
                    elif isinstance(msg, AIMessage):
                        formatted_history.append(msg)
                    else:
                        logger.warning(f"Unknown message type in history: {type(msg)}")
                            
            initial_state["messages"] = formatted_history

            config = {"configurable": {"thread_id": str(conversation_id)}}
            
            final_state = await self.workflow.ainvoke(initial_state, config=config)
            
            logger.info(f"Workflow finished for conversation {conversation_id}.")
            logger.debug(f"Final state: {final_state}")
            
            if not final_state or "final_response" not in final_state:
                 logger.error("Workflow finished without producing a final response in the state.")
                 final_state = final_state or {}
                 final_state["final_response"] = "I'm sorry, I couldn't process your request properly. Please try again or contact support."
                 final_state["needs_human_review"] = True
                 final_state["human_review_reason"] = "Workflow did not produce a final response"
                 logger.info("Added default final response to state")

            return final_state

        except Exception as e:
            logger.error(f"Error processing message in workflow: {e}", exc_info=True)
            if langsmith_service and langsmith_service.enabled and parent_trace_id:
                langsmith_service.create_feedback(
                    run_id=parent_trace_id,
                    key="workflow_error",
                    score=0,
                    comment=f"Workflow execution failed: {e}",
                    source_info={"conversation_id": conversation_id}
                )
            return {
                "messages": initial_state['messages'], 
                "current_message": current_message,
                "conversation_id": conversation_id,
                "account_id": chatwoot_conversation.get('account_id'),
                "intent": "error",
                "final_response": "I encountered an internal error. Please try again later.",
                "current_step": "error",
                "needs_human_review": True,
                "human_review_reason": f"Workflow execution failed: {e}"
            }