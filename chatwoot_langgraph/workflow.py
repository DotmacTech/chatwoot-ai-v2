"""
LangGraph workflow module for Chatwoot Automation.
Defines the workflow graph and specialized agents.
"""
import os
import logging
from typing import Dict, Any, Optional, List, TypedDict, Annotated, Literal, Union
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from chatwoot_langsmith import tracing_manager
from chatwoot_langchain import intent_classifier, INTENT_CATEGORIES

# Configure logging
logger = logging.getLogger(__name__)

# Define state schema
class AgentState(TypedDict):
    """State for the agent workflow"""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    intent: Optional[str]
    customer_id: Optional[str]
    conversation_id: Optional[str]
    next_agent: Optional[str]
    final_response: Optional[str]

# Define agent types
class AgentType(str, Enum):
    """Types of specialized agents"""
    ROUTER = "router"
    SALES = "sales"
    SUPPORT = "support"
    ACCOUNT = "account"
    GENERAL = "general"

# Create specialized agents
def create_agent(agent_type: AgentType, model_name: str = "gpt-3.5-turbo", temperature: float = 0.2):
    """Create a specialized agent with appropriate system prompt"""
    
    system_prompts = {
        AgentType.ROUTER: "You are a routing agent that determines which specialized agent should handle the customer request.",
        AgentType.SALES: "You are a sales agent that helps customers with product information, pricing, and purchasing.",
        AgentType.SUPPORT: "You are a support agent that helps customers troubleshoot issues and resolve technical problems.",
        AgentType.ACCOUNT: "You are an account management agent that helps customers with account-related inquiries.",
        AgentType.GENERAL: "You are a general information agent that provides basic information and assistance."
    }
    
    system_prompt = system_prompts.get(agent_type, system_prompts[AgentType.GENERAL])
    
    return ChatOpenAI(
        model=model_name,
        temperature=temperature
    ).bind(system=system_prompt)

# Define workflow nodes
def router(state: AgentState) -> AgentState:
    """Route the conversation to the appropriate specialized agent based on intent"""
    messages = state["messages"]
    
    # Extract the latest customer message
    latest_message = None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            latest_message = message.content
            break
    
    if not latest_message:
        return {**state, "next_agent": AgentType.GENERAL}
    
    # Classify intent using the existing intent classifier
    classification = intent_classifier.classify_intent(latest_message)
    intent = classification.get("intent", "general")
    confidence = classification.get("confidence", 0.0)
    
    # Map intent to agent type
    intent_to_agent = {
        "sales": AgentType.SALES,
        "support": AgentType.SUPPORT,
        "account": AgentType.ACCOUNT,
        "general": AgentType.GENERAL
    }
    
    next_agent = intent_to_agent.get(intent, AgentType.GENERAL)
    
    # Update state with intent and next agent
    return {
        **state,
        "intent": intent,
        "next_agent": next_agent
    }

def sales_agent(state: AgentState) -> AgentState:
    """Handle sales-related inquiries"""
    agent = create_agent(AgentType.SALES)
    messages = state["messages"]
    response = agent.invoke(messages)
    
    return {
        **state,
        "messages": messages + [response],
        "final_response": response.content
    }

def support_agent(state: AgentState) -> AgentState:
    """Handle support-related inquiries"""
    agent = create_agent(AgentType.SUPPORT)
    messages = state["messages"]
    response = agent.invoke(messages)
    
    return {
        **state,
        "messages": messages + [response],
        "final_response": response.content
    }

def account_agent(state: AgentState) -> AgentState:
    """Handle account-related inquiries"""
    agent = create_agent(AgentType.ACCOUNT)
    messages = state["messages"]
    response = agent.invoke(messages)
    
    return {
        **state,
        "messages": messages + [response],
        "final_response": response.content
    }

def general_agent(state: AgentState) -> AgentState:
    """Handle general inquiries"""
    agent = create_agent(AgentType.GENERAL)
    messages = state["messages"]
    response = agent.invoke(messages)
    
    return {
        **state,
        "messages": messages + [response],
        "final_response": response.content
    }

# Define the routing logic
def determine_next_agent(state: AgentState) -> str:
    """Determine which agent should handle the request next"""
    next_agent = state.get("next_agent")
    
    if not next_agent:
        return AgentType.GENERAL
    
    return next_agent

# Build the workflow graph
def build_workflow_graph():
    """Build the workflow graph with specialized agents"""
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node(AgentType.ROUTER, router)
    workflow.add_node(AgentType.SALES, sales_agent)
    workflow.add_node(AgentType.SUPPORT, support_agent)
    workflow.add_node(AgentType.ACCOUNT, account_agent)
    workflow.add_node(AgentType.GENERAL, general_agent)
    
    # Add conditional edge from router to specialized agents
    workflow.add_conditional_edges(
        AgentType.ROUTER,
        determine_next_agent,
        {
            AgentType.SALES: AgentType.SALES,
            AgentType.SUPPORT: AgentType.SUPPORT,
            AgentType.ACCOUNT: AgentType.ACCOUNT,
            AgentType.GENERAL: AgentType.GENERAL
        }
    )
    
    # Connect specialized agents to END
    workflow.add_edge(AgentType.SALES, END)
    workflow.add_edge(AgentType.SUPPORT, END)
    workflow.add_edge(AgentType.ACCOUNT, END)
    workflow.add_edge(AgentType.GENERAL, END)
    
    # Set entry point
    workflow.set_entry_point(AgentType.ROUTER)
    
    # Compile the graph
    return workflow.compile()

# Create the workflow manager
class WorkflowManager:
    """Manages the LangGraph workflow for customer interactions"""
    
    def __init__(self):
        """Initialize the workflow manager"""
        self.workflow = build_workflow_graph()
        logger.info("LangGraph workflow initialized")
    
    def process_message(self, message: str, customer_id: str = None, conversation_id: str = None) -> Dict[str, Any]:
        """
        Process a customer message through the workflow
        
        Args:
            message: The customer message
            customer_id: ID of the customer
            conversation_id: ID of the conversation
            
        Returns:
            Dictionary with the processed response and metadata
        """
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "intent": None,
            "customer_id": customer_id,
            "conversation_id": conversation_id,
            "next_agent": None,
            "final_response": None
        }
        
        # Trace the workflow execution if tracing is enabled
        if tracing_manager.enabled:
            with tracing_manager.get_tracer(tags=["workflow", "customer_message"]) as tracer:
                # Execute the workflow
                result = self.workflow.invoke(initial_state)
                
                # Add trace metadata
                tracer.add_metadata({
                    "customer_id": customer_id,
                    "conversation_id": conversation_id,
                    "intent": result.get("intent"),
                    "agent_used": result.get("next_agent")
                })
        else:
            # Execute the workflow without tracing
            result = self.workflow.invoke(initial_state)
        
        return {
            "response": result.get("final_response"),
            "intent": result.get("intent"),
            "agent_used": result.get("next_agent"),
            "conversation_id": conversation_id
        }

# Create a singleton instance
workflow_manager = WorkflowManager()
