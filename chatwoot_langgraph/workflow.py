"""
LangGraph workflow module for Chatwoot Automation.
Defines the complete workflow with empathetic specialized agents.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List, TypedDict, Union, Literal
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI as DeepSeekChat
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from chatwoot_langsmith import tracing_manager
from chatwoot_langchain import intent_classifier, INTENT_CATEGORIES
# Comment out external integrations
# from crm_integration import get_customer_info, update_customer_info
# from network_monitoring import get_network_status, check_service_address
# from billing_system import get_account_details, get_customer_invoices, get_payment_methods

# Configure logging
logger = logging.getLogger(__name__)

# Define state schema
class AgentState(TypedDict):
    """Full state for the agent workflow"""
    # Conversation data
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
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

# Define sentiment analysis model
class SentimentAnalysis(BaseModel):
    """Sentiment analysis of customer message"""
    emotional_tone: str = Field(description="The primary emotional tone (frustrated, happy, neutral, anxious, confused, angry)")
    urgency: int = Field(description="Urgency level from 1-5 where 5 is highest")
    sentiment_score: float = Field(description="Overall sentiment from -1.0 (negative) to 1.0 (positive)")
    key_concerns: List[str] = Field(description="List of main concerns or issues expressed")

# Define agent types
class AgentType(str, Enum):
    """Types of specialized agents"""
    ROUTER = "router"
    SALES = "sales"
    SUPPORT = "support"
    ACCOUNT = "account"
    GENERAL = "general"
    VERIFICATION = "verification"
    ONBOARDING = "onboarding"
    RETENTION = "retention"

# Define emotion-based response styles
EMOTION_RESPONSE_STYLES = {
    "frustrated": """Acknowledge their frustration immediately. Be brief, direct, and solution-focused. 
                     Avoid corporate language. Show understanding but focus on concrete next steps.""",
    
    "angry": """Remain calm and validating. Apologize sincerely if appropriate. Use phrases like 
                "I completely understand why this is frustrating" and focus on immediate resolution paths.""",
    
    "anxious": """Use a reassuring tone. Provide clear, step-by-step information. Avoid ambiguity.
                  Give timeframes when possible and set clear expectations about what will happen next.""",
    
    "confused": """Use simple, clear language. Break down complex concepts. Confirm understanding 
                   at each step and ask if further clarification is needed.""",
    
    "happy": """Match their positive energy. Be conversational and personable. Look for opportunities 
                to highlight additional benefits or services they might enjoy.""",
    
    "neutral": """Be professional and efficient while maintaining warmth. Balance thoroughness with brevity.
                  Focus on providing complete information without unnecessary details."""
}

# Create specialized agents
def create_agent(agent_type: AgentType, model_name: str = "deepseek-reasoner", temperature: float = 0.3, emotion: str = "neutral"):
    """Create a specialized agent with appropriate system prompt and emotional awareness"""
    
    # Base system prompts
    base_prompts = {
        AgentType.ROUTER: """You are an intelligent routing agent for an Internet Service Provider.
            Your job is to understand customer inquiries and route them to the appropriate specialized team.
            Carefully analyze the full context of the conversation to determine the most appropriate classification.""",
        
        AgentType.SALES: """You are a helpful, empathetic sales advisor for an Internet Service Provider.
            Your goal is to understand customer needs and help them find the perfect internet plan or upgrade.
            Provide accurate information about plans, pricing, promotions, and availability.
            Create a positive experience that builds trust while addressing their specific requirements.""",
        
        AgentType.SUPPORT: """You are a technical support specialist for an Internet Service Provider.
            Help customers troubleshoot and resolve internet connectivity issues with patience and clarity.
            Use a systematic approach to diagnose problems, starting with simple solutions before suggesting
            more complex fixes. Explain technical concepts in accessible language and provide step-by-step instructions.""",
        
        AgentType.ACCOUNT: """You are an account management specialist for an Internet Service Provider.
            Help customers with billing inquiries, account updates, service changes, and payment arrangements.
            Be detail-oriented and thorough while maintaining a friendly, helpful demeanor.
            Ensure customers understand their billing, services, and account options.""",
        
        AgentType.VERIFICATION: """You are a verification specialist for an Internet Service Provider.
            Your role is to verify customer identities securely and efficiently.
            Be thorough but respectful of customer privacy. Follow all security protocols
            while maintaining a helpful, patient demeanor.""",
        
        AgentType.ONBOARDING: """You are an onboarding specialist for an Internet Service Provider.
            Guide new customers through setting up their service, equipment installation, and initial configuration.
            Be thorough, patient, and provide clear step-by-step instructions.
            Ensure customers feel confident using their new service.""",
        
        AgentType.RETENTION: """You are a customer retention specialist for an Internet Service Provider.
            Your goal is to understand customer concerns that might lead to cancellation and find solutions
            that address their needs while maintaining their business. Be empathetic, solution-oriented,
            and focus on the value proposition of continued service.""",
        
        AgentType.GENERAL: """You are a helpful customer service agent for an Internet Service Provider.
            Provide accurate, helpful information about our services, policies, and procedures.
            Be warm, professional, and focused on resolving customer inquiries efficiently."""
    }
    
    # Get the base prompt for this agent type
    base_prompt = base_prompts.get(agent_type, base_prompts[AgentType.GENERAL])
    
    # Get the emotional response style
    emotion_style = EMOTION_RESPONSE_STYLES.get(emotion, EMOTION_RESPONSE_STYLES["neutral"])
    
    # Combine the base prompt with emotional guidance
    system_prompt = f"""{base_prompt}

EMOTIONAL CONTEXT:
The customer appears to be feeling {emotion}. {emotion_style}

COMMUNICATION GUIDELINES:
- Be empathetic and responsive to the customer's emotional state
- Use natural, conversational language rather than scripted responses
- Focus on understanding and resolving their specific situation
- Be precise with technical information but explain it clearly
- When you don't know something, be honest and explain how you'll help them get the information
- If the issue requires human intervention, acknowledge this respectfully

Always maintain a helpful, professional tone while addressing the customer's needs and emotional state.
"""
    
    # Check for API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        logger.error("DEEPSEEK_API_KEY environment variable not set!")
        raise ValueError("Missing API key for DeepSeek")
    
    return DeepSeekChat(
        model=model_name,
        temperature=temperature,
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com/v1",
        max_tokens=4096
    ).bind(system=system_prompt)

# Create sentiment analysis function
def analyze_sentiment(message: str) -> SentimentAnalysis:
    """Analyze the sentiment of a customer message"""
    sentiment_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at analyzing customer sentiment in messages.
                     Analyze the emotional tone, urgency, and overall sentiment of the message.
                     Identify key concerns or issues being expressed."""),
        ("human", "{message}")
    ])
    
    sentiment_model = DeepSeekChat(
        model="deepseek-reasoner",
        temperature=0.1,
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com/v1"
    )
    
    sentiment_chain = sentiment_prompt | sentiment_model | StrOutputParser()
    
    try:
        sentiment_json = sentiment_chain.invoke({"message": message})
        sentiment_data = json.loads(sentiment_json)
        return SentimentAnalysis(**sentiment_data)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        # Return default neutral sentiment
        return SentimentAnalysis(
            emotional_tone="neutral",
            urgency=2,
            sentiment_score=0.0,
            key_concerns=["general inquiry"]
        )

# Define workflow nodes
def initial_analysis(state: AgentState) -> AgentState:
    """Process incoming message with sentiment analysis and entity extraction"""
    # Find the latest user message
    latest_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            latest_message = msg.content
            break
    
    if not latest_message:
        return {**state, "next_step": "verification_check"}
    
    # Analyze sentiment
    sentiment = analyze_sentiment(latest_message)
    
    # Extract entities (simplified - would use NER in production)
    entities = {
        "account_number": None,
        "phone_number": None,
        "address": None,
        "service_type": None,
        "product_mention": None
    }
    
    # Quick message templates based on sentiment
    urgency_level = sentiment.urgency
    emotional_tone = sentiment.emotional_tone
    
    # Send immediate acknowledgment based on sentiment
    acknowledgments = {
        "angry": "I understand you're frustrated right now. I'm here to help resolve this situation as quickly as possible.",
        "frustrated": "I can see this has been challenging. Let me help you get this sorted out right away.",
        "anxious": "I understand your concern. Let me get the information you need to put your mind at ease.",
        "confused": "I'd be happy to clear things up for you. Let's go through this step by step.",
        "happy": "Thanks for your message! I'm glad to help you today.",
        "neutral": "Thank you for contacting our ISP support. I'll help you with your request."
    }
    
    acknowledgment = acknowledgments.get(emotional_tone, acknowledgments["neutral"])
    
    # If urgent, add urgency response
    if urgency_level >= 4:
        acknowledgment += " I can see this is urgent, so I'll prioritize your request."
    
    # Create the response
    response = AIMessage(content=acknowledgment)
    
    # Update state with all analysis
    return {
        **state,
        "messages": state["messages"] + [response],
        "sentiment": {
            "emotional_tone": emotional_tone,
            "urgency": urgency_level,
            "sentiment_score": sentiment.sentiment_score,
            "key_concerns": sentiment.key_concerns
        },
        "entities": entities,
        "next_step": "verification_check"
    }

def verification_check(state: AgentState) -> AgentState:
    """Check if the customer needs verification"""
    customer_id = state.get("customer_id")
    contact_info = state.get("contact_info")
    
    # If already verified, move on
    if state.get("verified"):
        return {**state, "next_step": "intent_classification"}
    
    # If we have customer ID but not verified, retrieve info
    if customer_id and not state.get("customer_info"):
        try:
            customer_info = get_customer_info(customer_id)
            if customer_info:
                return {
                    **state,
                    "customer_info": customer_info,
                    "verified": True,
                    "next_step": "intent_classification"
                }
        except Exception as e:
            logger.error(f"Error retrieving customer info: {str(e)}")
    
    # If we have contact info, try to find customer
    if contact_info and not customer_id:
        try:
            # This would call your CRM to find customers by contact
            customer_matches = find_customers_by_contact(contact_info)
            
            if len(customer_matches) == 1:
                # Single match found
                customer = customer_matches[0]
                return {
                    **state,
                    "customer_id": customer["id"],
                    "customer_info": customer,
                    "verified": True,
                    "next_step": "intent_classification"
                }
            elif len(customer_matches) > 1:
                # Multiple matches, need manual verification
                return {
                    **state,
                    "verified": False,
                    "needs_human_review": True,
                    "human_review_reason": "Multiple matching customer records",
                    "next_step": "verification_agent"
                }
        except Exception as e:
            logger.error(f"Error finding customer by contact: {str(e)}")
    
    # If no matches or errors, go to verification agent
    return {**state, "next_step": "verification_agent"}

def verification_agent(state: AgentState) -> AgentState:
    """Handle customer verification when needed"""
    # Use the verification agent with appropriate emotional context
    emotional_tone = state.get("sentiment", {}).get("emotional_tone", "neutral")
    agent = create_agent(AgentType.VERIFICATION, emotion=emotional_tone)
    
    # Add verification context to messages
    context_message = SystemMessage(content=f"""
    This customer needs verification. Current verification status: {state.get('verified', False)}
    Verification attempts: {state.get('verification_attempts', 0)}
    Contact info: {state.get('contact_info')}
    """)
    
    messages = state["messages"] + [context_message]
    response = agent.invoke(messages)
    
    # Update verification attempts
    verification_attempts = state.get("verification_attempts", 0) + 1
    
    # If too many attempts, escalate to human
    needs_human = verification_attempts >= 3
    human_reason = "Multiple failed verification attempts" if needs_human else state.get("human_review_reason")
    
    # Determine next step based on verification status
    next_step = "human_handoff" if needs_human else "intent_classification"
    
    return {
        **state,
        "messages": state["messages"] + [response],
        "verification_attempts": verification_attempts,
        "needs_human_review": needs_human or state.get("needs_human_review", False),
        "human_review_reason": human_reason,
        "next_step": next_step
    }

def intent_classification(state: AgentState) -> AgentState:
    """Classify the customer's intent and sub-intent"""
    # Find the latest user messages (use more context for better classification)
    user_messages = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_messages.append(msg.content)
    
    combined_message = " ".join(user_messages[-3:])  # Last 3 messages for context
    
    # Classify intent
    classification = intent_classifier.classify_intent(combined_message)
    intent = classification.get("intent", "general")
    sub_intent = classification.get("sub_intent")
    confidence = classification.get("confidence", 0.0)
    
    # Retrieve relevant data based on intent
    customer_id = state.get("customer_id")
    service_info = None
    account_status = None
    billing_data = None
    
    if customer_id and state.get("verified"):
        try:
            if intent == "support":
                # Get service info for support queries
                service_info = get_service_information(customer_id)
            
            elif intent == "account" or intent == "billing":
                # Get account and billing info
                account_status = get_account_details(customer_id)
                billing_data = get_customer_invoices(customer_id)
        except Exception as e:
            logger.error(f"Error retrieving customer data: {str(e)}")
    
    # Determine if human review is needed
    needs_human = state.get("needs_human_review", False)
    human_reason = state.get("human_review_reason")
    
    # Low confidence should trigger human review
    if confidence < 0.7 and not needs_human:
        needs_human = True
        human_reason = f"Low confidence intent classification: {intent} ({confidence:.2f})"
    
    # Determine next step based on intent and human review status
    next_step = "human_handoff" if needs_human else intent
    
    return {
        **state,
        "intent": intent,
        "sub_intent": sub_intent,
        "intent_confidence": confidence,
        "service_info": service_info,
        "account_status": account_status,
        "billing_data": billing_data,
        "needs_human_review": needs_human,
        "human_review_reason": human_reason,
        "next_step": next_step
    }

def process_by_specialized_agent(agent_type: str):
    """Generate a function to process messages with a specialized agent"""
    def process_fn(state: AgentState) -> AgentState:
        # Get emotional context
        emotional_tone = state.get("sentiment", {}).get("emotional_tone", "neutral")
        
        # Create agent with appropriate emotion handling
        agent = create_agent(getattr(AgentType, agent_type.upper()), emotion=emotional_tone)
        
        # Add context data for the agent
        context_data = {
            "customer_info": state.get("customer_info"),
            "service_info": state.get("service_info"),
            "account_status": state.get("account_status"),
            "billing_data": state.get("billing_data"),
            "intent": state.get("intent"),
            "sub_intent": state.get("sub_intent")
        }
        
        # Create a context message
        context_message = SystemMessage(content=f"""
        CUSTOMER CONTEXT:
        Intent: {context_data['intent']} / {context_data['sub_intent']}
        Verified: {state.get('verified', False)}
        Customer ID: {state.get('customer_id', 'Unknown')}
        
        Please provide a helpful, empathetic response to this customer's needs.
        """)
        
        # Add context to the messages
        messages = state["messages"] + [context_message]
        
        # Get response from agent
        response = agent.invoke(messages)
        
        # Check if agent indicates need for human review
        agent_requests_human = "human" in response.content.lower() and (
            "transfer" in response.content.lower() or 
            "escalate" in response.content.lower() or
            "specialist" in response.content.lower()
        )
        
        needs_human = state.get("needs_human_review", False) or agent_requests_human
        human_reason = state.get("human_review_reason")
        
        if agent_requests_human and not state.get("needs_human_review"):
            human_reason = f"Agent requested human assistance: {agent_type}"
        
        # Determine completion status
        completion_status = "completed" if not needs_human else "transferred_to_human"
        
        next_step = "human_handoff" if needs_human else "end"
        
        return {
            **state,
            "messages": state["messages"] + [response],
            "final_response": response.content,
            "needs_human_review": needs_human,
            "human_review_reason": human_reason,
            "completion_status": completion_status,
            "next_step": next_step
        }
        
    return process_fn

def human_handoff(state: AgentState) -> AgentState:
    """Prepare the conversation for human handoff with empathetic transition"""
    # Get emotional context
    emotional_tone = state.get("sentiment", {}).get("emotional_tone", "neutral")
    urgency = state.get("sentiment", {}).get("urgency", 3)
    
    # Create appropriate handoff message based on emotion and reason
    reason = state.get("human_review_reason", "specialized assistance")
    
    handoff_templates = {
        "angry": "I understand this is frustrating. I'm connecting you right away with a customer service specialist who can address this situation personally. They'll have all the context we've discussed.",
        
        "frustrated": "I can see this hasn't been resolved to your satisfaction yet. I'm going to connect you with a specialist who has additional tools to help with your specific situation.",
        
        "anxious": "I want to make sure you get the exact information you need. Let me connect you with a dedicated specialist who can provide you with definitive answers and support.",
        
        "confused": "To make sure we get this right for you, I'm connecting you with one of our specialists who can walk through this step-by-step with you and answer all your questions.",
        
        "happy": "I'd like to connect you with one of our specialists who can provide even more personalized assistance with your request. They'll be with you shortly.",
        
        "neutral": "I'm connecting you with a specialist who can better assist you with this matter. They'll have access to all the information we've discussed."
    }
    
    handoff_message = handoff_templates.get(emotional_tone, handoff_templates["neutral"])
    
    # For high urgency, add urgency acknowledgment
    if urgency >= 4:
        handoff_message += " Given the urgency of your situation, I've marked this as high priority."
    
    # Create the response
    response = AIMessage(content=handoff_message)
    
    # Add helpful context for the human agent
    human_agent_summary = f"""
    CONVERSATION SUMMARY FOR AGENT:
    Customer Sentiment: {emotional_tone} (Urgency: {urgency}/5)
    Primary Intent: {state.get('intent')}
    Sub-Intent: {state.get('sub_intent')}
    Verified: {state.get('verified', False)}
    Customer ID: {state.get('customer_id', 'Unknown')}
    
    Handoff Reason: {reason}
    
    Key Concerns: {', '.join(state.get('sentiment', {}).get('key_concerns', ['N/A']))}
    """
    
    return {
        **state,
        "messages": state["messages"] + [response],
        "final_response": handoff_message,
        "human_agent_summary": human_agent_summary,
        "completion_status": "transferred_to_human",
        "next_step": "end"
    }

# Define routing logic
def determine_next_step(state: AgentState) -> str:
    """Determine the next step in the workflow based on state"""
    return state.get("next_step", "end")

# Build the workflow graph
def build_workflow_graph():
    """Build the complete workflow graph with specialized agents"""
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("initial_analysis", initial_analysis)
    workflow.add_node("verification_check", verification_check)
    workflow.add_node("verification_agent", verification_agent)
    workflow.add_node("intent_classification", intent_classification)
    workflow.add_node("sales", process_by_specialized_agent("sales"))
    workflow.add_node("support", process_by_specialized_agent("support"))
    workflow.add_node("account", process_by_specialized_agent("account"))
    workflow.add_node("general", process_by_specialized_agent("general"))
    workflow.add_node("human_handoff", human_handoff)
    
    # Add edges
    workflow.add_conditional_edges(
        "initial_analysis",
        determine_next_step,
        {
            "verification_check": "verification_check",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "verification_check",
        determine_next_step,
        {
            "verification_agent": "verification_agent",
            "intent_classification": "intent_classification",
            "human_handoff": "human_handoff",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "verification_agent",
        determine_next_step,
        {
            "intent_classification": "intent_classification",
            "human_handoff": "human_handoff",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "intent_classification",
        determine_next_step,
        {
            "sales": "sales",
            "support": "support",
            "account": "account",
            "general": "general",
            "human_handoff": "human_handoff",
            "end": END
        }
    )
    
    # Add edges from specialized agents
    for agent_type in ["sales", "support", "account", "general"]:
        workflow.add_conditional_edges(
            agent_type,
            determine_next_step,
            {
                "human_handoff": "human_handoff",
                "end": END
            }
        )
    
    # Human handoff to end
    workflow.add_edge("human_handoff", END)
    
    # Set entry point
    workflow.set_entry_point("initial_analysis")
    
    # Compile the graph
    return workflow.compile()

# Create the workflow manager
class WorkflowManager:
    """Manages the LangGraph workflow for customer interactions"""
    
    def __init__(self):
        """Initialize the workflow manager"""
        self.workflow = build_workflow_graph()
        logger.info("LangGraph workflow initialized")
    
    def process_message(self, message: str, customer_id: str = None, conversation_id: str = None, 
                         contact_info: str = None, prev_messages: List = None) -> Dict[str, Any]:
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
        import time
        start_time = time.time()
        
        # Set up message history
        messages = [HumanMessage(content=message)]
        if prev_messages:
            messages = prev_messages + messages
        
        # Create initial state
        initial_state = {
            "messages": messages,
            "full_conversation_history": messages.copy(),
            "customer_id": customer_id,
            "customer_info": None,
            "verified": False,
            "verification_attempts": 0,
            "contact_info": contact_info,
            "intent": None,
            "sub_intent": None,
            "intent_confidence": None,
            "sentiment": None,
            "entities": None,
            "service_info": None,
            "account_status": None,
            "billing_data": None,
            "current_step": "initial_analysis",
            "next_step": None,
            "needs_human_review": False,
            "human_review_reason": None,
            "final_response": None,
            "completion_status": None,
            "processing_time": None,
            "token_usage": None
        }
        
        # Trace the workflow execution if tracing is enabled
        if tracing_manager and tracing_manager.enabled:
            with tracing_manager.get_tracer(tags=["workflow", "customer_message"]) as tracer:
                # Execute the workflow
                result = self.workflow.invoke(initial_state)
                
                # Add trace metadata
                tracer.add_metadata({
                    "customer_id": customer_id,
                    "conversation_id": conversation_id,
                    "intent": result.get("intent"),
                    "sentiment": result.get("sentiment"),
                    "verified": result.get("verified"),
                    "needs_human_review": result.get("needs_human_review"),
                    "human_review_reason": result.get("human_review_reason"),
                    "completion_status": result.get("completion_status")
                })
        else:
            # Execute the workflow without tracing
            result = self.workflow.invoke(initial_state)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        
        # Format the result for the API response
        response_data = {
            "response": result.get("final_response"),
            "intent": result.get("intent"),
            "sub_intent": result.get("sub_intent"),
            "intent_confidence": result.get("intent_confidence"),
            "sentiment": result.get("sentiment"),
            "verified": result.get("verified"),
            "needs_human_review": result.get("needs_human_review"),
            "human_review_reason": result.get("human_review_reason"),
            "human_agent_summary": result.get("human_agent_summary"),
            "conversation_id": conversation_id,
            "completion_status": result.get("completion_status"),
            "processing_time": processing_time
        }
        
        return response_data

# Create a singleton instance
workflow_manager = WorkflowManager()