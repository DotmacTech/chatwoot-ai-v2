"""
Intent classification module using LangChain.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

from chatwoot_langsmith import tracing_manager

# Configure logging
logger = logging.getLogger(__name__)

# Define intent categories
INTENT_CATEGORIES = {
    "support": [
        "technical_issue", 
        "bug_report", 
        "feature_request",
        "account_access",
        "password_reset",
        "installation_help",
        "integration_problem"
    ],
    "billing": [
        "payment_issue",
        "subscription_question",
        "refund_request",
        "invoice_request",
        "pricing_question",
        "upgrade_downgrade",
        "payment_method"
    ],
    "product": [
        "how_to",
        "feature_inquiry",
        "product_comparison",
        "compatibility",
        "limitations",
        "best_practices",
        "documentation"
    ],
    "sales": [
        "pricing_inquiry",
        "discount_request",
        "demo_request",
        "enterprise_question",
        "bulk_purchase",
        "contract_terms",
        "partnership"
    ],
    "general": [
        "greeting",
        "thank_you",
        "general_inquiry",
        "feedback",
        "complaint",
        "praise",
        "other"
    ]
}

class IntentClassification(BaseModel):
    """Intent classification result"""
    primary_intent: str = Field(description="Primary intent category")
    sub_intent: str = Field(description="Specific sub-intent within the primary category")
    confidence: float = Field(description="Confidence score between 0 and 1")
    entities: Dict[str, Any] = Field(description="Extracted entities from the message")
    sentiment: Dict[str, Any] = Field(description="Sentiment analysis of the message")
    requires_human: bool = Field(description="Whether human intervention is recommended")
    reason: Optional[str] = Field(description="Reason for human intervention if required")

class IntentClassifier:
    """A class for classifying customer intents using LangChain."""
    
    def __init__(self, model_name: str = None, temperature: float = 0.1):
        """
        Initialize the intent classifier.
        
        Args:
            model_name: The name of the LLM model to use
            temperature: The temperature parameter for the LLM
        """
        self.model_name = model_name or os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-reasoner")
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = ChatDeepSeek(
            model=self.model_name, 
            temperature=self.temperature,
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            streaming=False
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="output"
        )
        
        # Set up the classification chain
        self._setup_classifier()
        
        # Load feedback data if exists
        self.feedback_data_path = Path(__file__).parent.parent / "data" / "intent_feedback.json"
        self.feedback_data = self._load_feedback_data()
        
        logger.info(f"Intent classifier initialized with model: {self.model_name}")
        
    def _setup_classifier(self):
        """Set up the classification chain with prompt and output parser."""
        # Define the output schema
        response_schemas = [
            ResponseSchema(name="primary_intent", description="The primary intent category of the message (support, billing, product, sales, general)"),
            ResponseSchema(name="sub_intent", description="The specific sub-intent within the primary category"),
            ResponseSchema(name="confidence", description="Confidence score between 0 and 1"),
            ResponseSchema(name="entities", description="JSON object of extracted entities from the message"),
            ResponseSchema(name="sentiment", description="JSON object with sentiment analysis (positive, negative, neutral) and intensity (0-1)"),
            ResponseSchema(name="requires_human", description="Boolean indicating whether human intervention is recommended"),
            ResponseSchema(name="reason", description="Reason for human intervention if required")
        ]
        
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        
        # Create the prompt template
        template = """
        You are an AI assistant that classifies customer service messages into intent categories.
        
        Analyze the following message and classify it according to these categories:
        
        Primary intent categories:
        - support: Technical issues, bugs, feature requests, account access
        - billing: Payment issues, subscription questions, refunds, invoices
        - product: How-to questions, feature inquiries, compatibility
        - sales: Pricing, discounts, demos, enterprise questions
        - general: Greetings, thank you, general inquiries, feedback
        
        Sub-intent categories for 'support':
        {support_intents}
        
        Sub-intent categories for 'billing':
        {billing_intents}
        
        Sub-intent categories for 'product':
        {product_intents}
        
        Sub-intent categories for 'sales':
        {sales_intents}
        
        Sub-intent categories for 'general':
        {general_intents}
        
        Customer message: {message}
        
        {format_instructions}
        """
        
        # Format the intents for the prompt
        intent_formatters = {
            "support_intents": ", ".join(INTENT_CATEGORIES["support"]),
            "billing_intents": ", ".join(INTENT_CATEGORIES["billing"]),
            "product_intents": ", ".join(INTENT_CATEGORIES["product"]),
            "sales_intents": ", ".join(INTENT_CATEGORIES["sales"]),
            "general_intents": ", ".join(INTENT_CATEGORIES["general"])
        }
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["message"],
            partial_variables={
                **intent_formatters,
                "format_instructions": self.output_parser.get_format_instructions()
            }
        )
        
        # Create the chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            output_key="output"
        )
        
    def classify_intent(self, message: str) -> IntentClassification:
        """
        Classify the intent of a message using the LangChain pipeline.
        
        Args:
            message (str): The message to classify
            
        Returns:
            IntentClassification: The classified intent with category, subcategory, and confidence
        """
        try:
            # Run the classification chain
            result = self.chain.invoke({"message": message})
            
            # Parse the output
            parsed_output = self.output_parser.parse(result["output"])
            
            # Create the classification object
            classification = IntentClassification(
                primary_intent=parsed_output["primary_intent"],
                sub_intent=parsed_output["sub_intent"],
                confidence=float(parsed_output["confidence"]),
                entities=parsed_output["entities"],
                sentiment=parsed_output["sentiment"],
                requires_human=parsed_output["requires_human"],
                reason=parsed_output.get("reason", None)
            )
            
            logger.info(f"Classified message as {classification.primary_intent}/{classification.sub_intent} with confidence {classification.confidence}")
            
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            # Return a fallback classification
            return IntentClassification(
                primary_intent="general",
                sub_intent="other",
                confidence=0.0,
                entities={},
                sentiment={"sentiment": "neutral", "intensity": 0.5},
                requires_human=True,
                reason=f"Error in classification: {str(e)}"
            )
    
    def record_feedback(self, 
                       original_classification: Dict[str, Any], 
                       corrected_intent: str,
                       agent_id: str,
                       conversation_id: str) -> Dict[str, Any]:
        """
        Record feedback on intent classification for improving the system.
        
        Args:
            original_classification: The original classification result
            corrected_intent: The corrected intent category
            agent_id: ID of the agent providing the correction
            conversation_id: ID of the conversation
            
        Returns:
            Updated feedback stats
        """
        # Create trace for this operation
        trace_id = tracing_manager.create_trace(
            name="record_feedback",
            inputs={
                "original": original_classification,
                "corrected": corrected_intent,
                "conversation_id": conversation_id
            },
            tags=["intent", "feedback"]
        )
        
        try:
            # Create feedback entry
            timestamp = datetime.now().isoformat()
            feedback_entry = {
                "timestamp": timestamp,
                "conversation_id": conversation_id,
                "agent_id": agent_id,
                "original_intent": original_classification["primary_intent"],
                "original_sub_intent": original_classification["sub_intent"],
                "corrected_intent": corrected_intent,
                "confidence": original_classification["confidence"]
            }
            
            # Add to feedback data
            if "entries" not in self.feedback_data:
                self.feedback_data["entries"] = []
                
            self.feedback_data["entries"].append(feedback_entry)
            
            # Save feedback data
            self._save_feedback_data()
            
            # Calculate updated stats
            stats = self._calculate_feedback_stats()
            
            if trace_id:
                tracing_manager.end_trace(trace_id, outputs={"stats": stats})
            
            return stats
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            if trace_id:
                tracing_manager.end_trace(trace_id, error=str(e))
            return {"error": str(e)}
    
    def _load_feedback_data(self) -> Dict[str, Any]:
        """Load feedback data from file"""
        if self.feedback_data_path.exists():
            try:
                with open(self.feedback_data_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading feedback data: {e}")
                
        # Return empty data structure if file doesn't exist or error occurs
        return {"entries": [], "stats": {}}
    
    def _save_feedback_data(self) -> bool:
        """Save feedback data to file"""
        try:
            # Ensure the directory exists
            self.feedback_data_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.feedback_data_path, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Error saving feedback data: {e}")
            return False
    
    def _calculate_feedback_stats(self) -> Dict[str, Any]:
        """Calculate statistics from feedback data"""
        entries = self.feedback_data.get("entries", [])
        total_entries = len(entries)
        
        if total_entries == 0:
            return {
                "total_feedback": 0,
                "accuracy": 0,
                "category_accuracy": {},
                "common_corrections": {}
            }
            
        # Calculate overall accuracy
        correct_entries = sum(1 for entry in entries if entry["original_intent"] == entry["corrected_intent"])
        accuracy = correct_entries / total_entries
        
        # Calculate accuracy by category
        categories = {}
        for entry in entries:
            category = entry["original_intent"]
            if category not in categories:
                categories[category] = {"total": 0, "correct": 0}
                
            categories[category]["total"] += 1
            if entry["original_intent"] == entry["corrected_intent"]:
                categories[category]["correct"] += 1
                
        category_accuracy = {
            category: data["correct"] / data["total"] 
            for category, data in categories.items()
        }
        
        # Find common corrections
        corrections = {}
        for entry in entries:
            if entry["original_intent"] != entry["corrected_intent"]:
                key = f"{entry['original_intent']} -> {entry['corrected_intent']}"
                corrections[key] = corrections.get(key, 0) + 1
                
        # Sort corrections by frequency
        sorted_corrections = dict(sorted(
            corrections.items(), 
            key=lambda item: item[1], 
            reverse=True
        )[:10])  # Top 10 corrections
        
        stats = {
            "total_feedback": total_entries,
            "accuracy": accuracy,
            "category_accuracy": category_accuracy,
            "common_corrections": sorted_corrections
        }
        
        # Update stats in the feedback data
        self.feedback_data["stats"] = stats
        
        return stats
