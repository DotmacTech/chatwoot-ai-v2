"""
Intent Classifier Module

This module provides functionality to classify customer intents using LangChain.
It analyzes incoming messages to determine the appropriate routing for human agents.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.run_helpers import traceable

# Intent categories
INTENT_CATEGORIES = {
    "sales": {
        "description": "Sales inquiries, product information, pricing, and purchasing",
        "examples": [
            "I want to know more about your product",
            "What's the price of your premium plan?",
            "Do you offer discounts for annual subscriptions?",
            "I'm interested in buying your product",
            "Can you tell me about your features?"
        ]
    },
    "support": {
        "description": "Technical issues, troubleshooting, and product usage help",
        "examples": [
            "I'm having trouble logging in",
            "The app keeps crashing when I try to upload a file",
            "How do I reset my password?",
            "Is your service down right now?",
            "I can't figure out how to use this feature"
        ]
    },
    "account": {
        "description": "Account management, billing issues, and service changes",
        "examples": [
            "I need to update my payment method",
            "When does my subscription renew?",
            "I want to cancel my account",
            "Can I upgrade my plan?",
            "I was charged twice this month"
        ]
    },
    "general": {
        "description": "General inquiries, greetings, and other non-specific requests",
        "examples": [
            "Hello, is anyone there?",
            "I have a question",
            "Can you help me?",
            "Thank you for your help",
            "I'd like to speak with someone"
        ]
    }
}

class IntentClassifier:
    """
    A class for classifying customer intents using LangChain.
    """
    
    def __init__(self, model_name: str = None, temperature: float = 0.1):
        """
        Initialize the intent classifier.
        
        Args:
            model_name: The name of the LLM model to use
            temperature: The temperature parameter for the LLM
        """
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        self.temperature = temperature
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        self.langsmith_client = Client()
        self.feedback_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                              "data", "intent_feedback.json")
        self._setup_classifier()
        self._load_feedback_data()
    
    def _setup_classifier(self):
        """Set up the classification chain with prompt and output parser."""
        # Create a prompt template with examples for each intent category
        examples_text = ""
        for intent, data in INTENT_CATEGORIES.items():
            examples_text += f"\n{intent.upper()}: {data['description']}\nExamples:\n"
            for example in data['examples']:
                examples_text += f"- {example}\n"
        
        prompt = ChatPromptTemplate.from_template(
            """You are an intent classification system for a customer service platform.
            Your job is to analyze customer messages and determine their primary intent.
            
            The possible intent categories are:
            {examples}
            
            Analyze the following customer message and classify it into one of the intent categories.
            If you're uncertain, use the "general" category.
            
            Customer message: {message}
            
            Provide your response as a JSON object with the following structure:
            {{
                "intent": "category_name",
                "confidence": 0.0 to 1.0,
                "reasoning": "Brief explanation of why you classified it this way",
                "suggested_response": "A suggested initial response for a human agent"
            }}
            """
        )
        
        # Set up the chain with JSON output parser
        self.parser = JsonOutputParser()
        self.chain = prompt | self.llm | self.parser
        
        # Prepare the chain with examples - updated for newer LangChain versions
        self.chain = self.chain.with_config({"configurable": {"examples": examples_text}})
    
    def _load_feedback_data(self):
        """Load feedback data from file if it exists."""
        os.makedirs(os.path.dirname(self.feedback_data_path), exist_ok=True)
        
        if os.path.exists(self.feedback_data_path):
            try:
                with open(self.feedback_data_path, 'r') as f:
                    self.feedback_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self.feedback_data = {"corrections": [], "stats": {}}
        else:
            self.feedback_data = {"corrections": [], "stats": {}}
    
    def _save_feedback_data(self):
        """Save feedback data to file."""
        with open(self.feedback_data_path, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
    
    @traceable(run_type="llm")
    def classify_intent(self, message: str) -> Dict[str, Any]:
        """
        Classify the intent of a customer message.
        
        Args:
            message: The customer message to classify
            
        Returns:
            A dictionary containing the classification results
        """
        try:
            result = self.chain.invoke({"message": message})
            
            # Ensure the result has all required fields
            if not all(key in result for key in ["intent", "confidence", "reasoning", "suggested_response"]):
                missing_keys = [key for key in ["intent", "confidence", "reasoning", "suggested_response"] 
                               if key not in result]
                raise ValueError(f"Missing required fields in classification result: {missing_keys}")
            
            # Validate intent category
            if result["intent"] not in INTENT_CATEGORIES:
                result["intent"] = "general"
                result["confidence"] = min(result["confidence"], 0.5)
                result["reasoning"] += " (Fallback to general category due to invalid intent)"
            
            # Add timestamp
            result["timestamp"] = datetime.now().isoformat()
            
            return result
        except Exception as e:
            # Fallback to general intent if classification fails
            return {
                "intent": "general",
                "confidence": 0.3,
                "reasoning": f"Classification error: {str(e)}",
                "suggested_response": "Thank you for contacting us. How can I assist you today?",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
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
        if corrected_intent not in INTENT_CATEGORIES:
            raise ValueError(f"Invalid intent category: {corrected_intent}")
        
        # Record the correction
        correction = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "agent_id": agent_id,
            "original_intent": original_classification["intent"],
            "corrected_intent": corrected_intent,
            "original_confidence": original_classification["confidence"],
            "message": original_classification.get("message", "")
        }
        
        self.feedback_data["corrections"].append(correction)
        
        # Update stats
        stats = self.feedback_data.get("stats", {})
        
        # Initialize if not exists
        if "total_corrections" not in stats:
            stats["total_corrections"] = 0
        if "corrections_by_category" not in stats:
            stats["corrections_by_category"] = {}
        if "confusion_matrix" not in stats:
            stats["confusion_matrix"] = {}
        
        # Update total
        stats["total_corrections"] += 1
        
        # Update by category
        if original_classification["intent"] not in stats["corrections_by_category"]:
            stats["corrections_by_category"][original_classification["intent"]] = 0
        stats["corrections_by_category"][original_classification["intent"]] += 1
        
        # Update confusion matrix
        original = original_classification["intent"]
        corrected = corrected_intent
        
        if original not in stats["confusion_matrix"]:
            stats["confusion_matrix"][original] = {}
        if corrected not in stats["confusion_matrix"][original]:
            stats["confusion_matrix"][original][corrected] = 0
        stats["confusion_matrix"][original][corrected] += 1
        
        # Save updated data
        self.feedback_data["stats"] = stats
        self._save_feedback_data()
        
        # Record in LangSmith if available
        try:
            run_id = original_classification.get("run_id")
            if run_id:
                self.langsmith_client.create_feedback(
                    run_id=run_id,
                    key="intent_correction",
                    score=0.0,  # 0.0 indicates a correction was needed
                    value={"original": original, "corrected": corrected}
                )
        except Exception:
            # Silently continue if LangSmith feedback fails
            pass
        
        return stats
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics on intent classification feedback.
        
        Returns:
            Dictionary containing feedback statistics
        """
        return self.feedback_data.get("stats", {})
