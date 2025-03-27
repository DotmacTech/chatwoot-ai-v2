"""
LangChain integration package for Chatwoot Automation.
"""

from .intent_classifier import IntentClassifier, INTENT_CATEGORIES

# Create a singleton instance of the intent classifier
intent_classifier = IntentClassifier()

__all__ = ['intent_classifier', 'IntentClassifier', 'INTENT_CATEGORIES']
