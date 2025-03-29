from abc import ABC, abstractmethod
from typing import Any, Dict
from models.intent import IntentClassification
from utils.logging import AppLogger

logger = AppLogger(__name__)

class BaseHandler(ABC):
    """Base handler for processing messages"""

    def __init__(self):
        """Initialize the handler"""
        self.logger = logger

    @abstractmethod
    async def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process the incoming payload"""
        pass

    @abstractmethod
    async def classify_intent(self, message: str) -> IntentClassification:
        """Classify the intent of a message"""
        pass

    @abstractmethod
    async def generate_response(self, message: str, intent: IntentClassification) -> str:
        """Generate a response based on the message and intent"""
        pass
