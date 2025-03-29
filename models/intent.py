from pydantic import BaseModel, Field

class IntentRequest(BaseModel):
    """Request model for intent classification"""
    message: str = Field(..., description="Message to classify")

class IntentClassification(BaseModel):
    """Response model for intent classification"""
    intent: str = Field(..., description="Classified intent")
    confidence: float = Field(..., description="Confidence score")
    reasoning: str = Field(..., description="Reasoning behind the classification")
