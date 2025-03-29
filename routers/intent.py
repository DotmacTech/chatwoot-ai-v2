from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from typing import Dict, Any
from pydantic import BaseModel
from datetime import timedelta
from core.errors import ValidationError
from core.config import settings
from core.rate_limit import rate_limit
from core.cache import cached
from utils.templates import templates
from utils.logging import AppLogger
from models.intent import IntentClassification
from handlers.chatwoot import ChatwootHandler

router = APIRouter(tags=["intent"])
logger = AppLogger(__name__)

class IntentRequest(BaseModel):
    """Request model for intent classification"""
    message: str

class IntentFeedback(BaseModel):
    """Model for intent classification feedback"""
    message_id: str
    original_intent: str
    corrected_intent: str
    conversation_id: str

def _build_intent_cache_key(request: Request, intent_request: IntentRequest) -> str:
    """Build cache key for intent classification"""
    return f"intent:{hash(intent_request.message)}"

@router.get("/intent-dashboard", response_class=HTMLResponse)
@rate_limit(key_prefix="dashboard", max_requests=settings.RATE_LIMIT_DASHBOARD)
@cached(key_prefix="dashboard", expire=timedelta(minutes=5))
async def intent_dashboard(request: Request) -> HTMLResponse:
    """Render the intent classification dashboard"""
    try:
        # Get intent categories and statistics
        categories = {
            "SALES": {
                "description": "Sales and pricing related queries",
                "examples": ["I want to upgrade my plan", "What's your pricing?"]
            },
            "SUPPORT": {
                "description": "Technical support and troubleshooting",
                "examples": ["My service is down", "How do I configure this?"]
            },
            "FEEDBACK": {
                "description": "User feedback and suggestions",
                "examples": ["I love this feature", "This could be better"]
            }
        }
        
        # Get feedback statistics
        stats = {
            "total_corrections": 150,
            "corrections_by_category": {
                "SALES": 45,
                "SUPPORT": 85,
                "FEEDBACK": 20
            },
            "categories": ["SALES", "SUPPORT", "FEEDBACK"],
            "confusion_matrix": {
                "SALES": {"SALES": 40, "SUPPORT": 3, "FEEDBACK": 2},
                "SUPPORT": {"SALES": 5, "SUPPORT": 75, "FEEDBACK": 5},
                "FEEDBACK": {"SALES": 2, "SUPPORT": 3, "FEEDBACK": 15}
            }
        }
        
        return templates.TemplateResponse(
            "pages/intent_dashboard.html",
            {
                "request": request,
                "categories": categories,
                "stats": stats
            }
        )
    except Exception as e:
        logger.error("Error rendering intent dashboard", exc_info=True)
        raise ValidationError(
            message="Failed to render intent dashboard",
            details={"error": str(e)}
        )

@router.post("/estimate-intent")
@rate_limit(key_prefix="api", max_requests=settings.RATE_LIMIT_API)
@cached(
    key_prefix="intent",
    expire=timedelta(minutes=30),
    key_builder=_build_intent_cache_key
)
async def estimate_intent(request: Request, intent_request: IntentRequest) -> Dict[str, Any]:
    """Estimate intent for a given message"""
    try:
        handler = ChatwootHandler()
        intent = await handler.classify_intent(intent_request.message)
        return {"status": "success", "intent": intent.dict()}
    except Exception as e:
        logger.error("Error estimating intent", exc_info=True)
        raise ValidationError(
            message="Failed to estimate intent",
            details={"error": str(e)}
        )

@router.post("/intent-feedback")
@rate_limit(key_prefix="api", max_requests=settings.RATE_LIMIT_API)
async def submit_intent_feedback(request: Request, feedback: IntentFeedback) -> Dict[str, Any]:
    """Submit feedback for intent classification"""
    try:
        # TODO: Store feedback and update training data
        return {
            "status": "success",
            "message": "Feedback submitted successfully",
            "feedback_id": "123"  # TODO: Generate real ID
        }
    except Exception as e:
        logger.error("Error submitting intent feedback", exc_info=True)
        raise ValidationError(
            message="Failed to submit feedback",
            details={"error": str(e)}
        )
