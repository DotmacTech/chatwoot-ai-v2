from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from routers import health, webhook, intent, metrics
from core.config import settings
from utils.logging import AppLogger
from chatwoot_langsmith import setup_langsmith
from utils.security import verify_signature
from fastapi import Request, HTTPException, Body, Depends
import hmac
import hashlib
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from handlers import ChatwootHandler
from chatwoot_langsmith import tracing_manager, feedback_manager, cost_monitor
from chatwoot_langsmith.feedback import FeedbackType
from chatwoot_langchain import intent_classifier, INTENT_CATEGORIES
from models.webhook import WebhookPayload
from core.rate_limit import RateLimiter
from core.cache import cache
import httpx
import redis
import sys

# Initialize logger
logger = AppLogger(__name__)

# Load environment variables
load_dotenv(".env.test")

# Validate required environment variables
required_vars = [
    "CHATWOOT_API_KEY",
    "CHATWOOT_ACCOUNT_ID",
    "CHATWOOT_BASE_URL",
    "REDIS_HOST",
    "REDIS_PORT"
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# Create FastAPI app
app = FastAPI(
    title="Chatwoot AI Assistant",
    description="AI-powered assistant for Chatwoot using LangSmith and LangGraph",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis connection
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    password=settings.REDIS_PASSWORD or None,
    decode_responses=True
)

# Initialize rate limiter and handlers
rate_limiter = RateLimiter()
chatwoot_handler = ChatwootHandler()

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(webhook.router, prefix="/api/v1", tags=["webhook"])
app.include_router(intent.router, prefix="/api/v1", tags=["intent"])
app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])

# Set up LangSmith
setup_langsmith()

# Get webhook secret from environment variables
CHATWOOT_SECRET = os.getenv("CHATWOOT_WEBHOOK_SECRET")
if not CHATWOOT_SECRET:
    logger.warning("CHATWOOT_WEBHOOK_SECRET not set. Webhook signatures will not be verified!")

async def verify_webhook_signature(request: Request):
    """Verify Chatwoot webhook signature"""
    if not settings.CHATWOOT_WEBHOOK_SECRET:
        logger.warning("CHATWOOT_WEBHOOK_SECRET not set. Webhook signatures will not be verified!")
        return True
    
    signature = request.headers.get("X-Chatwoot-Signature")
    if not signature:
        logger.error("No webhook signature found in request")
        raise HTTPException(status_code=401, detail="No signature provided")
    
    # Get raw body
    body = await request.body()
    
    # Calculate expected signature
    expected_signature = hmac.new(
        settings.CHATWOOT_WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()
    
    # Compare signatures
    if not hmac.compare_digest(signature, expected_signature):
        logger.error("Invalid webhook signature")
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    return True

@app.on_event("startup")
async def startup_event():
    """Run startup tasks"""
    logger.info("Starting application...")
    
    # Test Redis connection
    try:
        redis_client.ping()
        logger.info("Redis connection successful")
    except redis.ConnectionError as e:
        logger.error("Failed to connect to Redis", exc_info=True)
        raise HTTPException(status_code=503, detail="Redis connection failed")

    # Test Chatwoot connection
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.CHATWOOT_BASE_URL}/api/v1/accounts/{settings.CHATWOOT_ACCOUNT_ID}/contacts",
                headers={"api_access_token": settings.CHATWOOT_API_TOKEN}
            )
            if response.status_code != 200:
                logger.error(f"Chatwoot connection failed with status {response.status_code}")
                raise HTTPException(status_code=503, detail="Chatwoot connection failed")
            logger.info("Chatwoot connection successful")
    except Exception as e:
        logger.error("Failed to connect to Chatwoot", exc_info=True)
        raise HTTPException(status_code=503, detail="Chatwoot connection failed")

    logger.info("Application startup complete")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = datetime.now()
    response = await call_next(request)
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info(
        f"Request: {request.method} {request.url.path}",
        extra={
            "duration": duration,
            "status_code": response.status_code,
            "client_host": request.client.host
        }
    )
    return response

@app.post("/feedback")
async def submit_feedback(feedback: Dict[str, Any] = Body(...)):
    """
    Submit feedback for a LangSmith run
    """
    if not feedback_manager.enabled:
        return {"status": "error", "message": "LangSmith feedback collection is not enabled"}
    
    try:
        feedback_id = feedback_manager.submit_feedback(
            run_id=feedback["run_id"],
            feedback_type=feedback["feedback_type"],
            score=feedback.get("score"),
            comment=feedback.get("comment"),
            metadata=feedback.get("metadata")
        )
        
        if feedback_id:
            return {"status": "ok", "feedback_id": feedback_id}
        else:
            return {"status": "error", "message": "Failed to submit feedback"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/feedback/{run_id}")
async def get_feedback(run_id: str):
    """
    Get all feedback for a specific run
    """
    if not feedback_manager.enabled:
        return {"status": "error", "message": "LangSmith feedback collection is not enabled"}
    
    try:
        feedback_list = feedback_manager.get_run_feedback(run_id)
        return {"status": "ok", "feedback": feedback_list}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/feedback-stats")
async def get_feedback_stats():
    """
    Get statistics on collected feedback
    """
    if not feedback_manager.enabled:
        return {"status": "error", "message": "LangSmith feedback collection is not enabled"}
    
    try:
        stats = feedback_manager.get_feedback_stats()
        return {"status": "ok", "stats": stats}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/monitoring-dashboard", response_class=HTMLResponse)
async def monitoring_dashboard(request: Request):
    """
    Monitoring dashboard for the application.
    Displays metrics, traces, and feedback data.
    """
    try:
        # Get usage statistics
        usage_stats = cost_monitor.get_usage_stats()
        
        # Create HTML content without using f-strings for JavaScript parts
        daily_tokens = usage_stats['daily']['tokens']['total']
        daily_limit = usage_stats['daily']['limit']
        daily_usage_percent = usage_stats['daily']['usage_percent']
        
        monthly_cost = usage_stats['monthly']['cost']
        monthly_budget = usage_stats['monthly']['budget']
        monthly_usage_percent = usage_stats['monthly']['usage_percent']
        
        daily_date = usage_stats['daily']['date']
        monthly_month = usage_stats['monthly']['month']
        
        input_tokens = usage_stats['daily']['tokens']['input']
        output_tokens = usage_stats['daily']['tokens']['output']
        daily_cost = usage_stats['daily']['cost']
        remaining_budget = usage_stats['monthly']['remaining']
        
        return templates.TemplateResponse(
            "pages/monitoring.html",
            {
                "request": request,
                "active_page": "monitoring",
                "daily_tokens": daily_tokens,
                "daily_limit": daily_limit,
                "daily_usage_percent": daily_usage_percent,
                "monthly_cost": monthly_cost,
                "monthly_budget": monthly_budget,
                "monthly_usage_percent": monthly_usage_percent,
                "daily_date": daily_date,
                "monthly_month": monthly_month,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "daily_cost": daily_cost,
                "remaining_budget": remaining_budget
            }
        )
    except Exception as e:
        logger.error(f"Failed to render monitoring dashboard: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """
    Dashboard for monitoring the system.
    """
    # Get usage stats
    usage_stats = cost_monitor.get_usage_stats()
    
    # Format stats for display
    daily_usage = usage_stats.get("daily_usage", {})
    monthly_usage = usage_stats.get("monthly_usage", {})
    
    # Format currency values
    daily_cost = "${:.2f}".format(daily_usage.get("cost", 0))
    monthly_cost = "${:.2f}".format(monthly_usage.get("cost", 0))
    
    # Calculate percentages of limits
    daily_token_limit = cost_monitor.daily_token_limit
    monthly_budget = cost_monitor.monthly_budget
    
    daily_token_percent = (daily_usage.get("total_tokens", 0) / daily_token_limit * 100) if daily_token_limit else 0
    monthly_cost_percent = (monthly_usage.get("cost", 0) / monthly_budget * 100) if monthly_budget else 0
    
    return templates.TemplateResponse(
        "pages/dashboard.html",
        {
            "request": request,
            "active_page": "dashboard",
            "daily_usage": daily_usage,
            "monthly_usage": monthly_usage,
            "daily_cost": daily_cost,
            "monthly_cost": monthly_cost,
            "daily_token_limit": daily_token_limit,
            "monthly_budget": monthly_budget,
            "daily_token_percent": daily_token_percent,
            "monthly_cost_percent": monthly_cost_percent
        }
    )

# Intent classification endpoints
@app.get("/intent-categories")
async def get_intent_categories():
    """
    Get all available intent categories.
    """
    return {"status": "ok", "categories": INTENT_CATEGORIES}

@app.post("/intent-feedback")
async def submit_intent_feedback(
    feedback: Dict[str, Any] = Body(...)
):
    """
    Submit feedback on an intent classification.
    
    Required fields in the request body:
    - conversation_id: ID of the conversation
    - original_intent: The original classified intent
    - corrected_intent: The correct intent as determined by the agent
    - agent_id: ID of the agent providing the feedback
    """
    required_fields = ["conversation_id", "original_intent", "corrected_intent", "agent_id"]
    for field in required_fields:
        if field not in feedback:
            return {"status": "error", "message": f"Missing required field: {field}"}
    
    try:
        # Get the original classification
        original_classification = {
            "intent": feedback["original_intent"],
            "confidence": feedback.get("original_confidence", 0.0),
            "message": feedback.get("message", "")
        }
        
        # Record the feedback
        stats = intent_classifier.record_feedback(
            original_classification=original_classification,
            corrected_intent=feedback["corrected_intent"],
            agent_id=feedback["agent_id"],
            conversation_id=feedback["conversation_id"]
        )
        
        return {
            "status": "ok", 
            "message": "Feedback recorded successfully",
            "stats": stats
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/intent-stats")
async def get_intent_stats():
    """
    Get statistics on intent classification and feedback.
    """
    try:
        stats = intent_classifier.get_feedback_stats()
        return {"status": "ok", "stats": stats}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/estimate-intent")
async def estimate_intent(
    request: Dict[str, str] = Body(...)
):
    """
    Estimate the intent of a message without recording it.
    Useful for testing the intent classifier.
    
    Required fields:
    - message: The message to classify
    """
    if "message" not in request:
        return {"status": "error", "message": "Missing required field: message"}
    
    try:
        # Check usage limits
        within_limits, limit_reason = cost_monitor.check_limits()
        if not within_limits:
            return {"status": "error", "message": f"Usage limits exceeded: {limit_reason}"}
        
        # Estimate cost
        input_tokens = cost_monitor.estimate_tokens(request["message"])
        estimated_cost = cost_monitor.estimate_cost(input_tokens=input_tokens, output_tokens=100)
        
        # Classify intent
        classification = intent_classifier.classify_intent(request["message"])
        
        # Track usage
        cost_monitor.track_usage(
            input_tokens=input_tokens,
            output_tokens=0,
            metadata={"source": "intent_estimation"}
        )
        
        return {
            "status": "ok",
            "classification": classification,
            "estimated_cost": estimated_cost
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Add endpoints for cost monitoring
@app.get("/usage-stats")
async def get_usage_stats():
    """
    Get current usage statistics
    """
    return cost_monitor.get_usage_stats()

@app.get("/usage-history")
async def get_usage_history(period: str = "daily", limit: int = 30):
    """
    Get usage history for a specific period
    """
    return {"history": cost_monitor.get_usage_history(period, limit)}

@app.post("/update-limits")
async def update_limits(limits: Dict[str, Any]):
    """
    Update usage limits
    """
    daily_token_limit = limits.get("daily_token_limit")
    monthly_budget = limits.get("monthly_budget")
    
    return cost_monitor.update_limits(
        daily_token_limit=daily_token_limit,
        monthly_budget=monthly_budget
    )

@app.post("/estimate-cost")
async def estimate_cost(request: Dict[str, Any]):
    """
    Estimate cost for processing text
    """
    input_text = request.get("input_text", "")
    expected_output_length = request.get("expected_output_length", 100)
    model = request.get("model")
    
    return cost_monitor.estimate_cost(
        input_text=input_text,
        expected_output_length=expected_output_length,
        model=model
    )

@app.get("/intent-dashboard", response_class=HTMLResponse)
async def intent_dashboard(request: Request):
    """
    Dashboard for intent classification statistics and testing.
    """
    # Get intent categories
    categories = INTENT_CATEGORIES
    
    # Get feedback stats
    stats = intent_classifier.get_feedback_stats()
    
    return templates.TemplateResponse(
        "pages/intent_dashboard.html",
        {
            "request": request,
            "active_page": "intent_dashboard",
            "categories": categories,
            "stats": stats
        }
    )

@app.get("/test-connections")
async def test_connections():
    """Test connections to external services"""
    results = {
        "chatwoot": {"status": "unknown"},
        "deepseek": {"status": "unknown"}
    }
    
    # Test Chatwoot connection
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.CHATWOOT_BASE_URL}/api/v1/accounts/{settings.CHATWOOT_ACCOUNT_ID}/contacts",
                headers={"api_access_token": settings.CHATWOOT_API_TOKEN}
            )
            if response.status_code == 200:
                results["chatwoot"] = {"status": "success"}
            else:
                results["chatwoot"] = {
                    "status": "error",
                    "details": f"Status code: {response.status_code}"
                }
    except Exception as e:
        results["chatwoot"] = {"status": "error", "details": str(e)}

    # Test DeepSeek connection (we'll just verify we have the API key)
    if settings.DEEPSEEK_API_KEY:
        results["deepseek"] = {"status": "configured"}
    else:
        results["deepseek"] = {"status": "error", "details": "API key not configured"}

    return results

@app.post("/webhook")
async def webhook(request: Request, verified: bool = Depends(verify_webhook_signature)):
    """Handle Chatwoot webhook"""
    try:
        payload_dict = await request.json()
        logger.info("Received webhook", extra={"payload": payload_dict})
        
        # Convert dict to WebhookPayload model
        try:
            from models.webhook import WebhookPayload
            payload = WebhookPayload(**payload_dict)
            
            # Enhanced debug logging
            logger.info("Parsed webhook payload", extra={
                "event": payload.event,
                "message_present": payload.message is not None,
                "conversation_present": payload.conversation is not None,
                "contact_present": payload.contact is not None,
                "message_id": payload.message.id if payload.message else None,
                "message_content": payload.message.content if payload.message else None,
                "message_type": payload.message.message_type if payload.message else None,
                "conversation_id": payload.conversation.id if payload.conversation else None,
                "contact_id": payload.contact.id if payload.contact else None,
                "contact_email": payload.contact.email if payload.contact else None
            })
            
        except Exception as e:
            logger.error(f"Error parsing webhook payload: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": f"Invalid payload format: {str(e)}"}
            )
        
        # Process webhook with Chatwoot handler
        result = await chatwoot_handler.process(payload)
        
        # Debug the result
        logger.info("Webhook processing result", extra={"result": result})
        
        return result
    except Exception as e:
        logger.error("Error processing webhook", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
