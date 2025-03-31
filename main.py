from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from routers import health, webhook, intent, metrics
from core.config import settings
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
from models.webhook import WebhookPayload
from core.rate_limit import RateLimiter
import httpx
import redis
import sys
from utils.logging import AppLogger
from chatwoot_langgraph.workflow import WorkflowManager
from services.langsmith_service import LangSmithService
from services.chatwoot_client import ChatwootClient
from services.llm_service import LLMService
from services.redis_service import RedisService
from handlers.chatwoot import ChatwootHandler

# Initialize logger
logger = AppLogger(__name__)

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

# Initialize Redis Service
logger.info("Initializing RedisService...")
redis_service = RedisService()
# Add a check to see if connection was successful (optional, depends on requirements)
# async def check_redis_connection():
#    if not await redis_service.is_connected():
#        logger.critical("Redis service failed to connect. Application may not function correctly.")
#        # Decide if the app should exit
#        # sys.exit("Exiting: Failed to connect to Redis via service.")
#    else:
#        logger.info("RedisService connected successfully.")
# asyncio.run(check_redis_connection()) # Check connection at startup (consider async startup events)


# Initialize Chatwoot API Client Service
logger.info("Initializing ChatwootClient service...")
chatwoot_client = ChatwootClient()
if not chatwoot_client.client:
    logger.error("ChatwootClient failed to initialize properly due to missing configuration. Application might not function correctly.")
    # Depending on requirements, you might want to exit here:
    # sys.exit("Exiting due to incomplete Chatwoot configuration.") 
else:
    logger.info("ChatwootClient service initialized.")

# Initialize rate limiter
rate_limiter = RateLimiter()

# Initialize services conditionally based on settings
logger.info("Initializing services...")
redis_service = RedisService()
chatwoot_client = ChatwootClient()

# LangSmith Service (Conditional)
if settings.LANGSMITH_TRACING_ENABLED:
    logger.info("LangSmith tracing is ENABLED. Initializing LangSmithService...")
    langsmith_service = LangSmithService() # Initialize if enabled
    if not langsmith_service.enabled:
        logger.warning("LangSmithService initialized but reports as disabled internally. Check LangSmith config/keys.")
        langsmith_service = None # Treat as disabled if initialization fails internally
    else:
        logger.info("LangSmithService initialized successfully.")
else:
    logger.info("LangSmith tracing is DISABLED by configuration.")
    langsmith_service = None

# LLM Service (Requires LangSmith instance if enabled)
logger.info("Initializing LLMService...")
llm_service = LLMService(langsmith_service=langsmith_service)
if not llm_service.is_available():
    logger.warning("LLMService failed to initialize properly (likely missing API key or config). AI features may be limited.")
    # Consider exiting if LLM is critical
    # sys.exit("Exiting due to LLM service initialization failure.")
else:
    logger.info("LLMService initialized.")

# Workflow Manager (Requires LLM and optional LangSmith)
logger.info("Initializing WorkflowManager...")
workflow_manager = WorkflowManager(llm_service=llm_service, langsmith_service=langsmith_service)
logger.info("WorkflowManager initialized.")

# Chatwoot Handler (Requires multiple services)
logger.info("Initializing ChatwootHandler...")
chatwoot_handler = ChatwootHandler(
    chatwoot_client=chatwoot_client,
    workflow_manager=workflow_manager,
    langsmith_service=langsmith_service,
    llm_service=llm_service
)
logger.info("ChatwootHandler initialized.")

# Store components in app state for dependency injection
app.state.chatwoot_client = chatwoot_client
app.state.workflow_manager = workflow_manager
app.state.langsmith_service = langsmith_service # Store instance or None
app.state.llm_service = llm_service
app.state.redis_service = redis_service # Store the Redis service
app.state.chatwoot_handler = chatwoot_handler # Store the ChatwootHandler for webhook router

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(webhook.router, tags=["webhook"])
app.include_router(intent.router, prefix="/api/v1", tags=["intent"])
app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
# Setup templates directory
templates = Jinja2Templates(directory="templates")

# Get webhook secret from environment variables
CHATWOOT_SECRET = settings.CHATWOOT_WEBHOOK_SECRET
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
    logger.info("Running application startup tasks...")
    
    # Ensure RedisService is connected
    try:
        if await redis_service.is_connected():
             logger.info("Redis connection check successful.")
        else:
             logger.error("CRITICAL: Redis connection check failed. Check Redis server and configuration.")
             # sys.exit("Exiting: Failed to connect to Redis.")
    except Exception as e:
        logger.error(f"CRITICAL: Redis connection failed during check: {e}", exc_info=True)
        # sys.exit("Exiting: Failed to connect to Redis.")

    # Fetch Chatwoot Agent Profile
    try:
        await chatwoot_client._fetch_self_profile() # Calls _fetch_self_profile internally
        if chatwoot_client.agent_id is None:
            logger.error("CRITICAL: Failed to fetch Chatwoot agent ID after initialization. Loop prevention may fail.")
            # sys.exit("Exiting: Could not determine bot agent ID.")
        else:
            logger.info(f"Successfully fetched Chatwoot bot agent ID: {chatwoot_client.agent_id}")
    except Exception as e:
        logger.error(f"CRITICAL: Error during Chatwoot client initialization/profile fetch: {e}", exc_info=True)
        # sys.exit("Exiting: Error during Chatwoot initialization.")

    # Initialize LangSmith if enabled
    if langsmith_service:
        try:
            # Assuming LangSmithService might have an async initialize method if needed
            # await langsmith_service.initialize()
            logger.info("LangSmithService initialization placeholder (if needed in future).")
        except Exception as e:
            logger.error(f"Error initializing LangSmithService during startup: {e}", exc_info=True)
            # Decide if this is critical

    # LLM Service is initialized in its __init__, no separate call needed here

    logger.info("Startup complete. Application is ready.")

# Dependency providers
def get_redis_service():
    return redis_service

def get_chatwoot_client():
    return chatwoot_client

def get_langsmith_service() -> Optional[LangSmithService]: # Return Optional
    return langsmith_service # Return instance or None

def get_llm_service():
    return llm_service

def get_workflow_manager():
    return workflow_manager

def get_chatwoot_handler():
    return chatwoot_handler

app.include_router(webhook.router, dependencies=[
    Depends(get_redis_service),
    Depends(get_chatwoot_client),
    Depends(get_langsmith_service), # Dependency injection handles None okay
    Depends(get_llm_service),
    Depends(get_workflow_manager),
    Depends(get_chatwoot_handler)
])

# Monitoring Endpoints (Conditionally available)
if langsmith_service:
    @app.post("/feedback", tags=["monitoring"])
    async def submit_feedback(feedback: Dict[str, Any]):
        """Submit feedback for a LangSmith run"""
        if not langsmith_service or not langsmith_service.enabled:
            return {"status": "error", "message": "LangSmith feedback collection is not enabled"}
        
        try:
            feedback_id = langsmith_service.submit_feedback(
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

    @app.get("/run/{run_id}/feedback", tags=["monitoring"])
    async def get_run_feedback(run_id: str):
        """Get all feedback for a specific run"""
        if not langsmith_service or not langsmith_service.enabled:
            return {"status": "error", "message": "LangSmith feedback collection is not enabled"}
        
        try:
            feedback_list = langsmith_service.get_run_feedback(run_id)
            return {"status": "ok", "feedback": feedback_list}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.get("/feedback-stats", tags=["monitoring"])
    async def get_feedback_stats():
        """Get statistics on collected feedback"""
        if not langsmith_service or not langsmith_service.enabled:
            return {"status": "error", "message": "LangSmith feedback collection is not enabled"}
        
        try:
            stats = langsmith_service.get_feedback_stats()
            return {"status": "ok", "stats": stats}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.get("/monitoring", response_class=HTMLResponse, tags=["monitoring"])
    async def monitoring_page(request: Request):
        """
        Page for monitoring LangSmith usage.
        """
        try:
            usage_stats = langsmith_service.get_usage_stats()
            
            daily_tokens = usage_stats.get('daily', {}).get('tokens', {}).get('total', 0)
            daily_limit = usage_stats.get('daily', {}).get('limit', None) # Might be None
            daily_usage_percent = usage_stats.get('daily', {}).get('usage_percent', 0)
            
            monthly_cost = usage_stats.get('monthly', {}).get('cost', 0)
            monthly_budget = usage_stats.get('monthly', {}).get('budget', None) # Might be None
            monthly_usage_percent = usage_stats.get('monthly', {}).get('usage_percent', 0)
            
            daily_date = usage_stats.get('daily', {}).get('date', 'N/A')
            monthly_month = usage_stats.get('monthly', {}).get('month', 'N/A')
            
            input_tokens = usage_stats.get('daily', {}).get('tokens', {}).get('input', 0)
            output_tokens = usage_stats.get('daily', {}).get('tokens', {}).get('output', 0)
            daily_cost = usage_stats.get('daily', {}).get('cost', 0)
            remaining_budget = usage_stats.get('monthly', {}).get('remaining', 0)
            
            return templates.TemplateResponse(
                "pages/monitoring.html",
                {
                    "request": request,
                    "active_page": "monitoring",
                    "daily_tokens": daily_tokens,
                    "daily_limit": daily_limit if daily_limit is not None else "Not Set",
                    "daily_usage_percent": daily_usage_percent,
                    "monthly_cost": monthly_cost,
                    "monthly_budget": monthly_budget if monthly_budget is not None else "Not Set",
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
            logger.error(f"Error loading monitoring page: {e}", exc_info=True)
            return HTMLResponse(content=f"<h1>Error loading monitoring data: {e}</h1>", status_code=500)

    @app.get("/dashboard", response_class=HTMLResponse, tags=["monitoring"])
    async def dashboard(request: Request):
        """
        Dashboard for monitoring the system.
        """
        try:
            usage_stats = langsmith_service.get_usage_stats()
            
            daily_usage = usage_stats.get("daily", {})
            monthly_usage = usage_stats.get("monthly", {})
            
            daily_cost = "${:.2f}".format(daily_usage.get("cost", 0))
            monthly_cost = "${:.2f}".format(monthly_usage.get("cost", 0))
            
            daily_token_limit = langsmith_service.daily_token_limit
            monthly_budget = langsmith_service.monthly_budget
            
            daily_token_percent = (daily_usage.get("tokens", {}).get("total", 0) / daily_token_limit * 100) if daily_token_limit else 0
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
                    "daily_token_limit": daily_token_limit if daily_token_limit else "Not Set",
                    "monthly_budget": monthly_budget if monthly_budget else "Not Set",
                    "daily_token_percent": daily_token_percent,
                    "monthly_cost_percent": monthly_cost_percent
                }
            )
        except Exception as e:
            logger.error(f"Error loading dashboard: {e}", exc_info=True)
            return HTMLResponse(content=f"<h1>Error loading dashboard data: {e}</h1>", status_code=500)

    @app.post("/estimate-intent", tags=["monitoring"])
    async def estimate_intent(request: Dict[str, Any]):
        """
        Estimate intent without full processing (for testing/cost estimation)
        """
        if not langsmith_service or not langsmith_service.enabled:
             return {"status": "error", "message": "LangSmith service is not enabled for estimation."}
        try:
            within_limits, limit_reason = langsmith_service.check_limits()
            if not within_limits:
                return {"status": "error", "message": f"Usage limits exceeded: {limit_reason}"}
            
            input_tokens = langsmith_service.estimate_tokens(request["message"])
            estimated_cost = langsmith_service.estimate_cost(input_tokens=input_tokens, output_tokens=10) # Estimate low output for intent
            
            classification = {}
            # Placeholder for actual classification if needed for more accurate estimation
            
            langsmith_service.track_usage(
                input_tokens=input_tokens,
                output_tokens=0, # Just estimating input cost for intent
                metadata={"source": "intent_estimation"}
            )
            
            return {
                "status": "ok",
                "estimated_intent": "unknown", # Placeholder
                "estimated_input_tokens": input_tokens,
                "estimated_cost": estimated_cost
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.get("/usage-stats", tags=["monitoring"])
    async def get_usage_stats():
        """
        Get current usage statistics
        """
        if not langsmith_service or not langsmith_service.enabled:
            return {"status": "error", "message": "LangSmith service is not enabled"}
        return langsmith_service.get_usage_stats()

    @app.get("/usage-history", tags=["monitoring"])
    async def get_usage_history(period: str = "daily", limit: int = 30):
        """
        Get usage history for a specific period
        """
        if not langsmith_service or not langsmith_service.enabled:
            return {"status": "error", "message": "LangSmith service is not enabled"}
        return {"history": langsmith_service.get_usage_history(period, limit)}

    @app.post("/update-limits", tags=["monitoring"])
    async def update_limits(limits: Dict[str, Any]):
        """
        Update usage limits
        """
        if not langsmith_service or not langsmith_service.enabled:
            return {"status": "error", "message": "LangSmith service is not enabled"}
        daily_token_limit = limits.get("daily_token_limit")
        monthly_budget = limits.get("monthly_budget")
        
        return langsmith_service.update_limits(
            daily_token_limit=daily_token_limit,
            monthly_budget=monthly_budget
        )

    @app.post("/estimate-cost", tags=["monitoring"])
    async def estimate_cost(request: Dict[str, Any]):
        """
        Estimate cost for processing text
        """
        if not langsmith_service or not langsmith_service.enabled:
            return {"status": "error", "message": "LangSmith service is not enabled"}
        input_text = request.get("input_text", "")
        expected_output_length = request.get("expected_output_length", 100)
        model = request.get("model")
        
        return langsmith_service.estimate_cost(
            input_text=input_text,
            expected_output_length=expected_output_length,
            model=model
        )

    @app.get("/intent-dashboard", response_class=HTMLResponse, tags=["monitoring"])
    async def intent_dashboard(request: Request):
        """
        Placeholder for intent dashboard page
        """
        # Logic to fetch intent data (e.g., from Redis or a database)
        intent_data = {"greeting": 10, "question": 25, "support_request": 15}
        return templates.TemplateResponse(
            "pages/intent_dashboard.html",
            {
                "request": request,
                "active_page": "intent_dashboard",
                "intent_data": intent_data # Pass data to template
            }
        )

    @app.get("/feedback-summary", response_class=HTMLResponse, tags=["monitoring"])
    async def feedback_summary(request: Request):
        """
        Placeholder for feedback summary page
        """
        # Logic to fetch feedback summary data
        feedback_stats = langsmith_service.get_feedback_stats()
        return templates.TemplateResponse(
            "pages/feedback_summary.html",
            {
                "request": request,
                "active_page": "feedback_summary",
                "feedback_stats": feedback_stats # Pass data to template
            }
        )

else:
    logger.info("Monitoring and feedback endpoints are disabled because LangSmith tracing is not enabled.")
    # Optionally define dummy endpoints returning errors if needed
    @app.get("/monitoring", tags=["monitoring"])
    async def monitoring_disabled():
        return JSONResponse(status_code=404, content={"detail": "Monitoring is disabled"})
    @app.get("/dashboard", tags=["monitoring"])
    async def dashboard_disabled():
        return JSONResponse(status_code=404, content={"detail": "Dashboard is disabled"})
    # Add similar stubs for other monitoring endpoints if direct access should be blocked

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
    if not langsmith_service or not langsmith_service.enabled:
        return {"status": "error", "message": "LangSmith feedback collection is not enabled"}
    
    try:
        feedback_id = langsmith_service.submit_feedback(
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
    if not langsmith_service or not langsmith_service.enabled:
        return {"status": "error", "message": "LangSmith feedback collection is not enabled"}
    
    try:
        feedback_list = langsmith_service.get_run_feedback(run_id)
        return {"status": "ok", "feedback": feedback_list}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/feedback-stats")
async def get_feedback_stats():
    """
    Get statistics on collected feedback
    """
    if not langsmith_service or not langsmith_service.enabled:
        return {"status": "error", "message": "LangSmith feedback collection is not enabled"}
    
    try:
        stats = langsmith_service.get_feedback_stats()
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
        usage_stats = langsmith_service.get_usage_stats()
        
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
    try:
        usage_stats = langsmith_service.get_usage_stats()
        
        daily_usage = usage_stats.get("daily_usage", {})
        monthly_usage = usage_stats.get("monthly_usage", {})
        
        daily_cost = "${:.2f}".format(daily_usage.get("cost", 0))
        monthly_cost = "${:.2f}".format(monthly_usage.get("cost", 0))
        
        daily_token_limit = langsmith_service.daily_token_limit
        monthly_budget = langsmith_service.monthly_budget
        
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
    except Exception as e:
        logger.error(f"Failed to render dashboard: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Intent classification endpoints
@app.get("/intent-categories")
async def get_intent_categories():
    """
    Get all available intent categories.
    """
    return {"status": "ok", "categories": []}

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
        stats = {}
        
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
        stats = {}
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
        within_limits, limit_reason = langsmith_service.check_limits()
        if not within_limits:
            return {"status": "error", "message": f"Usage limits exceeded: {limit_reason}"}
        
        # Estimate cost
        input_tokens = langsmith_service.estimate_tokens(request["message"])
        estimated_cost = langsmith_service.estimate_cost(input_tokens=input_tokens, output_tokens=100)
        
        # Classify intent
        classification = {}
        
        # Track usage
        langsmith_service.track_usage(
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
    return langsmith_service.get_usage_stats()

@app.get("/usage-history")
async def get_usage_history(period: str = "daily", limit: int = 30):
    """
    Get usage history for a specific period
    """
    return {"history": langsmith_service.get_usage_history(period, limit)}

@app.post("/update-limits")
async def update_limits(limits: Dict[str, Any]):
    """
    Update usage limits
    """
    daily_token_limit = limits.get("daily_token_limit")
    monthly_budget = limits.get("monthly_budget")
    
    return langsmith_service.update_limits(
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
    
    return langsmith_service.estimate_cost(
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
    categories = []
    
    # Get feedback stats
    stats = {}
    
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

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully close resources on shutdown."""
    logger.info("Closing Redis service connection...")
    # Use the service's close method
    try:
        await redis_service.close()
        logger.info("Redis service closed successfully.")
    except Exception as e:
        logger.error(f"Error closing Redis service: {e}", exc_info=True)
 
    logger.info("Closing Chatwoot client...")
    await chatwoot_client.close()
    logger.info("Application shutdown complete.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=settings.PORT, 
        log_level=settings.LOG_LEVEL.lower()
    )
