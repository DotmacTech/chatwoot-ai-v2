from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from typing import Dict, Any, Optional
from core.errors import ValidationError
from core.config import settings
from core.rate_limit import rate_limit
from utils.templates import templates
from utils.monitoring import CostMonitor
from utils.logging import AppLogger

router = APIRouter(tags=["monitoring"])
logger = AppLogger(__name__)
cost_monitor = CostMonitor()

@router.get("/monitoring-dashboard", response_class=HTMLResponse)
@rate_limit(key_prefix="dashboard", max_requests=settings.RATE_LIMIT_DASHBOARD)
async def monitoring_dashboard(request: Request) -> HTMLResponse:
    """Render the monitoring dashboard"""
    try:
        metrics = cost_monitor.get_metrics()
        return templates.TemplateResponse(
            "pages/monitoring.html",
            {
                "request": request,
                "metrics": metrics
            }
        )
    except Exception as e:
        logger.error("Error rendering monitoring dashboard", exc_info=True)
        raise ValidationError(
            message="Failed to render monitoring dashboard",
            details={"error": str(e)}
        )

@router.get("/dashboard", response_class=HTMLResponse)
@rate_limit(key_prefix="dashboard", max_requests=settings.RATE_LIMIT_DASHBOARD)
async def main_dashboard(request: Request) -> HTMLResponse:
    """Render the main system dashboard"""
    try:
        # Get system metrics
        daily_usage = cost_monitor.get_daily_usage()
        daily_token_limit = cost_monitor.get_daily_token_limit()
        daily_token_percent = (daily_usage.total_tokens / daily_token_limit * 100) if daily_token_limit else 0
        
        monthly_cost = cost_monitor.get_monthly_cost()
        monthly_budget = cost_monitor.get_monthly_budget()
        monthly_cost_percent = (monthly_cost / monthly_budget * 100) if monthly_budget else 0
        
        return templates.TemplateResponse(
            "pages/dashboard.html",
            {
                "request": request,
                "daily_usage": daily_usage,
                "daily_token_limit": daily_token_limit,
                "daily_token_percent": daily_token_percent,
                "monthly_cost": monthly_cost,
                "monthly_budget": monthly_budget,
                "monthly_cost_percent": monthly_cost_percent
            }
        )
    except Exception as e:
        logger.error("Error rendering main dashboard", exc_info=True)
        raise ValidationError(
            message="Failed to render main dashboard",
            details={"error": str(e)}
        )

@router.post("/update-limits")
@rate_limit(key_prefix="api", max_requests=settings.RATE_LIMIT_API)
async def update_limits(
    request: Request,
    daily_token_limit: Optional[int] = None,
    monthly_budget: Optional[float] = None
) -> Dict[str, Any]:
    """Update usage limits"""
    try:
        if daily_token_limit is not None:
            cost_monitor.set_daily_token_limit(daily_token_limit)
        if monthly_budget is not None:
            cost_monitor.set_monthly_budget(monthly_budget)
            
        return {
            "status": "ok",
            "message": "Limits updated successfully",
            "data": {
                "daily_token_limit": cost_monitor.get_daily_token_limit(),
                "monthly_budget": cost_monitor.get_monthly_budget()
            }
        }
    except Exception as e:
        logger.error("Error updating limits", exc_info=True)
        raise ValidationError(
            message="Failed to update limits",
            details={"error": str(e)}
        )
