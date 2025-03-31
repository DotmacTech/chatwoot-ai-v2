from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from core.config import settings
from utils.logging import AppLogger

router = APIRouter(tags=["metrics"])
logger = AppLogger(__name__)

class Trace(BaseModel):
    """Model for a LangSmith trace"""
    id: str
    name: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    status: str
    tags: List[str]
    error: Optional[str]

@router.get("/metrics")
async def get_metrics():
    """Get current metrics from the tracing system"""
    try:
        metrics = get_langsmith_service().get_metrics()
        
        # Add timestamp and uptime
        metrics["timestamp"] = datetime.now().isoformat()
        metrics["uptime_seconds"] = get_langsmith_service().metrics.get("uptime_seconds", 0)
        
        return metrics
    except Exception as e:
        logger.error("Failed to get metrics", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )

@router.post("/metrics/reset")
async def reset_metrics():
    """Reset all metrics in the tracing system"""
    try:
        get_langsmith_service().reset_metrics()
        return {
            "status": "success",
            "message": "Metrics reset successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error("Failed to reset metrics", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset metrics: {str(e)}"
        )

@router.get("/traces")
async def get_recent_traces(limit: int = 10) -> List[Trace]:
    """
    Get recent traces from LangSmith
    
    Args:
        limit: Maximum number of traces to return (default: 10)
    """
    if not get_langsmith_service().enabled:
        raise HTTPException(
            status_code=400,
            detail="LangSmith tracing is not enabled"
        )
    
    try:
        # Get recent runs from LangSmith
        runs = get_langsmith_service().client.list_runs(
            project_name=get_langsmith_service().project_name,
            limit=limit
        )
        
        # Format the runs for the response
        return [
            Trace(
                id=run.id,
                name=run.name,
                start_time=run.start_time,
                end_time=run.end_time,
                status=run.status,
                tags=run.tags,
                error=run.error
            )
            for run in runs
        ]
    except Exception as e:
        logger.error("Failed to get traces", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get traces: {str(e)}"
        )
