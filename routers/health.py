from fastapi import APIRouter, HTTPException
import httpx
from core.config import settings
from utils.logging import AppLogger

router = APIRouter(tags=["health"])
logger = AppLogger(__name__)

@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy"}

@router.get("/test-connections")
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
        logger.error("Chatwoot connection test failed", exc_info=True)
        results["chatwoot"] = {"status": "error", "details": str(e)}

    # Test DeepSeek connection
    if settings.DEEPSEEK_API_KEY:
        results["deepseek"] = {"status": "configured"}
    else:
        results["deepseek"] = {"status": "error", "details": "API key not configured"}

    # If all services are in error state, raise an error
    if all(r["status"] == "error" for r in results.values()):
        raise HTTPException(
            status_code=503,
            detail="All external services are unavailable"
        )

    return results
