from fastapi import APIRouter, Depends, Request
from typing import Optional, Dict, Any
from core.errors import WebhookError
from core.config import settings
from core.rate_limit import rate_limit
from models.webhook import WebhookPayload
from utils.security import verify_webhook_signature
from handlers.chatwoot import ChatwootHandler
from utils.logging import AppLogger

router = APIRouter(tags=["webhook"])
logger = AppLogger(__name__)

@router.post("/chatwoot-webhook")
@rate_limit(key_prefix="webhook", max_requests=settings.RATE_LIMIT_WEBHOOK)
async def chatwoot_webhook(
    request: Request,
    payload: WebhookPayload,
    signature: Optional[str] = Depends(verify_webhook_signature)
) -> Dict[str, Any]:
    """Handle incoming Chatwoot webhooks"""
    try:
        # Process the webhook
        handler = ChatwootHandler()
        result = await handler.process(payload)
        
        return {
            "status": "ok",
            "message": "Webhook processed successfully",
            "data": result
        }
    except Exception as e:
        logger.error("Error processing webhook", exc_info=True, extra={
            "event": payload.event,
            "error": str(e)
        })
        raise WebhookError(
            message="Failed to process webhook",
            details={"error": str(e)}
        )
