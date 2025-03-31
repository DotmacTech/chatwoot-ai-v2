from fastapi import APIRouter, Depends, Request
from typing import Optional, Dict, Any
from core.errors import WebhookError
from core.config import settings
from core.rate_limit import rate_limit
from models.webhook import WebhookPayload
from utils.security import verify_webhook_signature
from handlers import ChatwootHandler
from utils.logging import AppLogger

logger = AppLogger(__name__)

router = APIRouter(tags=["webhook"])

# --- Dependency to get the ChatwootHandler instance ---
def get_chatwoot_handler(request: Request) -> ChatwootHandler:
    """Retrieves the shared ChatwootHandler instance from app state."""
    handler = getattr(request.app.state, 'chatwoot_handler', None)
    if handler is None:
        # This should ideally not happen if main.py initializes correctly
        logger.error("ChatwootHandler not found in app state! This is a critical configuration error.")
        raise WebhookError(
            message="Internal server error: Handler not initialized",
            details={"error": "Handler not initialized"}
        )
    return handler
# ------------------------------------------------------

@router.post("/webhook")
@rate_limit(key_prefix="webhook", max_requests=settings.RATE_LIMIT_WEBHOOK)
async def chatwoot_webhook(
    request: Request,
    payload: WebhookPayload,
    signature: Optional[str] = Depends(verify_webhook_signature),
    handler: ChatwootHandler = Depends(get_chatwoot_handler)
) -> Dict[str, Any]:
    """Handle incoming Chatwoot webhooks"""
    event_type = payload.event
    logger.info(f"---> Webhook endpoint hit for event {event_type}")

    # Only process message_created events for now
    if event_type != "message_created":
        logger.info(f"Skipping event type '{event_type}'. Only processing 'message_created'.")
        return {"status": "skipped", "reason": f"Unsupported event type: {event_type}"}

    # Log more details for message_created events
    if event_type == "message_created":
        # Extract message details for logging - prioritize top-level
        message_content = getattr(payload, 'content', None)
        message_type = getattr(payload, 'message_type', None)
        conversation_id = None
        sender_name = None
        sender_type = None
        
        conversation = getattr(payload, 'conversation', None)
        if conversation:
            conversation_id = getattr(conversation, 'id', None)
            
        sender = getattr(payload, 'sender', None)
        if sender:
            sender_name = getattr(sender, 'name', None)
            sender_type = getattr(sender, 'type', None)
        
        # Attempt to get message type as int if string failed
        if message_type is None:
            raw_type_int = getattr(payload, 'message_type', None)
            if isinstance(raw_type_int, int):
                 message_type = "incoming" if raw_type_int == 0 else "outgoing"
                 logger.info(f"Router inferred message_type '{message_type}' from integer {raw_type_int}")

        # Fallback to message object if top-level fails (less likely for message_created based on sample)
        if message_type is None or message_content is None:
            message = getattr(payload, 'message', None)
            if message:
                logger.info("Router falling back to message object for details...")
                if message_content is None: 
                    message_content = getattr(message, 'content', None)
                if message_type is None:
                    message_type = getattr(message, 'message_type', None)
                    if isinstance(message_type, int):
                         message_type = "incoming" if message_type == 0 else "outgoing"
        
        logger.info(f"Webhook message details: type={message_type}, sender={sender_name}({sender_type}), conv_id={conversation_id}")
        if message_content:
            logger.info(f"Message content: '{message_content[:100]}...'")
    
    try:
        # Process the webhook
        result = await handler.process(payload)
        
        logger.info(f"Webhook processing result: {result}")
        
        return {
            "status": "ok",
            "message": "Webhook processed successfully",
            "data": result
        }
    except Exception as e:
        logger.error("Error processing webhook", exc_info=True, extra={
            "event": event_type,
            "error": str(e)
        })
        raise WebhookError(
            message="Failed to process webhook",
            details={"error": str(e)}
        )
