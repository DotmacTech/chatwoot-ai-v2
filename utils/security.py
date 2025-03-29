import hmac
import hashlib
import json
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException, Header
from core.config import settings
from utils.logging import AppLogger

logger = AppLogger(__name__)

def verify_signature(payload: Dict[str, Any], signature: str, secret: str) -> bool:
    """
    Verify the HMAC signature from Chatwoot webhook
    
    Args:
        payload: The webhook payload
        signature: The signature from X-Chatwoot-Signature header
        secret: The webhook secret key
        
    Returns:
        bool: True if signature is valid
    """
    try:
        computed_signature = hmac.new(
            key=secret.encode("utf-8"),
            msg=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            digestmod=hashlib.sha256
        ).hexdigest()
        
        is_valid = hmac.compare_digest(computed_signature, signature)
        if not is_valid:
            logger.warning(
                "Invalid webhook signature", 
                extra={
                    "computed": computed_signature,
                    "received": signature
                }
            )
        return is_valid
    except Exception as e:
        logger.error(
            f"Signature verification failed: {str(e)}", 
            exc_info=True,
            extra={"error": str(e)}
        )
        return False

async def verify_webhook_signature(
    request: Request,
    x_chatwoot_signature: Optional[str] = Header(None)
) -> Optional[str]:
    """FastAPI dependency for verifying webhook signatures
    
    Args:
        request: FastAPI request object
        x_chatwoot_signature: Signature from X-Chatwoot-Signature header
        
    Returns:
        str: The signature if valid
        
    Raises:
        HTTPException: If signature is invalid
    """
    if not settings.CHATWOOT_WEBHOOK_SECRET:
        logger.warning("Webhook secret not configured, skipping signature verification")
        return None
        
    if not x_chatwoot_signature:
        raise HTTPException(
            status_code=401,
            detail="Missing X-Chatwoot-Signature header"
        )
    
    # Get raw request body
    body = await request.body()
    payload = json.loads(body)
    
    if not verify_signature(payload, x_chatwoot_signature, settings.CHATWOOT_WEBHOOK_SECRET):
        raise HTTPException(
            status_code=401,
            detail="Invalid webhook signature"
        )
    
    return x_chatwoot_signature
