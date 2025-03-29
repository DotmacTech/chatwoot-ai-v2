from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any

class AppError(HTTPException):
    """Base error class for application-specific errors"""
    def __init__(
        self,
        status_code: int,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

class ValidationError(AppError):
    """Error raised when request validation fails"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=422,
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )

class AuthenticationError(AppError):
    """Error raised when authentication fails"""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            status_code=401,
            message=message,
            error_code="AUTHENTICATION_ERROR"
        )

class WebhookError(AppError):
    """Error raised when webhook processing fails"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=400,
            message=message,
            error_code="WEBHOOK_ERROR",
            details=details
        )

class RateLimitError(AppError):
    """Error raised when rate limit is exceeded"""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            status_code=429,
            message=message,
            error_code="RATE_LIMIT_ERROR"
        )

async def error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Global error handler for AppError exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.message,
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path
        }
    )
