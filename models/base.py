from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class MessageType(str, Enum):
    INCOMING = "incoming"
    OUTGOING = "outgoing"
    ACTIVITY = "activity"
    TEMPLATE = "template"

class WebhookEventType(str, Enum):
    MESSAGE_CREATED = "message_created"
    MESSAGE_UPDATED = "message_updated"
    CONVERSATION_CREATED = "conversation_created"
    CONVERSATION_STATUS_CHANGED = "conversation_status_changed"
    CONVERSATION_ASSIGNED = "conversation_assigned"

class Message(BaseModel):
    id: int
    content: str
    message_type: MessageType
    private: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class Conversation(BaseModel):
    id: int
    inbox_id: int
    status: Optional[str] = None
    assignee_id: Optional[int] = None
    contact_id: Optional[int] = None

class Account(BaseModel):
    id: int

class WebhookPayload(BaseModel):
    event: WebhookEventType
    message: Optional[Message] = None
    conversation: Optional[Conversation] = None
    account: Account
    
class WebhookResponse(BaseModel):
    status: str = Field(..., description="Status of the webhook processing")
    reason: Optional[str] = Field(None, description="Reason for skipping/error if applicable")
    message_id: Optional[int] = Field(None, description="ID of the processed message")
    intent: Optional[Dict[str, Any]] = Field(None, description="Classified intent details")
    response: Optional[str] = Field(None, description="Generated response if any")
