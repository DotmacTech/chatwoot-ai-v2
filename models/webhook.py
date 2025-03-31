from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class EventType(str, Enum):
    """Types of events from Chatwoot webhook"""
    MESSAGE_CREATED = "message_created"
    MESSAGE_UPDATED = "message_updated"
    CONVERSATION_CREATED = "conversation_created"
    CONVERSATION_STATUS_CHANGED = "conversation_status_changed"
    CONVERSATION_UPDATED = "conversation_updated"


class Account(BaseModel):
    """Account information"""
    id: int
    name: str


class ContactInbox(BaseModel):
    """Contact inbox information"""
    id: Optional[int] = None
    contact_id: Optional[int] = None
    inbox_id: Optional[int] = None
    source_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    hmac_verified: Optional[bool] = None
    pubsub_token: Optional[str] = None


class Sender(BaseModel):
    """Sender information"""
    account: Optional[Account] = None
    additional_attributes: Optional[Dict[str, Any]] = None
    avatar: Optional[str] = None
    custom_attributes: Optional[Dict[str, Any]] = None
    email: Optional[str] = None
    id: Optional[int] = None
    identifier: Optional[str] = None
    name: Optional[str] = None
    phone_number: Optional[str] = None
    thumbnail: Optional[str] = None
    type: Optional[str] = None


class Meta(BaseModel):
    """Meta information for conversation"""
    sender: Optional[Sender] = None
    assignee: Optional[Any] = None
    team: Optional[Any] = None
    hmac_verified: Optional[bool] = None


class MessageInConversation(BaseModel):
    """Message in conversation"""
    id: Optional[int] = None
    content: Optional[str] = None
    account_id: Optional[int] = None
    inbox_id: Optional[int] = None
    conversation_id: Optional[int] = None
    message_type: Optional[int] = None
    created_at: Optional[Union[int, str]] = None
    updated_at: Optional[str] = None
    private: Optional[bool] = None
    status: Optional[str] = None
    source_id: Optional[str] = None
    content_type: Optional[str] = None
    content_attributes: Optional[Dict[str, Any]] = {}
    sender_type: Optional[str] = None
    sender_id: Optional[int] = None
    external_source_ids: Optional[Dict[str, Any]] = {}
    additional_attributes: Optional[Dict[str, Any]] = {}
    processed_message_content: Optional[str] = None
    sentiment: Optional[Dict[str, Any]] = {}
    conversation: Optional[Dict[str, Any]] = None
    sender: Optional[Sender] = None


class Conversation(BaseModel):
    """Conversation information"""
    additional_attributes: Optional[Dict[str, Any]] = {}
    can_reply: Optional[bool] = None
    channel: Optional[str] = None
    contact_inbox: Optional[ContactInbox] = None
    id: Optional[int] = None
    inbox_id: Optional[int] = None
    messages: Optional[List[MessageInConversation]] = []
    labels: Optional[List[str]] = []
    meta: Optional[Meta] = None
    status: Optional[str] = None
    custom_attributes: Optional[Dict[str, Any]] = {}
    snoozed_until: Optional[Any] = None
    unread_count: Optional[int] = None
    first_reply_created_at: Optional[str] = None
    priority: Optional[Any] = None
    waiting_since: Optional[int] = None
    agent_last_seen_at: Optional[int] = None
    contact_last_seen_at: Optional[int] = None
    timestamp: Optional[int] = None
    created_at: Optional[Union[int, str]] = None


class Inbox(BaseModel):
    """Inbox information"""
    id: Optional[int] = None
    name: Optional[str] = None


class Contact(BaseModel):
    """Contact information"""
    additional_attributes: Optional[Dict[str, Any]] = {}
    custom_attributes: Optional[Dict[str, Any]] = {}
    email: Optional[str] = None
    id: Optional[int] = None
    identifier: Optional[str] = None
    name: Optional[str] = None
    phone_number: Optional[str] = None
    thumbnail: Optional[str] = None
    type: Optional[str] = None


class Message(BaseModel):
    """Message information"""
    id: Optional[int] = None
    content: Optional[str] = None
    message_type: Optional[str] = None
    content_type: Optional[str] = None
    content_attributes: Optional[Dict[str, Any]] = {}
    created_at: Optional[Union[int, str]] = None
    private: Optional[bool] = None
    source_id: Optional[str] = None
    sender: Optional[Sender] = None


class WebhookPayload(BaseModel):
    """Webhook payload from Chatwoot"""
    # Top-level fields
    event: Union[EventType, str]  # Accept any string value for event
    id: Optional[int] = None
    content: Optional[str] = None
    content_type: Optional[str] = None
    content_attributes: Optional[Dict[str, Any]] = {}
    message_type: Optional[str] = None
    private: Optional[bool] = None
    source_id: Optional[str] = None
    created_at: Optional[Union[int, str]] = None
    additional_attributes: Optional[Dict[str, Any]] = {}
    
    # Object fields
    message: Optional[Message] = None
    conversation: Optional[Conversation] = None
    account: Optional[Account] = None
    contact: Optional[Contact] = None
    sender: Optional[Sender] = None
    inbox: Optional[Inbox] = None

    class Config:
        # Allow extra fields that aren't defined in the model
        extra = "allow"
