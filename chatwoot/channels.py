"""
Channel-specific message handling for Chatwoot integration.
Supports WhatsApp, Facebook, Instagram, Email, and Website channels.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Channel type constants
CHANNEL_WHATSAPP = "Channel::Whatsapp"
CHANNEL_FACEBOOK = "Channel::FacebookPage" 
CHANNEL_INSTAGRAM = "Channel::Instagram"
CHANNEL_EMAIL = "Channel::Email"
CHANNEL_WEB = "Channel::WebWidget"
CHANNEL_API = "Channel::Api"
CHANNEL_TELEGRAM = "Channel::Telegram"

class ChannelHandler:
    """Base class for channel-specific message handling"""
    
    def __init__(self, conversation_data: Dict[str, Any]):
        """
        Initialize with conversation data from webhook
        
        Args:
            conversation_data: Conversation data from webhook
        """
        self.conversation = conversation_data
        self.inbox = conversation_data.get("inbox", {})
        self.channel_type = self.inbox.get("channel_type")
        
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message based on channel type
        
        Args:
            message: Message data from webhook
            
        Returns:
            Processing result with channel-specific data
        """
        logger.info(f"Processing message for channel: {self.channel_type}")
        
        # Add channel identifier to context
        result = {
            "channel_type": self.channel_type,
            "processed": True
        }
        
        # Channel-specific processing
        if self.channel_type == CHANNEL_WHATSAPP:
            return await self._process_whatsapp(message, result)
        elif self.channel_type == CHANNEL_FACEBOOK:
            return await self._process_facebook(message, result)
        elif self.channel_type == CHANNEL_INSTAGRAM:
            return await self._process_instagram(message, result)
        elif self.channel_type == CHANNEL_EMAIL:
            return await self._process_email(message, result)
        elif self.channel_type == CHANNEL_WEB:
            return await self._process_web(message, result)
        elif self.channel_type == CHANNEL_API:
            return await self._process_api(message, result)
        elif self.channel_type == CHANNEL_TELEGRAM:
            return await self._process_telegram(message, result)
        else:
            logger.warning(f"Unknown channel type: {self.channel_type}")
            result["processed"] = False
            result["error"] = "Unknown channel type"
            return result
    
    async def _process_whatsapp(self, message: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Process WhatsApp-specific message attributes"""
        # Extract WhatsApp-specific metadata if available
        result["platform"] = "whatsapp"
        
        # Handle WhatsApp-specific message types (text, image, audio, etc.)
        message_type = self._get_message_type(message)
        result["message_type"] = message_type
        
        if message_type == "image":
            result["has_attachment"] = True
            # Add logic for image processing
        elif message_type == "audio":
            result["has_attachment"] = True
            # Add logic for audio processing
        elif message_type == "location":
            # Add logic for location processing
            pass
            
        return result
    
    async def _process_facebook(self, message: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Process Facebook-specific message attributes"""
        result["platform"] = "facebook"
        # Add Facebook-specific processing
        return result
    
    async def _process_instagram(self, message: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Process Instagram-specific message attributes"""
        result["platform"] = "instagram"
        # Add Instagram-specific processing
        return result
    
    async def _process_email(self, message: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Process Email-specific message attributes"""
        result["platform"] = "email"
        
        # Extract email-specific data
        if "email" in message:
            result["email_data"] = {
                "subject": message.get("email", {}).get("subject", ""),
                "from": message.get("email", {}).get("from", ""),
                "to": message.get("email", {}).get("to", "")
            }
            
        return result
    
    async def _process_web(self, message: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Process Web Widget-specific message attributes"""
        result["platform"] = "web_widget"
        # Add Web Widget-specific processing
        return result
    
    async def _process_api(self, message: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Process API-specific message attributes"""
        result["platform"] = "api"
        # Add API-specific processing
        return result
    
    async def _process_telegram(self, message: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Process Telegram-specific message attributes"""
        result["platform"] = "telegram"
        # Add Telegram-specific processing
        return result
    
    def _get_message_type(self, message: Dict[str, Any]) -> str:
        """
        Determine message type based on content and attachments
        
        Args:
            message: Message data from webhook
            
        Returns:
            Message type (text, image, audio, video, etc.)
        """
        if message.get("attachments") and len(message.get("attachments", [])) > 0:
            attachment = message["attachments"][0]
            file_type = attachment.get("file_type", "")
            
            if "image" in file_type:
                return "image"
            elif "audio" in file_type:
                return "audio"
            elif "video" in file_type:
                return "video"
            elif "file" in file_type:
                return "file"
        
        # Check for location data
        if message.get("content_attributes", {}).get("location"):
            return "location"
            
        # Default to text
        return "text"

def get_channel_handler(conversation_data: Dict[str, Any]) -> ChannelHandler:
    """
    Factory function to get the appropriate channel handler
    
    Args:
        conversation_data: Conversation data from webhook
        
    Returns:
        ChannelHandler instance
    """
    return ChannelHandler(conversation_data)
