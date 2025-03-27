"""
Agent assignment and routing module for Chatwoot conversations.
Determines which agent (AI or human) should handle a conversation.
"""
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Types of agents that can handle conversations"""
    AI_SALES = "ai_sales"
    AI_SUPPORT = "ai_support"
    AI_GENERAL = "ai_general"
    HUMAN_SALES = "human_sales"
    HUMAN_SUPPORT = "human_support"
    HUMAN_MANAGER = "human_manager"

class RoutingRule:
    """Base class for routing rules"""
    
    def __init__(self, name: str, priority: int = 0):
        """
        Initialize a routing rule
        
        Args:
            name: Rule name for identification
            priority: Rule priority (higher = more important)
        """
        self.name = name
        self.priority = priority
    
    async def evaluate(self, 
                      message: Dict[str, Any], 
                      conversation: Dict[str, Any],
                      channel_data: Dict[str, Any]) -> Tuple[bool, Optional[AgentType]]:
        """
        Evaluate if this rule applies to the given message/conversation
        
        Args:
            message: Message data
            conversation: Conversation data
            channel_data: Channel-specific data
            
        Returns:
            Tuple of (rule_matched, agent_type)
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

class KeywordRule(RoutingRule):
    """Route based on keywords in message content"""
    
    def __init__(self, 
                name: str, 
                keywords: List[str], 
                agent_type: AgentType,
                priority: int = 0,
                match_all: bool = False):
        """
        Initialize with keywords to match
        
        Args:
            name: Rule name
            keywords: List of keywords to match
            agent_type: Agent type to assign if matched
            priority: Rule priority
            match_all: Whether all keywords must match (default: any)
        """
        super().__init__(name, priority)
        self.keywords = [k.lower() for k in keywords]
        self.agent_type = agent_type
        self.match_all = match_all
    
    async def evaluate(self, 
                      message: Dict[str, Any], 
                      conversation: Dict[str, Any],
                      channel_data: Dict[str, Any]) -> Tuple[bool, Optional[AgentType]]:
        """Check if message contains keywords"""
        content = message.get("content", "").lower()
        
        if self.match_all:
            # All keywords must be present
            matched = all(keyword in content for keyword in self.keywords)
        else:
            # Any keyword matches
            matched = any(keyword in content for keyword in self.keywords)
            
        return (matched, self.agent_type if matched else None)

class ChannelRule(RoutingRule):
    """Route based on channel type"""
    
    def __init__(self, 
                name: str, 
                channel_type: str, 
                agent_type: AgentType,
                priority: int = 0):
        """
        Initialize with channel type to match
        
        Args:
            name: Rule name
            channel_type: Channel type to match
            agent_type: Agent type to assign if matched
            priority: Rule priority
        """
        super().__init__(name, priority)
        self.channel_type = channel_type
        self.agent_type = agent_type
    
    async def evaluate(self, 
                      message: Dict[str, Any], 
                      conversation: Dict[str, Any],
                      channel_data: Dict[str, Any]) -> Tuple[bool, Optional[AgentType]]:
        """Check if conversation is from specified channel"""
        matched = channel_data.get("platform") == self.channel_type
        return (matched, self.agent_type if matched else None)

class RegexRule(RoutingRule):
    """Route based on regex pattern in message content"""
    
    def __init__(self, 
                name: str, 
                pattern: str, 
                agent_type: AgentType,
                priority: int = 0):
        """
        Initialize with regex pattern to match
        
        Args:
            name: Rule name
            pattern: Regex pattern to match
            agent_type: Agent type to assign if matched
            priority: Rule priority
        """
        super().__init__(name, priority)
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.agent_type = agent_type
    
    async def evaluate(self, 
                      message: Dict[str, Any], 
                      conversation: Dict[str, Any],
                      channel_data: Dict[str, Any]) -> Tuple[bool, Optional[AgentType]]:
        """Check if message matches regex pattern"""
        content = message.get("content", "")
        matched = bool(self.pattern.search(content))
        return (matched, self.agent_type if matched else None)

class Router:
    """Main routing engine for agent assignment"""
    
    def __init__(self, default_agent: AgentType = AgentType.AI_GENERAL):
        """
        Initialize router with rules
        
        Args:
            default_agent: Default agent type if no rules match
        """
        self.rules: List[RoutingRule] = []
        self.default_agent = default_agent
    
    def add_rule(self, rule: RoutingRule) -> None:
        """
        Add a routing rule
        
        Args:
            rule: RoutingRule instance
        """
        self.rules.append(rule)
        # Sort rules by priority (highest first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    async def route(self, 
                   message: Dict[str, Any], 
                   conversation: Dict[str, Any],
                   channel_data: Dict[str, Any]) -> Tuple[AgentType, str]:
        """
        Determine which agent should handle this conversation
        
        Args:
            message: Message data
            conversation: Conversation data
            channel_data: Channel-specific data
            
        Returns:
            Tuple of (agent_type, reason)
        """
        for rule in self.rules:
            matched, agent_type = await rule.evaluate(message, conversation, channel_data)
            
            if matched and agent_type:
                logger.info(f"Routing rule '{rule.name}' matched. Assigning to {agent_type.value}")
                return agent_type, f"Rule: {rule.name}"
        
        # No rules matched, use default
        logger.info(f"No routing rules matched. Using default agent: {self.default_agent.value}")
        return self.default_agent, "Default routing"

# Create default router with common rules
def create_default_router() -> Router:
    """
    Create a router with default routing rules
    
    Returns:
        Configured Router instance
    """
    router = Router()
    
    # Sales-related keywords
    router.add_rule(KeywordRule(
        name="Sales Inquiry",
        keywords=["pricing", "price", "cost", "package", "plan", "subscribe", "purchase"],
        agent_type=AgentType.AI_SALES,
        priority=10
    ))
    
    # Technical support keywords
    router.add_rule(KeywordRule(
        name="Technical Support",
        keywords=["help", "problem", "issue", "error", "not working", "broken", "fix"],
        agent_type=AgentType.AI_SUPPORT,
        priority=10
    ))
    
    # Human escalation patterns
    router.add_rule(RegexRule(
        name="Human Request",
        pattern=r"(speak|talk|connect)\s+(?:to|with)\s+(?:a|an|the)?\s*(human|agent|person|representative)",
        agent_type=AgentType.HUMAN_SUPPORT,
        priority=20  # Higher priority to catch escalation requests
    ))
    
    # Channel-specific rules
    router.add_rule(ChannelRule(
        name="WhatsApp Priority",
        channel_type="whatsapp",
        agent_type=AgentType.AI_GENERAL,
        priority=5
    ))
    
    return router
