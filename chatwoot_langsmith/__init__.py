"""
LangSmith integration package for Chatwoot Automation.
"""
# Import standard library modules first
import os
import logging
from typing import Dict, Any, Optional, List, Callable

# Import third-party modules
from dotenv import load_dotenv
import langsmith as ls

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize components
from chatwoot_langsmith.monitoring import tracing_manager
from chatwoot_langsmith.feedback import feedback_manager
from chatwoot_langsmith.cost_monitoring import cost_monitor
from chatwoot_langsmith.integration import setup_langsmith

__all__ = ['setup_langsmith', 'tracing_manager', 'feedback_manager', 'cost_monitor']
