"""
LangGraph workflow package for Chatwoot Automation.
Provides a stateful, multi-agent workflow for customer interactions.
"""
import os
import logging
from typing import Dict, Any, Optional, List

# Import third-party modules
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize components
from chatwoot_langgraph.workflow import workflow_manager

__all__ = ['workflow_manager']
