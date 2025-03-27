"""
LangSmith cost monitoring and usage limits module.
Tracks token usage, costs, and enforces usage limits.
"""
import os
import time
import logging
import json
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from dotenv import load_dotenv
from threading import Lock

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Default cost per 1K tokens for different models (in USD)
DEFAULT_COSTS = {
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "claude-2": {"input": 0.01, "output": 0.03},
    "claude-instant": {"input": 0.0015, "output": 0.0015},
    "llama-2": {"input": 0.0007, "output": 0.0007},
    "mistral": {"input": 0.0007, "output": 0.0007},
    "deepseek-coder": {"input": 0.0008, "output": 0.0016},
    "deepseek-chat": {"input": 0.0005, "output": 0.0010},
    "deepseek-llm": {"input": 0.0006, "output": 0.0012},
    "default": {"input": 0.01, "output": 0.01}  # Default fallback
}

class CostMonitor:
    """
    Monitors token usage, costs, and enforces usage limits.
    Provides real-time cost tracking and budget management.
    """
    
    def __init__(self):
        """Initialize cost monitoring system"""
        # Load configuration from environment variables
        self.daily_token_limit = int(os.getenv("DAILY_TOKEN_LIMIT", "100000"))
        self.monthly_budget = float(os.getenv("MONTHLY_BUDGET", "50.0"))  # USD
        self.cost_per_token = self._load_cost_config()
        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
        
        # Initialize usage tracking
        self.usage_lock = Lock()
        self.usage_file = os.path.join(os.path.dirname(__file__), "../data/usage.json")
        self.usage_data = self._load_usage_data()
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.usage_file), exist_ok=True)
        
        logger.info(f"Cost monitoring initialized with daily limit: {self.daily_token_limit} tokens, "
                   f"monthly budget: ${self.monthly_budget}")
    
    def _load_cost_config(self) -> Dict[str, Dict[str, float]]:
        """Load cost configuration from environment or defaults"""
        cost_config_str = os.getenv("MODEL_COSTS", "")
        
        if cost_config_str:
            try:
                return json.loads(cost_config_str)
            except json.JSONDecodeError:
                logger.warning("Invalid MODEL_COSTS format, using defaults")
                
        return DEFAULT_COSTS
    
    def _load_usage_data(self) -> Dict[str, Any]:
        """Load usage data from file or initialize if not exists"""
        if os.path.exists(self.usage_file):
            try:
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)
                    
                # Validate data structure
                if not all(k in data for k in ["daily", "monthly", "history"]):
                    logger.warning("Invalid usage data structure, reinitializing")
                    return self._initialize_usage_data()
                    
                return data
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading usage data: {e}")
                return self._initialize_usage_data()
        else:
            return self._initialize_usage_data()
    
    def _initialize_usage_data(self) -> Dict[str, Any]:
        """Initialize empty usage data structure"""
        today = datetime.now().strftime("%Y-%m-%d")
        current_month = datetime.now().strftime("%Y-%m")
        
        return {
            "daily": {
                "date": today,
                "tokens": {
                    "input": 0,
                    "output": 0,
                    "total": 0
                },
                "cost": 0.0
            },
            "monthly": {
                "month": current_month,
                "tokens": {
                    "input": 0,
                    "output": 0,
                    "total": 0
                },
                "cost": 0.0,
                "budget": self.monthly_budget
            },
            "history": [],
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_usage_data(self) -> None:
        """Save usage data to file"""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(self.usage_data, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving usage data: {e}")
    
    def _reset_daily_usage_if_needed(self) -> None:
        """Reset daily usage if date has changed"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.usage_data["daily"]["date"] != today:
            # Archive previous day's data
            prev_day = self.usage_data["daily"].copy()
            prev_day["id"] = f"day_{self.usage_data['daily']['date']}"
            self.usage_data["history"].append(prev_day)
            
            # Reset daily counters
            self.usage_data["daily"] = {
                "date": today,
                "tokens": {
                    "input": 0,
                    "output": 0,
                    "total": 0
                },
                "cost": 0.0
            }
    
    def _reset_monthly_usage_if_needed(self) -> None:
        """Reset monthly usage if month has changed"""
        current_month = datetime.now().strftime("%Y-%m")
        if self.usage_data["monthly"]["month"] != current_month:
            # Archive previous month's data
            prev_month = self.usage_data["monthly"].copy()
            prev_month["id"] = f"month_{self.usage_data['monthly']['month']}"
            self.usage_data["history"].append(prev_month)
            
            # Reset monthly counters
            self.usage_data["monthly"] = {
                "month": current_month,
                "tokens": {
                    "input": 0,
                    "output": 0,
                    "total": 0
                },
                "cost": 0.0,
                "budget": self.monthly_budget
            }
    
    def _calculate_cost(self, 
                       input_tokens: int, 
                       output_tokens: int, 
                       model: str) -> float:
        """
        Calculate cost for token usage
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name
            
        Returns:
            Cost in USD
        """
        # Get cost rates for the model, fallback to default if not found
        if model in self.cost_per_token:
            rates = self.cost_per_token[model]
        else:
            rates = self.cost_per_token["default"]
        
        # Calculate cost (rates are per 1K tokens)
        input_cost = (input_tokens / 1000) * rates["input"]
        output_cost = (output_tokens / 1000) * rates["output"]
        
        return input_cost + output_cost
    
    def track_usage(self, 
                   input_tokens: int, 
                   output_tokens: int, 
                   model: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Track token usage and calculate cost
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name (defaults to configured default)
            metadata: Additional metadata for the usage record
            
        Returns:
            Usage record with tokens and cost
        """
        with self.usage_lock:
            # Check if we need to reset daily/monthly usage
            self._reset_daily_usage_if_needed()
            self._reset_monthly_usage_if_needed()
            
            # Use default model if not specified
            model = model or self.default_model
            
            # Calculate cost
            cost = self._calculate_cost(input_tokens, output_tokens, model)
            
            # Update daily usage
            self.usage_data["daily"]["tokens"]["input"] += input_tokens
            self.usage_data["daily"]["tokens"]["output"] += output_tokens
            self.usage_data["daily"]["tokens"]["total"] += (input_tokens + output_tokens)
            self.usage_data["daily"]["cost"] += cost
            
            # Update monthly usage
            self.usage_data["monthly"]["tokens"]["input"] += input_tokens
            self.usage_data["monthly"]["tokens"]["output"] += output_tokens
            self.usage_data["monthly"]["tokens"]["total"] += (input_tokens + output_tokens)
            self.usage_data["monthly"]["cost"] += cost
            
            # Update last updated timestamp
            self.usage_data["last_updated"] = datetime.now().isoformat()
            
            # Create usage record
            usage_record = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens
                },
                "cost": cost,
                "metadata": metadata or {}
            }
            
            # Save usage data
            self._save_usage_data()
            
            return usage_record
    
    def check_limits(self) -> Tuple[bool, Optional[str]]:
        """
        Check if usage is within limits
        
        Returns:
            Tuple of (is_within_limits, reason_if_exceeded)
        """
        with self.usage_lock:
            # Check daily token limit
            daily_tokens = self.usage_data["daily"]["tokens"]["total"]
            if daily_tokens >= self.daily_token_limit:
                return False, f"Daily token limit exceeded: {daily_tokens}/{self.daily_token_limit}"
            
            # Check monthly budget
            monthly_cost = self.usage_data["monthly"]["cost"]
            if monthly_cost >= self.monthly_budget:
                return False, f"Monthly budget exceeded: ${monthly_cost:.2f}/${self.monthly_budget:.2f}"
            
            return True, None
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get current usage statistics
        
        Returns:
            Dictionary with usage statistics
        """
        with self.usage_lock:
            # Check if we need to reset daily/monthly usage
            self._reset_daily_usage_if_needed()
            self._reset_monthly_usage_if_needed()
            
            # Calculate remaining budget and tokens
            remaining_tokens = max(0, self.daily_token_limit - self.usage_data["daily"]["tokens"]["total"])
            remaining_budget = max(0, self.monthly_budget - self.usage_data["monthly"]["cost"])
            
            # Calculate usage percentages
            daily_usage_percent = min(100, (self.usage_data["daily"]["tokens"]["total"] / self.daily_token_limit) * 100) if self.daily_token_limit > 0 else 0
            monthly_usage_percent = min(100, (self.usage_data["monthly"]["cost"] / self.monthly_budget) * 100) if self.monthly_budget > 0 else 0
            
            # Get usage limits status
            limits_ok, limit_reason = self.check_limits()
            
            return {
                "daily": {
                    "date": self.usage_data["daily"]["date"],
                    "tokens": self.usage_data["daily"]["tokens"],
                    "cost": self.usage_data["daily"]["cost"],
                    "limit": self.daily_token_limit,
                    "remaining": remaining_tokens,
                    "usage_percent": daily_usage_percent
                },
                "monthly": {
                    "month": self.usage_data["monthly"]["month"],
                    "tokens": self.usage_data["monthly"]["tokens"],
                    "cost": self.usage_data["monthly"]["cost"],
                    "budget": self.monthly_budget,
                    "remaining": remaining_budget,
                    "usage_percent": monthly_usage_percent
                },
                "limits": {
                    "ok": limits_ok,
                    "reason": limit_reason
                },
                "last_updated": self.usage_data["last_updated"]
            }
    
    def get_usage_history(self, 
                         period: str = "daily", 
                         limit: int = 30) -> List[Dict[str, Any]]:
        """
        Get usage history for a specific period
        
        Args:
            period: Period type ('daily' or 'monthly')
            limit: Maximum number of records to return
            
        Returns:
            List of usage records
        """
        with self.usage_lock:
            # Filter history by period type
            prefix = f"{period}_"
            filtered_history = [
                record for record in self.usage_data["history"] 
                if "id" in record and record["id"].startswith(prefix)
            ]
            
            # Sort by date/month (descending)
            sorted_history = sorted(
                filtered_history, 
                key=lambda x: x["id"], 
                reverse=True
            )
            
            # Limit the number of records
            return sorted_history[:limit]
    
    def estimate_cost(self, 
                     input_text: str, 
                     expected_output_length: int, 
                     model: Optional[str] = None) -> Dict[str, Any]:
        """
        Estimate cost for processing text
        
        Args:
            input_text: Input text
            expected_output_length: Expected output length in tokens
            model: Model name (defaults to configured default)
            
        Returns:
            Dictionary with estimated tokens and cost
        """
        # Simple token estimation (4 chars ≈ 1 token)
        input_tokens = len(input_text) // 4
        output_tokens = expected_output_length
        
        # Use default model if not specified
        model = model or self.default_model
        
        # Calculate cost
        cost = self._calculate_cost(input_tokens, output_tokens, model)
        
        return {
            "model": model,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            "cost": cost
        }
    
    def update_limits(self, 
                     daily_token_limit: Optional[int] = None, 
                     monthly_budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Update usage limits
        
        Args:
            daily_token_limit: New daily token limit
            monthly_budget: New monthly budget in USD
            
        Returns:
            Dictionary with updated limits
        """
        with self.usage_lock:
            if daily_token_limit is not None:
                self.daily_token_limit = daily_token_limit
            
            if monthly_budget is not None:
                self.monthly_budget = monthly_budget
                self.usage_data["monthly"]["budget"] = monthly_budget
            
            # Save updated limits
            self._save_usage_data()
            
            return {
                "daily_token_limit": self.daily_token_limit,
                "monthly_budget": self.monthly_budget
            }

# Create a singleton instance
cost_monitor = CostMonitor()
