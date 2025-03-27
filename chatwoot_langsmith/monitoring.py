"""
LangSmith monitoring and tracing module.
Provides detailed tracing, metrics collection, and monitoring capabilities.
"""
import os
import time
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
from dotenv import load_dotenv
import langsmith as ls
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from langchain.schema import LLMResult, AgentAction, AgentFinish

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class TracingManager:
    """
    Manages detailed tracing for LangChain components and custom operations.
    Provides metrics collection and monitoring capabilities.
    """
    
    def __init__(self):
        """Initialize tracing and monitoring components"""
        self.api_key = os.getenv("LANGCHAIN_API_KEY")
        self.project_name = os.getenv("LANGCHAIN_PROJECT", "chatwoot-automation")
        
        if not self.api_key:
            logger.warning("LANGCHAIN_API_KEY not set. Tracing disabled.")
            self.enabled = False
        else:
            self.enabled = True
            self.client = ls.Client(api_key=self.api_key)
            self.tracer = LangChainTracer(
                project_name=self.project_name,
                client=self.client
            )
            logger.info(f"LangSmith tracing enabled for project: {self.project_name}")
            
        # Initialize metrics storage
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "response_times": []
        }
    
    def get_callback_manager(self) -> Optional[CallbackManager]:
        """
        Get a callback manager with the tracer
        
        Returns:
            CallbackManager instance or None if tracing is disabled
        """
        if not self.enabled:
            return None
            
        return CallbackManager(handlers=[self.tracer])
    
    def create_trace(self, 
                    name: str, 
                    inputs: Dict[str, Any] = None,
                    metadata: Dict[str, Any] = None,
                    tags: List[str] = None) -> Optional[str]:
        """
        Create a new trace for a custom operation
        
        Args:
            name: Name of the trace
            inputs: Input data
            metadata: Additional metadata
            tags: Tags for the trace
            
        Returns:
            Trace ID if successful, None otherwise
        """
        if not self.enabled:
            return None
            
        try:
            # Use the langsmith client directly to create a run
            run = self.client.create_run(
                name=name,
                inputs=inputs or {},
                run_type="chain",
                project_name=self.project_name,
                tags=tags or [],
                metadata=metadata or {}
            )
            trace_id = run.id
                
            logger.debug(f"Created trace: {trace_id}")
            return trace_id
        except Exception as e:
            logger.error(f"Error creating trace: {e}")
            return None
    
    def end_trace(self, 
                 trace_id: str, 
                 outputs: Dict[str, Any] = None,
                 error: Optional[str] = None) -> bool:
        """
        End a trace with outputs or error
        
        Args:
            trace_id: ID of the trace to end
            outputs: Output data
            error: Error message if the operation failed
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not trace_id:
            return False
            
        try:
            if error:
                self.client.update_run(
                    run_id=trace_id,
                    error=error,
                    end_time=datetime.now()
                )
            else:
                self.client.update_run(
                    run_id=trace_id,
                    outputs=outputs or {},
                    end_time=datetime.now()
                )
                
            logger.debug(f"Ended trace: {trace_id}")
            return True
        except Exception as e:
            logger.error(f"Error ending trace: {e}")
            return False
    
    def trace_function(self, 
                      name: str, 
                      func: Callable, 
                      *args, 
                      metadata: Dict[str, Any] = None, 
                      tags: List[str] = None, 
                      **kwargs) -> Any:
        """
        Trace a function execution
        
        Args:
            name: Name of the trace
            func: Function to trace
            *args: Arguments for the function
            metadata: Additional metadata
            tags: Tags for the trace
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function
        """
        # Record metrics
        self.metrics["total_requests"] += 1
        start_time = time.time()
        
        # Create trace
        trace_id = self.create_trace(
            name=name,
            inputs={"args": str(args), "kwargs": str(kwargs)},
            metadata=metadata,
            tags=tags
        )
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Record success and end trace
            self.metrics["successful_requests"] += 1
            if trace_id:
                self.end_trace(trace_id, outputs={"result": str(result)})
                
            # Record response time
            response_time = time.time() - start_time
            self.metrics["response_times"].append(response_time)
            
            return result
        except Exception as e:
            # Record failure and end trace with error
            self.metrics["failed_requests"] += 1
            if trace_id:
                self.end_trace(trace_id, error=str(e))
            
            # Re-raise the exception
            raise
    
    def record_token_usage(self, tokens: int, cost: float = None) -> None:
        """
        Record token usage and cost
        
        Args:
            tokens: Number of tokens used
            cost: Cost of the tokens (optional)
        """
        self.metrics["total_tokens"] += tokens
        
        if cost is not None:
            self.metrics["total_cost"] += cost
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics
        
        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.copy()
        
        # Calculate average response time
        if metrics["response_times"]:
            metrics["avg_response_time"] = sum(metrics["response_times"]) / len(metrics["response_times"])
        else:
            metrics["avg_response_time"] = 0
            
        # Calculate success rate
        total = metrics["total_requests"]
        if total > 0:
            metrics["success_rate"] = metrics["successful_requests"] / total
        else:
            metrics["success_rate"] = 0
            
        return metrics
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "response_times": []
        }
        
    def traceable(self, run_type: str = "chain", name: str = None, metadata: Dict[str, Any] = None, tags: List[str] = None):
        """
        Decorator for tracing functions with LangSmith
        
        Args:
            run_type: Type of run (chain, llm, tool, etc.)
            name: Name for the trace (defaults to function name)
            metadata: Additional metadata
            tags: Tags for the trace
            
        Returns:
            Decorated function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                # Use function name if name not provided
                trace_name = name or func.__name__
                
                # Create trace
                trace_id = self.create_trace(
                    name=trace_name,
                    inputs={"args": str(args), "kwargs": str(kwargs)},
                    metadata=metadata,
                    tags=tags
                )
                
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # End trace with success
                    if trace_id:
                        self.end_trace(trace_id, outputs={"result": str(result)})
                    
                    return result
                except Exception as e:
                    # End trace with error
                    if trace_id:
                        self.end_trace(trace_id, error=str(e))
                    
                    # Re-raise the exception
                    raise
            
            return wrapper
        
        return decorator

# Create a singleton instance
tracing_manager = TracingManager()
