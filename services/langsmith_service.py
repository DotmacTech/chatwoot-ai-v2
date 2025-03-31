"""
Unified LangSmith Service Module.

Provides a single interface for interacting with LangSmith, including tracing,
feedback, and potentially cost monitoring.
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager
from datetime import datetime

import langsmith as ls
from langsmith.run_helpers import traceable
from langsmith.schemas import Run, Feedback
import langsmith.utils as ls_utils  # Import the utils module which contains LangSmithRateLimitError

from core.config import settings  # Centralized configuration
from utils.logging import AppLogger # Consistent logging

logger = AppLogger(__name__)

class LangSmithService:
    """
    Unified service for all LangSmith operations.

    Implements a singleton pattern to ensure a single instance manages the client
    and state throughout the application. Handles initialization, tracing,
    feedback, and potentially cost monitoring based on central configuration.
    """
    _instance: Optional["LangSmithService"] = None

    @classmethod
    def get_instance(cls) -> "LangSmithService":
        """Get or create the singleton instance."""
        if cls._instance is None:
            logger.info("Creating LangSmithService singleton instance.")
            cls._instance = cls()
        return cls._instance

    def __init__(self, api_key: Optional[str] = None, project_name: Optional[str] = None, 
                 default_tags: Optional[List[str]] = None):
        """
        Initialize the LangSmith service.
        
        Args:
            api_key: LangSmith API key. If None, will use settings.LANGCHAIN_API_KEY
            project_name: LangSmith project name. If None, will use settings.LANGCHAIN_PROJECT
            default_tags: Default tags to add to all traces
        """
        self.api_key = api_key or settings.LANGCHAIN_API_KEY
        self.project_name = project_name or settings.LANGCHAIN_PROJECT
        self.default_tags = default_tags or settings.LANGSMITH_TAGS
        self.enabled = settings.LANGSMITH_TRACING_ENABLED # Check if tracing is enabled globally
        if not self.enabled:
            logger.warning("LangSmith tracing is DISABLED via LANGSMITH_TRACING_ENABLED=False.")
            self.client = None
        elif not self.api_key:
            logger.error("LangSmith API Key (LANGCHAIN_API_KEY) is not configured. LangSmith tracing will be disabled.")
            self.client = None
            self.enabled = False # Disable if key is missing even if flag was true
        else:
            try:
                self.client = ls.Client(api_key=self.api_key)
                logger.info(f"LangSmith service initialized for project '{self.project_name}'")
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith client: {e}")
                self.enabled = False

        # Track rate limit status to avoid excessive logging
        self._rate_limited = False
        self._last_rate_limit_log = 0
        self._rate_limit_log_interval = 60  # Only log rate limit errors once per minute
        
    def _initialize_client(self):
        """Initialize the LangSmith client with error handling."""
        try:
            logger.info(f"Initializing LangSmith client for project '{self.project_name}' at '{self.api_url}'...")
            self.client = ls.Client(api_url=self.api_url, api_key=self.api_key)
            # Perform a lightweight test connection
            self._test_connection()
            logger.info("LangSmith client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LangSmith client: {e}", exc_info=True)
            logger.warning("Disabling LangSmith service due to initialization failure.")
            self.enabled = False
            self.client = None

    def _test_connection(self):
        """Test the LangSmith connection with a lightweight operation."""
        if not self.client:
            raise ConnectionError("LangSmith client is not initialized.")
        try:
            # Example: List projects (adjust based on common/stable API)
            # Using list_projects as a basic check
            _ = list(self.client.list_projects(limit=1))
            logger.info("LangSmith connection test successful.")
        except Exception as test_err:
            # Log the specific error but re-raise a more general ConnectionError
            logger.error(f"LangSmith connection test failed: {test_err}", exc_info=True)
            raise ConnectionError("LangSmith connection test failed.") from test_err

    # --- Tracing Methods ---

    def create_trace(self, name: str, inputs: Optional[Dict] = None, metadata: Optional[Dict] = None, 
                    tags: Optional[List[str]] = None, run_type: str = "chain") -> Optional[str]:
        """
        Starts a new trace (run) in LangSmith.

        Args:
            name: The name for the run.
            inputs: Input data for the run.
            metadata: Additional metadata for the run.
            tags: Tags to associate with the run.
            run_type: The type of run (e.g., 'chain', 'llm', 'tool').

        Returns:
            The trace_id (run_id) if successful, otherwise None or a fallback ID.
            Returns None immediately if the service is disabled.
        """
        if not self.enabled or not self.client:
            logger.debug(f"Skipping create_trace '{name}': LangSmith disabled or client not initialized.")
            return None

        final_tags = (self.default_tags or []) + (tags or [])

        try:
            run: Optional[Run] = self.client.create_run(
                name=name,
                inputs=inputs or {},
                run_type=run_type,
                project_name=self.project_name,
                tags=final_tags,
                extra={"metadata": metadata or {}}
            )

            if run is None:
                logger.warning(f"LangSmith client returned None for create_run('{name}'), using fallback ID.")
                return f"fallback-none-{int(time.time())}"

            trace_id = self._extract_run_id(run)
            if trace_id:
                logger.debug(f"Created trace: {trace_id} (Name: '{name}')")
                # Reset rate limit flag if we're successful
                if self._rate_limited:
                    logger.info("LangSmith rate limits no longer in effect")
                    self._rate_limited = False
                return trace_id
            else:
                 logger.warning(f"Could not extract ID from run object for '{name}': {run}. Using fallback ID.")
                 return f"fallback-extract-{hash(str(run))}-{int(time.time())}"

        except ls_utils.LangSmithRateLimitError as e:
            current_time = time.time()
            # Only log rate limit errors once per interval to avoid log spam
            if not self._rate_limited or (current_time - self._last_rate_limit_log > self._rate_limit_log_interval):
                logger.info(f"LangSmith rate limit exceeded. Tracing will be disabled temporarily. Error: {e}")
                self._rate_limited = True
                self._last_rate_limit_log = current_time
            return f"fallback-ratelimit-{int(time.time())}"
        except Exception as e:
            # Don't log with exc_info for common errors to reduce log noise
            if isinstance(e, (ConnectionError, TimeoutError)):
                logger.info(f"LangSmith connection issue: {e}. Using fallback ID.")
            else:
                logger.warning(f"Error creating trace '{name}': {e}. Using fallback ID.", exc_info=True)
            return f"fallback-exception-{int(time.time())}"

    def end_trace(self, trace_id: str, outputs: Optional[Dict] = None, error: Optional[Union[str, Exception]] = None) -> bool:
        """
        Ends a trace (run) with outputs or an error message.

        Args:
            trace_id: The ID of the trace (run) to end.
            outputs: Output data from the run.
            error: Error message or Exception if the run failed.

        Returns:
            True if the update was attempted (even if it failed), False if skipped.
        """
        if not self.enabled or not self.client or not trace_id or trace_id.startswith("fallback-"):
            logger.debug(f"Skipping end_trace for '{trace_id}': Service disabled, client missing, invalid ID, or fallback ID.")
            return False

        try:
            error_message = str(error) if error else None
            end_time = datetime.utcnow() # Use UTC for consistency

            self.client.update_run(
                run_id=trace_id,
                outputs=outputs if not error_message else None, # Don't send outputs if error occurred
                error=error_message,
                end_time=end_time
            )
            status = "failed" if error_message else "completed"
            logger.debug(f"Ended trace: {trace_id} (Status: {status})")
            return True
        except Exception as e:
            logger.warning(f"Error ending trace {trace_id}: {e}", exc_info=True)
            # We attempted, so return True, but log the failure.
            return True # Indicate an attempt was made

    def _extract_run_id(self, run_obj: Union[Run, Dict, Any]) -> Optional[str]:
         """Robustly extracts the run ID from various possible return types."""
         if isinstance(run_obj, Run) and hasattr(run_obj, 'id'):
             return str(run_obj.id)
         elif isinstance(run_obj, dict) and 'id' in run_obj:
             return str(run_obj['id'])
         # Add more checks if other formats are observed
         return None

    @contextmanager
    def trace(self, name: str, inputs: Optional[Dict] = None, metadata: Optional[Dict] = None, 
              tags: Optional[List[str]] = None, run_type: str = "chain"):
        """
        Context manager for tracing a block of code.

        Example:
            with langsmith_service.trace("my_function", inputs={"arg": 1}):
                # code to trace
                result = ...
                # No need to manually call end_trace, context manager handles it.
                # If an exception occurs, it's automatically logged.
        """
        if not self.enabled or not self.client:
            logger.debug(f"LangSmith client not available or tracing disabled. Skipping trace start for '{name}'.")
            yield None
            return

        trace_id = self.create_trace(name, inputs, metadata, tags, run_type)
        error_occurred = None
        try:
            yield trace_id # Yield the ID in case it's needed within the block
        except Exception as e:
            error_occurred = e
            # Reraise the exception after attempting to log it
            raise
        finally:
            # Only call end_trace if a valid trace_id was obtained
            if trace_id and not trace_id.startswith("fallback-"):
                 self.end_trace(trace_id, error=error_occurred) # Outputs need manual handling if using context manager

    # --- Feedback Methods ---

    def create_feedback(self, run_id: str, key: str, score: Optional[Union[float, int, bool]] = None, 
                       value: Optional[Any] = None, comment: Optional[str] = None, 
                       source_info: Optional[Dict] = None) -> Optional[str]:
        """
        Creates feedback for a specific run.

        Args:
            run_id: The ID of the run to provide feedback for.
            key: The feedback key (e.g., "user_rating", "correctness").
            score: Numerical or boolean score.
            value: Any JSON-serializable value for the feedback.
            comment: Text comment.
            source_info: Optional dictionary with source metadata (e.g., {"user_id": "..."}).

        Returns:
            The feedback ID if successful, None otherwise.
        """
        if not self.enabled or not self.client or not run_id or run_id.startswith("fallback-"):
            logger.debug(f"Skipping create_feedback for run '{run_id}': Service disabled, client missing, invalid ID, or fallback ID.")
            return None

        try:
            feedback: Feedback = self.client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                value=value,
                comment=comment,
                source_info=source_info
            )
            feedback_id = str(feedback.id) if feedback and hasattr(feedback, 'id') else None
            if feedback_id:
                logger.debug(f"Created feedback {feedback_id} for run {run_id} (Key: {key})")
                return feedback_id
            else:
                logger.warning(f"Failed to get feedback ID after creating feedback for run {run_id}")
                return None
        except Exception as e:
            logger.warning(f"Error creating feedback for run {run_id} (Key: {key}): {e}", exc_info=True)
            return None

    # --- Cost Monitoring Methods (Placeholder) ---
    # TODO: Migrate cost monitoring logic here if desired

    def log_cost(self, run_id: str, total_tokens: Optional[int] = None, cost_usd: Optional[float] = None):
        """
        Logs cost information for a specific run (placeholder).
        Actual implementation might involve updating run metadata or using specific LangSmith features if available.
        """
        if not self.enabled or not self.client or not run_id or run_id.startswith("fallback-"):
            return

        logger.debug(f"Placeholder: Logging cost for run {run_id} - Tokens: {total_tokens}, Cost: {cost_usd}")
        # Example: Update run metadata (check LangSmith API for best practices)
        # try:
        #     self.client.update_run(
        #         run_id=run_id,
        #         extra={
        #             "metadata": {
        #                 "cost_monitoring": {
        #                     "total_tokens": total_tokens,
        #                     "cost_usd": cost_usd
        #                 }
        #                 # Consider merging with existing metadata if applicable
        #             }
        #         }
        #     )
        # except Exception as e:
        #     logger.warning(f"Failed to update run {run_id} with cost metadata: {e}")
        pass

# Make the singleton instance easily accessible if needed, though using
# get_instance() is generally preferred for clarity.
# langsmith_service = LangSmithService.get_instance()
