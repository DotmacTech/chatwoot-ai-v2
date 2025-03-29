"""
LangSmith integration module for message routing, tracing, and monitoring.
"""
import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager

import langsmith as ls
from langchain_core.tracers import LangChainTracer
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

# Configure logging
logger = logging.getLogger(__name__)

class LangSmithManager:
    """
    Manages LangSmith integration for message routing and tracing.
    Provides a central interface for all LangSmith operations.
    """
    
    def __init__(self):
        """Initialize LangSmith client and configuration"""
        self.api_key = os.getenv("LANGCHAIN_API_KEY")
        self.project_name = os.getenv("LANGCHAIN_PROJECT", "chatwoot-automation")
        self.default_tags = self._parse_tags(os.getenv("LANGSMITH_TAGS", "[]"))
        
        if not self.api_key:
            logger.warning("LANGCHAIN_API_KEY not set. LangSmith integration disabled.")
            self.enabled = False
            self.client = None
            self.tracer = None
        else:
            self.enabled = True
            self.client = ls.Client(api_key=self.api_key)
            self.tracer = LangChainTracer(
                project_name=self.project_name,
                client=self.client
            )
            logger.info(f"LangSmith integration enabled for project: {self.project_name}")
    
    def _parse_tags(self, tags_str: str) -> List[str]:
        """Parse tags from environment variable string"""
        if not tags_str or tags_str == "[]":
            return []
        
        tags_str = tags_str.strip("[]")
        if not tags_str:
            return []
            
        return [tag.strip().strip('"\'') for tag in tags_str.split(",")]
    
    @contextmanager
    def create_run(self, 
                  name: str,
                  inputs: Dict[str, Any],
                  tags: Optional[List[str]] = None,
                  metadata: Optional[Dict[str, Any]] = None):
        """
        Create a LangSmith run with context management.
        
        Args:
            name: Name of the run
            inputs: Input data for the run
            tags: Optional list of tags
            metadata: Optional metadata dictionary
        """
        if not self.enabled:
            try:
                logger.debug("LangSmith integration disabled. Using dummy run.")
                yield None
            except Exception as e:
                logger.error(f"Error in disabled LangSmith run: {str(e)}")
                raise
            return
            
        # Combine default tags with run-specific tags
        all_tags = self.default_tags.copy()
        if tags:
            all_tags.extend(tags)
            
        run = None
        try:
            # Create run
            try:
                logger.debug(f"Creating LangSmith run: {name} in project {self.project_name}")
                run = self.client.create_run(
                    name=name,
                    inputs=inputs,
                    run_type="chain",
                    project_name=self.project_name,
                    tags=all_tags,
                    extra=metadata or {}
                )
                if run:
                    logger.info(f"Successfully created LangSmith run: {run.id}")
                else:
                    logger.warning("LangSmith client returned None run object")
            except Exception as e:
                logger.error(f"Failed to create LangSmith run: {str(e)}", exc_info=True)
                logger.debug(f"Attempted run details - Name: {name}, Project: {self.project_name}, Tags: {all_tags}")
                
                # Create a dummy run context that does nothing
                yield None
                return
                
            # Start the run and provide it as context
            try:
                if run:
                    logger.debug(f"Starting LangSmith run: {run.id}")
                    self.client.update_run(run.id, {"status": "in_progress"})
                yield run
            except Exception as e:
                logger.error(f"Error during LangSmith run execution: {str(e)}", exc_info=True)
                yield None
                return
                
            # Update with success or failure
            try:
                if run:
                    self.client.update_run(
                        run.id,
                        {"status": "completed"}
                    )
                    logger.debug(f"Completed LangSmith run: {run.id}")
            except Exception as e:
                logger.error(f"Error updating LangSmith run status: {str(e)}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error in LangSmith run: {str(e)}", exc_info=True)
            yield None
    
    def record_feedback(self,
                       run_id: str,
                       key: str,
                       score: float,
                       comment: Optional[str] = None,
                       value: Optional[Any] = None,
                       feedback_type: Optional[str] = None) -> None:
        """
        Record feedback for a run.
        
        Args:
            run_id: ID of the run to record feedback for
            key: Key for the feedback
            score: Score value (typically between 0 and 1)
            comment: Optional comment
            value: Optional value associated with the feedback
            feedback_type: Optional type of feedback
        """
        if not self.enabled:
            logger.warning("LangSmith disabled, feedback not recorded")
            return
            
        try:
            self.client.create_feedback(
                run_id,
                key=key,
                score=score,
                comment=comment,
                value=value,
                feedback_type=feedback_type
            )
            logger.info(f"Recorded feedback for run {run_id}")
        except Exception as e:
            logger.error(f"Error recording feedback: {str(e)}")
    
    def get_run_feedback(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all feedback for a run"""
        if not self.enabled:
            return []
            
        try:
            feedbacks = self.client.list_feedback(run_ids=[run_id])
            return [feedback.dict() for feedback in feedbacks]
        except Exception as e:
            logger.error(f"Error getting feedback: {str(e)}")
            return []
    
    def wait_for_tracers(self):
        """Wait for all tracers to complete"""
        if self.enabled:
            wait_for_all_tracers()

# Create a singleton instance
langsmith_manager = LangSmithManager()

def setup_langsmith():
    """
    Initialize LangSmith integration.
    This function should be called at application startup.
    """
    if langsmith_manager.enabled:
        logger.info("LangSmith integration initialized")
    else:
        logger.warning("LangSmith integration disabled")
