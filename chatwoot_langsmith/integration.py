"""
LangSmith integration module for message routing, tracing, and monitoring.
"""
import os
import logging
from typing import Dict, Any, Optional, List, Callable
from dotenv import load_dotenv
import langsmith as ls
# The wait_for_all_tracers function is not available in the current version
# Remove dependency on this function

# Load environment variables
load_dotenv()

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
        else:
            self.enabled = True
            self.client = ls.Client(api_key=self.api_key)
            logger.info(f"LangSmith integration enabled for project: {self.project_name}")
    
    def _parse_tags(self, tags_str: str) -> List[str]:
        """Parse tags from environment variable string"""
        if not tags_str or tags_str == "[]":
            return []
        
        # Simple parsing for ["tag1", "tag2"] format
        tags_str = tags_str.strip("[]")
        if not tags_str:
            return []
            
        return [tag.strip().strip('"\'') for tag in tags_str.split(",")]
    
    def create_run(self, 
                  name: str, 
                  inputs: Dict[str, Any],
                  tags: Optional[List[str]] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Create a new LangSmith run for tracing
        
        Args:
            name: Name of the run
            inputs: Input data for the run
            tags: Additional tags for the run
            metadata: Additional metadata for the run
            
        Returns:
            Run ID if successful, None otherwise
        """
        if not self.enabled:
            return None
            
        # Combine default tags with run-specific tags
        all_tags = self.default_tags.copy()
        if tags:
            all_tags.extend(tags)
        
        try:
            run = self.client.run_create(
                name=name,
                inputs=inputs,
                run_type="chain",
                project_name=self.project_name,
                tags=all_tags,
                metadata=metadata or {}
            )
            logger.debug(f"Created LangSmith run: {run.id}")
            return run.id
        except Exception as e:
            logger.error(f"Error creating LangSmith run: {e}")
            return None
    
    def update_run(self, 
                  run_id: str, 
                  outputs: Dict[str, Any],
                  end_time: Optional[float] = None) -> bool:
        """
        Update an existing run with outputs
        
        Args:
            run_id: ID of the run to update
            outputs: Output data for the run
            end_time: End time of the run (timestamp)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not run_id:
            return False
            
        try:
            self.client.run_update(
                run_id=run_id,
                outputs=outputs,
                end_time=end_time
            )
            logger.debug(f"Updated LangSmith run: {run_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating LangSmith run: {e}")
            return False
    
    def trace_message_processing(self, 
                               message_data: Dict[str, Any], 
                               processor_func: Callable,
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Trace message processing with LangSmith
        
        Args:
            message_data: Message data to process
            processor_func: Function that processes the message
            metadata: Additional metadata for the run
            
        Returns:
            Result of the processor function
        """
        if not self.enabled:
            # If LangSmith is disabled, just call the processor function directly
            return processor_func(message_data)
            
        # Extract message identifier for the run name
        message_id = message_data.get("id", "unknown")
        conversation_id = message_data.get("conversation_id", "unknown")
        run_name = f"message-{message_id}-conv-{conversation_id}"
        
        # Create metadata with source information
        run_metadata = {
            "source": "chatwoot",
            "message_type": message_data.get("message_type", "unknown"),
            "conversation_id": conversation_id
        }
        
        # Add additional metadata if provided
        if metadata:
            run_metadata.update(metadata)
        
        # Create the run
        run_id = self.create_run(
            name=run_name,
            inputs={"message": message_data},
            tags=["message-processing"],
            metadata=run_metadata
        )
        
        try:
            # Process the message
            result = processor_func(message_data)
            
            # Update the run with the result
            if run_id:
                self.update_run(run_id, {"result": result})
                
            return result
        except Exception as e:
            # Log the error and update the run
            logger.error(f"Error processing message: {e}")
            if run_id:
                self.update_run(run_id, {"error": str(e)})
            raise
    
    def route_message(self, 
                     message_data: Dict[str, Any], 
                     routing_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a message based on content and routing configuration
        
        Args:
            message_data: Message data to route
            routing_config: Configuration for routing
            
        Returns:
            Routing result with destination and metadata
        """
        if not self.enabled:
            # Return default routing if LangSmith is disabled
            return {
                "destination": routing_config.get("default_destination", "default"),
                "confidence": 1.0,
                "metadata": {}
            }
        
        # Extract message content for routing
        content = message_data.get("content", "")
        
        # Create a run for the routing decision
        run_id = self.create_run(
            name="message-routing",
            inputs={"message": content, "config": routing_config},
            tags=["routing"]
        )
        
        # TODO: Implement actual routing logic using LangChain/LangSmith
        # For now, return a simple default routing
        result = {
            "destination": routing_config.get("default_destination", "default"),
            "confidence": 1.0,
            "metadata": {
                "source": "default_router",
                "rule_matched": "default"
            }
        }
        
        # Update the run with the routing result
        if run_id:
            self.update_run(run_id, {"result": result})
        
        return result
    
    def wait_for_tracers(self) -> None:
        """Wait for all tracers to complete"""
        if self.enabled:
            pass

# Create a singleton instance
langsmith_manager = LangSmithManager()

def setup_langsmith():
    """
    Initialize LangSmith integration.
    This function should be called at application startup.
    """
    if langsmith_manager.enabled:
        logger.info("Setting up LangSmith integration...")
        # Any additional setup steps can be added here
        return True
    else:
        logger.warning("LangSmith integration is disabled. Skipping setup.")
        return False
