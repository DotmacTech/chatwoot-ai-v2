"""
LLM Service Module

Provides a unified interface for interacting with various Large Language Models.
Handles client initialization, request formatting, and response parsing.
"""

from typing import List, Dict, Any, Optional, Union

from langchain_openai import ChatOpenAI 
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import LLMResult, ChatResult, Generation, ChatGeneration
 
from core.config import settings
from utils.logging import AppLogger # Assuming AppLogger handles setup internally
from services.langsmith_service import LangSmithService 
 
logger = AppLogger(__name__)

class LLMService:
    """
    Service to manage interactions with configured Large Language Models.
    Currently configured primarily for OpenAI.
    """

    def __init__(self, langsmith_service: Optional[LangSmithService] = None):
        """
        Initializes the LLM client based on settings.
        Now configured for OpenAI.
        Requires LangSmithService for tracing.
        """
        # Use the model name from environment, default to OpenAI model
        self.model_name = settings.DEFAULT_MODEL
        
        self.api_key = settings.OPENAI_API_KEY # Use standard OpenAI key
        self.api_base = None # Use default OpenAI API base
        self.langsmith_service = langsmith_service # For tracing

        if not self.api_key:
            logger.error(f"OpenAI API Key (OPENAI_API_KEY) is not configured. LLMService cannot function.")
            self.client = None
        else:
            api_key_to_log = f"{self.api_key[:5]}...{self.api_key[-4:]}" if self.api_key and len(self.api_key) > 9 else "Invalid or short key"
            logger.info(f"Attempting to initialize ChatOpenAI with API Key: {api_key_to_log}")
            try:
                # Using standard ChatOpenAI class
                self.client = ChatOpenAI(
                    model=self.model_name,
                    openai_api_key=self.api_key,
                    temperature=settings.DEFAULT_LLM_TEMPERATURE,
                    max_tokens=settings.DEFAULT_LLM_MAX_TOKENS,
                    model_kwargs={
                        "top_p": settings.DEFAULT_LLM_TOP_P,
                        "n": settings.DEFAULT_LLM_N,
                        "stop": settings.DEFAULT_LLM_STOP,
                        # Note: stream is handled separately if needed, not usually here
                    },
                )
                logger.info(f"LLMService initialized for model '{self.model_name}' with default settings.")
            except Exception as e:
                logger.error(f"Failed to initialize LLM client ({self.model_name}): {e}", exc_info=True)
                self.client = None

    def is_available(self) -> bool:
        """Check if the LLM client was initialized successfully."""
        return self.client is not None

    def _truncate_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Truncate messages if the history is too long."""
        MAX_MESSAGES = 10 
        if len(messages) > MAX_MESSAGES:
             # Keep first message for context and last MAX_MESSAGES-1 for recency
            logger.info(f"Truncating conversation history from {len(messages)} to {MAX_MESSAGES} messages (1 first + {MAX_MESSAGES-1} most recent)")
            messages = [messages[0]] + messages[-(MAX_MESSAGES-1):]
        return messages
 
    async def generate_response(
         self,
         messages: List[BaseMessage],
         temperature: Optional[float] = None,
         tags: Optional[List[str]] = None,
         metadata: Optional[Dict[str, Any]] = None,
         parent_trace_id: Optional[str] = None 
     ) -> Optional[str]:
        """
        Generates a response from the LLM based on a list of messages.
        Uses default parameters from settings unless overridden.
        """
        if not self.is_available():
            logger.error("LLM client is not available. Cannot generate response.")
            return None

        # Use provided temperature or default from settings
        current_temperature = temperature if temperature is not None else settings.DEFAULT_LLM_TEMPERATURE

        trace_tags = ["llm_call", self.model_name] + (tags or [])
        trace_metadata = {
            "model": self.model_name,
            "temperature": current_temperature,
            "message_count": len(messages),
            **(metadata or {}) 
        }
        trace_inputs = {"messages": [msg.to_json() for msg in messages]} 

        # Use LangSmithService for tracing this specific LLM call
        trace_id = None
        langsmith_context = None
        if self.langsmith_service:
            langsmith_context = self.langsmith_service.trace(
                name="LLM Generation",
                run_type="llm",
                inputs=trace_inputs,
                tags=trace_tags,
                metadata=trace_metadata,
                trace_id_param_name='trace_id' # Pass trace_id into context manager
            )
            trace_id = langsmith_context.__enter__() # Manually enter context
        
        logger.debug(f"Invoking LLM '{self.model_name}' with {len(messages)} messages.", extra={"trace_id": trace_id, "parent_trace_id": parent_trace_id})
        try:
            # Truncate messages if necessary
            processed_messages = self._truncate_messages(messages)
            
            # Prepare config for ainvoke
            config = {
                "tags": trace_tags,
                "metadata": trace_metadata,
                "temperature": current_temperature,
            }
            # Add LangSmith callback if service is available
            if self.langsmith_service and trace_id:
                 # Assuming get_callback_handler exists and works like this
                config["callbacks"] = [self.langsmith_service.get_callback_handler(trace_id)]

            response: Union[ChatResult, AIMessage] = await self.client.ainvoke(
                processed_messages,
                config=config
            )
                
            content = None
            token_usage = {}
            outputs = {}

            # Extract content from the first generation
            if isinstance(response, ChatResult) and response.generations and isinstance(response.generations[0], ChatGeneration):
                content = response.generations[0].message.content
                token_usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
                logger.debug(f"LLM '{self.model_name}' generated response. Tokens: {token_usage}", extra={"trace_id": trace_id})
                outputs = {"response": content, "token_usage": token_usage}
                
            # Handle case where response is an AIMessage directly
            elif isinstance(response, AIMessage):
                content = response.content
                logger.debug(f"LLM '{self.model_name}' returned AIMessage directly: {content[:50]}...", extra={"trace_id": trace_id})
                outputs = {"response": content}
                
            else:
                logger.warning(f"LLM '{self.model_name}' returned an unexpected response structure: {response}", extra={"trace_id": trace_id})
                if self.langsmith_service and langsmith_context:
                    langsmith_context.__exit__(Exception, "Unexpected LLM response structure", None) # Exit trace with error
                return None

            # Log outputs and end trace if LangSmith is enabled
            if self.langsmith_service and langsmith_context:
                 langsmith_context.__exit__(None, None, None) # Exit trace successfully
                 # Update trace with outputs AFTER exiting context seems safer if end_trace is separate
                 # Or, if __exit__ handles outputs, adjust accordingly. Let's assume end_trace is needed.
                 # This part might need refinement based on how LangSmithService trace context manager works.
                 # If context exit handles logging outputs, this end_trace call is redundant/wrong.
                 # Let's comment out the explicit end_trace for now, assuming __exit__ handles it.
                 # self.langsmith_service.end_trace(trace_id, outputs=outputs) 

            return content

        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"Error during LLM call ({self.model_name}): {error_type}: {e}", exc_info=True, extra={"trace_id": trace_id})
            if self.langsmith_service and langsmith_context:
                 langsmith_context.__exit__(type(e), e, e.__traceback__) # Exit trace with error details
                 # self.langsmith_service.end_trace(trace_id, error=str(e)) # Comment out if exit handles it
            return None
        finally:
            # Ensure context manager is exited if an unexpected error occurred before __exit__
            # This might be redundant if try/except covers all paths, but adds safety.
            if self.langsmith_service and langsmith_context and not langsmith_context._is_exited():
                 try:
                     langsmith_context.__exit__(None, None, None)
                 except Exception as exit_e:
                     logger.error(f"Error exiting Langsmith trace context: {exit_e}", extra={"trace_id": trace_id})

    # --- Potential future methods ---
    # async def stream_response(...):
    # async def count_tokens(...):
    # async def get_embeddings(...):
