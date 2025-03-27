"""
LangSmith feedback collection module.
Provides mechanisms for collecting, storing, and analyzing user feedback.
"""
import os
import logging
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv
import langsmith as ls
from langsmith.schemas import Run, Feedback, FeedbackSourceType

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class FeedbackType(str, Enum):
    """Types of feedback that can be collected"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    SCORE = "score"
    CORRECTION = "correction"
    FREE_FORM = "free_form"
    CUSTOMER_SATISFACTION = "customer_satisfaction"
    AGENT_SATISFACTION = "agent_satisfaction"
    ACCURACY = "accuracy"
    HELPFULNESS = "helpfulness"

class FeedbackManager:
    """
    Manages feedback collection, storage, and analysis.
    Provides mechanisms for submitting feedback on LangSmith runs.
    """
    
    def __init__(self):
        """Initialize feedback collection system"""
        self.api_key = os.getenv("LANGCHAIN_API_KEY")
        self.project_name = os.getenv("LANGCHAIN_PROJECT", "chatwoot-automation")
        
        if not self.api_key:
            logger.warning("LANGCHAIN_API_KEY not set. Feedback collection disabled.")
            self.enabled = False
        else:
            self.enabled = True
            self.client = ls.Client(api_key=self.api_key)
            logger.info(f"LangSmith feedback collection enabled for project: {self.project_name}")
        
        # Initialize feedback storage
        self.feedback_history = []
    
    def submit_feedback(self, 
                       run_id: str, 
                       feedback_type: Union[FeedbackType, str], 
                       score: Optional[float] = None, 
                       comment: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Submit feedback for a LangSmith run
        
        Args:
            run_id: ID of the run to submit feedback for
            feedback_type: Type of feedback
            score: Numeric score (if applicable)
            comment: Free-form comment
            metadata: Additional metadata
            
        Returns:
            Feedback ID if successful, None otherwise
        """
        if not self.enabled or not run_id:
            return None
        
        # Convert string to enum if needed
        if isinstance(feedback_type, str):
            try:
                feedback_type = FeedbackType(feedback_type)
            except ValueError:
                logger.warning(f"Invalid feedback type: {feedback_type}. Using FREE_FORM.")
                feedback_type = FeedbackType.FREE_FORM
        
        try:
            # Create feedback in LangSmith
            feedback = self.client.create_feedback(
                run_id=run_id,
                key=feedback_type.value,
                score=score,
                comment=comment,
                feedback_source_type=FeedbackSourceType.API,
                extra=metadata or {}
            )
            
            # Store in local history
            feedback_record = {
                "id": feedback.id,
                "run_id": run_id,
                "type": feedback_type.value,
                "score": score,
                "comment": comment,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            }
            self.feedback_history.append(feedback_record)
            
            logger.info(f"Submitted feedback for run {run_id}: {feedback_type.value}")
            return feedback.id
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return None
    
    def get_run_feedback(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Get all feedback for a specific run
        
        Args:
            run_id: ID of the run to get feedback for
            
        Returns:
            List of feedback items
        """
        if not self.enabled or not run_id:
            return []
            
        try:
            # Get feedback from LangSmith
            feedback_list = self.client.list_run_feedbacks(run_id=run_id)
            
            # Format the feedback
            formatted_feedback = []
            for feedback in feedback_list:
                formatted_feedback.append({
                    "id": feedback.id,
                    "key": feedback.key,
                    "score": feedback.score,
                    "comment": feedback.comment,
                    "created_at": feedback.created_at.isoformat() if feedback.created_at else None,
                    "modified_at": feedback.modified_at.isoformat() if feedback.modified_at else None
                })
            
            return formatted_feedback
        except Exception as e:
            logger.error(f"Error getting feedback for run {run_id}: {e}")
            return []
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics on collected feedback
        
        Returns:
            Dictionary of feedback statistics
        """
        if not self.enabled or not self.feedback_history:
            return {
                "total_feedback": 0,
                "feedback_by_type": {},
                "average_scores": {}
            }
        
        # Count feedback by type
        feedback_by_type = {}
        scores_by_type = {}
        
        for feedback in self.feedback_history:
            feedback_type = feedback["type"]
            
            # Count by type
            if feedback_type not in feedback_by_type:
                feedback_by_type[feedback_type] = 0
            feedback_by_type[feedback_type] += 1
            
            # Collect scores by type
            if feedback["score"] is not None:
                if feedback_type not in scores_by_type:
                    scores_by_type[feedback_type] = []
                scores_by_type[feedback_type].append(feedback["score"])
        
        # Calculate average scores
        average_scores = {}
        for feedback_type, scores in scores_by_type.items():
            if scores:
                average_scores[feedback_type] = sum(scores) / len(scores)
        
        return {
            "total_feedback": len(self.feedback_history),
            "feedback_by_type": feedback_by_type,
            "average_scores": average_scores
        }
    
    def create_feedback_form(self, 
                           run_id: str, 
                           feedback_types: List[FeedbackType] = None) -> Dict[str, Any]:
        """
        Create a feedback form configuration for a run
        
        Args:
            run_id: ID of the run to create a form for
            feedback_types: Types of feedback to collect
            
        Returns:
            Feedback form configuration
        """
        if not feedback_types:
            feedback_types = [
                FeedbackType.THUMBS_UP,
                FeedbackType.THUMBS_DOWN,
                FeedbackType.FREE_FORM
            ]
            
        return {
            "run_id": run_id,
            "feedback_types": [ft.value for ft in feedback_types],
            "timestamp": datetime.now().isoformat()
        }

# Create a singleton instance
feedback_manager = FeedbackManager()
