# tests/unit/test_workflow.py
import pytest
from chatwoot_langgraph.workflow import create_agent, AgentType, initial_analysis, analyze_sentiment
from langchain_core.messages import HumanMessage

@pytest.mark.unit
def test_agent_creation(mock_env):
    """Test specialized agent creation"""
    sales_agent = create_agent(AgentType.SALES)
    assert "deepseek-reasoner" in str(sales_agent)  # Check model name in string representation
    
    support_agent = create_agent(AgentType.SUPPORT)
    assert "deepseek-reasoner" in str(support_agent)  # Check model name in string representation

@pytest.mark.unit
def test_message_analysis(mock_env):
    """Test message analysis and sentiment detection"""
    # Test message sentiment analysis
    # Note: In test environment, the model might return default values
    message = "I need help with my service"
    sentiment = analyze_sentiment(message)
    
    # Check that we get a valid SentimentAnalysis object with expected fields
    assert hasattr(sentiment, "emotional_tone")
    assert hasattr(sentiment, "urgency")
    assert hasattr(sentiment, "sentiment_score")
    assert hasattr(sentiment, "key_concerns")
    
    # Urgency should be a number between 1-5
    assert 1 <= sentiment.urgency <= 5

@pytest.mark.unit
def test_initial_analysis(mock_env):
    """Test initial analysis of messages"""
    # Test with a message
    state = {
        "messages": [HumanMessage(content="I need help with my internet connection")],
        "intent": None,
        "customer_id": "test123",
        "conversation_id": "conv456",
        "next_step": None,
        "final_response": None
    }
    
    result = initial_analysis(state)
    
    # Check that we get the expected keys in the result
    assert "sentiment" in result
    assert "entities" in result
    assert "next_step" in result
    assert result["next_step"] == "verification_check"
    
    # Check that sentiment is a dictionary with expected keys
    assert "emotional_tone" in result["sentiment"]
    assert "urgency" in result["sentiment"]
    assert "sentiment_score" in result["sentiment"]
    assert "key_concerns" in result["sentiment"]
    
    # Check that we have a response message added
    assert len(result["messages"]) > len(state["messages"])