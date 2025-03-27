# tests/unit/test_workflow.py
import pytest
from chatwoot_langgraph.workflow import create_agent, AgentType, router
from langchain_core.messages import HumanMessage

@pytest.mark.unit
def test_agent_creation(mock_env):
    """Test specialized agent creation"""
    sales_agent = create_agent(AgentType.SALES)
    assert sales_agent.model == "deepseek-reasoner"
    assert "sales agent" in sales_agent.system[0].content.lower()
    
    support_agent = create_agent(AgentType.SUPPORT)
    assert "support agent" in support_agent.system[0].content.lower()

@pytest.mark.unit
def test_message_routing(mock_env):
    """Test message routing based on content"""
    # Test frustrated message
    state = {
        "messages": [HumanMessage(content="I'm extremely frustrated with my internet speed!")],
        "intent": None,
        "customer_id": "test123",
        "conversation_id": "conv456",
        "next_agent": None,
        "final_response": None
    }
    
    result = router(state)
    assert result["intent"] == "support"
    assert result["next_agent"] == "support"

    # Test happy message
    state = {
        "messages": [HumanMessage(content="I'm really happy with the service upgrade!")],
        "intent": None,
        "customer_id": "test123",
        "conversation_id": "conv456",
        "next_agent": None,
        "final_response": None
    }
    
    result = router(state)
    assert result["intent"] == "general"

    # Test neutral message
    state = {
        "messages": [HumanMessage(content="I need information about my bill.")],
        "intent": None,
        "customer_id": "test123",
        "conversation_id": "conv456",
        "next_agent": None,
        "final_response": None
    }
    
    result = router(state)
    assert result["intent"] == "account"