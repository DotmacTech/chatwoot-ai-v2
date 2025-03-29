"""
Test script for LangSmith API key validation.
"""
import os
import sys
from langsmith import Client

def test_langsmith_api():
    """Test if the LangSmith API key is valid by attempting to connect to LangSmith."""
    # Get the API key from environment variable
    api_key = os.getenv("LANGCHAIN_API_KEY")
    
    if not api_key:
        print("Error: LANGCHAIN_API_KEY environment variable is not set.")
        return False
    
    try:
        # Initialize the client
        client = Client(api_key=api_key)
        
        # Test the API key by listing projects (a simple operation)
        projects = client.list_projects()
        
        # Convert to list to force API call
        projects_list = list(projects)
        
        print(f"Success! API key is valid. Found {len(projects_list)} projects:")
        for project in projects_list:
            print(f"- {project.name}")
        
        return True
    except Exception as e:
        print(f"Error: Could not connect to LangSmith API: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_langsmith_api()
    sys.exit(0 if success else 1)
