#!/bin/sh
# Install all requirements at container startup
pip install -r requirements.txt
# Install additional packages that might not be in requirements.txt
pip install langchain_deepseek langchain_community langgraph
# Start your application
exec uvicorn main:app --host 0.0.0.0
