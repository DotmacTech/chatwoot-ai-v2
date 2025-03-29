#!/bin/sh

echo "Starting entrypoint script..."

# Install all requirements at container startup
echo "Installing requirements..."
pip install -r requirements.txt

# Install additional packages that might not be in requirements.txt
echo "Installing additional packages..."
pip install --no-cache-dir langchain_deepseek langchain_community langgraph

# Wait for Redis to be ready
echo "Waiting for Redis..."
timeout 30s sh -c 'until nc -z $REDIS_HOST $REDIS_PORT; do echo "Waiting for Redis at $REDIS_HOST:$REDIS_PORT..."; sleep 1; done' || echo "Redis wait timeout"

# Convert log level to lowercase
LOG_LEVEL_LOWER=$(echo "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')

# Start application with proper logging
echo "Starting FastAPI application..."
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level ${LOG_LEVEL_LOWER} --reload
