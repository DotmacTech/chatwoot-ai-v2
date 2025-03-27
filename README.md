# Chatwoot Automation with LangGraph

This project integrates Chatwoot with LangGraph to create a stateful, multi-agent workflow for handling customer interactions.

## Architecture

The system follows this flow:
[Customer] → [Chatwoot] → [LangSmith] → [LangGraph Workflow] → [Specialized Agents] → [Backend APIs] → [Response Generation] → [LangSmith] → [Chatwoot] → [Customer]

## Docker Setup

### Prerequisites

- Docker and Docker Compose installed on your system
- A `.env` file with all necessary environment variables

### Environment Variables

Create a `.env` file with the following variables:

```
# Chatwoot
CHATWOOT_BASE_URL=https://your-chatwoot-instance.com
CHATWOOT_API_TOKEN=your_api_token

# OpenAI
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL_NAME=gpt-3.5-turbo

# LangSmith
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=chatwoot-automation
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### Building and Running with Docker

1. Build the Docker image:

```bash
docker-compose build
```

2. Start the application:

```bash
docker-compose up -d
```

3. View logs:

```bash
docker-compose logs -f
```

4. Stop the application:

```bash
docker-compose down
```

## Development

### Running Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
uvicorn main:app --reload
```

## Testing

To test the webhook integration with Chatwoot, you can use a tool like ngrok to expose your local server to the internet:

```bash
ngrok http 8000
```

Then configure the webhook URL in Chatwoot to point to your ngrok URL.
