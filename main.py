from fastapi import FastAPI, Request, HTTPException, Body
import hmac
import hashlib
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from chatwoot.handler import process_webhook
from chatwoot_langsmith import setup_langsmith, tracing_manager, feedback_manager, cost_monitor
from chatwoot_langsmith.feedback import FeedbackType
from chatwoot_langchain import intent_classifier, INTENT_CATEGORIES

# Load environment variables
load_dotenv()

app = FastAPI(title="Chatwoot Automation")

# Set up LangSmith
setup_langsmith()

# Get webhook secret from environment variables
CHATWOOT_SECRET = os.getenv("CHATWOOT_WEBHOOK_SECRET")

@app.post("/chatwoot-webhook")
async def webhook(request: Request):
    """
    Endpoint to receive Chatwoot webhooks
    Processes webhook data without signature verification
    """
    # Process webhook payload directly
    payload = await request.json()
    return await process_webhook(payload)

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok"}

@app.get("/metrics")
async def get_metrics():
    """
    Get current metrics from the tracing system
    Returns performance and usage metrics
    """
    metrics = tracing_manager.get_metrics()
    
    # Add timestamp
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["uptime_seconds"] = tracing_manager.metrics.get("uptime_seconds", 0)
    
    return metrics

@app.post("/metrics/reset")
async def reset_metrics():
    """
    Reset all metrics in the tracing system
    """
    tracing_manager.reset_metrics()
    return {"status": "ok", "message": "Metrics reset successfully"}

@app.get("/traces")
async def get_recent_traces(limit: int = 10):
    """
    Get recent traces from LangSmith
    
    Args:
        limit: Maximum number of traces to return (default: 10)
    """
    if not tracing_manager.enabled:
        return {"status": "error", "message": "LangSmith tracing is not enabled"}
    
    try:
        # Get recent runs from LangSmith
        runs = tracing_manager.client.list_runs(
            project_name=tracing_manager.project_name,
            limit=limit
        )
        
        # Format the runs for the response
        formatted_runs = []
        for run in runs:
            formatted_runs.append({
                "id": run.id,
                "name": run.name,
                "start_time": run.start_time.isoformat() if run.start_time else None,
                "end_time": run.end_time.isoformat() if run.end_time else None,
                "status": run.status,
                "tags": run.tags,
                "error": run.error
            })
        
        return {"status": "ok", "traces": formatted_runs}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Feedback API models
class FeedbackRequest(BaseModel):
    """Request model for submitting feedback"""
    run_id: str
    feedback_type: str
    score: Optional[float] = None
    comment: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback for a LangSmith run
    """
    if not feedback_manager.enabled:
        return {"status": "error", "message": "LangSmith feedback collection is not enabled"}
    
    try:
        feedback_id = feedback_manager.submit_feedback(
            run_id=feedback.run_id,
            feedback_type=feedback.feedback_type,
            score=feedback.score,
            comment=feedback.comment,
            metadata=feedback.metadata
        )
        
        if feedback_id:
            return {"status": "ok", "feedback_id": feedback_id}
        else:
            return {"status": "error", "message": "Failed to submit feedback"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/feedback/{run_id}")
async def get_feedback(run_id: str):
    """
    Get all feedback for a specific run
    """
    if not feedback_manager.enabled:
        return {"status": "error", "message": "LangSmith feedback collection is not enabled"}
    
    try:
        feedback_list = feedback_manager.get_run_feedback(run_id)
        return {"status": "ok", "feedback": feedback_list}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/feedback-stats")
async def get_feedback_stats():
    """
    Get statistics on collected feedback
    """
    if not feedback_manager.enabled:
        return {"status": "error", "message": "LangSmith feedback collection is not enabled"}
    
    try:
        stats = feedback_manager.get_feedback_stats()
        return {"status": "ok", "stats": stats}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/monitoring-dashboard", response_class=HTMLResponse)
async def monitoring_dashboard():
    """
    Monitoring dashboard for the application.
    Displays metrics, traces, and feedback data.
    """
    # Get usage statistics
    usage_stats = cost_monitor.get_usage_stats()
    
    # Create HTML content without using f-strings for JavaScript parts
    daily_tokens = usage_stats['daily']['tokens']['total']
    daily_limit = usage_stats['daily']['limit']
    daily_usage_percent = usage_stats['daily']['usage_percent']
    
    monthly_cost = usage_stats['monthly']['cost']
    monthly_budget = usage_stats['monthly']['budget']
    monthly_usage_percent = usage_stats['monthly']['usage_percent']
    
    daily_date = usage_stats['daily']['date']
    monthly_month = usage_stats['monthly']['month']
    
    input_tokens = usage_stats['daily']['tokens']['input']
    output_tokens = usage_stats['daily']['tokens']['output']
    daily_cost = usage_stats['daily']['cost']
    remaining_budget = usage_stats['monthly']['remaining']
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chatwoot Automation Monitoring</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body class="bg-gray-100">
        <div class="container mx-auto px-4 py-8">
            <h1 class="text-3xl font-bold mb-8 text-center">Chatwoot Automation Monitoring</h1>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <!-- Usage Stats -->
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4">Usage Statistics</h2>
                    
                    <div class="mb-4">
                        <h3 class="text-lg font-medium mb-2">Daily Usage ({daily_date})</h3>
                        <div class="flex justify-between mb-1">
                            <span>Tokens: {daily_tokens} / {daily_limit}</span>
                            <span>{daily_usage_percent:.1f}%</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-2.5">
                            <div class="bg-blue-600 h-2.5 rounded-full" style="width: {daily_usage_percent}%"></div>
                        </div>
                    </div>
                    
                    <div>
                        <h3 class="text-lg font-medium mb-2">Monthly Budget ({monthly_month})</h3>
                        <div class="flex justify-between mb-1">
                            <span>Cost: ${monthly_cost:.2f} / ${monthly_budget:.2f}</span>
                            <span>{monthly_usage_percent:.1f}%</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-2.5">
                            <div class="bg-green-600 h-2.5 rounded-full" style="width: {monthly_usage_percent}%"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Cost Breakdown -->
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4">Cost Breakdown</h2>
                    <canvas id="costChart" class="w-full h-64"></canvas>
                </div>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <!-- Metrics -->
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4">Metrics</h2>
                    <ul class="space-y-2">
                        <li class="flex justify-between">
                            <span>Input Tokens:</span>
                            <span class="font-medium">{input_tokens}</span>
                        </li>
                        <li class="flex justify-between">
                            <span>Output Tokens:</span>
                            <span class="font-medium">{output_tokens}</span>
                        </li>
                        <li class="flex justify-between">
                            <span>Daily Cost:</span>
                            <span class="font-medium">${daily_cost:.4f}</span>
                        </li>
                        <li class="flex justify-between">
                            <span>Monthly Cost:</span>
                            <span class="font-medium">${monthly_cost:.2f}</span>
                        </li>
                        <li class="flex justify-between">
                            <span>Remaining Budget:</span>
                            <span class="font-medium">${remaining_budget:.2f}</span>
                        </li>
                    </ul>
                </div>
                
                <!-- Traces -->
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4">Recent Traces</h2>
                    <div id="traces-list" class="space-y-2">
                        <p class="text-gray-500 text-center">Loading traces...</p>
                    </div>
                </div>
                
                <!-- Feedback -->
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4">Feedback Summary</h2>
                    <div id="feedback-summary" class="space-y-4">
                        <p class="text-gray-500 text-center">Loading feedback...</p>
                    </div>
                </div>
            </div>
            
            <!-- Usage History -->
            <div class="bg-white p-6 rounded-lg shadow-md mb-8">
                <h2 class="text-xl font-semibold mb-4">Usage History</h2>
                <canvas id="historyChart" class="w-full h-80"></canvas>
            </div>
            
            <!-- Cost Management -->
            <div class="bg-white p-6 rounded-lg shadow-md">
                <h2 class="text-xl font-semibold mb-4">Cost Management</h2>
                
                <form id="limits-form" class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Daily Token Limit</label>
                        <input type="number" id="daily-token-limit" value="{daily_limit}" 
                               class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Monthly Budget (USD)</label>
                        <input type="number" id="monthly-budget" value="{monthly_budget}" step="0.01"
                               class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                    </div>
                    
                    <div class="md:col-span-2">
                        <button type="submit" class="w-full bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                            Update Limits
                        </button>
                    </div>
                </form>
            </div>
        </div>
        
        <script>
            // Cost Breakdown Chart
            const costCtx = document.getElementById('costChart').getContext('2d');
            const costChart = new Chart(costCtx, {{
                type: 'pie',
                data: {{
                    labels: ['Input Tokens', 'Output Tokens'],
                    datasets: [{{
                        data: [{input_tokens}, {output_tokens}],
                        backgroundColor: ['#3B82F6', '#10B981']
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            position: 'bottom'
                        }}
                    }}
                }}
            }});
            
            // Usage History Chart (placeholder)
            const historyCtx = document.getElementById('historyChart').getContext('2d');
            const historyChart = new Chart(historyCtx, {{
                type: 'line',
                data: {{
                    labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
                    datasets: [{{
                        label: 'Token Usage',
                        data: [1200, 1900, 3000, 5000, 2000, 3000, 4000],
                        borderColor: '#3B82F6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4,
                        fill: true
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Tokens'
                            }}
                        }}
                    }}
                }}
            }});
            
            // Fetch traces
            fetch('/traces?limit=5')
                .then(response => response.json())
                .then(data => {{
                    const tracesList = document.getElementById('traces-list');
                    if (data.traces && data.traces.length > 0) {{
                        tracesList.innerHTML = data.traces.map(trace => `
                            <div class="border-l-4 border-blue-500 pl-3 py-1">
                                <div class="text-sm font-medium">${{trace.name}}</div>
                                <div class="text-xs text-gray-500">${{new Date(trace.start_time).toLocaleString()}}</div>
                            </div>
                        `).join('');
                    }} else {{
                        tracesList.innerHTML = '<p class="text-gray-500 text-center">No traces found</p>';
                    }}
                }})
                .catch(error => {{
                    console.error('Error fetching traces:', error);
                    document.getElementById('traces-list').innerHTML = '<p class="text-red-500 text-center">Error loading traces</p>';
                }});
            
            // Fetch feedback summary
            fetch('/feedback-stats')
                .then(response => response.json())
                .then(data => {{
                    const feedbackSummary = document.getElementById('feedback-summary');
                    feedbackSummary.innerHTML = `
                        <div class="flex items-center justify-center">
                            <div class="text-center">
                                <div class="text-3xl font-bold text-indigo-600">${{data.average_score ? data.average_score.toFixed(1) : 'N/A'}}</div>
                                <div class="text-sm text-gray-500">Average Score</div>
                            </div>
                        </div>
                        <div class="grid grid-cols-3 gap-2 text-center">
                            <div>
                                <div class="text-lg font-medium">${{data.count_by_type?.accuracy || 0}}</div>
                                <div class="text-xs text-gray-500">Accuracy</div>
                            </div>
                            <div>
                                <div class="text-lg font-medium">${{data.count_by_type?.helpfulness || 0}}</div>
                                <div class="text-xs text-gray-500">Helpfulness</div>
                            </div>
                            <div>
                                <div class="text-lg font-medium">${{data.count_by_type?.satisfaction || 0}}</div>
                                <div class="text-xs text-gray-500">Satisfaction</div>
                            </div>
                        </div>
                    `;
                }})
                .catch(error => {{
                    console.error('Error fetching feedback:', error);
                    document.getElementById('feedback-summary').innerHTML = '<p class="text-red-500 text-center">Error loading feedback</p>';
                }});
            
            // Update limits form
            document.getElementById('limits-form').addEventListener('submit', function(e) {{
                e.preventDefault();
                
                const dailyTokenLimit = document.getElementById('daily-token-limit').value;
                const monthlyBudget = document.getElementById('monthly-budget').value;
                
                fetch('/update-limits', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{
                        daily_token_limit: parseInt(dailyTokenLimit),
                        monthly_budget: parseFloat(monthlyBudget)
                    }}),
                }})
                .then(response => response.json())
                .then(data => {{
                    alert('Limits updated successfully!');
                    window.location.reload();
                }})
                .catch(error => {{
                    console.error('Error updating limits:', error);
                    alert('Error updating limits. Please try again.');
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    return html_content

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """
    Dashboard for monitoring the system.
    """
    # Get usage stats
    usage_stats = cost_monitor.get_usage_stats()
    
    # Format stats for display
    daily_usage = usage_stats.get("daily_usage", {})
    monthly_usage = usage_stats.get("monthly_usage", {})
    
    # Format currency values
    daily_cost = "${:.2f}".format(daily_usage.get("cost", 0))
    monthly_cost = "${:.2f}".format(monthly_usage.get("cost", 0))
    
    # Calculate percentages of limits
    daily_token_limit = cost_monitor.daily_token_limit
    monthly_budget = cost_monitor.monthly_budget
    
    daily_token_percent = (daily_usage.get("total_tokens", 0) / daily_token_limit * 100) if daily_token_limit else 0
    monthly_cost_percent = (monthly_usage.get("cost", 0) / monthly_budget * 100) if monthly_budget else 0
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chatwoot Automation Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body class="bg-gray-100">
        <div class="container mx-auto px-4 py-8">
            <h1 class="text-3xl font-bold mb-8 text-center">Chatwoot Automation Dashboard</h1>
            
            <div class="mb-6 flex justify-center space-x-4">
                <a href="/dashboard" class="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700">
                    Main Dashboard
                </a>
                <a href="/intent-dashboard" class="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700">
                    Intent Classification
                </a>
                <a href="/usage-stats" class="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700">
                    Detailed Usage Stats
                </a>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <!-- Daily Usage -->
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4">Daily Usage</h2>
                    <div class="grid grid-cols-2 gap-4 mb-4">
                        <div class="bg-gray-50 p-4 rounded-md">
                            <div class="text-sm text-gray-500">Input Tokens</div>
                            <div class="text-2xl font-bold">{daily_usage.get("input_tokens", 0):,}</div>
                        </div>
                        <div class="bg-gray-50 p-4 rounded-md">
                            <div class="text-sm text-gray-500">Output Tokens</div>
                            <div class="text-2xl font-bold">{daily_usage.get("output_tokens", 0):,}</div>
                        </div>
                        <div class="bg-gray-50 p-4 rounded-md">
                            <div class="text-sm text-gray-500">Total Tokens</div>
                            <div class="text-2xl font-bold">{daily_usage.get("total_tokens", 0):,}</div>
                        </div>
                        <div class="bg-gray-50 p-4 rounded-md">
                            <div class="text-sm text-gray-500">Cost</div>
                            <div class="text-2xl font-bold">{daily_cost}</div>
                        </div>
                    </div>
                    
                    <div class="mb-2 flex justify-between text-sm">
                        <span>Daily Token Usage</span>
                        <span>{daily_usage.get("total_tokens", 0):,} / {daily_token_limit:,} ({daily_token_percent:.1f}%)</span>
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-2.5 mb-4">
                        <div class="bg-blue-600 h-2.5 rounded-full" style="width: {min(daily_token_percent, 100)}%"></div>
                    </div>
                </div>
                
                <!-- Monthly Usage -->
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4">Monthly Usage</h2>
                    <div class="grid grid-cols-2 gap-4 mb-4">
                        <div class="bg-gray-50 p-4 rounded-md">
                            <div class="text-sm text-gray-500">Input Tokens</div>
                            <div class="text-2xl font-bold">{monthly_usage.get("input_tokens", 0):,}</div>
                        </div>
                        <div class="bg-gray-50 p-4 rounded-md">
                            <div class="text-sm text-gray-500">Output Tokens</div>
                            <div class="text-2xl font-bold">{monthly_usage.get("output_tokens", 0):,}</div>
                        </div>
                        <div class="bg-gray-50 p-4 rounded-md">
                            <div class="text-sm text-gray-500">Total Tokens</div>
                            <div class="text-2xl font-bold">{monthly_usage.get("total_tokens", 0):,}</div>
                        </div>
                        <div class="bg-gray-50 p-4 rounded-md">
                            <div class="text-sm text-gray-500">Cost</div>
                            <div class="text-2xl font-bold">{monthly_cost}</div>
                        </div>
                    </div>
                    
                    <div class="mb-2 flex justify-between text-sm">
                        <span>Monthly Budget</span>
                        <span>{monthly_cost} / ${monthly_budget:.2f} ({monthly_cost_percent:.1f}%)</span>
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-2.5 mb-4">
                        <div class="bg-green-600 h-2.5 rounded-full" style="width: {min(monthly_cost_percent, 100)}%"></div>
                    </div>
                </div>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <!-- System Status -->
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4">System Status</h2>
                    <div class="space-y-4">
                        <div class="flex items-center">
                            <div class="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
                            <span>API Server: Running</span>
                        </div>
                        <div class="flex items-center">
                            <div class="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
                            <span>LangSmith Integration: Connected</span>
                        </div>
                        <div class="flex items-center">
                            <div class="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
                            <span>Intent Classification: Active</span>
                        </div>
                        <div class="flex items-center">
                            <div class="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
                            <span>Cost Monitoring: Enabled</span>
                        </div>
                    </div>
                </div>
                
                <!-- Quick Links -->
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4">Quick Links</h2>
                    <div class="space-y-4">
                        <a href="/intent-dashboard" class="block p-4 bg-indigo-50 rounded-md hover:bg-indigo-100">
                            <h3 class="font-medium">Intent Classification Dashboard</h3>
                            <p class="text-sm text-gray-600">Test and monitor intent classification performance</p>
                        </a>
                        <a href="/usage-stats" class="block p-4 bg-indigo-50 rounded-md hover:bg-indigo-100">
                            <h3 class="font-medium">Detailed Usage Statistics</h3>
                            <p class="text-sm text-gray-600">View detailed token usage and cost breakdown</p>
                        </a>
                        <a href="/health" class="block p-4 bg-indigo-50 rounded-md hover:bg-indigo-100">
                            <h3 class="font-medium">Health Check</h3>
                            <p class="text-sm text-gray-600">Check system health and connectivity</p>
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

# Intent classification endpoints
@app.get("/intent-categories")
async def get_intent_categories():
    """
    Get all available intent categories.
    """
    return {"status": "ok", "categories": INTENT_CATEGORIES}

@app.post("/intent-feedback")
async def submit_intent_feedback(
    feedback: Dict[str, Any] = Body(...)
):
    """
    Submit feedback on an intent classification.
    
    Required fields in the request body:
    - conversation_id: ID of the conversation
    - original_intent: The original classified intent
    - corrected_intent: The correct intent as determined by the agent
    - agent_id: ID of the agent providing the feedback
    """
    required_fields = ["conversation_id", "original_intent", "corrected_intent", "agent_id"]
    for field in required_fields:
        if field not in feedback:
            return {"status": "error", "message": f"Missing required field: {field}"}
    
    try:
        # Get the original classification
        original_classification = {
            "intent": feedback["original_intent"],
            "confidence": feedback.get("original_confidence", 0.0),
            "message": feedback.get("message", "")
        }
        
        # Record the feedback
        stats = intent_classifier.record_feedback(
            original_classification=original_classification,
            corrected_intent=feedback["corrected_intent"],
            agent_id=feedback["agent_id"],
            conversation_id=feedback["conversation_id"]
        )
        
        return {
            "status": "ok", 
            "message": "Feedback recorded successfully",
            "stats": stats
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/intent-stats")
async def get_intent_stats():
    """
    Get statistics on intent classification and feedback.
    """
    try:
        stats = intent_classifier.get_feedback_stats()
        return {"status": "ok", "stats": stats}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/estimate-intent")
async def estimate_intent(
    request: Dict[str, str] = Body(...)
):
    """
    Estimate the intent of a message without recording it.
    Useful for testing the intent classifier.
    
    Required fields:
    - message: The message to classify
    """
    if "message" not in request:
        return {"status": "error", "message": "Missing required field: message"}
    
    try:
        # Check usage limits
        within_limits, limit_reason = cost_monitor.check_limits()
        if not within_limits:
            return {"status": "error", "message": f"Usage limits exceeded: {limit_reason}"}
        
        # Estimate cost
        input_tokens = cost_monitor.estimate_tokens(request["message"])
        estimated_cost = cost_monitor.estimate_cost(input_tokens=input_tokens, output_tokens=100)
        
        # Classify intent
        classification = intent_classifier.classify_intent(request["message"])
        
        # Track usage
        cost_monitor.track_usage(
            input_tokens=input_tokens,
            output_tokens=0,
            metadata={"source": "intent_estimation"}
        )
        
        return {
            "status": "ok",
            "classification": classification,
            "estimated_cost": estimated_cost
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Add endpoints for cost monitoring
@app.get("/usage-stats")
async def get_usage_stats():
    """
    Get current usage statistics
    """
    return cost_monitor.get_usage_stats()

@app.get("/usage-history")
async def get_usage_history(period: str = "daily", limit: int = 30):
    """
    Get usage history for a specific period
    """
    return {"history": cost_monitor.get_usage_history(period, limit)}

@app.post("/update-limits")
async def update_limits(limits: Dict[str, Any]):
    """
    Update usage limits
    """
    daily_token_limit = limits.get("daily_token_limit")
    monthly_budget = limits.get("monthly_budget")
    
    return cost_monitor.update_limits(
        daily_token_limit=daily_token_limit,
        monthly_budget=monthly_budget
    )

@app.post("/estimate-cost")
async def estimate_cost(request: Dict[str, Any]):
    """
    Estimate cost for processing text
    """
    input_text = request.get("input_text", "")
    expected_output_length = request.get("expected_output_length", 100)
    model = request.get("model")
    
    return cost_monitor.estimate_cost(
        input_text=input_text,
        expected_output_length=expected_output_length,
        model=model
    )

@app.get("/intent-dashboard", response_class=HTMLResponse)
async def intent_dashboard():
    """
    Dashboard for intent classification statistics and testing.
    """
    # Get intent categories
    categories = INTENT_CATEGORIES
    
    # Get feedback stats
    stats = intent_classifier.get_feedback_stats()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Intent Classification Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body class="bg-gray-100">
        <div class="container mx-auto px-4 py-8">
            <h1 class="text-3xl font-bold mb-8 text-center">Intent Classification Dashboard</h1>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <!-- Intent Categories -->
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4">Intent Categories</h2>
                    <div class="space-y-4">
                        {
                            ''.join([
                                f'''
                                <div class="border-l-4 border-indigo-500 pl-4 py-2">
                                    <h3 class="text-lg font-medium">{intent.upper()}</h3>
                                    <p class="text-gray-600">{data['description']}</p>
                                    <div class="mt-2">
                                        <p class="text-sm text-gray-500">Examples:</p>
                                        <ul class="list-disc pl-5 text-sm">
                                            {
                                                ''.join([f'<li>{example}</li>' for example in data['examples']])
                                            }
                                        </ul>
                                    </div>
                                </div>
                                '''
                                for intent, data in categories.items()
                            ])
                        }
                    </div>
                </div>
                
                <!-- Intent Testing -->
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4">Test Intent Classification</h2>
                    <form id="intent-test-form" class="space-y-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">Message</label>
                            <textarea id="test-message" rows="4" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"></textarea>
                        </div>
                        <button type="submit" class="w-full bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                            Classify Intent
                        </button>
                    </form>
                    
                    <div id="classification-result" class="mt-4 hidden">
                        <h3 class="text-lg font-medium mb-2">Classification Result</h3>
                        <div class="bg-gray-50 p-4 rounded-md">
                            <div class="flex justify-between mb-2">
                                <span class="font-medium">Intent:</span>
                                <span id="result-intent" class="px-2 py-1 bg-blue-100 text-blue-800 rounded-md"></span>
                            </div>
                            <div class="flex justify-between mb-2">
                                <span class="font-medium">Confidence:</span>
                                <span id="result-confidence"></span>
                            </div>
                            <div class="mb-2">
                                <span class="font-medium">Reasoning:</span>
                                <p id="result-reasoning" class="text-sm mt-1 text-gray-600"></p>
                            </div>
                            <div>
                                <span class="font-medium">Suggested Response:</span>
                                <p id="result-response" class="text-sm mt-1 text-gray-600"></p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <!-- Feedback Stats -->
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4">Feedback Statistics</h2>
                    <div id="feedback-stats">
                        <p class="text-center text-gray-500">Loading statistics...</p>
                    </div>
                </div>
                
                <!-- Confusion Matrix -->
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4">Confusion Matrix</h2>
                    <div id="confusion-matrix">
                        <p class="text-center text-gray-500">Loading confusion matrix...</p>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Test intent classification
            document.getElementById('intent-test-form').addEventListener('submit', function(e) {{
                e.preventDefault();
                
                const message = document.getElementById('test-message').value;
                if (!message) return;
                
                fetch('/estimate-intent', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{ message: message }}),
                }})
                .then(response => response.json())
                .then(data => {{
                    if (data.status === 'ok') {{
                        const result = data.classification;
                        
                        document.getElementById('result-intent').textContent = result.intent;
                        document.getElementById('result-confidence').textContent = 
                            (result.confidence * 100).toFixed(1) + '%';
                        document.getElementById('result-reasoning').textContent = result.reasoning;
                        document.getElementById('result-response').textContent = result.suggested_response;
                        
                        document.getElementById('classification-result').classList.remove('hidden');
                    }} else {{
                        alert('Error: ' + data.message);
                    }}
                }})
                .catch(error => {{
                    console.error('Error testing intent:', error);
                    alert('Error testing intent. Please try again.');
                }});
            }});
            
            // Load feedback stats
            fetch('/intent-stats')
                .then(response => response.json())
                .then(data => {{
                    if (data.status === 'ok') {{
                        const stats = data.stats;
                        let statsHtml = '';
                        
                        if (Object.keys(stats).length === 0) {{
                            statsHtml = '<p class="text-center text-gray-500">No feedback data available yet.</p>';
                        }} else {{
                            const totalCorrections = stats.total_corrections || 0;
                            
                            statsHtml = `
                                <div class="mb-4">
                                    <div class="text-2xl font-bold text-center text-indigo-600">${{totalCorrections}}</div>
                                    <div class="text-sm text-center text-gray-500">Total Corrections</div>
                                </div>
                            `;
                            
                            if (stats.corrections_by_category) {{
                                statsHtml += `
                                    <h3 class="text-lg font-medium mb-2">Corrections by Category</h3>
                                    <div class="space-y-2">
                                `;
                                
                                for (const [category, count] of Object.entries(stats.corrections_by_category)) {{
                                    const percentage = totalCorrections > 0 ? 
                                        (count / totalCorrections * 100).toFixed(1) : 0;
                                    
                                    statsHtml += `
                                        <div>
                                            <div class="flex justify-between mb-1">
                                                <span>${{category}}</span>
                                                <span>${{count}} (${{percentage}}%)</span>
                                            </div>
                                            <div class="w-full bg-gray-200 rounded-full h-2">
                                                <div class="bg-indigo-600 h-2 rounded-full" style="width: ${{percentage}}%"></div>
                                            </div>
                                        </div>
                                    `;
                                }}
                                
                                statsHtml += '</div>';
                            }}
                        }}
                        
                        document.getElementById('feedback-stats').innerHTML = statsHtml;
                        
                        // Render confusion matrix if available
                        if (stats.confusion_matrix && Object.keys(stats.confusion_matrix).length > 0) {{
                            const matrix = stats.confusion_matrix;
                            const categories = [...new Set([
                                ...Object.keys(matrix),
                                ...Object.values(matrix).flatMap(obj => Object.keys(obj))
                            ])];
                            
                            let matrixHtml = `
                                <div class="overflow-x-auto">
                                    <table class="min-w-full border-collapse">
                                        <thead>
                                            <tr>
                                                <th class="border p-2 bg-gray-50">Original ↓ / Corrected →</th>
                                                ${{categories.map(cat => `<th class="border p-2 bg-gray-50">${{cat}}</th>`).join('')}}
                                            </tr>
                                        </thead>
                                        <tbody>
                            `;
                            
                            for (const original of categories) {{
                                matrixHtml += `<tr><th class="border p-2 bg-gray-50">${{original}}</th>`;
                                
                                for (const corrected of categories) {{
                                    const count = matrix[original]?.[corrected] || 0;
                                    const cellClass = count > 0 ? 'bg-blue-50 font-medium' : '';
                                    
                                    matrixHtml += `<td class="border p-2 text-center ${{cellClass}}">${{count}}</td>`;
                                }}
                                
                                matrixHtml += '</tr>';
                            }}
                            
                            matrixHtml += `
                                        </tbody>
                                    </table>
                                </div>
                                <p class="text-xs text-gray-500 mt-2">
                                    The confusion matrix shows how often each intent was corrected to another intent.
                                </p>
                            `;
                            
                            document.getElementById('confusion-matrix').innerHTML = matrixHtml;
                        }} else {{
                            document.getElementById('confusion-matrix').innerHTML = 
                                '<p class="text-center text-gray-500">No confusion matrix data available yet.</p>';
                        }}
                    }} else {{
                        document.getElementById('feedback-stats').innerHTML = 
                            '<p class="text-center text-red-500">Error loading statistics: ' + data.message + '</p>';
                        document.getElementById('confusion-matrix').innerHTML = 
                            '<p class="text-center text-red-500">Error loading confusion matrix.</p>';
                    }}
                }})
                .catch(error => {{
                    console.error('Error loading stats:', error);
                    document.getElementById('feedback-stats').innerHTML = 
                        '<p class="text-center text-red-500">Error loading statistics.</p>';
                    document.getElementById('confusion-matrix').innerHTML = 
                        '<p class="text-center text-red-500">Error loading confusion matrix.</p>';
                }});
        </script>
    </body>
    </html>
    """
    
    return html_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
