"""
Firebase Cloud Functions entry point for VoiceRad API.

Wraps the FastAPI backend as a Firebase Cloud Function (2nd gen)
using functions-framework and ASGI adapter.
"""

import os
import sys

# Ensure DEMO_MODE for Firebase deployment (GPU models not available in Cloud Functions)
os.environ.setdefault("DEMO_MODE", "1")

# Add the backend directory to the path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from firebase_functions import https_fn, options
from backend.app import app as fastapi_app

# Configure Cloud Function options
# - 256MB memory for demo mode (increase for GPU/model inference)
# - 60s timeout for API requests
# - Allow unauthenticated access (public API)
function_options = options.HttpsOptions(
    memory=options.MemoryOption.MB_256,
    timeout_sec=60,
    region="us-central1",
    cors=options.CorsOptions(
        cors_origins="*",
        cors_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    ),
)


@https_fn.on_request(opts=function_options)
def api(req: https_fn.Request) -> https_fn.Response:
    """
    Main Firebase Cloud Function that proxies all /api/* requests
    to the FastAPI application.

    Uses the ASGI adapter to bridge between Firebase's WSGI-style
    request/response and FastAPI's ASGI interface.
    """
    import asyncio
    from io import BytesIO

    # Convert Firebase request to ASGI scope
    scope = {
        "type": "http",
        "method": req.method,
        "path": req.path,
        "query_string": req.query_string,
        "headers": [(k.lower().encode(), v.encode()) for k, v in req.headers.items()],
        "root_path": "",
    }

    # Collect the response
    response_started = False
    response_status = 200
    response_headers = {}
    body_parts = []

    async def receive():
        body = req.get_data()
        return {"type": "http.request", "body": body or b""}

    async def send(message):
        nonlocal response_started, response_status, response_headers
        if message["type"] == "http.response.start":
            response_started = True
            response_status = message["status"]
            response_headers = {
                k.decode(): v.decode()
                for k, v in message.get("headers", [])
            }
        elif message["type"] == "http.response.body":
            body_parts.append(message.get("body", b""))

    # Run the ASGI app
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(fastapi_app(scope, receive, send))
    finally:
        loop.close()

    # Build response
    body = b"".join(body_parts)
    content_type = response_headers.get("content-type", "application/json")

    return https_fn.Response(
        response=body,
        status=response_status,
        headers=response_headers,
    )
