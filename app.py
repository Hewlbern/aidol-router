from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import json
import asyncio
import logging
from dotenv import load_dotenv
from memory import (
    get_or_create_conversation,
    get_conversation_messages,
    add_message_to_conversation,
    load_conversation,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="OpenRouter Service")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",  # Next.js default ports
        "https://www.dreammachine.one",
        "https://dreammachine.one",
        "https://api.dreammachine.one",
        "https://www.api.dreammachine.one",
        "https://goonerai.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable must be set in .env file")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# System prompt to replicate ourdream.ai/chat experience
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are an advanced AI companion designed to engage users in meaningful and personalized conversations. "
    "Your responses should be empathetic, contextually relevant, and adapt to the user's emotional state. "
    "Maintain a friendly and supportive tone, and remember details from previous interactions to provide a "
    "cohesive and engaging experience. Avoid controversial topics and ensure that all interactions are "
    "appropriate and respectful."
)

# Default AI model
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "nousresearch/hermes-3-llama-3.1-70b")

# Optional OpenRouter headers
HTTP_REFERER = os.getenv("HTTP_REFERER", "")
X_TITLE = os.getenv("X_TITLE", "")

class ChatRequest(BaseModel):
    model: str = "nousresearch/hermes-3-llama-3.1-70b"
    messages: list[dict[str, str]]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 1000
    system_prompt: str | None = None
    conversation_id: str | None = None
    user_id: str | None = None  # Optional: user ID for database tracking
    character_id: str | None = None  # Optional: character ID for database tracking

class WebSocketMessage(BaseModel):
    type: str  # "start_conversation", "message", "ping"
    character_id: str | None = None
    user_id: str | None = None
    message: str | None = None
    conversation_id: str | None = None
    system_prompt: str | None = None  # Optional: custom system prompt, uses DEFAULT_SYSTEM_PROMPT if not provided
    model: str | None = None  # Optional: AI model to use, uses DEFAULT_MODEL if not provided

def prepare_messages(request: ChatRequest, conversation_id: str) -> list[dict[str, str]]:
    """Prepare messages with system prompt and conversation history"""
    system_prompt = request.system_prompt or DEFAULT_SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]
    
    # Load conversation history if conversation_id is provided
    if conversation_id:
        history_messages = get_conversation_messages(conversation_id)
        if history_messages:
            # Filter out system messages from history (we'll add our own)
            history_messages = [msg for msg in history_messages if msg.get("role") != "system"]
            messages.extend(history_messages)
    
    # Add new messages from request
    # Check if first message is already a system message
    if request.messages and request.messages[0].get("role") == "system":
        # If system message provided, replace ours
        messages = request.messages
        if conversation_id:
            # Still add history after system message
            history_messages = get_conversation_messages(conversation_id)
            history_messages = [msg for msg in history_messages if msg.get("role") != "system"]
            messages.extend(history_messages)
    else:
        messages.extend(request.messages)
    
    return messages

def get_headers() -> dict[str, str]:
    """Get headers for OpenRouter API"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if HTTP_REFERER:
        headers["HTTP-Referer"] = HTTP_REFERER
    if X_TITLE:
        headers["X-Title"] = X_TITLE
    return headers

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Get conversation history by UUID (from JSON file - source of truth)
    
    Returns the full conversation including:
    - id: Conversation UUID
    - created_at: ISO timestamp when conversation was created
    - updated_at: ISO timestamp when conversation was last updated
    - messages: Array of message objects with role, content, and timestamp
    
    Example response:
    {
        "id": "uuid-here",
        "created_at": "2025-12-15T02:00:49.882510",
        "updated_at": "2025-12-15T02:00:56.084928",
        "messages": [
            {
                "role": "user",
                "content": "Hello!",
                "timestamp": "2025-12-15T02:00:52.394667"
            }
        ]
    }
    """
    try:
        conversation = load_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Conversation not found",
                    "conversation_id": conversation_id
                }
            )
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to load conversation",
                "conversation_id": conversation_id,
                "detail": str(e)
            }
        )


@app.post("/chat")
async def chat(request: ChatRequest):
    """Generate a non-streaming chat response"""
    # Get or create conversation (with optional user_id and character_id for database tracking)
    conversation_id = get_or_create_conversation(
        request.conversation_id,
        request.user_id,
        request.character_id
    )
    
    # Save user messages to conversation
    for msg in request.messages:
        if msg.get("role") != "system":
            add_message_to_conversation(conversation_id, msg["role"], msg["content"], request.user_id, request.character_id)
    
    # Prepare messages with history
    messages = prepare_messages(request, conversation_id)
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            OPENROUTER_API_URL,
            headers=get_headers(),
            json={
                "model": request.model,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            },
            timeout=60.0,
        )
        if response.status_code != 200:
            error_detail = response.text
            return {"error": f"OpenRouter API error: {response.status_code}", "detail": error_detail}, response.status_code
        
        result = response.json()
        
        # Save assistant response to conversation
        if "choices" in result and len(result["choices"]) > 0:
            assistant_message = result["choices"][0].get("message", {}).get("content", "")
            if assistant_message:
                add_message_to_conversation(conversation_id, "assistant", assistant_message, request.user_id, request.character_id)
        
        # Add conversation_id to response
        result["conversation_id"] = conversation_id
        return result

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Generate a streaming chat response"""
    # Get or create conversation (with optional user_id and character_id for database tracking)
    conversation_id = get_or_create_conversation(
        request.conversation_id,
        request.user_id,
        request.character_id
    )
    
    # Save user messages to conversation
    for msg in request.messages:
        if msg.get("role") != "system":
            add_message_to_conversation(conversation_id, msg["role"], msg["content"], request.user_id, request.character_id)
    
    # Prepare messages with history
    messages = prepare_messages(request, conversation_id)
    
    # Collect full response for saving
    full_response = ""
    
    async def generate():
        nonlocal full_response
        # Send conversation_id as the first event so client can use it immediately
        yield f"data: {json.dumps({'conversation_id': conversation_id, 'type': 'conversation_start'})}\n\n"
        
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                OPENROUTER_API_URL,
                headers=get_headers(),
                json={
                    "model": request.model,
                    "messages": messages,
                    "stream": True,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                },
                timeout=60.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            # Save complete assistant response
                            if full_response:
                                add_message_to_conversation(conversation_id, "assistant", full_response, request.user_id, request.character_id)
                            # Send done event with conversation_id
                            yield f"data: {json.dumps({'conversation_id': conversation_id, 'type': 'done'})}\n\n"
                            break
                        try:
                            # Parse and extract content
                            chunk_data = json.loads(data)
                            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                delta = chunk_data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_response += content
                            yield f"data: {data}\n\n"
                        except json.JSONDecodeError:
                            continue

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat with character
    
    Message format:
    {
        "type": "start_conversation" | "message",
        "character_id": "uuid",
        "user_id": "uuid",
        "message": "user message content",
        "conversation_id": "uuid" (optional, for continuing conversation)
    }
    """
    await websocket.accept()
    conversation_id = None
    character_id = None
    user_id = None
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            try:
                message_data = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid JSON format"
                })
                continue
            
            message_type = message_data.get("type")
            
            # Handle ping/pong for connection keepalive
            if message_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            
            # Start new conversation
            if message_type == "start_conversation":
                character_id = message_data.get("character_id")
                user_id = message_data.get("user_id")
                initial_message = message_data.get("message", "")
                provided_conversation_id = message_data.get("conversation_id")  # Optional: use existing chat ID
                system_prompt = message_data.get("system_prompt")  # Optional: custom system prompt
                
                if not character_id:
                    await websocket.send_json({
                        "type": "error",
                        "error": "character_id is required"
                    })
                    continue
                
                # Create new conversation or use provided conversation_id
                # If conversation_id is provided (e.g., from frontend chat ID), use it
                # Otherwise, create a new one
                conversation_id = get_or_create_conversation(
                    provided_conversation_id,  # Use provided ID if available
                    user_id,
                    character_id
                )
                
                # Send conversation_id to client
                await websocket.send_json({
                    "type": "conversation_started",
                    "conversation_id": conversation_id
                })
                
                # If there's an initial message, process it
                if initial_message:
                    model = message_data.get("model")  # Optional: custom model
                    await process_websocket_message(
                        websocket,
                        conversation_id,
                        initial_message,
                        character_id,
                        user_id,
                        system_prompt,
                        model
                    )
            
            # Send message in existing conversation
            elif message_type == "message":
                message_content = message_data.get("message")
                conversation_id = message_data.get("conversation_id") or conversation_id
                character_id = message_data.get("character_id") or character_id
                user_id = message_data.get("user_id") or user_id
                system_prompt = message_data.get("system_prompt")  # Optional: custom system prompt
                model = message_data.get("model")  # Optional: custom model
                
                if not message_content:
                    await websocket.send_json({
                        "type": "error",
                        "error": "message is required"
                    })
                    continue
                
                if not conversation_id:
                    await websocket.send_json({
                        "type": "error",
                        "error": "conversation_id is required. Send start_conversation first."
                    })
                    continue
                
                await process_websocket_message(
                    websocket,
                    conversation_id,
                    message_content,
                    character_id,
                    user_id,
                    system_prompt,
                    model
                )
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "error": f"Unknown message type: {message_type}"
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for conversation {conversation_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass

async def process_websocket_message(
    websocket: WebSocket,
    conversation_id: str,
    message_content: str,
    character_id: str | None,
    user_id: str | None,
    system_prompt: str | None = None,
    model: str | None = None
):
    """
    Process a message and stream the response via WebSocket
    
    Args:
        websocket: WebSocket connection
        conversation_id: UUID of the conversation
        message_content: User's message content
        character_id: Optional character ID
        user_id: Optional user ID
        system_prompt: Optional custom system prompt (uses DEFAULT_SYSTEM_PROMPT if not provided)
        model: Optional AI model to use (uses DEFAULT_MODEL if not provided)
    """
    try:
        # Save user message to Supabase Storage (JSON is source of truth)
        logger.info(f"Saving user message to conversation {conversation_id}: {message_content[:50]}...")
        add_message_to_conversation(conversation_id, "user", message_content, user_id, character_id)
        logger.info(f"User message saved successfully to conversation {conversation_id}")
        
        # Notify client that message was received
        await websocket.send_json({
            "type": "message_received",
            "conversation_id": conversation_id
        })
        
        # Prepare messages with history
        # Use provided system_prompt or fall back to DEFAULT_SYSTEM_PROMPT
        history_messages = get_conversation_messages(conversation_id)
        prompt_to_use = system_prompt or DEFAULT_SYSTEM_PROMPT
        messages = [{"role": "system", "content": prompt_to_use}]
        messages.extend([msg for msg in history_messages if msg.get("role") != "system"])
        messages.append({"role": "user", "content": message_content})
        
        # Use provided model or fall back to DEFAULT_MODEL
        model_to_use = model or DEFAULT_MODEL
        
        # Stream response from OpenRouter
        full_response = ""
        await websocket.send_json({
            "type": "response_start",
            "conversation_id": conversation_id
        })
        
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                OPENROUTER_API_URL,
                headers=get_headers(),
                json={
                    "model": model_to_use,
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.7,
                    "max_tokens": 1000,
                },
                timeout=60.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            # Save complete assistant response
                            if full_response:
                                logger.info(f"Saving assistant response to conversation {conversation_id}: {len(full_response)} chars")
                                add_message_to_conversation(conversation_id, "assistant", full_response, user_id, character_id)
                                logger.info(f"Assistant response saved successfully to conversation {conversation_id}")
                            await websocket.send_json({
                                "type": "response_complete",
                                "conversation_id": conversation_id
                            })
                            break
                        try:
                            chunk_data = json.loads(data)
                            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                delta = chunk_data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_response += content
                                    # Send chunk to client
                                    await websocket.send_json({
                                        "type": "chunk",
                                        "content": content,
                                        "conversation_id": conversation_id
                                    })
                        except json.JSONDecodeError:
                            continue
        
    except Exception as e:
        logger.error(f"Error processing WebSocket message: {e}")
        await websocket.send_json({
            "type": "error",
            "error": str(e),
            "conversation_id": conversation_id
        })

