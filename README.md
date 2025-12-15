# Router Backend - AI Idol Chat Service

FastAPI backend for managing chat interactions with AI Idols—intelligent companions that remember past conversations. Routes requests to OpenRouter API while maintaining persistent conversation memory stored both locally (JSON) and in Supabase.

## Features

- **Dual Storage**: Local JSON files + Supabase Storage for redundancy
- **Persistent Memory**: Full conversation history across sessions
- **Multiple APIs**: REST, WebSocket, and SSE streaming
- **User/Character Tracking**: Links conversations to users and AI Idol characters
- **Context-Aware**: Automatically includes history in API requests

## Setup

**Prerequisites:** Python 3.12+, Supabase account, OpenRouter API key

```bash
# Install dependencies
make install  # or pip install -r requirements.txt

# Create .env file
OPENROUTER_API_KEY=your_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_STORAGE_BUCKET=conversations

# Run
make run  # or docker-compose up -d
```

## API Endpoints

**Health Check**
```
GET /health
```

**Get Conversation**
```
GET /conversation/{conversation_id}
```

**Chat (Non-Streaming)**
```
POST /chat
Body: {
  "model": "nousresearch/hermes-3-llama-3.1-70b",
  "messages": [{"role": "user", "content": "Hello!"}],
  "conversation_id": "optional-uuid",
  "user_id": "optional-uuid",
  "character_id": "optional-uuid"
}
```

**Chat (Streaming)**
```
POST /chat/stream
Body: Same as /chat
```

**WebSocket Chat**
```
WS /ws/chat
Message: {
  "type": "start_conversation" | "message" | "ping",
  "character_id": "uuid",
  "user_id": "uuid",
  "message": "content",
  "conversation_id": "uuid (optional)"
}
```

## Conversation Memory

Conversations are saved with full history. When `conversation_id` is provided, history is automatically loaded and included in API requests.

```json
{
  "id": "conversation-uuid",
  "created_at": "2025-12-15T00:18:46.655592",
  "updated_at": "2025-12-15T00:18:51.271819",
  "messages": [
    {"role": "user", "content": "Hello!", "timestamp": "..."},
    {"role": "assistant", "content": "Hello!", "timestamp": "..."}
  ]
}
```

## Supabase Setup

1. Create project at [supabase.com](https://supabase.com)
2. Get URL and service role key from Settings > API
3. Storage bucket is created automatically
4. (Optional) Create `chats` table:
   ```sql
   CREATE TABLE chats (
     id UUID PRIMARY KEY,
     user_id UUID,
     character_id UUID,
     updated_at TIMESTAMP WITH TIME ZONE
   );
   ```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Service role key (not anon) |
| `SUPABASE_STORAGE_BUCKET` | No | Bucket name (default: `conversations`) |
| `SYSTEM_PROMPT` | No | Custom system prompt |
| `HTTP_REFERER` | No | OpenRouter referer header |
| `X_TITLE` | No | OpenRouter title header |

## Development

```bash
make run              # Start service
curl localhost:8000/health  # Test endpoint
docker-compose up -d  # Docker deployment
```

## License

MIT
