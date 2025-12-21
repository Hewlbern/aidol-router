import httpx
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
# Look for .env file in the same directory as this file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
# Also try loading from current directory (for compatibility)
load_dotenv()

# ElevenLabs API configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

# Default voice ID (can be overridden)
DEFAULT_VOICE_ID = os.getenv("ELEVENLABS_DEFAULT_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

# Default model ID (can be overridden via environment variable)
# Common models: "eleven_turbo_v2_5", "eleven_turbo_v2", "eleven_multilingual_v2", "eleven_monolingual_v1"
# Check https://elevenlabs.io/docs/api-reference/text-to-speech for latest models
DEFAULT_MODEL_ID = os.getenv("ELEVENLABS_DEFAULT_MODEL_ID", "eleven_turbo_v2_5")

async def text_to_speech(
    text: str,
    voice_id: str | None = None,
    model_id: str | None = None,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    style: float = 0.0,
    use_speaker_boost: bool = True
) -> bytes:
    """
    Convert text to speech using ElevenLabs API
    
    Args:
        text: Text to convert to speech
        voice_id: ElevenLabs voice ID (uses DEFAULT_VOICE_ID if not provided)
        model_id: Model ID (uses DEFAULT_MODEL_ID from env or "eleven_turbo_v2_5" if not provided)
        stability: Stability setting (0.0-1.0)
        similarity_boost: Similarity boost setting (0.0-1.0)
        style: Style setting (0.0-1.0)
        use_speaker_boost: Whether to use speaker boost
    
    Returns:
        bytes: Audio data in MP3 format
    
    Raises:
        ValueError: If ELEVENLABS_API_KEY is not set
        httpx.HTTPStatusError: If API request fails
    """
    if not ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY environment variable must be set in .env file")
    
    voice_to_use = voice_id or DEFAULT_VOICE_ID
    model_to_use = model_id or DEFAULT_MODEL_ID
    
    url = f"{ELEVENLABS_API_URL}/{voice_to_use}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY,
    }
    
    payload = {
        "text": text,
        "model_id": model_to_use,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": use_speaker_boost,
        }
    }
    
    logger.info(f"Requesting TTS from ElevenLabs for text: {text[:50]}...")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            headers=headers,
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
        
        logger.info(f"Successfully received audio from ElevenLabs ({len(response.content)} bytes)")
        return response.content

async def text_to_speech_stream(
    text: str,
    voice_id: str | None = None,
    model_id: str | None = None,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    style: float = 0.0,
    use_speaker_boost: bool = True
):
    """
    Stream text to speech audio from ElevenLabs API
    
    Args:
        text: Text to convert to speech
        voice_id: ElevenLabs voice ID (uses DEFAULT_VOICE_ID if not provided)
        model_id: Model ID (uses DEFAULT_MODEL_ID from env or "eleven_turbo_v2_5" if not provided)
        stability: Stability setting (0.0-1.0)
        similarity_boost: Similarity boost setting (0.0-1.0)
        style: Style setting (0.0-1.0)
        use_speaker_boost: Whether to use speaker boost
    
    Yields:
        bytes: Audio chunks in MP3 format
    
    Raises:
        ValueError: If ELEVENLABS_API_KEY is not set
        httpx.HTTPStatusError: If API request fails
    """
    if not ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY environment variable must be set in .env file")
    
    voice_to_use = voice_id or DEFAULT_VOICE_ID
    model_to_use = model_id or DEFAULT_MODEL_ID
    
    url = f"{ELEVENLABS_API_URL}/{voice_to_use}/stream"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY,
    }
    
    payload = {
        "text": text,
        "model_id": model_to_use,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": use_speaker_boost,
        }
    }
    
    logger.info(f"Streaming TTS from ElevenLabs for text: {text[:50]}...")
    
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=payload,
            timeout=30.0,
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                yield chunk

