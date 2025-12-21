import httpx
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"

# Default Whisper model
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")

async def speech_to_text(
    audio_data: bytes,
    model: str | None = None,
    language: str | None = None,
    prompt: str | None = None,
    response_format: str = "json",
    temperature: float = 0.0
) -> str:
    """
    Convert speech to text using OpenAI Whisper API
    
    Args:
        audio_data: Audio file data as bytes
        model: Whisper model to use (defaults to "whisper-1")
        language: Language code (e.g., "en", "es", "fr") - optional, auto-detected if not provided
        prompt: Optional text prompt to guide the model's style or continue a previous audio segment
        response_format: Format of the response (default: "json")
        temperature: Sampling temperature (0.0-1.0), lower is more deterministic
    
    Returns:
        str: Transcribed text
    
    Raises:
        ValueError: If OPENAI_API_KEY is not set
        httpx.HTTPStatusError: If API request fails
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable must be set in .env file")
    
    model_to_use = model or DEFAULT_MODEL
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    
    files = {
        "file": ("audio.mp3", audio_data, "audio/mpeg")
    }
    
    data = {
        "model": model_to_use,
        "response_format": response_format,
        "temperature": str(temperature),
    }
    
    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt
    
    logger.info(f"Requesting transcription from OpenAI Whisper (model: {model_to_use})...")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            OPENAI_API_URL,
            headers=headers,
            files=files,
            data=data,
        )
        response.raise_for_status()
        
        result = response.json()
        transcribed_text = result.get("text", "")
        
        logger.info(f"Successfully transcribed audio: {len(transcribed_text)} characters")
        return transcribed_text

