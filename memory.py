import json
import os
import uuid
import logging
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "conversations")

# Initialize Supabase client
_supabase_client: Optional[Client] = None

# Cache for bucket existence check (to avoid repeated API calls)
_bucket_verified: bool = False

def get_supabase_client() -> Client:
    """Get or create Supabase client"""
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            missing = []
            if not SUPABASE_URL:
                missing.append("SUPABASE_URL")
            if not SUPABASE_SERVICE_ROLE_KEY:
                missing.append("SUPABASE_SERVICE_ROLE_KEY")
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Please set them in your .env file. "
                f"Get SUPABASE_SERVICE_ROLE_KEY from: Supabase Dashboard > Settings > API > service_role key (secret)"
            )
        
        # Verify the key looks like a service role key (JWT token)
        if not SUPABASE_SERVICE_ROLE_KEY.startswith('eyJ'):
            logger.warning("SUPABASE_SERVICE_ROLE_KEY doesn't look like a JWT token. "
                          "Make sure you're using the service_role key, not the anon key.")
        
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _supabase_client

def ensure_storage_bucket():
    """Ensure the storage bucket exists (creates if it doesn't) - only checks once per application lifetime"""
    global _bucket_verified
    
    # If we've already verified the bucket exists, skip the check
    if _bucket_verified:
        return
    
    try:
        supabase = get_supabase_client()
        
        # Try to get the bucket - if it exists, we're good
        try:
            bucket_response = supabase.storage.get_bucket(STORAGE_BUCKET)
            if bucket_response:
                logger.info(f"Bucket '{STORAGE_BUCKET}' verified and exists")
                _bucket_verified = True
                return
        except Exception as get_error:
            # Bucket doesn't exist, try to create it
            logger.info(f"Bucket '{STORAGE_BUCKET}' not found, attempting to create...")
            pass
        
        # Create bucket if it doesn't exist
        try:
            create_response = supabase.storage.create_bucket(
                STORAGE_BUCKET,
                options={
                    "public": False,  # Private bucket for conversations
                    "allowed_mime_types": ["application/json"],
                    "file_size_limit": 10485760  # 10MB limit
                }
            )
            
            # Handle response - could be a dict with 'error' key or an object with .error attribute
            if isinstance(create_response, dict):
                if create_response.get('error'):
                    error_msg = create_response.get('error')
                    # Check if error is because bucket already exists
                    error_str = str(error_msg).lower()
                    if "already exists" in error_str or "duplicate" in error_str or "bucket already exists" in error_str:
                        logger.info(f"Bucket '{STORAGE_BUCKET}' already exists")
                        _bucket_verified = True
                        return
                    else:
                        logger.warning(f"Error creating bucket: {error_msg}")
                        logger.warning(f"Bucket creation failed. Please create the bucket manually in Supabase Dashboard:")
                        logger.warning(f"  1. Go to: https://supabase.com/dashboard/project/{SUPABASE_URL.split('//')[1].split('.')[0]}/storage/buckets")
                        logger.warning(f"  2. Create a new bucket named: {STORAGE_BUCKET}")
                        logger.warning(f"  3. Set it to Private")
                        raise Exception(f"Failed to create bucket: {error_msg}")
                else:
                    logger.info(f"Bucket '{STORAGE_BUCKET}' created successfully")
                    _bucket_verified = True
            elif hasattr(create_response, 'error') and create_response.error:
                error_msg = create_response.error
                error_str = str(error_msg).lower()
                if "already exists" in error_str or "duplicate" in error_str:
                    logger.info(f"Bucket '{STORAGE_BUCKET}' already exists")
                    _bucket_verified = True
                    return
                else:
                    logger.warning(f"Error creating bucket: {error_msg}")
                    raise Exception(f"Failed to create bucket: {error_msg}")
            else:
                logger.info(f"Bucket '{STORAGE_BUCKET}' created successfully")
                _bucket_verified = True
        except Exception as create_error:
            # Check if error is because bucket already exists
            error_str = str(create_error).lower()
            if "already exists" in error_str or "duplicate" in error_str or "bucket already exists" in error_str:
                logger.info(f"Bucket '{STORAGE_BUCKET}' already exists (created by another process)")
                _bucket_verified = True
                return
            
            # If it's an authorization error, provide helpful instructions
            if "unauthorized" in error_str or "signature" in error_str:
                logger.warning(f"⚠️  Authorization failed when creating bucket. This usually means:")
                logger.warning(f"  1. SUPABASE_SERVICE_ROLE_KEY is incorrect or missing")
                logger.warning(f"  2. You're using the anon key instead of the service_role key")
                logger.warning(f"  3. The bucket needs to be created manually in Supabase Dashboard")
                logger.warning(f"")
                logger.warning(f"To fix:")
                logger.warning(f"  - Verify SUPABASE_SERVICE_ROLE_KEY in your .env file")
                logger.warning(f"  - Get it from: Supabase Dashboard > Settings > API > service_role key (secret)")
                logger.warning(f"  - Or create the bucket manually: Storage > New Bucket > Name: {STORAGE_BUCKET}")
                logger.warning(f"")
                logger.warning(f"Continuing anyway - will try to use existing bucket or operations will fail with clear errors.")
                # Don't raise - allow the app to continue, file operations will fail with clear errors if bucket doesn't exist
                return
            
            logger.error(f"Failed to create bucket '{STORAGE_BUCKET}': {create_error}")
            raise Exception(f"Failed to create bucket: {create_error}")
                
    except Exception as e:
        logger.error(f"Error in ensure_storage_bucket: {e}")
        raise Exception(f"Failed to ensure storage bucket exists: {e}")

def get_conversation_path(conversation_id: str) -> str:
    """Get the storage path for a conversation"""
    return f"{conversation_id}.json"

def create_chat_record(conversation_id: str, user_id: str, character_id: str):
    """
    Create or update a record in the chats table.
    The chats table only stores metadata - the JSON file in Supabase Storage contains all messages.
    
    Args:
        conversation_id: UUID of the conversation (used as chat.id, also the JSON filename)
        user_id: UUID of the user
        character_id: UUID of the character
    """
    try:
        supabase = get_supabase_client()
        
        # Insert or update chat record
        # Use conversation_id as the chat.id to link to JSON file: {conversation_id}.json
        result = supabase.table("chats").upsert(
            {
                "id": conversation_id,
                "user_id": user_id,
                "character_id": character_id,
                "updated_at": datetime.utcnow().isoformat()
            },
            on_conflict="id"
        ).execute()
        
        if hasattr(result, 'error') and result.error:
            logger.warning(f"Failed to create chat record: {result.error}")
        else:
            logger.debug(f"Created/updated chat record: conversation_id={conversation_id}, user_id={user_id}, character_id={character_id}")
    except Exception as e:
        # Don't fail the conversation creation if database record fails
        logger.warning(f"Failed to create chat record in database: {e}")

def update_chat_timestamp(conversation_id: str):
    """
    Update the updated_at timestamp in the chats table when the JSON file is modified.
    This provides a reference point for when the conversation was last updated.
    
    Args:
        conversation_id: UUID of the conversation (chat.id)
    """
    try:
        supabase = get_supabase_client()
        
        # Update the updated_at timestamp to reflect JSON file modification
        result = supabase.table("chats").update({
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", conversation_id).execute()
        
        if hasattr(result, 'error') and result.error:
            logger.debug(f"Failed to update chat timestamp: {result.error}")
        else:
            logger.debug(f"Updated chat timestamp for conversation {conversation_id}")
    except Exception as e:
        # Don't fail if timestamp update fails - JSON is source of truth
        logger.debug(f"Failed to update chat timestamp: {e}")

def create_conversation(user_id: Optional[str] = None, character_id: Optional[str] = None) -> str:
    """
    Create a new conversation and return its UUID
    
    Args:
        user_id: Optional user ID to link conversation in database
        character_id: Optional character ID to link conversation in database
    
    Returns:
        conversation_id: UUID of the created conversation
    """
    conversation_id = str(uuid.uuid4())
    conversation_data = {
        "id": conversation_id,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "messages": []
    }
    save_conversation(conversation_id, conversation_data)
    
    # If user_id and character_id provided, create database record
    if user_id and character_id:
        create_chat_record(conversation_id, user_id, character_id)
    
    return conversation_id

def load_conversation(conversation_id: str) -> Optional[dict]:
    """Load a conversation by UUID from Supabase Storage"""
    ensure_storage_bucket()
    supabase = get_supabase_client()
    file_path = get_conversation_path(conversation_id)
    
    try:
        response = supabase.storage.from_(STORAGE_BUCKET).download(file_path)
        if response:
            # Response is bytes, decode to string then parse JSON
            content = response.decode('utf-8')
            return json.loads(content)
    except Exception as e:
        # File doesn't exist or other error
        error_str = str(e).lower()
        if "not found" in error_str or "does not exist" in error_str:
            # File doesn't exist - this is expected for new conversations
            return None
        else:
            # Other error - log it
            logger.warning(f"Error loading conversation {conversation_id}: {e}")
            return None
    
    return None

def save_conversation(conversation_id: str, conversation_data: dict):
    """Save conversation data to Supabase Storage"""
    ensure_storage_bucket()
    supabase = get_supabase_client()
    file_path = get_conversation_path(conversation_id)
    
    conversation_data["updated_at"] = datetime.utcnow().isoformat()
    
    try:
        # Convert to JSON string then to bytes
        json_content = json.dumps(conversation_data, indent=2, ensure_ascii=False)
        json_bytes = json_content.encode('utf-8')
        
        file_options = {"content-type": "application/json"}
        
        # Try to upload first (for new files)
        # If file exists, use update instead
        try:
            upload_response = supabase.storage.from_(STORAGE_BUCKET).upload(
                file_path,
                json_bytes,
                file_options=file_options
            )
            
            # Check for errors in response
            if hasattr(upload_response, 'error') and upload_response.error:
                error_msg = upload_response.error
                # If it's a duplicate error, try update instead
                if "duplicate" in str(error_msg).lower() or "already exists" in str(error_msg).lower():
                    logger.debug(f"File exists, using update instead of upload for {conversation_id}")
                    update_response = supabase.storage.from_(STORAGE_BUCKET).update(
                        file_path,
                        json_bytes,
                        file_options=file_options
                    )
                    if hasattr(update_response, 'error') and update_response.error:
                        raise Exception(f"Failed to update conversation: {update_response.error}")
                else:
                    raise Exception(f"Failed to save conversation: {error_msg}")
            
            logger.debug(f"Successfully saved conversation {conversation_id} to storage")
        except Exception as upload_error:
            # Check if it's a duplicate error
            error_str = str(upload_error).lower()
            if "duplicate" in error_str or "already exists" in error_str:
                # File exists, use update instead
                logger.debug(f"File exists, using update instead of upload for {conversation_id}")
                try:
                    update_response = supabase.storage.from_(STORAGE_BUCKET).update(
                        file_path,
                        json_bytes,
                        file_options=file_options
                    )
                    if hasattr(update_response, 'error') and update_response.error:
                        raise Exception(f"Failed to update conversation: {update_response.error}")
                    logger.debug(f"Successfully updated conversation {conversation_id} in storage")
                except Exception as update_error:
                    logger.error(f"Exception updating conversation {conversation_id}: {update_error}")
                    raise Exception(f"Failed to update conversation to Supabase Storage: {update_error}")
            else:
                # Some other error, re-raise it
                raise
        
    except Exception as e:
        logger.error(f"Exception saving conversation {conversation_id}: {e}")
        raise Exception(f"Failed to save conversation to Supabase Storage: {e}")

def ensure_chat_record_exists(conversation_id: str, user_id: Optional[str] = None, character_id: Optional[str] = None) -> bool:
    """
    Ensure the chat record exists in the database. If not, try to create it.
    The chat record is just metadata - the JSON file contains all messages.
    
    Args:
        conversation_id: UUID of the conversation (used as chat.id, also JSON filename)
        user_id: Optional user ID (will try to get from existing chat if not provided)
        character_id: Optional character ID (will try to get from existing chat if not provided)
    
    Returns:
        True if chat record exists or was created, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        # Check if chat exists
        chat_check = supabase.table("chats").select("id, user_id, character_id").eq("id", conversation_id).execute()
        
        if chat_check.data and len(chat_check.data) > 0:
            # Chat exists
            return True
        
        # Chat doesn't exist - try to create it if we have user_id and character_id
        if user_id and character_id:
            create_chat_record(conversation_id, user_id, character_id)
            return True
        else:
            logger.debug(f"Chat {conversation_id} not found in database and cannot create without user_id and character_id")
            return False
    except Exception as e:
        logger.warning(f"Error checking/creating chat record: {e}")
        return False

def add_message_to_conversation(conversation_id: str, role: str, content: str, user_id: Optional[str] = None, character_id: Optional[str] = None):
    """
    Add a message to a conversation.
    JSON file in Supabase Storage is the ONLY source of truth for messages.
    The database chats table only stores metadata (user_id, character_id, timestamps).
    
    Args:
        conversation_id: UUID of the conversation (also the JSON filename: {conversation_id}.json)
        role: Message role ('user' or 'assistant')
        content: Message content
        user_id: Optional user ID (used to create/update chat record metadata)
        character_id: Optional character ID (used to create/update chat record metadata)
    """
    try:
        # Load conversation from JSON (source of truth)
        conversation = load_conversation(conversation_id)
        
        if conversation is None:
            # Create new conversation if it doesn't exist
            logger.info(f"Creating new conversation {conversation_id} for message")
            conversation = {
                "id": conversation_id,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "messages": []
            }
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        conversation["messages"].append(message)
        logger.debug(f"Added {role} message to conversation {conversation_id}: {content[:50]}...")
        
        # Save to JSON file (source of truth)
        save_conversation(conversation_id, conversation)
        logger.debug(f"Successfully saved conversation {conversation_id} with {len(conversation['messages'])} messages to JSON")
        
        # Update chat record metadata (ensure it exists and update timestamp)
        if user_id and character_id:
            ensure_chat_record_exists(conversation_id, user_id, character_id)
        update_chat_timestamp(conversation_id)
    except Exception as e:
        logger.error(f"Error adding message to conversation {conversation_id}: {e}")
        raise

def get_conversation_messages(conversation_id: str) -> list[dict[str, str]]:
    """Get all messages from a conversation in the format expected by the API"""
    conversation = load_conversation(conversation_id)
    
    if conversation is None:
        return []
    
    # Return messages in the format expected by OpenRouter API
    # (role and content only, no timestamp)
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in conversation.get("messages", [])
    ]

def get_or_create_conversation(conversation_id: Optional[str] = None, user_id: Optional[str] = None, character_id: Optional[str] = None) -> str:
    """
    Get existing conversation ID or create a new one
    
    Args:
        conversation_id: Optional existing conversation ID
        user_id: Optional user ID for database tracking
        character_id: Optional character ID for database tracking
    
    Returns:
        conversation_id: UUID of the conversation
    """
    if conversation_id:
        # Verify conversation exists, create if it doesn't
        if load_conversation(conversation_id) is None:
            # Create new conversation with the provided ID
            conversation_data = {
                "id": conversation_id,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "messages": []
            }
            save_conversation(conversation_id, conversation_data)
            
            # If user_id and character_id provided, create database record
            if user_id and character_id:
                create_chat_record(conversation_id, user_id, character_id)
        return conversation_id
    else:
        return create_conversation(user_id, character_id)
