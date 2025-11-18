"""
Utility functions for the email agent MVP.
Provides helpers for ID generation, timestamps, trace logging, config loading, and LLM initialization.
"""

import json
import uuid
import yaml
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global constants
USER_EMAIL = "vince.j.kaminski@enron.com"  # User's email address - change this to update user email globally


def set_global_seeds(seed: int = 42) -> None:
    """
    Set global random seeds for reproducibility.
    
    This sets seeds for:
    - Python's random module
    - NumPy's random module (if available)
    - PYTHONHASHSEED environment variable
    
    Args:
        seed: Integer seed value (default: 42)
    """
    import os
    import random
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed if available
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    # Set hash seed for dictionary ordering (Python 3.3+)
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    unique_id = str(uuid.uuid4())
    return f"{prefix}_{unique_id}" if prefix else unique_id


def get_timestamp() -> str:
    """Get fixed timestamp in ISO 8601 format with timezone (for reproducibility: November 3, 2025)."""
    # Fixed date: November 3, 2025, 12:00:00 UTC for reproducibility
    fixed_date = datetime(2025, 11, 3, 12, 0, 0, tzinfo=timezone.utc)
    return fixed_date.isoformat()


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def append_trace_event(
    trace_file: str,
    event_type: str,
    session_id: str,
    payload: dict,
    event_id: Optional[str] = None
) -> str:
    """
    Append a trace event to the session-specific trace file.
    
    Args:
        trace_file: Base path to the trace JSONL file (will be modified for session-specific storage)
        event_type: Type of event (user_input, tool_call, tool_result, agent_response)
        session_id: Session identifier
        payload: Event-specific data
        event_id: Optional event ID (generated if not provided)
    
    Returns:
        The event_id used for this event
    """
    if event_id is None:
        event_id = generate_id()
    
    event = {
        "ts": get_timestamp(),
        "event_id": event_id,
        "session_id": session_id,
        "event_type": event_type,
        "payload": payload
    }
    
    # Create session-specific trace file path
    trace_path = Path(trace_file)
    session_trace_dir = trace_path.parent / "traces"
    session_trace_file = session_trace_dir / f"{session_id}.jsonl"
    
    # Ensure trace file directory exists
    session_trace_dir.mkdir(parents=True, exist_ok=True)
    
    # Append event to session-specific file
    with open(session_trace_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    
    return event_id


def read_trace_events(trace_file: str, session_id: Optional[str] = None) -> list:
    """
    Read trace events from the session-specific trace file.
    
    Args:
        trace_file: Base path to the trace JSONL file (will be modified for session-specific storage)
        session_id: Session ID to read events for (required for session-specific storage)
    
    Returns:
        List of trace events for the specified session
    """
    if session_id is None:
        return []
    
    # Create session-specific trace file path
    trace_path = Path(trace_file)
    session_trace_dir = trace_path.parent / "traces"
    session_trace_file = session_trace_dir / f"{session_id}.jsonl"
    
    if not session_trace_file.exists():
        return []
    
    events = []
    with open(session_trace_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                event = json.loads(line)
                events.append(event)
    
    return events


def ensure_data_directories(config: dict) -> None:
    """Ensure all required data directories exist."""
    data_config = config.get("data", {})
    
    # Create mailbox directory
    mailbox_dir = Path(data_config.get("mailbox_dir", "data/interactive_agent/mailbox"))
    mailbox_dir.mkdir(parents=True, exist_ok=True)
    
    # Create drafts directory
    drafts_dir = Path(data_config.get("drafts_dir", "data/interactive_agent/drafts"))
    drafts_dir.mkdir(parents=True, exist_ok=True)
    
    # Create outbox directory
    outbox_dir = Path(data_config.get("outbox_dir", "data/interactive_agent/outbox"))
    outbox_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trace file parent directory
    trace_file = Path(data_config.get("trace_file", "data/interactive_agent/trace.jsonl"))
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize trace file if it doesn't exist
    if not trace_file.exists():
        trace_file.touch()


def compare_attack_bench_files(original_file: Path, cached_file: Path) -> Dict[str, Any]:
    """
    Compare attack benchmark files to ensure only attack_emails differ.
    
    This function verifies that cached files only differ from original files
    in the attack_emails attribute, ensuring the caching system works correctly.
    
    Args:
        original_file: Path to the original attack benchmark file
        cached_file: Path to the cached attack benchmark file
    
    Returns:
        Dictionary with comparison results:
        - valid: bool - True if files are valid (only attack_emails differ)
        - attack_emails_differ: bool - True if attack_emails are different
        - differences: List[str] - List of field names that differ
        - summary: str - Human-readable summary
    """
    result = {
        "valid": True,
        "attack_emails_differ": False,
        "differences": [],
        "summary": ""
    }
    
    try:
        # Load both files
        with open(original_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        with open(cached_file, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
        
        # Compare all keys and values except attack_emails
        for key in original_data:
            if key == "attack_emails":
                # Check if attack_emails differ
                if original_data[key] != cached_data.get(key):
                    result["attack_emails_differ"] = True
                continue
            
            # For all other keys, they must be identical
            if key not in cached_data:
                result["valid"] = False
                result["differences"].append(f"{key} (missing in cached)")
            elif original_data[key] != cached_data[key]:
                result["valid"] = False
                result["differences"].append(key)
        
        # Check for extra keys in cached file (excluding optimization_metadata)
        for key in cached_data:
            if key not in original_data and key != "optimization_metadata":
                result["valid"] = False
                result["differences"].append(f"{key} (extra in cached)")
        
        # Generate summary
        if result["valid"] and result["attack_emails_differ"]:
            result["summary"] = "✅ Files are identical except for attack_emails (as expected)"
        elif result["valid"] and not result["attack_emails_differ"]:
            result["summary"] = "✅ Files are completely identical"
        else:
            result["summary"] = f"❌ Files differ in fields: {', '.join(result['differences'])}"
        
        return result
        
    except Exception as e:
        return {
            "valid": False,
            "attack_emails_differ": False,
            "differences": [f"Error: {e}"],
            "summary": f"❌ Error comparing files: {e}"
        }


def validate_cache_integrity(cache_dir: str = "data/benchmark/attack_bench_cache", 
                           original_dir: str = "data/benchmark/attack_bench") -> Dict[str, Any]:
    """
    Validate the integrity of all cached attack benchmark files.
    
    This function compares all cached files with their original counterparts
    to ensure the caching system is working correctly.
    
    Args:
        cache_dir: Directory containing cached files
        original_dir: Directory containing original files
    
    Returns:
        Dictionary with validation results:
        - total_files: int - Total number of files compared
        - valid_files: int - Number of files that are correctly cached
        - invalid_files: int - Number of files with issues
        - results: List[Dict] - Detailed results for each file
        - summary: str - Overall validation summary
    """
    cache_path = Path(cache_dir)
    original_path = Path(original_dir)
    
    if not cache_path.exists():
        return {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "results": [],
            "summary": "❌ Cache directory does not exist"
        }
    
    # Find all cached files
    cached_files = list(cache_path.rglob("*.json"))
    results = []
    valid_files = 0
    invalid_files = 0
    
    for cached_file in cached_files:
        # Find corresponding original file
        relative_path = cached_file.relative_to(cache_path)
        original_file = original_path / relative_path
        
        if not original_file.exists():
            results.append({
                "cached_file": str(cached_file),
                "original_file": str(original_file),
                "status": "missing_original",
                "result": {
                    "identical": False,
                    "summary": "❌ Original file not found"
                }
            })
            invalid_files += 1
            continue
        
        # Compare files
        comparison_result = compare_attack_bench_files(original_file, cached_file)
        
        # Use the simplified valid flag
        is_valid = comparison_result["valid"]
        
        if is_valid:
            valid_files += 1
        else:
            invalid_files += 1
        
        results.append({
            "cached_file": str(cached_file),
            "original_file": str(original_file),
            "status": "valid" if is_valid else "invalid",
            "result": comparison_result
        })
    
    total_files = len(cached_files)
    
    # Generate summary
    if invalid_files == 0:
        summary = f"✅ All {total_files} cached files are valid"
    else:
        summary = f"⚠️ {valid_files}/{total_files} cached files are valid, {invalid_files} have issues"
    
    return {
        "total_files": total_files,
        "valid_files": valid_files,
        "invalid_files": invalid_files,
        "results": results,
        "summary": summary
    }


def print_cache_validation_report(validation_result: Dict[str, Any]) -> None:
    """
    Print a formatted validation report for cache integrity.
    
    Args:
        validation_result: Result from validate_cache_integrity()
    """
    print("=" * 80)
    print("CACHE INTEGRITY VALIDATION REPORT")
    print("=" * 80)
    print(f"Total files: {validation_result['total_files']}")
    print(f"Valid files: {validation_result['valid_files']}")
    print(f"Invalid files: {validation_result['invalid_files']}")
    print(f"Summary: {validation_result['summary']}")
    print()
    
    if validation_result['invalid_files'] > 0:
        print("INVALID FILES:")
        print("-" * 40)
        for result in validation_result['results']:
            if result['status'] == 'invalid':
                print(f"❌ {result['cached_file']}")
                print(f"   Original: {result['original_file']}")
                print(f"   Issue: {result['result']['summary']}")
                if result['result']['differences']:
                    print("   Differences:")
                    for diff in result['result']['differences']:
                        print(f"     - {diff}")
                print()
    
    print("VALID FILES:")
    print("-" * 40)
    for result in validation_result['results']:
        if result['status'] == 'valid':
            print(f"✅ {result['cached_file']}")
            if result['result']['attack_emails_differ']:
                print("   (attack_emails differ as expected)")
            else:
                print("   (completely identical)")
    print("=" * 80)


# ============================================================================
# LLM Utility Functions
# ============================================================================
# Functions for LLM initialization supporting multiple providers (OpenAI, Gemini).


def detect_provider(model_name: str) -> Literal["openai", "gemini"]:
    """
    Detect the provider based on model name.
    
    Args:
        model_name: Name of the model (e.g., "gpt-4o", "gemini-2.5-pro")
    
    Returns:
        Provider name: "openai" or "gemini"
    """
    model_lower = model_name.lower()
    
    # Gemini models start with "gemini"
    if model_lower.startswith("gemini"):
        return "gemini"
    
    # Default to OpenAI for all other models
    return "openai"


def get_openai_client(api_key: Optional[str] = None):
    """
    Get an OpenAI client instance.
    
    Args:
        api_key: Optional API key (defaults to OPENAI_API_KEY env var)
    
    Returns:
        OpenAI client instance
    """
    from openai import OpenAI
    
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    
    return OpenAI(api_key=api_key)


def get_gemini_client(api_key: Optional[str] = None):
    """
    Get a Google Generative AI client instance.
    
    Args:
        api_key: Optional API key (defaults to GEMINI_API_KEY env var)
    
    Returns:
        Google Generative AI client instance
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai package is required for Gemini models. "
            "Install it with: pip install google-generativeai"
        )
    
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    
    genai.configure(api_key=api_key)
    return genai


def call_openai_chat_completion(
    client,
    model: str,
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Call OpenAI chat completion API.
    
    Args:
        client: OpenAI client instance
        model: Model name
        messages: List of message dicts with "role" and "content"
        temperature: Temperature parameter
        max_tokens: Maximum tokens for response
        max_completion_tokens: Maximum completion tokens (for reasoning models)
        top_p: Top-p parameter
        presence_penalty: Presence penalty
        frequency_penalty: Frequency penalty
        response_format: Response format (e.g., {"type": "json_object"})
    
    Returns:
        Response object from OpenAI API
    """
    params = {
        "model": model,
        "messages": messages,
    }
    
    # Add optional parameters
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        params["max_completion_tokens"] = max_completion_tokens
    if top_p is not None:
        params["top_p"] = top_p
    if presence_penalty is not None:
        params["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        params["frequency_penalty"] = frequency_penalty
    if response_format is not None:
        params["response_format"] = response_format
    
    return client.chat.completions.create(**params)


def call_gemini_chat_completion(
    _genai_module,
    model: str,
    messages: list,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
) -> Any:
    """
    Call Gemini chat completion API.
    
    Args:
        _genai_module: Google Generative AI module (from get_gemini_client) - unused, genai imported directly
        model: Model name (e.g., "gemini-2.5-pro")
        messages: List of message dicts with "role" and "content"
        temperature: Temperature parameter
        max_output_tokens: Maximum output tokens
        top_p: Top-p parameter
    
    Returns:
        Response object from Gemini API (wrapped to be OpenAI-compatible)
    """
    import google.generativeai as genai
    
    # Extract system instruction from messages
    system_instruction = None
    chat_messages = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            # Collect system instructions
            if system_instruction is None:
                system_instruction = content
            else:
                system_instruction += "\n\n" + content
        else:
            # Add to chat history
            chat_messages.append({
                "role": "user" if role == "user" else "model",
                "parts": [content]
            })
    
    # Build generation config
    generation_config = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if max_output_tokens is not None:
        generation_config["max_output_tokens"] = max_output_tokens
    if top_p is not None:
        generation_config["top_p"] = top_p
    
    gen_config = genai.types.GenerationConfig(**generation_config) if generation_config else None
    
    # Get the model instance - system_instruction is passed during model creation
    if system_instruction:
        model_instance = genai.GenerativeModel(
            model,
            system_instruction=system_instruction
        )
    else:
        model_instance = genai.GenerativeModel(model)
    
    # Build the prompt from messages
    # For chat, we need to handle the conversation history
    if len(chat_messages) == 0:
        raise ValueError("No user messages found in messages list")
    
    # If we have multiple messages, use chat history
    if len(chat_messages) > 1:
        # Separate history from the last message
        history = chat_messages[:-1]
        last_message = chat_messages[-1]["parts"][0]
        
        # Start chat with history
        chat = model_instance.start_chat(history=history)
        
        # Send the last message
        response = chat.send_message(last_message, generation_config=gen_config if gen_config else None)
    else:
        # Single message - use generate_content
        prompt = chat_messages[0]["parts"][0]
        if gen_config:
            response = model_instance.generate_content(
                prompt,
                generation_config=gen_config
            )
        else:
            response = model_instance.generate_content(prompt)
    
    # Convert Gemini response to OpenAI-like format for compatibility
    class GeminiResponse:
        """Wrapper to make Gemini response compatible with OpenAI response format."""
        def __init__(self, gemini_response):
            self.choices = [GeminiChoice(gemini_response)]
            self.usage = GeminiUsage(gemini_response)
    
    class GeminiChoice:
        """Wrapper for Gemini choice."""
        def __init__(self, gemini_response):
            self.message = GeminiMessage(gemini_response)
            # Check finish reason
            if hasattr(gemini_response, "candidates") and gemini_response.candidates:
                finish_reason = gemini_response.candidates[0].finish_reason if gemini_response.candidates else None
                if finish_reason is not None:
                    # Handle finish_reason - it might be an enum or string
                    if hasattr(finish_reason, "name"):
                        self.finish_reason = finish_reason.name.lower()
                    elif isinstance(finish_reason, str):
                        self.finish_reason = finish_reason.lower()
                    else:
                        self.finish_reason = str(finish_reason).lower()
                else:
                    self.finish_reason = "stop"
            else:
                self.finish_reason = "stop"
    
    class GeminiMessage:
        """Wrapper for Gemini message."""
        def __init__(self, gemini_response):
            # Check for refusal/safety blocks FIRST before trying to get content
            self.refusal = None
            finish_reason_code = None
            finish_reason_name = None
            
            if hasattr(gemini_response, "candidates") and gemini_response.candidates:
                candidate = gemini_response.candidates[0]
                if hasattr(candidate, "finish_reason"):
                    finish_reason = candidate.finish_reason
                    # finish_reason can be an enum or integer
                    if hasattr(finish_reason, "name"):
                        finish_reason_name = finish_reason.name
                        finish_reason_code = finish_reason.value if hasattr(finish_reason, "value") else None
                    elif isinstance(finish_reason, int):
                        finish_reason_code = finish_reason
                        # Map common codes: 1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION
                        finish_reason_map = {1: "STOP", 2: "MAX_TOKENS", 3: "SAFETY", 4: "RECITATION"}
                        finish_reason_name = finish_reason_map.get(finish_reason, f"UNKNOWN_{finish_reason}")
                    else:
                        finish_reason_name = str(finish_reason)
                    
                    # Check if blocked by safety (code 3 or name contains SAFETY)
                    if finish_reason_code == 3 or (finish_reason_name and "SAFETY" in finish_reason_name.upper()):
                        self.refusal = f"Blocked by safety filter: {finish_reason_name} (code: {finish_reason_code})"
                    elif finish_reason_code == 4 or (finish_reason_name and "RECITATION" in finish_reason_name.upper()):
                        self.refusal = f"Blocked by recitation filter: {finish_reason_name} (code: {finish_reason_code})"
                    elif finish_reason_code == 2:
                        # MAX_TOKENS - not a block, but content might be truncated
                        pass
            
            # Get text from response - handle cases where content might be blocked
            # IMPORTANT: Check candidates/parts FIRST before trying response.text to avoid errors
            try:
                # First check candidates (safer - doesn't trigger the quick accessor error)
                if hasattr(gemini_response, "candidates") and gemini_response.candidates:
                    candidate = gemini_response.candidates[0]
                    # Check if candidate has content with parts
                    if hasattr(candidate, "content"):
                        if hasattr(candidate.content, "parts") and candidate.content.parts:
                            # Extract text from parts safely
                            parts_text = []
                            for part in candidate.content.parts:
                                if hasattr(part, "text"):
                                    parts_text.append(part.text)
                            self.content = "".join(parts_text)
                        else:
                            # No parts in content - this usually means blocked/empty response
                            self.content = ""
                    else:
                        # No content attribute - response was likely blocked
                        self.content = ""
                # Only try response.text if we didn't get content from candidates
                elif hasattr(gemini_response, "text"):
                    # This might throw an error if there are no parts, so wrap in try-except
                    try:
                        self.content = gemini_response.text
                    except Exception:
                        # If response.text fails, it means no valid parts
                        self.content = ""
                else:
                    self.content = ""
            except Exception as e:
                # If we can't get text, this is an error
                error_details = f"Finish reason: {finish_reason_name} (code: {finish_reason_code})" if finish_reason_name else "Unknown finish reason"
                if self.refusal:
                    raise RuntimeError(
                        f"Gemini model response was blocked: {self.refusal}. "
                        f"{error_details}. "
                        f"Original error: {e}"
                    ) from e
                else:
                    raise RuntimeError(
                        f"Gemini model response has no valid content. "
                        f"{error_details}. "
                        f"Error accessing response: {e}"
                    ) from e
            
            # If content is empty and we have a refusal or finish_reason indicates a block, that's an error
            if not self.content:
                if self.refusal:
                    raise RuntimeError(
                        f"Gemini model response was blocked and returned no content: {self.refusal}. "
                        f"Finish reason: {finish_reason_name} (code: {finish_reason_code})"
                    )
                elif finish_reason_code == 3:  # SAFETY
                    raise RuntimeError(
                        f"Gemini model response was blocked by safety filter (finish_reason code: 3). "
                        f"No content was returned."
                    )
                elif finish_reason_code == 4:  # RECITATION
                    raise RuntimeError(
                        f"Gemini model response was blocked by recitation filter (finish_reason code: 4). "
                        f"No content was returned."
                    )
                elif finish_reason_code == 2 and not self.content:
                    # MAX_TOKENS with no content - this is unusual
                    # It could mean the response was so truncated that no parts were returned
                    # Or it could indicate a block. Either way, we can't proceed without content.
                    # Log additional debug info before raising
                    debug_info = []
                    if hasattr(gemini_response, "candidates") and gemini_response.candidates:
                        candidate = gemini_response.candidates[0]
                        debug_info.append(f"candidate has content: {hasattr(candidate, 'content')}")
                        if hasattr(candidate, "content"):
                            debug_info.append(f"content has parts: {hasattr(candidate.content, 'parts')}")
                            if hasattr(candidate.content, "parts"):
                                debug_info.append(f"parts count: {len(candidate.content.parts) if candidate.content.parts else 0}")
                    raise RuntimeError(
                        f"Gemini model response was truncated (finish_reason=MAX_TOKENS, code: 2) "
                        f"but no content was extracted. This may indicate: "
                        f"(1) The response was severely truncated, (2) The token limit is too low, "
                        f"or (3) The response was blocked. Consider increasing max_output_tokens. "
                        f"Debug info: {', '.join(debug_info) if debug_info else 'No debug info available'}"
                    )
    
    class GeminiUsage:
        """Wrapper for Gemini usage stats."""
        def __init__(self, gemini_response):
            # Gemini provides usage info in usage_metadata
            if hasattr(gemini_response, "usage_metadata"):
                self.prompt_tokens = gemini_response.usage_metadata.prompt_token_count
                self.completion_tokens = gemini_response.usage_metadata.candidates_token_count
                self.total_tokens = gemini_response.usage_metadata.total_token_count
            else:
                self.prompt_tokens = 0
                self.completion_tokens = 0
                self.total_tokens = 0
    
    return GeminiResponse(response)


def create_llm_client(provider: Optional[str] = None, model_name: Optional[str] = None, api_key: Optional[str] = None):
    """
    Create an LLM client for the specified provider.
    
    Args:
        provider: Provider name ("openai" or "gemini"). If None, auto-detects from model_name.
        model_name: Model name (used for auto-detection if provider is None)
        api_key: Optional API key (defaults to env vars)
    
    Returns:
        Client instance (OpenAI client or genai module)
    """
    if provider is None:
        if model_name is None:
            raise ValueError("Either provider or model_name must be provided")
        provider = detect_provider(model_name)
    
    if provider == "openai":
        return get_openai_client(api_key)
    elif provider == "gemini":
        return get_gemini_client(api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def call_llm_chat_completion(
    model: str,
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    response_format: Optional[Dict[str, Any]] = None,
    client: Optional[Any] = None,
) -> Any:
    """
    Unified function to call LLM chat completion for any provider.
    
    Args:
        model: Model name (e.g., "gpt-4o", "gemini-2.5-pro")
        messages: List of message dicts with "role" and "content"
        temperature: Temperature parameter
        max_tokens: Maximum tokens (OpenAI)
        max_completion_tokens: Maximum completion tokens (OpenAI reasoning models)
        max_output_tokens: Maximum output tokens (Gemini)
        top_p: Top-p parameter
        presence_penalty: Presence penalty (OpenAI only)
        frequency_penalty: Frequency penalty (OpenAI only)
        response_format: Response format (OpenAI only)
        client: Optional pre-created client (auto-created if not provided)
    
    Returns:
        Response object (compatible format across providers)
    """
    provider = detect_provider(model)
    
    if client is None:
        client = create_llm_client(provider=provider, model_name=model)
    
    if provider == "openai":
        return call_openai_chat_completion(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            response_format=response_format,
        )
    elif provider == "gemini":
        # Map max_tokens or max_completion_tokens to max_output_tokens for Gemini
        if max_output_tokens is None:
            max_output_tokens = max_completion_tokens if max_completion_tokens is not None else max_tokens
        
        return call_gemini_chat_completion(
            _genai_module=client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

