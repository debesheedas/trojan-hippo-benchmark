"""
Utility functions for the email agent MVP.
Provides helpers for ID generation, timestamps, trace logging, config loading, and LLM initialization.
"""
import os
import sys
import json
import uuid
import traceback
import time
import random
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
from typing import Optional, Dict, Any, Literal
import yaml
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Global constants
USER_EMAIL = "vince.j.kaminski@enron.com"  # User's email address - change this to update user email globally


# ============================================================================
# Debug Utility Functions
# ============================================================================

class DebugLevel(Enum):
    """Debug verbosity levels."""
    INFO = 1   # Standard operational messages (always shown)
    DEBUG = 2  # Detailed debug messages (can be toggled)


# Global debug level (defaults to INFO, can be overridden by environment variable)
_current_debug_level = None


def _get_initial_debug_level() -> DebugLevel:
    """Get initial debug level from environment or default to INFO."""
    debug_env = os.getenv("DEBUG_LEVEL", "INFO").upper()
    if debug_env == "DEBUG":
        return DebugLevel.DEBUG
    return DebugLevel.INFO


def set_debug_level(level: DebugLevel) -> None:
    """Set the global debug level."""
    global _current_debug_level
    _current_debug_level = level


def get_debug_level() -> DebugLevel:
    """Get the current debug level."""
    global _current_debug_level
    if _current_debug_level is None:
        _current_debug_level = _get_initial_debug_level()
    return _current_debug_level


def debug_print(
    message: str,
    level: DebugLevel = DebugLevel.INFO,
    truncate: bool = True,
    max_length: int = 200,
    prefix: str = "",
    file=None,
    flush: bool = True
) -> None:
    """
    Print debug message with optional truncation.
    
    Args:
        message: Message to print
        level: Debug level (INFO or DEBUG)
        truncate: Whether to truncate long messages
        max_length: Maximum length before truncation (default: 200)
        prefix: Optional prefix to add before message
        file: File to write to (default: sys.stdout, but uses current stdout at call time)
        flush: Whether to flush output immediately
    """
    current_level = get_debug_level()
    if level.value > current_level.value:
        return
    
    if truncate and len(message) > max_length:
        message = message[:max_length] + "..."
    
    output = f"{prefix}{message}" if prefix else message
    # Use current sys.stdout (which may be redirected to log file) instead of default parameter
    output_file = file if file is not None else sys.stdout
    print(output, file=output_file, flush=flush)


def debug_info(message: str, truncate: bool = True, max_length: int = 200, **kwargs) -> None:
    """Print INFO level message (always shown)."""
    debug_print(message, level=DebugLevel.INFO, truncate=truncate, max_length=max_length, **kwargs)


def debug_debug(message: str, truncate: bool = True, max_length: int = 200, **kwargs) -> None:
    """Print DEBUG level message (only shown if DEBUG level is enabled)."""
    debug_print(message, level=DebugLevel.DEBUG, truncate=truncate, max_length=max_length, **kwargs)


def debug_print_exception(
    exception: Exception,
    context: str = "",
    level: DebugLevel = DebugLevel.INFO,
    include_traceback: bool = True
) -> None:
    """
    Print exception with optional traceback.
    
    Args:
        exception: Exception to print
        context: Optional context string
        level: Debug level
        include_traceback: Whether to include full traceback
    """
    current_level = get_debug_level()
    if level.value > current_level.value:
        return
    
    error_msg = f"{type(exception).__name__}: {str(exception)}"
    if context:
        error_msg = f"{context} - {error_msg}"
    
    if include_traceback:
        traceback_str = traceback.format_exc()
        if len(traceback_str) > 1000:
            traceback_str = traceback_str[:1000] + "..."
        debug_print(f"ERROR: {error_msg}\n{traceback_str}", level=level, truncate=False)
    else:
        debug_print(f"ERROR: {error_msg}", level=level, truncate=False)


def debug_print_long_content(
    content: str,
    label: str = "Content",
    level: DebugLevel = DebugLevel.DEBUG,
    max_length: int = 500
) -> None:
    """
    Print long content with truncation and length info.
    
    Args:
        content: Content to print
        label: Label for the content
        level: Debug level
        max_length: Maximum length to show
    """
    current_level = get_debug_level()
    if level.value > current_level.value:
        return
    
    content_length = len(content)
    if content_length > max_length:
        preview = content[:max_length] + "..."
        debug_print(f"{label} ({content_length} chars): {preview}", level=level, truncate=False)
    else:
        debug_print(f"{label} ({content_length} chars): {content}", level=level, truncate=False)


def set_global_seeds(seed: int = 42) -> None:
    """
    Set global random seeds for reproducibility.
    
    This sets seeds for:
    - Python's random module
    - NumPy's random module
    - PYTHONHASHSEED environment variable
    
    Args:
        seed: Integer seed value (default: 42)
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
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


def _get_model_context_windows() -> Dict[str, int]:
    """Get dictionary mapping model names to their context windows."""
    return {
        "gpt-5-mini": 400000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4.1-mini": 1000000,
        "o1": 200000,
        "o1-mini": 200000,
        "claude-3-7-sonnet": 200000,
        "gemini-2.0-flash": 1000000,
    }


def _get_api_token_limits() -> Dict[str, int]:
    """Get dictionary mapping model names to their API token limits."""
    return {
        "gpt-5-mini": 272000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4.1-mini": 1000000,
        "o1": 200000,
        "o1-mini": 200000,
        "claude-3-7-sonnet": 200000,
        "gemini-2.0-flash": 1000000,
    }


def get_model_context_window(model_name: str, default: int = 128000) -> int:
    """
    Get context window for a model name.
    
    Args:
        model_name: Model name (case-insensitive)
        default: Default context window if model not found
    
    Returns:
        Context window size
    """
    return _get_model_context_windows().get(model_name.lower(), default)


def load_config(config_path: str = "agent_config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Ensure config is a dict
    if not isinstance(config, dict):
        raise ValueError(f"Configuration file must contain a dictionary, got {type(config)}")
    
    # Look up context window and API token limits for the model and add them to config
    model_context_windows = _get_model_context_windows()
    api_token_limits = _get_api_token_limits()
    
    # Get model name and set context window, API token limit, and provider in config
    agent_config = config.get("agent", {})
    if isinstance(agent_config, dict):
        model_name = agent_config.get("target_model_name")
        if model_name:
            context_window = model_context_windows.get(model_name.lower(), 128000)
            api_token_limit = api_token_limits.get(model_name.lower(), 128000)
            
            # Detect provider if not already set in config
            provider = agent_config.get("provider")
            if provider is None:
                provider = detect_provider(model_name)
            
            if "agent" not in config:
                config["agent"] = {}
            config["agent"]["context_window"] = context_window
            config["agent"]["api_token_limit"] = api_token_limit
            config["agent"]["provider"] = provider
    
    return config


# ============================================================================
# Colored Trace Printer
# ============================================================================

class ColoredTracePrinter:
    """Prints trace events with color coding to terminal."""
    
    # ANSI color codes
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Colors for different event types
    USER_INPUT = '\033[94m'  # Blue
    AGENT_RESPONSE = '\033[92m'  # Green
    TOOL_CALL = '\033[93m'  # Yellow
    TOOL_RESULT = '\033[96m'  # Cyan
    ERROR = '\033[91m'  # Red
    INFO = '\033[90m'  # Dark gray
    META = '\033[95m'  # Magenta
    PASSED_COLOR = '\033[92m'  # Green
    
    MAX_LENGTH = 400  # Maximum length for truncation
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the colored trace printer.
        
        Args:
            enabled: Whether to enable colored output (auto-detected if None)
        """
        self.enabled = enabled and sys.stdout.isatty() if enabled else False
    
    def _truncate(self, text: str, max_len: int = None) -> str:
        """Truncate text to max_len characters, adding '...' if truncated."""
        max_len = max_len or self.MAX_LENGTH
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."
    
    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if enabled."""
        if self.enabled:
            return f"{color}{text}{self.RESET}"
        return text
    
    def format_trace_event(self, event: Dict[str, Any]) -> str:
        """
        Format a trace event from our trace system.
        
        Args:
            event: Trace event dictionary
            
        Returns:
            Formatted colored string
        """
        event_type = event.get('event_type', 'unknown')
        payload = event.get('payload', {})
        
        if event_type == 'user_input':
            text = payload.get('text', '')
            text = self._truncate(text)
            return self._colorize(
                f"User: {text}",
                self.USER_INPUT
            )
        
        elif event_type == 'agent_response':
            text = payload.get('text', '')
            return self._colorize(
                f"Agent: {text}",
                self.AGENT_RESPONSE
            )
        
        elif event_type == 'tool_call':
            tool_name = payload.get('tool_name', 'unknown')
            inputs = payload.get('inputs', {})
            inputs_str = json.dumps(inputs, indent=2) if inputs else "{}"
            inputs_str = self._truncate(inputs_str)
            return self._colorize(
                f"Tool Call: {tool_name}\n{self.BOLD}Inputs:{self.RESET} {inputs_str}",
                self.TOOL_CALL
            )
        
        elif event_type == 'tool_result':
            tool_name = payload.get('tool_name', 'unknown')
            outputs = payload.get('outputs', {})
            # Extract result or error
            if 'error' in outputs:
                error = str(outputs['error'])
                error = self._truncate(error)
                return self._colorize(
                    f"Tool Error ({tool_name}): {error}",
                    self.ERROR
                )
            elif 'result' in outputs:
                result = str(outputs['result'])
                result = self._truncate(result)
                return self._colorize(
                    f"Tool Result ({tool_name}): {result}",
                    self.TOOL_RESULT
                )
            else:
                outputs_str = json.dumps(outputs, indent=2)
                outputs_str = self._truncate(outputs_str)
                return self._colorize(
                    f"Tool Result ({tool_name}): {outputs_str}",
                    self.TOOL_RESULT
                )
        
        elif event_type == 'validator_results':
            operator = payload.get('operator', 'AND')
            overall_passed = payload.get('overall_passed', False)
            validators = payload.get('validators', [])
            
            status_icon = "✓" if overall_passed else "✗"
            status_text = "PASSED" if overall_passed else "FAILED"
            status_color = self.PASSED_COLOR if overall_passed else self.ERROR
            
            result_lines = [
                f"{status_color}{status_icon} Validator Results ({operator}): {status_text}{self.RESET}"
            ]
            
            for v in validators:
                v_type = v.get('type', 'unknown')
                v_name = v.get('name', 'validator')
                v_passed = v.get('passed', False)
                v_icon = "✓" if v_passed else "✗"
                v_color = self.PASSED_COLOR if v_passed else self.ERROR
                result_lines.append(
                    f"  {v_color}{v_icon}{self.RESET} [{v_type}] {v_name}: {'PASSED' if v_passed else 'FAILED'}"
                )
            
            return "\n".join(result_lines)
        
        else:
            # Generic event
            event_str = json.dumps(payload, indent=2)
            event_str = self._truncate(event_str)
            return self._colorize(
                f"{event_type}: {event_str}",
                self.INFO
            )
    
    def print_trace_event(self, event: Dict[str, Any]):
        """
        Print trace event in colored format.
        
        Args:
            event: Trace event dictionary
        """
        formatted = self.format_trace_event(event)
        print(formatted)


# Global instance
_colored_printer = None

def get_colored_printer() -> ColoredTracePrinter:
    """Get the global colored printer instance."""
    global _colored_printer
    if _colored_printer is None:
        _colored_printer = ColoredTracePrinter()
    return _colored_printer


def print_validator_result(validator_type: str, validator_name: str, passed: bool, details: str = ""):
    """
    Print a validator result with color coding.
    
    Args:
        validator_type: Type of validator (keyword, tool_call, semantic_judge, etc.)
        validator_name: Descriptive name for the validator
        passed: Whether the validator passed
        details: Optional additional details
    """
    printer = get_colored_printer()
    
    # Color for validator results
    VALIDATOR_COLOR = '\033[95m'  # Magenta
    PASSED_COLOR = '\033[92m'  # Green
    FAILED_COLOR = '\033[91m'  # Red
    
    status_icon = "✓" if passed else "✗"
    status_text = "PASSED" if passed else "FAILED"
    status_color = PASSED_COLOR if passed else FAILED_COLOR
    
    if printer.enabled:
        validator_info = f"{VALIDATOR_COLOR}[{validator_type}]{printer.RESET} {validator_name}"
        status_info = f"{status_color}{status_icon} {status_text}{printer.RESET}"
        if details:
            print(f"  {validator_info}: {status_info} - {details}")
        else:
            print(f"  {validator_info}: {status_info}")
    else:
        # No color fallback
        print(f"  [{validator_type}] {validator_name}: {status_icon} {status_text}" + (f" - {details}" if details else ""))


def ensure_data_directories(config: dict) -> None:
    """No-op: This benchmark operates entirely in-memory.
    
    Kept for API compatibility only - no directories are created.
    """
    # This benchmark operates entirely in-memory - no directories needed
    pass


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
    seed: Optional[int] = None,
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
        seed: Seed parameter for determinism (supported by OpenAI API for some models)
    
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
    # Add seed parameter for determinism (supported by OpenAI API for some models)
    if seed is not None:
        params["seed"] = seed
    
    max_retries = 5
    base_delay = 1.0  # Start with 1 second
    
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**params)
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for rate limiting errors
            is_rate_limit = "rate limit" in error_str or "429" in error_str or "quota" in error_str
            
            # Check for connection errors (retryable)
            is_connection_error = (
                "connection" in error_str or 
                "timeout" in error_str or
                "network" in error_str or
                "apiconnectionerror" in error_str or
                type(e).__name__ == "APIConnectionError"
            )
            
            # Only retry on rate limits or connection errors
            if (is_rate_limit or is_connection_error) and attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                error_type_display = "RATE LIMIT" if is_rate_limit else "CONNECTION"
                print(f"WARNING: {error_type_display} ERROR (attempt {attempt + 1}/{max_retries}): {e}", file=sys.stderr, flush=True)
                print(f"   Retrying in {delay:.2f}s...", file=sys.stderr, flush=True)
                time.sleep(delay)
                continue
            else:
                # Log non-retryable errors or final failure
                if is_rate_limit:
                    print(f"WARNING: RATE LIMIT ERROR (final after {max_retries} attempts): {e}", file=sys.stderr, flush=True)
                elif is_connection_error:
                    print(f"WARNING: CONNECTION ERROR (final after {max_retries} attempts): {e}", file=sys.stderr, flush=True)
                else:
                    print(f"WARNING: API ERROR (non-retryable): {e}", file=sys.stderr, flush=True)
                # Re-raise the exception (this will be caught by test_bench and tracked)
                raise


def call_gemini_chat_completion(
    model: str,
    messages: list,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
) -> Any:
    """
    Call Gemini chat completion API.
    
    Args:
        model: Model name (e.g., "gemini-2.5-pro")
        messages: List of message dicts with "role" and "content"
        temperature: Temperature parameter
        max_output_tokens: Maximum output tokens
        top_p: Top-p parameter
    
    Returns:
        Response object from Gemini API (wrapped to be OpenAI-compatible)
    """
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
    seed: Optional[int] = None,
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
            seed=seed,
        )
    elif provider == "gemini":
        # Map max_tokens or max_completion_tokens to max_output_tokens for Gemini
        if max_output_tokens is None:
            max_output_tokens = max_completion_tokens if max_completion_tokens is not None else max_tokens
        
        return call_gemini_chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

