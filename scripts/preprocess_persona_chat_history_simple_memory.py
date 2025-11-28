#!/usr/bin/env python3
"""
Preprocess persona chat history to build simple memory (explicit memory).

This script:
1. Loads user messages from persona 66's chat history
2. Processes each message through the agent (with simple memory enabled, mem0 disabled)
3. Saves the session history (all user messages + agent responses)
4. Saves the final simple memory state as memory_set_3.json
5. Then the test case can use these pre-processed files
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
import pandas as pd

# Add project root to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

# Import after path setup
try:
    from agent.utils import load_config
    from agent.agent_core import invoke_agent, clear_agent_cache
    from agent.backend.memory_manager import get_memory_manager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root and dependencies are installed")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def load_persona_messages(persona_id: int = 66, num_messages: int = None) -> List[str]:
    """
    Load user messages from persona's chat history.
    
    Args:
        persona_id: Persona ID to load
        num_messages: Number of messages to return (None = all messages)
    
    Returns:
        List of user message strings
    """
    print(f"Loading persona {persona_id} chat history...")
    
    dataset = load_dataset("bowen-upenn/ImplicitPersona", download_mode="reuse_cache_if_exists")
    benchmark_text = dataset['benchmark_text']
    df = benchmark_text.to_pandas()
    
    persona_data = df[df['persona_id'] == persona_id].copy()
    if len(persona_data) == 0:
        raise ValueError(f"No data found for persona_id {persona_id}")
    
    # Get chat history link
    chat_history_link = persona_data['chat_history_32k_link'].iloc[0]
    
    # Download and parse chat history
    from huggingface_hub import hf_hub_download
    local_path = hf_hub_download(
        repo_id="bowen-upenn/ImplicitPersona",
        filename=chat_history_link,
        repo_type="dataset"
    )
    
    # Read the file - it might be JSON or plain text
    with open(local_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
    
    # Try to parse as JSON first
    chat_data = None
    try:
        chat_data = json.loads(file_content)
        print(f"  Successfully parsed as JSON")
    except json.JSONDecodeError:
        # Not JSON, treat as plain text
        print(f"  File is not JSON, treating as plain text")
        chat_data = file_content
    
    # Extract messages - handle different formats
    print(f"  Chat data type: {type(chat_data)}")
    
    messages = []
    user_messages = []
    
    # If it's a string, we need to parse it using regex patterns
    if isinstance(chat_data, str):
        print(f"  Chat data is a string (length: {len(chat_data)})")
        # Use regex to extract user messages
        import re
        # Pattern 1: "User:" or "user:" followed by text until next "User:" or "Assistant:"
        user_pattern = re.compile(r'(?i)(?:User|user):\s*(.+?)(?=\n(?:Assistant|assistant|User|user):|$)', re.DOTALL)
        matches = user_pattern.findall(chat_data)
        if matches:
            user_messages = [m.strip() for m in matches if m.strip()]
            print(f"  Found {len(user_messages)} user messages via text parsing")
            if user_messages:
                print(f"  Sample user messages:")
                for i, msg in enumerate(user_messages[:3], 1):
                    print(f"    {i}. {msg[:100]}...")
        
        # Limit to num_messages if specified
        if num_messages is not None:
            if len(user_messages) > num_messages:
                print(f"  Using first {num_messages} messages for preprocessing")
                user_messages = user_messages[:num_messages]
            else:
                print(f"  Using all {len(user_messages)} messages (requested {num_messages})")
        
        return user_messages
    
    # If it's a dict or list, try to extract messages
    elif isinstance(chat_data, (dict, list)):
        print(f"  Chat data is {type(chat_data).__name__}")
        # Try common formats
        if isinstance(chat_data, list):
            # List of message objects
            for item in chat_data:
                if isinstance(item, dict):
                    role = item.get("role", "").lower()
                    content = item.get("content", item.get("text", ""))
                    if role == "user" and content:
                        user_messages.append(content)
                    messages.append(item)
        elif isinstance(chat_data, dict):
            # Dict with messages key
            if "messages" in chat_data:
                messages = chat_data["messages"]
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "").lower()
                        content = msg.get("content", msg.get("text", ""))
                        if role == "user" and content:
                            user_messages.append(content)
            # Also try direct content extraction if it's a conversation format
            elif "conversation" in chat_data:
                messages = chat_data["conversation"]
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "").lower()
                        content = msg.get("content", msg.get("text", ""))
                        if role == "user" and content:
                            user_messages.append(content)
            # Try extracting from nested structures
            else:
                # Look for any list in the dict that might contain messages
                for key, value in chat_data.items():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                role = item.get("role", "").lower()
                                content = item.get("content", item.get("text", item.get("message", "")))
                                if role == "user" and content:
                                    user_messages.append(content)
        
        if not user_messages:
            print(f"  Warning: Could not extract user messages from structured format")
            print(f"  Chat data keys: {list(chat_data.keys()) if isinstance(chat_data, dict) else 'N/A'}")
            print(f"  Chat data preview (first 500 chars): {str(chat_data)[:500]}")
        
        # Limit to num_messages if specified
        if num_messages is not None:
            if len(user_messages) > num_messages:
                print(f"  Using first {num_messages} messages for preprocessing")
                user_messages = user_messages[:num_messages]
            else:
                print(f"  Using all {len(user_messages)} messages (requested {num_messages})")
        
        return user_messages
    
    else:
        print(f"  Warning: Unexpected chat data type: {type(chat_data)}")
        return []


def process_messages_and_build_memory(
    user_messages: List[str],
    config: Dict[str, Any],
    session_id: str = "persona_66_preprocessing",
    memory_file: str = None
) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Process user messages through the agent and build simple memory.
    
    Returns:
        - List of conversation turns (user message + agent response)
        - Final memory state (dict with long_term list)
    """
    print(f"\nProcessing {len(user_messages)} messages through agent...")
    
    # Clear any existing cache
    clear_agent_cache()
    
    # Get simple memory config
    simple_memory_config = config.get("memory", {}).get("simple_memory", {})
    if not simple_memory_config.get("enabled", False):
        raise ValueError("simple_memory must be enabled in config")
    
    # Use a temporary memory file for preprocessing
    import tempfile
    temp_dir = Path(tempfile.gettempdir())
    if memory_file is None:
        memory_file = str(temp_dir / "agent_memory_persona66_temp.json")
    
    # Remove existing temp memory file if it exists
    temp_memory_path = Path(memory_file)
    if temp_memory_path.exists():
        temp_memory_path.unlink()
        print(f"✓ Removed existing temp memory file: {memory_file}")
    
    # Update config to use the temp memory file
    config["memory"]["simple_memory"]["memory_file"] = memory_file
    config["data"] = config.get("data", {})
    config["data"]["memory_file"] = memory_file
    print(f"✓ Using memory file: {memory_file}")
    
    conversation_history = []
    
    # Process each user message
    for i, user_msg in enumerate(user_messages, 1):
        print(f"\n[{i}/{len(user_messages)}] Processing message...")
        print(f"  User: {user_msg[:100]}...")
        
        # Invoke agent directly
        try:
            result = invoke_agent(
                text=user_msg,
                session_id=session_id,
                config=config
            )
            
            agent_response = result.get("output", result.get("response", ""))
            print(f"  Agent: {agent_response[:100] if agent_response else 'No response'}...")
            
            # Add to conversation history
            conversation_history.append({
                "role": "user",
                "content": user_msg
            })
            conversation_history.append({
                "role": "assistant",
                "content": agent_response if agent_response else ""
            })
            
            # Note: The agent automatically updates simple memory after each turn
            
        except Exception as e:
            print(f"  Error processing message: {e}")
            import traceback
            traceback.print_exc()
            # Still add the user message
            conversation_history.append({
                "role": "user",
                "content": user_msg
            })
            conversation_history.append({
                "role": "assistant",
                "content": f"Error: {str(e)}"
            })
    
    print(f"\n✓ Processed {len(user_messages)} messages")
    print(f"  Conversation history: {len(conversation_history)} messages")
    
    # Get final memory state (whatever the agent saved via update_memory tool)
    print(f"\nExtracting final memory state...")
    try:
        memory_manager = get_memory_manager(memory_file=memory_file, force_new=False)
        memory_state = memory_manager.get_context()
        long_term_memories = memory_state.get("long_term", [])
        print(f"✓ Found {len(long_term_memories)} long-term memories")
        
        if long_term_memories:
            print("  Sample memories:")
            for i, mem in enumerate(long_term_memories[:5], 1):
                print(f"    {i}. {mem[:100]}...")
        else:
            print("  ⚠️ Warning: No long-term memories found!")
            print("  Note: Simple memory only saves when agent calls update_memory tool")
        
        final_memory = {
            "long_term": long_term_memories
        }
        
    except Exception as e:
        print(f"  ⚠️ Warning: Could not extract memory state: {e}")
        import traceback
        traceback.print_exc()
        final_memory = {"long_term": []}
    
    return conversation_history, final_memory


def save_session_history(conversation_history: List[Dict[str, str]], output_path: Path):
    """Save conversation history to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conversation_history, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved session history to {output_path}")


def save_memory_set(memory_state: Dict[str, Any], memory_set_name: str):
    """
    Save the pre-processed memory to the initial_memory directory.
    This is where test cases will look for pre-processed memory sets.
    
    The memory is saved as: data/benchmark/initial_memory/{memory_set_name}.json
    """
    target_file = Path("data/benchmark/initial_memory") / f"{memory_set_name}.json"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(memory_state, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved memory set to {target_file}")
    return target_file


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess persona chat history to build simple memory")
    parser.add_argument(
        "--persona-id",
        type=int,
        default=66,
        help="Persona ID to process (default: 66)"
    )
    parser.add_argument(
        "--num-messages",
        type=int,
        default=None,
        help="Number of messages to process (default: None = all messages)"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID for processing (default: persona_{id}_preprocessing)"
    )
    
    args = parser.parse_args()
    
    persona_id = args.persona_id
    num_messages = args.num_messages  # None means all messages
    session_id = args.session_id or f"persona_{persona_id}_preprocessing"
    
    print("="*80)
    print("Preprocessing Persona Chat History (Simple Memory)")
    print("="*80)
    print(f"Persona ID: {persona_id}")
    if num_messages is None:
        print(f"Number of messages: ALL (processing entire chat history)")
    else:
        print(f"Number of messages: {num_messages}")
    print(f"Session ID: {session_id}")
    print("="*80)
    
    try:
        # Load config
        config = load_config()
        
        # IMPORTANT: Enable simple memory and disable mem0
        if "memory" not in config:
            config["memory"] = {}
        
        # Enable simple memory
        if "simple_memory" not in config["memory"]:
            config["memory"]["simple_memory"] = {}
        config["memory"]["simple_memory"]["enabled"] = True
        print(f"✓ Enabled simple_memory")
        
        # Disable mem0
        if "mem0_memory" not in config["memory"]:
            config["memory"]["mem0_memory"] = {}
        config["memory"]["mem0_memory"]["enabled"] = False
        print(f"✓ Disabled mem0_memory")
        
        # Load user messages
        user_messages = load_persona_messages(persona_id, num_messages)
        
        # Check if we got any messages
        if not user_messages or len(user_messages) == 0:
            print(f"\n✗ Error: No user messages were loaded!")
            if num_messages:
                print(f"  Expected {num_messages} messages from persona {persona_id}")
            else:
                print(f"  Expected to find messages from persona {persona_id}")
            print(f"  This might indicate:")
            print(f"    1. The chat history format is different than expected")
            print(f"    2. The persona_id {persona_id} doesn't exist in the dataset")
            print(f"    3. The chat history file doesn't contain user messages")
            return 1
        
        if num_messages and len(user_messages) < num_messages:
            print(f"\n⚠️ Warning: Only found {len(user_messages)} messages, expected {num_messages}")
            print(f"  Will process {len(user_messages)} messages instead")
        elif num_messages is None:
            print(f"\n✓ Loaded all {len(user_messages)} user messages from chat history")
        
        # Process messages through agent
        conversation_history, memory_state = process_messages_and_build_memory(
            user_messages,
            config,
            session_id
        )
        
        # Save session history (for reference)
        session_history_path = Path("data/benchmark/initial_sessions/persona_66_session_simple.json")
        save_session_history(conversation_history, session_history_path)
        
        # Save memory set
        memory_set_name = "memory_set_3"
        memory_file_path = save_memory_set(memory_state, memory_set_name)
        
        # Verify the memory file
        if memory_file_path.exists():
            with open(memory_file_path, 'r', encoding='utf-8') as f:
                saved_memory = json.load(f)
            long_term_count = len(saved_memory.get("long_term", []))
            print(f"✓ Verified memory file: {long_term_count} long-term memories saved")
            
            if long_term_count == 0:
                print(f"  ⚠️ Warning: Memory file is empty! No memories were extracted.")
            else:
                print(f"  ✓ Memory set '{memory_set_name}' is ready for use in test cases")
        else:
            print(f"  ⚠️ Error: Memory file was not created!")
            return 1
        
        print(f"\n{'='*80}")
        print("Preprocessing Complete!")
        print(f"{'='*80}")
        print(f"1. Session history saved to: {session_history_path}")
        print(f"2. Memory set saved to: {memory_file_path}")
        print(f"3. Update test case to use: memory_set_3")
        print(f"{'='*80}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

