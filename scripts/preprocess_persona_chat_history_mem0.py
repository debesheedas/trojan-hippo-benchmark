#!/usr/bin/env python3
"""
Preprocess persona chat history to build mem0 memories.

This script:
1. Loads user messages from persona 66's chat history
2. Processes each message through the agent (with mem0 enabled)
3. Saves the session history (all user messages + agent responses)
4. Saves the final mem0 memory state
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
                    print(f"    {i}. {msg[:80]}...")
            return user_messages[:num_messages]
        else:
            print(f"  ⚠️ Could not parse user messages from text format")
            print(f"  Preview (first 500 chars): {chat_data[:500]}")
    
    # If it's a list, extract messages directly
    elif isinstance(chat_data, list):
        messages = chat_data
        print(f"  Chat data is a list with {len(messages)} items")
    
    # If it's a dict, look for message arrays
    elif isinstance(chat_data, dict):
        print(f"  Chat data is a dict with keys: {list(chat_data.keys())}")
        # Try common keys that might contain messages
        for key in ['messages', 'conversation', 'chat_history', 'history', 'data']:
            if key in chat_data:
                val = chat_data[key]
                if isinstance(val, list):
                    messages = val
                    print(f"  Found '{key}' key with {len(messages)} items")
                    break
                elif isinstance(val, str):
                    # Recursively parse string value
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, list):
                            messages = parsed
                            print(f"  Successfully parsed '{key}' string as JSON list")
                            break
                    except:
                        pass
        
        # If still no messages, check if the dict itself has role/content
        if not messages and 'role' in chat_data and 'content' in chat_data:
            messages = [chat_data]
            print(f"  Treating dict as single message")
    
    # Extract user messages from structured format
    if messages:
        print(f"  Total messages to process: {len(messages)}")
        role_counts = {}
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                role_counts[role] = role_counts.get(role, 0) + 1
                
                if role == "user" and content:
                    user_messages.append(content)
                    if len(user_messages) <= 3:
                        print(f"    User message {len(user_messages)}: {content[:80]}...")
            elif isinstance(msg, str):
                # If message is just a string, assume it's a user message
                user_messages.append(msg)
                if len(user_messages) <= 3:
                    print(f"    User message {len(user_messages)} (string): {msg[:80]}...")
            else:
                print(f"  ⚠️ Message {i} is not a dict or string: {type(msg)}")
        
        if messages and isinstance(messages[0], dict):
            print(f"  Message role breakdown: {role_counts}")
    
    print(f"  Found {len(user_messages)} user messages with content")
    if num_messages is None:
        print(f"  Using all {len(user_messages)} messages for preprocessing")
    else:
        print(f"  Using first {num_messages} messages for preprocessing")
        user_messages = user_messages[:num_messages]
    
    if len(user_messages) == 0:
        print(f"  ⚠️ WARNING: No user messages found!")
        print(f"  This might indicate an issue with the chat history format.")
        if messages:
            print(f"  Sample of first message: {messages[0]}")
        elif isinstance(chat_data, str):
            print(f"  Chat data preview (first 500 chars): {chat_data[:500]}")
    
    # Return all messages (caller will slice if needed)
    return user_messages


def process_messages_and_build_memory(
    user_messages: List[str],
    config: Dict[str, Any],
    session_id: str = "persona_66_preprocessing"
) -> tuple[List[Dict[str, str]], str]:
    """
    Process user messages through the agent and build mem0 memories.
    
    Returns:
        - List of conversation turns (user message + agent response)
        - Path to the mem0 vectorstore
    """
    print(f"\nProcessing {len(user_messages)} messages through agent...")
    
    # Clear any existing cache
    clear_agent_cache()
    
    # Clear mem0 manager cache to ensure fresh manager is created
    try:
        from agent.backend.mem0_memory_manager import _mem0_manager_cache
        _mem0_manager_cache.clear()
        print("✓ Cleared mem0 manager cache")
    except Exception as e:
        print(f"⚠️ Could not clear mem0 manager cache: {e}")
    
    # Get mem0 config
    mem0_config = config.get("memory", {}).get("mem0_memory", {})
    if not mem0_config.get("enabled", False):
        raise ValueError("mem0_memory must be enabled in config")
    
    # Use a temporary vectorstore path for preprocessing
    # We'll copy it to the final location after processing
    # Use a temp path that won't conflict with anything
    import tempfile
    temp_dir = Path(tempfile.gettempdir())
    vectorstore_path = str(temp_dir / "mem0_vectorstore_persona66_temp")
    
    # Remove existing temp vectorstore if it exists
    temp_vectorstore_path = Path(vectorstore_path)
    if temp_vectorstore_path.exists():
        import shutil
        shutil.rmtree(temp_vectorstore_path)
        print(f"✓ Removed existing temp vectorstore: {vectorstore_path}")
    
    # Update config to use the temp vectorstore path
    # This ensures the agent uses this path when processing messages
    config["memory"]["mem0_memory"]["vectorstore_path"] = vectorstore_path
    print(f"✓ Using vectorstore path: {vectorstore_path}")
    
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
            
            # Note: The agent automatically indexes memories via mem0 after each turn
            
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
    
    # Verify that memories were created
    print(f"\nVerifying mem0 memories were created...")
    try:
        from agent.backend.mem0_memory_manager import get_mem0_memory_manager
        
        mem0_manager = get_mem0_memory_manager(
            llm_provider=mem0_config.get("llm_provider", "openai"),
            llm_model=mem0_config.get("llm_model", "gpt-4o-mini"),
            llm_temperature=mem0_config.get("llm_temperature", 0.0),
            embedding_provider=mem0_config.get("embedding_provider", "openai"),
            embedding_model=mem0_config.get("embedding_model", "text-embedding-3-small"),
            vector_store_provider=mem0_config.get("vector_store_provider", "faiss"),
            vectorstore_path=vectorstore_path,
            top_k=mem0_config.get("top_k", 3),
            user_id="vince",
            agent_id=None,  # Memories are stored with agent_id=None
            force_new=False
        )
        
        # Get all memories
        memories = mem0_manager.get_all_memories(user_id="vince", agent_id=None, limit=1000)
        print(f"✓ Found {len(memories)} memories in vectorstore")
        
        if memories:
            print("  Sample memories:")
            for i, mem in enumerate(memories[:3], 1):
                if isinstance(mem, dict):
                    memory_text = mem.get("memory", "")
                    print(f"    {i}. {memory_text[:100]}...")
        else:
            print("  ⚠️ Warning: No memories found! The vectorstore may be empty.")
            
        # Check if vectorstore directory exists and has files
        vectorstore_path_obj = Path(vectorstore_path)
        faiss_files = list(vectorstore_path_obj.glob("*.faiss"))
        print(f"  FAISS files: {len(faiss_files)}")
        
    except Exception as e:
        print(f"  ⚠️ Warning: Could not verify memories: {e}")
        import traceback
        traceback.print_exc()
    
    return conversation_history, vectorstore_path


def save_session_history(conversation_history: List[Dict[str, str]], output_path: Path):
    """Save conversation history to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conversation_history, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved session history to {output_path}")


def copy_vectorstore_to_memory_set(source_vectorstore: Path, mem0_memory_set: str):
    """
    Copy the pre-processed vectorstore to the initial_mem0_memory directory.
    This is where test cases will look for pre-processed vectorstores.
    
    The vectorstore is stored directly in: data/benchmark/initial_mem0_memory/{mem0_memory_set}/
    """
    target_dir = Path("data/benchmark/initial_mem0_memory") / mem0_memory_set
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    
    if not source_vectorstore.exists():
        print(f"⚠️ Warning: Source vectorstore not found at {source_vectorstore}")
        return False
    
    import shutil
    if target_dir.exists():
        shutil.rmtree(target_dir)
    
    shutil.copytree(source_vectorstore, target_dir)
    print(f"✓ Copied vectorstore to {target_dir}")
    return True


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess persona chat history to build mem0 memories")
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
    print("Preprocessing Persona Chat History")
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
        
        # IMPORTANT: Set defense_type to "none" for preprocessing
        # We want all memories to be indexed without any defense restrictions
        # Later, we can redo preprocessing with different defenses if needed
        if "memory" in config and "mem0_memory" in config["memory"]:
            original_defense_type = config["memory"]["mem0_memory"].get("defense_type", "none")
            config["memory"]["mem0_memory"]["defense_type"] = "none"
            print(f"✓ Set defense_type to 'none' for preprocessing (was: {original_defense_type})")
            print(f"  This ensures all memories are indexed without restrictions")
        
        # Load user messages (pass None to get all messages)
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
        conversation_history, vectorstore_path = process_messages_and_build_memory(
            user_messages,
            config,
            session_id
        )
        
        # Save session history (for reference, not used by mem0)
        session_history_path = Path("data/benchmark/initial_sessions/persona_66_session.json")
        save_session_history(conversation_history, session_history_path)
        
        # Copy the vectorstore to the initial_mem0_memory directory
        # This is where test cases will look for pre-processed vectorstores
        mem0_memory_set_name = "mem0_memory_set_1"
        source_vectorstore = Path(vectorstore_path)
        
        print(f"\n{'='*80}")
        print(f"Copying vectorstore to initial_mem0_memory directory...")
        print(f"{'='*80}")
        
        if not source_vectorstore.exists():
            print(f"⚠️ Error: Vectorstore not found at {source_vectorstore}")
            print(f"  The vectorstore should have been created during message processing.")
            print(f"  Check if mem0 is indexing correctly and the path is correct.")
            raise FileNotFoundError(f"Vectorstore not found at {source_vectorstore}")
        
        # Verify vectorstore has files
        faiss_files = list(source_vectorstore.glob("*.faiss"))
        if not faiss_files:
            print(f"⚠️ Warning: Vectorstore exists but has no FAISS files!")
            print(f"  This may indicate that memories were not created.")
        else:
            print(f"✓ Vectorstore has {len(faiss_files)} FAISS file(s)")
        
        # Copy to final location
        success = copy_vectorstore_to_memory_set(source_vectorstore, mem0_memory_set_name)
        
        if not success:
            raise RuntimeError(f"Failed to copy vectorstore to {mem0_memory_set_name}")
        
        # Verify the copy was successful
        target_dir = Path("data/benchmark/initial_mem0_memory") / mem0_memory_set_name
        if target_dir.exists():
            target_faiss_files = list(target_dir.glob("*.faiss"))
            print(f"✓ Verified copy: {len(target_faiss_files)} FAISS file(s) in target directory")
            
            # Verify memories are accessible in the copied location
            try:
                from agent.backend.mem0_memory_manager import get_mem0_memory_manager
                mem0_config = config.get("memory", {}).get("mem0_memory", {})
                
                mem0_manager = get_mem0_memory_manager(
                    llm_provider=mem0_config.get("llm_provider", "openai"),
                    llm_model=mem0_config.get("llm_model", "gpt-4o-mini"),
                    llm_temperature=mem0_config.get("llm_temperature", 0.0),
                    embedding_provider=mem0_config.get("embedding_provider", "openai"),
                    embedding_model=mem0_config.get("embedding_model", "text-embedding-3-small"),
                    vector_store_provider=mem0_config.get("vector_store_provider", "faiss"),
                    vectorstore_path=str(target_dir),
                    top_k=mem0_config.get("top_k", 3),
                    user_id="vince",
                    agent_id=None,  # Memories are stored with agent_id=None
                    force_new=False
                )
                
                memories = mem0_manager.get_all_memories(user_id="vince", agent_id=None, limit=1000)
                print(f"✓ Verified memories in copied location: {len(memories)} memories found")
            except Exception as e:
                print(f"⚠️ Warning: Could not verify memories in copied location: {e}")
        
        # Clean up temporary vectorstore AFTER successful copy
        temp_vectorstore = Path(vectorstore_path)
        if temp_vectorstore.exists() and temp_vectorstore.name.endswith("_temp"):
            print(f"\nCleaning up temporary vectorstore: {vectorstore_path}")
            import shutil
            shutil.rmtree(temp_vectorstore)
            print("✓ Cleaned up temporary vectorstore")
        
        print("\n" + "="*80)
        print("✓ Preprocessing complete!")
        print("="*80)
        print(f"Session history (reference): {session_history_path}")
        print(f"Mem0 vectorstore (final location): data/benchmark/initial_mem0_memory/{mem0_memory_set_name}/")
        print("\nNext steps:")
        print("1. Update test case to use mem0_memory_set_1")
        print("2. Remove chat history messages from test case")
        print("3. Keep only the questions in the test case")
        print("\nNote: The vectorstore is stored at:")
        print(f"   data/benchmark/initial_mem0_memory/{mem0_memory_set_name}/")
        print("   Test cases will automatically copy from there to their test directories.")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

