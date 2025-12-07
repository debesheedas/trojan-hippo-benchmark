#!/usr/bin/env python3
"""
Unified Preprocessing Script for Persona Chat History

Preprocesses persona chat history to build memory for any memory backend.
Replaces the three separate scripts:
- preprocess_persona_chat_history_explicit_memory.py
- preprocess_persona_chat_history_rag.py
- preprocess_persona_chat_history_mem0.py

Usage:
    python scripts/preprocess_persona_chat_history.py --memory-backend explicit
    python scripts/preprocess_persona_chat_history.py --memory-backend mem0 --persona-id 66
    python scripts/preprocess_persona_chat_history.py --memory-backend rag --num-messages 10
"""

import json
import sys
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datasets import load_dataset

# Add project root to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

# Import after path setup
try:
    from agent.utils import load_config
    from agent.agent_core import invoke_agent, clear_agent_cache
    from agent.backend.memory_manager import get_memory_manager
    from agent.backend.rag_memory_manager import get_rag_memory_manager
    from agent.backend.mem0_memory_manager import get_mem0_memory_manager, _mem0_manager_cache
    from benchmark.rag_memory_loader import create_rag_memory_set_from_text
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


def process_messages_common(
    user_messages: List[str],
    config: Dict[str, Any],
    session_id: str = "persona_66_preprocessing"
) -> List[Dict[str, str]]:
    """
    Common message processing logic - processes messages through agent.
    
    Returns:
        List of conversation turns (user message + agent response)
    """
    print(f"\nProcessing {len(user_messages)} messages through agent...")
    
    # Clear any existing cache
    clear_agent_cache()
    
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
    
    return conversation_history


def process_explicit_memory(
    user_messages: List[str],
    config: Dict[str, Any],
    session_id: str = "persona_66_preprocessing"
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Process messages and build explicit memory.
    
    Returns:
        - List of conversation turns
        - Final memory state (dict with long_term list)
    """
    # Get explicit memory config
    explicit_memory_config = config.get("memory", {}).get("explicit_memory", {})
    if not explicit_memory_config.get("enabled", False):
        raise ValueError("explicit_memory must be enabled in config")
    
    # Use a temporary memory file for preprocessing
    temp_dir = Path(tempfile.gettempdir())
    memory_file = str(temp_dir / "agent_memory_persona66_temp.json")
    
    # Remove existing temp memory file if it exists
    temp_memory_path = Path(memory_file)
    if temp_memory_path.exists():
        temp_memory_path.unlink()
        print(f"✓ Removed existing temp memory file: {memory_file}")
    
    # Update config to use the temp memory file
    config["memory"]["explicit_memory"]["memory_file"] = memory_file
    config["data"] = config.get("data", {})
    config["data"]["memory_file"] = memory_file
    print(f"✓ Using memory file: {memory_file}")
    
    # Process messages
    conversation_history = process_messages_common(user_messages, config, session_id)
    
    # Get final memory state
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
            print("  Note: Explicit memory only saves when agent calls update_memory tool")
        
        final_memory = {
            "long_term": long_term_memories
        }
        
    except Exception as e:
        print(f"  ⚠️ Warning: Could not extract memory state: {e}")
        import traceback
        traceback.print_exc()
        final_memory = {"long_term": []}
    
    return conversation_history, final_memory


def process_rag_memory(
    user_messages: List[str],
    config: Dict[str, Any],
    session_id: str = "persona_66_preprocessing"
) -> Tuple[List[Dict[str, str]], str]:
    """
    Process messages and build RAG memory.
    
    Returns:
        - List of conversation turns
        - Full conversation text (for chunking)
    """
    # Get RAG config
    rag_config = config.get("memory", {}).get("rag_memory", {})
    if not rag_config.get("enabled", False):
        raise ValueError("rag_memory must be enabled in config")
    
    # Use a temporary vectorstore path for preprocessing
    temp_dir = Path(tempfile.gettempdir())
    vectorstore_path = str(temp_dir / "rag_vectorstore_persona66_temp")
    
    # Remove existing temp vectorstore if it exists
    temp_vectorstore_path = Path(vectorstore_path)
    if temp_vectorstore_path.exists():
        shutil.rmtree(temp_vectorstore_path)
        print(f"✓ Removed existing temp vectorstore: {vectorstore_path}")
    
    # Update config to use the temp vectorstore path
    config["memory"]["rag_memory"]["vectorstore_path"] = vectorstore_path
    print(f"✓ Using vectorstore path: {vectorstore_path}")
    
    # Process messages
    conversation_history = process_messages_common(user_messages, config, session_id)
    
    # Extract all conversation text from RAG memory
    print(f"\nExtracting RAG memory content...")
    try:
        rag_memory_manager = get_rag_memory_manager(
            embedding_model=rag_config.get("embedding_model", "text-embedding-3-small"),
            top_k=rag_config.get("top_k", 3),
            chunk_size=rag_config.get("chunk_size", 512),
            vectorstore_path=vectorstore_path,
            force_new=False
        )
        
        # Get all documents from the vectorstore
        all_documents = rag_memory_manager.documents if hasattr(rag_memory_manager, 'documents') else []
        
        # Combine all conversation turns into full text
        full_conversation_text = "\n\n".join(all_documents)
        
        print(f"✓ Extracted {len(all_documents)} conversation chunks from RAG memory")
        if all_documents:
            print("  Sample chunks:")
            for i, doc in enumerate(all_documents[:3], 1):
                print(f"    {i}. {doc[:100]}...")
        
        return conversation_history, full_conversation_text
        
    except Exception as e:
        print(f"  ⚠️ Warning: Could not extract RAG memory content: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: use conversation history to build text
        full_text = ""
        for turn in conversation_history:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "user":
                full_text += f"User: {content}\n\n"
            elif role == "assistant":
                full_text += f"Assistant: {content}\n\n"
        return conversation_history, full_text


def process_mem0_memory(
    user_messages: List[str],
    config: Dict[str, Any],
    session_id: str = "persona_66_preprocessing"
) -> Tuple[List[Dict[str, str]], str]:
    """
    Process messages and build mem0 memory.
    
    Returns:
        - List of conversation turns
        - Path to the mem0 vectorstore
    """
    # Clear mem0 manager cache to ensure fresh manager is created
    try:
        _mem0_manager_cache.clear()
        print("✓ Cleared mem0 manager cache")
    except Exception as e:
        print(f"⚠️ Could not clear mem0 manager cache: {e}")
    
    # Get mem0 config
    mem0_config = config.get("memory", {}).get("mem0_memory", {})
    if not mem0_config.get("enabled", False):
        raise ValueError("mem0_memory must be enabled in config")
    
    # Use a temporary vectorstore path for preprocessing
    temp_dir = Path(tempfile.gettempdir())
    vectorstore_path = str(temp_dir / "mem0_vectorstore_persona66_temp")
    
    # Remove existing temp vectorstore if it exists
    temp_vectorstore_path = Path(vectorstore_path)
    if temp_vectorstore_path.exists():
        shutil.rmtree(temp_vectorstore_path)
        print(f"✓ Removed existing temp vectorstore: {vectorstore_path}")
    
    # Update config to use the temp vectorstore path
    config["memory"]["mem0_memory"]["vectorstore_path"] = vectorstore_path
    print(f"✓ Using vectorstore path: {vectorstore_path}")
    
    # Process messages
    conversation_history = process_messages_common(user_messages, config, session_id)
    
    # Verify that memories were created
    print(f"\nVerifying mem0 memories were created...")
    try:
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
            agent_id=None,
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


def save_explicit_memory_set(memory_state: Dict[str, Any], memory_set_name: str) -> Path:
    """
    Save the pre-processed explicit memory to the unified initial_memory directory.
    
    New structure: initial_memory/{set_number}/explicit.json
    Also saves to old location for backward compatibility.
    
    Returns:
        Path to saved memory file (new structure)
    """
    # Extract set number from memory_set_name (e.g., "memory_set_3" -> "3", or "3" -> "3")
    import re
    match = re.search(r'(\d+)$', memory_set_name)
    set_number = match.group(1) if match else memory_set_name
    
    # New unified structure
    target_file = Path("data/benchmark/initial_environment/initial_memory") / set_number / "explicit.json"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(memory_state, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved memory set to {target_file}")
    
    # Also save to old location for backward compatibility
    old_target_file = Path("data/benchmark/initial_environment/initial_explicit_memory") / f"{memory_set_name}.json"
    old_target_file.parent.mkdir(parents=True, exist_ok=True)
    with open(old_target_file, 'w', encoding='utf-8') as f:
        json.dump(memory_state, f, indent=2, ensure_ascii=False)
    print(f"✓ Also saved to old location: {old_target_file}")
    
    return target_file


def save_rag_memory_set(conversation_text: str, rag_memory_set: str, chunk_size: int = 512) -> Path:
    """
    Save the pre-processed RAG memory as a JSON file with chunks.
    
    New structure: initial_memory/{set_number}/rag.json
    Also saves to old location for backward compatibility.
    
    Returns:
        Path to saved RAG memory file (new structure)
    """
    # Extract set number from rag_memory_set (e.g., "rag_memory_set_1" -> "1", or "1" -> "1")
    import re
    match = re.search(r'(\d+)$', rag_memory_set)
    set_number = match.group(1) if match else rag_memory_set
    
    # New unified structure
    output_dir = Path("data/benchmark/initial_environment/initial_memory") / set_number
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "rag.json"
    
    # Use the helper function to create the RAG memory set JSON
    create_rag_memory_set_from_text(
        text=conversation_text,
        output_path=str(output_file),
        chunk_size=chunk_size,
        metadata={
            "source": "persona_chat_history",
            "type": "preprocessed_conversation"
        }
    )
    
    print(f"✓ Saved RAG memory set to {output_file}")
    
    # Also save to old location for backward compatibility
    old_output_dir = Path("data/benchmark/initial_environment/initial_rag_memory")
    old_output_dir.mkdir(parents=True, exist_ok=True)
    old_output_file = old_output_dir / f"{rag_memory_set}.json"
    # Copy the file
    import shutil
    shutil.copy2(output_file, old_output_file)
    print(f"✓ Also saved to old location: {old_output_file}")
    
    return output_file


def save_mem0_memory_set(source_vectorstore: Path, mem0_memory_set: str) -> Path:
    """
    Copy the pre-processed vectorstore to the unified initial_memory directory.
    
    New structure: initial_memory/{set_number}/mem0/
    Also saves to old location for backward compatibility.
    
    Returns:
        Path to copied vectorstore directory (new structure)
    """
    # Extract set number from mem0_memory_set (e.g., "mem0_memory_set_1" -> "1", or "1" -> "1")
    import re
    match = re.search(r'(\d+)$', mem0_memory_set)
    set_number = match.group(1) if match else mem0_memory_set
    
    # New unified structure
    target_dir = Path("data/benchmark/initial_environment/initial_memory") / set_number / "mem0"
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    
    if not source_vectorstore.exists():
        print(f"⚠️ Warning: Source vectorstore not found at {source_vectorstore}")
        raise FileNotFoundError(f"Source vectorstore not found at {source_vectorstore}")
    
    if target_dir.exists():
        shutil.rmtree(target_dir)
    
    shutil.copytree(source_vectorstore, target_dir)
    print(f"✓ Copied vectorstore to {target_dir}")
    
    # Also copy to old location for backward compatibility
    old_target_dir = Path("data/benchmark/initial_environment/initial_mem0_memory") / mem0_memory_set
    old_target_dir.parent.mkdir(parents=True, exist_ok=True)
    if old_target_dir.exists():
        shutil.rmtree(old_target_dir)
    shutil.copytree(source_vectorstore, old_target_dir)
    print(f"✓ Also copied to old location: {old_target_dir}")
    
    return target_dir


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess persona chat history to build memory for any backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess for explicit memory
  python scripts/preprocess_persona_chat_history.py --memory-backend explicit
  
  # Preprocess for mem0 with specific persona
  python scripts/preprocess_persona_chat_history.py --memory-backend mem0 --persona-id 66
  
  # Preprocess for RAG with limited messages
  python scripts/preprocess_persona_chat_history.py --memory-backend rag --num-messages 10
        """
    )
    
    parser.add_argument(
        "--memory-backend",
        type=str,
        choices=["explicit", "mem0", "rag"],
        required=True,
        help="Memory backend to use (explicit, mem0, or rag)"
    )
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
    parser.add_argument(
        "--memory-set-name",
        type=str,
        default=None,
        help="Name for the memory set (default: {backend}_memory_set_1)"
    )
    
    args = parser.parse_args()
    
    persona_id = args.persona_id
    num_messages = args.num_messages
    session_id = args.session_id or f"persona_{persona_id}_preprocessing"
    memory_backend = args.memory_backend
    
    # Default memory set names
    if args.memory_set_name:
        memory_set_name = args.memory_set_name
    else:
        if memory_backend == "explicit":
            memory_set_name = "memory_set_3"
        elif memory_backend == "mem0":
            memory_set_name = "mem0_memory_set_1"
        else:  # rag
            memory_set_name = "rag_memory_set_1"
    
    print("="*80)
    print(f"Preprocessing Persona Chat History ({memory_backend.upper()} Memory)")
    print("="*80)
    print(f"Persona ID: {persona_id}")
    if num_messages is None:
        print(f"Number of messages: ALL (processing entire chat history)")
    else:
        print(f"Number of messages: {num_messages}")
    print(f"Session ID: {session_id}")
    print(f"Memory Set Name: {memory_set_name}")
    print("="*80)
    
    try:
        # Load config
        config = load_config()
        
        # Configure memory backend
        if "memory" not in config:
            config["memory"] = {}
        
        # Enable the specified backend and disable others
        for backend_name in ["explicit", "mem0", "rag"]:
            if backend_name not in config["memory"]:
                config["memory"][backend_name] = {}
            
            if backend_name == memory_backend:
                config["memory"][backend_name]["enabled"] = True
                print(f"✓ Enabled {backend_name}_memory")
                
                # Set defense_type to "none" for preprocessing (no restrictions)
                if backend_name == "mem0":
                    original_defense = config["memory"][backend_name].get("defense_type", "none")
                    config["memory"][backend_name]["defense_type"] = "none"
                    print(f"✓ Set defense_type to 'none' for preprocessing (was: {original_defense})")
            else:
                config["memory"][backend_name]["enabled"] = False
                print(f"✓ Disabled {backend_name}_memory")
        
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
        
        # Process messages based on backend
        if memory_backend == "explicit":
            conversation_history, memory_state = process_explicit_memory(
                user_messages, config, session_id
            )
            
            # Save session history
            session_history_path = Path(f"data/benchmark/initial_environment/initial_sessions/persona_{persona_id}_session_explicit.json")
            save_session_history(conversation_history, session_history_path)
            
            # Save memory set
            memory_file_path = save_explicit_memory_set(memory_state, memory_set_name)
            
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
            print(f"3. Update test case to use: {memory_set_name}")
            print(f"{'='*80}")
            
        elif memory_backend == "rag":
            conversation_history, conversation_text = process_rag_memory(
                user_messages, config, session_id
            )
            
            # Save session history
            session_history_path = Path(f"data/benchmark/initial_environment/initial_sessions/persona_{persona_id}_session.json")
            save_session_history(conversation_history, session_history_path)
            
            # Save the RAG memory set as JSON
            chunk_size = config.get("memory", {}).get("rag_memory", {}).get("chunk_size", 512)
            
            print(f"\n{'='*80}")
            print(f"Saving RAG memory set...")
            print(f"{'='*80}")
            
            if not conversation_text or not conversation_text.strip():
                print(f"⚠️ Warning: Conversation text is empty!")
                print(f"  This may indicate that RAG memory was not properly stored.")
                print(f"  Falling back to conversation history...")
                # Build text from conversation history
                conversation_text = ""
                for turn in conversation_history:
                    role = turn.get("role", "")
                    content = turn.get("content", "")
                    if role == "user":
                        conversation_text += f"User: {content}\n\n"
                    elif role == "assistant":
                        conversation_text += f"Assistant: {content}\n\n"
            
            # Save RAG memory set
            output_file = save_rag_memory_set(conversation_text, memory_set_name, chunk_size)
            
            # Verify the file was created
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    chunks = data.get("chunks", [])
                    print(f"✓ Verified RAG memory set: {len(chunks)} chunks saved")
            else:
                print(f"⚠️ Warning: RAG memory set file not found at {output_file}")
            
            # Clean up temporary vectorstore AFTER successful save
            temp_vectorstore = Path(config["memory"]["rag_memory"]["vectorstore_path"])
            if temp_vectorstore.exists() and temp_vectorstore.name.endswith("_temp"):
                print(f"\nCleaning up temporary vectorstore: {temp_vectorstore}")
                shutil.rmtree(temp_vectorstore)
                print("✓ Cleaned up temporary vectorstore")
            
            print("\n" + "="*80)
            print("✓ Preprocessing complete!")
            print("="*80)
            print(f"Session history (reference): {session_history_path}")
            print(f"RAG memory set (final location): {output_file}")
            print("\nNext steps:")
            print(f"1. Update test case to use {memory_set_name}")
            print("2. Remove chat history messages from test case")
            print("3. Keep only the questions in the test case")
            
        elif memory_backend == "mem0":
            conversation_history, vectorstore_path = process_mem0_memory(
                user_messages, config, session_id
            )
            
            # Save session history
            session_history_path = Path(f"data/benchmark/initial_environment/initial_sessions/persona_{persona_id}_session.json")
            save_session_history(conversation_history, session_history_path)
            
            # Copy the vectorstore to the initial_mem0_memory directory
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
            target_dir = save_mem0_memory_set(source_vectorstore, memory_set_name)
            
            # Verify the copy was successful
            if target_dir.exists():
                target_faiss_files = list(target_dir.glob("*.faiss"))
                print(f"✓ Verified copy: {len(target_faiss_files)} FAISS file(s) in target directory")
                
                # Verify memories are accessible in the copied location
                try:
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
                        agent_id=None,
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
                shutil.rmtree(temp_vectorstore)
                print("✓ Cleaned up temporary vectorstore")
            
            print("\n" + "="*80)
            print("✓ Preprocessing complete!")
            print("="*80)
            print(f"Session history (reference): {session_history_path}")
            print(f"Mem0 vectorstore (final location): {target_dir}")
            print("\nNext steps:")
            print(f"1. Update test case to use {memory_set_name}")
            print("2. Remove chat history messages from test case")
            print("3. Keep only the questions in the test case")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

