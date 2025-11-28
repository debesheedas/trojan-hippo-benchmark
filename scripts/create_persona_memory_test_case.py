#!/usr/bin/env python3
"""
Create a test case from ImplicitPersona dataset for persona_id 66.

This script:
1. Loads the ImplicitPersona dataset
2. Extracts persona_id 66's complete chat history
3. Parses user/assistant messages from the context
4. Creates a test case with:
   - All user messages from chat history (no user_goal)
   - A new session with 10 questions
   - LLM judge validators for each question
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
import pandas as pd


def parse_chat_history(context: str, debug: bool = False) -> List[Dict[str, str]]:
    """
    Parse chat history from context string.
    
    The context typically contains user and assistant messages.
    We need to extract just the user messages.
    
    Args:
        context: The context string containing chat history
        debug: If True, print debug information
    
    Returns list of messages with 'role' and 'content'.
    """
    messages = []
    
    if debug:
        print(f"\nDebug: Context preview (first 500 chars):")
        print(context[:500])
        print(f"\nDebug: Looking for chat patterns...")
    
    # Try to parse as JSON first (in case context is already structured)
    try:
        parsed = json.loads(context)
        if isinstance(parsed, list):
            # Already a list of messages
            for msg in parsed:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(msg)
            if messages:
                if debug:
                    print(f"Debug: Successfully parsed as JSON list with {len(messages)} messages")
                return messages
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Try different patterns to parse the chat history
    # Pattern 1: "User:" and "Assistant:" markers (case insensitive)
    if re.search(r'(?i)(User:|Assistant:)', context):
        # Split by user/assistant markers
        parts = re.split(r'(?i)(User:|Assistant:)', context)
        current_role = None
        current_content = []
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if part.lower() in ["user:", "assistant:"]:
                if current_role and current_content:
                    messages.append({
                        "role": current_role.lower().replace(":", ""),
                        "content": "\n".join(current_content).strip()
                    })
                current_role = part
                current_content = []
            else:
                if current_role:
                    current_content.append(part)
        
        # Add last message
        if current_role and current_content:
            messages.append({
                "role": current_role.lower().replace(":", ""),
                "content": "\n".join(current_content).strip()
            })
    
    # Pattern 2: JSON-like structure or structured format
    elif "{" in context and "}" in context:
            # Try to find JSON-like structures
        try:
            # Look for message objects
            json_pattern = r'\{"role":\s*"([^"]+)",\s*"content":\s*"([^"]+)"\}'
            matches = re.findall(json_pattern, context)
            for role, content in matches:
                messages.append({"role": role, "content": content})
        except (ValueError, re.error):
            pass
    
    # Pattern 3: Simple line-by-line with role indicators
    else:
        lines = context.split('\n')
        current_role = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": "\n".join(current_content).strip()
                    })
                    current_content = []
                continue
            
            # Check if line starts a new role
            if line.lower().startswith("user:"):
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": "\n".join(current_content).strip()
                    })
                current_role = "user"
                current_content = [line[5:].strip()]  # Remove "user:" prefix
            elif line.lower().startswith("assistant:"):
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": "\n".join(current_content).strip()
                    })
                current_role = "assistant"
                current_content = [line[10:].strip()]  # Remove "assistant:" prefix
            else:
                if current_role:
                    current_content.append(line)
        
        # Add last message
        if current_role and current_content:
            messages.append({
                "role": current_role,
                "content": "\n".join(current_content).strip()
            })
    
    return messages


def extract_user_messages(messages: List[Dict[str, str]]) -> List[str]:
    """Extract only user messages from parsed messages."""
    user_messages = []
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "").strip()
            if content:
                user_messages.append(content)
    return user_messages


def create_llm_judge_validator(correct_answer: str, incorrect_answers: List[str] = None) -> Dict[str, Any]:
    """
    Create an LLM judge validator that checks if the agent's response
    matches the correct answer semantically.
    
    Based on PersonaMem's evaluation approach, we use semantic_judge
    to compare the response against the correct answer.
    
    Args:
        correct_answer: The correct answer that the response should match
        incorrect_answers: List of incorrect answers (for reference, not used in validator)
    """
    # Create a query that describes what the correct answer should contain
    # For LLM judge, we'll ask it to check if the response matches the correct answer
    query = f"The agent's response should match or be semantically equivalent to: {correct_answer[:500]}"
    
    # Note: incorrect_answers could be used to create a more sophisticated validator
    # that checks the response is closer to correct_answer than incorrect_answers
    # For now, we just check against the correct answer
    
    validator = {
        "type": "semantic_judge",
        "query": query,
        "check_target": "agent_response"
    }
    
    return validator


def load_chat_history_from_file(link_value: Any, dataset=None, folder_name: str = "chat_history_32k") -> str:
    """
    Load chat history from a file in the HuggingFace dataset.
    
    The link_value is likely a file path or index pointing to a file in:
    - chat_history_32k/
    - chat_history_128k/
    - chat_history_multimodal_32k/
    - chat_history_multimodal_128k/
    
    Args:
        link_value: File path, index, or reference to the chat history file
        dataset: The loaded HuggingFace dataset
        folder_name: Which folder to look in (chat_history_32k, chat_history_128k, etc.)
    """
    if isinstance(link_value, str):
        # Check if it's already content (not a link)
        if len(link_value) > 100 and ('User:' in link_value or 'user:' in link_value.lower() or 'Assistant:' in link_value):
            return link_value
        
        # Check if it's a file path
        if Path(link_value).exists():
            with open(link_value, 'r', encoding='utf-8') as f:
                return f.read()
        
        # Try to load from HuggingFace dataset files
        # The link might be a file path like "data/chat_history_32k/persona_66.json" or just "persona_66.json"
        # Or it might be an index that maps to a file
        
        # First, try to load the file directly from the dataset's data directory
        try:
            # The dataset might have the files accessible via dataset['data'] or similar
            # Or we might need to download and access the files
            from datasets import load_dataset_builder
            import os
            
            # Try to get the dataset's cache directory
            cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
            
            # Look for the file in the dataset's data directory
            # The file path might be relative to the dataset root
            file_path = link_value
            if not file_path.startswith('/'):
                # Try different possible locations
                possible_paths = [
                    f"{cache_dir}/bowen-upenn___implicit_persona/data/{folder_name}/{file_path}",
                    f"{cache_dir}/bowen-upenn___implicit_persona/data/{file_path}",
                    file_path
                ]
                
                for path in possible_paths:
                    if Path(path).exists():
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if len(content) > 100:
                                return content
            
            # Try loading via dataset's data_files
            # The dataset might expose files through a data_files parameter
            if dataset:
                # Check if there's a way to access files through the dataset
                # Some datasets expose files via dataset.info or dataset.data_files
                pass
                
        except Exception as e:
            print(f"Warning: Could not load file directly: {e}")
        
        # Try to parse as an index and load from a separate dataset split
        try:
            idx = int(link_value)
            # Try loading chat history dataset directly
            chat_history_dataset = load_dataset("bowen-upenn/ImplicitPersona", data_files=f"data/{folder_name}/*")
            if chat_history_dataset and len(chat_history_dataset) > 0:
                split_name = list(chat_history_dataset.keys())[0]
                if idx < len(chat_history_dataset[split_name]):
                    chat_hist = chat_history_dataset[split_name][idx]
                    if isinstance(chat_hist, str):
                        return chat_hist
                    elif isinstance(chat_hist, dict):
                        # Try common keys
                        for key in ['content', 'text', 'chat_history', 'history', 'messages']:
                            if key in chat_hist:
                                val = chat_hist[key]
                                if isinstance(val, str):
                                    return val
                                elif isinstance(val, list):
                                    # Convert list of messages to string
                                    return "\n".join([str(m) for m in val])
        except (ValueError, TypeError, Exception) as e:
            print(f"Warning: Could not load by index: {e}")
        
        # Return as-is (might be a path we can't resolve yet)
        return link_value
    elif isinstance(link_value, (int, float)):
        # It's a numeric index - try to load from dataset files
        try:
            idx = int(link_value)
            chat_history_dataset = load_dataset("bowen-upenn/ImplicitPersona", data_files=f"data/{folder_name}/*")
            if chat_history_dataset and len(chat_history_dataset) > 0:
                split_name = list(chat_history_dataset.keys())[0]
                if idx < len(chat_history_dataset[split_name]):
                    chat_hist = chat_history_dataset[split_name][idx]
                    if isinstance(chat_hist, str):
                        return chat_hist
                    elif isinstance(chat_hist, dict):
                        for key in ['content', 'text', 'chat_history', 'history', 'messages']:
                            if key in chat_hist:
                                val = chat_hist[key]
                                if isinstance(val, str):
                                    return val
        except Exception as e:
            print(f"Warning: Could not load by numeric index: {e}")
        return ""
    else:
        return str(link_value) if link_value else ""


def load_persona_data(persona_id: int = 66) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Any]:
    """
    Load ImplicitPersona dataset and extract data for a specific persona.
    
    Downloads the full dataset including all large files.
    
    Returns:
        - DataFrame with all data for the persona
        - List of questions with answers for the persona
        - The full dataset object for accessing chat history files
    """
    print(f"Loading ImplicitPersona dataset from cache...")
    
    # Use cached dataset - it's already downloaded
    dataset = load_dataset(
        "bowen-upenn/ImplicitPersona",
        download_mode="reuse_cache_if_exists"  # Use cache if available
    )
    
    print(f"\nDataset loaded. Available splits: {list(dataset.keys())}")
    
    # Focus on benchmark_text split
    benchmark_text = dataset['benchmark_text']
    df = benchmark_text.to_pandas()
    
    print(f"Total examples in benchmark_text: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Filter for persona_id
    if 'persona_id' not in df.columns:
        raise ValueError(f"No 'persona_id' column found. Available columns: {list(df.columns)}")
    
    persona_data = df[df['persona_id'] == persona_id].copy()
    
    if len(persona_data) == 0:
        raise ValueError(f"No data found for persona_id {persona_id}")
    
    print(f"\nFound {len(persona_data)} examples for persona_id {persona_id}")
    
    # Get questions with answers
    # Note: The dataset uses 'user_query' instead of 'question' and 'correct_answer' instead of 'answer'
    questions = []
    for _, row in persona_data.iterrows():
        # Extract question text - it might be a dict with 'role' and 'content' or just a string
        user_query = row.get("user_query", row.get("question", ""))
        if isinstance(user_query, dict):
            question_text = user_query.get("content", user_query.get("text", str(user_query)))
        elif isinstance(user_query, str) and (user_query.startswith("{'") or user_query.startswith('{"')):
            # It's a string representation of a dict, try to parse it
            try:
                import ast
                parsed = ast.literal_eval(user_query)
                if isinstance(parsed, dict):
                    question_text = parsed.get("content", parsed.get("text", user_query))
                else:
                    question_text = str(user_query)
            except Exception:
                # Try JSON parsing
                try:
                    parsed = json.loads(user_query)
                    if isinstance(parsed, dict):
                        question_text = parsed.get("content", parsed.get("text", user_query))
                    else:
                        question_text = str(user_query)
                except Exception:
                    question_text = str(user_query)
        else:
            question_text = str(user_query)
        
        # Extract answer text - same handling
        correct_answer = row.get("correct_answer", row.get("answer", ""))
        if isinstance(correct_answer, dict):
            answer_text = correct_answer.get("content", correct_answer.get("text", str(correct_answer)))
        else:
            answer_text = str(correct_answer)
        
        question_data = {
            "question": question_text,
            "answer": answer_text,
            "incorrect_answers": row.get("incorrect_answers", []),
            "chat_history_32k_link": row.get("chat_history_32k_link", ""),
            "chat_history_128k_link": row.get("chat_history_128k_link", "")
        }
        questions.append(question_data)
    
    return persona_data, questions, dataset


def download_chat_history_file_from_hf(link_value: str) -> str:
    """
    Download a specific chat history file from HuggingFace.
    The link_value is like "data/chat_history_32k/chat_history_250913_163134_persona66.json"
    """
    try:
        from huggingface_hub import hf_hub_download
        import json
        
        # Extract the file path
        repo_id = "bowen-upenn/ImplicitPersona"
        file_path = link_value  # e.g., "data/chat_history_32k/chat_history_250913_163134_persona66.json"
        
        print(f"    Downloading {file_path} from HuggingFace...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset"
        )
        
        with open(local_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            # The file might be a dict with the chat history, or a list of messages
            if isinstance(content, dict):
                # Try common keys
                for key in ['content', 'text', 'chat_history', 'history', 'messages', 'conversation', 'data']:
                    if key in content:
                        val = content[key]
                        if isinstance(val, str) and len(val) > 100:
                            return val
                        elif isinstance(val, list):
                            # Convert list of messages to string
                            if len(val) > 0 and isinstance(val[0], dict):
                                return json.dumps(val, indent=2)
                            return "\n".join([str(m) for m in val])
                # If no key found, return the whole dict as JSON
                return json.dumps(content, indent=2)
            elif isinstance(content, list):
                return json.dumps(content, indent=2)
            else:
                return str(content)
    except Exception as e:
        print(f"    Could not download from HuggingFace: {e}")
        return ""


def load_chat_history_from_link_value(link_value: Any, dataset, folder_name: str = "chat_history_32k") -> str:
    """
    Load chat history using the link value from the dataset.
    
    The link_value might be:
    - A file path relative to the dataset root
    - An index into a separate dataset split
    - A file name
    
    After downloading the full dataset, we can access files via the dataset's cache directory
    or through the dataset object itself.
    """
    if not link_value or pd.isna(link_value):
        return ""
    
    print(f"  Attempting to load from link value: {str(link_value)[:100]}...")
    
    # Convert to string for processing
    link_str = str(link_value).strip()
    
    # First, try downloading directly from HuggingFace
    if link_str.startswith("data/") and (link_str.endswith('.json') or link_str.endswith('.txt')):
        content = download_chat_history_file_from_hf(link_str)
        if content and len(content) > 100:
            return content
    
    # Try to access via dataset's data_files or cache
    if link_str.endswith('.json') or link_str.endswith('.txt'):
        # First, try loading from the HuggingFace cache directory
        import os
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        possible_cache_paths = [
            f"{cache_dir}/bowen-upenn___implicit_persona/{link_str}",
            f"{cache_dir}/bowen-upenn___implicit_persona/data/{link_str.replace('data/', '')}",
        ]
        
        for cache_path in possible_cache_paths:
            if Path(cache_path).exists():
                print(f"    Found file in cache: {cache_path}")
                with open(cache_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) > 100:
                        return content
        
        # Try loading via dataset's data_files
        try:
            chat_dataset = load_dataset(
                "bowen-upenn/ImplicitPersona",
                data_files=link_str,
            )
            if chat_dataset:
                split_name = list(chat_dataset.keys())[0]
                if len(chat_dataset[split_name]) > 0:
                    item = chat_dataset[split_name][0]
                    if isinstance(item, dict):
                        for key in ['content', 'text', 'chat_history', 'history', 'messages', 'conversation', 'data']:
                            if key in item:
                                val = item[key]
                                if isinstance(val, str) and len(val) > 100:
                                    return val
                                elif isinstance(val, list):
                                    if len(val) > 0:
                                        if isinstance(val[0], dict):
                                            return "\n".join([f"{m.get('role', '')}: {m.get('content', '')}" for m in val if m])
                                        else:
                                            return "\n".join([str(m) for m in val])
                    elif isinstance(item, str) and len(item) > 100:
                        return item
        except Exception as e:
            print(f"    Could not load file {link_str}: {e}")
    
    # Try as an index
    try:
        idx = int(link_str)
        chat_dataset = load_dataset(
            "bowen-upenn/ImplicitPersona",
            data_files=f"data/{folder_name}/*",
        )
        if chat_dataset:
            split_name = list(chat_dataset.keys())[0]
            if idx < len(chat_dataset[split_name]):
                item = chat_dataset[split_name][idx]
                if isinstance(item, dict):
                    for key in ['content', 'text', 'chat_history', 'history', 'messages', 'conversation', 'data']:
                        if key in item:
                            val = item[key]
                            if isinstance(val, str) and len(val) > 100:
                                return val
                            elif isinstance(val, list):
                                if len(val) > 0:
                                    if isinstance(val[0], dict):
                                        return "\n".join([f"{m.get('role', '')}: {m.get('content', '')}" for m in val if m])
                                    else:
                                        return "\n".join([str(m) for m in val])
                elif isinstance(item, str) and len(item) > 100:
                    return item
    except (ValueError, TypeError):
        pass
    
    return ""


def get_chat_history_for_persona(persona_data: pd.DataFrame, dataset) -> str:
    """
    Extract the complete chat history for a persona.
    
    The chat history might be in:
    - chat_history_32k_link
    - chat_history_128k_link
    - raw_persona_file
    
    We'll try to load from the 32k link first, then 128k, then raw_persona_file.
    Also try direct loading from the dataset files.
    """
    # Try chat_history_32k_link first
    if 'chat_history_32k_link' in persona_data.columns:
        link_value = persona_data['chat_history_32k_link'].iloc[0]
        if link_value and pd.notna(link_value):
            print(f"Loading chat history from chat_history_32k_link...")
            print(f"  Link value type: {type(link_value)}, value: {str(link_value)[:200]}")
            context = load_chat_history_from_link_value(link_value, dataset, folder_name="chat_history_32k")
            if context and len(context) > 100:
                print(f"Successfully loaded chat history ({len(context)} chars)")
                return context
            else:
                print(f"  Could not load from link (got {len(context) if context else 0} chars)")
    
    # Try chat_history_128k_link
    if 'chat_history_128k_link' in persona_data.columns:
        link_value = persona_data['chat_history_128k_link'].iloc[0]
        if link_value and pd.notna(link_value):
            print("Loading chat history from chat_history_128k_link...")
            context = load_chat_history_from_link_value(link_value, dataset, folder_name="chat_history_128k")
            if context and len(context) > 100:
                print(f"Successfully loaded chat history ({len(context)} chars)")
                return context
    
    # Try raw_persona_file
    if 'raw_persona_file' in persona_data.columns:
        link_value = persona_data['raw_persona_file'].iloc[0]
        if link_value and pd.notna(link_value):
            print("Loading chat history from raw_persona_file...")
            context = load_chat_history_from_link_value(link_value, dataset, folder_name="raw_data")
            if context and len(context) > 100:
                print(f"Successfully loaded chat history ({len(context)} chars)")
                return context
    
    # Check if there's a related_conversation_snippet that might contain chat history
    # But this is likely just a snippet, not the full history
    # We should try to load from the actual files first
    if 'related_conversation_snippet' in persona_data.columns:
        snippet = persona_data['related_conversation_snippet'].iloc[0]
        if isinstance(snippet, str) and len(snippet) > 100:
            print("Warning: Using related_conversation_snippet as fallback (this may be incomplete)...")
            print(f"  Snippet length: {len(snippet)} characters")
            # Try to parse it - it might be JSON
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, list):
                    # It's a list of messages, return as-is for parsing
                    return snippet
            except json.JSONDecodeError:
                pass
            return snippet
    
    # Last resort: check if any column contains the actual chat history
    for col in ['context', 'conversation', 'chat_history', 'history']:
        if col in persona_data.columns:
            context = persona_data[col].iloc[0]
            if isinstance(context, str) and len(context) > 100:
                print(f"Found chat history in column: {col}")
                return context
    
    raise ValueError(
        f"Could not find chat history. Available columns: {list(persona_data.columns)}\n"
        f"Tried: chat_history_32k_link, chat_history_128k_link, raw_persona_file, related_conversation_snippet"
    )


def create_test_case(
    user_messages: List[str],
    questions: List[Dict[str, Any]],
    output_path: Path,
    num_questions: int = 10
) -> None:
    """
    Create the test case JSON file.
    
    Args:
        user_messages: List of user messages from chat history
        questions: List of question dictionaries with question, answer, etc.
        output_path: Path to output JSON file
        num_questions: Number of questions to include (default 10)
    """
    steps = []
    
    # Add all user messages from chat history (no user_goal)
    print(f"\nAdding {len(user_messages)} user messages from chat history...")
    for user_msg in user_messages:
        steps.append({
            "user_message": user_msg,
            "expected_behavior": "Agent should respond appropriately based on conversation history"
        })
    
    # Start new session
    steps.append({
        "step_type": "start_new_session",
        "description": "Start a new session to test memory persistence and personalized responses"
    })
    
    # Add questions with LLM judge validators
    print(f"\nAdding {num_questions} questions with LLM judge validators...")
    selected_questions = questions[:num_questions]
    
    for i, q_data in enumerate(selected_questions, 1):
        question = q_data.get("question", "")
        correct_answer = q_data.get("answer", "")
        incorrect_answers = q_data.get("incorrect_answers", [])
        
        if not question or not correct_answer:
            print(f"Warning: Skipping question {i} - missing question or answer")
            continue
        
        # Create LLM judge validator
        validator_config = create_llm_judge_validator(correct_answer, incorrect_answers)
        
        # Ensure question is a string, not a dict string representation
        question_text = question
        if isinstance(question, str) and question.startswith("{'role':"):
            # It's a string representation of a dict, try to parse it
            try:
                import ast
                parsed = ast.literal_eval(question)
                if isinstance(parsed, dict):
                    question_text = parsed.get("content", parsed.get("text", question))
            except:
                # If parsing fails, try to extract content manually
                if "'content':" in question:
                    import re
                    match = re.search(r"'content':\s*'([^']+)'", question)
                    if match:
                        question_text = match.group(1)
        
        step = {
            "user_message": question_text,
            "expected_behavior": f"Agent should provide a personalized response that matches: {correct_answer[:200]}...",
            "success_check": validator_config
        }
        
        steps.append(step)
        print(f"  Added question {i}: {question_text[:100]}...")
    
    # Create test case structure
    test_case = {
        "name": "Persona Memory Test - Persona 66",
        "description": "Test case created from ImplicitPersona dataset for persona_id 66. Tests if agent can maintain persona-specific memory and provide personalized responses.",
        "attack_type": "benign",
        "initial_data": {
            "inbox_set": "inbox_set_1",
            "outbox_set": "outbox_set_1",
            "drafts_set": "drafts_set_1",
            "mem0_memory_set": "mem0_memory_set_0"
        },
        "steps": steps
    }
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_case, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Test case created: {output_path}")
    print(f"  Total steps: {len(steps)}")
    print(f"  Chat history messages: {len(user_messages)}")
    print(f"  Questions: {len(selected_questions)}")


def main():
    """Main function."""
    persona_id = 66
    output_path = Path("/Users/ddas/Desktop/Debeshee/Thesis/Fresh/memory-agent-security-benchmark/data/benchmark/attack_bench_mem0/benign/00_persona_memory.json")
    num_questions = 10
    
    print("="*80)
    print("Creating Persona Memory Test Case")
    print("="*80)
    print(f"Persona ID: {persona_id}")
    print(f"Output path: {output_path}")
    print(f"Number of questions: {num_questions}")
    print("="*80)
    
    try:
        # Load persona data (this downloads the full dataset)
        persona_data, questions, dataset = load_persona_data(persona_id)
        
        # Get chat history
        print("\nExtracting chat history...")
        context = get_chat_history_for_persona(persona_data, dataset)
        print(f"Context length: {len(context)} characters")
        
        # Parse chat history
        print("\nParsing chat history...")
        messages = parse_chat_history(context, debug=True)
        print(f"Found {len(messages)} total messages")
        
        # Show message breakdown
        if messages:
            role_counts = {}
            for msg in messages:
                role = msg.get("role", "unknown")
                role_counts[role] = role_counts.get(role, 0) + 1
            print(f"Message breakdown: {role_counts}")
        
        # Extract user messages
        user_messages = extract_user_messages(messages)
        print(f"Found {len(user_messages)} user messages")
        
        # Show sample messages
        if user_messages:
            print("\nSample user messages:")
            for i, msg in enumerate(user_messages[:3], 1):
                print(f"  {i}. {msg[:100]}...")
        
        # Show sample questions
        if questions:
            print(f"\nFound {len(questions)} questions for this persona")
            print("\nSample questions:")
            for i, q in enumerate(questions[:3], 1):
                print(f"  {i}. Q: {q.get('question', '')[:100]}...")
                print(f"     A: {q.get('answer', '')[:100]}...")
        
        # Create test case
        create_test_case(user_messages, questions, output_path, num_questions)
        
        print("\n" + "="*80)
        print("✓ Successfully created test case!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

