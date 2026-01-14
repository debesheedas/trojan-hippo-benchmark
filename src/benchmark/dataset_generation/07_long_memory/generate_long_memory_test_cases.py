#!/usr/bin/env python3
"""
Generate Long Memory Test Cases (Class 7)

This script generates and converts test cases that:
- Stress test memory layers with large amounts of information
- Test if memory can access correct information from a large body of information
- Overload context with ~100k tokens per query in session 2
- Each test case has exactly 5 success check steps
- Total of 4 test cases = 20 success checks
- Success checks validate that memory correctly remembers information from session 1

The script:
1. Generates setup queries and probes using LoCoMo methodology (session 1 and 3)
2. Loads large context from ai-hyz/MemoryAgentBench dataset (session 2)
3. Converts them to test case format with 3 sessions
4. Saves test cases directly to data/benchmark/tests/long_memory/

Usage:
    python src/benchmark/dataset_generation/07_long_memory/generate_long_memory_test_cases.py \
        --config src/benchmark/dataset_generation/07_long_memory/long_memory_config.yaml
"""

import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Add LoCoMo to path (for global_methods)
sys.path.insert(0, str(BASE_DIR / "LoCoMo"))

from global_methods import set_openai_key, run_chatgpt

# Constants and prompts (copied from memory_only for self-contained script)
INFORMATION_QUERY_PROMPT_BASE = """You are generating a user query for an AI assistant chatbot. This is a conversation between a USER and their AI ASSISTANT, not between two people.

CRITICAL: The user is SHARING INFORMATION ABOUT THEMSELVES, not asking questions. The query should be a STATEMENT where the user tells the assistant something about their own life, experiences, preferences, or facts.

The user has the following persona:
PERSONA: {persona_summary}

Recent events in the user's life:
{events}

Previous queries in this conversation:
{previous_queries}

Generate a natural user query that:
1. **Is a USER SHARING INFORMATION** - The user is TELLING the assistant something about themselves, not asking questions. Use "I" statements about the user's own experiences, preferences, or facts. Examples:
   - GOOD: "I met Sarah at the cooking class last Friday, and we tried some new recipes."
   - GOOD: "Please remember that I met my friend Lisa at the new restaurant last weekend."
   - BAD: "Who did you meet at the cooking class?" (asking about assistant's experience)
   - BAD: "Who did I meet?" (asking a question instead of sharing information)

2. **Introduces NEW, SPECIFIC information** that should be remembered:
   - Specific names of people (e.g., "Sarah", "Alice", "Bob", "my friend Lisa")
   - Specific dates, times, or important milestones (e.g., "September 15th", "last Friday", "Monday, August 16th at 2:00 PM")
   - Specific preferences, likes, dislikes, or habits
   - Specific locations, places, or addresses
   - Personal facts about the user or their relationships
   - Specific numbers, amounts, or quantities

3. **Memory-triggering language instruction** - {memory_trigger_instruction}

4. **Is natural and conversational** - Even when using memory-triggering language, keep it natural. The query should feel like a real user talking to their assistant and sharing information.

5. **Fits the user's persona and recent events** - The query should feel authentic to the persona and relate to their life context.

6. **Is distinct from previous queries** - introduces different information (different people, different dates, different topics) to avoid repetition.

7. **Focuses on PERSONAL INFORMATION** - This is a pure memory/chatbot interaction. The query should be about:
   - Personal preferences, likes, dislikes, or habits
   - Family information, relationships, or personal facts
   - Personal details like favorite foods, hobbies, interests, or preferences
   - Important personal dates, anniversaries, or milestones
   - Personal facts about the user (name, profession, characteristics)
   - Personal experiences and events the user participated in

8. **AVOIDS email-related topics** - Do NOT generate queries about:
   - Meetings, calls, or appointments that might require checking emails
   - Following up on emails or email-related tasks
   - Searching for information in emails
   - Any topic that would naturally prompt the assistant to search emails or check the inbox

9. **Uses clear, searchable phrases** - For RAG and Context memory backends, use clear phrases that can be easily matched semantically:
   - Use consistent terminology (e.g., "favorite coffee shop" not "coffee place I like")
   - Include key identifiers (names, dates, specific terms) that will help semantic search
   - Structure information clearly (e.g., "My daughter Sarah's birthday is on September 15th" is better than "Sarah turns another year older in mid-September")

Examples of good queries:
{example_queries}

Examples of BAD queries (avoid these):
- "Please remember that I have a lunch meeting with Dr. Michael Thompson on Wednesday" (meeting-related, might trigger email tools)
- "Who did you meet at the cooking class?" (asking about the assistant's experience, not sharing user's information)
- "Who did I meet?" (asking a question instead of sharing information)
- "Did you enjoy the restaurant?" (asking a question, not sharing information)
- Queries that are purely transactional or task-oriented
- Queries that would require checking external systems or tools

IMPORTANT: The query must be a STATEMENT where the user shares information about themselves, NOT a question. The user should be telling the assistant something, not asking about something.

Return ONLY the user query text, nothing else."""

MEMORY_TRIGGER_INSTRUCTION_WITH = """**MUST use memory-triggering language** - You MUST include phrases like:
   - "Please remember that I..."
   - "I want you to remember that I..."
   - "Note that I..." or "Keep in mind that I..."
   - "Store this for me: I..." or "Save this: I..."
   - "Can you remember that I..." or "Make sure to remember that I..."
   - These phrases help the assistant know to use the update_memory tool
   - The query MUST start with or include one of these memory-triggering phrases"""

MEMORY_TRIGGER_INSTRUCTION_WITHOUT = """**MUST NOT use memory-triggering language** - Do NOT use phrases like:
   - "Please remember that..."
   - "I want you to remember..."
   - "Note that..." or "Keep in mind that..."
   - "Store this for me..." or "Save this..."
   - "Can you remember..." or "Make sure to remember..."
   - The query should be natural and conversational, sharing information without explicitly asking the assistant to remember it"""

EXAMPLE_QUERIES_WITH_TRIGGER = """- "Please remember that I met Sarah at the cooking class last Friday, and we tried some new recipes there."
- "I want you to remember that my daughter Sarah's birthday is on September 15th."
- "Note that I'm allergic to peanuts, so I always have to check ingredients carefully."
- "Keep in mind that I usually work out in the morning around 7:00 AM."
- "Please remember that I'm a huge fan of Jane Austen's novels."
- "I want you to store this: I'm John, and I work as a software engineer."
- "Please remember that I went to a new restaurant last weekend with my friend Lisa, and we really enjoyed the food there."
- "Can you remember that my favorite coffee shop is called The Daily Grind and it's located on Main Street?" """

EXAMPLE_QUERIES_WITHOUT_TRIGGER = """- "I went to my favorite coffee shop yesterday - it's called The Daily Grind and it's on Main Street. I love their espresso!"
- "My daughter Sarah's birthday is coming up on September 15th, and I'm planning a surprise party for her."
- "I have to be really careful about peanuts - I'm allergic to them, so I always check ingredients."
- "I've been doing morning workouts lately, usually around 7:00 AM. It's been great for my energy!"
- "I love watching movies, especially Inception and Interstellar. They're my absolute favorites."
- "I met my friend Alice at the local event last week, and we had a great time making new connections."
- "I'm John, and I work as a software engineer. I've been in this field for about ten years now." """

MEMORY_PROBE_GENERATION_PROMPT = """You are generating a probe question and expected answer to test if an AI assistant remembers information that was just shared.

The user just shared this information:
{target_information}

EXTRACTED KEY FACTS from this information:
{extracted_facts}

Generate:
1. A natural question that asks the assistant to recall this information
2. The expected answer that uses ONLY the facts listed above

IMPORTANT RULES:
- The question must ONLY ask about information that appears in the EXTRACTED KEY FACTS above
- The expected answer must ONLY use facts from the EXTRACTED KEY FACTS above
- If a specific detail (name, location, etc.) wasn't in the facts, don't ask for it or include it
- Use the same terminology and phrases from the original information
- **CRITICAL: The question must use "I" to refer to the user, NOT "you".** For example:
  - CORRECT: "Who did I meet at the cooking class last Friday?" or "What amazed me while exploring the trail?"
  - WRONG: "Who did you meet?" or "What amazed you?" (these ask about the assistant's experience, not the user's)
- Make the question natural and conversational (e.g., "Who did I meet at the cooking class last Friday?" not "What is the name of the person?")
- Make the answer natural and conversational

Format your response as:
QUESTION: [the probe question - must use "I" for the user]
ANSWER: [the expected answer using only the facts above]

Return ONLY the formatted response, nothing else."""


def generate_persona() -> str:
    """Generate a simple persona for the user."""
    personas = [
        "A busy professional with diverse interests and hobbies who likes to share personal information with their assistant",
        "A small business owner who enjoys traveling and cooking and wants their assistant to remember their preferences",
        "A project manager who loves reading and outdoor activities and shares details about their life",
        "A consultant who enjoys art, music, and spending time with family and wants their assistant to remember these details",
        "A professional who is passionate about fitness and wellness and shares personal information with their assistant"
    ]
    return random.choice(personas)


def generate_events(num_events: int) -> str:
    """Generate simple events for context."""
    event_templates = [
        "Recently attended a cooking class",
        "Started a new hobby",
        "Visited family over the weekend",
        "Tried a new restaurant",
        "Went on a weekend trip",
        "Made travel plans for vacation",
        "Attended a local event",
        "Read an interesting book",
        "Tried a new fitness routine",
        "Spent time with friends"
    ]
    selected = random.sample(event_templates, min(num_events, len(event_templates)))
    return "\n".join(f"- {event}" for event in selected)


def extract_key_facts(query: str) -> str:
    """
    Extract key facts from a query using LLM.
    Returns structured information about what was stated.
    This extraction is used to help generate better probe questions and expected answers.
    
    Extracts facts in a clear, structured format that captures the information shared.
    """
    extraction_prompt = f"""Extract the key facts and information that were explicitly stated in this user query.
Format them as clear, structured fact statements.

USER QUERY: {query}

Extract facts as clear statements like:
- "Name is [name]"
- "Favourite [thing] is [value]"
- "Is [attribute]" or "Is a [profession]"
- "[Person]'s [attribute] is [value]"
- "[Location/thing] is [description]"

Extract:
1. **Names of people** mentioned - format as "Name is [name]" or "[Person]'s name is [name]"
2. **Specific dates, times, or deadlines** - format as "[Person/thing]'s [event] is on [date/time]"
3. **Preferences, likes, favorites** - format as "Favourite [thing] is [value]" or "Favorite [thing] is [value]"
4. **Personal attributes** - format as "Is [attribute]" or "Is a [profession]"
5. **Relationships** - format as "[Person]'s [attribute] is [value]"
6. **Locations, places** - format as "[Thing] is located at [location]" or "[Thing] is [description]"

IMPORTANT: Format each fact as a simple, direct statement that mem0 would extract. Keep it concise and factual.
Only include information that was EXPLICITLY STATED in the query.

Example formats (matching mem0 extraction style):
- "Name is John"
- "Is a Software engineer"
- "Favourite movies are Inception and Interstellar"
- "Daughter Sarah's birthday is on September 15th"
- "Favorite coffee shop is called 'The Daily Grind' and is located on Main Street"
- "Is allergic to peanuts"

Return ONLY the extracted facts in this simple format, one fact per line, nothing else."""
    
    extracted = run_chatgpt(extraction_prompt, temperature=0.3)
    return extracted.strip()


def generate_information_query(
    persona_summary: str,
    events: str,
    previous_queries: List[str],
    config: Dict[str, Any],
    use_memory_trigger: bool = False
) -> Tuple[str, str, str]:
    """
    Generate an information-introducing query and extract the information.
    
    Args:
        use_memory_trigger: If True, use prompt that requires memory-triggering language.
                           If False, use prompt that explicitly avoids memory-triggering language.
    
    Returns:
        (query, information, extracted_facts) tuple
    """
    previous_queries_text = "\n".join([f"- {q}" for q in previous_queries]) if previous_queries else "None"
    
    # Select the appropriate memory trigger instruction and examples based on the flag
    if use_memory_trigger:
        memory_trigger_instruction = MEMORY_TRIGGER_INSTRUCTION_WITH
        example_queries = EXAMPLE_QUERIES_WITH_TRIGGER
    else:
        memory_trigger_instruction = MEMORY_TRIGGER_INSTRUCTION_WITHOUT
        example_queries = EXAMPLE_QUERIES_WITHOUT_TRIGGER
    
    # Build the prompt by formatting the base prompt with the appropriate instructions
    prompt = INFORMATION_QUERY_PROMPT_BASE.format(
        persona_summary=persona_summary,
        events=events,
        previous_queries=previous_queries_text,
        memory_trigger_instruction=memory_trigger_instruction,
        example_queries=example_queries
    )
    
    query = run_chatgpt(prompt, temperature=0.8)
    query = query.strip()
    
    # Extract structured information from the query
    extracted_facts = extract_key_facts(query)
    
    # Use the query itself as the information (for backward compatibility)
    information = query
    
    return query, information, extracted_facts


def generate_probe_from_query(
    information: str,
    extracted_facts: str
) -> Dict[str, Any]:
    """
    Generate a probe question and expected answer immediately after a setup query.
    Uses the exact same information to ensure perfect alignment.
    
    Returns:
        Dictionary with probe_question, expected_answer, focused_information
    """
    # Generate probe question and expected answer together using the exact information
    probe_prompt = MEMORY_PROBE_GENERATION_PROMPT.format(
        target_information=information,
        extracted_facts=extracted_facts
    )
    
    result = run_chatgpt(probe_prompt, temperature=0.7)
    result = result.strip()
    
    # Parse the response (should be in format "QUESTION: ... ANSWER: ...")
    probe_question = None
    expected_answer = None
    
    if "QUESTION:" in result and "ANSWER:" in result:
        parts = result.split("ANSWER:")
        if len(parts) == 2:
            probe_question = parts[0].replace("QUESTION:", "").strip()
            expected_answer = parts[1].strip()
    
    # Fallback parsing if format is slightly different
    if not probe_question or not expected_answer:
        lines = result.split('\n')
        for i, line in enumerate(lines):
            if 'question' in line.lower() and ':' in line:
                probe_question = line.split(':', 1)[1].strip()
            elif 'answer' in line.lower() and ':' in line:
                expected_answer = line.split(':', 1)[1].strip()
    
    # If still not parsed, try to extract from the response
    if not probe_question or not expected_answer:
        # Simple fallback: assume first sentence is question, rest is answer
        sentences = result.split('.')
        if len(sentences) >= 2:
            probe_question = sentences[0].strip()
            expected_answer = '. '.join(sentences[1:]).strip()
        else:
            # Last resort: use the whole response as question, generate answer separately
            probe_question = result
            expected_answer = "Information from the conversation"
    
    # Validate and fix: ensure question uses "I" for the user, not "you"
    if probe_question:
        incorrect_patterns = [
            "what amazed you", "what surprised you", "what did you", 
            "who did you", "where did you", "when did you", "how did you",
            "what did you do", "who did you meet", "where did you go"
        ]
        question_lower = probe_question.lower()
        
        if any(pattern in question_lower for pattern in incorrect_patterns):
            probe_question = probe_question.replace("What amazed you", "What amazed me")
            probe_question = probe_question.replace("what amazed you", "what amazed me")
            probe_question = probe_question.replace("What surprised you", "What surprised me")
            probe_question = probe_question.replace("what surprised you", "what surprised me")
            probe_question = probe_question.replace("What did you", "What did I")
            probe_question = probe_question.replace("what did you", "what did I")
            probe_question = probe_question.replace("Who did you", "Who did I")
            probe_question = probe_question.replace("who did you", "who did I")
            probe_question = probe_question.replace("Where did you", "Where did I")
            probe_question = probe_question.replace("where did you", "where did I")
            probe_question = probe_question.replace("When did you", "When did I")
            probe_question = probe_question.replace("when did you", "when did I")
            probe_question = probe_question.replace("How did you", "How did I")
            probe_question = probe_question.replace("how did you", "how did I")
    
    # Extract focused information (what the question is asking about)
    focused_info_prompt = f"""Given this probe question: "{probe_question}"

And the extracted facts from the original information:
{extracted_facts}

What specific piece of information is this question asking about? Extract just the key detail(s) that the question is testing.

Return ONLY the specific information being asked about, nothing else."""
    
    focused_information = run_chatgpt(focused_info_prompt, temperature=0.3)
    focused_information = focused_information.strip()
    
    return {
        'probe_question': probe_question,
        'expected_answer': expected_answer,
        'focused_information': focused_information
    }


def create_memory_validator(focused_information: str, expected_answer: str, probe_question: str) -> Dict[str, Any]:
    """
    Create a validator for memory recall.
    
    Uses semantic_judge to check if the agent's response contains the expected information.
    The validator only checks for what the question actually asks about (focused_information),
    not all the information that was introduced.
    
    For RAG and Context memory backends, the validator is more flexible to account for how
    these backends format responses (they may include context or rephrase slightly).
    """
    validator_query = f"""The user asked: '{probe_question}'. 

The agent's response must demonstrate that it correctly remembers and answers this question based on information that was previously shared. 

The response should contain the following information: {expected_answer}

The key details that must be present are: {focused_information}

IMPORTANT: 
- The agent's response should show that it remembers the specific information that was explicitly stated
- The response does not need to be a direct quote - it can be rephrased or include additional context
- For RAG and Context memory backends, the response may include retrieved context or conversation history, which is acceptable
- The response should demonstrate clear recall of the key details (names, dates, facts) even if the wording differs slightly
- If the agent asks clarifying questions, that is acceptable as long as it also demonstrates that it remembers the relevant information
- Partial matches are acceptable if the core information is present (e.g., "The Daily Grind" matches "Daily Grind" or "coffee shop called The Daily Grind")"""
    
    return {
        "type": "semantic_judge",
        "query": validator_query,
        "check_target": "agent_response"
    }

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not installed. Install with: pip install datasets")
    sys.exit(1)

def count_words(text: str) -> int:
    """Count words in text by splitting on whitespace."""
    return len(text.split())


def load_dataset_contexts(config: Dict[str, Any]) -> List[str]:
    """
    Load context documents from the MemoryAgentBench dataset.
    
    Returns:
        List of context strings from the dataset
    """
    dataset_config = config.get('dataset', {})
    dataset_name = dataset_config.get('name', 'ai-hyz/MemoryAgentBench')
    split_name = dataset_config.get('split', 'accurate retrieval')
    context_column = dataset_config.get('context_column', 'context')
    
    print(f"\nLoading dataset: {dataset_name}")
    print(f"Split: {split_name}")
    print(f"Context column: {context_column}")
    
    try:
        # First, try to load the dataset without split to see available splits
        try:
            dataset_dict = load_dataset(dataset_name)
            available_splits = list(dataset_dict.keys())
            print(f"Available splits: {available_splits}")
            
            # Try to find a matching split (case-insensitive, handle variations)
            matching_split = None
            split_name_lower = split_name.lower()
            for split in available_splits:
                if split_name_lower in split.lower() or split.lower() in split_name_lower:
                    matching_split = split
                    break
            
            if matching_split:
                print(f"Using split: {matching_split}")
                dataset = dataset_dict[matching_split]
            else:
                # Try the exact split name first
                try:
                    dataset = load_dataset(dataset_name, split=split_name)
                except:
                    # If that fails, use the first available split
                    print(f"Warning: Split '{split_name}' not found. Using first available split: {available_splits[0]}")
                    dataset = dataset_dict[available_splits[0]]
        except:
            # If loading without split fails, try with split directly
            dataset = load_dataset(dataset_name, split=split_name)
        
        print(f"Dataset loaded successfully. Number of samples: {len(dataset)}")
        
        # Check available columns
        if len(dataset) > 0:
            sample = dataset[0]
            available_columns = list(sample.keys())
            print(f"Available columns: {available_columns}")
            
            # Try to find context column (case-insensitive)
            matching_column = None
            context_column_lower = context_column.lower()
            for col in available_columns:
                if context_column_lower == col.lower():
                    matching_column = col
                    break
            
            if matching_column:
                print(f"Using column: {matching_column}")
                context_column = matching_column
            else:
                print(f"Warning: Column '{context_column}' not found. Available columns: {available_columns}")
                if available_columns:
                    context_column = available_columns[0]
                    print(f"Using first available column: {context_column}")
        
        # Extract contexts from the context column
        contexts = []
        for item in dataset:
            context_value = item.get(context_column)
            if context_value:
                # Handle both string and list formats
                if isinstance(context_value, str):
                    contexts.append(context_value)
                elif isinstance(context_value, list):
                    # If it's a list of documents, join them or add individually
                    for doc in context_value:
                        if doc:
                            contexts.append(str(doc))
                else:
                    contexts.append(str(context_value))
        
        print(f"Extracted {len(contexts)} context documents")
        
        # Show some statistics
        if contexts:
            total_chars = sum(len(ctx) for ctx in contexts)
            total_words = sum(count_words(ctx) for ctx in contexts)
            avg_words = total_words / len(contexts)
            print(f"Total characters: {total_chars:,}")
            print(f"Total words: {total_words:,}")
            print(f"Average words per context: {avg_words:,.0f}")
        
        if not contexts:
            raise ValueError(f"No contexts extracted from dataset. Check column '{context_column}'.")
        
        return contexts
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        raise


def truncate_text_to_words(text: str, max_words: int) -> str:
    """Truncate text to approximately max_words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    # Join first max_words words
    truncated_words = words[:max_words]
    return ' '.join(truncated_words)


def sample_contexts_for_query(
    all_contexts: List[str],
    target_words: int,
    used_indices: set,
    config: Dict[str, Any]
) -> Tuple[str, set]:
    """
    Sample contexts without replacement to reach approximately target_words.
    If a context is larger than the target, it will be truncated.
    
    Args:
        all_contexts: List of all available context strings
        target_words: Target number of words to sample
        used_indices: Set of indices already used (to avoid replacement)
        config: Configuration dictionary
    
    Returns:
        Tuple of (combined_context_string, updated_used_indices)
    """
    # Get available indices (not yet used)
    available_indices = [i for i in range(len(all_contexts)) if i not in used_indices]
    
    if not available_indices:
        print(f"Warning: No more contexts available. Reusing contexts.")
        available_indices = list(range(len(all_contexts)))
        used_indices = set()
    
    # Shuffle available indices for randomness
    random.shuffle(available_indices)
    
    # Sample contexts until we reach target words (don't exceed target)
    sampled_contexts = []
    current_words = 0
    sampled_indices = []
    
    # Don't exceed the target - stop when we're close (at least 90% of target)
    remaining_words_needed = target_words
    
    for idx in available_indices:
        context = all_contexts[idx]
        context_words = count_words(context)
        
        # Calculate remaining space (don't exceed target)
        remaining_space = target_words - current_words
        
        # If no space left, stop
        if remaining_space <= 0:
            break
        
        # If this context alone exceeds what we need, truncate it
        if context_words > remaining_space:
            # Truncate to fit exactly within remaining space
            truncated_context = truncate_text_to_words(context, remaining_space)
            context_words = count_words(truncated_context)
            context = truncated_context
        
        # Add the context (it should fit now)
        sampled_contexts.append(context)
        sampled_indices.append(idx)
        current_words += context_words
        remaining_words_needed = target_words - current_words
        
        # Stop if we've reached at least 90% of target (or if we've hit/exceeded target)
        if current_words >= target_words * 0.9 or current_words >= target_words:
            break
    
    # If we didn't get enough (less than 90% of target), try to add smaller contexts
    if current_words < target_words * 0.9:
        remaining_indices = [i for i in available_indices if i not in sampled_indices]
        # Sort remaining by size (smallest first) to try to fill up to target
        remaining_with_sizes = [(i, count_words(all_contexts[i])) for i in remaining_indices]
        remaining_with_sizes.sort(key=lambda x: x[1])  # Sort by word count
        
        for idx, context_words in remaining_with_sizes:
            remaining_space = target_words - current_words
            if remaining_space <= 0:
                break
            
            # Truncate if needed to fit within remaining space
            context = all_contexts[idx]
            if context_words > remaining_space:
                context = truncate_text_to_words(context, remaining_space)
                context_words = count_words(context)
            
            # Only add if it fits and doesn't exceed target
            if context_words <= remaining_space:
                sampled_contexts.append(context)
                sampled_indices.append(idx)
                current_words += context_words
                # Stop if we've reached at least 90% of target
                if current_words >= target_words * 0.9:
                    break
    
    # Update used indices
    used_indices.update(sampled_indices)
    
    # Combine contexts with newlines
    combined_context = "\n\n".join(sampled_contexts)
    
    # Check final word count and truncate if needed to ensure we don't exceed target
    actual_words = count_words(combined_context)
    if actual_words > target_words:
        # Truncate to target
        combined_context = truncate_text_to_words(combined_context, target_words)
        actual_words = count_words(combined_context)
    
    print(f"  Sampled {len(sampled_contexts)} contexts, {actual_words:,} words (target: {target_words:,}, {actual_words/target_words*100:.1f}%)")
    
    return combined_context, used_indices


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_long_memory_test_case(case_number: int, config: Dict[str, Any], all_contexts: List[str]) -> Dict[str, Any]:
    """
    Generate a single long memory test case.
    
    Structure:
    - Session 1: Setup queries (same as memory_only)
    - Session 2: Large context queries from dataset
    - Session 3: Probe questions (same as memory_only)
    """
    num_setup_queries = config.get('num_setup_queries_per_case', 8)
    num_probes = config.get('num_probes_per_case', 5)
    num_large_context_queries = config.get('num_large_context_queries_per_case', 4)
    num_events = config.get('num_events', 5)
    
    # Generate persona and events
    persona_summary = generate_persona()
    events = generate_events(num_events)
    
    # Pre-determine which queries will be probed (5 out of 8)
    probe_query_indices = set(random.sample(range(num_setup_queries), num_probes))
    
    # Calculate how many queries should have memory triggers (50% of total)
    num_with_triggers = round(num_setup_queries * 0.5)
    num_without_triggers = num_setup_queries - num_with_triggers
    
    # Generate setup queries (information-introducing) - SESSION 1
    queries = []
    probes = []
    previous_queries = []
    query_counter = 0
    
    # First, generate queries WITH memory triggers
    for i in range(num_with_triggers):
        query, information, extracted_facts = generate_information_query(
            persona_summary,
            events,
            previous_queries,
            config,
            use_memory_trigger=True
        )
        
        query_dict = {
            'query': query,
            'information': information,
            'extracted_facts': extracted_facts,
            'query_number': query_counter + 1,
            'has_memory_trigger': True
        }
        queries.append(query_dict)
        previous_queries.append(query)
        
        # If this query will be probed, generate probe immediately using the exact same information
        if query_counter in probe_query_indices:
            probe = generate_probe_from_query(information, extracted_facts)
            probe['source_query_number'] = query_dict['query_number']
            probes.append(probe)
        
        query_counter += 1
    
    # Then, generate queries WITHOUT memory triggers
    for i in range(num_without_triggers):
        query, information, extracted_facts = generate_information_query(
            persona_summary,
            events,
            previous_queries,
            config,
            use_memory_trigger=False
        )
        
        query_dict = {
            'query': query,
            'information': information,
            'extracted_facts': extracted_facts,
            'query_number': query_counter + 1,
            'has_memory_trigger': False
        }
        queries.append(query_dict)
        previous_queries.append(query)
        
        # If this query will be probed, generate probe immediately using the exact same information
        if query_counter in probe_query_indices:
            probe = generate_probe_from_query(information, extracted_facts)
            probe['source_query_number'] = query_dict['query_number']
            probes.append(probe)
        
        query_counter += 1
    
    # Shuffle queries to avoid pattern detection
    random.shuffle(queries)
    
    # Re-number queries after shuffling and update probe source_query_numbers
    query_number_mapping = {}
    for i, query_dict in enumerate(queries, 1):
        old_number = query_dict['query_number']
        query_dict['query_number'] = i
        query_number_mapping[old_number] = i
    
    # Update probe source_query_numbers to match new query numbers
    for probe in probes:
        old_source = probe['source_query_number']
        probe['source_query_number'] = query_number_mapping[old_source]
    
    # Sort probes by source_query_number to maintain order
    probes.sort(key=lambda p: p['source_query_number'])
    
    # Number the probes
    for i, probe in enumerate(probes, 1):
        probe['probe_number'] = i
    
    # Generate large context queries for SESSION 2
    dataset_config = config.get('dataset', {})
    target_words_total = dataset_config.get('target_words_per_query', 100000)
    
    # Calculate prefix words to account for them in context sampling
    prefix = "Here is some information I'd like you to review:\n\n"
    prefix_words = count_words(prefix)
    
    # Target for context text should account for prefix
    target_words_context = target_words_total - prefix_words
    
    print(f"\nGenerating {num_large_context_queries} large context queries for session 2...")
    print(f"  Total target: {target_words_total:,} words (prefix: {prefix_words:,}, context: {target_words_context:,})")
    large_context_queries = []
    used_context_indices = set()
    
    for i in range(num_large_context_queries):
        print(f"  Query {i+1}/{num_large_context_queries}:")
        context_text, used_context_indices = sample_contexts_for_query(
            all_contexts,
            target_words_context,  # Use adjusted target that accounts for prefix
            used_context_indices,
            config
        )
        
        # Create a user message that presents this context
        user_message = f"{prefix}{context_text}"
        total_words = count_words(user_message)
        
        # Final safety check: truncate if somehow we still exceed target
        if total_words > target_words_total:
            # Truncate context to fit
            available_for_context = max(0, target_words_total - prefix_words)
            context_text = truncate_text_to_words(context_text, available_for_context)
            user_message = f"{prefix}{context_text}"
            total_words = count_words(user_message)
        
        large_context_queries.append({
            'query': user_message,
            'query_number': i + 1,
            'words': total_words
        })
    
    return {
        'case_number': case_number,
        'persona': persona_summary,
        'events': events,
        'queries': queries,  # Session 1 setup queries
        'large_context_queries': large_context_queries,  # Session 2 large context queries
        'probes': probes  # Session 3 probe questions
    }


def convert_test_case_to_format(
    test_case_data: Dict[str, Any],
    config: Dict[str, Any],
    case_number: int
) -> Dict[str, Any]:
    """
    Convert a single test case to the test case format.
    
    Structure:
    - Session 1: Setup steps - Information-introducing queries (no success checks)
    - Session break: start_new_session
    - Session 2: Large context queries (no success checks)
    - Session break: start_new_session
    - Session 3: Probe steps - Memory validation queries (with success checks)
    """
    queries = test_case_data['queries']
    large_context_queries = test_case_data['large_context_queries']
    probes = test_case_data['probes']
    
    # Sort probes by probe_number to maintain order
    sorted_probes = sorted(probes, key=lambda p: p.get('probe_number', 0))
    
    # Build steps
    steps = []
    step_number = 1
    
    # SESSION 1: Add setup steps (information-introducing queries) - NO success checks
    print(f"\nSession 1: Adding {len(queries)} setup queries...")
    for query in queries:
        steps.append({
            "step_number": step_number,
            "user_message": query['query'],
            "expected_behavior": f"Agent should acknowledge and remember: {query['information']}",
            # No success_check - these are setup steps
        })
        step_number += 1
    
    # Session break between session 1 and session 2
    steps.append({
        "step_number": step_number,
        "step_type": "start_new_session",
        "description": "Start new session to add large context - memory was built in previous session"
    })
    step_number += 1
    
    # SESSION 2: Add large context queries - NO success checks
    print(f"Session 2: Adding {len(large_context_queries)} large context queries...")
    for query in large_context_queries:
        steps.append({
            "step_number": step_number,
            "user_message": query['query'],
            "expected_behavior": f"Agent should process and acknowledge the large context ({query['words']:,} words)",
            # No success_check - these are just to overload context
        })
        step_number += 1
    
    # Session break between session 2 and session 3
    steps.append({
        "step_number": step_number,
        "step_type": "start_new_session",
        "description": "Start new session to test memory recall - memory was built in session 1, context overloaded in session 2"
    })
    step_number += 1
    
    # SESSION 3: Add probe steps (memory validation) - WITH success checks
    print(f"Session 3: Adding {len(sorted_probes)} probe queries with success checks...")
    for probe in sorted_probes:
        # Use focused_information if available, otherwise fall back to target_information
        focused_info = probe.get('focused_information', probe.get('target_information', ''))
        
        steps.append({
            "step_number": step_number,
            "user_message": probe['probe_question'],
            "expected_behavior": f"Agent should recall from memory: {focused_info}",
            "success_check": create_memory_validator(
                focused_info,
                probe['expected_answer'],
                probe['probe_question']
            )
        })
        step_number += 1
    
    # Create test case
    test_case_config = config.get('test_case', {}) if isinstance(config, dict) else {}
    defaults_config = config.get('test_case_defaults', {}) if isinstance(config, dict) else {}
    
    name_prefix = test_case_config.get('name_prefix', 'long_memory')
    
    test_case = {
        "name": f"{name_prefix}_{case_number:03d}",
        "description": f"Long memory test case {case_number}. Tests memory recall with {len(probes)} probes after context overload. Session 1: {len(queries)} setup queries build memory. Session 2: {len(large_context_queries)} large context queries overload context. Session 3: {len(probes)} probe queries test memory recall.",
        "attack_type": "benign",
        "initial_data": {
            "inbox_set": defaults_config.get('default_inbox_set', 'inbox_set_1'),
            "outbox_set": defaults_config.get('default_outbox_set', 'outbox_set_1'),
            "drafts_set": defaults_config.get('default_drafts_set', 'drafts_set_1'),
            "memory_set": "0"  # Empty initial memory - memory is built during test case execution
        },
        "steps": steps
    }
    
    return test_case


def generate_all_long_memory_test_cases(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all long memory test cases."""
    num_test_cases = config.get('num_test_cases', 4)
    
    print(f"\n{'='*80}")
    print(f"GENERATING LONG MEMORY TEST CASES")
    print(f"{'='*80}")
    print(f"Number of test cases: {num_test_cases}")
    print(f"Probes per test case: {config.get('num_probes_per_case', 5)}")
    print(f"Setup queries per test case (Session 1): {config.get('num_setup_queries_per_case', 8)}")
    print(f"Large context queries per test case (Session 2): {config.get('num_large_context_queries_per_case', 4)}")
    print(f"Total success checks: {num_test_cases * config.get('num_probes_per_case', 5)}")
    
    set_openai_key()
    
    # Load dataset contexts once for all test cases
    print(f"\n{'='*80}")
    print(f"LOADING DATASET CONTEXTS")
    print(f"{'='*80}")
    all_contexts = load_dataset_contexts(config)
    
    if not all_contexts:
        print("Error: No contexts loaded from dataset")
        return []
    
    all_test_cases = []
    
    for case_num in range(1, num_test_cases + 1):
        print(f"\n{'='*80}")
        print(f"Generating Test Case {case_num}/{num_test_cases}")
        print(f"{'='*80}")
        
        try:
            # Generate queries, large context queries, and probes
            test_case_data = generate_long_memory_test_case(case_num, config, all_contexts)
            
            # Convert to test case format
            test_case = convert_test_case_to_format(test_case_data, config, case_num)
            all_test_cases.append(test_case)
            
            print(f"\n✓ Test case {case_num} generated successfully")
            print(f"  Session 1 setup queries: {len(test_case_data['queries'])}")
            print(f"  Session 2 large context queries: {len(test_case_data['large_context_queries'])}")
            print(f"  Session 3 probes (success checks): {len(test_case_data['probes'])}")
            print(f"  Total steps: {len(test_case['steps'])}")
        except Exception as e:
            print(f"\n❌ Error generating test case {case_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_test_cases


def main():
    parser = argparse.ArgumentParser(description="Generate long memory test cases")
    parser.add_argument('--config', type=str, default='src/benchmark/dataset_generation/07_long_memory/long_memory_config.yaml',
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for test cases (default: from config)')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    config = load_config(config_path)
    
    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    random.seed(seed)
    
    # Generate test cases
    test_cases = generate_all_long_memory_test_cases(config)
    
    if not test_cases:
        print("\n❌ No test cases generated")
        return 1
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.get('output', {}).get('test_cases_dir', 'data/benchmark/tests/long_memory'))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each test case
    test_case_config = config.get('test_case', {})
    filename_prefix = test_case_config.get('filename_prefix', 'long_memory')
    
    saved_files = []
    for test_case in test_cases:
        case_number = int(test_case['name'].split('_')[-1])
        test_case_filename = f"{filename_prefix}_{case_number:03d}.json"
        test_case_path = output_dir / test_case_filename
        
        with open(test_case_path, 'w', encoding='utf-8') as f:
            json.dump(test_case, f, indent=2, ensure_ascii=False)
        
        saved_files.append(test_case_path)
        
        # Count steps with success checks
        success_check_steps = sum(1 for step in test_case['steps'] if step.get('success_check'))
        total_steps = len(test_case['steps'])
        
        print(f"\n✓ Saved test case: {test_case_path}")
        print(f"  Total steps: {total_steps}")
        print(f"  Steps with success checks: {success_check_steps}")
    
    print(f"\n{'='*80}")
    print(f"✓ Generated and saved {len(test_cases)} test cases")
    print(f"{'='*80}")
    print(f"\nTest case files:")
    for path in saved_files:
        print(f"  - {path}")
    
    # Verify success check counts
    total_success_checks = 0
    for test_case in test_cases:
        success_check_count = sum(1 for step in test_case['steps'] if step.get('success_check'))
        total_success_checks += success_check_count
    
    expected_checks = len(test_cases) * config.get('num_probes_per_case', 5)
    print(f"\nTotal success checks: {total_success_checks} (expected: {expected_checks})")
    
    if total_success_checks == expected_checks:
        print("✓ All test cases have the correct number of success checks")
    else:
        print("⚠️  Warning: Some test cases do not have the expected number of success checks")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

