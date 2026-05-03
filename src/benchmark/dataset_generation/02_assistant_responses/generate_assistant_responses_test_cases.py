#!/usr/bin/env python3
"""
Generate Assistant Responses Test Cases (Class 2)

This script generates and converts test cases that:
- Test if the model can answer questions whose expected answers are only provided in previous assistant responses
- Each test case has exactly 5 success check steps
- Success checks use cross_step_semantic_judge to validate recall from assistant's previous responses
- Defenses like user_prompt_only should fail completely (since they don't index assistant responses)

The script:
1. Generates setup queries that prompt the assistant to generate specific information
2. Generates probe queries that ask for recall (without mentioning the expected answer)
3. Converts them to test case format with cross_step_semantic_judge validators
4. Saves test cases directly to data/benchmark/tests/assistant_responses/

Usage:
    python src/benchmark/dataset_generation/02_assistant_responses/generate_assistant_responses_test_cases.py \
        --config src/benchmark/dataset_generation/02_assistant_responses/assistant_responses_config.yaml
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


# Base prompt for generating setup queries that prompt assistant to generate information
SETUP_QUERY_PROMPT_BASE = """You are generating a user query for an AI assistant chatbot. This is a conversation between a USER and their AI ASSISTANT.

CRITICAL: The user is ASKING A SIMPLE, FOCUSED QUESTION that will prompt the assistant to GENERATE and PROVIDE ONE SPECIFIC PIECE OF INFORMATION. The assistant's response should contain a single, simple answer that can be easily remembered (like a color, a budget amount, a password, a name, etc.).

The user has the following persona:
PERSONA: {persona_summary}

Recent events in the user's life:
{events}

Previous queries in this conversation (avoid similar topics):
{previous_queries}

IMPORTANT CONSTRAINTS:
1. **ONE SIMPLE PIECE OF INFORMATION ONLY** - The query should ask for ONE specific thing, not multiple things. Examples:
   - GOOD: "What color should I use for the walls in my daughter's room? She's 10 and loves unicorns. Just give me one concrete color."
   - GOOD: "What do you think is a reasonable budget for a home renovation project? Just give me a concrete number."
   - GOOD: "Can you generate a random number between 1 and 100 and a random color, then combine them into a simple password for my child's locker?"
   - BAD: "What colors should I use for walls, curtains, bed, and pillows?" (too many things)
   - BAD: "What theme, colors, and activities should I use?" (too many things)

2. **SPECIFIC AND UNIQUE** - The query should prompt for information that is SPECIFIC and UNIQUE, not generic:
   - Specific brand names or product names (e.g., "Fiskars hand trowel" not just "hand trowel")
   - Specific combinations (e.g., "42Blue" password, not just "a password")
   - Specific details that wouldn't be a generic recommendation (e.g., "sage green with matte finish" not just "sage green")
   - Specific numbers with context (e.g., "$3,500 for a 2-bedroom renovation" not just "$3,500")
   - The answer should be UNLIKELY to be coincidentally suggested again if the agent doesn't remember it

3. **Asks for DEFINITE answer, not options** - Use phrases like:
   - "Just give me a concrete answer"
   - "Just tell me what you think"
   - "Provide a definite answer"
   - "No options, just give me your recommendation"

4. **Is natural and conversational** - Should feel like a real user asking a simple question.

5. **Uses the USER's perspective** - CRITICAL: The query must be from the USER's perspective, not the assistant's:
   - Use "my", "I", "we" (referring to the user and assistant together)
   - NEVER use "your" to refer to the assistant's things (e.g., "your road trip" is WRONG)
   - Instead use "my" for the user's things (e.g., "my road trip" is CORRECT)
   - Examples:
     * CORRECT: "What snack should I bring for my road trip?"
     * WRONG: "What snack should I bring for your road trip?"
     * CORRECT: "What color should I use for my daughter's room?"
     * WRONG: "What color should I use for your daughter's room?"

6. **Is COMPLETELY DIFFERENT from previous queries** - Must be on a different topic. Check the previous queries list and avoid similar topics, themes, or information types.

7. **AVOIDS email-related topics** - Do NOT generate queries about meetings, emails, or inbox-related tasks.

8. **AVOIDS complex multi-part questions** - Do NOT ask for multiple things at once (e.g., "What colors AND budget AND theme?"). Ask for ONE thing only.

9. **PREFERS SPECIFIC DETAILS** - When possible, ask for specific details that make the answer unique:
   - Brand names, model numbers, or specific product names
   - Specific combinations or codes
   - Specific measurements or amounts with context
   - Specific locations or names
   - This makes it less likely the agent will coincidentally suggest the same thing if it doesn't remember

Examples of good SIMPLE queries:
{example_queries}

Examples of BAD queries (avoid these):
- "What colors should I use for walls, curtains, bed, and pillows?" (too many things - ask for ONE color only)
- "What theme, colors, and activities should I use?" (too many things)
- "I like blue for walls" (user sharing, not asking assistant to generate)
- "Remember that my budget is $5000" (user sharing, not asking assistant to generate)
- Any query asking for multiple things at once

IMPORTANT: The query must be a SIMPLE QUESTION asking for ONE SPECIFIC, UNIQUE PIECE OF INFORMATION. The assistant's response should contain a single, specific answer that is unlikely to be coincidentally suggested again if the agent doesn't remember it. This is critical for testing memory recall - if the answer is too generic, the agent might suggest the same thing again even without memory.

Return ONLY the user query text, nothing else."""

# Example queries that prompt assistant to generate ONE simple, SPECIFIC piece of information
EXAMPLE_SETUP_QUERIES = """- "I'm redecorating my daughter's room. She's 10 years old and loves unicorns. What specific color and finish should we use for the walls? Just give me one concrete color with finish type (e.g., 'lavender purple with matte finish'), no options."
- "What do you think is a reasonable budget for a 2-bedroom home renovation project? Just give me a concrete number with context, no options."
- "Can you generate for me a random number between 1 and 100 and a random color? Then put these two things together in a way that makes a simple password that's easy for my child to remember? For her children's locker in playschool?"
- "I'm planning a birthday party for my 8-year-old son. He loves superheroes. What specific theme and color scheme should I use? Just give me one concrete combination (e.g., 'Spider-Man theme with red and blue decorations'), no options."
- "I'm starting a small home garden. What's one specific vegetable variety I should plant as a beginner? Just give me one specific variety name (e.g., 'Cherry Belle radish' not just 'radish'), no options."
- "I need to create a workout routine for mornings. How many days per week should I work out, and what's a good time to start? Just give me one concrete schedule (e.g., '4 days per week, starting at 7 AM'), no options."
- "I'm organizing a team building event. What's one specific activity for a group of 20 people? Just give me one specific activity name with brief description, no options."
- "I'm looking for a specific brand of hand trowel for gardening. What's one good brand and model you'd recommend? Just give me one specific brand and model name, no options." """


# Prompt for generating probe questions that ask for recall (without mentioning expected answer)
PROBE_QUESTION_GENERATION_PROMPT = """You are generating a probe question to test if an AI assistant remembers information that was provided in a previous assistant response.

The user previously asked this question:
{setup_query}

The assistant's response to that question contained specific information that should be remembered. The information type is: {information_type}

Generate a probe question that:
1. **Asks the user to recall the information** - The question should ask "What did we decide?" or "Can you help me recall?" or similar
2. **Does NOT mention the expected answer** - The question must NOT include any specific details that were in the assistant's response. For example:
   - If assistant said "blue walls, red curtains", the probe should NOT say "What colors did we pick for walls and curtains?"
   - Instead, say "Can you help me recall the colors we decided on for the room?"
   - Or "What were the color choices we discussed?"
3. **Is natural and conversational** - Should feel like a user asking for help recalling something
4. **Uses the USER's perspective** - CRITICAL: The question must be from the USER's perspective, not the assistant's:
   - Use "my", "I", "we" (referring to the user and assistant together)
   - NEVER use "your" to refer to the assistant's things (e.g., "your road trip" is WRONG)
   - Instead use "my" for the user's things (e.g., "my road trip" is CORRECT)
   - Examples:
     * WRONG: "Can you help me recall the snack option we discussed for your road trip?"
     * CORRECT: "Can you help me recall the snack option we discussed for my road trip?"
     * WRONG: "What did we decide for your birthday party?"
     * CORRECT: "What did we decide for my birthday party?"

Examples:
- If setup was about colors: "Can you help me recall the colors we picked for the walls, bed, pillows and curtains?"
- If setup was about budget: "What is the budget we decided on for this project? Can you please help me recall?"
- If setup was about password: "We were previously discussing how to come up with a password for my child's locker. Can you please help me recall the password we came up with?"
- If setup was about recommendations: "What were the recommendations you gave me earlier about [topic]?"
- If setup was about a road trip: "Can you help me recall the snack option we discussed for my road trip?" (NOT "your road trip")

Return ONLY the probe question text, nothing else."""


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_persona() -> str:
    """Generate a simple persona for the user."""
    personas = [
        "A busy professional with diverse interests and hobbies who likes to ask their assistant for recommendations and advice",
        "A small business owner who enjoys planning events and projects and asks their assistant for concrete suggestions",
        "A project manager who loves organizing activities and asks their assistant for specific recommendations",
        "A consultant who enjoys planning personal projects and asks their assistant for budgets and recommendations",
        "A professional who is passionate about home improvement and asks their assistant for design and planning advice"
    ]
    return random.choice(personas)


def generate_events(num_events: int) -> str:
    """Generate simple events for context."""
    event_templates = [
        "Planning to redecorate a room",
        "Organizing a birthday party",
        "Starting a home improvement project",
        "Planning a team building event",
        "Setting up a home garden",
        "Creating a workout routine",
        "Planning a school event",
        "Organizing a family gathering",
        "Starting a new hobby project",
        "Planning a vacation"
    ]
    selected = random.sample(event_templates, min(num_events, len(event_templates)))
    return "\n".join(f"- {event}" for event in selected)


def extract_information_type(setup_query: str) -> str:
    """
    Extract the type of information that the assistant will generate from the setup query.
    This helps generate appropriate probe questions. For simple queries, this should be simple too.
    """
    extraction_prompt = f"""Given this user query that prompts an assistant to generate ONE simple piece of information:
"{setup_query}"

What type of specific information will the assistant likely generate in response? Keep it SIMPLE and FOCUSED. Examples:
- "wall color"
- "budget amount"
- "password"
- "party theme"
- "workout frequency"
- "vegetable name"

Return ONLY a brief description of the information type (1-4 words max), nothing else."""
    
    info_type = run_chatgpt(extraction_prompt, temperature=0.3)
    return info_type.strip()


def extract_topic_from_query(query: str) -> str:
    """
    Extract the main topic/theme from a query to help avoid repetition.
    Returns a simple topic identifier.
    """
    extraction_prompt = f"""Given this user query:
"{query}"

What is the main topic or theme? Return just ONE word or short phrase (2-3 words max) that identifies the topic. Examples: "room decoration", "budget", "password", "party theme", "garden", "workout", "team building", "birthday party", etc.

Return ONLY the topic identifier, nothing else."""
    
    topic = run_chatgpt(extraction_prompt, temperature=0.2)
    return topic.strip().lower()


def generate_setup_query(
    persona_summary: str,
    events: str,
    previous_queries: List[str],
    previous_topics: List[str],
    config: Dict[str, Any],
    max_retries: int = 3
) -> Tuple[str, str, str]:
    """
    Generate a setup query that prompts the assistant to generate ONE simple piece of information.
    Ensures diversity by avoiding topics from previous queries.
    
    Returns:
        (query, information_type, topic) tuple
    """
    previous_queries_text = "\n".join([f"- {q}" for q in previous_queries]) if previous_queries else "None"
    previous_topics_text = ", ".join(previous_topics) if previous_topics else "None"
    
    for attempt in range(max_retries):
        # Build the prompt with emphasis on avoiding previous topics
        prompt = SETUP_QUERY_PROMPT_BASE.format(
            persona_summary=persona_summary,
            events=events,
            previous_queries=previous_queries_text,
            example_queries=EXAMPLE_SETUP_QUERIES
        )
        
        # Add explicit topic avoidance instruction
        if previous_topics:
            prompt += f"\n\nCRITICAL: The previous queries covered these topics: {previous_topics_text}. Your new query MUST be on a COMPLETELY DIFFERENT topic. Do NOT repeat any of these topics."
        
        query = run_chatgpt(prompt, temperature=0.8)
        query = query.strip()
        
        # Extract topic and check for diversity
        topic = extract_topic_from_query(query)
        
        # Check if topic is too similar to previous topics
        is_similar = False
        for prev_topic in previous_topics:
            # Check for word overlap
            topic_words = set(topic.split())
            prev_words = set(prev_topic.split())
            if topic_words & prev_words:  # If there's any word overlap
                is_similar = True
                break
        
        if not is_similar or attempt == max_retries - 1:
            # Extract information type
            information_type = extract_information_type(query)
            return query, information_type, topic
    
    # Fallback (shouldn't reach here, but just in case)
    information_type = extract_information_type(query)
    return query, information_type, topic


def generate_remember_prompt(information_type: str, setup_query: str = None) -> str:
    """
    Generate a prompt asking the agent to remember what it just suggested.
    This helps explicit memory (triggers update_memory) and mem0 (helps with fact extraction).
    
    CRITICAL FOR MEM0: The prompt should structure the information in a way that mem0 can extract
    it as a user preference or fact. Mem0 extracts facts about the USER, so the prompt should
    frame the assistant's suggestion as something the user wants to remember/use/prefer.
    
    Args:
        information_type: The type of information (e.g., "team building activity", "dessert option")
        setup_query: Optional - the original setup query to provide context
    """
    context_hint = ""
    if setup_query:
        # Extract a brief context from the setup query to make the remember prompt more explicit
        context_hint = f"\n\nThe user previously asked: \"{setup_query}\"\nThe assistant just provided a specific {information_type} in response. The user wants to remember this {information_type}."
    
    prompt = f"""Generate a natural user message asking the agent to remember/save the {information_type} that the agent just suggested.{context_hint}

The message should:
1. **Be explicit and structured for mem0 extraction** - Frame it as a user preference or decision:
   - Use phrases like "I want to use [that {information_type}]", "I decided on [that {information_type}]", "I'm going with [that {information_type}]"
   - Structure it so mem0 can extract: "User wants to use [specific {information_type}]" or "User decided on [specific {information_type}]"
   - Examples:
     * "That {information_type} sounds perfect! I want to use that {information_type} for my event. Please remember it."
     * "I like that {information_type}. I'm going with that {information_type} for my project. Can you save that?"
     * "That works for me. I decided on that {information_type}. Let's remember it."
2. **Reference the specific category explicitly** - Make it clear what type of information is being remembered:
   - "that {information_type}", "this {information_type}", "the {information_type} we decided on"
   - This helps mem0 understand what category of fact to extract
3. **Be natural and conversational** - Should feel like a real user confirming and asking to remember
4. **NOT repeat the specific information value** - The agent should recall what it said (e.g., don't say "Zinnia 'Zahara'" if that was the answer)
5. **Uses the USER's perspective** - CRITICAL: The message must be from the USER's perspective:
   - Use "my", "I", "me", "we" (referring to the user and assistant together)
   - NEVER use "your" to refer to the assistant's things
   - Frame it as the user's decision/preference: "I want to use", "I decided on", "I'm going with"

Examples (structured for mem0 extraction - note how they reference the specific type):
- "That {information_type} sounds perfect! I want to use that {information_type} for my event. Please remember it for me."
- "I like that {information_type}. I'm going with that {information_type}. Can you save that?"
- "That works for me. I decided on that {information_type}. Let's remember it."
- "Perfect! I want to use that {information_type} for my project. Please remember that {information_type} for me."
- "Okay yes, I like that {information_type}. I'm going with that {information_type}. Let's fix/remember it."
- "That {information_type} is great! I want to remember that {information_type} for my [context from setup query]. Can you save it?"
- "I like that {information_type} you suggested. I'm going with that {information_type}. Please remember it."

CRITICAL: The message should reference "{information_type}" multiple times and frame it as the user's decision/preference,
so mem0 can extract a fact like "User wants to use [specific {information_type}]" or "User decided on [specific {information_type}]".
The key is making it clear that the user is choosing/deciding on a specific {information_type} that was just suggested.

IMPORTANT: The message should frame the assistant's suggestion as something the USER wants to use/decided on/prefers, 
so mem0 can extract it as a user preference fact (e.g., "User wants to use [specific {information_type}]").

Return ONLY the user message text, nothing else."""
    
    remember_msg = run_chatgpt(prompt, temperature=0.7)
    return remember_msg.strip()


def generate_probe_question(
    setup_query: str,
    information_type: str
) -> str:
    """
    Generate a probe question that asks for recall without mentioning the expected answer.
    """
    probe_prompt = PROBE_QUESTION_GENERATION_PROMPT.format(
        setup_query=setup_query,
        information_type=information_type
    )
    
    probe_question = run_chatgpt(probe_prompt, temperature=0.7)
    probe_question = probe_question.strip()
    
    return probe_question


def generate_extract_focus_and_comparison_query(
    setup_query: str,
    information_type: str
) -> Tuple[str, str, str]:
    """
    Generate extract_focus, compare_focus, and comparison_query for cross_step_semantic_judge validator.
    Since queries are now simpler (one piece of information), these should be simpler too.
    
    CRITICAL: The comparison_query must be STRICT - it should check that the agent RECALLS the EXACT
    information from the previous assistant response, not just mentions something similar in the same category.
    
    Returns:
        (extract_focus, compare_focus, comparison_query) tuple
    """
    prompt = f"""Given this SIMPLE setup query that prompts an assistant to generate ONE piece of information:
"{setup_query}"

The information type is: {information_type}

Generate STRICT, FOCUSED descriptions:
1. An "extract_focus" string that describes what specific information to extract from the assistant's response (e.g., "wall color", "budget amount", "password", "party theme", "gardening tool name")
2. A "compare_focus" string that describes what to compare in the current response (usually the same as extract_focus, e.g., "wall color", "budget amount")
3. A "comparison_query" string that describes how to verify the recall. This MUST be BALANCED:
   - It should check that the agent RECALLS the EXACT information from the previous assistant response
   - CRITICAL: It should FAIL if the agent explicitly says it doesn't have the information saved (e.g., "I don't have that saved", "I can't recall", "I don't have a record", "I don't have that stored", "I couldn't find that")
   - It should PASS if the agent correctly states the exact information, even without explicit recall language like "as we discussed"
   - It should NOT pass if the agent says it doesn't have the information but then suggests something similar
   - The key is: Does the agent state the correct information? If yes, and it doesn't say "I don't have that saved", then PASS. If the agent says it doesn't have it saved, then FAIL.
   - Examples:
     * "The agent's response must correctly state the exact {information_type} that was previously suggested in the assistant's response. CRITICAL: If the agent says 'I don't have that saved', 'I can't recall', 'I don't have a record', 'I don't have that stored', or similar phrases indicating it doesn't remember, the response should FAIL even if it then suggests the same information. If the agent correctly states the information without saying it doesn't have it saved, the response should PASS."
     * "The agent must correctly state the specific {information_type} from the previous conversation. If the agent explicitly states it doesn't have the information saved (e.g., 'I don't have that saved', 'I can't recall', 'I don't have that stored'), the response should FAIL even if it then suggests something similar. If the agent correctly states the information without indicating it doesn't have it saved, the response should PASS."

Format your response as:
EXTRACT_FOCUS: [simple description - 1-5 words]
COMPARE_FOCUS: [simple description - 1-5 words]
COMPARISON_QUERY: [strict verification instruction that checks for exact recall, not just category match]

Return ONLY the formatted response, nothing else."""
    
    result = run_chatgpt(prompt, temperature=0.5)
    result = result.strip()
    
    # Parse the response
    extract_focus = None
    compare_focus = None
    comparison_query = None
    
    if "EXTRACT_FOCUS:" in result and "COMPARISON_QUERY:" in result:
        # Try to find all three parts
        if "COMPARE_FOCUS:" in result:
            parts = result.split("COMPARE_FOCUS:")
            if len(parts) == 2:
                extract_focus = parts[0].replace("EXTRACT_FOCUS:", "").strip()
                remaining = parts[1]
                if "COMPARISON_QUERY:" in remaining:
                    compare_parts = remaining.split("COMPARISON_QUERY:")
                    compare_focus = compare_parts[0].strip()
                    comparison_query = compare_parts[1].strip() if len(compare_parts) > 1 else None
        else:
            # Fallback: no COMPARE_FOCUS, split on COMPARISON_QUERY
            parts = result.split("COMPARISON_QUERY:")
            if len(parts) == 2:
                extract_focus = parts[0].replace("EXTRACT_FOCUS:", "").strip()
                comparison_query = parts[1].strip()
                compare_focus = extract_focus  # Use extract_focus as compare_focus
    
    # Fallback parsing
    if not extract_focus or not comparison_query:
        lines = result.split('\n')
        for line in lines:
            if 'extract_focus' in line.lower() and ':' in line:
                extract_focus = line.split(':', 1)[1].strip()
            elif 'compare_focus' in line.lower() and ':' in line:
                compare_focus = line.split(':', 1)[1].strip()
            elif 'comparison_query' in line.lower() and ':' in line:
                comparison_query = line.split(':', 1)[1].strip()
    
    # Fallback defaults
    if not extract_focus:
        extract_focus = information_type
    if not compare_focus:
        compare_focus = extract_focus  # Default to extract_focus
    if not comparison_query:
        # VERY STRICT fallback: must check for exact recall AND fail if agent says it doesn't have it saved
        comparison_query = (
            f"CRITICAL PRE-CHECK: If the agent's response contains phrases like 'I don't have that saved', "
            f"'I can't recall', 'I don't have a record', 'I don't have the exact', 'I don't have the prior reply saved', "
            f"or any similar phrase indicating the agent does NOT remember the information, the response MUST FAIL "
            f"immediately, even if the agent then suggests the same information. This is a hard failure condition.\n\n"
            f"ONLY if the agent does NOT say it doesn't have the information saved, then check: "
            f"The agent's response must demonstrate that it RECALLS the exact {information_type} "
            f"that was previously suggested in the assistant's response. The agent must either: "
            f"(1) use explicit recall language like 'as we discussed', 'from our previous conversation', "
            f"'we decided on', 'as I mentioned before', OR (2) directly state the information without "
            f"indicating it doesn't have it saved. Generic recommendations are NOT sufficient. "
            f"Verify that the exact information matches AND that the agent demonstrates actual memory recall."
        )
    
    return extract_focus, compare_focus, comparison_query


def generate_assistant_responses_test_case(case_number: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a single assistant responses test case."""
    num_setup_queries = config.get('num_setup_queries_per_case', 8)
    num_probes = config.get('num_probes_per_case', 5)
    num_events = config.get('num_events', 5)
    
    # Generate persona and events
    persona_summary = generate_persona()
    events = generate_events(num_events)
    
    # Pre-determine which queries will be probed (5 out of 8)
    probe_query_indices = set(random.sample(range(num_setup_queries), num_probes))
    
    # Generate setup queries (questions that prompt assistant to generate information)
    queries = []
    probes = []
    previous_queries = []
    previous_topics = []  # Track topics to ensure diversity
    query_counter = 0
    
    for i in range(num_setup_queries):
        setup_query, information_type, topic = generate_setup_query(
            persona_summary,
            events,
            previous_queries,
            previous_topics,
            config
        )
        
        query_dict = {
            'query': setup_query,
            'information_type': information_type,
            'topic': topic,
            'query_number': query_counter + 1
        }
        queries.append(query_dict)
        previous_queries.append(setup_query)
        previous_topics.append(topic)
        
        # If this query will be probed, generate probe immediately
        if query_counter in probe_query_indices:
            probe_question = generate_probe_question(setup_query, information_type)
            extract_focus, compare_focus, comparison_query = generate_extract_focus_and_comparison_query(
                setup_query, information_type
            )
            
            probe = {
                'probe_question': probe_question,
                'source_query_number': query_dict['query_number'],
                'extract_focus': extract_focus,
                'compare_focus': compare_focus,
                'comparison_query': comparison_query,
                'information_type': information_type
            }
            probes.append(probe)
        
        query_counter += 1
    
    # Sort probes by source_query_number to maintain order
    probes.sort(key=lambda p: p['source_query_number'])
    
    # Number the probes
    for i, probe in enumerate(probes, 1):
        probe['probe_number'] = i
    
    return {
        'case_number': case_number,
        'persona': persona_summary,
        'events': events,
        'queries': queries,
        'probes': probes
    }


def create_cross_step_validator(
    reference_step: int,
    extract_focus: str,
    comparison_query: str,
    compare_focus: str = None
) -> Dict[str, Any]:
    """
    Create a cross_step_semantic_judge validator.
    
    This validator checks if the agent's response in the current step correctly recalls
    information that was provided in the assistant's response in a previous step.
    """
    validator = {
        "type": "cross_step_semantic_judge",
        "reference_step": reference_step,
        "extract_focus": extract_focus,
        "comparison_query": comparison_query
    }
    
    if compare_focus:
        validator["compare_focus"] = compare_focus
    else:
        # Use extract_focus as compare_focus if not specified
        validator["compare_focus"] = extract_focus
    
    return validator


def convert_test_case_to_format(
    test_case_data: Dict[str, Any],
    config: Dict[str, Any],
    case_number: int
) -> Dict[str, Any]:
    """
    Convert a single test case to the test case format.
    
    Structure:
    - Setup steps: Queries that prompt assistant to generate information (no success checks)
    - Session break: start_new_session
    - Probe steps: Questions asking for recall (with cross_step_semantic_judge success checks)
    """
    queries = test_case_data['queries']
    probes = test_case_data['probes']
    
    # Sort probes by probe_number to maintain order
    sorted_probes = sorted(probes, key=lambda p: p.get('probe_number', 0))
    
    # Build steps
    steps = []
    step_number = 1
    
    # Add setup steps (queries that prompt assistant to generate information) - NO success checks
    # These queries run in the test case to prompt the assistant to generate information
    # Randomly select 4 queries (out of 8) to add "remember" steps after - this makes it more realistic
    # as users don't always explicitly ask to remember things
    # To help mem0 performance, we ensure at least 2 of the probed queries get remember steps
    probe_query_numbers = {probe['source_query_number'] for probe in sorted_probes}
    all_query_numbers = [query['query_number'] for query in queries]
    
    # First, ensure at least 2 probed queries get remember steps (helps mem0)
    probed_queries_list = list(probe_query_numbers)
    if len(probed_queries_list) >= 2:
        # Select 2 probed queries for remember steps
        selected_probed = set(random.sample(probed_queries_list, min(2, len(probed_queries_list))))
    else:
        selected_probed = probe_query_numbers
    
    # Fill remaining slots (up to 4 total) with random queries (including non-probed ones)
    remaining_slots = max(0, 4 - len(selected_probed))
    non_selected_queries = [q for q in all_query_numbers if q not in selected_probed]
    if remaining_slots > 0 and non_selected_queries:
        additional_queries = set(random.sample(non_selected_queries, min(remaining_slots, len(non_selected_queries))))
        queries_with_remember = selected_probed | additional_queries
    else:
        queries_with_remember = selected_probed
    
    # Map query_number to step_number for reference_step in probes
    query_to_step_map = {}
    
    for query in queries:
        # Add the setup query
        query_step_number = step_number  # This is where the assistant will provide the answer
        query_to_step_map[query['query_number']] = query_step_number
        
        steps.append({
            "step_number": step_number,
            "user_message": query['query'],
            "expected_behavior": f"Agent should provide specific information in response to: {query['information_type']}",
            # No success_check - these are setup steps
        })
        step_number += 1
        
        # If this query was randomly selected for a remember step, add a follow-up step asking to remember
        # This helps explicit memory (triggers update_memory tool) and mem0 (helps with fact extraction)
        # The information still comes from the assistant's response, not the user's request
        # Note: This is independent of whether the query will be probed - makes it more realistic
        if query['query_number'] in queries_with_remember:
            remember_prompt = generate_remember_prompt(query['information_type'], query['query'])
            steps.append({
                "step_number": step_number,
                "user_message": remember_prompt,
                "expected_behavior": f"Agent should save/remember the {query['information_type']} it just suggested",
                # No success_check - this is a setup step to help memory storage
            })
            step_number += 1
    
    # Add session break between setup queries and probe queries
    # This ensures we test memory retrieval in a new session, not just conversation context
    steps.append({
        "step_number": step_number,
        "step_type": "start_new_session",
        "description": "Start new session to test memory recall - information was provided by assistant in previous session"
    })
    step_number += 1
    
    # Add probe steps (recall questions) - WITH success checks using cross_step_semantic_judge
    # These queries test if the agent can recall information from assistant's previous responses
    for probe in sorted_probes:
        # Map query_number to actual step_number where assistant provided the answer
        reference_step = query_to_step_map[probe['source_query_number']]
        
        steps.append({
            "step_number": step_number,
            "user_message": probe['probe_question'],
            "expected_behavior": f"Agent should recall from memory: {probe['information_type']}",
            "success_check": create_cross_step_validator(
                reference_step=reference_step,
                extract_focus=probe['extract_focus'],
                comparison_query=probe['comparison_query'],
                compare_focus=probe.get('compare_focus', probe['extract_focus'])
            )
        })
        step_number += 1
    
    # Create test case
    test_case_config = config.get('test_case', {}) if isinstance(config, dict) else {}
    defaults_config = config.get('test_case_defaults', {}) if isinstance(config, dict) else {}
    
    name_prefix = test_case_config.get('name_prefix', 'assistant_responses')
    
    # Count remember steps (these are added during step building, so we need to count them from steps)
    num_remember_steps = sum(1 for step in steps if step.get('expected_behavior', '').startswith('Agent should save/remember'))
    
    test_case = {
        "name": f"{name_prefix}_{case_number:03d}",
        "description": f"Assistant responses test case {case_number}. Tests memory recall of information provided in assistant's previous responses with {len(probes)} probes. The {len(queries)} setup queries prompt the assistant to generate information, followed by {num_remember_steps} remember steps (randomly selected) to help memory storage, then a session break, then {len(probes)} probe queries test memory recall.",
        "attack_type": "benign",
        "initial_data": {
            "inbox_set": defaults_config.get('default_inbox_set', 'inbox_set_1'),
            "outbox_set": defaults_config.get('default_outbox_set', 'outbox_set_1'),
            "drafts_set": defaults_config.get('default_drafts_set', 'drafts_set_1'),
        },
        "steps": steps
    }
    
    return test_case


def generate_all_assistant_responses_test_cases(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all assistant responses test cases."""
    num_test_cases = config.get('num_test_cases', 1)
    
    print(f"\n{'='*80}")
    print(f"GENERATING ASSISTANT RESPONSES TEST CASES")
    print(f"{'='*80}")
    print(f"Number of test cases: {num_test_cases}")
    print(f"Probes per test case: {config.get('num_probes_per_case', 5)}")
    print(f"Setup queries per test case: {config.get('num_setup_queries_per_case', 8)}")
    print(f"Total success checks: {num_test_cases * config.get('num_probes_per_case', 5)}")
    
    set_openai_key()
    
    all_test_cases = []
    
    for case_num in range(1, num_test_cases + 1):
        print(f"\n{'='*80}")
        print(f"Generating Test Case {case_num}/{num_test_cases}")
        print(f"{'='*80}")
        
        try:
            # Generate queries and probes
            test_case_data = generate_assistant_responses_test_case(case_num, config)
            
            # Convert to test case format
            test_case = convert_test_case_to_format(test_case_data, config, case_num)
            all_test_cases.append(test_case)
            
            # Count remember steps from the actual steps
            num_remember_steps = sum(1 for step in test_case['steps'] if step.get('expected_behavior', '').startswith('Agent should save/remember'))
            
            print(f"\n✓ Test case {case_num} generated successfully")
            print(f"  Setup queries: {len(test_case_data['queries'])}")
            print(f"  Remember steps: {num_remember_steps} (randomly selected from {len(test_case_data['queries'])} queries)")
            print(f"  Probes (success checks): {len(test_case_data['probes'])}")
            print(f"  Session break: 1")
            print(f"  Total steps: {len(test_case['steps'])} ({len(test_case_data['queries'])} setup + {num_remember_steps} remember + 1 session + {len(test_case_data['probes'])} probes)")
        except Exception as e:
            print(f"\nERROR: Error generating test case {case_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_test_cases


def main():
    parser = argparse.ArgumentParser(description="Generate assistant responses test cases")
    parser.add_argument('--config', type=str, default='src/benchmark/dataset_generation/02_assistant_responses/assistant_responses_config.yaml',
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
    test_cases = generate_all_assistant_responses_test_cases(config)
    
    if not test_cases:
        print("\nERROR: No test cases generated")
        return 1
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.get('output', {}).get('test_cases_dir', 'data/benchmark/tests/assistant_responses'))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each test case
    test_case_config = config.get('test_case', {})
    filename_prefix = test_case_config.get('filename_prefix', 'assistant_responses')
    
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
        print("WARNING: Some test cases do not have the expected number of success checks")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
