#!/usr/bin/env python3
"""
Generate Memory-Only Test Cases (Class 1)

This script generates and converts test cases that:
- Only test memory recall (no tool calls)
- Each test case has exactly 5 success check steps
- Total of 4 test cases = 20 success checks
- Success checks validate that memory correctly remembers information the user previously shared

The script:
1. Generates queries and probes using LoCoMo methodology
2. Converts them to test case format
3. Saves test cases directly to data/benchmark/tests/memory_only/

Usage:
    python src/benchmark/dataset_generation/01_memory_only/generate_memory_only_test_cases.py \
        --config src/benchmark/dataset_generation/01_memory_only/memory_only_config.yaml
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


# Prompt for generating information-introducing queries
INFORMATION_QUERY_PROMPT = """You are generating a user query for an AI assistant that will introduce specific information that should be remembered.

The user has the following persona:
PERSONA: {persona_summary}

Recent events in the user's life:
{events}

Previous queries in this conversation:
{previous_queries}

Generate a natural user query that:
1. **Introduces NEW, SPECIFIC information** that should be remembered:
   - Specific names of people (e.g., "Dr. Sarah Johnson", "my friend Alice", "my manager Bob")
   - Specific dates, times, or deadlines (use clear, unambiguous dates like "Monday, August 16th at 2:00 PM")
   - Specific preferences, allergies, or important details
   - Specific locations, addresses, or places
   - Specific numbers, amounts, or quantities
   - Specific plans, goals, or intentions

2. **Uses memory-triggering language** - The query should naturally prompt the assistant to remember the information. Use phrases like:
   - "Please remember that..." or "I want you to remember..."
   - "Note that..." or "Keep in mind that..."
   - "Can you remember..." or "Make sure to remember..."
   - OR structure it as a clear fact-sharing statement that implies memory storage

3. **Is natural and conversational** - not just a statement of facts, but incorporate the memory-triggering language naturally

4. **Fits the user's persona and recent events**

5. **Is distinct from previous queries** - introduces different information (different people, different dates, different topics)

6. **Is appropriate for an email assistant context** - the user might mention this in relation to email tasks

7. **States information clearly and explicitly** - the information should be explicitly stated as facts, not just implied

8. **Uses unique identifiers** - if mentioning multiple people/meetings/items, use distinct names/details to avoid ambiguity

9. **Structures information as clear facts** - present the information in a way that can be easily extracted and stored as discrete facts

Examples of good queries:
- "Please remember that I have a lunch meeting with Dr. Michael Thompson on Wednesday, August 25th at 12:30 PM"
- "I want you to remember that Dr. Emily Rodriguez should be added to the project kick-off meeting on Monday, August 16th at 2:00 PM"
- "Note that I need to follow up with Mr. Patel regarding the budget proposal before our meeting on Friday at 10:00 AM"

Return ONLY the user query text, nothing else."""


# Prompt for generating memory probe questions
MEMORY_PROBE_PROMPT = """You are generating a probe question to test if an AI assistant remembers specific information that was previously shared.

The user previously shared this information:
TARGET INFORMATION: {target_information}

EXTRACTED KEY FACTS: {extracted_facts}

The user's persona:
PERSONA: {persona_summary}

Recent events:
{events}

Generate a natural, conversational question that:
1. **Asks about SPECIFIC, UNAMBIGUOUS information** that was clearly stated in the target information
2. **References specific details** - use names, dates, or other unique identifiers to avoid ambiguity
3. **Is natural** - sounds like something a user would ask in conversation
4. **Tests memory recall** - the assistant should need to remember the information to answer correctly
5. **Is appropriate for an email assistant context**
6. **Only asks about information that was EXPLICITLY STATED** - don't ask about actions that were only discussed but not completed
7. **Uses clear, specific language** - avoid vague terms like "the upcoming meeting" if multiple meetings exist; instead use specific details like "the meeting with Dr. X on Friday"

Examples of good probe questions:
- "What was the name of Dr. Sarah Johnson that you mentioned earlier?" (if Dr. Sarah Johnson was shared)
- "When did you say the meeting with Dr. Sarah Johnson was scheduled?" (if meeting time with specific person was shared)
- "What was the deadline for sending the travel itinerary?" (if specific deadline was shared)

Examples of BAD probe questions (avoid these):
- "When is the upcoming meeting?" (too vague if multiple meetings exist)
- "Who did you add to contacts?" (assumes action was completed when it was only discussed)
- "What was that thing you mentioned?" (too vague)

Return ONLY the probe question text, nothing else."""


# Prompt for generating expected answer to probe
MEMORY_PROBE_ANSWER_PROMPT = """Given the following probe question and the information that was previously shared, generate the expected answer.

PROBE QUESTION: {probe_question}

INFORMATION THAT WAS PREVIOUSLY SHARED: {target_information}

EXTRACTED KEY FACTS: {extracted_facts}

FOCUSED INFORMATION (what the question specifically asks about): {focused_information}

Generate a concise expected answer that:
1. **Directly answers the probe question** using the information that was explicitly stated
2. **Only includes information that was CLEARLY STATED** in the original information - don't infer or assume actions were completed
3. **Only includes information relevant to what the question asks** - don't include extra details
4. **Is natural and conversational**
5. **Contains the key details that must be present** for the answer to be correct - include specific names, dates, times as they were stated
6. **Matches what was actually communicated** - if the original query only discussed something (didn't complete it), the answer should reflect that
7. **Uses the exact facts from extracted_facts** - reference the structured facts to ensure accuracy

IMPORTANT: 
- The expected answer should reflect what was ACTUALLY STATED in the original information, not what might have been implied or assumed
- Include specific details (names, dates, times) as they were stated in the original query
- The answer should be something that can be verified against the extracted facts

Return ONLY the expected answer text, nothing else."""


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_persona() -> str:
    """Generate a simple persona for the user."""
    personas = [
        "A busy professional who manages multiple email accounts and schedules",
        "A small business owner who handles customer communications",
        "A project manager coordinating team communications",
        "A consultant managing client relationships via email",
        "An executive assistant organizing schedules and communications"
    ]
    return random.choice(personas)


def generate_events(num_events: int) -> str:
    """Generate simple events for context."""
    event_templates = [
        "Recently attended a conference",
        "Started a new project",
        "Had a meeting with stakeholders",
        "Received important emails",
        "Scheduled upcoming appointments",
        "Made travel plans",
        "Coordinated with team members",
        "Handled customer inquiries"
    ]
    selected = random.sample(event_templates, min(num_events, len(event_templates)))
    return "\n".join(f"- {event}" for event in selected)


def extract_key_facts(query: str) -> str:
    """
    Extract key facts from a query using LLM.
    Returns structured information about what was stated.
    This extraction is used to help generate better probe questions and expected answers.
    """
    extraction_prompt = f"""Extract the key facts and information that were explicitly stated in this user query.

USER QUERY: {query}

Extract:
1. **Names of people** mentioned (with titles/roles if given) - include full names and titles
2. **Specific dates, times, or deadlines** mentioned - include complete date/time information
3. **Specific actions or tasks** mentioned (but note if they were only discussed vs. completed)
4. **Specific details** like locations, amounts, preferences, etc.
5. **What was actually stated** vs. what was only discussed/requested
6. **Relationships between facts** - e.g., "Person X is scheduled for Meeting Y on Date Z"

Format as a clear, structured list of facts. Only include information that was EXPLICITLY STATED in the query.
Structure each fact clearly so it can be easily stored and retrieved by different memory systems.

Example format:
- Person: [Name with title]
- Date/Time: [Complete date and time]
- Event/Meeting: [Description]
- Relationship: [How facts relate to each other]

Return ONLY the extracted facts in a clear, structured format, nothing else."""
    
    extracted = run_chatgpt(extraction_prompt, temperature=0.3)
    return extracted.strip()


def generate_information_query(
    persona_summary: str,
    events: str,
    previous_queries: List[str],
    config: Dict[str, Any]
) -> Tuple[str, str, str]:
    """
    Generate an information-introducing query and extract the information.
    
    Returns:
        (query, information, extracted_facts) tuple
    """
    previous_queries_text = "\n".join([f"- {q}" for q in previous_queries]) if previous_queries else "None"
    
    prompt = INFORMATION_QUERY_PROMPT.format(
        persona_summary=persona_summary,
        events=events,
        previous_queries=previous_queries_text
    )
    
    query = run_chatgpt(prompt, temperature=0.8)
    query = query.strip()
    
    # Extract structured information from the query
    extracted_facts = extract_key_facts(query)
    
    # Use the query itself as the information (for backward compatibility)
    information = query
    
    return query, information, extracted_facts


def generate_memory_probe(
    target_information: str,
    extracted_facts: str,
    persona_summary: str,
    events: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a memory probe question and expected answer.
    
    Returns:
        Dictionary with probe_question, expected_answer, target_information, focused_information
    """
    # First, generate the probe question
    probe_prompt = MEMORY_PROBE_PROMPT.format(
        target_information=target_information,
        extracted_facts=extracted_facts,
        persona_summary=persona_summary,
        events=events
    )
    
    probe_question = run_chatgpt(probe_prompt, temperature=0.7)
    probe_question = probe_question.strip()
    
    # Extract what specific information the probe question is asking about
    focused_info_prompt = f"""Given this probe question: "{probe_question}"

And the extracted facts from the original information:
{extracted_facts}

What specific piece of information is this question asking about? Extract just the key detail(s) that the question is testing. Be specific and unambiguous.

Return ONLY the specific information being asked about, nothing else."""
    
    focused_information = run_chatgpt(focused_info_prompt, temperature=0.3)
    focused_information = focused_information.strip()
    
    # Generate expected answer
    answer_prompt = MEMORY_PROBE_ANSWER_PROMPT.format(
        probe_question=probe_question,
        target_information=target_information,
        extracted_facts=extracted_facts,
        focused_information=focused_information
    )
    
    expected_answer = run_chatgpt(answer_prompt, temperature=0.3)
    expected_answer = expected_answer.strip()
    
    return {
        'probe_question': probe_question,
        'expected_answer': expected_answer,
        'target_information': target_information,
        'focused_information': focused_information
    }


def generate_memory_only_test_case(case_number: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a single memory-only test case."""
    num_setup_queries = config.get('num_setup_queries_per_case', 8)
    num_probes = config.get('num_probes_per_case', 5)
    num_events = config.get('num_events', 5)
    
    # Generate persona and events
    persona_summary = generate_persona()
    events = generate_events(num_events)
    
    # Generate setup queries (information-introducing)
    queries = []
    previous_queries = []
    
    for i in range(num_setup_queries):
        query, information, extracted_facts = generate_information_query(
            persona_summary,
            events,
            previous_queries,
            config
        )
        queries.append({
            'query': query,
            'information': information,
            'extracted_facts': extracted_facts,
            'query_number': i + 1
        })
        previous_queries.append(query)
    
    # Generate probes (memory validation)
    probes = []
    for i in range(num_probes):
        # Select a random query to probe about
        target_query = random.choice(queries)
        target_information = target_query['information']
        extracted_facts = target_query.get('extracted_facts', target_information)
        
        probe = generate_memory_probe(
            target_information,
            extracted_facts,
            persona_summary,
            events,
            config
        )
        probe['probe_number'] = i + 1
        probe['source_query_number'] = target_query['query_number']
        probes.append(probe)
    
    return {
        'case_number': case_number,
        'persona': persona_summary,
        'events': events,
        'queries': queries,
        'probes': probes
    }


def create_memory_validator(focused_information: str, expected_answer: str, probe_question: str) -> Dict[str, Any]:
    """
    Create a validator for memory recall.
    
    Uses semantic_judge to check if the agent's response contains the expected information.
    The validator only checks for what the question actually asks about (focused_information),
    not all the information that was introduced.
    """
    # Create a more precise validator query that focuses on what was actually stated
    validator_query = f"""The user asked: '{probe_question}'. 

The agent's response must demonstrate that it correctly remembers and answers this question based on information that was previously shared. 

The response should contain the following information: {expected_answer}

The key details that must be present are: {focused_information}

IMPORTANT: The agent's response should show that it remembers the specific information that was explicitly stated. The response does not need to be a direct quote, but it should demonstrate clear recall of the key details. If the agent asks clarifying questions, that is acceptable as long as it also demonstrates that it remembers the relevant information."""
    
    return {
        "type": "semantic_judge",
        "query": validator_query,
        "check_target": "agent_response"
    }


def convert_test_case_to_format(
    test_case_data: Dict[str, Any],
    config: Dict[str, Any],
    case_number: int
) -> Dict[str, Any]:
    """
    Convert a single test case to the test case format.
    
    Structure:
    - Setup steps: Information-introducing queries (no success checks)
    - Session break: start_new_session
    - Probe steps: Memory validation queries (with success checks)
    """
    queries = test_case_data['queries']
    probes = test_case_data['probes']
    
    # Sort probes by probe_number to maintain order
    sorted_probes = sorted(probes, key=lambda p: p.get('probe_number', 0))
    
    # Build steps
    steps = []
    step_number = 1
    
    # Add setup steps (information-introducing queries) - NO success checks
    # These queries run in the test case to build memory naturally
    for query in queries:
        steps.append({
            "step_number": step_number,
            "user_message": query['query'],
            "expected_behavior": f"Agent should acknowledge and remember: {query['information']}",
            # No success_check - these are setup steps
        })
        step_number += 1
    
    # Add session break between setup queries and probe queries
    # This ensures we test memory retrieval in a new session, not just conversation context
    steps.append({
        "step_number": step_number,
        "step_type": "start_new_session",
        "description": "Start new session to test memory recall - memory was built in previous session"
    })
    step_number += 1
    
    # Add probe steps (memory validation) - WITH success checks
    # These queries test if the agent can recall information from memory built in setup steps
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
    
    name_prefix = test_case_config.get('name_prefix', 'memory_only')
    
    test_case = {
        "name": f"{name_prefix}_{case_number:03d}",
        "description": f"Memory-only test case {case_number}. Tests memory recall with {len(probes)} probes. The {len(queries)} setup queries run in the test case to build memory, then a session break, then {len(probes)} probe queries test memory recall.",
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


def generate_all_memory_only_test_cases(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all memory-only test cases."""
    num_test_cases = config.get('num_test_cases', 4)
    
    print(f"\n{'='*80}")
    print(f"GENERATING MEMORY-ONLY TEST CASES")
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
            test_case_data = generate_memory_only_test_case(case_num, config)
            
            # Convert to test case format
            test_case = convert_test_case_to_format(test_case_data, config, case_num)
            all_test_cases.append(test_case)
            
            print(f"\n✓ Test case {case_num} generated successfully")
            print(f"  Setup queries: {len(test_case_data['queries'])}")
            print(f"  Probes (success checks): {len(test_case_data['probes'])}")
            print(f"  Total steps: {len(test_case['steps'])}")
        except Exception as e:
            print(f"\n❌ Error generating test case {case_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_test_cases


def main():
    parser = argparse.ArgumentParser(description="Generate memory-only test cases")
    parser.add_argument('--config', type=str, default='src/benchmark/dataset_generation/01_memory_only/memory_only_config.yaml',
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
    test_cases = generate_all_memory_only_test_cases(config)
    
    if not test_cases:
        print("\n❌ No test cases generated")
        return 1
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.get('output', {}).get('test_cases_dir', 'data/benchmark/tests/memory_only'))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each test case
    test_case_config = config.get('test_case', {})
    filename_prefix = test_case_config.get('filename_prefix', 'memory_only')
    
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
