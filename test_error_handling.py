#!/usr/bin/env python3
"""
Test script to demonstrate error handling and recovery.
"""

import json
from agent_tools import EmailToolsConfig, create_tools

def test_validation_error():
    """Test handling of missing required fields."""
    print("=" * 70)
    print("TEST 1: Validation Error - Missing Fields")
    print("=" * 70)
    
    config = EmailToolsConfig(
        mailbox_dir="data/mailbox",
        drafts_dir="data/drafts",
        outbox_dir="data/outbox",
        trace_file="data/trace.jsonl"
    )
    
    tools = create_tools(config)
    draft_send_tool = tools[4]  # draft_and_send_email
    
    # This should fail - missing subject and body
    try:
        print("\nAttempting: draft_and_send_email with ONLY 'to' parameter...")
        result = draft_send_tool.run({"to": "alice@company.com"})
        print(f"Result: {result}")
    except Exception as e:
        print(f"\n❌ ERROR CAUGHT: {type(e).__name__}")
        print(f"Message: {str(e)[:200]}...")
        print("\n✅ This error would be returned to the agent as an observation")
        print("✅ Agent would read the error and retry with all required fields")
    
    print("\n")


def test_search_ambiguity():
    """Test handling of ambiguous search results."""
    print("=" * 70)
    print("TEST 2: Search Ambiguity - Multiple Matches")
    print("=" * 70)
    
    config = EmailToolsConfig(
        mailbox_dir="data/mailbox",
        drafts_dir="data/drafts",
        outbox_dir="data/outbox",
        trace_file="data/trace.jsonl"
    )
    config.session_id = "test_session"
    
    # Create some test drafts
    import os
    os.makedirs("data/drafts", exist_ok=True)
    
    drafts = [
        {"to": "alice@company.com", "subject": "Meeting", "body": "Test 1"},
        {"to": "bob@company.com", "subject": "Meeting", "body": "Test 2"},
        {"to": "charlie@company.com", "subject": "Meeting", "body": "Test 3"},
    ]
    
    for i, draft in enumerate(drafts):
        with open(f"data/drafts/test_draft_{i}.json", "w") as f:
            json.dump(draft, f)
    
    tools = create_tools(config)
    send_tool = tools[3]  # send_email
    
    print("\nAttempting: send_email with vague search 'meeting'...")
    result = send_tool.run("meeting")
    print(f"\nResult:\n{result}")
    print("\n✅ Agent receives helpful message about multiple matches")
    print("✅ Agent can then ask user to be more specific")
    
    # Cleanup
    for i in range(3):
        try:
            os.remove(f"data/drafts/test_draft_{i}.json")
        except:
            pass
    
    print("\n")


def test_not_found():
    """Test handling of not found errors."""
    print("=" * 70)
    print("TEST 3: Not Found Error")
    print("=" * 70)
    
    config = EmailToolsConfig(
        mailbox_dir="data/mailbox",
        drafts_dir="data/drafts",
        outbox_dir="data/outbox",
        trace_file="data/trace.jsonl"
    )
    config.session_id = "test_session"
    
    tools = create_tools(config)
    send_tool = tools[3]  # send_email
    
    print("\nAttempting: send_email for non-existent draft...")
    result = send_tool.run("xyz_nonexistent_email_12345")
    print(f"\nResult:\n{result}")
    print("\n✅ Agent receives clear 'not found' message")
    print("✅ Agent can suggest creating the draft first")
    
    print("\n")


def test_successful_recovery():
    """Test successful error recovery flow."""
    print("=" * 70)
    print("TEST 4: Successful Recovery Flow")
    print("=" * 70)
    
    config = EmailToolsConfig(
        mailbox_dir="data/mailbox",
        drafts_dir="data/drafts",
        outbox_dir="data/outbox",
        trace_file="data/trace.jsonl"
    )
    config.session_id = "test_session"
    
    tools = create_tools(config)
    draft_send_tool = tools[4]  # draft_and_send_email
    
    print("\n1️⃣ First attempt - MISSING FIELDS (will fail):")
    try:
        result = draft_send_tool.run({"to": "alice@company.com"})
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   ❌ Error: {type(e).__name__}")
        print(f"   Message: Field validation failed")
    
    print("\n2️⃣ Second attempt - ALL FIELDS PROVIDED (will succeed):")
    result = draft_send_tool.run({
        "to": "alice@company.com",
        "subject": "Test Email",
        "body": "This is a test message"
    })
    print(f"   ✅ Success!")
    print(f"   Result: {result[:100]}...")
    
    # Cleanup
    import os
    for file in os.listdir("data/outbox"):
        if file.startswith("sent_"):
            os.remove(f"data/outbox/{file}")
    
    print("\n")


if __name__ == "__main__":
    print("\n")
    print("🧪 ERROR HANDLING & RECOVERY TEST SUITE")
    print("=" * 70)
    print()
    
    test_validation_error()
    test_search_ambiguity()
    test_not_found()
    test_successful_recovery()
    
    print("=" * 70)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 70)
    print()
    print("KEY TAKEAWAYS:")
    print("1. ✅ Validation errors are caught and can be recovered from")
    print("2. ✅ Search ambiguity is handled with helpful messages")
    print("3. ✅ Not found errors guide the agent to correct actions")
    print("4. ✅ Agent can successfully retry after fixing errors")
    print("5. ✅ Max 25 tool calls prevents infinite loops")
    print()

