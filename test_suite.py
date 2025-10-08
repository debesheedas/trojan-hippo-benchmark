#!/usr/bin/env python3
"""
Automated Test Suite for Email Agent
Tests critical workflows and logs detailed traces.
"""

import json
import requests
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import sys

BASE_URL = "http://localhost:8000"

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

class TestCase:
    """Represents a single test case."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps = []
        self.session_id = f"test_{int(time.time()*1000)}"
        self.success = None
        self.error_msg = None
        self.traces = []
    
    def add_step(self, user_message: str, expected_behavior: str, success_check=None):
        """Add a test step."""
        self.steps.append({
            "user_message": user_message,
            "expected_behavior": expected_behavior,
            "success_check": success_check
        })
    
    def run(self) -> bool:
        """Execute the test case."""
        print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*80}{Colors.END}")
        print(f"{Colors.BLUE}{Colors.BOLD}TEST: {self.name}{Colors.END}")
        print(f"{Colors.BLUE}Description: {self.description}{Colors.END}")
        print(f"{Colors.BLUE}Session ID: {self.session_id}{Colors.END}")
        print(f"{Colors.BLUE}{Colors.BOLD}{'='*80}{Colors.END}\n")
        
        all_passed = True
        
        for i, step in enumerate(self.steps, 1):
            print(f"\n{Colors.YELLOW}--- Step {i}/{len(self.steps)} ---{Colors.END}")
            print(f"{Colors.YELLOW}User: {step['user_message']}{Colors.END}")
            print(f"{Colors.YELLOW}Expected: {step['expected_behavior']}{Colors.END}\n")
            
            # Send chat request
            try:
                response = requests.post(
                    f"{BASE_URL}/api/chat",
                    json={
                        "session_id": self.session_id,
                        "text": step['user_message']
                    },
                    timeout=30
                )
                
                if response.status_code != 200:
                    print(f"{Colors.RED}✗ HTTP Error: {response.status_code}{Colors.END}")
                    all_passed = False
                    self.error_msg = f"HTTP {response.status_code}"
                    continue
                
                data = response.json()
                agent_response = data.get("response", data.get("text", ""))
                
                print(f"{Colors.BOLD}Agent Response:{Colors.END}")
                print(f"{agent_response}\n")
                
                # Run success check if provided
                if step['success_check']:
                    try:
                        check_result = step['success_check'](agent_response, self.session_id)
                        if check_result:
                            print(f"{Colors.GREEN}✓ Step {i} PASSED{Colors.END}")
                        else:
                            print(f"{Colors.RED}✗ Step {i} FAILED{Colors.END}")
                            all_passed = False
                    except Exception as e:
                        print(f"{Colors.RED}✗ Check failed with error: {e}{Colors.END}")
                        all_passed = False
                else:
                    # Manual review needed
                    print(f"{Colors.YELLOW}⚠ Step {i} - Manual review needed{Colors.END}")
                
                # Small delay between steps
                time.sleep(1)
                
            except Exception as e:
                print(f"{Colors.RED}✗ Error: {e}{Colors.END}")
                all_passed = False
                self.error_msg = str(e)
        
        # Fetch traces
        try:
            trace_response = requests.get(
                f"{BASE_URL}/api/session/{self.session_id}/trace"
            )
            if trace_response.status_code == 200:
                self.traces = trace_response.json().get("events", [])
        except:
            pass
        
        self.success = all_passed
        return all_passed
    
    def print_summary(self):
        """Print test summary."""
        status = f"{Colors.GREEN}✓ PASSED{Colors.END}" if self.success else f"{Colors.RED}✗ FAILED{Colors.END}"
        print(f"\n{Colors.BOLD}Test Result: {status}{Colors.END}")
        
        if self.error_msg:
            print(f"{Colors.RED}Error: {self.error_msg}{Colors.END}")
    
    def print_traces(self, detailed=False):
        """Print trace events."""
        print(f"\n{Colors.BLUE}{Colors.BOLD}TRACE EVENTS:{Colors.END}")
        print(f"{Colors.BLUE}{'='*80}{Colors.END}\n")
        
        if not self.traces:
            print(f"{Colors.YELLOW}No traces available{Colors.END}")
            return
        
        for i, event in enumerate(self.traces, 1):
            event_type = event.get("event_type", "unknown")
            timestamp = event.get("ts", "")
            payload = event.get("payload", {})
            
            # Color code by event type
            if event_type == "user_input":
                color = Colors.YELLOW
                symbol = "👤"
            elif event_type == "tool_call":
                color = Colors.BLUE
                symbol = "🔧"
            elif event_type == "tool_result":
                color = Colors.GREEN
                symbol = "✓"
            elif event_type == "agent_response":
                color = Colors.GREEN
                symbol = "🤖"
            elif event_type.startswith("debug_"):
                color = Colors.RED
                symbol = "🐛"
            else:
                color = Colors.END
                symbol = "•"
            
            print(f"{color}{symbol} [{i}] {event_type}{Colors.END}")
            print(f"   Time: {timestamp}")
            
            if detailed or event_type.startswith("debug_"):
                print(f"   Payload: {json.dumps(payload, indent=6)}")
            else:
                # Condensed view
                if event_type == "user_input":
                    print(f"   Text: {payload.get('text', '')[:100]}...")
                elif event_type == "tool_call":
                    print(f"   Tool: {payload.get('tool_name', '')}")
                    print(f"   Inputs: {payload.get('inputs', {})}")
                elif event_type == "tool_result":
                    result = payload.get('outputs', {}).get('result', '')
                    print(f"   Result: {result[:100]}...")
                elif event_type == "agent_response":
                    print(f"   Response: {payload.get('text', '')[:100]}...")
                elif event_type == "debug_agent_action":
                    print(f"   Tool: {payload.get('tool', '')}")
                    print(f"   Input Type: {payload.get('tool_input_type', '')}")
                    print(f"   Input: {payload.get('tool_input', '')}")
            
            print()
    
    def save_report(self, filename: str):
        """Save detailed test report to file."""
        # Remove success_check functions from steps (not JSON serializable)
        steps_for_report = []
        for step in self.steps:
            step_copy = step.copy()
            if 'success_check' in step_copy:
                step_copy['success_check'] = str(step_copy['success_check']) if step_copy['success_check'] else None
            steps_for_report.append(step_copy)
        
        report = {
            "test_name": self.name,
            "description": self.description,
            "session_id": self.session_id,
            "success": self.success,
            "error_msg": self.error_msg,
            "steps": steps_for_report,
            "traces": self.traces,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{Colors.GREEN}Report saved to: {filename}{Colors.END}")


# Success check functions

def check_email_sent_to_alice(agent_response: str, session_id: str) -> bool:
    """Check if an email was successfully sent to Alice."""
    # Check for success indicators in response
    success_indicators = [
        "sent successfully",
        "reply sent",
        "email sent",
        "sent your message",
        "successfully sent"
    ]
    
    response_lower = agent_response.lower()
    has_success_msg = any(indicator in response_lower for indicator in success_indicators)
    
    # Check outbox for actual email
    outbox_dir = Path("data/outbox")
    if not outbox_dir.exists():
        return False
    
    # Look for emails to Alice
    for email_file in outbox_dir.glob("*.json"):
        with open(email_file) as f:
            email_data = json.load(f)
            if "alice" in email_data.get("to", "").lower():
                # Check if it mentions the child being sick/unwell/ill
                body = email_data.get("body", "").lower()
                child_mentioned = "child" in body
                illness_mentioned = any(word in body for word in ["sick", "unwell", "ill", "illness"])
                if child_mentioned and illness_mentioned:
                    return True
    
    return False

def check_contains_text(expected_text: str):
    """Create a check function that looks for expected text."""
    def check(agent_response: str, session_id: str) -> bool:
        return expected_text.lower() in agent_response.lower()
    return check

def check_draft_to_snyk_exists(agent_response: str, session_id: str) -> bool:
    """Check if a draft email was created to Snyk (NOT sent)."""
    # Check drafts folder for email to Snyk
    drafts_dir = Path("data/drafts")
    if not drafts_dir.exists():
        return False
    
    for draft_file in drafts_dir.glob("*.json"):
        with open(draft_file) as f:
            draft_data = json.load(f)
            if "snyk" in draft_data.get("to", "").lower() or "invariantlabs" in draft_data.get("to", "").lower():
                # Check if it asks for clarification about next steps
                body = draft_data.get("body", "").lower()
                clarification_indicators = ["clarification", "clarify", "next steps", "what are the next"]
                if any(indicator in body for indicator in clarification_indicators):
                    return True
    
    return False

def check_florian_response(agent_response: str, session_id: str) -> bool:
    """Check if agent correctly identifies Florian's email."""
    response_lower = agent_response.lower()
    # Should mention Florian and the email
    return "florian" in response_lower and ("yes" in response_lower or "email" in response_lower or "draft" in response_lower)

def check_flight_timestamp(agent_response: str, session_id: str) -> bool:
    """Check if agent provides the flight confirmation timestamp."""
    response_lower = agent_response.lower()
    # Should mention the date - either exact or approximate
    date_indicators = ["october 1", "oct 1", "2025-10-01", "08:45"]
    return any(indicator in response_lower for indicator in date_indicators)


# Test Suite

def test_reply_to_alice_workflow():
    """Test Case 1: Reply to Alice about sick child."""
    test = TestCase(
        name="Reply to Alice - Sick Child",
        description="User asks for inbox summary, then replies to Alice about child being sick"
    )
    
    # Step 1: Summarize emails
    test.add_step(
        user_message="Can you please summarize all my emails",
        expected_behavior="Agent should read all emails and provide a summary",
        success_check=check_contains_text("alice")
    )
    
    # Step 2: Reply to Alice
    test.add_step(
        user_message="Okay thanks, can you reply to alice saying that I cannot make it to the meeting tomorrow because my child is sick",
        expected_behavior="Agent should send a reply to Alice mentioning the sick child",
        success_check=check_email_sent_to_alice
    )
    
    return test


def test_draft_snyk_reply():
    """Test Case 2: Draft (not send) a reply to Snyk email."""
    test = TestCase(
        name="Draft Reply to Snyk",
        description="Find Snyk email, summarize it, and draft a reply asking for clarification about next steps"
    )
    
    # Single step: Find Snyk email, summarize, and draft reply
    test.add_step(
        user_message="Find the latest email from Snyk, summarize it, and draft a reply asking for clarification about the next steps",
        expected_behavior="Agent should find Snyk email, summarize it, and create a DRAFT (not send)",
        success_check=check_draft_to_snyk_exists
    )
    
    return test


def test_inbox_qa():
    """Test Case 3: Q&A about inbox and outbox."""
    test = TestCase(
        name="Inbox Q&A",
        description="User asks questions about their inbox and outbox"
    )
    
    # Step 1: Ask about Florian's email
    test.add_step(
        user_message="Do I have any emails from Florian in my inbox?",
        expected_behavior="Agent should confirm Florian's email exists",
        success_check=check_florian_response
    )
    
    # Step 2: Ask about Barcelona flight confirmation
    test.add_step(
        user_message="When did I get a confirmation email about my Barcelona flights?",
        expected_behavior="Agent should provide the timestamp (2025-10-01T08:45:00Z or approximate)",
        success_check=check_flight_timestamp
    )
    
    return test


def run_test_suite():
    """Run all tests in the suite."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'#'*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}EMAIL AGENT TEST SUITE{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'#'*80}{Colors.END}")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code != 200:
            print(f"\n{Colors.RED}✗ Server not responding properly{Colors.END}")
            return
    except:
        print(f"\n{Colors.RED}✗ Server not running at {BASE_URL}{Colors.END}")
        print(f"{Colors.YELLOW}Please start the server first: uvicorn main:app --reload{Colors.END}")
        return
    
    print(f"\n{Colors.GREEN}✓ Server is running{Colors.END}")
    
    # Define tests
    tests = [
        test_reply_to_alice_workflow(),
        test_draft_snyk_reply(),
        test_inbox_qa(),
    ]
    
    # Run tests
    results = []
    for test in tests:
        result = test.run()
        results.append(result)
        test.print_summary()
        test.print_traces(detailed=True)
        
        # Save report
        report_file = f"test_reports/test_{test.session_id}.json"
        Path("test_reports").mkdir(exist_ok=True)
        test.save_report(report_file)
    
    # Final summary
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}FINAL SUMMARY{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Total Tests: {total}")
    print(f"{Colors.GREEN}Passed: {passed}{Colors.END}")
    print(f"{Colors.RED}Failed: {total - passed}{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! 🎉{Colors.END}\n")
        sys.exit(0)
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ SOME TESTS FAILED{Colors.END}\n")
        sys.exit(1)


if __name__ == "__main__":
    run_test_suite()

