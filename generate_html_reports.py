#!/usr/bin/env python3
"""
HTML Report Generator for Memory Agent Security Benchmark Test Results

This script reads JSON test result files and generates beautiful HTML reports
that can be presented to supervisors or stakeholders.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse


class TestResultHTMLGenerator:
    def __init__(self, results_dir: str, output_dir: str = "html_reports"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_test_result(self, json_file: Path) -> Dict[str, Any]:
        """Load and parse a test result JSON file."""
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_original_test_case(self, test_file_path: str) -> Optional[Dict[str, Any]]:
        """Load the original test case file to get additional context."""
        try:
            test_file = Path(test_file_path)
            if test_file.exists():
                with open(test_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load original test case {test_file_path}: {e}")
        return None
    
    def format_timestamp(self, timestamp: str) -> str:
        """Format timestamp for display."""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except:
            return timestamp
    
    def format_trace_event(self, event: Dict[str, Any], session_id: Optional[str] = None) -> str:
        """Format a single trace event for HTML display."""
        event_type = event.get('event_type', 'unknown')
        timestamp = self.format_timestamp(event.get('ts', ''))
        payload = event.get('payload', {})
        event_session_id = event.get('session_id', session_id)
        
        if event_type == 'user_input':
            session_info = f" (Session: {event_session_id})" if event_session_id else ""
            return f"""
            <div class="trace-event user-input">
                <div class="event-header">
                    <span class="event-type">👤 User Input{session_info}</span>
                    <span class="timestamp">{timestamp}</span>
                </div>
                <div class="event-content">
                    <p><strong>Message:</strong> {payload.get('text', '')}</p>
                </div>
            </div>
            """
        
        elif event_type == 'agent_response':
            session_info = f" (Session: {event_session_id})" if event_session_id else ""
            return f"""
            <div class="trace-event agent-response">
                <div class="event-header">
                    <span class="event-type">🤖 Agent Response{session_info}</span>
                    <span class="timestamp">{timestamp}</span>
                </div>
                <div class="event-content">
                    <p>{payload.get('text', '').replace(chr(10), '<br>')}</p>
                </div>
            </div>
            """
        
        elif event_type == 'tool_call':
            tool_name = payload.get('tool_name', 'unknown')
            inputs = payload.get('inputs', {})
            session_info = f" (Session: {event_session_id})" if event_session_id else ""
            return f"""
            <div class="trace-event tool-call">
                <div class="event-header">
                    <span class="event-type">🔧 Tool Call: {tool_name}{session_info}</span>
                    <span class="timestamp">{timestamp}</span>
                </div>
                <div class="event-content">
                    <p><strong>Tool:</strong> {tool_name}</p>
                    <p><strong>Inputs:</strong></p>
                    <pre>{json.dumps(inputs, indent=2)}</pre>
                </div>
            </div>
            """
        
        elif event_type == 'tool_result':
            tool_name = payload.get('tool_name', 'unknown')
            outputs = payload.get('outputs', {})
            session_info = f" (Session: {event_session_id})" if event_session_id else ""
            return f"""
            <div class="trace-event tool-result">
                <div class="event-header">
                    <span class="event-type">✅ Tool Result: {tool_name}{session_info}</span>
                    <span class="timestamp">{timestamp}</span>
                </div>
                <div class="event-content">
                    <p><strong>Tool:</strong> {tool_name}</p>
                    <p><strong>Outputs:</strong></p>
                    <pre>{json.dumps(outputs, indent=2)}</pre>
                </div>
            </div>
            """
        
        elif event_type == 'session_change':
            old_session = payload.get('old_session_id', 'unknown')
            new_session = payload.get('new_session_id', 'unknown')
            reason = payload.get('reason', 'unknown')
            return f"""
            <div class="trace-event session-change">
                <div class="event-header">
                    <span class="event-type">🔄 Session Change</span>
                    <span class="timestamp">{timestamp}</span>
                </div>
                <div class="event-content">
                    <p><strong>From Session:</strong> {old_session}</p>
                    <p><strong>To Session:</strong> {new_session}</p>
                    <p><strong>Reason:</strong> {reason}</p>
                </div>
            </div>
            """
        
        else:
            session_info = f" (Session: {event_session_id})" if event_session_id else ""
            return f"""
            <div class="trace-event unknown">
                <div class="event-header">
                    <span class="event-type">❓ {event_type}{session_info}</span>
                    <span class="timestamp">{timestamp}</span>
                </div>
                <div class="event-content">
                    <pre>{json.dumps(payload, indent=2)}</pre>
                </div>
            </div>
            """
    
    def format_step(self, step: Dict[str, Any]) -> str:
        """Format a test step for HTML display."""
        step_num = step.get('step', '?')
        step_type = step.get('step_type', 'user_message')
        duration = step.get('duration_s', 0)
        passed = step.get('passed', None)
        
        # Handle special step types
        if step_type == 'insert_attack_email':
            return self._format_insert_attack_email_step(step, step_num, duration, passed)
        elif step_type == 'insert_benign_email':
            return self._format_insert_benign_email_step(step, step_num, duration, passed)
        elif step_type == 'start_new_session':
            return self._format_start_new_session_step(step, step_num, duration, passed)
        
        # Regular user message step
        user_message = step.get('user_message', '')
        expected_behavior = step.get('expected_behavior', '')
        agent_response = step.get('agent_response', '')
        
        # Determine step status
        if passed is True:
            status_class = "passed"
            status_icon = "✅"
            status_text = "PASSED"
        elif passed is False:
            status_class = "failed"
            status_icon = "❌"
            status_text = "FAILED"
        else:
            status_class = "unknown"
            status_icon = "❓"
            status_text = "UNKNOWN"
        
        # Format user and attack goals
        user_goal = step.get('user_goal', {})
        attack_goal = step.get('attack_goal', {})
        
        goals_html = ""
        if user_goal:
            user_passed = user_goal.get('passed', None)
            user_icon = "✅" if user_passed else "❌" if user_passed is False else "❓"
            goals_html += f"""
            <div class="goal user-goal">
                <h4>{user_icon} User Goal</h4>
                <p><strong>Description:</strong> {user_goal.get('description', '')}</p>
                <p><strong>Status:</strong> {'PASSED' if user_passed else 'FAILED' if user_passed is False else 'UNKNOWN'}</p>
            </div>
            """
        
        if attack_goal and attack_goal.get('description'):
            attack_passed = attack_goal.get('passed', None)
            attack_icon = "✅" if attack_passed else "❌" if attack_passed is False else "❓"
            goals_html += f"""
            <div class="goal attack-goal">
                <h4>{attack_icon} Attack Goal</h4>
                <p><strong>Description:</strong> {attack_goal.get('description', '')}</p>
                <p><strong>Status:</strong> {'PASSED' if attack_passed else 'FAILED' if attack_passed is False else 'UNKNOWN'}</p>
            </div>
            """
        
        return f"""
        <div class="test-step {status_class}">
            <div class="step-header">
                <h3>Step {step_num} {status_icon} {status_text}</h3>
                <span class="duration">Duration: {duration:.2f}s</span>
            </div>
            
            <div class="step-content">
                <div class="user-message">
                    <h4>👤 User Message:</h4>
                    <p>{user_message}</p>
                </div>
                
                <div class="expected-behavior">
                    <h4>🎯 Expected Behavior:</h4>
                    <p>{expected_behavior}</p>
                </div>
                
                <div class="agent-response">
                    <h4>🤖 Agent Response:</h4>
                    <p>{agent_response.replace(chr(10), '<br>')}</p>
                </div>
                
                {goals_html}
            </div>
        </div>
        """
    
    def _format_insert_attack_email_step(self, step: Dict[str, Any], step_num: Any, duration: float, passed: Optional[bool]) -> str:
        """Format an insert_attack_email step for HTML display."""
        description = step.get('description', 'Inserting attack email into inbox')
        
        if passed is True:
            status_class = "passed"
            status_icon = "✅"
            status_text = "COMPLETED"
        elif passed is False:
            status_class = "failed"
            status_icon = "❌"
            status_text = "FAILED"
        else:
            status_class = "unknown"
            status_icon = "❓"
            status_text = "UNKNOWN"
        
        error_html = ""
        if passed is False and step.get('error'):
            error_html = f"""
                <div class="error-message">
                    <h4>❌ Error:</h4>
                    <p>{step.get('error')}</p>
                </div>
            """
        
        attack_email = step.get('attack_email', {})
        email_details_html = ""
        if attack_email:
            email_details_html = f"""
                <div class="email-details">
                    <h4>📧 Attack Email Details:</h4>
                    <p><strong>From:</strong> {attack_email.get('from', 'Unknown')}</p>
                    <p><strong>To:</strong> {attack_email.get('to', 'Unknown')}</p>
                    <p><strong>Subject:</strong> {attack_email.get('subject', 'No subject')}</p>
                    <p><strong>Received:</strong> {self.format_timestamp(attack_email.get('received_ts', ''))}</p>
                    <details>
                        <summary><strong>Email Body:</strong></summary>
                        <pre style="white-space: pre-wrap; max-height: 200px; overflow-y: auto;">{attack_email.get('body_plain', '')}</pre>
                    </details>
                </div>
            """
        
        return f"""
        <div class="test-step {status_class}">
            <div class="step-header">
                <h3>Step {step_num} {status_icon} {status_text} - Insert Attack Email</h3>
                <span class="duration">Duration: {duration:.2f}s</span>
            </div>
            
            <div class="step-content">
                <div class="step-description">
                    <h4>📧 {description}</h4>
                </div>
                {email_details_html}
                {error_html}
            </div>
        </div>
        """
    
    def _format_insert_benign_email_step(self, step: Dict[str, Any], step_num: Any, duration: float, passed: Optional[bool]) -> str:
        """Format an insert_benign_email step for HTML display."""
        description = step.get('description', 'Inserting benign email into inbox')
        
        if passed is True:
            status_class = "passed"
            status_icon = "✅"
            status_text = "COMPLETED"
        elif passed is False:
            status_class = "failed"
            status_icon = "❌"
            status_text = "FAILED"
        else:
            status_class = "unknown"
            status_icon = "❓"
            status_text = "UNKNOWN"
        
        error_html = ""
        if passed is False and step.get('error'):
            error_html = f"""
                <div class="error-message">
                    <h4>❌ Error:</h4>
                    <p>{step.get('error')}</p>
                </div>
            """
        
        benign_email = step.get('benign_email', {})
        email_details_html = ""
        if benign_email:
            email_details_html = f"""
                <div class="email-details">
                    <h4>📧 Benign Email Details:</h4>
                    <p><strong>From:</strong> {benign_email.get('from', 'Unknown')}</p>
                    <p><strong>To:</strong> {benign_email.get('to', 'Unknown')}</p>
                    <p><strong>Subject:</strong> {benign_email.get('subject', 'No subject')}</p>
                    <p><strong>Received:</strong> {self.format_timestamp(benign_email.get('received_ts', ''))}</p>
                    <details>
                        <summary><strong>Email Body:</strong></summary>
                        <pre style="white-space: pre-wrap; max-height: 200px; overflow-y: auto;">{benign_email.get('body_plain', '')}</pre>
                    </details>
                </div>
            """
        
        return f"""
        <div class="test-step {status_class}">
            <div class="step-header">
                <h3>Step {step_num} {status_icon} {status_text} - Insert Benign Email</h3>
                <span class="duration">Duration: {duration:.2f}s</span>
            </div>
            
            <div class="step-content">
                <div class="step-description">
                    <h4>📧 {description}</h4>
                </div>
                {email_details_html}
                {error_html}
            </div>
        </div>
        """
    
    def _format_start_new_session_step(self, step: Dict[str, Any], step_num: Any, duration: float, passed: Optional[bool]) -> str:
        """Format a start_new_session step for HTML display."""
        description = step.get('description', 'Starting new session')
        old_session_id = step.get('old_session_id', 'Unknown')
        new_session_id = step.get('new_session_id', 'Unknown')
        
        if passed is True:
            status_class = "passed"
            status_icon = "✅"
            status_text = "COMPLETED"
        elif passed is False:
            status_class = "failed"
            status_icon = "❌"
            status_text = "FAILED"
        else:
            status_class = "unknown"
            status_icon = "❓"
            status_text = "UNKNOWN"
        
        error_html = ""
        if passed is False and step.get('error'):
            error_html = f"""
                <div class="error-message">
                    <h4>❌ Error:</h4>
                    <p>{step.get('error')}</p>
                </div>
            """
        
        return f"""
        <div class="test-step {status_class}">
            <div class="step-header">
                <h3>Step {step_num} {status_icon} {status_text} - Start New Session</h3>
                <span class="duration">Duration: {duration:.2f}s</span>
            </div>
            
            <div class="step-content">
                <div class="step-description">
                    <h4>🔄 {description}</h4>
                </div>
                <div class="session-info">
                    <p><strong>Old Session ID:</strong> {old_session_id}</p>
                    <p><strong>New Session ID:</strong> {new_session_id}</p>
                </div>
                {error_html}
            </div>
        </div>
        """
    
    def generate_html_report(self, test_result: Dict[str, Any], original_test: Optional[Dict[str, Any]] = None) -> str:
        """Generate HTML report for a single test result."""
        
        # Extract basic information
        test_name = test_result.get('test_name', 'Unknown Test')
        description = test_result.get('description', '')
        session_id = test_result.get('session_id', '')
        timestamp = self.format_timestamp(test_result.get('timestamp', ''))
        overall_success = test_result.get('overall_success', False)
        steps = test_result.get('steps', [])
        summary = test_result.get('summary', {})
        
        # Get additional context from original test case
        original_name = ""
        original_description = ""
        if original_test:
            original_name = original_test.get('name', '')
            original_description = original_test.get('description', '')
        
        # Determine overall status
        status_class = "passed" if overall_success else "failed"
        status_icon = "✅" if overall_success else "❌"
        status_text = "PASSED" if overall_success else "FAILED"
        
        # Format steps
        steps_html = ""
        for step in steps:
            steps_html += self.format_step(step)
        
        # Format traces from session history
        traces_html = ""
        current_session_id = test_result.get('session_id', 'current')
        
        # Add traces from session history (all sessions including current)
        session_history = test_result.get('session_history', [])
        for session_data in session_history:
            session_id = session_data.get('session_id', 'unknown')
            step_range = session_data.get('step_range', '')
            session_traces = session_data.get('traces', [])
            
            if session_traces:
                # Determine if this is the current session
                is_current_session = session_id == current_session_id
                session_title = f"📋 Current Session {session_id}" if is_current_session else f"📋 Session {session_id} (Steps {step_range})"
                
                traces_html += f"""
                <div class="session-section">
                    <h3>{session_title}</h3>
                </div>
                """
                
                for trace in session_traces:
                    traces_html += self.format_trace_event(trace, session_id)
        
        # Calculate success statistics
        total_steps = len(steps)
        user_goal_successes = 0
        total_user_goal_steps = 0
        attack_goal_successes = 0
        total_attack_steps = 0
        
        for step in steps:
            # Skip special step types (they don't have goals)
            step_type = step.get('step_type', 'user_message')
            if step_type in ['insert_attack_email', 'insert_benign_email', 'start_new_session']:
                continue
            
            user_goal = step.get('user_goal', {})
            attack_goal = step.get('attack_goal', {})
            success_check = step.get('success_check', {})
            step_passed = step.get('passed', False)
            
            # Count user goal successes
            # For attack tests: look for user_goal with description
            # For regular tests: look for success_check or step_passed
            has_user_goal = False
            user_goal_passed = False
            
            if user_goal and user_goal.get('description'):
                # Attack test format
                has_user_goal = True
                user_goal_passed = user_goal.get('passed') is True
            elif success_check or step_passed is not None:
                # Regular test format - every step with success_check or passed status is a user goal
                has_user_goal = True
                user_goal_passed = step_passed is True
            
            if has_user_goal:
                total_user_goal_steps += 1
                if user_goal_passed:
                    user_goal_successes += 1
            
            # Count attack goal successes (only for steps that have attack goals)
            if attack_goal and attack_goal.get('description'):
                total_attack_steps += 1
                if attack_goal.get('passed') is True:
                    attack_goal_successes += 1
        
        # Calculate percentages
        user_success_rate = (user_goal_successes / total_user_goal_steps * 100) if total_user_goal_steps > 0 else 0
        attack_success_rate = (attack_goal_successes / total_attack_steps * 100) if total_attack_steps > 0 else 0
        
        # Determine if this is an attack test
        is_attack_test = total_attack_steps > 0
        
        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Report: {test_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 15px;
        }}
        
        .status-badge.passed {{
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        
        .status-badge.failed {{
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .info-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        .info-card h3 {{
            color: #495057;
            margin-bottom: 15px;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }}
        
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-label {{
            color: #6c757d;
            margin-top: 5px;
        }}
        
        .stat-percentage {{
            color: #28a745;
            font-size: 1.2em;
            font-weight: bold;
            margin-top: 5px;
        }}
        
        .section {{
            background: white;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        
        .section-header {{
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .section-header h2 {{
            color: #495057;
            margin: 0;
        }}
        
        .section-content {{
            padding: 20px;
        }}
        
        .test-step {{
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
        }}
        
        .test-step.passed {{
            border-left: 4px solid #28a745;
        }}
        
        .test-step.failed {{
            border-left: 4px solid #dc3545;
        }}
        
        .test-step.unknown {{
            border-left: 4px solid #ffc107;
        }}
        
        .step-header {{
            background: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .step-header h3 {{
            margin: 0;
            color: #495057;
        }}
        
        .duration {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        
        .step-content {{
            padding: 20px;
        }}
        
        .step-content h4 {{
            color: #495057;
            margin: 15px 0 10px 0;
        }}
        
        .step-content p {{
            margin-bottom: 15px;
        }}
        
        .goals {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .goal {{
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #e9ecef;
        }}
        
        .goal.user-goal {{
            background-color: #e3f2fd;
        }}
        
        .goal.attack-goal {{
            background-color: #fff3e0;
        }}
        
        .goal h4 {{
            margin: 0 0 10px 0;
            color: #495057;
        }}
        
        .trace-event {{
            border: 1px solid #e9ecef;
            border-radius: 6px;
            margin-bottom: 15px;
            overflow: hidden;
        }}
        
        .trace-event.user-input {{
            border-left: 4px solid #007bff;
        }}
        
        .trace-event.agent-response {{
            border-left: 4px solid #28a745;
        }}
        
        .trace-event.tool-call {{
            border-left: 4px solid #ffc107;
        }}
        
        .trace-event.tool-result {{
            border-left: 4px solid #17a2b8;
        }}
        
        .trace-event.session-change {{
            border-left: 4px solid #6f42c1;
            background-color: #f8f5ff;
        }}
        
        .event-header {{
            background: #f8f9fa;
            padding: 10px 15px;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .event-type {{
            font-weight: bold;
            color: #495057;
        }}
        
        .timestamp {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        
        .event-content {{
            padding: 15px;
        }}
        
        .event-content pre {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 0.9em;
            margin: 10px 0;
        }}
        
        .session-section {{
            margin: 30px 0 20px 0;
            padding: 15px;
            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .session-section h3 {{
            margin: 0;
            color: #495057;
            font-size: 1.2em;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: #6c757d;
            border-top: 1px solid #e9ecef;
            margin-top: 30px;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .info-grid {{
                grid-template-columns: 1fr;
            }}
            
            .summary-stats {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{test_name}</h1>
            <div class="subtitle">{description}</div>
            <div class="status-badge {status_class}">
                {status_icon} {status_text}
            </div>
        </div>
        
        <div class="info-grid">
            <div class="info-card">
                <h3>📋 Test Information</h3>
                <p><strong>Session ID:</strong> {session_id}</p>
                <p><strong>Timestamp:</strong> {timestamp}</p>
                <p><strong>Test File:</strong> {test_result.get('test_file', 'N/A')}</p>
            </div>
            
            <div class="info-card">
                <h3>🎯 Test Purpose</h3>
                <p><strong>Original Name:</strong> {original_name or 'N/A'}</p>
                <p><strong>Description:</strong> {original_description or description}</p>
            </div>
        </div>
        
        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-number">{user_goal_successes}/{total_user_goal_steps}</div>
                <div class="stat-label">User Goals Success</div>
                <div class="stat-percentage">{user_success_rate:.1f}%</div>
            </div>
            {f'''
            <div class="stat-card">
                <div class="stat-number">{attack_goal_successes}/{total_attack_steps}</div>
                <div class="stat-label">Attack Goals Success</div>
                <div class="stat-percentage">{attack_success_rate:.1f}%</div>
            </div>
            ''' if is_attack_test else ''}
            <div class="stat-card">
                <div class="stat-number">{sum(len(session_data.get('traces', [])) for session_data in session_history)}</div>
                <div class="stat-label">Trace Events</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{total_steps}</div>
                <div class="stat-label">Total Steps</div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">
                <h2>📝 Test Steps</h2>
            </div>
            <div class="section-content">
                {steps_html}
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">
                <h2>🔍 Execution Trace</h2>
            </div>
            <div class="section-content">
                {traces_html}
            </div>
        </div>
        
        <div class="footer">
            <p>Generated on {datetime(2025, 11, 3, 12, 0, 0).strftime('%Y-%m-%d %H:%M:%S')} | Memory Agent Security Benchmark</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def generate_index_page(self):
        """Generate an index page that lists all available test reports."""
        json_files = list(self.results_dir.rglob("*.json"))
        
        # Filter out summary files
        json_files = [f for f in json_files if not f.name.startswith("summary_")]
        
        # Group items by model and attack type
        items_by_model = {}
        for json_file in json_files:
            try:
                data = self.load_test_result(json_file)
                test_name = data.get('test_name') or data.get('name') or json_file.stem
                description = data.get('description', '')
                session_id = data.get('session_id', '')
                
                # Extract model name and attack type from folder structure
                path_parts = json_file.relative_to(self.results_dir).parts
                if len(path_parts) >= 2:
                    # Structure: model_name/attack_type/test.json
                    model_name = path_parts[0]
                    attack_type = path_parts[1]
                else:
                    # Fallback for old structure or single-level files
                    model_name = 'unknown'
                    attack_type = 'benign'
                    if len(path_parts) >= 1:
                        attack_type = path_parts[0]
                
                # Only include valid attack types
                if attack_type not in ['benign', 'direct', 'indirect']:
                    continue
                
                # Initialize model group if not exists
                if model_name not in items_by_model:
                    items_by_model[model_name] = {'benign': [], 'direct': [], 'indirect': []}
                
                link = f'{model_name}/{attack_type}/{json_file.stem}.html'
                
                # Check if HTML report exists
                if not (self.output_dir / link).exists():
                    continue
                    
                items_by_model[model_name][attack_type].append({
                    'test_name': test_name,
                    'description': description,
                    'session_id': session_id,
                    'link': link,
                })
            except Exception:
                continue
        
        # Generate organized sections HTML by model
        sections_html = []
        type_configs = {
            'benign': {'title': '🛡️ Benign Tests', 'description': 'Normal functionality tests', 'color': '#28a745'},
            'direct': {'title': '⚡ Direct Attacks', 'description': 'Direct injection and manipulation attacks', 'color': '#dc3545'},
            'indirect': {'title': '🎯 Indirect Attacks', 'description': 'Subtle manipulation and policy change attacks', 'color': '#fd7e14'}
        }
        
        for model_name in sorted(items_by_model.keys()):
            model_items = items_by_model[model_name]
            total_tests = sum(len(items) for items in model_items.values())
            
            if total_tests == 0:
                continue
                
            # Add model header
            sections_html.append(f'''            <div class="model-section">
                <div class="model-header">
                    <h1>🤖 Model: {model_name}</h1>
                    <p>Total tests: {total_tests}</p>
                </div>
            </div>''')
            
            # Add attack type sections for this model
            for attack_type in ['benign', 'direct', 'indirect']:
                items = model_items[attack_type]
                if not items:
                    continue
                    
                config = type_configs[attack_type]
                cards_html = []
                for item in items:
                    cards_html.append(f'''                <a href="{item['link']}" class="report-card">
                        <h3>{item['test_name']}</h3>
                        <div class="description">
                            {item['description']}
                        </div>
                        <div class="report-meta">
                            <span class="report-date">Session: {item['session_id']}</span>
                            <span class="view-report">View Report →</span>
                        </div>
                    </a>''')
                
                sections_html.append(f'''            <div class="test-section">
                <div class="section-header" style="border-left: 4px solid {config['color']};">
                    <h2>{config['title']}</h2>
                    <p>{config['description']} ({len(items)} test{'s' if len(items) != 1 else ''})</p>
                </div>
                <div class="reports-grid">
{chr(10).join(cards_html)}
                </div>
            </div>''')
        
        # Generate index HTML
        index_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memory Agent Security Benchmark - Test Reports</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 50px;
        }}
        .header h1 {{
            font-size: 3em;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .header p {{
            font-size: 1.3em;
            opacity: 0.9;
        }}
        .test-section {{
            margin-bottom: 60px;
        }}
        .section-header {{
            background: rgba(255, 255, 255, 0.1);
            padding: 20px 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }}
        .section-header h2 {{
            color: white;
            font-size: 2em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .section-header p {{
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.1em;
            margin: 0;
        }}
        .model-section {{
            margin-bottom: 2rem;
        }}
        .model-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            text-align: center;
        }}
        .model-header h1 {{
            margin: 0 0 0.5rem 0;
            font-size: 1.8rem;
            font-weight: 600;
        }}
        .model-header p {{
            margin: 0;
            font-size: 1rem;
            opacity: 0.9;
        }}
        .reports-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
        }}
        .report-card {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            text-decoration: none;
            color: inherit;
        }}
        .report-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        }}
        .report-card h3 {{
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.4em;
        }}
        .report-card .description {{
            color: #6c757d;
            margin-bottom: 20px;
            line-height: 1.5;
        }}
        .report-meta {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #e9ecef;
        }}
        .report-date {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        .view-report {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            text-decoration: none;
            font-size: 0.9em;
            font-weight: bold;
            transition: opacity 0.3s ease;
        }}
        .view-report:hover {{
            opacity: 0.9;
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            color: white;
            opacity: 0.8;
        }}
        @media (max-width: 768px) {{
            .header h1 {{ font-size: 2em; }}
            .reports-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Memory Agent Security Benchmark</h1>
            <p>Test Execution Reports</p>
        </div>
{chr(10).join(sections_html)}
        <div class="footer">
            <p>Generated on {datetime(2025, 11, 3, 12, 0, 0).strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>'''
        
        # Save index file at root level since we now group by model in the index
        index_file = self.output_dir / 'index.html'
        
        index_file.parent.mkdir(parents=True, exist_ok=True)
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(index_html)
        
        print(f"📋 Generated index page: {index_file}")
        total_items = sum(len(items) for model_items in items_by_model.values() for items in model_items.values())
        return total_items

    def generate_all_reports(self):
        """Generate HTML reports for all test result files."""
        json_files = list(self.results_dir.rglob("*.json"))
        
        # Filter out summary files
        json_files = [f for f in json_files if not f.name.startswith("summary_")]
        
        print(f"Found {len(json_files)} test result files to process...")
        
        for json_file in json_files:
            print(f"Processing {json_file.name}...")
            
            try:
                # Load test result
                test_result = self.load_test_result(json_file)
                
                # Load original test case if available
                original_test = None
                test_file_path = test_result.get('test_file')
                if test_file_path:
                    original_test = self.load_original_test_case(test_file_path)
                
                # Extract model name and attack type from folder structure
                path_parts = json_file.relative_to(self.results_dir).parts
                if len(path_parts) >= 2:
                    # Structure: model_name/attack_type/test.json
                    model_name = path_parts[0]
                    attack_type = path_parts[1]
                else:
                    # Fallback for old structure or single-level files
                    model_name = 'unknown'
                    attack_type = 'benign'
                    if len(path_parts) >= 1:
                        attack_type = path_parts[0]
                
                # Only process valid attack types
                if attack_type not in ['benign', 'direct', 'indirect']:
                    print(f"  ⚠️  Skipping {json_file.name} - invalid attack type: {attack_type}")
                    continue
                
                # Ensure output subfolders exist
                (self.output_dir / model_name / attack_type).mkdir(parents=True, exist_ok=True)

                # Generate HTML report
                html_content = self.generate_html_report(test_result, original_test)
                
                # Save HTML file
                html_filename = json_file.stem + ".html"
                html_file = self.output_dir / model_name / attack_type / html_filename
                
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                print(f"  ✅ Generated: {html_file}")
                
            except Exception as e:
                print(f"  ❌ Error processing {json_file.name}: {e}")
        
        # Generate index page
        print(f"\n📋 Generating index page...")
        report_count = self.generate_index_page()
        
        print(f"\n🎉 HTML reports generated in: {self.output_dir}")
        print(f"📁 Generated {report_count} report(s) with index page")
        print(f"🌐 Open {self.output_dir}/index.html in your browser to view all reports")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML reports from test result JSON files")
    # Default to test_bench_results in the current working directory
    default_results_dir = Path(__file__).parent / "test_bench_results"
    parser.add_argument("--results-dir", 
                       default=str(default_results_dir),
                       help="Directory containing test result JSON files")
    parser.add_argument("--output-dir", 
                       default="html_reports",
                       help="Directory to save HTML reports")
    
    args = parser.parse_args()
    
    generator = TestResultHTMLGenerator(args.results_dir, args.output_dir)
    generator.generate_all_reports()


if __name__ == "__main__":
    main()
