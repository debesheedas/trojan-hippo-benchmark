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
    
    def format_trace_event(self, event: Dict[str, Any], session_id: str = None) -> str:
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
        user_message = step.get('user_message', '')
        expected_behavior = step.get('expected_behavior', '')
        agent_response = step.get('agent_response', '')
        duration = step.get('duration_s', 0)
        passed = step.get('passed', None)
        
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
    
    def generate_html_report(self, test_result: Dict[str, Any], original_test: Optional[Dict[str, Any]] = None) -> str:
        """Generate HTML report for a single test result."""
        
        # Extract basic information
        test_name = test_result.get('test_name', 'Unknown Test')
        description = test_result.get('description', '')
        session_id = test_result.get('session_id', '')
        timestamp = self.format_timestamp(test_result.get('timestamp', ''))
        overall_success = test_result.get('overall_success', False)
        steps = test_result.get('steps', [])
        traces = test_result.get('traces', [])
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
        
        # Format traces - include both main traces and session history
        traces_html = ""
        current_session_id = test_result.get('session_id', 'current')
        
        # First, add traces from session history (previous sessions only)
        session_history = test_result.get('session_history', [])
        for session_data in session_history:
            session_id = session_data.get('session_id', 'unknown')
            step_range = session_data.get('step_range', '')
            session_traces = session_data.get('traces', [])
            
            # Only show sessions that are different from the current session
            if session_traces and session_id != current_session_id:
                traces_html += f"""
                <div class="session-section">
                    <h3>📋 Session {session_id} (Steps {step_range})</h3>
                </div>
                """
                
                for trace in session_traces:
                    traces_html += self.format_trace_event(trace, session_id)
        
        # Then add current session traces
        if traces:
            traces_html += f"""
            <div class="session-section">
                <h3>📋 Current Session {current_session_id}</h3>
            </div>
            """
            
            for trace in traces:
                traces_html += self.format_trace_event(trace, current_session_id)
        
        # Calculate success statistics
        total_steps = len(steps)
        user_goal_successes = 0
        total_user_goal_steps = 0
        attack_goal_successes = 0
        total_attack_steps = 0
        
        for step in steps:
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
                <div class="stat-number">{len(traces)}</div>
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
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Memory Agent Security Benchmark</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def generate_all_reports(self):
        """Generate HTML reports for all test result files."""
        json_files = list(self.results_dir.glob("*.json"))
        
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
                
                # Generate HTML report
                html_content = self.generate_html_report(test_result, original_test)
                
                # Save HTML file
                html_filename = json_file.stem + ".html"
                html_file = self.output_dir / html_filename
                
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                print(f"  ✅ Generated: {html_file}")
                
            except Exception as e:
                print(f"  ❌ Error processing {json_file.name}: {e}")
        
        print(f"\n🎉 HTML reports generated in: {self.output_dir}")
        print(f"📁 Open any .html file in your browser to view the reports")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML reports from test result JSON files")
    parser.add_argument("--results-dir", 
                       default="/Users/ddas/Desktop/Debeshee/Thesis/memory-agent-security-benchmark/test_bench_results",
                       help="Directory containing test result JSON files")
    parser.add_argument("--output-dir", 
                       default="html_reports",
                       help="Directory to save HTML reports")
    
    args = parser.parse_args()
    
    generator = TestResultHTMLGenerator(args.results_dir, args.output_dir)
    generator.generate_all_reports()


if __name__ == "__main__":
    main()
