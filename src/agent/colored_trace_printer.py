#!/usr/bin/env python3
"""
Colored trace event printer for terminal output.
Provides beautiful, color-coded output for different event types during benchmark runs.
"""

import json
import sys
from typing import Any, Dict, Optional


class ColoredTracePrinter:
    """Prints trace events with color coding to terminal."""
    
    # ANSI color codes
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Colors for different event types
    USER_INPUT = '\033[94m'  # Blue
    AGENT_RESPONSE = '\033[92m'  # Green
    TOOL_CALL = '\033[93m'  # Yellow
    TOOL_RESULT = '\033[96m'  # Cyan
    ERROR = '\033[91m'  # Red
    INFO = '\033[90m'  # Dark gray
    META = '\033[95m'  # Magenta
    PASSED_COLOR = '\033[92m'  # Green
    
    MAX_LENGTH = 400  # Maximum length for truncation
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the colored trace printer.
        
        Args:
            enabled: Whether to enable colored output (auto-detected if None)
        """
        self.enabled = enabled and sys.stdout.isatty() if enabled else False
    
    def _truncate(self, text: str, max_len: int = None) -> str:
        """Truncate text to max_len characters, adding '...' if truncated."""
        max_len = max_len or self.MAX_LENGTH
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."
    
    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if enabled."""
        if self.enabled:
            return f"{color}{text}{self.RESET}"
        return text
    
    def format_langchain_update(self, update: Dict[str, Any]) -> Optional[str]:
        """
        Format LangChain's verbose update output.
        Handles [updates] and [values] style outputs from LangChain callbacks.
        
        Args:
            update: Dictionary containing LangChain update data
            
        Returns:
            Formatted string or None if not applicable
        """
        if not isinstance(update, dict):
            return None
        
        # Check for 'messages' in updates (LangChain callback format)
        if 'messages' in update:
            messages = update.get('messages', [])
            if messages:
                last_msg = messages[-1]
                
                # Check if it's a tool call
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    tool_calls = last_msg.tool_calls
                    for tool_call in tool_calls:
                        tool_name = tool_call.get('name', tool_call.get('tool', 'unknown'))
                        args = tool_call.get('args', {})
                        args_str = json.dumps(args, indent=2) if args else "{}"
                        # args_str = self._truncate(args_str)
                        return self._colorize(
                            f"Tool Call: {tool_name}\n{self.BOLD}Inputs:{self.RESET} {args_str}",
                            self.TOOL_CALL
                        )
                
                # Check if it's an agent response
                if hasattr(last_msg, 'content') and last_msg.content:
                    content = str(last_msg.content)
                    if content.strip():
                        # Don't truncate agent responses - show full content
                        return self._colorize(
                            f"Agent: {content}",
                            self.AGENT_RESPONSE
                        )
        
        # Check for 'tools' in updates (tool results)
        if 'tools' in update:
            tools = update.get('tools', {})
            if 'messages' in tools:
                tool_messages = tools['messages']
                for msg in tool_messages:
                    if hasattr(msg, 'name') and hasattr(msg, 'content'):
                        tool_name = msg.name
                        content = str(msg.content)
                        content = self._truncate(content)
                        return self._colorize(
                            f"Tool Result ({tool_name}): {content}",
                            self.TOOL_RESULT
                        )
        
        # Check for 'model' in updates
        if 'model' in update:
            model = update.get('model', {})
            if 'messages' in model:
                messages = model['messages']
                for msg in messages:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_name = tool_call.get('name', tool_call.get('tool', 'unknown'))
                            args = tool_call.get('args', {})
                            args_str = json.dumps(args, indent=2) if args else "{}"
                            # args_str = self._truncate(args_str)
                            return self._colorize(
                                f"Tool Call: {tool_name}\n{self.BOLD}Inputs:{self.RESET} {args_str}",
                                self.TOOL_CALL
                            )
                    elif hasattr(msg, 'content') and msg.content:
                        content = str(msg.content)
                        if content.strip():
                            # Don't truncate agent responses - show full content
                            return self._colorize(
                                f"Agent: {content}",
                                self.AGENT_RESPONSE
                            )
        
        return None
    
    def format_trace_event(self, event: Dict[str, Any]) -> str:
        """
        Format a trace event from our trace system.
        
        Args:
            event: Trace event dictionary
            
        Returns:
            Formatted colored string
        """
        event_type = event.get('event_type', 'unknown')
        payload = event.get('payload', {})
        
        if event_type == 'user_input':
            text = payload.get('text', '')
            text = self._truncate(text)
            return self._colorize(
                f"User: {text}",
                self.USER_INPUT
            )
        
        elif event_type == 'agent_response':
            text = payload.get('text', '')
            # Don't truncate agent responses - show full content
            return self._colorize(
                f"Agent: {text}",
                self.AGENT_RESPONSE
            )
        
        elif event_type == 'tool_call':
            tool_name = payload.get('tool_name', 'unknown')
            inputs = payload.get('inputs', {})
            inputs_str = json.dumps(inputs, indent=2) if inputs else "{}"
            inputs_str = self._truncate(inputs_str)
            return self._colorize(
                f"Tool Call: {tool_name}\n{self.BOLD}Inputs:{self.RESET} {inputs_str}",
                self.TOOL_CALL
            )
        
        elif event_type == 'tool_result':
            tool_name = payload.get('tool_name', 'unknown')
            outputs = payload.get('outputs', {})
            # Extract result or error
            if 'error' in outputs:
                error = str(outputs['error'])
                error = self._truncate(error)
                return self._colorize(
                    f"Tool Error ({tool_name}): {error}",
                    self.ERROR
                )
            elif 'result' in outputs:
                result = str(outputs['result'])
                result = self._truncate(result)
                return self._colorize(
                    f"Tool Result ({tool_name}): {result}",
                    self.TOOL_RESULT
                )
            else:
                outputs_str = json.dumps(outputs, indent=2)
                outputs_str = self._truncate(outputs_str)
                return self._colorize(
                    f"Tool Result ({tool_name}): {outputs_str}",
                    self.TOOL_RESULT
                )
        
        elif event_type == 'validator_results':
            operator = payload.get('operator', 'AND')
            overall_passed = payload.get('overall_passed', False)
            validators = payload.get('validators', [])
            
            status_icon = "✓" if overall_passed else "✗"
            status_text = "PASSED" if overall_passed else "FAILED"
            status_color = self.PASSED_COLOR if overall_passed else self.ERROR
            
            result_lines = [
                f"{status_color}{status_icon} Validator Results ({operator}): {status_text}{self.RESET}"
            ]
            
            for v in validators:
                v_type = v.get('type', 'unknown')
                v_name = v.get('name', 'validator')
                v_passed = v.get('passed', False)
                v_icon = "✓" if v_passed else "✗"
                v_color = self.PASSED_COLOR if v_passed else self.ERROR
                result_lines.append(
                    f"  {v_color}{v_icon}{self.RESET} [{v_type}] {v_name}: {'PASSED' if v_passed else 'FAILED'}"
                )
            
            return "\n".join(result_lines)
        
        else:
            # Generic event
            event_str = json.dumps(payload, indent=2)
            event_str = self._truncate(event_str)
            return self._colorize(
                f"{event_type}: {event_str}",
                self.INFO
            )
    
    def print_langchain_callback(self, update: Dict[str, Any]):
        """
        Print LangChain callback update in colored format.
        Call this from LangChain callbacks.
        
        Args:
            update: LangChain update dictionary
        """
        formatted = self.format_langchain_update(update)
        if formatted:
            print(formatted)
    
    def print_trace_event(self, event: Dict[str, Any]):
        """
        Print trace event in colored format.
        
        Args:
            event: Trace event dictionary
        """
        formatted = self.format_trace_event(event)
        print(formatted)


# Global instance
_printer = None

def get_colored_printer() -> ColoredTracePrinter:
    """Get the global colored printer instance."""
    global _printer
    if _printer is None:
        _printer = ColoredTracePrinter()
    return _printer


def print_validator_result(validator_type: str, validator_name: str, passed: bool, details: str = ""):
    """
    Print a validator result with color coding.
    
    Args:
        validator_type: Type of validator (keyword, tool_call, semantic_judge, etc.)
        validator_name: Descriptive name for the validator
        passed: Whether the validator passed
        details: Optional additional details
    """
    printer = get_colored_printer()
    
    # Color for validator results
    VALIDATOR_COLOR = '\033[95m'  # Magenta
    PASSED_COLOR = '\033[92m'  # Green
    FAILED_COLOR = '\033[91m'  # Red
    
    status_icon = "✓" if passed else "✗"
    status_text = "PASSED" if passed else "FAILED"
    status_color = PASSED_COLOR if passed else FAILED_COLOR
    
    if printer.enabled:
        validator_info = f"{VALIDATOR_COLOR}[{validator_type}]{printer.RESET} {validator_name}"
        status_info = f"{status_color}{status_icon} {status_text}{printer.RESET}"
        if details:
            print(f"  {validator_info}: {status_info} - {details}")
        else:
            print(f"  {validator_info}: {status_info}")
    else:
        # No color fallback
        print(f"  [{validator_type}] {validator_name}: {status_icon} {status_text}" + (f" - {details}" if details else ""))

