"""
Generalized test validation functions for the Email Agent test bench.

This module provides flexible validation functions that can be configured
via parameters in the test JSON files.
"""

import json
import re
import fnmatch
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from agent.utils import (
    debug_info, debug_debug, debug_print_exception, load_config,
    call_llm_chat_completion, detect_provider, append_trace_event
)
from agent.utils import print_validator_result


class TestValidator:
    """Base class for test validators."""
    
    def __init__(self, config: Dict[str, Any], test_config: Optional[Dict[str, Any]] = None):
        self.config = config
        self.test_config = test_config
    
    def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
        """Validate the test result. Override in subclasses."""
        raise NotImplementedError

    def score(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> float:
        """Numeric score in [0,1] indicating partial success. Default maps bool to float."""
        try:
            return 1.0 if self.validate(agent_response, session_id, traces) else 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def _check_keywords_in_text(
        text: str,
        keywords_lower: List[str],
        require_all: bool = False,
        min_keywords: Optional[int] = None
    ) -> bool:
        """
        Helper method to check if keywords are present in text.
        
        Args:
            text: Text to search in (should be lowercase)
            keywords_lower: List of lowercase keywords to find
            require_all: If True, all keywords must be present
            min_keywords: Minimum number of keywords that must be present
            
        Returns:
            True if keywords match the requirements, False otherwise
        """
        if not keywords_lower:
            return True
        
        found_keywords = set()
        text_lower = text.lower()
        
        for keyword in keywords_lower:
            if keyword in text_lower:
                found_keywords.add(keyword)
                # Early exit optimizations
                if min_keywords is not None and len(found_keywords) >= min_keywords:
                    break
                elif min_keywords is None and not require_all and len(found_keywords) > 0:
                    break
        
        found_count = len(found_keywords)
        
        if min_keywords is not None:
            return found_count >= min_keywords
        elif require_all:
            return found_count == len(keywords_lower)
        else:
            return found_count > 0


class KeywordValidator(TestValidator):
    """
    Validates that specific keywords/phrases are present or absent in the agent response.
    
    Config options:
    - keywords: List of keywords/phrases to look for
    - require_all: If True, all keywords must be present. If False, any keyword is sufficient.
    - min_required: Minimum number of keywords that must be present (overrides require_all)
    - check_absence: If True, validates that keywords are NOT present (default: False)
    - case_sensitive: Whether to do case-sensitive matching (default: False)
    """
    
    def __init__(self, config: Dict[str, Any], test_config: Optional[Dict[str, Any]] = None):
        super().__init__(config, test_config)
        # Pre-normalize keywords for performance
        self._keywords = config.get("keywords", [])
        self._case_sensitive = config.get("case_sensitive", False)
        self._keywords_normalized = [kw if self._case_sensitive else kw.lower() for kw in self._keywords]
        self._require_all = config.get("require_all", False)
        self._min_required = config.get("min_required", None)
        self._check_absence = config.get("check_absence", False)
    
    def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
        if not self._keywords:
            return True  # No keywords to check
        
        text = agent_response if self._case_sensitive else agent_response.lower()
        
        if self._check_absence:
            # Check that none of the keywords are present
            return not any(keyword in text for keyword in self._keywords_normalized)
        
        # For presence check, use the base class helper for case-insensitive or inline for case-sensitive
        if self._case_sensitive:
            # Case-sensitive: need to check against original text and keywords
            found_keywords = set()
            for keyword in self._keywords:
                if keyword in agent_response:
                    found_keywords.add(keyword)
                    # Early exit optimizations
                    if self._min_required is not None and len(found_keywords) >= self._min_required:
                        break
                    elif self._min_required is None and not self._require_all and len(found_keywords) > 0:
                        break
            
            found_count = len(found_keywords)
            if self._min_required is not None:
                return found_count >= self._min_required
            elif self._require_all:
                return found_count == len(self._keywords)
            else:
                return found_count > 0
        else:
            # Case-insensitive: use the base class helper
            return self._check_keywords_in_text(
                text,
                self._keywords_normalized,
                self._require_all,
                self._min_required
            )


class ToolCallValidator(TestValidator):
    """
    Validates that specific tools were called with expected parameters.
    
    Config options:
    - required_tools: List of tool names that must be called
    - tool_params: Dict mapping tool names to expected parameter values
    - require_all_tools: If True, all required tools must be called
    - min_tools: Minimum number of required tools that must be called
    - check_params: Whether to validate tool parameters (default: True)
    """
    
    def __init__(self, config: Dict[str, Any], test_config: Optional[Dict[str, Any]] = None):
        super().__init__(config, test_config)
        # Pre-normalize for performance
        self._required_tools = set(config.get("required_tools", []))  # Use set for O(1) lookup
        self._tool_params = config.get("tool_params", {})
        self._require_all_tools = config.get("require_all_tools", True)
        self._min_tools = config.get("min_tools", None)
        self._check_params = config.get("check_params", True)
        
        # Pre-normalize string parameters
        self._normalized_tool_params = {}
        for tool_name, params in self._tool_params.items():
            self._normalized_tool_params[tool_name] = {}
            for param_name, expected_value in params.items():
                if isinstance(expected_value, str):
                    self._normalized_tool_params[tool_name][param_name] = expected_value.lower()
                elif isinstance(expected_value, list):
                    self._normalized_tool_params[tool_name][param_name] = [kw.lower() for kw in expected_value]
                else:
                    self._normalized_tool_params[tool_name][param_name] = expected_value
    
    def _collect_tool_calls(self, traces: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """Collect tool calls from traces, returning dict mapping tool_name to inputs."""
        tool_inputs = {}
        if not traces:
            return tool_inputs
        
        for trace in traces:
            if trace.get("event_type") == "tool_call":
                payload = trace.get("payload", {})
                tool_name = payload.get("tool_name", "")
                if tool_name in self._required_tools:
                    tool_inputs[tool_name] = payload.get("inputs", {})
        return tool_inputs
    
    def _validate_tool_count(self, called_tools_count: int) -> bool:
        """Validate that the number of called tools meets requirements."""
        if self._min_tools is not None:
            return called_tools_count >= self._min_tools
        elif self._require_all_tools:
            return called_tools_count >= len(self._required_tools)
        else:
            return called_tools_count > 0
    
    def _validate_param_value(self, actual_value: Any, expected_value: Any, normalized_value: Any, 
                             min_required: Optional[int] = None) -> bool:
        """Validate a single parameter value against expected value."""
        if isinstance(expected_value, str):
            return normalized_value in str(actual_value).lower()
        
        elif isinstance(expected_value, list):
            # List of keywords - normalize whitespace for multi-line matching
            actual_normalized = re.sub(r'\s+', ' ', str(actual_value).lower())
            normalized_keywords = normalized_value
            
            if min_required is not None:
                # Count keywords found (with whitespace normalization)
                found_count = sum(
                    1 for kw in normalized_keywords
                    if re.sub(r'\s+', ' ', kw) in actual_normalized
                )
                return found_count >= min_required
            else:
                # Check if any keyword is present
                return any(re.sub(r'\s+', ' ', kw) in actual_normalized for kw in normalized_keywords)
        
        elif isinstance(expected_value, dict):
            return self._validate_nested_params(actual_value, expected_value)
        
        else:
            # Exact match
            return actual_value == expected_value
    
    def _validate_tool_params(self, tool_inputs: Dict[str, Dict[str, Any]]) -> bool:
        """Validate parameters for all tools that have parameter expectations."""
        if not self._check_params or not self._tool_params:
            return True
        
        for tool_name, actual_inputs in tool_inputs.items():
            if tool_name not in self._tool_params:
                continue
            
            expected_params = self._tool_params[tool_name]
            normalized_params = self._normalized_tool_params[tool_name]
            
            for param_name, expected_value in expected_params.items():
                # Skip validation-only parameters
                if param_name == 'min_required':
                    continue
                
                if param_name not in actual_inputs:
                    return False
                
                actual_value = actual_inputs[param_name]
                normalized_value = normalized_params.get(param_name, expected_value)
                min_required = expected_params.get('min_required')
                
                if not self._validate_param_value(actual_value, expected_value, normalized_value, min_required):
                    return False
        
        return True
    
    def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
        if not self._required_tools:
            return True  # No tools to check
        
        if not traces:
            return False
        
        # Collect tool calls
        tool_inputs = self._collect_tool_calls(traces)
        called_tools_count = len(tool_inputs)
        
        # Validate tool count
        if not self._validate_tool_count(called_tools_count):
            return False
        
        # Validate tool parameters
        return self._validate_tool_params(tool_inputs)

    def score(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> float:
        """
        Partial scoring rule:
        - Each required tool contributes equally to the score.
        - If a required tool was not called: 0 for that portion.
        - If called and params expected:
            - Params match -> full credit for that portion (1.0 share)
            - Params mismatch -> half credit (0.5 share)
        - If no params expected or check_params is False -> full credit if called.
        """
        if not self._required_tools:
            return 1.0
        
        if not traces:
            return 0.0
        
        # Collect tool calls
        tool_inputs = self._collect_tool_calls(traces)
        num_required = len(self._required_tools)
        
        # Score each required tool
        per_tool_scores = []
        for tool_name in self._required_tools:
            if tool_name not in tool_inputs:
                per_tool_scores.append(0.0)
                continue
            
            # Tool was called
            if not self._check_params or tool_name not in self._tool_params:
                per_tool_scores.append(1.0)
                continue
            
            # Check parameters
            actual_inputs = tool_inputs[tool_name]
            expected_params = self._tool_params[tool_name]
            normalized_params = self._normalized_tool_params.get(tool_name, {})
            
            params_match = True
            for param_name, expected_value in expected_params.items():
                if param_name == 'min_required':
                    continue
                
                if param_name not in actual_inputs:
                    params_match = False
                    break
                
                actual_value = actual_inputs[param_name]
                normalized_value = normalized_params.get(param_name, expected_value)
                min_required = expected_params.get('min_required')
                
                if not self._validate_param_value(actual_value, expected_value, normalized_value, min_required):
                    params_match = False
                    break
            
            # Full credit if params match, half credit if called but params don't match
            per_tool_scores.append(1.0 if params_match else 0.5)
        
        return sum(per_tool_scores) / num_required if per_tool_scores else 0.0
    
    def _validate_nested_params(self, actual: Any, expected: Dict[str, Any]) -> bool:
        """Validate nested parameters."""
        if not isinstance(actual, dict):
            return False
        
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            
            actual_value = actual[key]
            if isinstance(expected_value, str):
                if expected_value.lower() not in str(actual_value).lower():
                    return False
            elif actual_value != expected_value:
                return False
        
        return True


class FileSystemValidator(TestValidator):
    """
    Validates that specific files exist and match JSON field expectations.
    
    This validator is used to check if files (typically email JSON files) exist and have
    specific field values (e.g., checking that an email was sent to a specific recipient).
    Content validation is handled separately by semantic_judge validators.
    
    Config options:
    - check_files: List of file patterns to check (supports glob patterns like "data/outbox/*.json")
    - json_fields: Dict mapping file patterns to expected JSON field values
      Example: {"data/outbox/*.json": {"to": "user@example.com"}}
    - require_all_files: If True, all files must exist (default: True)
    - min_files: Minimum number of files that must exist
    """
    
    def __init__(self, config: Dict[str, Any], test_config: Optional[Dict[str, Any]] = None):
        super().__init__(config, test_config)
        self._check_files = config.get("check_files", [])
        self._json_fields = config.get("json_fields", {})
        self._require_all_files = config.get("require_all_files", True)
        self._min_files = config.get("min_files", None)
        
        # Pre-resolve file patterns to test environment paths
        self._resolved_files = [self._resolve_path(pattern) for pattern in self._check_files]
        
        # Pre-resolve JSON field patterns
        self._resolved_json_fields = {
            self._resolve_path(pattern): fields 
            for pattern, fields in self._json_fields.items()
        }
    
    def _resolve_path(self, file_pattern: str) -> str:
        """Resolve file pattern to test environment path if test_config is available."""
        if not self.test_config:
            return file_pattern
        
        # Map common data directory patterns to test environment paths
        data_mappings = {
            "data/outbox": self.test_config["data"]["outbox_dir"],
            "data/drafts": self.test_config["data"]["drafts_dir"],
            "data/mailbox": self.test_config["data"]["mailbox_dir"],
            "data/sessions": self.test_config["data"].get("sessions_dir", "data/agent/sessions"),
            "data/agent/outbox": self.test_config["data"]["outbox_dir"],
            "data/agent/drafts": self.test_config["data"]["drafts_dir"],
            "data/agent/mailbox": self.test_config["data"]["mailbox_dir"],
            "data/agent/sessions": self.test_config["data"].get("sessions_dir", "data/agent/sessions"),
        }
        
        # Replace data directory paths with test environment paths
        resolved_pattern = file_pattern
        for original_path, test_path in data_mappings.items():
            if file_pattern.startswith(original_path):
                resolved_pattern = file_pattern.replace(original_path, test_path)
                break
        
        return resolved_pattern
    
    def _find_files_matching_patterns(self) -> Tuple[List[str], Dict[str, List[str]]]:
        """Find all files matching the patterns, returning (all_files, pattern_to_files mapping)."""
        found_files = []
        pattern_to_files = {}
        
        for resolved_pattern in self._resolved_files:
            pattern_files = []
            
            # Handle glob patterns
            if '*' in resolved_pattern or '?' in resolved_pattern:
                pattern_str = str(resolved_pattern)
                last_sep_idx = max(pattern_str.rfind('/'), pattern_str.rfind('\\'))
                
                if last_sep_idx >= 0:
                    dir_part, glob_part = pattern_str[:last_sep_idx], pattern_str[last_sep_idx + 1:]
                    dir_path = Path(dir_part)
                    if dir_path.exists() and dir_path.is_dir():
                        pattern_files = [str(p) for p in dir_path.glob(glob_part) if p.is_file()]
                    else:
                        pattern_files = [str(p) for p in Path(".").glob(resolved_pattern) if p.is_file()]
                else:
                    pattern_files = [str(p) for p in Path(".").glob(resolved_pattern) if p.is_file()]
            else:
                # No glob - check as regular path
                pattern_path = Path(resolved_pattern)
                if pattern_path.is_file():
                    pattern_files = [str(pattern_path)]
                elif pattern_path.is_dir():
                    pattern_files = [str(p) for p in pattern_path.glob("*") if p.is_file()]
            
            found_files.extend(pattern_files)
            if pattern_files:
                pattern_to_files[resolved_pattern] = pattern_files
        
        # Remove duplicates while preserving order
        found_files = list(dict.fromkeys(found_files))
        return found_files, pattern_to_files
    
    def _validate_file_count(self, found_files: List[str]) -> bool:
        """Validate that the number of found files meets requirements."""
        if self._min_files is not None:
            return len(found_files) >= self._min_files
        elif self._require_all_files:
            return len(found_files) >= len(self._check_files)
        return True
    
    def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
        if not self._check_files:
            return True  # No files to check
        
        # Find all matching files
        found_files, pattern_to_files = self._find_files_matching_patterns()
        
        # Validate file count
        if not self._validate_file_count(found_files):
            return False
        
        # Validate JSON fields if specified
        if self._resolved_json_fields:
            return self._validate_json_fields(found_files, pattern_to_files)
        
        return True
    
    def _validate_json_fields(self, found_files: List[str], pattern_to_files: Dict[str, List[str]]) -> bool:
        """Validate JSON fields match expectations."""
        for resolved_pattern, expected_fields in self._resolved_json_fields.items():
            pattern_files = pattern_to_files.get(resolved_pattern, [])
            if not pattern_files:
                pattern_files = [f for f in found_files if self._matches_pattern(f, resolved_pattern)]
            
            # Check all matching files - if any file matches the JSON fields, validation passes
            for file_path in pattern_files:
                if self._check_json_fields(file_path, expected_fields):
                    return True
        
        return False
    
    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches the pattern."""
        try:
            file_path_obj = Path(file_path)
            pattern_obj = Path(pattern)
            
            # If pattern contains glob characters, use fnmatch
            if '*' in pattern or '?' in pattern:
                # Try various path matching strategies
                file_str = str(file_path_obj.resolve()) if file_path_obj.exists() else str(file_path_obj)
                pattern_str = str(pattern_obj)
                
                if fnmatch.fnmatch(file_str, pattern_str) or fnmatch.fnmatch(str(file_path_obj), pattern_str):
                    return True
                
                # For patterns like "dir/*.json", check directory and filename separately
                if '/' in pattern_str or '\\' in pattern_str:
                    pattern_dir = pattern_obj.parent
                    pattern_glob = pattern_obj.name
                    if (file_path_obj.parent == pattern_dir or 
                        str(file_path_obj.parent) == str(pattern_dir)):
                        return fnmatch.fnmatch(file_path_obj.name, pattern_glob)
                return False
            else:
                # No glob - use direct path comparison
                return (str(file_path_obj) == str(pattern_obj) or 
                        (file_path_obj.exists() and pattern_obj.exists() and 
                         file_path_obj.resolve() == pattern_obj.resolve()))
        except Exception as e:
            # Fallback: simple string containment check
            debug_debug(f"Could not match file path pattern, using simple string containment fallback")
            debug_print_exception(e, context="Matching file path pattern", include_traceback=True)
            return pattern in file_path
    
    def _check_json_fields_data(self, data: Dict[str, Any], expected_fields: Dict[str, Any]) -> bool:
        """Check if JSON data contains expected field values (works with data object, not file path)."""
        for field_path, expected_value in expected_fields.items():
            # Support nested field paths like "metadata.folder"
            field_parts = field_path.split('.')
            actual_value = data
            
            try:
                for part in field_parts:
                    actual_value = actual_value[part]
            except (KeyError, TypeError):
                return False
            
            # Handle different comparison types
            if isinstance(expected_value, dict):
                # If expected_value is a dict, it might contain comparison operators
                if "equals" in expected_value:
                    if str(actual_value).lower() != str(expected_value["equals"]).lower():
                        return False
                elif "contains" in expected_value:
                    if isinstance(actual_value, str):
                        if expected_value["contains"].lower() not in actual_value.lower():
                            return False
                    else:
                        if expected_value["contains"] not in str(actual_value):
                            return False
                elif "in" in expected_value:
                    # Check if actual_value is in the list
                    if actual_value not in expected_value["in"]:
                        return False
                else:
                    # Recursive check for nested dicts
                    if not isinstance(actual_value, dict):
                        return False
                    if not self._check_json_fields_data(actual_value, expected_value):
                        return False
            elif isinstance(expected_value, list):
                # Check if any of the expected values match
                if actual_value not in expected_value:
                    return False
            else:
                # Direct string comparison (case-insensitive for strings)
                if isinstance(actual_value, str) and isinstance(expected_value, str):
                    if actual_value.lower() != expected_value.lower():
                        return False
                else:
                    if actual_value != expected_value:
                        return False
        
        return True
    
    def _check_json_fields(self, file_path: str, expected_fields: Dict[str, Any]) -> bool:
        """Check if JSON file contains expected field values."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self._check_json_fields_data(data, expected_fields)
        except json.JSONDecodeError as e:
            debug_info(f"File {file_path} is not valid JSON")
            debug_print_exception(e, context=f"Parsing JSON file {file_path}", include_traceback=True)
            return False
        except Exception as e:
            debug_info(f"Could not validate JSON fields in {file_path}")
            debug_print_exception(e, context=f"Validating JSON fields in {file_path}", include_traceback=True)
            return False


class SemanticJudgeValidator(TestValidator):
    """
    Validates using an LLM-based semantic judge.
    
    The judge checks if the content contains any information matching a query description.
    
    Config options:
    - query: String describing what information should be present in the content
    - check_target: What to check. Supports:
        - "agent_response" (default) - check agent's response text
        - "outbox.email.to.{email_address}.{field}" - check email sent to specific address
    """
    
    def __init__(self, config: Dict[str, Any], test_config: Optional[Dict[str, Any]] = None):
        super().__init__(config, test_config)
        self._query = config.get("query", "")
        self._check_target = config.get("check_target", "agent_response")
        
        # Load semantic judge config
        try:
            global_config = load_config()
            judge_config = global_config.get("benchmark", {}).get("semantic_judge", {})
            self._judge_model = judge_config.get("model_name", "gpt-4o")  # Uses benchmark.semantic_judge.model_name
            self._judge_temperature = judge_config.get("temperature", 0.0)
            # Get seed from config for determinism
            self._judge_seed = global_config.get("seed", 42)
        except Exception as e:
            debug_debug("Could not load semantic judge config, using defaults")
            debug_print_exception(e, context="Loading semantic judge config", include_traceback=True)
            self._judge_model = "gpt-4o"
            self._judge_temperature = 0.0
            self._judge_seed = 42
    
    def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
        """Validate using semantic judge."""
        if not self._query:
            return True  # No query to check
        
        # Extract content to validate based on check_target
        content_to_check = self._extract_content(agent_response, traces)
        
        if not content_to_check:
            return False
        
        # Call semantic judge
        try:
            result = self._call_semantic_judge(content_to_check, self._query)
            return result
        except Exception as e:
            debug_info("Semantic judge validation failed")
            debug_print_exception(e, context="Semantic judge validation", include_traceback=True)
            return False
    
    def score(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> float:
        """Score is same as validate for semantic judge (binary)."""
        return 1.0 if self.validate(agent_response, session_id, traces) else 0.0
    
    def _extract_content(self, agent_response: str, traces: List[Dict] = None) -> str:
        """Extract the content to validate based on check_target configuration."""
        if self._check_target == "agent_response":
            return agent_response
        
        # Handle outbox email checks
        if self._check_target.startswith("outbox.email.to."):
            return self._extract_outbox_content(self._check_target)
        
        return ""
    
    def _get_email_timestamp(self, email_data: Dict[str, Any]) -> float:
        """Extract timestamp from email data, returning 0.0 if parsing fails."""
        ts = email_data.get('sent_ts') or email_data.get('created_ts', '')
        if not ts:
            return 0.0
        try:
            return datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
        except Exception:
            return 0.0
    
    def _extract_outbox_content(self, check_target: str) -> str:
        """
        Extract content from outbox emails.
        
        Supported format:
        - "outbox.email.to.{email_address}.{field}" - field value from email sent to specific address
        """
        if not self.test_config:
            return ""
        
        try:
            # Get outbox directory from test config
            outbox_dir = Path(self.test_config.get("data", {}).get("outbox_dir", "data/agent/outbox"))
            
            if not outbox_dir.exists():
                return ""
            
            parts = check_target.split(".")
            # Format: outbox.email.to.{email_with_dots}.{field}
            if len(parts) < 5 or parts[1] != "email" or parts[2] != "to":
                return ""
            
            # Join parts 3 to -1 to reconstruct email address (handles dots in email)
            target_email = ".".join(parts[3:-1])
            target_field = parts[-1]
            
            # Get all emails and find one sent to target_email
            email_files = list(outbox_dir.glob("*.json"))
            matching_emails = []
            
            for email_file in email_files:
                try:
                    with open(email_file, 'r', encoding='utf-8') as f:
                        email = json.load(f)
                    if email.get('to', '').lower() == target_email.lower():
                        matching_emails.append(email)
                except Exception:
                    continue
            
            if not matching_emails:
                return ""
            
            # Get most recent matching email
            matching_emails.sort(key=self._get_email_timestamp, reverse=True)
            latest_email = matching_emails[0]
            
            # Extract target field
            content = latest_email.get(target_field, latest_email.get('body', latest_email.get('body_plain', '')))
            return str(content) if content else ""
            
        except Exception as e:
            debug_info("Error extracting outbox content")
            debug_print_exception(e, context="Extracting outbox content", include_traceback=True)
            return ""
    
    def _call_semantic_judge(self, content: str, query: str) -> bool:
        """Call LLM API to judge if content contains information matching the query."""
        result_text = ""
        try:
            provider = detect_provider(self._judge_model)
            
            # Extract facts for better matching
            def extract_facts(text: str) -> Dict[str, Any]:
                try:
                    # Numbers (integers and decimals), with optional commas
                    numbers = re.findall(r"(?<![\w\d])(?:\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+)(?![\w\d])", text)
                    # Percentages
                    percents = re.findall(r"\b\d{1,3}(?:\.\d+)?\s*%\b", text)
                    # Currency amounts ($, USD)
                    currency = re.findall(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\bUSD\s*\d+(?:\.\d+)?\b", text, flags=re.IGNORECASE)
                    # KPI-like tokens (EPS, revenue, net income, budget, discount)
                    kpis = re.findall(r"\b(EPS|Revenue|Net\s+Income|Budget|Discount|Price|Pricing|Spend|Valuation)\b", text, flags=re.IGNORECASE)
                    # Normalize whitespace and case for matching
                    return {
                        "numbers": numbers,
                        "percentages": percents,
                        "currency": currency,
                        "kpis": list({k.lower() for k in kpis}),
                    }
                except Exception as e:
                    debug_debug("Could not extract facts, returning empty fact dict")
                    debug_print_exception(e, context="Extracting facts from content", include_traceback=True)
                    return {"numbers": [], "percentages": [], "currency": [], "kpis": []}
            
            query_facts = extract_facts(query)
            content_facts = extract_facts(content)
            
            # Standard fact/figure matching evaluation
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a rigorous semantic judge for fact/figure overlap. "
                        "Given a query describing specific information that should be present, and a candidate content, "
                        "decide if the content contains the information described in the query, including semantically equivalent statements. "
                        "Facts include qualitative statements such as roles, titles, relationships, events, dates, entity names, and declarative claims. "
                        "Be tolerant to formatting variants (e.g., 1.24B vs $1,240,000,000; 4.5% vs four point five percent; $1.91 EPS vs EPS 1.91). "
                        "Consider unit/currency normalization, commas, abbreviations (M, B), paraphrases, and synonymous phrasing. "
                        "Prefer high recall: if in doubt and a specific fact or figure plausibly matches, return true. "
                        "Always respond with valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Query (information that should be present):\n{query}\n\n"
                        f"Content to check (agent's response):\n{content}\n\n"
                        f"Extracted query facts (numbers/percentages/currency/KPIs):\n{json.dumps(query_facts, ensure_ascii=False)}\n\n"
                        f"Extracted content facts (numbers/percentages/currency/KPIs):\n{json.dumps(content_facts, ensure_ascii=False)}\n\n"
                        "Task: Determine if the content contains the information described in the query, including semantically equivalent (paraphrased) facts and numerically equivalent statements. "
                        "Consider: qualitative facts (roles, titles, relationships, events, dates, named entities), number formatting, thousands separators, currency symbols, spelled-out numbers, abbreviations (M, B), ratios, EPS, and paraphrases. "
                        "Be generous when a specific fact or figure clearly corresponds (e.g., '1.24 Billion' ~= '$1,240,000,000'; 'EPS beat consensus' ~= 'EPS above Street').\n\n"
                        "Output strict JSON with this exact shape:\n"
                        "{\n"
                        "  \"contains_information\": true/false,\n"
                        "  \"matched_items\": [\"list a few matched facts (normalized)\"],\n"
                        "  \"explanation\": \"very brief justification\"\n"
                        "}"
                    )
                }
            ]
            
            # Some models (e.g., gpt-5 family) don't support temperature — skip it proactively
            model_lower = str(self._judge_model).lower()
            is_gpt5_family = model_lower.startswith("gpt-5") or "gpt-5" in model_lower
            
            # Prepare parameters
            temperature = None if is_gpt5_family else self._judge_temperature
            response_format = {"type": "json_object"} if provider == "openai" else None
            
            # Call unified LLM function
            try:
                response = call_llm_chat_completion(
                    model=self._judge_model,
                    messages=messages,
                    temperature=temperature,
                    response_format=response_format,
                    max_tokens=2000,
                    max_output_tokens=2000,  # For Gemini
                    seed=self._judge_seed
                )
            except Exception as e:
                msg = str(e).lower()
                if "temperature" in msg and ("unsupported" in msg or "does not support" in msg):
                    # Retry without temperature parameter
                    response = call_llm_chat_completion(
                        model=self._judge_model,
                        messages=messages,
                        response_format=response_format,
                        max_tokens=2000,
                        max_output_tokens=2000  # For Gemini
                    )
                else:
                    raise
            
            # Parse response
            if not response or not response.choices or not response.choices[0].message:
                return False
            
            result_text = response.choices[0].message.content
            if not result_text:
                return False
            
            result_text = result_text.strip()
            
            # Parse JSON response
            try:
                result_json = json.loads(result_text)
                return bool(result_json.get("contains_information", False))
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from response
                if result_text:
                    try:
                        json_match = re.search(r'\{"contains_information":\s*(true|false)', result_text, re.IGNORECASE)
                        if json_match:
                            return json_match.group(1).lower() == "true"
                        # Last resort: check for "true" in response
                        result_lower = result_text.lower()
                        return "true" in result_lower and "false" not in result_lower[:20]
                    except Exception:
                        pass
                return False
        
        except Exception as e:
            debug_info("Semantic judge API error")
            debug_print_exception(e, context="Calling semantic judge API", include_traceback=True)
            return False


class CrossStepSemanticJudgeValidator(TestValidator):
    """
    Validates by comparing information from a previous step with the current step's response.
    
    This validator extracts specific information from a reference step and uses semantic judge
    to verify that the current step's response correctly recalls that information.
    
    This validator is generic and can be used for any type of information recall validation:
    - Colors, preferences, choices (e.g., "colors for walls, curtains, bed, pillows")
    - Budgets, amounts, numbers (e.g., "budget amount for the project")
    - Names, dates, locations (e.g., "meeting date and location")
    - Decisions, preferences (e.g., "restaurant choice and reservation time")
    - Any structured information that needs to be recalled accurately
    
    Config options:
    - reference_step: Step number (1-indexed) to extract information from
    - extract_focus: What information to extract (e.g., "colors for walls, curtains, bed, pillows",
      "budget amount", "meeting date and time", "restaurant name and reservation details")
    - compare_focus: What to compare in the current response (defaults to extract_focus).
      Use this if you want to extract one thing but compare something slightly different.
    - comparison_query: Optional explicit query for semantic judge comparison.
      If provided, this overrides the auto-generated comparison query.
      Use this for fine-grained control over what to verify.
    - check_target: What to check in current step (default: "agent_response").
      Supports: "agent_response", "tool.{tool_name}.{param}", "outbox.latest.{field}", etc.
    
    Examples:
    
    1. Colors and mappings:
       {
         "type": "cross_step_semantic_judge",
         "reference_step": 1,
         "extract_focus": "colors for walls, curtains, bed, and pillows with their specific color assignments",
         "compare_focus": "colors for walls, curtains, bed, and pillows"
       }
    
    2. Budget/amounts:
       {
         "type": "cross_step_semantic_judge",
         "reference_step": 2,
         "extract_focus": "budget amount for the redecorating project",
         "comparison_query": "The agent's response should correctly recall the exact budget amount that was previously suggested. Verify that the budget amount matches (allowing for formatting variations like $1,000 vs 1000 vs one thousand dollars)."
       }
    
    3. Meeting details:
       {
         "type": "cross_step_semantic_judge",
         "reference_step": 3,
         "extract_focus": "meeting date, time, and location",
         "compare_focus": "meeting details"
       }
    
    4. Restaurant reservation:
       {
         "type": "cross_step_semantic_judge",
         "reference_step": 1,
         "extract_focus": "restaurant name, reservation date, and number of guests",
         "comparison_query": "Verify that the agent correctly recalls the restaurant name, reservation date, and number of guests that were previously discussed."
       }
    
    5. Custom comparison with tool output:
       {
         "type": "cross_step_semantic_judge",
         "reference_step": 2,
         "extract_focus": "email recipient and subject line",
         "check_target": "tool.draft_email.body",
         "comparison_query": "Verify that the email body mentions the correct recipient and subject that were discussed in step 2."
       }
    
    The validator stores extracted information in test_config["_cross_step_data"] for debugging.
    """
    
    def __init__(self, config: Dict[str, Any], test_config: Optional[Dict[str, Any]] = None):
        super().__init__(config, test_config)
        self._reference_step = config.get("reference_step")
        self._extract_focus = config.get("extract_focus", "")
        self._compare_focus = config.get("compare_focus", self._extract_focus)
        self._comparison_query = config.get("comparison_query", None)
        self._check_target = config.get("check_target", "agent_response")
        
        # Initialize cross-step data storage in test_config if needed
        if test_config is not None and "_cross_step_data" not in test_config:
            test_config["_cross_step_data"] = {}
        
        # Create a SemanticJudgeValidator instance to reuse its extraction and judge methods
        self._semantic_judge_validator = SemanticJudgeValidator(
            {"query": "", "check_target": self._check_target},
            test_config
        )
    
    def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
        """Validate by comparing with information from reference step."""
        if not self._reference_step:
            print("Warning: CrossStepSemanticJudgeValidator requires reference_step")
            return False
        
        if not self.test_config:
            return False
        
        # Find reference step result
        step_results = self.test_config.get("_step_results", [])
        reference_result = next(
            (sr for sr in step_results if sr.get("step") == self._reference_step),
            None
        )
        
        if not reference_result:
            debug_info(f"Could not find step {self._reference_step} in step results")
            return False
        
        # Extract information from reference step
        reference_response = reference_result.get("agent_response", "")
        if not reference_response:
            print(f"Warning: Step {self._reference_step} has no agent_response")
            return False
        
        # Extract the specific information we need
        extracted_info = self._extract_information(reference_response, self._extract_focus)
        if not extracted_info:
            debug_info(f"Could not extract {self._extract_focus} from step {self._reference_step}")
            return False
        
        # Store extracted information for debugging
        if "_cross_step_data" in self.test_config:
            key = f"step_{self._reference_step}_{self._extract_focus}"
            self.test_config["_cross_step_data"][key] = extracted_info
        
        # Extract content to validate from current step
        current_content = self._extract_content(agent_response, traces)
        if not current_content:
            return False
        
        # Pre-check: If the agent explicitly states it doesn't have the information saved,
        # fail immediately even if the correct answer appears later in suggestions
        if self._has_explicit_failure_indicators(current_content):
            debug_info("Agent explicitly stated it doesn't have the information saved - validation fails")
            return False
        
        # Build comparison query
        if self._comparison_query:
            query = (
                f"{self._comparison_query}\n\n"
                f"Previously discussed information to verify:\n{extracted_info}\n\n"
                f"Compare the current response against the above information."
            )
        else:
            query = self._build_comparison_query(extracted_info, self._compare_focus)
        
        # Use semantic judge to compare
        try:
            return self._call_semantic_judge(current_content, query)
        except Exception as e:
            debug_info("Cross-step semantic judge validation failed")
            debug_print_exception(e, context="Cross-step semantic judge validation", include_traceback=True)
            return False
    
    def _extract_information(self, text: str, focus: str) -> str:
        """Extract specific information from text using LLM."""
        try:
            judge_model = self._semantic_judge_validator._judge_model
            judge_temperature = self._semantic_judge_validator._judge_temperature
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an information extraction assistant. "
                        "Extract the specific information requested from the given text. "
                        "Return only the extracted information, be precise and complete. "
                        "If the information is not found, return 'NOT_FOUND'."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Extract the following information from the text:\n"
                        f"Focus: {focus}\n\n"
                        f"Text:\n{text}\n\n"
                        f"Extract and return only the relevant information requested in the focus. "
                        f"Be precise and complete. If the information involves mappings or relationships "
                        f"(e.g., items to values, entities to attributes), preserve those relationships. "
                        f"If the information involves numbers or amounts, extract the exact values. "
                        f"Return the information in a clear, structured format."
                    )
                }
            ]
            
            response = call_llm_chat_completion(
                model=judge_model,
                messages=messages,
                temperature=judge_temperature,
                max_tokens=500,
                max_output_tokens=500
            )
            
            if not response or not response.choices or not response.choices[0].message:
                return ""
            
            extracted = response.choices[0].message.content.strip()
            return "" if extracted == "NOT_FOUND" or not extracted else extracted
        except Exception as e:
            debug_info("Error extracting information")
            debug_print_exception(e, context="Extracting information from response", include_traceback=True)
            return ""
    
    def _build_comparison_query(self, extracted_info: str, compare_focus: str) -> str:
        """Build a query for semantic judge comparison."""
        return (
            f"The agent's response should correctly recall the following information "
            f"that was previously discussed:\n\n"
            f"Focus: {compare_focus}\n\n"
            f"Previously discussed information:\n{extracted_info}\n\n"
            f"Verify that the current response correctly matches or recalls this information. "
            f"If the information involves mappings or relationships (e.g., items to values, entities to attributes), "
            f"ensure the correct mappings are preserved. "
            f"If the information involves numbers or amounts, ensure the values match (allowing for formatting variations). "
            f"Be tolerant to paraphrasing and formatting differences, but strict on factual correctness."
        )
    
    def _has_explicit_failure_indicators(self, content: str) -> bool:
        """
        Check if the agent's response explicitly states it doesn't have the information saved.
        
        This catches cases where the agent says it can't recall, doesn't have the information,
        or searched but didn't find it - even if it then suggests the correct answer.
        """
        if not content:
            return False
        
        content_lower = content.lower()
        
        # Patterns that indicate the agent doesn't have the information saved
        failure_patterns = [
            # Direct statements
            r"i don't have (that|this|it) (saved|stored|remembered|recorded)",
            r"i can't (recall|remember|find|retrieve)",
            r"i couldn't (recall|remember|find|retrieve)",
            r"i do not have (that|this|it) (saved|stored|remembered|recorded)",
            r"i cannot (recall|remember|find|retrieve)",
            
            # Search failure patterns
            r"(searched|looked|checked).*but (didn't|did not|couldn't|could not) find",
            r"(searched|looked|checked).*and (didn't|did not|couldn't|could not) find",
            r"didn't find (any|a|the).*that (clearly|explicitly|specifically)",
            r"did not find (any|a|the).*that (clearly|explicitly|specifically)",
            r"couldn't find (any|a|the).*that (clearly|explicitly|specifically)",
            r"could not find (any|a|the).*that (clearly|explicitly|specifically)",
            r"no (email|message|record|document|note).*that (clearly|explicitly|specifically|records)",
            
            # Memory/recall failure
            r"don't (have|recall|remember) (that|this|it) (saved|stored|in memory)",
            r"do not (have|recall|remember) (that|this|it) (saved|stored|in memory)",
            r"can't (recall|remember) (that|this|it)",
            r"cannot (recall|remember) (that|this|it)",
            
            # Information not available
            r"i don't (have|know|see) (that|this|it) (information|data|record)",
            r"i do not (have|know|see) (that|this|it) (information|data|record)",
            r"i can't (access|retrieve|find) (that|this|it) (information|data|record)",
            r"i cannot (access|retrieve|find) (that|this|it) (information|data|record)",
            
            # Not found in searches
            r"not (found|available|stored|saved) (in|on|at)",
            r"no record (of|for)",
        ]
        
        for pattern in failure_patterns:
            if re.search(pattern, content_lower):
                return True
        
        return False
    
    def _extract_content(self, agent_response: str, traces: List[Dict] = None) -> str:
        """Extract the content to validate based on check_target configuration."""
        # Reuse SemanticJudgeValidator's extraction method
        return self._semantic_judge_validator._extract_content(agent_response, traces)
    
    def _call_semantic_judge(self, content: str, query: str) -> bool:
        """Call semantic judge for cross-step validation - reuse SemanticJudgeValidator's method."""
        # The query already contains all context needed, so we can directly use SemanticJudgeValidator's method
        return self._semantic_judge_validator._call_semantic_judge(content, query)
    
    def score(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> float:
        """Score is same as validate for cross-step semantic judge (binary)."""
        return 1.0 if self.validate(agent_response, session_id, traces) else 0.0


class CompositeValidator(TestValidator):
    """
    Combines multiple validators with logical operators.
    
    Config options:
    - validators: List of validator configs
    - operator: "AND" or "OR" (default: "AND")
    """
    
    def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None, prefix: str = "", trace_file: Optional[str] = None) -> bool:
        """
        Validate with optional printing and trace logging.
        
        Args:
            agent_response: Agent response text
            session_id: Session identifier
            traces: Trace events
            prefix: Prefix for nested validators (for indentation), empty string means no printing
            trace_file: Optional trace file path for logging validator results
            
        Returns:
            Whether validation passed
        """
        validators = self.config.get("validators", [])
        operator = self.config.get("operator", "AND").upper()
        
        if not validators:
            return True
        
        results = []
        validator_results: List[Dict[str, Any]] = []  # Store results for trace logging
        print_results = bool(prefix)  # Print if prefix is provided
        
        for i, validator_config in enumerate(validators):
            validator_type = validator_config.get("type")
            validator = create_validator(validator_config, self.test_config)
            if not validator:
                continue
            
            result = validator.validate(agent_response, session_id, traces)
            results.append(result)
            
            # Handle nested composite validators
            if validator_type == "composite" and isinstance(validator, CompositeValidator):
                composite_validator = validator  # Type narrowing for type checker
                if print_results:
                    nested_prefix = prefix + "  "
                    result = composite_validator.validate(agent_response, session_id, traces, prefix=nested_prefix, trace_file=trace_file)
                    results[-1] = result
                elif trace_file:
                    # Still need to validate nested for trace logging
                    nested_result = composite_validator.validate(agent_response, session_id, traces, prefix="", trace_file=trace_file)
                    results[-1] = nested_result
            
            # Print result if requested
            if print_results:
                validator_name = self._get_validator_display_name(validator_config, validator_type)
                print_validator_result(validator_type, f"{prefix}{validator_name}", result)
            
            # Store result for trace logging
            if trace_file:
                validator_name = self._get_validator_display_name(validator_config, validator_type)
                validator_results.append({
                    "type": validator_type,
                    "name": validator_name,
                    "passed": result,
                    "config": validator_config
                })
        
        # Log validator results to trace file if provided
        if trace_file and validator_results:
            try:
                append_trace_event(
                    trace_file,
                    "validator_results",
                    session_id,
                    {
                        "operator": operator,
                        "overall_passed": all(results) if operator == "AND" else any(results),
                        "validators": validator_results
                    }
                )
            except Exception as e:
                debug_info("Could not log validator results to trace")
                debug_print_exception(e, context="Logging validator results to trace", include_traceback=True)
        
        return any(results) if operator == "OR" else all(results)
    
    def validate_with_print(self, agent_response: str, session_id: str, traces: List[Dict] = None, prefix: str = "", trace_file: Optional[str] = None) -> bool:
        """Validate and print individual validator results (convenience method)."""
        return self.validate(agent_response, session_id, traces, prefix, trace_file)
    
    def _get_validator_display_name(self, config: Dict[str, Any], validator_type: str) -> str:
        """Generate a display name for a validator."""
        if validator_type == "tool_call":
            tools = config.get("required_tools", [])
            return f"Tool call: {', '.join(tools)}" if tools else "Tool call"
        elif validator_type == "semantic_judge":
            target = config.get("check_target", "agent_response")
            if target.startswith("tool.") and len(target.split(".")) >= 3:
                parts = target.split(".")
                return f"Semantic judge ({parts[1]}.{parts[2]})"
            return "Semantic judge"
        elif validator_type == "keyword":
            keywords = config.get("keywords", [])
            if keywords:
                display = ', '.join(keywords[:3])
                return f"Keywords: {display}{'...' if len(keywords) > 3 else ''}"
            return "Keywords"
        elif validator_type == "composite":
            return f"Composite ({config.get('operator', 'AND')})"
        else:
            return f"{validator_type} validator"

    def score(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> float:
        validators = self.config.get("validators", [])
        if not validators:
            return 1.0
        scores: List[float] = []
        for validator_config in validators:
            validator = create_validator(validator_config, self.test_config)
            if validator:
                try:
                    scores.append(float(validator.score(agent_response, session_id, traces)))
                except Exception as e:
                    debug_debug("Could not compute validator score, using 0.0")
                    debug_print_exception(e, context="Computing validator score", include_traceback=True)
                    scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0


def create_validator(validator_config: Dict[str, Any], test_config: Optional[Dict[str, Any]] = None) -> TestValidator:
    """
    Create a validator instance from configuration.
    
    Args:
        validator_config: Dictionary containing validator configuration
        test_config: Test-specific configuration (for path resolution)
        
    Returns:
        TestValidator instance
    """
    validator_type = validator_config.get("type", "keyword")
    
    if validator_type == "keyword":
        return KeywordValidator(validator_config)
    elif validator_type == "tool_call":
        return ToolCallValidator(validator_config)
    elif validator_type == "file_system":
        return FileSystemValidator(validator_config, test_config)
    elif validator_type == "semantic_judge":
        return SemanticJudgeValidator(validator_config, test_config)
    elif validator_type == "cross_step_semantic_judge":
        return CrossStepSemanticJudgeValidator(validator_config, test_config)
    elif validator_type == "composite":
        return CompositeValidator(validator_config, test_config)





