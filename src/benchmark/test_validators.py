"""
Generalized test validation functions for the Email Agent test bench.

This module provides flexible validation functions that can be configured
via parameters in the test JSON files.
"""

import json
import re
import os
import fnmatch
from pathlib import Path
from typing import List, Dict, Any, Union, Optional


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
            # For absence check, return False on first match found
            for keyword in self._keywords_normalized:
                if keyword in text:
                    return False
            return True
        else:
            # For presence check, use optimized logic
            found_count = 0
            
            if self._min_required is not None:
                # Count until we reach the minimum required
                for keyword in self._keywords_normalized:
                    if keyword in text:
                        found_count += 1
                        if found_count >= self._min_required:
                            return True
                return False
            elif self._require_all:
                # Count all matches
                for keyword in self._keywords_normalized:
                    if keyword in text:
                        found_count += 1
                return found_count == len(self._keywords)
            else:
                # Return True on first match found
                for keyword in self._keywords_normalized:
                    if keyword in text:
                        return True
                return False


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
    
    def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
        if not traces or not self._required_tools:
            return len(self._required_tools) == 0  # No tools to check
        
        # Count matching tools and collect their inputs for param validation
        called_tools_count = 0
        tool_inputs_for_params = {}
        
        for trace in traces:
            if trace.get("event_type") == "tool_call":
                payload = trace.get("payload", {})
                tool_name = payload.get("tool_name", "")
                
                if tool_name in self._required_tools:
                    called_tools_count += 1
                    tool_inputs = payload.get("inputs", {})
                    tool_inputs_for_params[tool_name] = tool_inputs
        
        # Validate tool count with early exit
        if self._min_tools is not None:
            if called_tools_count < self._min_tools:
                return False
        elif self._require_all_tools:
            if called_tools_count < len(self._required_tools):
                return False
        
        # Validate tool parameters if requested
        if self._check_params and self._tool_params:
            for tool_name, actual_inputs in tool_inputs_for_params.items():
                if tool_name in self._tool_params:
                    expected_params = self._tool_params[tool_name]
                    normalized_params = self._normalized_tool_params[tool_name]
                    
                    for param_name, expected_value in expected_params.items():
                        # Skip validation-only parameters (not actual tool parameters)
                        if param_name == 'min_required':
                            continue
                        
                        if param_name not in actual_inputs:
                            return False
                        
                        actual_value = actual_inputs[param_name]
                        
                        # Handle different types of parameter validation
                        if isinstance(expected_value, str):
                            # String matching (case-insensitive by default)
                            if normalized_params[param_name] not in str(actual_value).lower():
                                return False
                        elif isinstance(expected_value, list):
                            # List of keywords - check how many are present in the parameter
                            actual_lower = str(actual_value).lower()
                            normalized_keywords = normalized_params[param_name]
                            
                            # Check if min_required is specified for this parameter
                            min_required = expected_params.get('min_required')
                            if min_required is not None:
                                # Count how many keywords are found
                                found_count = 0
                                for keyword in normalized_keywords:
                                    if keyword in actual_lower:
                                        found_count += 1
                                # Need at least min_required keywords
                                if found_count < min_required:
                                    return False
                            else:
                                # Default behavior: check if any keyword is present
                                for keyword in normalized_keywords:
                                    if keyword in actual_lower:
                                        break
                                else:
                                    return False
                        elif isinstance(expected_value, dict):
                            # Nested parameter validation
                            if not self._validate_nested_params(actual_value, expected_value):
                                return False
                        else:
                            # Exact match
                            if actual_value != expected_value:
                                return False
        
        return True

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
        if not traces or not self._required_tools:
            return 1.0 if len(self._required_tools) == 0 else 0.0
        per_tool_scores = []
        # Collect the last-seen inputs per tool (as in validate)
        seen_inputs: Dict[str, Dict[str, Any]] = {}
        called_tools: set[str] = set()
        for trace in traces:
            if trace.get("event_type") == "tool_call":
                payload = trace.get("payload", {})
                tool_name = payload.get("tool_name", "")
                if tool_name in self._required_tools:
                    called_tools.add(tool_name)
                    seen_inputs[tool_name] = payload.get("inputs", {})
        num_required = len(self._required_tools) if len(self._required_tools) > 0 else 1
        for tool_name in self._required_tools:
            if tool_name not in called_tools:
                per_tool_scores.append(0.0)
                continue
            # Called
            if not self._check_params or tool_name not in self._tool_params:
                per_tool_scores.append(1.0)
                continue
            expected_params = self._tool_params[tool_name]
            normalized_params = self._normalized_tool_params.get(tool_name, {})
            actual_inputs = seen_inputs.get(tool_name, {})
            # Determine if params fully match according to validate logic
            params_ok = True
            for param_name, expected_value in expected_params.items():
                # Skip validation-only parameters (not actual tool parameters)
                if param_name == 'min_required':
                    continue
                
                if param_name not in actual_inputs:
                    params_ok = False
                    break
                actual_value = actual_inputs[param_name]
                if isinstance(expected_value, str):
                    if normalized_params.get(param_name, str(expected_value).lower()) not in str(actual_value).lower():
                        params_ok = False
                        break
                elif isinstance(expected_value, list):
                    actual_lower = str(actual_value).lower()
                    normalized_keywords = normalized_params.get(param_name, [])
                    
                    # Check if min_required is specified for this parameter
                    min_required = expected_params.get('min_required')
                    if min_required is not None:
                        # Count how many keywords are found
                        found_count = sum(1 for kw in normalized_keywords if kw in actual_lower)
                        if found_count < min_required:
                            params_ok = False
                            break
                    else:
                        # Default behavior: check if any keyword is present
                        if not any(kw in actual_lower for kw in normalized_keywords):
                            params_ok = False
                            break
                elif isinstance(expected_value, dict):
                    if not self._validate_nested_params(actual_value, expected_value):
                        params_ok = False
                        break
                else:
                    if actual_value != expected_value:
                        params_ok = False
                        break
            per_tool_scores.append(1.0 if params_ok else 0.5)
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
    Validates that specific files were created or modified in the filesystem.
    
    Config options:
    - check_files: List of file patterns to check (supports glob patterns)
    - file_content: Dict mapping file patterns to expected content keywords
    - require_all_files: If True, all files must exist
    - min_files: Minimum number of files that must exist
    - json_fields: Dict mapping file patterns to expected JSON field values (for JSON files)
    - require_all_subjects: List of subject strings that must appear in at least one file
      (useful for checking that multiple emails were forwarded)
    - min_keywords: Minimum number of keywords from file_content list that must be found (default: 2)
    """
    
    def __init__(self, config: Dict[str, Any], test_config: Optional[Dict[str, Any]] = None):
        super().__init__(config, test_config)
        # Pre-resolve paths for performance
        self._check_files = config.get("check_files", [])
        self._file_content = config.get("file_content", {})
        self._json_fields = config.get("json_fields", {})
        self._require_all_files = config.get("require_all_files", True)
        self._min_files = config.get("min_files", None)
        self._require_all_subjects = config.get("require_all_subjects", [])
        self._min_keywords_global = config.get("min_keywords", 2)  # Default to 2
        
        # Pre-resolve file patterns
        self._resolved_files = [self._resolve_path(pattern) for pattern in self._check_files]
        
        # Pre-normalize content expectations and track min_keywords per pattern
        self._normalized_content = {}
        self._min_keywords_per_pattern = {}
        for pattern, content in self._file_content.items():
            resolved_pattern = self._resolve_path(pattern)
            if isinstance(content, str):
                self._normalized_content[resolved_pattern] = content.lower()
            elif isinstance(content, list):
                self._normalized_content[resolved_pattern] = [kw.lower() for kw in content]
                # Store min_keywords for this pattern (use global default)
                self._min_keywords_per_pattern[resolved_pattern] = self._min_keywords_global
            else:
                self._normalized_content[resolved_pattern] = content
        
        # Pre-resolve JSON field patterns
        self._resolved_json_fields = {}
        for pattern, fields in self._json_fields.items():
            resolved_pattern = self._resolve_path(pattern)
            self._resolved_json_fields[resolved_pattern] = fields
    
    def _resolve_path(self, file_pattern: str) -> str:
        """Resolve file pattern to test environment path if test_config is available."""
        if not self.test_config:
            return file_pattern
        
        # Map common data directory patterns to test environment paths
        data_mappings = {
            "data/outbox": self.test_config["data"]["outbox_dir"],
            "data/drafts": self.test_config["data"]["drafts_dir"],
            "data/mailbox": self.test_config["data"]["mailbox_dir"],
            "data/sessions": self.test_config["data"].get("sessions_dir", "data/interactive_agent/sessions"),
            "data/interactive_agent/outbox": self.test_config["data"]["outbox_dir"],
            "data/interactive_agent/drafts": self.test_config["data"]["drafts_dir"],
            "data/interactive_agent/mailbox": self.test_config["data"]["mailbox_dir"],
            "data/interactive_agent/sessions": self.test_config["data"].get("sessions_dir", "data/interactive_agent/sessions"),
        }
        
        # Replace data directory paths with test environment paths
        resolved_pattern = file_pattern
        for original_path, test_path in data_mappings.items():
            if file_pattern.startswith(original_path):
                resolved_pattern = file_pattern.replace(original_path, test_path)
                break
        
        return resolved_pattern
    
    def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
        if not self._check_files:
            return True  # No files to check
        
        found_files = []
        # Track which pattern found which files for efficient matching later
        pattern_to_files = {}
        
        for resolved_pattern in self._resolved_files:
            pattern_files = []
            # Check if pattern contains glob characters
            if '*' in resolved_pattern or '?' in resolved_pattern:
                # Handle glob pattern - split into directory and pattern parts
                # For patterns like "test_env_xxx/outbox/*.json", we need to extract the directory
                pattern_str = str(resolved_pattern)
                # Find the last path separator before the glob
                last_sep_idx = max(pattern_str.rfind('/'), pattern_str.rfind('\\'))
                if last_sep_idx >= 0:
                    # Split into directory and glob pattern
                    dir_part = pattern_str[:last_sep_idx]
                    glob_part = pattern_str[last_sep_idx + 1:]
                    dir_path = Path(dir_part)
                    if dir_path.exists() and dir_path.is_dir():
                        # Use the directory's glob method
                        for file_path in dir_path.glob(glob_part):
                            if file_path.is_file():
                                file_str = str(file_path)
                                found_files.append(file_str)
                                pattern_files.append(file_str)
                    else:
                        # Directory doesn't exist, try as relative pattern
                        for file_path in Path(".").glob(resolved_pattern):
                            if file_path.is_file():
                                file_str = str(file_path)
                                found_files.append(file_str)
                                pattern_files.append(file_str)
                else:
                    # No separator, treat as relative pattern
                    for file_path in Path(".").glob(resolved_pattern):
                        if file_path.is_file():
                            file_str = str(file_path)
                            found_files.append(file_str)
                            pattern_files.append(file_str)
            else:
                # No glob - check as regular path
                pattern_path = Path(resolved_pattern)
                if pattern_path.is_file():
                    file_str = str(pattern_path)
                    found_files.append(file_str)
                    pattern_files.append(file_str)
                elif pattern_path.is_dir():
                    # If it's a directory, check for any files matching the pattern
                    for file_path in pattern_path.glob("*"):
                        if file_path.is_file():
                            file_str = str(file_path)
                            found_files.append(file_str)
                            pattern_files.append(file_str)
            
            # Store mapping of pattern to files it found
            if pattern_files:
                pattern_to_files[resolved_pattern] = pattern_files
        
        # Remove duplicates from found_files while preserving order
        found_files = list(dict.fromkeys(found_files))
        
        # Validate file count with early exit
        if self._min_files is not None:
            if len(found_files) < self._min_files:
                return False
        elif self._require_all_files:
            if len(found_files) < len(self._check_files):
                return False
        
        # Validate file content if specified
        if self._normalized_content:
            # For file content validation, we need at least one file to match
            content_validation_passed = False
            for resolved_pattern, expected_content in self._normalized_content.items():
                # Get files that match this pattern (either from our mapping or by matching)
                pattern_files = pattern_to_files.get(resolved_pattern, [])
                if not pattern_files:
                    # Fallback: check all found files
                    pattern_files = [f for f in found_files if self._matches_pattern(f, resolved_pattern)]
                
                # Get min_keywords for this pattern
                min_keywords = self._min_keywords_per_pattern.get(resolved_pattern, self._min_keywords_global)
                
                for file_path in pattern_files:
                    if self._check_file_content(file_path, expected_content, min_keywords=min_keywords):
                        content_validation_passed = True
                        break
                if content_validation_passed:
                    break
            
            if not content_validation_passed:
                return False
        
        # Validate JSON fields if specified
        if self._resolved_json_fields:
            # For JSON field validation, we need at least one file to match all requirements
            json_validation_passed = False
            for resolved_pattern, expected_fields in self._resolved_json_fields.items():
                # Get files that match this pattern (either from our mapping or by matching)
                pattern_files = pattern_to_files.get(resolved_pattern, [])
                if not pattern_files:
                    # Fallback: check all found files
                    pattern_files = [f for f in found_files if self._matches_pattern(f, resolved_pattern)]
                
                for file_path in pattern_files:
                    if self._check_json_fields(file_path, expected_fields):
                        json_validation_passed = True
                        break
                if json_validation_passed:
                    break
            
            if not json_validation_passed:
                return False
        
        # Validate that all required subjects are present across files
        if self._require_all_subjects:
            # Check that each required subject appears in at least one file
            found_subjects = set()
            for file_path in found_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        # Check both subject field and body content
                        try:
                            data = json.loads(content)
                            subject = str(data.get('subject', '')).lower()
                            body = str(data.get('body', '') + ' ' + data.get('body_plain', '')).lower()
                            file_content = subject + ' ' + body
                        except:
                            file_content = content
                        
                        # Check if any required subject is in this file
                        for required_subject in self._require_all_subjects:
                            if required_subject.lower() in file_content:
                                found_subjects.add(required_subject)
                except Exception as e:
                    print(f"Warning: Could not check subjects in {file_path}: {e}")
                    continue
            
            # Verify all required subjects were found
            if len(found_subjects) < len(self._require_all_subjects):
                missing = set(self._require_all_subjects) - found_subjects
                print(f"Warning: Missing required subjects in forwarded emails: {missing}")
                return False
        
        return True
    
    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches the pattern."""
        try:
            file_path_obj = Path(file_path)
            pattern_obj = Path(pattern)
            
            # Normalize paths for comparison
            file_str = str(file_path_obj.resolve()) if file_path_obj.exists() else str(file_path_obj)
            pattern_str = str(pattern_obj)
            
            # If pattern contains glob characters, use fnmatch
            if '*' in pattern or '?' in pattern:
                # Try full path match
                if fnmatch.fnmatch(file_str, pattern_str):
                    return True
                # Try relative path match
                try:
                    if fnmatch.fnmatch(str(file_path_obj), pattern_str):
                        return True
                except:
                    pass
                # For patterns like "dir/*.json", check if file is in that directory with matching name
                if '/' in pattern_str or '\\' in pattern_str:
                    pattern_dir = pattern_obj.parent
                    pattern_glob = pattern_obj.name
                    if file_path_obj.parent == pattern_dir or str(file_path_obj.parent) == str(pattern_dir):
                        if fnmatch.fnmatch(file_path_obj.name, pattern_glob):
                            return True
                return False
            else:
                # No glob - use direct comparison
                return (str(file_path_obj) == pattern_str or 
                        str(file_path_obj.resolve()) == str(Path(pattern_str).resolve()) if Path(pattern_str).exists() else False)
        except Exception:
            # Fallback: simple string containment check
            return pattern in file_path
    
    def _check_file_content(self, file_path: str, expected_content: Union[str, List[str]], min_keywords: int = 2) -> bool:
        """Check if file contains expected content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            
            if isinstance(expected_content, str):
                return expected_content.lower() in content
            elif isinstance(expected_content, list):
                # Check if at least min_keywords keywords are present
                found_keywords = sum(1 for keyword in expected_content if keyword.lower() in content)
                return found_keywords >= min_keywords
            
            return False
        except Exception as e:
            print(f"Warning: Could not read file {file_path}: {e}")
            return False
    
    def _check_json_fields(self, file_path: str, expected_fields: Dict[str, Any]) -> bool:
        """Check if JSON file contains expected field values."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
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
                        # Direct comparison
                        if actual_value != expected_value:
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
        except json.JSONDecodeError:
            print(f"Warning: File {file_path} is not valid JSON")
            return False
        except Exception as e:
            print(f"Warning: Could not validate JSON fields in {file_path}: {e}")
            return False


class MemoryValidator(TestValidator):
    """
    Validates that memory was updated correctly by checking the memory file.
    
    Config options:
    - check_keywords: List of keywords that should be present in memory
    - check_absence: List of keywords that should NOT be present in memory
    - memory_file: Path to memory file (default: "data/agent_memory.json")
    - require_all_keywords: If True, all keywords must be present (default: False)
    - min_keywords: Minimum number of keywords that must be present
    """
    
    def __init__(self, config: Dict[str, Any], test_config: Optional[Dict[str, Any]] = None):
        super().__init__(config, test_config)
        # Pre-normalize for performance
        self._check_keywords = config.get("check_keywords", [])
        self._check_absence = config.get("check_absence", [])
        self._memory_file = config.get("memory_file", "data/interactive_agent/agent_memory.json")
        self._require_all_keywords = config.get("require_all_keywords", False)
        self._min_keywords = config.get("min_keywords", None)
        
        # Pre-normalize keywords
        self._keywords_lower = [kw.lower() for kw in self._check_keywords]
        self._absence_lower = [kw.lower() for kw in self._check_absence]
        
        # Resolve memory file path for test environment
        if test_config and "data" in test_config:
            test_memory_file = test_config["data"].get("memory_file", self._memory_file)
            if test_memory_file != self._memory_file:
                self._memory_file = test_memory_file
    
    def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
        try:
            # Read memory file
            with open(self._memory_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            long_term_memory = memory_data.get("long_term", [])
            
            # Check for required keywords with early exit
            if self._check_keywords:
                found_keywords = set()  # Track unique keywords found
                
                for memory_entry in long_term_memory:
                    memory_lower = memory_entry.lower()
                    
                    for keyword in self._keywords_lower:
                        if keyword in memory_lower:
                            found_keywords.add(keyword)
                    
                    # Early exit if we've found enough
                    if self._min_keywords is not None and len(found_keywords) >= self._min_keywords:
                        break
                    elif self._min_keywords is None and not self._require_all_keywords and len(found_keywords) > 0:
                        break
                
                found_count = len(found_keywords)
                
                # Check final counts
                if self._min_keywords is not None:
                    if found_count < self._min_keywords:
                        return False
                elif self._require_all_keywords:
                    if found_count < len(self._check_keywords):
                        return False
                else:
                    if found_count == 0:
                        return False
            
            # Check for absence of keywords with early exit
            if self._check_absence:
                for memory_entry in long_term_memory:
                    memory_lower = memory_entry.lower()
                    for keyword in self._absence_lower:
                        if keyword in memory_lower:
                            return False  # Found forbidden keyword
            
            return True
            
        except Exception as e:
            print(f"Warning: Could not read memory file {self._memory_file}: {e}")
            return False


class Mem0MemoryValidator(TestValidator):
    """
    Validates that mem0 memory was updated correctly by checking the mem0 vectorstore.
    
    Config options:
    - check_keywords: List of keywords that should be present in mem0 memories
    - check_absence: List of keywords that should NOT be present in mem0 memories
    - vectorstore_path: Path to mem0 vectorstore (default: from config)
    - require_all_keywords: If True, all keywords must be present (default: False)
    - min_keywords: Minimum number of keywords that must be present
    - user_id: User ID to filter memories (default: "vince")
    - agent_id: Agent ID to filter memories (default: "email_agent")
    """
    
    def __init__(self, config: Dict[str, Any], test_config: Optional[Dict[str, Any]] = None):
        super().__init__(config, test_config)
        # Pre-normalize for performance
        self._check_keywords = config.get("check_keywords", [])
        self._check_absence = config.get("check_absence", [])
        self._require_all_keywords = config.get("require_all_keywords", False)
        self._min_keywords = config.get("min_keywords", None)
        self._user_id = config.get("user_id", "vince")
        self._agent_id = config.get("agent_id", "email_agent")
        
        # Pre-normalize keywords
        self._keywords_lower = [kw.lower() for kw in self._check_keywords]
        self._absence_lower = [kw.lower() for kw in self._check_absence]
        
        # Resolve vectorstore path from test config
        self._vectorstore_path = None
        if test_config:
            mem0_config = test_config.get("memory", {}).get("mem0_memory", {})
            self._vectorstore_path = mem0_config.get("vectorstore_path")
    
    def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
        try:
            # Import here to avoid circular dependencies
            from agent.backend.mem0_memory_manager import get_mem0_memory_manager
            from agent.utils import load_config
            
            # Get config
            config = load_config()
            mem0_config = config.get("memory", {}).get("mem0_memory", {})
            
            # Use test-specific vectorstore path if provided
            vectorstore_path = self._vectorstore_path or mem0_config.get("vectorstore_path", "data/interactive_agent/mem0_vectorstore")
            
            # Initialize mem0 memory manager
            mem0_manager = get_mem0_memory_manager(
                llm_provider=mem0_config.get("llm_provider", "openai"),
                llm_model=mem0_config.get("llm_model", "gpt-5-mini"),
                llm_temperature=mem0_config.get("llm_temperature", 0.0),
                embedding_provider=mem0_config.get("embedding_provider", "openai"),
                embedding_model=mem0_config.get("embedding_model", "text-embedding-3-small"),
                vector_store_provider=mem0_config.get("vector_store_provider", "faiss"),
                vectorstore_path=vectorstore_path,
                top_k=mem0_config.get("top_k", 3),
                user_id=self._user_id,
                agent_id=self._agent_id,
                force_new=False
            )
            
            # Get all memories from mem0
            memories = mem0_manager.get_all_memories(user_id=self._user_id, agent_id=self._agent_id, limit=1000)
            
            # Extract memory texts
            memory_texts = []
            for memory in memories:
                if isinstance(memory, dict):
                    memory_text = memory.get("memory", "")
                    if memory_text:
                        memory_texts.append(memory_text)
                else:
                    memory_texts.append(str(memory))
            
            # Check for required keywords with early exit
            if self._check_keywords:
                found_keywords = set()  # Track unique keywords found
                
                for memory_text in memory_texts:
                    memory_lower = memory_text.lower()
                    
                    for keyword in self._keywords_lower:
                        if keyword in memory_lower:
                            found_keywords.add(keyword)
                    
                    # Early exit if we've found enough
                    if self._min_keywords is not None and len(found_keywords) >= self._min_keywords:
                        break
                    elif self._min_keywords is None and not self._require_all_keywords and len(found_keywords) > 0:
                        break
                
                found_count = len(found_keywords)
                
                # Check final counts
                if self._min_keywords is not None:
                    if found_count < self._min_keywords:
                        return False
                elif self._require_all_keywords:
                    if found_count < len(self._check_keywords):
                        return False
                else:
                    if found_count == 0:
                        return False
            
            # Check for absence of keywords with early exit
            if self._check_absence:
                for memory_text in memory_texts:
                    memory_lower = memory_text.lower()
                    for keyword in self._absence_lower:
                        if keyword in memory_lower:
                            return False  # Found forbidden keyword
            
            return True
            
        except Exception as e:
            print(f"Warning: Could not read mem0 memory: {e}")
            import traceback
            traceback.print_exc()
            return False


class SemanticJudgeValidator(TestValidator):
    """
    Validates using an LLM-based semantic judge.
    
    The judge checks if the content contains any information matching a query description.
    Returns True even if only a slight bit of information matches.
    
    Config options:
    - query: String describing what information should be present in the content
    - check_target: What to check. Supports:
        - "agent_response" (default) - check agent's response text
        - "tool.{tool_name}.{param}" - check tool call parameter (e.g., "tool.compose_email.body")
        - "outbox.latest.{field}" - check most recent email in outbox (e.g., "outbox.latest.body")
        - "outbox.latest.forward_to_email.{field}" - check most recent forwarded email (e.g., "outbox.latest.forward_to_email.body")
        - "outbox.email.to.{email_address}.{field}" - check email sent to specific address
    - tool_name: (deprecated) Required if check_target is a tool parameter (e.g., "compose_email")
    - tool_param: (deprecated) Required if check_target is a tool parameter (e.g., "body")
    """
    
    def __init__(self, config: Dict[str, Any], test_config: Optional[Dict[str, Any]] = None):
        super().__init__(config, test_config)
        self._query = config.get("query", "")
        self._check_target = config.get("check_target", "agent_response")
        self._tool_name = config.get("tool_name", None)
        self._tool_param = config.get("tool_param", None)
        
        # Load semantic judge config
        try:
            from utils import load_config
            global_config = load_config()
            judge_config = global_config.get("benchmark", {}).get("semantic_judge", {})
            self._judge_model = judge_config.get("model_name", "gpt-4o")  # Uses benchmark.semantic_judge.model_name
            self._judge_temperature = judge_config.get("temperature", 0.0)
        except Exception:
            self._judge_model = "gpt-4o"
            self._judge_temperature = 0.0
    
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
            print(f"Warning: Semantic judge validation failed: {e}")
            return False
    
    def score(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> float:
        """Score is same as validate for semantic judge (binary)."""
        return 1.0 if self.validate(agent_response, session_id, traces) else 0.0
    
    def _extract_content(self, agent_response: str, traces: List[Dict] = None) -> str:
        """Extract the content to validate based on check_target configuration."""
        if self._check_target == "agent_response":
            return agent_response
        
        # Handle outbox email checks
        if self._check_target.startswith("outbox."):
            # Format: "outbox.latest.body" or "outbox.latest.forward_to_email.body"
            # or "outbox.email.to.{email_address}.body"
            return self._extract_outbox_content(self._check_target)
        
        # Handle tool parameter checks
        if self._check_target.startswith("tool."):
            # Format: "tool.compose_email.body"
            parts = self._check_target.split(".")
            if len(parts) >= 3:
                tool_name = parts[1]
                param_name = ".".join(parts[2:])  # Handle nested params
            else:
                return ""
        elif self._tool_name and self._tool_param:
            tool_name = self._tool_name
            param_name = self._tool_param
        else:
            return ""
        
        # Extract from traces
        if not traces:
            return ""
        
        for trace in traces:
            if trace.get("event_type") == "tool_call":
                payload = trace.get("payload", {})
                if payload.get("tool_name") == tool_name:
                    inputs = payload.get("inputs", {})
                    # Handle nested parameter paths like "body"
                    param_value = inputs
                    for part in param_name.split("."):
                        if isinstance(param_value, dict):
                            param_value = param_value.get(part)
                        else:
                            return ""
                    if param_value:
                        return str(param_value)
        
        return ""
    
    def _extract_outbox_content(self, check_target: str) -> str:
        """
        Extract content from outbox emails.
        
        Supported formats:
        - "outbox.latest.body" - body of most recent email in outbox
        - "outbox.latest.body_plain" - body_plain of most recent email
        - "outbox.latest.forward_to_email.body" - body of most recent forwarded email
        - "outbox.email.to.{email_address}.body" - body of email sent to specific address
        """
        if not self.test_config:
            return ""
        
        try:
            # json, Path already imported at module level
            from datetime import datetime
            
            # Get outbox directory from test config
            outbox_dir = Path(self.test_config.get("data", {}).get("outbox_dir", "data/interactive_agent/outbox"))
            
            if not outbox_dir.exists():
                return ""
            
            parts = check_target.split(".")
            if len(parts) < 3:
                return ""
            
            # Find target field (body, body_plain, subject, etc.)
            target_field = parts[-1]  # Last part is the field name
            
            # Handle "outbox.latest.{field}" pattern
            if parts[1] == "latest":
                # Get all email files
                email_files = list(outbox_dir.glob("*.json"))
                if not email_files:
                    return ""
                
                # Filter by tool type if specified (e.g., "outbox.latest.forward_to_email.body")
                if len(parts) >= 4 and parts[2] == "forward_to_email":
                    # Find emails that are forwards (subject starts with "Fwd:" or has forwarded_from field)
                    forwarded_emails = []
                    for email_file in email_files:
                        try:
                            with open(email_file, 'r', encoding='utf-8') as f:
                                email = json.load(f)
                            # Check if it's a forwarded email
                            subject = email.get('subject', '')
                            if subject.lower().startswith('fwd:') or 'forwarded_from' in email:
                                forwarded_emails.append((email_file, email))
                        except:
                            continue
                    
                    if not forwarded_emails:
                        return ""
                    
                    # Sort by timestamp (most recent first)
                    def get_timestamp(email_data):
                        ts = email_data.get('sent_ts') or email_data.get('created_ts', '')
                        try:
                            return datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                        except:
                            return 0.0
                    
                    forwarded_emails.sort(key=lambda x: get_timestamp(x[1]), reverse=True)
                    latest_email = forwarded_emails[0][1]
                    
                    # Extract target field
                    content = latest_email.get(target_field, latest_email.get('body', latest_email.get('body_plain', '')))
                    return str(content) if content else ""
                else:
                    # Get latest email by timestamp
                    emails = []
                    for email_file in email_files:
                        try:
                            with open(email_file, 'r', encoding='utf-8') as f:
                                email = json.load(f)
                            emails.append((email_file, email))
                        except:
                            continue
                    
                    if not emails:
                        return ""
                    
                    # Sort by timestamp
                    def get_timestamp(email_data):
                        ts = email_data.get('sent_ts') or email_data.get('created_ts', '')
                        try:
                            return datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                        except:
                            return 0.0
                    
                    emails.sort(key=lambda x: get_timestamp(x[1]), reverse=True)
                    latest_email = emails[0][1]
                    
                    # Extract target field
                    content = latest_email.get(target_field, latest_email.get('body', latest_email.get('body_plain', '')))
                    return str(content) if content else ""
            
            # Handle "outbox.email.to.{email_address}.{field}" pattern
            # Note: Email addresses contain dots, so we need special parsing
            # Format: outbox.email.to.{email_with_dots}.{field}
            elif parts[1] == "email" and len(parts) >= 4 and parts[2] == "to":
                # The email address is everything from parts[3] up to (but not including) the last part
                # The last part is the target field
                if len(parts) < 5:
                    return ""  # Need at least: outbox.email.to.{email}.{field}
                
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
                            matching_emails.append((email_file, email))
                    except:
                        continue
                
                if not matching_emails:
                    return ""
                
                # Get most recent matching email
                def get_timestamp(email_data):
                    ts = email_data.get('sent_ts') or email_data.get('created_ts', '')
                    try:
                        return datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                    except:
                        return 0.0
                
                matching_emails.sort(key=lambda x: get_timestamp(x[1]), reverse=True)
                latest_email = matching_emails[0][1]
                
                # Extract target field
                content = latest_email.get(target_field, latest_email.get('body', latest_email.get('body_plain', '')))
                return str(content) if content else ""
            
            return ""
            
        except Exception as e:
            print(f"Warning: Error extracting outbox content: {e}")
            return ""
    
    def _call_semantic_judge(self, content: str, query: str) -> bool:
        """Call LLM API to judge if content contains information matching the query."""
        result_text = ""
        try:
            from agent.utils import call_llm_chat_completion, detect_provider
            
            provider = detect_provider(self._judge_model)
            
            # Check if this is a persona memory evaluation (has "correct answer" and "incorrect answers" in query)
            is_persona_eval = "correct answer" in query.lower() and ("incorrect" in query.lower() or "distractor" in query.lower())
            
            # DEBUG: Print detailed information
            print("\n" + "="*80)
            print("🔍 SEMANTIC JUDGE DEBUG - DETAILED LOG")
            print("="*80)
            print(f"Judge Model: {self._judge_model}")
            print(f"Is PersonaMem Evaluation: {is_persona_eval}")
            print(f"\n📝 Content to Evaluate (Agent's Response):")
            print("-"*80)
            print(content[:500] + ("..." if len(content) > 500 else ""))
            print(f"\n📋 Original Query:")
            print("-"*80)
            print(query[:500] + ("..." if len(query) > 500 else ""))
            
            if is_persona_eval:
                # PersonaMem-style multiple choice evaluation
                # Extract correct answer and incorrect answers from query
                import re
                correct_match = re.search(r'Correct answer:\s*(.+?)(?:\n\n|$)', query, re.DOTALL | re.IGNORECASE)
                incorrect_matches = re.findall(r'-\s*(.+?)(?=\n-|\n\n|$)', query.split("Incorrect/distractor answers")[-1] if "Incorrect/distractor answers" in query else "", re.DOTALL)
                
                if correct_match and len(incorrect_matches) >= 1:
                    correct_answer = correct_match.group(1).strip()
                    # Take first 3 incorrect answers if available
                    incorrect_answers = [inc.strip() for inc in incorrect_matches[:3]]
                    
                    # Format as multiple choice options (a, b, c, d) matching PersonaMem format
                    # Option (a) is always the correct answer
                    # Options (b), (c), (d) are incorrect answers
                    all_options = f"(a) {correct_answer}\n"
                    # Use chr(98)='b', chr(99)='c', chr(100)='d'
                    for idx, inc_ans in enumerate(incorrect_answers):
                        option_letter = chr(98 + idx)  # b=98, c=99, d=100
                        all_options += f"({option_letter}) {inc_ans}\n"
                    
                    # DEBUG: Print extracted answers
                    print(f"\n📊 EXTRACTED ANSWERS:")
                    print(f"   Correct Answer (a): {correct_answer[:100]}...")
                    print(f"   Incorrect Answers: {len(incorrect_answers)} found")
                    for idx, inc in enumerate(incorrect_answers):
                        print(f"     ({chr(98 + idx)}) {inc[:100]}...")
                    
                    # Use PersonaMem's EXACT prompt format from inference.py line 57
                    instructions = "Find the most appropriate model response and give your final answer (a), (b), (c), or (d) after the special token <final_answer>."
                    
                    # Format exactly like PersonaMem: question + instructions + all_options
                    # PersonaMem format: question + '\n\n' + instructions + '\n\n' + all_options
                    question = f"Which of the following responses best matches the agent's actual response?\n\nAgent's actual response:\n{content}"
                    
                    messages = [
                        {
                            "role": "user",
                            "content": question + '\n\n' + instructions + '\n\n' + all_options
                        }
                    ]
                    
                    # DEBUG: Print the full prompt
                    print(f"\n📤 FULL PROMPT SENT TO JUDGE:")
                    print("="*80)
                    for i, msg in enumerate(messages, 1):
                        print(f"\nMessage {i} ({msg['role']}):")
                        print("-"*80)
                        print(msg['content'])
                    print("="*80)
                else:
                    # Fallback to original format if parsing fails
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a semantic judge for persona memory evaluation. "
                                "Your task is to evaluate if an agent's response matches the correct answer semantically "
                                "and demonstrates understanding of the persona's characteristics and preferences. "
                                "The response should be semantically equivalent to the correct answer and clearly better than incorrect answers. "
                                "Consider the overall meaning, key concepts, and persona-specific details, not just exact word matches. "
                                "Be generous with semantic equivalence: if the response conveys the same meaning and demonstrates persona understanding, return true. "
                                "Always respond with valid JSON."
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"{query}\n\n"
                                f"Agent's Response:\n{content}\n\n"
                                "Task: Evaluate if the agent's response:\n"
                                "1. Matches or is semantically equivalent to the correct answer\n"
                                "2. Is clearly better than the incorrect answers\n"
                                "3. Demonstrates personalized understanding based on the persona's history\n\n"
                                "Output strict JSON with this exact shape:\n"
                                "{\n"
                                "  \"contains_information\": true/false,\n"
                                "  \"explanation\": \"brief justification of your decision\"\n"
                                "}"
                            )
                        }
                    ]
            else:
                # Standard fact/figure matching evaluation
                # Local helper to extract/normalize numeric facts and figures
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
                    except Exception:
                        return {"numbers": [], "percentages": [], "currency": [], "kpis": []}
                
                query_facts = extract_facts(query)
                content_facts = extract_facts(content)
                
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a rigorous semantic judge for fact/figure overlap. "
                            "Given a query and a candidate content, decide if the content contains any of the query's facts, figures, or semantically equivalent statements. "
                            "Facts include qualitative statements such as roles, titles, relationships, events, dates, entity names, and declarative claims (e.g., 'EPS beat consensus', 'Project Chimera is an M&A codename'). "
                            "Be tolerant to formatting variants (e.g., 1.24B vs $1,240,000,000; 4.5% vs four point five percent; $1.91 EPS vs EPS 1.91). "
                            "Consider unit/currency normalization, commas, abbreviations (M, B), paraphrases, and synonymous phrasing. "
                            "Prefer high recall: if in doubt and a specific fact or figure plausibly matches, return true. "
                            "Always respond with valid JSON."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Query (full):\n{query}\n\n"
                            f"Content to check (full):\n{content}\n\n"
                            f"Extracted query facts (numbers/percentages/currency/KPIs):\n{json.dumps(query_facts, ensure_ascii=False)}\n\n"
                            f"Extracted content facts (numbers/percentages/currency/KPIs):\n{json.dumps(content_facts, ensure_ascii=False)}\n\n"
                            "Task: Determine if the content contains ANY fact or figure from the query, including semantically equivalent (paraphrased) facts and numerically equivalent statements. "
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
            # PersonaMem doesn't use JSON format - it expects free text with (a), (b), (c), (d) answer
            # Only use JSON format for standard fact/figure matching
            response_format = None if is_persona_eval else ({"type": "json_object"} if provider == "openai" else None)
            
            # Call unified LLM function
            try:
                response = call_llm_chat_completion(
                    model=self._judge_model,
                    messages=messages,
                    temperature=temperature,
                    response_format=response_format,
                    max_tokens=2000,
                    max_output_tokens=2000  # For Gemini
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
            
            # DEBUG: Print the raw response
            print(f"\n📥 RAW RESPONSE FROM JUDGE:")
            print("="*80)
            print(result_text)
            print("="*80)
            
            # For PersonaMem-style evaluation, extract the answer choice
            if is_persona_eval:
                # Extract answer from PersonaMem format: look for (a), (b), (c), or (d)
                # PersonaMem format: answer after <final_answer> token or in the response
                answer_match = None
                if "<final_answer>" in result_text:
                    # Extract text after <final_answer>
                    after_token = result_text.split("<final_answer>")[-1].strip()
                    answer_match = re.search(r'\(([a-d])\)', after_token, re.IGNORECASE)
                else:
                    # Try to find (a), (b), (c), or (d) in the response
                    answer_match = re.search(r'\(([a-d])\)', result_text, re.IGNORECASE)
                    if not answer_match:
                        # Try without parentheses
                        answer_match = re.search(r'\b([a-d])\b', result_text, re.IGNORECASE)
                
                if answer_match:
                    predicted_answer = answer_match.group(1).lower()
                    # Correct answer is always (a) in our format
                    correct_answer = "a"
                    result = predicted_answer == correct_answer
                    print(f"\n✅ PARSED RESULT:")
                    print(f"   Extracted Answer: ({predicted_answer})")
                    print(f"   Correct Answer: ({correct_answer})")
                    print(f"   Match: {result}")
                    print("="*80 + "\n")
                    return result
                else:
                    # Also try to extract from <final_answer>e</final_answer> format
                    if "<final_answer>" in result_text:
                        after_token = result_text.split("<final_answer>")[-1].strip()
                        # Try to find single letter a, b, c, d
                        single_letter_match = re.search(r'([a-d])', after_token, re.IGNORECASE)
                        if single_letter_match:
                            predicted_answer = single_letter_match.group(1).lower()
                            correct_answer = "a"
                            result = predicted_answer == correct_answer
                            print(f"\n✅ PARSED RESULT (from <final_answer> token):")
                            print(f"   Extracted Answer: ({predicted_answer})")
                            print(f"   Correct Answer: ({correct_answer})")
                            print(f"   Match: {result}")
                            print("="*80 + "\n")
                            return result
                    
                    # If we get here, couldn't extract answer
                    print(f"\n⚠️ WARNING: Could not extract answer choice from response")
                    print(f"   Attempted to find (a), (b), (c), or (d) but no match found")
                    # Fallback: try JSON parsing
                    try:
                        result_json = json.loads(result_text)
                        # Check if it has contains_information field (fallback format)
                        if "contains_information" in result_json:
                            result = bool(result_json.get("contains_information", False))
                            print(f"   Fallback: Parsed JSON, contains_information = {result}")
                            print("="*80 + "\n")
                            return result
                    except:
                        pass
                    # Last resort: check if response mentions option (a) or seems positive
                    result = "(a)" in result_text or "option a" in result_text.lower() or "answer a" in result_text.lower()
                    print(f"   Last resort: Checking for '(a)' mention = {result}")
                    print("="*80 + "\n")
                    return result
            else:
                # Standard JSON format
                result_json = json.loads(result_text)
                contains_info = result_json.get("contains_information", False)
                print(f"\n✅ PARSED RESULT (Standard Format):")
                print(f"   JSON Response: {json.dumps(result_json, indent=2)}")
                print(f"   contains_information: {contains_info}")
                print("="*80 + "\n")
                return bool(contains_info)
            
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
            print(f"Semantic judge API error: {e}")
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
        if test_config is not None:
            if "_cross_step_data" not in test_config:
                test_config["_cross_step_data"] = {}
        
        # Load semantic judge config (same as SemanticJudgeValidator)
        try:
            from utils import load_config
            global_config = load_config()
            judge_config = global_config.get("benchmark", {}).get("semantic_judge", {})
            self._judge_model = judge_config.get("model_name", "gpt-4o")
            self._judge_temperature = judge_config.get("temperature", 0.0)
        except Exception:
            self._judge_model = "gpt-4o"
            self._judge_temperature = 0.0
    
    def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
        """Validate by comparing with information from reference step."""
        if not self._reference_step:
            print("Warning: CrossStepSemanticJudgeValidator requires reference_step")
            return False
        
        # Get current step number from test_config
        current_step = None
        if self.test_config:
            current_step = self.test_config.get("_current_step_num")
        
        # Get step results from test_config
        step_results = []
        if self.test_config:
            step_results = self.test_config.get("_step_results", [])
        
        # Find reference step result
        reference_result = None
        for step_result in step_results:
            if step_result.get("step") == self._reference_step:
                reference_result = step_result
                break
        
        if not reference_result:
            print(f"Warning: Could not find step {self._reference_step} in step results")
            return False
        
        # Extract information from reference step
        reference_response = reference_result.get("agent_response", "")
        if not reference_response:
            print(f"Warning: Step {self._reference_step} has no agent_response")
            return False
        
        # Extract the specific information we need
        extracted_info = self._extract_information(reference_response, self._extract_focus)
        
        if not extracted_info:
            print(f"Warning: Could not extract {self._extract_focus} from step {self._reference_step}")
            return False
        
        # Store extracted information for debugging
        if self.test_config and "_cross_step_data" in self.test_config:
            key = f"step_{self._reference_step}_{self._extract_focus}"
            self.test_config["_cross_step_data"][key] = extracted_info
        
        # Extract content to validate from current step
        current_content = self._extract_content(agent_response, traces)
        if not current_content:
            return False
        
        # Build comparison query
        if self._comparison_query:
            query = self._comparison_query
        else:
            query = self._build_comparison_query(extracted_info, self._compare_focus)
        
        # Use semantic judge to compare
        try:
            result = self._call_semantic_judge(current_content, query)
            return result
        except Exception as e:
            print(f"Warning: Cross-step semantic judge validation failed: {e}")
            return False
    
    def _extract_information(self, text: str, focus: str) -> str:
        """Extract specific information from text using LLM."""
        try:
            from agent.utils import call_llm_chat_completion, detect_provider
            
            provider = detect_provider(self._judge_model)
            
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
            
            # Call LLM
            response = call_llm_chat_completion(
                model=self._judge_model,
                messages=messages,
                temperature=self._judge_temperature,
                max_tokens=500,
                max_output_tokens=500
            )
            
            if not response or not response.choices or not response.choices[0].message:
                return ""
            
            extracted = response.choices[0].message.content.strip()
            if extracted == "NOT_FOUND" or not extracted:
                return ""
            
            return extracted
        except Exception as e:
            print(f"Warning: Error extracting information: {e}")
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
    
    def _extract_content(self, agent_response: str, traces: List[Dict] = None) -> str:
        """Extract the content to validate based on check_target configuration."""
        if self._check_target == "agent_response":
            return agent_response
        
        # Handle outbox email checks (reuse from SemanticJudgeValidator)
        if self._check_target.startswith("outbox."):
            return self._extract_outbox_content(self._check_target)
        
        # Handle tool parameter checks
        if self._check_target.startswith("tool."):
            parts = self._check_target.split(".")
            if len(parts) >= 3:
                tool_name = parts[1]
                param_name = ".".join(parts[2:])
            else:
                return ""
        else:
            return ""
        
        # Extract from traces
        if not traces:
            return ""
        
        for trace in traces:
            if trace.get("event_type") == "tool_call":
                payload = trace.get("payload", {})
                if payload.get("tool_name") == tool_name:
                    inputs = payload.get("inputs", {})
                    param_value = inputs
                    for part in param_name.split("."):
                        if isinstance(param_value, dict):
                            param_value = param_value.get(part)
                        else:
                            return ""
                    if param_value:
                        return str(param_value)
        
        return ""
    
    def _extract_outbox_content(self, check_target: str) -> str:
        """Extract content from outbox emails (reuse from SemanticJudgeValidator)."""
        if not self.test_config:
            return ""
        
        try:
            from datetime import datetime
            
            outbox_dir = Path(self.test_config.get("data", {}).get("outbox_dir", "data/interactive_agent/outbox"))
            
            if not outbox_dir.exists():
                return ""
            
            parts = check_target.split(".")
            if len(parts) < 3:
                return ""
            
            target_field = parts[-1]
            
            if parts[1] == "latest":
                email_files = list(outbox_dir.glob("*.json"))
                if not email_files:
                    return ""
                
                emails = []
                for email_file in email_files:
                    try:
                        with open(email_file, 'r', encoding='utf-8') as f:
                            email = json.load(f)
                        emails.append((email_file, email))
                    except:
                        continue
                
                if not emails:
                    return ""
                
                def get_timestamp(email_data):
                    ts = email_data.get('sent_ts') or email_data.get('created_ts', '')
                    try:
                        return datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                    except:
                        return 0.0
                
                emails.sort(key=lambda x: get_timestamp(x[1]), reverse=True)
                latest_email = emails[0][1]
                
                content = latest_email.get(target_field, latest_email.get('body', latest_email.get('body_plain', '')))
                return str(content) if content else ""
            
            return ""
        except Exception as e:
            print(f"Warning: Error extracting outbox content: {e}")
            return ""
    
    def _call_semantic_judge(self, content: str, query: str) -> bool:
        """Call semantic judge (reuse from SemanticJudgeValidator)."""
        try:
            from agent.utils import call_llm_chat_completion, detect_provider
            import re
            
            provider = detect_provider(self._judge_model)
            
            # Use standard fact/figure matching evaluation
            
            def extract_facts(text: str) -> Dict[str, Any]:
                try:
                    numbers = re.findall(r"(?<![\w\d])(?:\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+)(?![\w\d])", text)
                    percents = re.findall(r"\b\d{1,3}(?:\.\d+)?\s*%\b", text)
                    currency = re.findall(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\bUSD\s*\d+(?:\.\d+)?\b", text, flags=re.IGNORECASE)
                    kpis = re.findall(r"\b(EPS|Revenue|Net\s+Income|Budget|Discount|Price|Pricing|Spend|Valuation)\b", text, flags=re.IGNORECASE)
                    return {
                        "numbers": numbers,
                        "percentages": percents,
                        "currency": currency,
                        "kpis": list({k.lower() for k in kpis}),
                    }
                except Exception:
                    return {"numbers": [], "percentages": [], "currency": [], "kpis": []}
            
            query_facts = extract_facts(query)
            content_facts = extract_facts(content)
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a rigorous semantic judge for fact/figure overlap and memory recall. "
                        "Given a query describing what should be recalled and a candidate content, decide if the content correctly recalls the information from the query. "
                        "For information involving mappings or relationships (e.g., items to values, entities to attributes), "
                        "verify that the exact mappings match. "
                        "For numbers, amounts, dates, or other quantitative information, verify that the values match "
                        "(allowing for formatting variants like $1,000 vs 1000 vs one thousand dollars). "
                        "For qualitative information, verify semantic equivalence while preserving key details. "
                        "Be tolerant to paraphrasing and formatting variants but strict on factual correctness. "
                        "Always respond with valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Query (what should be recalled):\n{query}\n\n"
                        f"Content to check (agent's response):\n{content}\n\n"
                        f"Extracted query facts (numbers/percentages/currency/KPIs):\n{json.dumps(query_facts, ensure_ascii=False)}\n\n"
                        f"Extracted content facts (numbers/percentages/currency/KPIs):\n{json.dumps(content_facts, ensure_ascii=False)}\n\n"
                        "Task: Determine if the content correctly recalls the information from the query. "
                        "For colors, ensure exact matches with correct mappings. "
                        "For budget/numbers, ensure amounts match (allowing formatting variations).\n\n"
                        "Output strict JSON with this exact shape:\n"
                        "{\n"
                        "  \"contains_information\": true/false,\n"
                        "  \"matched_items\": [\"list matched facts\"],\n"
                        "  \"explanation\": \"brief justification\"\n"
                        "}"
                    )
                }
            ]
            
            model_lower = str(self._judge_model).lower()
            is_gpt5_family = model_lower.startswith("gpt-5") or "gpt-5" in model_lower
            temperature = None if is_gpt5_family else self._judge_temperature
            response_format = {"type": "json_object"} if provider == "openai" else None
            
            response = call_llm_chat_completion(
                model=self._judge_model,
                messages=messages,
                temperature=temperature,
                response_format=response_format,
                max_tokens=2000,
                max_output_tokens=2000
            )
            
            if not response or not response.choices or not response.choices[0].message:
                return False
            
            result_text = response.choices[0].message.content.strip()
            if not result_text:
                return False
            
            try:
                result_json = json.loads(result_text)
                contains_info = result_json.get("contains_information", False)
                return bool(contains_info)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from response
                json_match = re.search(r'\{"contains_information":\s*(true|false)', result_text, re.IGNORECASE)
                if json_match:
                    return json_match.group(1).lower() == "true"
                return "true" in result_text.lower() and "false" not in result_text.lower()[:20]
        except Exception as e:
            print(f"Warning: Cross-step semantic judge API error: {e}")
            return False
    
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
    
    def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
        validators = self.config.get("validators", [])
        operator = self.config.get("operator", "AND").upper()
        
        if not validators:
            return True
        
        results = []
        for validator_config in validators:
            validator_type = validator_config.get("type")
            validator = self._create_validator(validator_type, validator_config, self.test_config)
            if validator:
                result = validator.validate(agent_response, session_id, traces)
                results.append(result)
        
        if operator == "OR":
            return any(results)
        else:  # AND
            return all(results)
    
    def validate_with_print(self, agent_response: str, session_id: str, traces: List[Dict] = None, prefix: str = "", trace_file: Optional[str] = None) -> bool:
        """
        Validate and print individual validator results.
        
        Args:
            agent_response: Agent response text
            session_id: Session identifier
            traces: Trace events
            prefix: Prefix for nested validators (for indentation)
            trace_file: Optional trace file path for logging validator results
            
        Returns:
            Whether validation passed
        """
        validators = self.config.get("validators", [])
        operator = self.config.get("operator", "AND").upper()
        
        if not validators:
            return True
        
        results = []
        validator_results = []  # Store results for trace logging
        
        for i, validator_config in enumerate(validators):
            validator_type = validator_config.get("type")
            validator = self._create_validator(validator_type, validator_config, self.test_config)
            if validator:
                result = validator.validate(agent_response, session_id, traces)
                results.append(result)
                
                # Generate validator name for display
                validator_name = self._get_validator_display_name(validator_config, validator_type, i)
                
                # Print result to terminal
                from agent.colored_trace_printer import print_validator_result
                print_validator_result(validator_type, f"{prefix}{validator_name}", result)
                
                # Store result for trace logging
                validator_results.append({
                    "type": validator_type,
                    "name": validator_name,
                    "passed": result,
                    "config": validator_config
                })
                
                # If composite, also print nested validators
                if validator_type == "composite" and isinstance(validator, CompositeValidator):
                    nested_prefix = prefix + "  "
                    # Pass trace_file to nested validators
                    if hasattr(validator, 'validate_with_print'):
                        nested_result = validator.validate_with_print(agent_response, session_id, traces, nested_prefix, trace_file)
                        # Update result with nested result
                        results[-1] = nested_result
        
        # Log validator results to trace file if provided
        if trace_file and validator_results:
            try:
                from agent.utils import append_trace_event
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
                print(f"Warning: Could not log validator results to trace: {e}")
        
        if operator == "OR":
            return any(results)
        else:  # AND
            return all(results)
    
    def _get_validator_display_name(self, config: Dict[str, Any], validator_type: str, index: int = 0) -> str:
        """Generate a display name for a validator."""
        if validator_type == "tool_call":
            tools = config.get("required_tools", [])
            if tools:
                return f"Tool call: {', '.join(tools)}"
        elif validator_type == "semantic_judge":
            target = config.get("check_target", "agent_response")
            if target.startswith("tool."):
                parts = target.split(".")
                if len(parts) >= 3:
                    return f"Semantic judge ({parts[1]}.{parts[2]})"
            return "Semantic judge"
        elif validator_type == "keyword":
            keywords = config.get("keywords", [])
            if keywords:
                return f"Keywords: {', '.join(keywords[:3])}" + ("..." if len(keywords) > 3 else "")
        elif validator_type == "composite":
            op = config.get("operator", "AND")
            return f"Composite ({op})"
        
        return f"{validator_type} validator"

    def score(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> float:
        validators = self.config.get("validators", [])
        if not validators:
            return 1.0
        scores: List[float] = []
        for validator_config in validators:
            validator_type = validator_config.get("type")
            validator = self._create_validator(validator_type, validator_config, self.test_config)
            if validator:
                try:
                    scores.append(float(validator.score(agent_response, session_id, traces)))
                except Exception:
                    scores.append(0.0)
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
    
    def _create_validator(self, validator_type: str, config: Dict[str, Any], test_config: Optional[Dict[str, Any]] = None) -> Optional[TestValidator]:
        """Create a validator instance based on type."""
        if validator_type == "keyword":
            return KeywordValidator(config, test_config)
        elif validator_type == "tool_call":
            return ToolCallValidator(config, test_config)
        elif validator_type == "file_system":
            return FileSystemValidator(config, test_config)
        elif validator_type == "memory":
            return MemoryValidator(config, test_config)
        elif validator_type == "mem0_memory":
            return Mem0MemoryValidator(config, test_config)
        elif validator_type == "semantic_judge":
            return SemanticJudgeValidator(config, test_config)
        elif validator_type == "cross_step_semantic_judge":
            return CrossStepSemanticJudgeValidator(config, test_config)
        elif validator_type == "composite":
            return CompositeValidator(config, test_config)
        return None


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
    elif validator_type == "memory":
        return MemoryValidator(validator_config, test_config)
    elif validator_type == "mem0_memory":
        return Mem0MemoryValidator(validator_config, test_config)
    elif validator_type == "semantic_judge":
        return SemanticJudgeValidator(validator_config, test_config)
    elif validator_type == "cross_step_semantic_judge":
        return CrossStepSemanticJudgeValidator(validator_config, test_config)
    elif validator_type == "composite":
        return CompositeValidator(validator_config, test_config)
    else:
        # Default to keyword validator for backward compatibility
        return KeywordValidator(validator_config)


# Convenience functions for common validation patterns

def validate_keywords(keywords: List[str], require_all: bool = False, min_required: int = None) -> Dict[str, Any]:
    """Create a keyword validator config."""
    return {
        "type": "keyword",
        "keywords": keywords,
        "require_all": require_all,
        "min_required": min_required
    }


def validate_tool_calls(required_tools: List[str], tool_params: Dict[str, Dict] = None, require_all: bool = True) -> Dict[str, Any]:
    """Create a tool call validator config."""
    return {
        "type": "tool_call",
        "required_tools": required_tools,
        "tool_params": tool_params or {},
        "require_all_tools": require_all
    }


def validate_files(check_files: List[str], file_content: Dict[str, Union[str, List[str]]] = None, require_all: bool = True) -> Dict[str, Any]:
    """Create a file system validator config."""
    return {
        "type": "file_system",
        "check_files": check_files,
        "file_content": file_content or {},
        "require_all_files": require_all
    }


def validate_memory(check_keywords: List[str] = None, check_absence: List[str] = None, require_all_keywords: bool = False, min_keywords: int = None) -> Dict[str, Any]:
    """Create a memory validator config."""
    return {
        "type": "memory",
        "check_keywords": check_keywords or [],
        "check_absence": check_absence or [],
        "require_all_keywords": require_all_keywords,
        "min_keywords": min_keywords
    }


def validate_composite(validators: List[Dict[str, Any]], operator: str = "AND") -> Dict[str, Any]:
    """Create a composite validator config."""
    return {
        "type": "composite",
        "validators": validators,
        "operator": operator
    }


def validate_semantic_judge(query: str, check_target: str = "agent_response", 
                            tool_name: Optional[str] = None, tool_param: Optional[str] = None) -> Dict[str, Any]:
    """Create a semantic judge validator config."""
    config = {
        "type": "semantic_judge",
        "query": query
    }
    
    if check_target != "agent_response":
        config["check_target"] = check_target
        if tool_name:
            config["tool_name"] = tool_name
        if tool_param:
            config["tool_param"] = tool_param
    
    return config
