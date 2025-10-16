"""
Generalized test validation functions for the Email Agent test bench.

This module provides flexible validation functions that can be configured
via parameters in the test JSON files.
"""

import json
import re
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
    
    def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
        keywords = self.config.get("keywords", [])
        require_all = self.config.get("require_all", False)
        min_required = self.config.get("min_required", None)
        check_absence = self.config.get("check_absence", False)
        case_sensitive = self.config.get("case_sensitive", False)
        
        if not keywords:
            return True  # No keywords to check
        
        text = agent_response if case_sensitive else agent_response.lower()
        found_keywords = []
        
        for keyword in keywords:
            search_term = keyword if case_sensitive else keyword.lower()
            if search_term in text:
                found_keywords.append(keyword)
        
        if check_absence:
            # For absence check, we want NO keywords to be found
            return len(found_keywords) == 0
        else:
            # For presence check, use normal logic
            if min_required is not None:
                return len(found_keywords) >= min_required
            elif require_all:
                return len(found_keywords) == len(keywords)
            else:
                return len(found_keywords) > 0


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
    
    def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
        if not traces:
            return False
        
        required_tools = self.config.get("required_tools", [])
        tool_params = self.config.get("tool_params", {})
        require_all_tools = self.config.get("require_all_tools", True)
        min_tools = self.config.get("min_tools", None)
        check_params = self.config.get("check_params", True)
        
        if not required_tools:
            return True  # No tools to check
        
        # Extract tool calls from traces
        tool_calls = []
        for trace in traces:
            if trace.get("event_type") == "tool_call":
                payload = trace.get("payload", {})
                tool_name = payload.get("tool_name", "")
                tool_inputs = payload.get("inputs", {})
                tool_calls.append({
                    "name": tool_name,
                    "inputs": tool_inputs
                })
        
        # Check which required tools were called
        called_tools = []
        for tool_call in tool_calls:
            if tool_call["name"] in required_tools:
                called_tools.append(tool_call)
        
        # Validate tool count
        if min_tools is not None:
            if len(called_tools) < min_tools:
                return False
        elif require_all_tools:
            if len(called_tools) < len(required_tools):
                return False
        
        # Validate tool parameters if requested
        if check_params and tool_params:
            for tool_call in called_tools:
                tool_name = tool_call["name"]
                if tool_name in tool_params:
                    expected_params = tool_params[tool_name]
                    actual_inputs = tool_call["inputs"]
                    
                    for param_name, expected_value in expected_params.items():
                        if param_name not in actual_inputs:
                            return False
                        
                        actual_value = actual_inputs[param_name]
                        
                        # Handle different types of parameter validation
                        if isinstance(expected_value, str):
                            # String matching (case-insensitive by default)
                            # Check if expected value is contained in actual value
                            if expected_value.lower() not in str(actual_value).lower():
                                return False
                        elif isinstance(expected_value, list):
                            # List of keywords - check if any are present in the parameter
                            actual_lower = str(actual_value).lower()
                            found_keywords = [kw for kw in expected_value if kw.lower() in actual_lower]
                            if not found_keywords:
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
    """
    
    def _resolve_path(self, file_pattern: str) -> str:
        """Resolve file pattern to test environment path if test_config is available."""
        if not self.test_config:
            return file_pattern
        
        # Map common data directory patterns to test environment paths
        data_mappings = {
            "data/outbox": self.test_config["data"]["outbox_dir"],
            "data/drafts": self.test_config["data"]["drafts_dir"],
            "data/mailbox": self.test_config["data"]["mailbox_dir"],
            "data/sessions": self.test_config["data"]["sessions_dir"],
        }
        
        # Replace data directory paths with test environment paths
        resolved_pattern = file_pattern
        for original_path, test_path in data_mappings.items():
            if file_pattern.startswith(original_path):
                resolved_pattern = file_pattern.replace(original_path, test_path)
                break
        
        return resolved_pattern
    
    def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
        check_files = self.config.get("check_files", [])
        file_content = self.config.get("file_content", {})
        require_all_files = self.config.get("require_all_files", True)
        min_files = self.config.get("min_files", None)
        
        if not check_files:
            return True  # No files to check
        
        found_files = []
        
        for file_pattern in check_files:
            # Resolve path to test environment
            resolved_pattern = self._resolve_path(file_pattern)
            
            # Convert pattern to Path and check if files exist
            pattern_path = Path(resolved_pattern)
            if pattern_path.is_file():
                found_files.append(str(pattern_path))
            elif pattern_path.is_dir():
                # If it's a directory, check for any files matching the pattern
                for file_path in pattern_path.glob("*"):
                    if file_path.is_file():
                        found_files.append(str(file_path))
            else:
                # Try glob pattern
                for file_path in Path(".").glob(resolved_pattern):
                    if file_path.is_file():
                        found_files.append(str(file_path))
        
        # Validate file count
        if min_files is not None:
            if len(found_files) < min_files:
                return False
        elif require_all_files:
            if len(found_files) < len(check_files):
                return False
        
        # Validate file content if specified
        if file_content:
            for file_path in found_files:
                for pattern, expected_content in file_content.items():
                    # Resolve pattern to test environment path
                    resolved_pattern = self._resolve_path(pattern)
                    if self._matches_pattern(file_path, resolved_pattern):
                        if not self._check_file_content(file_path, expected_content):
                            return False
        
        return True
    
    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches the pattern."""
        try:
            return Path(file_path).match(pattern)
        except:
            return pattern in file_path
    
    def _check_file_content(self, file_path: str, expected_content: Union[str, List[str]]) -> bool:
        """Check if file contains expected content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            
            if isinstance(expected_content, str):
                return expected_content.lower() in content
            elif isinstance(expected_content, list):
                # Check if at least 2 keywords are present (more flexible than requiring all)
                found_keywords = sum(1 for keyword in expected_content if keyword.lower() in content)
                return found_keywords >= 2
            
            return False
        except Exception as e:
            print(f"Warning: Could not read file {file_path}: {e}")
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
    
    def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
        check_keywords = self.config.get("check_keywords", [])
        check_absence = self.config.get("check_absence", [])
        memory_file = self.config.get("memory_file", "data/agent_memory.json")
        require_all_keywords = self.config.get("require_all_keywords", False)
        min_keywords = self.config.get("min_keywords", None)
        
        # Resolve memory file path for test environment
        if self.test_config and "data" in self.test_config:
            # Use test environment memory file if available
            test_memory_file = self.test_config["data"].get("memory_file", memory_file)
            if test_memory_file != memory_file:
                memory_file = test_memory_file
        
        try:
            # Read memory file
            with open(memory_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            long_term_memory = memory_data.get("long_term", [])
            memory_text = " ".join(long_term_memory).lower()
            
            # Check for required keywords
            if check_keywords:
                found_keywords = []
                for keyword in check_keywords:
                    if keyword.lower() in memory_text:
                        found_keywords.append(keyword)
                
                if min_keywords is not None:
                    if len(found_keywords) < min_keywords:
                        return False
                elif require_all_keywords:
                    if len(found_keywords) < len(check_keywords):
                        return False
                else:
                    if len(found_keywords) == 0:
                        return False
            
            # Check for absence of keywords
            if check_absence:
                for keyword in check_absence:
                    if keyword.lower() in memory_text:
                        return False
            
            return True
            
        except Exception as e:
            print(f"Warning: Could not read memory file {memory_file}: {e}")
            return False


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
