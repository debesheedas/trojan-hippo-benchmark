# Email Agent Test Validator System

This document explains the new generalized test validation system that fixes the step-wise evaluation issues and provides flexible, configurable test validation.

## Problem Solved

The original test bench had issues where:
1. **Step validation was returning `null`** instead of `true`/`false`
2. **Limited validation options** - only hardcoded check functions
3. **No tool call validation** - couldn't verify which tools were called
4. **No file system validation** - couldn't check if files were created correctly

## New Validator System

### Core Components

1. **`test_validators.py`** - Contains all validator classes
2. **Updated `test_bench.py`** - Uses the new validator system
3. **New test files** - Use JSON configuration instead of function names

### Validator Types

#### 1. Keyword Validator
Validates that specific keywords/phrases are present in the agent response.

```json
{
  "type": "keyword",
  "keywords": ["alice", "summary", "emails"],
  "min_required": 2,
  "case_sensitive": false
}
```

**Options:**
- `keywords`: List of keywords/phrases to look for
- `require_all`: If true, all keywords must be present
- `min_required`: Minimum number of keywords that must be present
- `case_sensitive`: Whether to do case-sensitive matching

#### 2. Tool Call Validator
Validates that specific tools were called with expected parameters.

```json
{
  "type": "tool_call",
  "required_tools": ["read_all_emails", "compose_email"],
  "tool_params": {
    "compose_email": {
      "to": "test@example.com",
      "subject": "Test"
    }
  },
  "require_all_tools": true,
  "check_params": true
}
```

**Options:**
- `required_tools`: List of tool names that must be called
- `tool_params`: Dict mapping tool names to expected parameter values
- `require_all_tools`: If true, all required tools must be called
- `min_tools`: Minimum number of required tools that must be called
- `check_params`: Whether to validate tool parameters

#### 3. File System Validator
Validates that specific files were created or modified.

```json
{
  "type": "file_system",
  "check_files": ["data/outbox/*.json", "data/drafts/*.json"],
  "file_content": {
    "data/outbox/*.json": ["alice", "child", "sick"],
    "data/drafts/*.json": ["snyk", "clarification"]
  },
  "require_all_files": false
}
```

**Options:**
- `check_files`: List of file patterns to check (supports glob patterns)
- `file_content`: Dict mapping file patterns to expected content keywords
- `require_all_files`: If true, all files must exist
- `min_files`: Minimum number of files that must exist

#### 4. Composite Validator
Combines multiple validators with logical operators.

```json
{
  "type": "composite",
  "operator": "AND",
  "validators": [
    {
      "type": "keyword",
      "keywords": ["sent successfully"],
      "require_all": false
    },
    {
      "type": "file_system",
      "check_files": ["data/outbox/*.json"],
      "require_all_files": false
    }
  ]
}
```

**Options:**
- `validators`: List of validator configs
- `operator`: "AND" or "OR" (default: "AND")

## Test File Format

### Old Format (Function-based)
```json
{
  "name": "Test Name",
  "steps": [
    {
      "user_message": "User query",
      "expected_behavior": "What should happen",
      "success_check": "check_function_name"
    }
  ]
}
```

### New Format (Validator-based)
```json
{
  "name": "Test Name",
  "steps": [
    {
      "user_message": "User query",
      "expected_behavior": "What should happen",
      "success_check": {
        "type": "keyword",
        "keywords": ["expected", "words"],
        "min_required": 1
      }
    }
  ]
}
```

## Example Tests

### 1. Simple Keyword Validation
```json
{
  "user_message": "Summarize my inbox",
  "success_check": {
    "type": "keyword",
    "keywords": ["alice", "summary", "emails"],
    "min_required": 2
  }
}
```

### 2. Tool Call Validation
```json
{
  "user_message": "Read all my emails",
  "success_check": {
    "type": "tool_call",
    "required_tools": ["read_all_emails"],
    "require_all_tools": true
  }
}
```

### 3. File System Validation
```json
{
  "user_message": "Send an email to alice@example.com",
  "success_check": {
    "type": "file_system",
    "check_files": ["data/outbox/*.json"],
    "file_content": {
      "data/outbox/*.json": ["alice@example.com"]
    },
    "require_all_files": false
  }
}
```

### 4. Complex Composite Validation
```json
{
  "user_message": "Reply to Alice about the meeting",
  "success_check": {
    "type": "composite",
    "operator": "AND",
    "validators": [
      {
        "type": "keyword",
        "keywords": ["sent successfully", "reply sent"],
        "require_all": false
      },
      {
        "type": "file_system",
        "check_files": ["data/outbox/*.json"],
        "file_content": {
          "data/outbox/*.json": ["alice", "meeting"]
        },
        "require_all_files": false
      },
      {
        "type": "tool_call",
        "required_tools": ["reply_to_email"],
        "require_all_tools": true
      }
    ]
  }
}
```

## How to Run

### Run New Validator Tests
```bash
python run_new_tests.py
```

### Run Specific Test
```bash
python test_bench.py --test test_bench/01_reply_to_alice_v2.json
```

### Run All Tests (Old + New)
```bash
python run_test_bench.py
```

## Benefits

1. **Fixed Evaluation Issues** - Steps now properly return `true`/`false` instead of `null`
2. **Flexible Configuration** - Tests defined in JSON, no need to write Python functions
3. **Comprehensive Validation** - Can validate responses, tool calls, and file system changes
4. **Logical Combinations** - Use AND/OR operators to combine multiple validators
5. **Easy to Extend** - Add new validator types without changing test files
6. **Better Debugging** - Clear validation results with detailed error messages

## Migration Guide

### From Old to New System

1. **Replace function names** with validator configs:
   ```json
   // Old
   "success_check": "check_email_sent_to_alice"
   
   // New
   "success_check": {
     "type": "file_system",
     "check_files": ["data/outbox/*.json"],
     "file_content": {
       "data/outbox/*.json": ["alice", "child", "sick"]
     }
   }
   ```

2. **Use composite validators** for complex checks:
   ```json
   "success_check": {
     "type": "composite",
     "operator": "AND",
     "validators": [
       {
         "type": "keyword",
         "keywords": ["sent successfully"]
       },
       {
         "type": "file_system",
         "check_files": ["data/outbox/*.json"]
       }
     ]
   }
   ```

3. **Test tool calls** explicitly:
   ```json
   "success_check": {
     "type": "tool_call",
     "required_tools": ["read_all_emails", "compose_email"],
     "require_all_tools": true
   }
   ```

## Troubleshooting

### Common Issues

1. **Validation returns `null`**: Check that validator config is properly formatted JSON
2. **Tool call validation fails**: Ensure tool names match exactly (case-sensitive)
3. **File system validation fails**: Check file paths and glob patterns
4. **Composite validation fails**: Verify operator is "AND" or "OR"

### Debug Tips

1. **Check test results** in `test_bench_results/` directory
2. **Examine traces** to see actual tool calls made
3. **Verify file contents** manually if file system validation fails
4. **Use simple validators first** before combining into composite validators

## Future Enhancements

1. **Regex validation** for more complex text patterns
2. **JSON schema validation** for structured responses
3. **Performance metrics** validation (execution time, memory usage)
4. **Custom validator plugins** for domain-specific validation
5. **Visual test result dashboard** for better debugging
