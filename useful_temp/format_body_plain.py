#!/usr/bin/env python3
"""
Script to format the contents of scratchpad.txt as a JSON-escaped string
suitable for use as the "body_plain" attribute in JSON files.
"""

import json
import sys
from pathlib import Path

def format_body_plain(input_file: str = "scratchpad.txt") -> str:
    """
    Read a text file and format it as a JSON-escaped string.
    
    Args:
        input_file: Path to the input text file
        
    Returns:
        JSON-escaped string ready to paste into JSON files
    """
    file_path = Path(input_file)
    
    if not file_path.exists():
        print(f"Error: File '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    
    # Read the file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Remove trailing newline if present (common when editing)
    if content.endswith('\n'):
        content = content.rstrip('\n')
    
    # Use json.dumps to properly escape the string for JSON
    # This handles: quotes, backslashes, newlines, tabs, etc.
    json_escaped = json.dumps(content, ensure_ascii=False)
    
    return json_escaped


if __name__ == "__main__":
    # Allow specifying a different file as argument
    input_file = sys.argv[1] if len(sys.argv) > 1 else "scratchpad.txt"
    
    formatted = format_body_plain(input_file)
    
    # Output the formatted string
    print(formatted)
    
    # Also print a helpful message with the full attribute line
    print("\n--- Copy the line above ---", file=sys.stderr)
    print(f'Or use this complete attribute line:', file=sys.stderr)
    print(f'        "body_plain": {formatted},', file=sys.stderr)

