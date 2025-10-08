#!/usr/bin/env python3
"""
Quick setup verification script.
Checks that all required files and directories exist.
"""

import sys
from pathlib import Path

def check_file(path: str, description: str) -> bool:
    """Check if a file exists."""
    if Path(path).exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}: {path} (MISSING)")
        return False

def check_dir(path: str, description: str) -> bool:
    """Check if a directory exists."""
    if Path(path).is_dir():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}: {path} (MISSING)")
        return False

def main():
    print("🔍 Verifying Email Agent MVP Setup\n")
    
    all_ok = True
    
    print("📁 Core Files:")
    all_ok &= check_file("main.py", "FastAPI application")
    all_ok &= check_file("agent_tools.py", "Agent tools")
    all_ok &= check_file("utils.py", "Utility functions")
    all_ok &= check_file("config.yaml", "Configuration")
    all_ok &= check_file("requirements.txt", "Dependencies")
    
    print("\n📁 Documentation:")
    all_ok &= check_file("README.md", "Main documentation")
    all_ok &= check_file("tests/smoke_test.md", "Test procedures")
    
    print("\n📁 UI:")
    all_ok &= check_file("static/index.html", "Web interface")
    
    print("\n📁 Data Directories:")
    all_ok &= check_dir("data/mailbox", "Mailbox directory")
    all_ok &= check_dir("data/drafts", "Drafts directory")
    
    print("\n📧 Sample Emails:")
    for i in range(1, 6):
        all_ok &= check_file(f"data/mailbox/sample-{i:03d}.json", f"Sample email {i}")
    
    print("\n📝 Trace File:")
    all_ok &= check_file("data/trace.jsonl", "Trace log")
    
    print("\n" + "="*60)
    
    if all_ok:
        print("✅ ALL CHECKS PASSED!")
        print("\nYou can now run the application:")
        print("  ./run.sh")
        print("  OR")
        print("  pip install -r requirements.txt")
        print("  uvicorn main:app --reload")
        print("\nThen open: http://localhost:8000")
        return 0
    else:
        print("❌ SOME CHECKS FAILED!")
        print("\nPlease ensure all files are created correctly.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

