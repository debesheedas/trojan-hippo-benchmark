#!/bin/bash
# Script to run the email agent test suite

echo "============================================================"
echo "EMAIL AGENT - AUTOMATED TEST RUNNER"
echo "============================================================"
echo ""

# Check if server is running
if ! curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "⚠️  Server is not running!"
    echo ""
    echo "Please start the server in another terminal:"
    echo "  cd $(pwd)"
    echo "  source venv/bin/activate"
    echo "  uvicorn main:app --reload --host 0.0.0.0 --port 8000"
    echo ""
    exit 1
fi

echo "✓ Server is running"
echo ""

# Clean up old test data
echo "Cleaning up old test data..."
rm -f data/outbox/*.json 2>/dev/null || true
rm -rf test_reports 2>/dev/null || true
mkdir -p test_reports

# Clear trace file or back it up
if [ -f "data/trace.jsonl" ]; then
    mv data/trace.jsonl "data/trace_backup_$(date +%s).jsonl"
    touch data/trace.jsonl
fi

echo "✓ Cleanup complete"
echo ""

# Activate virtual environment and run tests
echo "Running test suite..."
echo "============================================================"
echo ""

source venv/bin/activate
python3 test_suite.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Test reports saved to: ./test_reports/"
echo "Trace logs saved to: ./data/trace.jsonl"
echo "============================================================"

exit $EXIT_CODE

