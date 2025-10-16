# Email Agent Test Bench - Pure Python vs UI Modes

This document explains how to run the Email Agent in two different modes:

1. **Pure Python Mode** (Fast, for benchmarks and automated testing)
2. **UI Mode** (Interactive web interface)

## Pure Python Mode (Test Bench)

### What is it?
- Direct function calls to the agent (no HTTP overhead)
- Maximum speed for automated testing and benchmarks
- Same agent logic as UI mode, but bypasses web server
- Perfect for running test suites and performance benchmarks

### Test Structure

Each test is defined in a JSON file in the `test_bench/` directory:

```json
{
  "name": "Test Name",
  "description": "What this test does",
  "steps": [
    {
      "user_message": "User query to send",
      "expected_behavior": "What should happen",
      "success_check": "check_function_name"  // Optional validation
    }
  ]
}
```

### Available Tests

1. **01_reply_to_alice.json** - Tests email summarization and replying to Alice about sick child
2. **02_draft_snyk.json** - Tests finding Snyk email and drafting a reply
3. **03_inbox_qa.json** - Tests Q&A about inbox contents and timestamps

### How to Run Pure Python Mode

#### Option 1: Run all tests
```bash
python run_test_bench.py
```

#### Option 2: Run specific test
```bash
python test_bench.py --test test_bench/01_reply_to_alice.json
```

#### Option 3: Run with custom directory
```bash
python test_bench.py --dir my_tests
```

#### Option 4: Single message (for quick testing)
```bash
python benchmark_runner.py --message "Summarize my inbox" --session test_001
```

### Test Results

Results are saved in `test_bench_results/` directory:
- Individual test results: `{test_name}_{session_id}.json`
- Summary reports: `summary_{timestamp}.json`

Each result includes:
- Test execution details
- Step-by-step results
- Success/failure status
- Execution times
- Full agent responses
- Trace events

### Success Validation

Tests automatically validate results using check functions:
- `check_email_sent_to_alice` - Verifies email was sent to Alice
- `check_draft_to_snyk_exists` - Verifies draft was created (not sent)
- `check_florian_response` - Verifies Florian email was found
- `check_flight_timestamp` - Verifies flight confirmation timestamp
- `check_contains_text_alice` - Verifies response contains "alice"

## UI Mode (Interactive Web Interface)

### What is it?
- Full web interface for interactive testing
- Same agent backend as pure Python mode
- Session management and memory visualization
- Real-time memory updates via Server-Sent Events

### How to Run UI Mode

#### Start the server:
```bash
# Using the provided script
./run.sh

# Or directly with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Access the UI:
Open your browser to: `http://localhost:8000`

#### Run the original test suite (HTTP-based):
```bash
python test_suite.py
```

## Key Differences

| Aspect | Pure Python Mode | UI Mode |
|--------|------------------|---------|
| **Speed** | Maximum (no HTTP) | Slower (HTTP overhead) |
| **Use Case** | Automated testing, benchmarks | Interactive exploration |
| **Session Management** | Simple in-memory | Persistent with file storage |
| **Memory Updates** | File-based only | Real-time SSE streaming |
| **Debugging** | Direct function calls | Web interface + traces |
| **Deployment** | Single Python script | Web server required |

## Test Execution Flow

### Pure Python Mode:
1. Load test definition from JSON
2. Create fresh agent instance
3. For each step:
   - Call `invoke_agent()` directly (no HTTP)
   - Validate response using check functions
   - Record results and timing
4. Save detailed results to JSON files

### UI Mode:
1. Start FastAPI server
2. Load web interface
3. User interacts via browser
4. Each interaction goes through `/api/chat` endpoint
5. Same agent logic, but with HTTP overhead

## Memory and Session Handling

### Pure Python Mode:
- **Sessions**: Simple in-memory storage per test run
- **Memory**: Uses existing long-term memory from `data/agent_memory.json`
- **Traces**: Written to `data/traces/{session_id}.jsonl`

### UI Mode:
- **Sessions**: Persistent storage in `data/sessions/`
- **Memory**: Real-time updates with SSE streaming
- **Traces**: Same format as pure Python mode

## Adding New Tests

1. Create a new JSON file in `test_bench/` directory
2. Define test steps with user messages and expected behavior
3. Add success check functions to `test_bench.py` if needed
4. Run with `python run_test_bench.py`

Example new test:
```json
{
  "name": "My New Test",
  "description": "Tests a specific workflow",
  "steps": [
    {
      "user_message": "What emails do I have?",
      "expected_behavior": "Agent should list emails",
      "success_check": "check_contains_text_emails"
    }
  ]
}
```

## Performance Comparison

The pure Python mode is significantly faster:
- **No HTTP overhead** (no request/response serialization)
- **No web server** (no FastAPI, uvicorn overhead)
- **Direct function calls** (no network latency)
- **Simpler session management** (no file I/O for sessions)

Expected speedup: **5-10x faster** for automated testing.

## Troubleshooting

### Pure Python Mode Issues:
- Check that `config.yaml` exists and is valid
- Ensure `data/` directory structure is correct
- Verify OpenAI API key is set (or using mock mode)
- Check `test_bench_results/` for detailed error logs

### UI Mode Issues:
- Ensure server is running on port 8000
- Check browser console for JavaScript errors
- Verify FastAPI server logs for backend errors
- Check `data/sessions/` and `data/traces/` for session data

## Next Steps

1. **Run the test bench** to verify pure Python mode works
2. **Compare results** between pure Python and UI modes
3. **Add more test cases** as needed for your benchmark
4. **Optimize further** by removing unnecessary API endpoints if UI is not needed
