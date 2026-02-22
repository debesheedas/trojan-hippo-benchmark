# CI Tests

This directory contains continuous integration (CI) tests to ensure that refactoring doesn't break the benchmark implementation.

## Model: gpt-5-mini (fixed)

**CI runs and ground truth use the model `gpt-5-mini` only.** This is hardcoded in:

- `run_ci_tests.sh` (benchmark + compare)
- `compare_results.py` (when results dir is `CI-tests/results`, model defaults to gpt-5-mini)
- `update_ground_truth_from_results.py` (gold is built from gpt-5-mini results)
- `.github/workflows/regression_test.yml` (benchmark step uses `--model gpt-5-mini`)

Do not change the model in one place without updating the others. In the future the model may be made configurable; until then it remains fixed for reproducibility.

## Directory Structure

```
CI-tests/
  ├── testcases/          # Test case files
  │   ├── test1.json     # Regression test case 1
  │   └── test2.json     # Regression test case 2
  ├── ground_truth/       # Ground truth files
  │   ├── test1.json     # Expected results for test1
  │   └── test2.json     # Expected results for test2
  ├── results/            # Benchmark results (generated during CI runs)
  ├── compare_results.py  # Script to compare results against ground truth
  └── README.md          # This file
```

## Test Overview

The regression test runs multiple test cases across all 23 combinations of:
- **Memory backends**: `none`, `explicit`, `mem0`, `rag`, `context` (5 backends)
- **Defense types**: `none`, `user_prompt_only`, `no_untrusted_tools`, `limit_memory_length`, `provable_policy` (5 defenses)

Note: Invalid combinations are skipped:
- `limit_memory_length` is not applicable for the `context` backend
- `user_prompt_only` is not applicable for the `explicit` backend

## How It Works

The regression test uses the **normal benchmark infrastructure** - no special test code needed:

1. **Run benchmark**: Uses `scripts/run_benchmark.py` to run all combinations (just like normal benchmarks)
2. **Compare results**: Uses `CI-tests/compare_results.py` to compare actual results against ground truth

This ensures we're testing the exact same code path as normal benchmarks.

## Ground Truth

Each test case has its own ground truth file in the `ground_truth/` directory:
- `ground_truth/test1.json` - Expected results for `test1.json`
- `ground_truth/test2.json` - Expected results for `test2.json`
- For additional tests, use `ground_truth/{test_name}.json` format

The `compare_results.py` script automatically detects the correct ground truth file based on the test file name (e.g., `test1.json` → `ground_truth/test1.json`).

### Format

```json
{
  "memory_backend": {
    "defense_type": true/false
  }
}
```

- `true` = test should pass
- `false` = test should fail

You need to manually populate the ground truth file after running each test case once to establish the baseline.

## Running Tests Locally

1. Make sure you have your OpenAI API key set in `.env`:
   ```bash
   echo "OPENAI_API_KEY=your-key-here" > .env
   ```

2. Run the benchmark for all combinations (for each test case):
   ```bash
   # For test1
   python scripts/run_benchmark.py \
     --test CI-tests/testcases/test1.json \
     --memory-backend none explicit mem0 rag context \
     --defense-type none user_prompt_only no_untrusted_tools limit_memory_length provable_policy \
     --results-dir CI-tests/results \
     --num-workers 1 \
     --force
   
   # For test2
   python scripts/run_benchmark.py \
     --test CI-tests/testcases/test2.json \
     --memory-backend none explicit mem0 rag context \
     --defense-type none user_prompt_only no_untrusted_tools limit_memory_length provable_policy \
     --results-dir CI-tests/results \
     --num-workers 1 \
     --force
   ```

3. Compare results against ground truth (for each test case):
   ```bash
   # For test1 (uses ground_truth.json)
   python CI-tests/compare_results.py \
     --results-dir CI-tests/results \
     --test-file CI-tests/testcases/test1.json
   
   # For test2 (auto-detects ground_truth_test2.json)
   python CI-tests/compare_results.py \
     --results-dir CI-tests/results \
     --test-file CI-tests/testcases/test2.json
   ```

   (Model name is automatically read from `agent_config.yaml`)
   (Ground truth file is auto-detected from test file name)

## GitHub Actions

The test runs automatically on every commit via GitHub Actions (`.github/workflows/regression_test.yml`).

### Setting up GitHub Secrets

To enable the GitHub Actions workflow, you need to add your OpenAI API key as a secret:

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `OPENAI_API_KEY`
5. Value: Your OpenAI API key
6. Click **Add secret**

The workflow will automatically:
1. Run the benchmark for all combinations for each test case (serial execution, `--num-workers 1`)
2. Compare results against ground truth for each test case
3. Fail if any results don't match their respective ground truth

## Updating Ground Truth

If you make changes that intentionally affect test results, you'll need to update the corresponding ground truth file:

1. Run the benchmark locally for the specific test case (see "Running Tests Locally" above)
2. Check the results in `CI-tests/results/`
3. Update the appropriate ground truth file:
   - `CI-tests/ground_truth/test1.json` for test1
   - `CI-tests/ground_truth/test2.json` for test2
   - `CI-tests/ground_truth/{test_name}.json` for other tests
4. Commit the updated ground truth file
