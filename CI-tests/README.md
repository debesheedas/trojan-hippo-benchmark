# CI Tests

This directory contains continuous integration (CI) tests to ensure that refactoring doesn't break the benchmark implementation.

## Directory Structure

```
CI-tests/
  ├── testcases/          # Test case files
  │   └── test1.json     # The regression test case
  ├── results/            # Benchmark results (generated during CI runs)
  ├── ground_truth.json   # Expected results for each combination
  ├── compare_results.py  # Script to compare results against ground truth
  └── README.md          # This file
```

## Test Overview

The regression test runs a single test case (`testcases/test1.json`) across all 24 combinations of:
- **Memory backends**: `none`, `explicit`, `mem0`, `rag`, `context` (5 backends)
- **Defense types**: `none`, `user_prompt_only`, `no_untrusted_tools`, `limit_memory_length`, `provable_policy` (5 defenses)

Note: `limit_memory_length` is not applicable for the `context` backend, so there are 24 valid combinations.

## How It Works

The regression test uses the **normal benchmark infrastructure** - no special test code needed:

1. **Run benchmark**: Uses `scripts/run_benchmark.py` to run all combinations (just like normal benchmarks)
2. **Compare results**: Uses `CI-tests/compare_results.py` to compare actual results against ground truth

This ensures we're testing the exact same code path as normal benchmarks.

## Ground Truth

The `ground_truth.json` file contains the expected pass/fail results for each combination. You need to manually populate this file after running the test case once to establish the baseline.

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

## Running Tests Locally

1. Make sure you have your OpenAI API key set in `.env`:
   ```bash
   echo "OPENAI_API_KEY=your-key-here" > .env
   ```

2. Run the benchmark for all combinations:
   ```bash
   python scripts/run_benchmark.py \
     --test CI-tests/testcases/test1.json \
     --memory-backend none explicit mem0 rag context \
     --defense-type none user_prompt_only no_untrusted_tools limit_memory_length provable_policy \
     --results-dir CI-tests/results \
     --num-workers 1 \
     --force
   ```

3. Compare results against ground truth:
   ```bash
   python CI-tests/compare_results.py \
     --results-dir CI-tests/results \
     --test-file CI-tests/testcases/test1.json
   ```

   (Model name is automatically read from `agent_config.yaml`)

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
1. Run the benchmark for all combinations (serial execution, `--num-workers 1`)
2. Compare results against ground truth
3. Fail if any results don't match

## Updating Ground Truth

If you make changes that intentionally affect test results, you'll need to update `ground_truth.json`:

1. Run the benchmark locally (see "Running Tests Locally" above)
2. Check the results in `CI-tests/results/`
3. Update `CI-tests/ground_truth.json` with the new expected results
4. Commit the updated ground truth file
