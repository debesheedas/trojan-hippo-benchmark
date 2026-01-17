# GitHub Actions Regression Test - How It Works

## Overview

The GitHub Actions workflow runs a regression test that:
1. Sets up the environment and dependencies
2. Configures the OpenAI API key as a secret
3. Runs the benchmark for all combinations using the normal benchmark infrastructure
4. Compares results against ground truth
5. Passes/fails based on whether results match

## Step-by-Step Process

### 1. Environment Setup
- Checks out the code
- Sets up Python 3.11
- Installs dependencies from `requirements.txt`
- Creates `.env` file with `OPENAI_API_KEY` from GitHub Secrets

### 2. Run Benchmark

The workflow runs benchmarks for multiple test cases. For each test case:

```bash
python scripts/run_benchmark.py 
  --test CI-tests/testcases/test1.json 
  --memory-backend none explicit mem0 rag context 
  --defense-type none user_prompt_only no_untrusted_tools limit_memory_length provable_policy 
  --results-dir CI-tests/results 
  --num-workers 1 
  --force
```

**What this does:**
- Runs each test file (e.g., `test1.json`, `test2.json`) for **all 24 combinations** (5 backends × 5 defenses, minus 1 invalid combination)
- Uses the **exact same code path** as normal benchmarks (no special test code)
- Saves results to `CI-tests/results/` with structure:
  ```
  CI-tests/results/
    {model_name}/
      {memory_backend}/
        {defense_type}/
          {attack_type}/
            {test_file_name}.json
  ```

### 3. Compare Results

For each test case, the workflow compares results:

```bash
python CI-tests/compare_results.py 
  --results-dir CI-tests/results 
  --test-file CI-tests/testcases/test1.json
```

**What this does:**
- Reads each result file from `CI-tests/results/`
- Auto-detects the ground truth file:
  - `test1.json` → `ground_truth/test1.json`
  - `test2.json` → `ground_truth/test2.json`
  - `testN.json` → `ground_truth/testN.json`
- Compares `overall_success` from each result against the corresponding ground truth file
- Reports mismatches
- Exits with code 0 (success) if all match, 1 (failure) if any don't match

## Parallel Workers on GitHub Actions

**Question:** Do multiple workers work fine on GitHub Actions?

**Answer:** Yes, but we use `--num-workers 1` (serial execution) for CI/CD because:

1. **ProcessPoolExecutor works fine**: The parallelization uses Python's `ProcessPoolExecutor`, which works perfectly on GitHub Actions (Ubuntu). Each combination runs in a separate process.

2. **Why serial for CI/CD:**
   - **Rate limits**: Serial execution avoids hitting API rate limits
   - **Predictability**: Easier to debug if something fails
   - **Resource usage**: GitHub Actions runners have limited resources
   - **Cost**: Serial execution uses fewer API calls per second

3. **If you want to test parallel execution:**
   - You can change `--num-workers 1` to `--num-workers 4` or higher
   - It will work, but you may hit rate limits depending on your API tier
   - For production CI/CD, serial is recommended

## Result File Structure

Results are saved with the same structure as normal benchmarks:
```
CI-tests/results/
  gpt-4o-mini/              # Model name from agent_config.yaml
    explicit/               # Memory backend
      none/                 # Defense type
        benign/             # Attack type (from test file)
          test1.json        # Result file with overall_success
    mem0/
      none/
        benign/
          test1.json
    ...
```

The `compare_results.py` script uses the same `get_result_path()` function that the benchmark uses, ensuring consistency.

## Ground Truth Format

Each test case has its own ground truth file in the `ground_truth/` directory:
- `CI-tests/ground_truth/test1.json` - Expected results for `test1.json`
- `CI-tests/ground_truth/test2.json` - Expected results for `test2.json`
- `CI-tests/ground_truth/{test_name}.json` - Expected results for other tests

The `compare_results.py` script automatically detects the correct ground truth file based on the test file name (e.g., `test1.json` → `ground_truth/test1.json`).

Each ground truth file contains expected results:
```json
{
  "none": {
    "none": false,
    "user_prompt_only": false,
    ...
  },
  "explicit": {
    "none": true,
    "user_prompt_only": true,
    ...
  },
  ...
}
```

Each value is `true` (test should pass) or `false` (test should fail).

## Key Points

1. **No special test code**: Uses the exact same `run_benchmark.py` that you use normally
2. **Same result structure**: Results are saved with the same folder structure as normal benchmarks
3. **Simple comparison**: Just compares `overall_success` boolean against ground truth
4. **Serial execution**: Uses `--num-workers 1` to avoid rate limits and ensure reliability
5. **Self-contained**: All test files and results are in the `CI-tests/` directory
