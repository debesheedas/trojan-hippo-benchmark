# GitHub Actions Setup Guide

This guide walks you through setting up GitHub Secrets and testing the CI/CD workflow.

## Step 1: Add OpenAI API Key to GitHub Secrets

### Method 1: Via GitHub Web Interface (Recommended)

1. **Navigate to your repository on GitHub**
   - Go to: `https://github.com/YOUR_USERNAME/YOUR_REPO_NAME`
   - Make sure you're logged in and have admin/owner permissions

2. **Go to Settings**
   - Click on the **"Settings"** tab (top navigation bar, right side)
   - If you don't see "Settings", you may not have admin access to the repository

3. **Navigate to Secrets**
   - In the left sidebar, click **"Secrets and variables"**
   - Then click **"Actions"**

4. **Add New Secret**
   - Click the **"New repository secret"** button (top right)
   - **Name**: Enter exactly: `OPENAI_API_KEY`
     - ⚠️ **Important**: The name must match exactly (case-sensitive)
   - **Secret**: Paste your OpenAI API key
     - You can get your API key from: https://platform.openai.com/api-keys
   - Click **"Add secret"**

5. **Verify the Secret**
   - You should see `OPENAI_API_KEY` listed in the secrets
   - The value will be hidden (showing only `••••••••`)
   - You can click on it to update or delete it later

### Method 2: Via GitHub CLI (Alternative)

If you prefer using the command line:

```bash
# Install GitHub CLI if you haven't already
# brew install gh  # macOS
# or download from: https://cli.github.com/

# Authenticate with GitHub
gh auth login

# Set the secret (replace YOUR_API_KEY with your actual key)
gh secret set OPENAI_API_KEY --repo YOUR_USERNAME/YOUR_REPO_NAME --body "YOUR_API_KEY"
```

## Step 2: Verify the Workflow File

Make sure the workflow file exists and is correct:

1. **Check the workflow file exists:**
   ```bash
   ls -la .github/workflows/regression_test.yml
   ```

2. **Verify it references the secret correctly:**
   - The workflow should have: `${{ secrets.OPENAI_API_KEY }}`
   - This is already set up in `.github/workflows/regression_test.yml`

## Step 3: Test the GitHub Actions Workflow

### Option A: Test by Pushing to Main/Master Branch

1. **Make a small change to trigger the workflow:**
   ```bash
   # Make a small change (e.g., update a comment)
   echo "# Test commit" >> README.md
   
   # Commit and push
   git add README.md
   git commit -m "test: trigger CI workflow"
   git push origin main  # or 'master' depending on your default branch
   ```

2. **Check GitHub Actions:**
   - Go to your repository on GitHub
   - Click on the **"Actions"** tab (top navigation)
   - You should see a new workflow run appear
   - Click on it to see the progress
   - The workflow will show:
     - ✅ Green checkmark if all tests pass
     - ❌ Red X if tests fail
     - 🟡 Yellow circle if still running

3. **View Logs:**
   - Click on the workflow run
   - Click on the **"test"** job
   - Expand each step to see detailed logs
   - Look for:
     - "✓ .env file created successfully" (confirms API key was set)
     - Benchmark execution logs
     - Comparison results

### Option B: Test via Pull Request

1. **Create a test branch:**
   ```bash
   git checkout -b test-ci-workflow
   ```

2. **Make a small change:**
   ```bash
   echo "# CI test" >> README.md
   git add README.md
   git commit -m "test: verify CI workflow"
   ```

3. **Push and create PR:**
   ```bash
   git push origin test-ci-workflow
   ```
   - Then create a Pull Request on GitHub
   - The workflow will run automatically on the PR

4. **Check the PR:**
   - The PR will show a status check
   - Click "Details" to see the workflow run

### Option C: Manual Workflow Trigger (if enabled)

If you've enabled manual triggers in the workflow:

1. Go to **Actions** tab
2. Select **"Regression Test"** workflow
3. Click **"Run workflow"** button
4. Select branch and click **"Run workflow"**

## Step 4: Verify the Workflow is Working

### Success Indicators:

✅ **Workflow completes successfully if:**
- All steps show green checkmarks
- "Compare results with ground truth" step shows: `✓ SUCCESS: All results match ground truth!`
- Exit code is 0

❌ **Workflow fails if:**
- API key is missing or incorrect (you'll see authentication errors)
- Tests don't match ground truth (you'll see which combinations failed)
- Python/dependency errors (check installation step)

### Common Issues and Solutions:

#### Issue 1: "OPENAI_API_KEY not found"
**Solution:**
- Double-check the secret name is exactly `OPENAI_API_KEY` (case-sensitive)
- Make sure you added it to the correct repository
- Verify you have admin access to the repository

#### Issue 2: "Authentication failed" or "Invalid API key"
**Solution:**
- Verify your API key is correct
- Check if the API key has expired or been revoked
- Make sure there are no extra spaces when copying the key
- Regenerate the key if needed: https://platform.openai.com/api-keys

#### Issue 3: "Tests don't match ground truth"
**Solution:**
- This is expected if you've made code changes
- Run tests locally first to verify the new results
- Update `CI-tests/ground_truth.json` with the new expected results
- Commit and push the updated ground truth

#### Issue 4: "Workflow not running"
**Solution:**
- Check that the workflow file is in `.github/workflows/`
- Verify the branch name matches (main/master)
- Make sure you've pushed to the correct branch
- Check GitHub Actions is enabled for your repository (Settings → Actions)

## Step 5: Monitor Workflow Runs

### Viewing Workflow History:

1. Go to **Actions** tab
2. Click on **"Regression Test"** in the left sidebar
3. You'll see all workflow runs with:
   - Status (✅/❌/🟡)
   - Commit message
   - Duration
   - Who triggered it

### Setting up Notifications:

1. Go to repository **Settings** → **Notifications**
2. Enable notifications for:
   - Workflow runs
   - Failed workflows
   - Workflow approvals (if required)

## Step 6: Best Practices

1. **Keep API Key Secure:**
   - Never commit API keys to the repository
   - Use GitHub Secrets for all sensitive data
   - Rotate keys periodically

2. **Monitor API Usage:**
   - Check OpenAI dashboard for usage: https://platform.openai.com/usage
   - Set up usage alerts if needed
   - CI runs can consume API credits

3. **Update Ground Truth:**
   - Always update `ground_truth.json` when making intentional changes
   - Test locally before pushing
   - Document why results changed

4. **Review Workflow Logs:**
   - Check logs regularly for warnings
   - Look for rate limit errors
   - Monitor execution time

## Troubleshooting Commands

If you need to debug locally:

```bash
# Test the workflow commands locally
export OPENAI_API_KEY="your-key-here"
echo "OPENAI_API_KEY=${OPENAI_API_KEY}" > .env

# Run the benchmark
python scripts/run_benchmark.py \
  --test CI-tests/testcases/test1.json \
  --memory-backend none explicit mem0 rag context \
  --defense-type none user_prompt_only no_untrusted_tools limit_memory_length provable_policy \
  --results-dir CI-tests/results \
  --num-workers 1 \
  --force

# Compare results
python CI-tests/compare_results.py \
  --results-dir CI-tests/results \
  --test-file CI-tests/testcases/test1.json
```

## Next Steps

Once the workflow is working:
1. ✅ Set up branch protection rules (optional)
2. ✅ Add status badges to README (optional)
3. ✅ Configure workflow notifications
4. ✅ Document any custom configurations

---

**Need Help?**
- Check GitHub Actions documentation: https://docs.github.com/en/actions
- Review workflow logs for detailed error messages
- Test commands locally first before pushing
