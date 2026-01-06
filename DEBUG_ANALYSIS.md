# Debug Analysis: Provable Policy Defense Issue

## Problem
The provable_policy defense is not blocking `send_email` even though:
1. The session is correctly upgraded to U when U-labeled memories are loaded (log line 395)
2. The defense logic appears correct in the code

## Code Flow Analysis

### 1. Agent Creation (Session 2)
- `_create_agent_executor_for_python()` is called with `session_id=bench_25d43488`
- Line 355: `tools_config.session_id = session_id` ✅ Sets session_id on tools config
- Line 377-384: Tools are created with `tools_config` (which has session_id) ✅
- Line 494-497: `get_long_term_as_text()` is called, upgrades session to U ✅

### 2. When send_email is Called
- Line 1022-1023: Gets `session_id` from `self.config.session_id`
- Line 1035: Calls `ProvablePolicyManager.is_trusted(session_id)`
- Line 1039: Should block if `not is_trusted`

## Debug Statements Added

### In `memory_manager.py` - `get_long_term_as_text()`:
- Logs when checking memories for U labels
- Shows each memory's label and text preview
- Logs when U-labeled memory is found and session is upgraded
- Logs if no U-labeled memories are found

### In `email_tools.py` - `send_email._run()`:
- Logs the session_id received by the tool
- Logs whether config exists and its type
- Logs the result of `is_trusted()` and session label
- Logs whether the tool will be blocked

## Expected Debug Output

When the benchmark runs, you should see:

### During Agent Initialization (Step 11):
```
🔍 [DEBUG] get_long_term_as_text: Checking X memories for U labels (session_id=bench_25d43488)
🔍 [DEBUG] Memory 0: label=U, text_preview='User practices yoga...'
🛡️ [DEBUG] Found U-labeled memory! Upgrading session 'bench_25d43488' to UNTRUSTED
🛡️ [Provable Policy] Session 'bench_25d43488' upgraded to UNTRUSTED (U)
```

### When send_email is Called (Step 13):
```
🔍 [DEBUG] send_email: session_id=bench_25d43488, config=True, config_type=EmailToolsConfig
🔍 [DEBUG] send_email: config.session_id=bench_25d43488
🔍 [DEBUG] send_email: is_trusted=False, session_label=U, will_block=True
```

## Potential Issues to Check

1. **If `session_id=None` in send_email debug output:**
   - The `tools_config.session_id` isn't being set correctly
   - The tool config isn't being passed correctly to the tool

2. **If `is_trusted=True` when it should be False:**
   - The session label isn't being upgraded correctly
   - There's a bug in `ProvablePolicyManager.is_trusted()`
   - The session_id being checked is different from the one that was upgraded

3. **If memories don't have U labels:**
   - Memories aren't being saved with labels correctly
   - The memory file format is incorrect

## Next Steps

1. Run the benchmark with the debug statements
2. Check the debug output to identify where the issue occurs
3. Fix the issue based on the debug output

## Files Modified

- `src/agent/backend/memory_manager.py`: Added debug logging in `get_long_term_as_text()`
- `src/agent/tool_specifications/email_tools.py`: Added debug logging in `send_email._run()`

