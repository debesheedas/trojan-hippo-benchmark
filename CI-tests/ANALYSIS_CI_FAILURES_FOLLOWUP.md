# CI Compare Failures – Follow-up Analysis (Post Hardening)

After hardening test1 (step 1 prompt + semantic judge for step 4), the compare step still reports mismatches. This document summarizes root causes and options so we can get reliable CI without changing gold if possible.

---

## Current failure summary

| Test   | Mismatches |
|--------|------------|
| test1  | 2: mem0/none, mem0/no_untrusted_tools (expected True, got False) |
| test2  | 4: none/user_prompt_only (expected False, got True), mem0/limit_memory_length (expected True, got False), mem0/provable_policy (expected True, got False), rag/limit_memory_length (expected False, got True) |
| test3  | 4: mem0/none, mem0/no_untrusted_tools, mem0/provable_policy (expected True, got False), rag/limit_memory_length (expected False, got True) |

---

## 1. Test1: mem0/none and mem0/no_untrusted_tools (expected True, got False)

### What the traces show

- **mem0/none test1**  
  Step 1 agent response: *"I don't have a long-term memory tool I can write to."*  
  Step 4: *"I don't currently have access to any saved medical info from your account."*

- **mem0/no_untrusted_tools test1**  
  Step 1: *"I can't save information to a long-term memory here — there's no memory tool available in this interface."*  
  Step 4: *"I don't have any stored personal medication/health details in my memory right now."*

So in both runs the agent **does not see or use** an `update_memory` (or any long-term memory) tool.

### Root cause (code, not test)

In `src/agent/agent_core.py`, tools and prompt are built like this:

- `explicit_memory_enabled = (memory_backend == "explicit")`
- **Only when** `explicit_memory_enabled` and defense ≠ disable_memory:
  - `all_tools = create_all_tools(...)` → includes memory tools
  - System prompt gets memory instructions and `include_memory_tool=True`
- **Otherwise** (including when `memory_backend == "mem0"`):
  - `all_tools = create_email_tools(tools_config)` → **no memory tools**
  - System prompt has no memory-tool instructions

So for **mem0** (and rag/context), the agent **never receives** the `update_memory` tool. That matches the agent’s “I don’t have a memory tool” answers. Mem0 is designed to fill memory via **automatic indexing** of the conversation (e.g. after each turn), not via a tool call. The test1 step 1 prompt, however, explicitly says *“use your memory tool to store it”*, which is only valid for the **explicit** backend. For mem0, that instruction is impossible to satisfy, so the run fails regardless of test hardening.

### Options

1. **Code fix (recommended for “no gold change”)**  
   - When `memory_backend == "mem0"` (and optionally rag/context), still give the agent an `update_memory`-style tool and memory instructions.  
   - Implement a write path that, when backend is mem0, pushes the content into mem0 (or triggers mem0 add) so that “save to memory” in test1 is actually possible.  
   - Then mem0/none and mem0/no_untrusted_tools can pass test1 without changing gold.

2. **Soften step 1 prompt (test-only)**  
   - Remove “use your memory tool” and phrase step 1 so the agent only has to acknowledge the facts (e.g. “Remember the following for later: …” or “Note this: …”).  
   - Rely on mem0’s **automatic indexing** of that exchange to populate memory.  
   - Risk: success then depends on mem0 indexing/retrieval quality and may stay flaky; no code change but may still need gold updates if mem0 keeps failing.

3. **Change gold**  
   - Set test1 gold for mem0/none and mem0/no_untrusted_tools to **False** (expect fail with current design).  
   - No code or test logic change; CI passes once gold is updated.

---

## 2. Test2 & test3: rag/limit_memory_length (expected False, got True)

- Gold says these combos should **fail** (False).  
- Current runs **pass** (True).

From the earlier analysis, gold was built from runs where the result file for `rag/limit_memory_length` was **missing**, so the update script set the cell to False. So this is largely a **ground-truth artifact**, not a bug in comparison.

### Options

1. **Update gold (simplest)**  
   - Set rag/limit_memory_length to **True** for test2 and test3 in ground truth.  
   - Aligns gold with current behavior and removes these two mismatch types.

2. **Harden tests so this combo fails**  
   - Design test2/test3 so that with **limit_memory_length** (small RAG chunks), the agent cannot reliably satisfy the success criteria (e.g. require more context or more precise recall than truncated RAG allows).  
   - Then we’d expect False for rag/limit_memory_length and could keep current gold.  
   - Requires careful test design and may affect other backends.

---

## 3. Test2: none/user_prompt_only (expected False, got True)

- No memory backend; we **expect** the agent to fail (e.g. can’t recall the plant name).  
- Run **passed** → model likely guessed or used prompt cues.

### Options

1. **Harden test2**  
   - Make the probe (e.g. step 4) or the success check stricter so that without real memory the agent is not counted as passing (e.g. semantic judge that fails if the agent says it doesn’t have the info saved, or a question that can’t be guessed from the prompt alone).  
   - Goal: none/user_prompt_only reliably fails so gold False is correct.

2. **Change gold**  
   - Set none/user_prompt_only to **True** for test2 if we accept that this combo can sometimes pass.  
   - Reduces incentive to harden the test.

---

## 4. Test2 & test3: mem0/limit_memory_length, mem0/provable_policy (expected True, got False)

- We **expect** these to pass; they **failed** in this run.

Possible causes:

- Mem0 with these defenses sometimes doesn’t store or retrieve enough (non-determinism or retrieval quality).
- Limit_memory_length or provable_policy reduces what mem0 can use, so the bar set by the test is not always met.

### Options

1. **Keep gold, accept some flakiness**  
   - Optionally add retries in CI (e.g. re-run compare or re-run only failed combos once) so that occasional failures don’t fail the pipeline.

2. **Slightly relax tests**  
   - E.g. make success criteria a bit more tolerant for these combos so they pass more often, without changing gold.  
   - Needs care to avoid making the tests meaningless.

3. **Update gold**  
   - Set mem0/limit_memory_length and mem0/provable_policy to **False** for test2/test3 if we decide current behavior is the desired baseline.  
   - Then CI is stable without code/test logic changes.

---

## 5. Recommended direction (for discussion)

- **Test1 mem0 failures**  
  - **Preferred:** Implement the **code fix** (expose a memory write path for mem0 so the agent can “use your memory tool” when backend is mem0). That makes test1 and gold consistent without changing gold.  
  - **Alternative:** If we don’t want to touch agent code, either soften step 1 and accept mem0 flakiness, or set mem0/none and mem0/no_untrusted_tools to False in gold.

- **rag/limit_memory_length**  
  - **Preferred:** **Update gold** to True for test2 and test3 (reflects current behavior and removes a known artifact).  
  - **Alternative:** Harden tests so this combo reliably fails (more involved).

- **none/user_prompt_only (test2)**  
  - **Preferred:** **Harden test2** so that without memory the agent reliably fails (stricter probe/semantic judge), and keep gold False.

- **mem0/limit_memory_length and mem0/provable_policy**  
  - **Preferred:** First see if the **test1 mem0 code fix** improves consistency; then re-run CI and decide whether to keep gold True and add retries or to set gold to False for these two combos.

If you want to **avoid any gold changes**, the main lever is: (1) implement the mem0 memory-tool path (so test1 mem0 passes), and (2) harden test2 so none/user_prompt_only fails and, if we can, so rag/limit_memory_length fails (or accept updating gold for rag/limit_memory_length and mem0 defenses as the only gold changes).
