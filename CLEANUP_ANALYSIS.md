# Code Cleanup Analysis

## Summary
This document outlines code cleanup opportunities found in the `src/` directory, including code deduplication, dead code removal, and other improvements.

---

## 1. Code Deduplication Opportunities

### 1.1 UNTRUSTED_TOOLS Constant Duplication
**Issue**: `UNTRUSTED_TOOLS` is defined in multiple places:
- `src/agent/tools_registry.py` (line 14)
- `src/agent/tool_specifications/memory_tools.py` (line 17)

**Impact**: Medium - Risk of inconsistency if one is updated but not the other

**Solution**: 
- Keep definition in `tools_registry.py` (central location)
- Import from `tools_registry` in `memory_tools.py`
- Remove duplicate definition

**Files to modify**:
- `src/agent/tool_specifications/memory_tools.py` - Remove duplicate, add import

---

### 1.2 Memory Context Retrieval Pattern
**Issue**: Similar patterns for retrieving memory context in `agent_core.py`:
- Lines 472-493: Debug code that retrieves memory context (commented out or unused)
- Lines 510-575: Active memory context retrieval for RAG and mem0

**Impact**: Low - Debug code is already commented, but could be cleaner

**Solution**: 
- Remove commented debug code (lines 472-497)
- The active code is necessary and well-structured

**Files to modify**:
- `src/agent/agent_core.py` - Remove commented debug section

---

### 1.3 Defense Manager Pattern Similarity
**Issue**: `Mem0DefenseManager` and `RAGDefenseManager` have similar structure:
- Both check `no_untrusted_tools` using `SessionTrustManager`
- Both have `should_index_memory` and filtering methods

**Impact**: Low - They serve different backends, some duplication is acceptable

**Solution**: 
- Keep as-is (different backends may diverge in future)
- Consider extracting common logic to a base class if more defenses are added

---

### 1.4 Duplicate Cache Declaration
**Issue**: In `mem0_memory_manager.py`:
- Line 598: `_mem0_manager_cache: Dict[str, Mem0MemoryManager] = {}`
- Line 602: Duplicate declaration (same line repeated)

**Impact**: Low - Python allows this, but it's redundant

**Solution**: Remove duplicate declaration

**Files to modify**:
- `src/agent/backend/mem0_memory_manager.py` - Remove duplicate line 602

---

### 1.5 Session Trust Checking Pattern
**Issue**: Session trust checking appears in multiple places:
- `agent_core.py`: `SessionTrustManager` class
- `mem0_defense_manager.py`: Uses `SessionTrustManager`
- `rag_defense_manager.py`: Uses `SessionTrustManager`
- `email_tools.py`: Multiple tools call `SessionTrustManager.set_untrusted()`

**Impact**: Low - This is intentional design (centralized trust management)

**Solution**: Keep as-is - this is good architecture

---

## 2. Dead Code Removal

### 2.1 Commented Debug Code in agent_core.py
**Issue**: Lines 472-497 contain commented-out debug code for printing system prompt

**Impact**: Low - Dead code that can be removed

**Solution**: Remove commented debug section

**Files to modify**:
- `src/agent/agent_core.py` - Remove lines 472-497

---

### 2.2 Unused Import Check
**Issue**: Need to verify if all imports are used

**Impact**: Low - Minor cleanup

**Solution**: Review imports in each file and remove unused ones

**Files to check**:
- All Python files in `src/`

---

## 3. Code Organization Improvements

### 3.1 Long Function Parameters
**Issue**: Some functions have very long parameter lists:
- `get_mem0_memory_manager()`: 10+ parameters
- `get_rag_memory_manager()`: 6+ parameters
- `_create_agent_executor_for_python()`: Complex nested config handling

**Impact**: Low - These are factory functions, long parameter lists are acceptable

**Solution**: Keep as-is - refactoring would require significant changes

---

### 3.2 Error Handling Consolidation
**Issue**: Similar try-except patterns repeated in multiple places:
- Memory manager initialization
- Tool execution
- API calls

**Impact**: Low - Error handling is context-specific

**Solution**: Keep as-is - context-specific error handling is appropriate

---

## 4. Constants and Configuration

### 4.1 Hardcoded Values
**Issue**: Some hardcoded values that could be constants:
- `"vince"` as user_id (appears in multiple places)
- `80` as default `limit_memory_size`
- `15` as `max_short_term` messages

**Impact**: Low - These are reasonable defaults

**Solution**: Consider extracting to constants if they're used in many places

---

## 5. Recommended Cleanup Actions

### High Priority (Safe, High Impact)
1. ✅ Remove duplicate `UNTRUSTED_TOOLS` definition in `memory_tools.py`
2. ✅ Remove duplicate cache declaration in `mem0_memory_manager.py`
3. ✅ Remove commented debug code in `agent_core.py` (lines 472-497)

### Medium Priority (Safe, Medium Impact)
4. Review and remove unused imports across all files
5. Extract hardcoded `"vince"` user_id to a constant if used >5 times

### Low Priority (Requires Testing)
6. Consider extracting common defense logic if more defenses are added
7. Consider consolidating memory context retrieval if patterns become more similar

---

## Testing Strategy

After cleanup:
1. Run existing tests to ensure nothing broke
2. Run a sample benchmark to verify functionality
3. Test interactive agent to ensure it still works
4. Check that all memory backends (explicit, mem0, rag) still function

---

## Files Summary

**Files to modify**:
- `src/agent/tool_specifications/memory_tools.py` - Remove duplicate UNTRUSTED_TOOLS
- `src/agent/backend/mem0_memory_manager.py` - Remove duplicate cache declaration
- `src/agent/agent_core.py` - Remove commented debug code

**Files to review** (for unused imports):
- All files in `src/agent/`
- All files in `src/benchmark/`
- All files in `src/interactive_agent/`

