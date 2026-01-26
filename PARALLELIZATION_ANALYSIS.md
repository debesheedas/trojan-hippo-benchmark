# Parallelization Analysis for OpenEvolve Optimizer

## Current Implementation

### Early Stopping
✅ **Partially Implemented**
- Line 630-633: Checks if `agentdojo_score >= early_stop_score` and breaks immediately
- Line 678: Also checks `binary_success` for final success determination
- **Issue**: Does NOT check `binary_success` immediately after scoring (only checks score)
- **Recommendation**: Add `binary_success` check right after line 764 (when candidate.binary_success is set)

### Sequential Flow
1. **Initial candidate scoring** (line 436) - Sequential
2. **Generate diverse initial candidates** (line 450-485) - Sequential (mutator generates 10 at once)
3. **Score initial candidates** (line 487-520) - Sequential loop
4. **Main optimization loop** (line 525-667):
   - Sample parents (line 540) - Sequential
   - Generate variants via mutator (line 550-580) - Sequential (mutator generates batch at once)
   - Score variants (line 590-633) - **SEQUENTIAL LOOP** ⚠️
   - Add to database (line 620) - Sequential

## Parallelization Opportunities

### 1. **Candidate Scoring** (HIGHEST PRIORITY) ⭐
**Location**: Line 590-633 in `openevolve_optimizer.py`

**Current**: Sequential loop scoring each variant one by one
```python
for i, variant_email in enumerate(new_variants, 1):
    # ... create candidate ...
    self._score_candidate(...)  # Blocks until complete
    # ... add to database ...
```

**Parallelizable**: YES - Each candidate scoring is completely independent:
- Creates fresh in-memory environment
- Runs full test sequence
- No shared state between candidates
- Only writes to database at end (can be synchronized)

**Benefits**:
- With 4 workers and 8 candidates: ~2x speedup (from ~16 min to ~8 min)
- Each candidate takes 1-2 minutes, so 4 workers can process 4 simultaneously

**Challenges**:
- Need thread-safe logging
- Need synchronized database access
- Need proper progress tracking
- Early stopping needs coordination (if one succeeds, cancel others)

### 2. **Initial Candidate Scoring** (MEDIUM PRIORITY)
**Location**: Line 487-520 in `openevolve_optimizer.py`

**Current**: Sequential loop scoring 10 initial candidates
```python
for i, candidate_email in enumerate(initial_candidates, 1):
    # ... create candidate ...
    self._score_candidate(...)  # Blocks until complete
    # ... add to database ...
```

**Parallelizable**: YES - Same as above, completely independent

**Benefits**:
- With 4 workers and 10 candidates: ~2.5x speedup (from ~20 min to ~8 min)

### 3. **Mutator Generation** (LOW PRIORITY)
**Location**: Line 550-580 in `openevolve_optimizer.py`

**Current**: Single LLM call generates batch of variants at once

**Parallelizable**: PARTIALLY - Could generate multiple batches in parallel, but:
- Mutator already generates 10 variants in one call
- LLM API rate limits might be hit
- Less benefit than candidate scoring

**Recommendation**: Skip for now, focus on candidate scoring

## Design Proposal

### Configuration
Add to `benchmark_config.yaml`:
```yaml
openevolve:
  num_workers: 4  # Number of parallel workers for candidate scoring (1 = sequential)
  # ... existing config ...
```

### Implementation Strategy

#### Option 1: ThreadPoolExecutor (Recommended)
**Pros**:
- Simple to implement
- Good for I/O-bound tasks (LLM API calls)
- Easy to cancel remaining tasks on early stop
- Shared memory (can share logger, database with locks)

**Cons**:
- Python GIL limits CPU-bound parallelism (but we're I/O-bound)
- Need thread-safe logging

**Implementation**:
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Thread-safe logger wrapper
class ThreadSafeLogger:
    def __init__(self, logger):
        self.logger = logger
        self.lock = threading.Lock()
    
    def info(self, msg):
        with self.lock:
            self.logger.info(msg)
    # ... other methods ...

# Thread-safe database wrapper
class ThreadSafeDatabase:
    def __init__(self, database):
        self.database = database
        self.lock = threading.Lock()
    
    def add(self, candidate):
        with self.lock:
            return self.database.add(candidate)
```

#### Option 2: ProcessPoolExecutor
**Pros**:
- True parallelism (no GIL)
- Process isolation (no shared state issues)

**Cons**:
- More complex (need to serialize/deserialize candidates, results)
- Cannot share logger easily (need queue-based logging)
- Cannot share database easily (need IPC)
- Higher overhead for process creation

**Recommendation**: Use ThreadPoolExecutor (we're I/O-bound, not CPU-bound)

### Detailed Design

#### 1. Thread-Safe Components

**ThreadSafeLogger**:
- Wraps existing logger with lock
- All log calls synchronized
- Preserves log order per thread

**ThreadSafeDatabase**:
- Wraps CandidateDatabase with lock
- `add()` method synchronized
- `get_best()`, `get_statistics()` synchronized

**Early Stopping Coordination**:
- Use `threading.Event()` to signal early stop
- When one candidate succeeds (binary_success=True or score>=10):
  - Set event
  - Cancel remaining futures
  - Return immediately

#### 2. Parallel Scoring Function

```python
def _score_candidate_parallel(self, candidate, ...):
    """Score a single candidate - designed for parallel execution."""
    try:
        # Check early stop event
        if self._early_stop_event.is_set():
            return None  # Cancelled
        
        # Score candidate (creates fresh environment, no shared state)
        result = self.scorer.score(...)
        
        # Update candidate
        candidate.agentdojo_score = result.agentdojo_critic_score or 1
        candidate.partial_score = result.partial_score
        candidate.binary_success = result.attack_goal_passed
        
        # Check for early stop
        if candidate.binary_success or candidate.agentdojo_score >= self.early_stop_score:
            self._early_stop_event.set()
        
        return candidate
    except Exception as e:
        self._thread_safe_logger.warning(f"Failed to score candidate {candidate.id}: {e}")
        return None
```

#### 3. Parallel Scoring Loop

```python
def _score_candidates_parallel(self, candidates, ...):
    """Score multiple candidates in parallel."""
    if self.num_workers == 1:
        # Sequential mode (for debugging)
        return self._score_candidates_sequential(candidates, ...)
    
    # Parallel mode
    self._early_stop_event = threading.Event()
    scored_candidates = []
    
    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
        # Submit all scoring tasks
        future_to_candidate = {
            executor.submit(self._score_candidate_parallel, candidate, ...): candidate
            for candidate in candidates
        }
        
        # Process completed tasks
        for future in as_completed(future_to_candidate):
            if self._early_stop_event.is_set():
                # Cancel remaining tasks
                for f in future_to_candidate:
                    f.cancel()
                break
            
            candidate = future_to_candidate[future]
            try:
                scored = future.result()
                if scored:
                    scored_candidates.append(scored)
                    # Add to database (thread-safe)
                    self._thread_safe_database.add(scored)
                    
                    # Log progress
                    self._thread_safe_logger.info(
                        f"   Variant {len(scored_candidates)}/{len(candidates)}: "
                        f"score={scored.agentdojo_score}/10"
                    )
            except Exception as e:
                self._thread_safe_logger.warning(f"Scoring failed for candidate: {e}")
    
    return scored_candidates
```

#### 4. Progress Tracking

**Per-Candidate Logging**:
- Each worker logs when it starts/finishes a candidate
- Include candidate ID in logs for debugging
- Use thread-safe logger

**Progress Summary**:
- Track completed/total candidates
- Log periodically (every N candidates or every N seconds)
- Show estimated time remaining

**Example Log Format**:
```
[Worker-1] Scoring candidate abc123...
[Worker-2] Scoring candidate def456...
[Worker-1] Completed candidate abc123: score=5/10
[Worker-3] Scoring candidate ghi789...
[Worker-2] Completed candidate def456: score=7/10
Progress: 2/8 candidates completed (25%)
```

#### 5. Early Stopping Coordination

**Immediate Stop on Success**:
```python
# In _score_candidate_parallel:
if candidate.binary_success or candidate.agentdojo_score >= self.early_stop_score:
    self._early_stop_event.set()
    self._thread_safe_logger.info(f"SUCCESS! Candidate {candidate.id} achieved score {candidate.agentdojo_score}/10")
    return candidate  # Return immediately

# In _score_candidates_parallel:
if self._early_stop_event.is_set():
    # Find the successful candidate
    for candidate in scored_candidates:
        if candidate.binary_success or candidate.agentdojo_score >= self.early_stop_score:
            return [candidate]  # Return only the successful one
```

### Configuration Integration

**benchmark_config.yaml**:
```yaml
openevolve:
  num_workers: 4  # Parallel workers (1 = sequential, 4 = recommended, 8+ = aggressive)
  # ... existing config ...
```

**Code Changes**:
```python
# In __init__:
self.num_workers = self.openevolve_config.get("num_workers", 1)  # Default to 1 for safety

# Initialize thread-safe components if num_workers > 1
if self.num_workers > 1:
    self._thread_safe_logger = ThreadSafeLogger(self.logger)
    self._thread_safe_database = ThreadSafeDatabase(self.database)
    self._early_stop_event = None  # Created per scoring batch
```

### Testing Strategy

1. **num_workers=1**: Should behave exactly like current implementation
2. **num_workers=4**: Should complete ~4x faster (with 8+ candidates)
3. **Early stopping**: Should stop immediately when success found
4. **Logging**: Should be readable and not interleaved incorrectly
5. **Database**: Should maintain MAP Elites correctness

### Potential Issues & Solutions

1. **LLM API Rate Limits**:
   - Solution: Add rate limiting per worker
   - Or: Use semaphore to limit concurrent API calls

2. **Memory Usage**:
   - Each worker creates fresh in-memory environment
   - With 4 workers: 4x memory (should be fine, environments are small)

3. **Log Interleaving**:
   - Solution: Thread-safe logger with locks
   - Or: Use queue-based logging

4. **Database Race Conditions**:
   - Solution: Thread-safe database wrapper with locks
   - MAP Elites algorithm is already safe (just dict operations)

5. **Early Stop Coordination**:
   - Solution: Use threading.Event() to signal all workers
   - Cancel remaining futures immediately

### Performance Estimates

**Current (Sequential)**:
- 10 initial candidates × 2 min = 20 minutes
- 5 iterations × 8 candidates × 2 min = 80 minutes
- Total: ~100 minutes

**With 4 Workers (Parallel)**:
- 10 initial candidates ÷ 4 × 2 min = 5 minutes
- 5 iterations × 8 candidates ÷ 4 × 2 min = 20 minutes
- Total: ~25 minutes (4x speedup)

**With Early Stop**:
- If success found early: Could save 50-75% of time
- Combined with parallelization: Even better

## Implementation Checklist

- [ ] Add `num_workers` config option
- [ ] Create ThreadSafeLogger wrapper
- [ ] Create ThreadSafeDatabase wrapper  
- [ ] Create `_score_candidate_parallel()` method
- [ ] Create `_score_candidates_parallel()` method
- [ ] Add early stop event coordination
- [ ] Update initial candidate scoring to use parallel
- [ ] Update iteration variant scoring to use parallel
- [ ] Add progress tracking/logging
- [ ] Test with num_workers=1 (should match current behavior)
- [ ] Test with num_workers=4
- [ ] Test early stopping
- [ ] Verify thread safety

## Code Structure

```
openevolve_optimizer.py:
  - ThreadSafeLogger (new class)
  - ThreadSafeDatabase (new class)
  - OpenEvolveOptimizer:
    - __init__: Initialize thread-safe components if num_workers > 1
    - _score_candidate_parallel: Parallel-safe scoring
    - _score_candidates_parallel: Parallel scoring loop
    - _score_candidates_sequential: Fallback for num_workers=1
    - optimize_attack: Use parallel scoring when num_workers > 1
```
