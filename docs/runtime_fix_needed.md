# Runtime Fix Needed: RAMWatcher Export

## Issue Summary
Tests are failing with `AttributeError: module 'src.agent.agent_core' has no attribute 'RAMWatcher'`

## Root Cause
The `RAMWatcher` class is not exported from `src/agent/agent_core.py`, but tests expect it to be available as `src.agent.agent_core.RAMWatcher`.

## Expected Fix
Add the following export to `src/agent/agent_core.py`:

```python
# ... existing imports ...

# Export RAMWatcher for test compatibility
RAMWatcher = RAMWatcher  # or from wherever it's defined
```

Or ensure `RAMWatcher` is properly imported and available in the module's namespace.

## Impact
- Blocks test execution beyond ~73% completion
- Prevents full test suite validation
- Affects `TestSkillTriggers.test_belly_trigger_detection` and potentially other tests

## Verification
After fix, run:
```bash
cd /c/Homework/agent_hackathon/pokemon-md-agent
export PYTHONPATH=/c/Homework/agent_hackathon/pokemon-md-agent/src
python -m pytest tests/test_skill_triggers.py::TestSkillTriggers::test_belly_trigger_detection -v
```

## Priority
High - Required for complete test suite execution.