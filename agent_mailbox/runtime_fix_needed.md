# Runtime Fix Needed: RAMWatcher Import Error

## Issue Summary
Current test suite fails on `TestSkillTriggers.test_belly_trigger_detection` due to missing `RAMWatcher` class in `src.agent.agent_core`.

## Failing Test Details
**Test**: `TestSkillTriggers.test_belly_trigger_detection`
**Error**: `AttributeError: module 'src.agent.agent_core' has no attribute 'RAMWatcher'`
**Traceback Header**:
```
ERROR at setup of TestSkillTriggers.test_belly_trigger_detection:
AttributeError: module 'src.agent.agent_core' has no attribute 'RAMWatcher'
```

**Current pytest invocation**:
```bash
python -m pytest -q --maxfail=1 -m "not slow and not network and not bench and not longctx"
```

## Required Fix
**Import path required by tests**: `from src.agent.agent_core import RAMWatcher`

**Expected minimal fix by runtime team**:
- Export `RAMWatcher` in `agent_core` module (re-export from another module or implement directly)
- Ensure the exported class matches the test API expectations
- This is a runtime-owned issue; tests are correct and should not be modified

## Impact
- Blocks full test suite execution
- Fast lane tests pass until this point (~77% completion)
- Prevents validation of skill trigger functionality

## Additional Issue: Missing evaluate_condition Method

**New Issue Discovered**: `TestRAMPredicates.test_evaluate_condition_comparison` fails due to missing `evaluate_condition` method in `RAMPredicates` class.

**Failing Test**: `tests/test_skills.py::TestRAMPredicates::test_evaluate_condition_comparison`
**Error**: `AttributeError: 'RAMPredicates' object has no attribute 'evaluate_condition'`
**Impact**: Test suite fails at ~59% completion, blocking further validation

**Required Fix**: Implement `evaluate_condition` method in `RAMPredicates` class to match test expectations.

## Next Steps
1. Runtime team implements/fixes `RAMWatcher` export in `src/agent/agent_core.py`
2. Runtime team implements `evaluate_condition` method in `RAMPredicates` class
3. Re-run test suite to validate fixes
4. Update this note once resolved</content>
<parameter name="filePath">C:\Homework\agent_hackathon\pokemon-md-agent\agent_mailbox\runtime_fix_needed.md