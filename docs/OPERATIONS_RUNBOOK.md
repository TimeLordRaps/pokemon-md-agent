# Operations Runbook: Dashboard & Memory System

## Quick Start

### 1. Verify System is Operational
```bash
cd pokemon-md-agent

# Run core tests (should see 99 passed in <10s)
python -m pytest tests/test_on_device_buffer.py tests/test_circular_buffer.py \
  tests/test_local_ann_index.py tests/test_content_api.py \
  tests/test_embeddings.py tests/test_auto_retrieve.py -q

# Check You.com API access
python scripts/check_you_api.py --live

# Verify budget
python -c "from src.dashboard.content_api import BudgetTracker; \
  b = BudgetTracker(); \
  print(f'Budget: {b.used_this_month}/{b.monthly_limit} used, {b.remaining()} remaining')"
```

### 2. Check GitHub Pages Accessibility
```bash
# Test main page
curl -I https://[username].github.io/pokemon-md-agent/

# Test documentation index
curl https://[username].github.io/pokemon-md-agent/docs/

# Test demo video accessible
curl -I https://[username].github.io/pokemon-md-agent/assets/agent_demo.mp4
```

---

## Daily Checks (Morning)

### 1. Budget Status
```python
from src.dashboard.content_api import BudgetTracker
import json
from pathlib import Path

bt = BudgetTracker()
print(f"Monthly Budget: {bt.used_this_month}/{bt.monthly_limit}")
print(f"Remaining: {bt.remaining()} calls")
print(f"Daily burn rate estimate: {bt.used_this_month / 31:.1f} calls/day")

# Warning thresholds
if bt.remaining() < 100:
    print("WARNING: Less than 100 calls remaining!")
if bt.remaining() < 50:
    print("CRITICAL: Less than 50 calls remaining!")
```

### 2. System Health
```bash
# Check local storage
du -sh ~/.cache/pmd-red/

# Verify budget file integrity
python -c "
import json
from pathlib import Path
f = Path.home() / '.cache' / 'pmd-red' / 'youcom_budget.json'
with open(f) as fh:
    data = json.load(fh)
    print('Budget file OK')
    print(f'  Used: {data[\"used_this_month\"]}')
    print(f'  Month started: {data[\"month_start\"]}')
"
```

### 3. Dashboard Deployment Status
```bash
# Check if changes need deployment
cd docs/docs
git status

# If changes found:
cd ../..
scripts/finalize_and_snapshot.sh

# Verify deployment
git status  # Should show clean working tree
```

---

## Memory System Monitoring

### Check Circular Buffer Health
```python
from src.retrieval.circular_buffer import CircularBuffer

# Default configuration
buffer = CircularBuffer(
    window_seconds=3600,    # 60 minutes
    max_entries=1000,
    enable_async=True
)

# Monitor during operation
print(f"Entries: {len(buffer._frames)}")
print(f"Fill rate: {len(buffer._frames) / buffer.max_entries * 100:.1f}%")
print(f"Oldest frame age: {time.time() - buffer._frames[0][1] if buffer._frames else 'N/A'}s")
```

### Check On-Device Buffer Stuckness
```python
from src.retrieval.on_device_buffer import OnDeviceBuffer

buffer = OnDeviceBuffer(max_entries=1000, ttl_minutes=60)

# During operation, monitor stuckness
stats = buffer.stats()
print(f"Stuckness score: {stats['stuckness_score']:.3f}")
print(f"Is stuck: {stats['is_stuck']}")
print(f"Entries: {stats['total_entries']}")
print(f"Capacity: {stats['capacity_utilization']:.1%}")

# If stuck:
# 1. Check if agent is in a loop (visual inspection of game)
# 2. Trigger stickiness recovery: agent reset or force action
```

### Check ANN Index Performance
```python
import time
from src.retrieval.local_ann_index import LocalANNIndex
import numpy as np

ann = LocalANNIndex(dimension=384)

# Benchmark query performance
query_vec = np.random.rand(384).astype(np.float32)
start = time.time()
results = ann.search(query_vec, k=5)
latency_ms = (time.time() - start) * 1000

print(f"ANN query latency: {latency_ms:.2f}ms")
print(f"Results found: {len(results)}")

# Alert if degraded
if latency_ms > 100:
    print("WARNING: ANN query latency degraded!")
```

---

## Gatekeeper Monitoring

### Check Gate Token Status
```python
from src.retrieval.gatekeeper import RetrievalGatekeeper

gk = RetrievalGatekeeper(max_tokens_per_hour=1000)

stats = gk.get_stats()
print(f"Active tokens: {stats['active_tokens']}")
print(f"Hourly usage: {stats['hourly_usage']}/{stats['budget_remaining']} remaining")
print(f"Cache size: {stats['cache_size']}")

# Tokens should expire after token_lifetime_seconds (default 300s)
# Usage should reset hourly
```

### Test Gate Logic
```python
from src.retrieval.gatekeeper import RetrievalGatekeeper

gk = RetrievalGatekeeper()

# Test shallow checks
query = "how to defeat Articuno in Pokemon Mystery Dungeon"
context = {
    "shallow_hits": 5,  # >= 3 required
    "game_state": {"floor": 3, "hp": 20}
}

status, token, metadata = gk.check_and_gate(query, context)
print(f"Gate status: {status.value}")
print(f"Confidence: {metadata.get('shallow_confidence', 0):.2f}")
print(f"Reasons: {metadata.get('shallow_reasons', [])}")

if token:
    print(f"Gate token created: {token.token_id}")
```

---

## Content API Monitoring

### Check API Status
```python
from src.dashboard.content_api import ContentAPI
import asyncio

async def check_api():
    api = ContentAPI(api_key="YOUR_YOU_COM_KEY", mock_mode=False)

    # Test a simple fetch
    pages = await api.fetch([
        "https://[username].github.io/pokemon-md-agent/docs/"
    ])

    if pages and pages[0].error is None:
        print(f"API OK: Fetched {len(pages[0].content)} chars")
        print(f"Status: {pages[0].status_code}")
    else:
        print(f"API ERROR: {pages[0].error}")

    print(f"Budget remaining: {api.get_budget_status()['remaining']}")

asyncio.run(check_api())
```

### Monitor Request Rate
```python
# The API uses TokenBucket for rate limiting (default 10 RPS)
from src.dashboard.content_api import ContentAPI

api = ContentAPI(rps_limit=10)  # 10 requests per second

# Check rate limiter stats
print(f"Rate limit: {api.rps_limit} RPS")
print(f"Burst capacity: {api.rate_limiter.capacity}")

# Should automatically throttle if >10 RPS attempted
```

---

## Troubleshooting

### Issue: "Gatekeeper blocked: insufficient_shallow_hits"

**Diagnosis**:
```python
from src.retrieval.on_device_buffer import OnDeviceBuffer

buffer = OnDeviceBuffer()
# The buffer should have entries from the agent's recent activity
stats = buffer.stats()
print(f"Buffer entries: {stats['total_entries']}")

# If empty, the buffer hasn't received data yet
# If populated but no matches, query doesn't match recent activity
```

**Resolution**:
1. Ensure agent has been running for >1 minute (to populate buffer)
2. Ensure query is related to recent game activity
3. Check buffer isn't in stuckness state

---

### Issue: "Budget exceeded - no more API calls allowed"

**Diagnosis**:
```python
from src.dashboard.content_api import BudgetTracker

bt = BudgetTracker()
print(f"Used: {bt.used_this_month}")
print(f"Limit: {bt.monthly_limit}")

# Check if we're at month boundary
import time
from datetime import datetime
current_month = time.gmtime(time.time()).tm_mon
start_month = time.gmtime(bt.month_start).tm_mon
print(f"Current month: {current_month}, Start month: {start_month}")
```

**Resolution**:
1. If at month boundary, budget should reset automatically
2. If over budget mid-month, contact team lead to increase quota
3. Manual reset (testing only): `rm ~/.cache/pmd-red/youcom_budget.json`

---

### Issue: "ANN query latency >100ms"

**Diagnosis**:
```python
from src.retrieval.local_ann_index import LocalANNIndex

ann = LocalANNIndex()
print(f"Index size: {ann._index.ntotal if hasattr(ann._index, 'ntotal') else 'unknown'}")
print(f"Index trained: {hasattr(ann._index, 'is_trained') and ann._index.is_trained}")

# FAISS flat index performance degrades with size
# Expect ~10ms for 10k entries, ~50ms for 100k entries
```

**Resolution**:
1. If index >100k entries, consider switching to IVF index: `LocalANNIndex(index_type='ivf')`
2. Prune old entries from buffer
3. Check system resource contention (CPU, memory)

---

### Issue: "GitHub Pages deployment failed"

**Diagnosis**:
```bash
# Check git status
cd docs/docs
git status

# Check if gh-pages branch exists
git branch -a | grep gh-pages

# Check remote origin
git remote -v
```

**Resolution**:
1. Ensure gh-pages branch exists: `git branch gh-pages origin/gh-pages`
2. Ensure GitHub Actions workflow is configured
3. Manual deployment: `scripts/finalize_and_snapshot.sh`
4. If still failing, check GitHub Actions logs in web UI

---

## Performance Baseline

Expected latencies (p95) for reference:

| Component | Latency | Typical | Alert Threshold |
|-----------|---------|---------|-----------------|
| Buffer search | <1ms | 0.1ms | >10ms |
| ANN query | <50ms | 5ms | >100ms |
| Gatekeeper check | <1ms | 0.5ms | >5ms |
| Content API | <2s | 1.5s | >5s |
| Full pipeline | <5s | 3s | >10s |

---

## Maintenance Schedule

### Daily (Automated)
- Budget tracking
- Token cleanup (expired tokens)
- Cache pruning (old entries)

### Weekly (Manual)
- Review budget burn rate
- Check for any error patterns in logs
- Test GitHub Pages accessibility
- Verify ANN index performance

### Monthly (Manual)
- Full system health check (run all tests)
- Review and optimize gatekeeper shallow check thresholds
- Archive old budget tracking data if desired
- Plan for quota increases if needed

### Quarterly (Manual)
- Index optimization (consider HNSW for scale)
- Silo configuration review
- Embedding model updates
- GitHub Pages redesign/updates

---

## Alerting & Notifications

Set up monitoring for:

```bash
# Budget warning (70%+ used)
cron: 0 9 * * * python check_budget.py

# API health check (every 30min)
cron: */30 * * * * python scripts/check_you_api.py --live

# Test coverage (daily)
cron: 0 2 * * * python -m pytest tests/ -q

# Deploy status check (every 6h)
cron: 0 */6 * * * scripts/finalize_and_snapshot.sh
```

---

## Emergency Procedures

### Budget Exhausted Mid-Month
1. **Immediate**: Disable Content API calls in gatekeeper
2. **Temporary**: Use mock mode: `ContentAPI(mock_mode=True)`
3. **Escalation**: Contact team lead for quota increase
4. **Recovery**: Reset gatekeeper: `gk.reset_budget()`

### ANN Index Corrupted
1. **Immediate**: Clear index file
2. **Rebuild**: Re-add entries from circular buffer
3. **Test**: Run ANN tests to verify

### GitHub Pages Inaccessible
1. **Immediate**: Check network connectivity
2. **Verify**: Check GitHub Actions status
3. **Manual Deploy**: Run `scripts/finalize_and_snapshot.sh`
4. **Fallback**: Host on alternative CDN if GitHub down

### Agent Loop Detection (Stuckness)
1. **Automatic**: Gatekeeper blocks API calls
2. **Manual**: Force agent reset
3. **Analyze**: Review trajectory logs for pattern
4. **Recovery**: Train new skill or adjust routing

---

## Useful Commands

```bash
# Full system test
pytest tests/test_{on_device_buffer,circular_buffer,local_ann_index,content_api,embeddings,auto_retrieve}.py -v

# Quick smoke test
pytest tests/ -k "test_store" -q

# Budget check
python -c "from src.dashboard.content_api import BudgetTracker; print(f'{BudgetTracker().remaining()} calls remaining')"

# Disk usage
du -sh ~/.cache/pmd-red/

# Reset budget (TESTING ONLY!)
rm ~/.cache/pmd-red/youcom_budget.json

# Check API key
echo $YOU_API_KEY | cut -c1-20

# Deploy to GitHub Pages
cd docs && git add -A && git commit -m "Deploy" && git push
```

---

## Support

For issues not covered here:
1. Check `docs/DASHBOARD_AND_MEMORY_INTEGRATION_VERIFIED.md` for known issues
2. Review test failures: `pytest tests/ -v --tb=short`
3. Check component logs: Configured with logging module
4. Contact: See project README for team contacts

---

*Last updated: October 31, 2025*
*System version: 1.0 (Production Ready)*
