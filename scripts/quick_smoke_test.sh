#!/bin/bash
set -e

echo "=== Quick Smoke Test (T-minus 3hrs) ==="

# Test 1: Screenshot capture (30s timeout)
echo "[1/3] Testing screenshot capture..."
python -c "
import tempfile
from pathlib import Path
from src.environment.mgba_controller import MGBAController
c = MGBAController()
c.connect()
with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / 'test.png'
    img = c.capture_screenshot(str(path))
    assert img is not None
    assert img.shape == (160, 240, 3)
c.disconnect()
print('✓ Screenshot capture works')
"

# Test 2: WRAM read (10s timeout)
echo "[2/3] Testing WRAM read..."
python -c "
from src.environment.mgba_controller import MGBAController
c = MGBAController()
c.connect()
data = c.memory_domain_read_range('wram', 0x0000, 256)
assert len(data) == 256
c.disconnect()
print('✓ WRAM read works')
"

# Test 3: Reconnect stress (5 cycles)
echo "[3/3] Testing reconnect stability..."
for i in {1..5}; do
  python -c "
from src.environment.mgba_controller import MGBAController
c = MGBAController()
c.connect()
c.disconnect()
  "
  echo "  Cycle $i/5 OK"
done
echo "✓ Reconnect stable"

echo ""
echo "=== All smoke tests passed ==="