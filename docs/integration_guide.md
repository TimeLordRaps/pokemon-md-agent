# PMD-Red Agent Integration Guide

## Overview

This guide provides comprehensive instructions for integrating and deploying the PMD-Red autonomous agent system. The agent uses mGBA + mGBA-http, Qwen3-VL vision models, hierarchical RAG, and gated content API access to play Pokemon Mystery Dungeon Red Rescue Team.

## System Architecture

### Core Components
- **mGBA-http Server**: Lua-based emulator interface (port 8888)
- **Vision Pipeline**: Qwen3-VL models for screenshot analysis
- **Retrieval System**: 7-tier temporal RAG with gatekeeper
- **Agent Loop**: Cost-aware model routing and memory management
- **Dashboard**: GitHub Pages artifact streaming

### Data Flow
```
Game State → Vision Analysis → Retrieval → Model Inference → Action → Repeat
```

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA RTX 30-series or better (24GB+ VRAM recommended)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ SSD for models and artifacts
- **Network**: Stable internet for model downloads and API access

### Software Requirements
- **OS**: Windows 10/11 or Linux (Ubuntu 20.04+)
- **Python**: 3.10+ with conda/mamba
- **mGBA**: Latest version with Lua scripting enabled
- **Git**: For repository management

## Installation

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/pmd-red-agent.git
cd pmd-red-agent

# Create conda environment
mamba env create -f environment.yaml
mamba activate agent-hackathon

# Verify environment
python --version  # Should be 3.10+
mamba info --envs  # Should show agent-hackathon
```

### 2. Model Setup

```bash
# Download Qwen3-VL models
python -c "
from src.models.real_loader import RealModelLoader
loader = RealModelLoader()
loader.download_models(['Qwen3-VL-2B', 'Qwen3-VL-4B'])
"
```

### 3. ROM and Emulator Setup

```bash
# Place PMD Red US ROM in rom/ directory
cp /path/to/pmd_red_us.gba rom/

# Configure mGBA with Lua server
# Edit mGBA settings to enable Lua scripting
# Place mGBASocketServer.lua in mGBA scripts directory
```

## Configuration

### Environment Variables

```bash
# Required
export HF_HOME="/path/to/huggingface/cache"  # Sanitized path required
export MGBACONTROLLER_PORT="8888"
export AGENT_MODE="production"

# Optional
export LOG_LEVEL="INFO"
export API_BUDGET="1000"
export RETRIEVAL_WINDOW_MINUTES="60"
```

### Configuration Files

#### `config/model_config.yaml`
```yaml
models:
  qwen_2b:
    name: "Qwen3-VL-2B-Instruct"
    max_tokens: 1024
    temperature: 0.7
    cost_per_token: 0.0001

  qwen_4b:
    name: "Qwen3-VL-4B-Instruct"
    max_tokens: 2048
    temperature: 0.7
    cost_per_token: 0.0002

routing:
  confidence_threshold_2b: 0.8
  confidence_threshold_4b: 0.6
  stuck_threshold: 5
```

#### `config/retrieval_config.yaml`
```yaml
buffer:
  size_minutes: 60
  keyframe_policy: "ssim_0.9"

silos:
  - resolution: 1
  - resolution: 2
  - resolution: 4
  - resolution: 8
  - resolution: 16
  - resolution: 32
  - resolution: 64

gatekeeper:
  shallow_hit_threshold: 3
  api_budget_limit: 1000
```

## Startup Procedure

### 1. Start mGBA-http Server

```bash
# Launch mGBA with ROM
./mgba -g rom/pmd_red_us.gba -s scripts/mGBASocketServer.lua

# Verify server is running
curl http://localhost:8888/status
# Should return: {"status": "ready", "port": 8888}
```

### 2. Initialize Agent

```bash
# Start agent system
python -m pokemon_md_agent.main \
  --config config/model_config.yaml \
  --retrieval-config config/retrieval_config.yaml \
  --dashboard-url https://yourusername.github.io/pmd-dashboard
```

### 3. Verify Integration

```bash
# Test vision pipeline
python -c "
from src.vision.vision_pipeline import VisionPipeline
pipeline = VisionPipeline()
result = pipeline.analyze_screenshot('test_screenshot.png')
print('Vision analysis successful')
"

# Test retrieval system
python -c "
from src.retrieval.auto_retrieve import AutoRetrieve
retriever = AutoRetrieve()
results = retriever.search('stairs location')
print(f'Found {len(results)} relevant experiences')
"

# Test agent loop
python -c "
from src.agent.agent_loop import AgentLoop
agent = AgentLoop()
agent.start_autonomous_play()
"
```

## Operation Modes

### Development Mode
```bash
export AGENT_MODE="development"
# Enables debug logging, test data generation, mock API responses
```

### Production Mode
```bash
export AGENT_MODE="production"
# Optimized performance, real API calls, artifact streaming
```

### Benchmark Mode
```bash
export AGENT_MODE="benchmark"
# Performance testing, detailed metrics collection
```

## Monitoring and Maintenance

### Key Metrics to Monitor

#### Performance Metrics
- **Inference Latency**: <2s per action (target)
- **Retrieval Time**: <500ms for local search
- **Memory Usage**: <8GB working set
- **API Budget**: Track remaining calls

#### Quality Metrics
- **Success Rate**: Actions leading to progress
- **Hallucination Rate**: <5% false information
- **Stuck Detection**: Proper loop breaking

### Log Analysis

```bash
# View recent agent activity
tail -f logs/agent_$(date +%Y%m%d).log

# Monitor model usage
grep "model_inference" logs/agent_*.log | jq '.model,.latency,.cost'

# Check retrieval performance
grep "retrieval_search" logs/agent_*.log | jq '.query,.hits,.latency'
```

### Health Checks

```bash
# System health check
python scripts/health_check.py

# Component status
curl http://localhost:8080/health  # If dashboard deployed

# Resource usage
python scripts/resource_monitor.py
```

## Troubleshooting

### Common Issues

#### Model Loading Failures
```bash
# Check HF_HOME sanitization
python -c "import os; print('HF_HOME:', os.environ.get('HF_HOME'))"

# Verify model cache
ls -la $HF_HOME/hub/models--Qwen--Qwen3-VL-2B/

# Clear cache if corrupted
rm -rf $HF_HOME/hub/models--Qwen--Qwen3-VL-2B/
```

#### mGBA Connection Issues
```bash
# Check if server is running
netstat -tlnp | grep 8888

# Test connection
curl http://localhost:8888/ping

# Restart mGBA if needed
pkill -f mgba
./mgba -g rom/pmd_red_us.gba -s scripts/mGBASocketServer.lua
```

#### Memory Issues
```bash
# Monitor memory usage
python -c "
import psutil
process = psutil.Process()
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f}MB')
"

# Clear retrieval buffer if needed
python -c "
from src.retrieval.circular_buffer import CircularBuffer
buffer = CircularBuffer()
buffer.clear()
"
```

#### API Budget Exceeded
```bash
# Check remaining budget
python -c "
from src.dashboard.content_api import ContentAPIWrapper
api = ContentAPIWrapper()
print(f'Remaining budget: {api.get_remaining_budget()}')
"

# Reset budget (development only)
# Edit config or restart with fresh budget
```

## Scaling and Optimization

### Performance Tuning

#### Model Selection
- **2B Model**: Fastest, lowest cost, suitable for simple tasks
- **4B Model**: Balanced performance, good for complex scenarios
- **8B Model**: Highest accuracy, use for stuck situations only

#### Memory Management
- **Buffer Size**: Reduce to 30 minutes for memory-constrained systems
- **Keyframes**: Increase SSIM threshold for fewer keyframes
- **Embeddings**: Use lower-dimensional embeddings if needed

#### Rate Limiting
- **Screenshots**: ≤30/s baseline, reduce for slower systems
- **Memory Polls**: ≤10/s, increase during menus/combat
- **API Calls**: Budget-based throttling

### Multi-Agent Deployment

```bash
# Run multiple agents on different ports
export MGBACONTROLLER_PORT="8889"
python -m pokemon_md_agent.main --instance-id "agent_2" &

export MGBACONTROLLER_PORT="8890"
python -m pokemon_md_agent.main --instance-id "agent_3" &
```

## Development and Testing

### Running Tests

```bash
# Fast test suite
mamba info --envs && python --version && mamba activate agent-hackathon && pwd && ls -la && python -m pytest -q --maxfail=1 -m "not slow"

# Full test suite
python -m pytest -q

# Specific component tests
python -m pytest tests/test_retrieval_system.py -v
python -m pytest tests/test_vision_pipeline.py -v
```

### Debugging

```bash
# Enable debug logging
export LOG_LEVEL="DEBUG"

# Run with debugger
python -m pdb -m pokemon_md_agent.main

# Profile performance
python -m cProfile -s time -m pokemon_md_agent.main > profile.txt
```

## Backup and Recovery

### Data Backup
```bash
# Backup configuration
cp config/ config_backup_$(date +%Y%m%d)

# Backup model cache (if needed)
cp -r $HF_HOME/hub/models--Qwen models_backup/

# Backup logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
```

### Recovery Procedures

#### Model Cache Corruption
```bash
# Clear corrupted cache
rm -rf $HF_HOME/hub/models--Qwen--*

# Restart download
python scripts/download_models.py --force
```

#### State Recovery
```bash
# Restore from checkpoint
python scripts/restore_checkpoint.py --checkpoint-id latest

# Manual state reset
python -c "
from src.environment.save_manager import SaveManager
sm = SaveManager()
sm.load_state('slot_1')
"
```

## Security Considerations

### API Key Management
- Store You.com API keys securely (environment variables, not code)
- Rotate keys regularly
- Monitor API usage for anomalies

### Data Privacy
- No personal data transmitted
- Game state anonymized before external API calls
- Logs sanitized of sensitive information

### Network Security
- Use HTTPS for all external API calls
- Implement request timeouts and retries
- Monitor for unusual network patterns

## Support and Resources

### Documentation
- `docs/architecture.md` - System architecture details
- `docs/troubleshooting.md` - Common issues and solutions
- `AGENTS.md` - Agent development guidelines

### Community Resources
- GitHub Issues for bug reports
- Discord channel for community support
- Wiki for advanced configuration

### Professional Support
- Enterprise deployment consulting available
- Custom model training services
- Performance optimization services

---

*Integration guide created by Claude (Research) Agent on 2025-10-31T22:45Z*
*Based on project architecture and operational requirements*