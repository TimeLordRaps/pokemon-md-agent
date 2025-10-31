# GitHub Upload Checklist

**Goal**: Prepare Pokemon MD Agent repository for public GitHub upload

**Status**: ✅ Ready for initial push

---

## ✅ Pre-Upload Verification

### Repository Health
- [x] `.gitignore` updated (comprehensive exclusions)
- [x] No ROMs, SAVs, or large video files staged
- [x] No credentials or secrets in tracked files
- [x] Clean commit history (see: `git log --oneline`)
- [x] `README.md` has Quick Start section
- [x] Production documentation complete

### Commit Quality
```bash
git log --oneline | head -5
# Output:
# 4dfabbc feat(demo): add production-ready 3-minute video montage pipeline
# 84d2ce1 vision: add dataset dumper tools (sprites/quad capture) + tests
# e3650d9 netio: add adaptive socket + guarded screenshot path (opt-in, no controller changes)
# 50196ed fix(asyncio): remove nested asyncio.run() calls in Agent Core
# 5a71773 feat(orchestrator): add test orchestration scripts and update documentation
```

### File Structure
```bash
pokemon-md-agent/
├── src/                          # Source code (tracked)
│   ├── agent/                   # Agent core + routing
│   ├── environment/             # mGBA controller + RAM decoders
│   ├── vision/                  # Grid parsing + ASCII rendering
│   ├── skills/                  # Skill DSL + runtime
│   ├── retrieval/               # RAG + keyframe system
│   └── mgba-harness/            # CLI tools
│
├── scripts/                     # Utilities (tracked)
│   ├── final_demo_runner.py    # Main demo orchestrator
│   ├── generate_montage_video.py # Video generation
│   └── *.sh / *.ps1            # Test scripts
│
├── tests/                       # Unit tests (tracked)
│
├── config/                      # Address configs (tracked)
│
├── docs/                        # Architecture docs (tracked)
│
├── README.md                    # Quick start + overview
├── PRODUCTION_RUNBOOK.md        # Operator guide
├── DEMO_EXECUTION_SUMMARY.md    # Demo context
├── GITHUB_UPLOAD_CHECKLIST.md   # This file
│
├── .gitignore                   # Excludes: *.gba, *.sav, *.mp4, logs/, runs/, etc.
├── .gitattributes               # (optional) LFS config
├── pyproject.toml               # Package config
├── requirements.txt             # Dependencies
└── LICENSE                      # MIT or similar
```

### Excluded Properly
```bash
# Should NOT be tracked:
rom/                            # Game ROMs (too large, proprietary)
*.gba, *.sav                   # Game assets
*.mp4, *.avi, *.webm          # Video outputs
runs/, snapshots/, logs/       # Runtime outputs
unsloth_compiled_cache/        # Large compiled models
.env, secrets.json             # Credentials
__pycache__, *.pyc             # Python cache
.vscode/, .idea/               # IDE config
```

---

## 📋 GitHub Repository Setup

### 1. Create Repository

**GitHub UI:**
1. Go to https://github.com/new
2. **Repository name**: `pokemon-md-agent`
3. **Description**: "Multi-model Qwen3-VL agent with hierarchical RAG for autonomous Pokemon Mystery Dungeon Red gameplay"
4. **Visibility**: Public
5. **License**: MIT (or chosen license)
6. **Initialize**: No (we already have commits)

### 2. Add Remote & Push

```bash
# From pokemon-md-agent directory
git remote add origin https://github.com/YOUR_USERNAME/pokemon-md-agent.git
git branch -M main
git push -u origin main
```

**Verify:**
```bash
git remote -v
# Should show:
# origin  https://github.com/YOUR_USERNAME/pokemon-md-agent.git (fetch)
# origin  https://github.com/YOUR_USERNAME/pokemon-md-agent.git (push)
```

### 3. Create GitHub Pages (Optional but Recommended)

**If you want live dashboard:**

1. **Enable Pages** in repo settings:
   - Settings → Pages
   - Source: Deploy from a branch
   - Branch: `gh-pages`
   - Folder: `/docs` or `/root`

2. **Create initial dashboard**:
   ```bash
   mkdir -p docs
   cat > docs/index.html <<'HTML'
   <!DOCTYPE html>
   <html>
   <head><title>Pokemon MD Agent Dashboard</title></head>
   <body><h1>Pokemon MD Agent Live Dashboard</h1><p>Coming soon...</p></body>
   </html>
   HTML
   git add docs/index.html
   git commit -m "docs: add GitHub Pages placeholder"
   git push
   ```

3. **Access**: https://YOUR_USERNAME.github.io/pokemon-md-agent/

---

## 🔐 Security Pre-Flight

### Check for Secrets
```bash
# Search for patterns (should return nothing)
git log -p -S 'api_key' | head -5       # API keys
git log -p -S 'password' | head -5      # Passwords
git log -p -S 'token' | head -5         # Tokens
grep -r 'sk_' . --include="*.py"        # OpenAI keys
grep -r 'sk-' . --include="*.py"        # Alternative keys
```

### Review Tracked Config Files
- [ ] `config/addresses/pmd_red_us_v1.json` - ROM addresses (OK to track)
- [ ] `.env.example` - Template with no real values (OK)
- [ ] No actual `.env` file (should be in .gitignore)

### Dangerous Patterns (verify absent)
- [ ] No AWS keys, GCP keys, Azure keys
- [ ] No OpenAI API keys
- [ ] No You.com API keys (if used)
- [ ] No GitHub tokens
- [ ] No database passwords

**Verify clean:**
```bash
git log -p | grep -i 'password\|api_key\|token\|secret' || echo "✓ No secrets found"
```

---

## 📊 Repository Quality Checks

### Code Quality
```bash
# Check Python files
python -m py_compile src/**/*.py                  # Should succeed
python -m pytest tests/ --collect-only            # List test discovery
```

### Documentation
- [x] README.md - Quick start + overview
- [x] PRODUCTION_RUNBOOK.md - Setup & troubleshooting
- [x] DEMO_EXECUTION_SUMMARY.md - Demo context
- [x] AGENTS.md - Agent instructions (from parent project)
- [x] docs/*.md - Architecture guides
- [ ] (Optional) CONTRIBUTING.md - Contribution guidelines
- [ ] (Optional) CODE_OF_CONDUCT.md - Community standards

### License
- [x] LICENSE file present (MIT)
- [x] All source files have license headers (or single LICENSE file is sufficient)

### CI/CD (Optional)
- [ ] `.github/workflows/test.yml` - Run tests on push
- [ ] `.github/workflows/lint.yml` - Code style checks

---

## 📋 Pre-Push Checklist

### Final Verification
```bash
# 1. Check git status (should be clean)
git status

# 2. Verify remotes configured
git remote -v

# 3. Check no large files
find . -type f -size +50M ! -path './.git/*' | head

# 4. Verify .gitignore effective
ls -la | grep -E '\.env|\.gba|\.sav'  # Should be none

# 5. Check latest commits
git log --oneline -3
```

### Execute Upload
```bash
# 1. Set remote
git remote add origin https://github.com/YOUR_USERNAME/pokemon-md-agent.git

# 2. Push main branch
git push -u origin main

# 3. Verify on GitHub
# Visit: https://github.com/YOUR_USERNAME/pokemon-md-agent
```

---

## 🎯 Post-Upload Actions

### 1. Verify Repository
- [ ] Navigate to GitHub repo URL
- [ ] Check README displays correctly
- [ ] Verify all files present
- [ ] Check no secrets exposed

### 2. Setup Issues & Discussions
- [ ] Enable "Issues" (Settings → Features)
- [ ] Enable "Discussions" (Settings → Features)
- [ ] Pin PRODUCTION_RUNBOOK.md issue or discussion

### 3. Add Badges (Optional)
```markdown
# Pokemon Mystery Dungeon Red - Autonomous Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)](#)

... rest of README
```

### 4. Enable Branch Protection (Recommended)
- [ ] Settings → Branches → Add rule for `main`
- [ ] Require pull requests before merging
- [ ] Require status checks to pass (if CI/CD configured)

---

## 🚀 Next Steps After Upload

### Immediate (Day 1)
- [ ] Push initial repository
- [ ] Verify public access
- [ ] Test README instructions (from fresh clone)
- [ ] Record demo video

### Short-term (Week 1)
- [ ] Add contributing guidelines
- [ ] Set up issue templates
- [ ] Enable discussions for feature requests
- [ ] Add GitHub Actions CI (optional)

### Medium-term (Week 2+)
- [ ] Build GitHub Pages dashboard
- [ ] Document architecture decisions
- [ ] Add example trajectories/videos
- [ ] Create quick video walkthrough

---

## 🔗 Useful Commands

### Git Flow
```bash
# Clone for first-time setup
git clone https://github.com/YOUR_USERNAME/pokemon-md-agent.git
cd pokemon-md-agent

# Setup environment
mamba create -n agent-hackathon python=3.11 -y
mamba activate agent-hackathon
pip install -r requirements.txt

# Run demo
python scripts/final_demo_runner.py
```

### Size Analysis
```bash
# Find large files in history
git rev-list --all --objects | sort -k2 | tail -5

# Check .gitignore effectiveness
git check-ignore -v src/environment/*.pyc

# Estimate final push size
du -sh .git/
```

---

## 📞 Troubleshooting

### Already Have Remote
```bash
git remote remove origin
git remote add origin https://github.com/USERNAME/pokemon-md-agent.git
```

### Large Files in History
```bash
# Before pushing, if you accidentally committed large files:
git reset HEAD~1
# Or use BFG repo-cleaner: https://rtyley.github.io/bfg-repo-cleaner/
```

### Push Rejected
```bash
# If remote has commits you don't have:
git pull origin main --allow-unrelated-histories
git push -u origin main
```

---

## ✅ Final Checklist

- [ ] `.gitignore` configured (no ROMs, videos, creds)
- [ ] All commits meaningful (good messages)
- [ ] No secrets in git history
- [ ] README has Quick Start
- [ ] Production docs complete
- [ ] Remote configured
- [ ] Main branch pushed
- [ ] GitHub Pages enabled (optional)
- [ ] Issues enabled
- [ ] Repository is public & discoverable

---

**Status**: Ready for GitHub upload ✅

**Last Updated**: 2025-10-30
