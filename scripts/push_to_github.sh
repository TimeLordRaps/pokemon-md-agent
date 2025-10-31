#!/bin/bash
# Quick push to GitHub after demo
# Usage: bash scripts/push_to_github.sh

set -e

echo "========================================"
echo "GitHub Push Automation"
echo "========================================"

# Verify we're in the right directory
if [ ! -f "README.md" ]; then
    echo "ERROR: Not in pokemon-md-agent directory"
    exit 1
fi

# Check git status
echo ""
echo "Current git status:"
git status --short | head -5
echo ""

# Ask for confirmation
read -p "Ready to push to GitHub? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 1
fi

# Configure remote if not already done
echo ""
echo "Configuring remote..."
if ! git remote | grep -q "origin"; then
    echo "Adding remote: https://github.com/TimeLordRaps/pokemon-md-agent.git"
    git remote add origin https://github.com/TimeLordRaps/pokemon-md-agent.git
else
    echo "Remote already configured"
    git remote -v | grep origin
fi

# Ensure we're on main branch
echo ""
echo "Switching to main branch..."
git branch -M main

# Push
echo ""
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "========================================"
echo "SUCCESS! Repository pushed to GitHub"
echo "========================================"
echo ""
echo "View at: https://github.com/TimeLordRaps/pokemon-md-agent"
echo ""

# Show what was pushed
echo "Latest commits:"
git log --oneline -3

echo ""
echo "Done!"
