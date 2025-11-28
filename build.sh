#!/bin/bash
# Vercel build script - keeps output small by not including unnecessary files

# Install dependencies
pip install -q -r requirements.txt

# Clean up unnecessary files to reduce deployment size
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

echo "Build complete - dependencies installed"
