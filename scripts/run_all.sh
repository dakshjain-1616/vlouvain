#!/bin/bash
WORKSPACE="/root/AutoProjects/latest-auto-ai-research/data/builds/a6d8bcd1-3207-44f7-8034-5a9f28a439ad/workspace"

echo "=== STEP 1: Installing dependencies ==="
pip3 install numpy faiss-cpu scipy scikit-learn click tqdm pytest pytest-timeout
echo "INSTALL_EXIT_CODE: $?"

echo ""
echo "=== STEP 2: Installing package in editable mode ==="
cd "$WORKSPACE"
pip3 install -e .
echo "EDITABLE_INSTALL_EXIT_CODE: $?"

echo ""
echo "=== STEP 3: Running tests ==="
cd "$WORKSPACE"
python3 -m pytest tests/ -v --timeout=120 2>&1 | head -100

echo ""
echo "=== STEP 4: Running demo ==="
cd "$WORKSPACE"
python3 demo.py 2>&1 | head -80
