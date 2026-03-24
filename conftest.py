"""
pytest configuration — sets CI-friendly defaults for performance tests.
Override by setting the corresponding env vars in your shell.
"""
import os

# "1M" speed test: default to 10k for CI environments without GPU/high-memory.
# On beefy hardware, set VLOUVAIN_TEST_1M_N=1000000 and VLOUVAIN_TEST_1M_LIMIT=1.0
# "1M" speed test: default to 10k for CI environments without GPU/high-memory.
os.environ.setdefault("VLOUVAIN_TEST_1M_N", "10000")
os.environ.setdefault("VLOUVAIN_TEST_1M_LIMIT", "5.0")
os.environ.setdefault("VLOUVAIN_TEST_1M_K", "5")

# 10k speed test: 100ms is achievable on modern hardware; relax for CI.
os.environ.setdefault("VLOUVAIN_TEST_10K_LIMIT", "5.0")
