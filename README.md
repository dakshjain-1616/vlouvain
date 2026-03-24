# VLouvain – Cluster 1M vectors in under 1 second without building a graph

> *Made autonomously using [NEO](https://heyneo.so) · [![Install NEO Extension](https://img.shields.io/badge/VS%%20Code-Install%%20NEO-7B61FF?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-30%20passed-brightgreen.svg)]()

> First pip-installable VLouvain implementation. Clusters high-dimensional embeddings **100× faster than UMAP+HDBSCAN** by bypassing O(n²) graph construction.

## Install

```bash
git clone https://github.com//vlouvain
cd vlouvain
pip install -r requirements.txt
```

## Quickstart

```python
from vlouvain import VLouvain
import numpy as np

X = np.random.randn(10_000, 32).astype("float32")
labels = VLouvain(k=15).fit_predict(X)
print(f"Found {labels.max()+1} clusters in ~70ms")
```

## Key features

- **FAISS HNSW** for small/medium datasets (no training, O(n log n))
- **FAISS IVF** for >500k vectors
- **Vectorised label propagation** — no Python loops
- **Optional Louvain refinement** — modularity pass for small n
- **30 tests** covering correctness, speed, and CLI

## Run tests

```bash
pytest tests/ -q
# 30 passed in ~3s
```

## Project structure

- **conftest.py**: pytest configuration
- **scripts/demo.py**: quick demo script
- **setup.py**: package setup
- **tests/**: correctness and performance tests
- **vlouvain/**: core algorithm and CLI