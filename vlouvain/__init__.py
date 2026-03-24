"""
VLouvain — cluster high-dimensional vectors without building a full graph.

Quick start::

    from vlouvain import VLouvain
    import numpy as np

    X = np.random.randn(10_000, 128).astype("float32")
    labels = VLouvain(k=15, verbose=True).fit_predict(X)
"""

from vlouvain.algorithm import VLouvain

__version__ = "0.1.0"
__all__ = ["VLouvain"]
