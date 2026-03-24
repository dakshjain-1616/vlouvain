"""
Benchmark utilities: compare VLouvain against KMeans (and optionally HDBSCAN).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_clustered_vectors(
    n: int,
    n_clusters: int = 100,
    dim: int = 32,
    random_state: int = 42,
    std: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate n vectors drawn from n_clusters isotropic Gaussians on the unit sphere.

    Returns
    -------
    X : (n, dim) float32 array — L2-normalised embeddings
    true_labels : (n,) int32 array — ground-truth cluster ids
    """
    rng = np.random.default_rng(random_state)
    dim = int(os.getenv("VLOUVAIN_BENCH_DIM", str(dim)))
    std = float(os.getenv("VLOUVAIN_BENCH_STD", str(std)))

    # Sample cluster centres on unit sphere
    centres = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    # Assign points to clusters
    sizes = rng.multinomial(n, np.ones(n_clusters) / n_clusters)
    parts: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for cid, size in enumerate(sizes):
        if size == 0:
            continue
        noise = rng.standard_normal((size, dim)).astype(np.float32) * std
        pts = centres[cid] + noise
        pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-10
        parts.append(pts)
        labels_list.append(np.full(size, cid, dtype=np.int32))

    X = np.concatenate(parts, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], labels[perm]


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _run_vlouvain(X: np.ndarray, k: int, **kwargs) -> tuple[np.ndarray, float]:
    from vlouvain.algorithm import VLouvain

    model = VLouvain(k=k, verbose=False, **kwargs)
    t0 = time.perf_counter()
    labels = model.fit_predict(X)
    elapsed = time.perf_counter() - t0
    return labels, elapsed


def _run_kmeans(X: np.ndarray, n_clusters: int) -> tuple[np.ndarray, float]:
    from sklearn.cluster import MiniBatchKMeans

    n_clusters = min(n_clusters, len(X))
    batch = int(os.getenv("VLOUVAIN_KMEANS_BATCH", "10240"))
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch,
        random_state=42,
        n_init=3,
    )
    t0 = time.perf_counter()
    labels = model.fit_predict(X)
    elapsed = time.perf_counter() - t0
    return labels, elapsed


def _run_hdbscan(X: np.ndarray) -> tuple[np.ndarray, float]:
    try:
        import hdbscan

        min_cluster_size = int(os.getenv("VLOUVAIN_HDBSCAN_MIN_CLUSTER", "50"))
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        t0 = time.perf_counter()
        labels = model.fit_predict(X)
        elapsed = time.perf_counter() - t0
        return labels, elapsed
    except ImportError:
        return np.zeros(len(X), dtype=np.int32), -1.0


def _silhouette(X: np.ndarray, labels: np.ndarray, max_samples: int = 5000) -> float:
    from sklearn.metrics import silhouette_score

    unique = np.unique(labels[labels >= 0])
    if len(unique) < 2:
        return 0.0
    n = len(X)
    if n > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, max_samples, replace=False)
        X = X[idx]
        labels = labels[idx]
    try:
        return float(silhouette_score(X, labels, sample_size=None))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main benchmark entry-point
# ---------------------------------------------------------------------------

def run_benchmark(
    n: int = 1_000_000,
    n_clusters: int = 100,
    k: int = 5,
    include_hdbscan: bool = False,
    hdbscan_max_n: int = 50_000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a full benchmark comparing VLouvain, MiniBatchKMeans (and optionally
    HDBSCAN) on synthetic clustered vectors.

    Parameters
    ----------
    n            : number of vectors
    n_clusters   : number of ground-truth clusters
    k            : k for VLouvain
    include_hdbscan : whether to run HDBSCAN (slow for large n)
    hdbscan_max_n : maximum n before HDBSCAN is skipped
    verbose      : print results table

    Returns
    -------
    dict with keys: n, dim, n_clusters, vlouvain, kmeans, hdbscan (optional)
    Each method entry: {time, n_found, silhouette}
    """
    dim = int(os.getenv("VLOUVAIN_BENCH_DIM", "32"))
    n_clusters_env = int(os.getenv("VLOUVAIN_BENCH_CLUSTERS", str(n_clusters)))

    if verbose:
        print(f"\n{'='*60}")
        print(f"  VLouvain Benchmark")
        print(f"  n={n:,}  dim={dim}  true_clusters={n_clusters_env}")
        print(f"{'='*60}")

    # --- Data generation ---
    if verbose:
        print("Generating data …", end=" ", flush=True)
    t0 = time.perf_counter()
    X, true_labels = generate_clustered_vectors(
        n, n_clusters=n_clusters_env, dim=dim
    )
    gen_time = time.perf_counter() - t0
    if verbose:
        print(f"{gen_time:.2f}s")

    results: Dict[str, Any] = {
        "n": n,
        "dim": dim,
        "true_n_clusters": n_clusters_env,
    }

    # --- VLouvain ---
    if verbose:
        print("Running VLouvain …", end=" ", flush=True)
    vl_labels, vl_time = _run_vlouvain(X, k=k)
    vl_sil = _silhouette(X, vl_labels)
    results["vlouvain"] = {
        "time": vl_time,
        "n_found": int(np.max(vl_labels) + 1),
        "silhouette": vl_sil,
    }
    if verbose:
        print(f"{vl_time:.3f}s  |  clusters={results['vlouvain']['n_found']}  |  sil={vl_sil:.4f}")

    # --- MiniBatchKMeans ---
    if verbose:
        print("Running MiniBatchKMeans …", end=" ", flush=True)
    km_labels, km_time = _run_kmeans(X, n_clusters=n_clusters_env)
    km_sil = _silhouette(X, km_labels)
    results["kmeans"] = {
        "time": km_time,
        "n_found": n_clusters_env,
        "silhouette": km_sil,
    }
    if verbose:
        print(f"{km_time:.3f}s  |  clusters={n_clusters_env}  |  sil={km_sil:.4f}")

    # --- HDBSCAN (optional, only for smaller n) ---
    if include_hdbscan and n <= hdbscan_max_n:
        if verbose:
            print("Running HDBSCAN …", end=" ", flush=True)
        hdb_labels, hdb_time = _run_hdbscan(X)
        if hdb_time >= 0:
            hdb_sil = _silhouette(X, hdb_labels)
            results["hdbscan"] = {
                "time": hdb_time,
                "n_found": int(np.max(hdb_labels) + 1),
                "silhouette": hdb_sil,
            }
            if verbose:
                print(f"{hdb_time:.3f}s  |  clusters={results['hdbscan']['n_found']}  |  sil={hdb_sil:.4f}")
        else:
            if verbose:
                print("skipped (hdbscan not installed)")
    elif include_hdbscan:
        if verbose:
            print(f"HDBSCAN skipped (n={n:,} > {hdbscan_max_n:,})")

    if verbose:
        print(f"\nSummary: VLouvain {vl_time:.3f}s vs KMeans {km_time:.3f}s")
        speedup = km_time / max(vl_time, 1e-6)
        print(f"VLouvain is {speedup:.1f}x faster than MiniBatchKMeans")
        print(f"{'='*60}\n")

    return results
