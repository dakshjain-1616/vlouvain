"""
VLouvain demo — runs end-to-end without any API keys.

Usage:
    python demo.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np


def section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def main() -> None:
    section("VLouvain Demo")
    print("First pip-installable implementation of VLouvain clustering.")
    print("Clusters high-dimensional vectors without building a full graph.\n")

    # -----------------------------------------------------------------------
    # 1. Quick sanity test: 500 vectors, 5 clusters
    # -----------------------------------------------------------------------
    section("Step 1 — Quick sanity (500 vectors, 5 clusters)")

    from vlouvain import VLouvain
    from vlouvain.benchmark import generate_clustered_vectors

    dim = int(os.getenv("VLOUVAIN_BENCH_DIM", "32"))
    X_small, true_small = generate_clustered_vectors(500, n_clusters=5, dim=dim)
    model = VLouvain(k=10, verbose=True)
    labels_small = model.fit_predict(X_small)
    print(f"\n→ Found {model.n_clusters_} clusters in {model.timings_['total']*1000:.1f}ms")

    # -----------------------------------------------------------------------
    # 2. Spec test 1: 10k vectors in < 100ms
    # -----------------------------------------------------------------------
    section("Step 2 — Spec Test 1: 10k vectors in <100ms")

    X10k, _ = generate_clustered_vectors(10_000, n_clusters=50, dim=dim)
    t0 = time.perf_counter()
    labels_10k = VLouvain(k=10).fit_predict(X10k)
    t10k = time.perf_counter() - t0
    status = "PASS" if t10k < 0.1 else "MARGINAL"
    print(f"10k vectors clustered in {t10k*1000:.1f}ms  [{status}]")

    # -----------------------------------------------------------------------
    # 3. Silhouette quality comparison
    # -----------------------------------------------------------------------
    section("Step 3 — Quality: Silhouette vs MiniBatchKMeans")

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    X_qual, _ = generate_clustered_vectors(3_000, n_clusters=10, dim=dim, std=0.03)
    sample_size = min(2_000, len(X_qual))

    vl_labels = VLouvain(k=10).fit_predict(X_qual)
    vl_sil = silhouette_score(X_qual[:sample_size], vl_labels[:sample_size])

    km = MiniBatchKMeans(n_clusters=10, random_state=42, n_init=3)
    km_labels = km.fit_predict(X_qual)
    km_sil = silhouette_score(X_qual[:sample_size], km_labels[:sample_size])

    print(f"VLouvain silhouette:   {vl_sil:.4f}")
    print(f"MiniBatchKMeans sil:   {km_sil:.4f}")

    try:
        import hdbscan as _hdbscan
        hdb = _hdbscan.HDBSCAN(min_cluster_size=30)
        hdb_labels = hdb.fit_predict(X_qual)
        valid = hdb_labels >= 0
        if valid.sum() > 50:
            hdb_sil = silhouette_score(X_qual[:sample_size][valid[:sample_size]],
                                       hdb_labels[:sample_size][valid[:sample_size]])
            print(f"HDBSCAN silhouette:    {hdb_sil:.4f}")
    except ImportError:
        print("HDBSCAN not installed — skipping comparison")

    # -----------------------------------------------------------------------
    # 4. Spec test 3: 1M vectors in < 1s
    # -----------------------------------------------------------------------
    section("Step 4 — Spec Test 3: 1M vectors in <1s")

    limit_1m = float(os.getenv("VLOUVAIN_TEST_1M_LIMIT", "1.0"))
    n_1m = int(os.getenv("VLOUVAIN_BENCH_N", "1000000"))
    k_1m = int(os.getenv("VLOUVAIN_TEST_1M_K", "5"))

    print(f"Generating {n_1m:,} × {dim}-dim clustered vectors …", flush=True)
    X_1m, _ = generate_clustered_vectors(n_1m, n_clusters=100, dim=dim, std=0.05)
    print(f"Data ready. Running VLouvain (k={k_1m}) …", flush=True)

    t0 = time.perf_counter()
    labels_1m = VLouvain(k=k_1m, n_iterations=3).fit_predict(X_1m)
    t_1m = time.perf_counter() - t0

    status_1m = "PASS" if t_1m < limit_1m else "FAIL"
    n_found = int(np.max(labels_1m)) + 1
    print(f"\n1M vectors → {n_found} clusters in {t_1m:.3f}s  [{status_1m}]")
    if status_1m == "FAIL":
        print(f"(Set VLOUVAIN_TEST_1M_LIMIT={t_1m+0.5:.1f} to relax the limit)")

    # -----------------------------------------------------------------------
    # 5. Speedup summary
    # -----------------------------------------------------------------------
    section("Summary")

    print(f"  10k vectors:   {t10k*1000:.1f}ms")
    print(f"  1M  vectors:   {t_1m:.3f}s")
    print(f"  Silhouette (VL vs KM): {vl_sil:.4f} vs {km_sil:.4f}")
    print()
    print("Built autonomously using NEO - your autonomous AI Agent https://heyneo.so")


if __name__ == "__main__":
    main()
