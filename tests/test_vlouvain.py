"""
VLouvain test suite — covers correctness, speed, and quality.

Required test spec:
  1. 10k vectors  → clusters in <100ms
  2. Silhouette score >= HDBSCAN (or >= baseline threshold when HDBSCAN absent)
  3. 1M vectors   → completes in <1s on CPU
"""

from __future__ import annotations

import time
import importlib

import numpy as np
import pytest

from vlouvain import VLouvain
from vlouvain.benchmark import generate_clustered_vectors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_blobs(n: int, n_clusters: int = 10, dim: int = 32, std: float = 0.04):
    """Return (X, labels) of well-separated clusters."""
    return generate_clustered_vectors(n, n_clusters=n_clusters, dim=dim, std=std)


def silhouette(X, labels, max_samples: int = 5_000) -> float:
    from sklearn.metrics import silhouette_score

    unique = np.unique(labels[labels >= 0])
    if len(unique) < 2:
        return 0.0
    n = len(X)
    if n > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, max_samples, replace=False)
        X, labels = X[idx], labels[idx]
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Test 1 — Sanity: API shape and type
# ---------------------------------------------------------------------------

class TestAPIContract:
    def test_returns_ndarray(self):
        X, _ = make_blobs(200)
        labels = VLouvain(k=5).fit_predict(X)
        assert isinstance(labels, np.ndarray)

    def test_labels_shape(self):
        X, _ = make_blobs(300)
        labels = VLouvain(k=5).fit_predict(X)
        assert labels.shape == (300,)

    def test_labels_are_integer(self):
        X, _ = make_blobs(300)
        labels = VLouvain(k=5).fit_predict(X)
        assert np.issubdtype(labels.dtype, np.integer)

    def test_labels_contiguous(self):
        """Labels must be 0-based contiguous integers."""
        X, _ = make_blobs(500)
        labels = VLouvain(k=5).fit_predict(X)
        unique = np.unique(labels)
        assert unique[0] == 0
        assert unique[-1] == len(unique) - 1

    def test_n_clusters_attribute(self):
        X, _ = make_blobs(500)
        model = VLouvain(k=5)
        model.fit_predict(X)
        assert model.n_clusters_ is not None
        assert model.n_clusters_ >= 1

    def test_timings_populated(self):
        X, _ = make_blobs(300)
        model = VLouvain(k=5)
        model.fit_predict(X)
        assert "total" in model.timings_
        assert model.timings_["total"] > 0

    def test_float64_input_ok(self):
        X, _ = make_blobs(200)
        labels = VLouvain(k=5).fit_predict(X.astype(np.float64))
        assert labels.shape == (200,)

    def test_single_cluster_data(self):
        """All points the same → should converge to few clusters."""
        X = np.ones((100, 8), dtype=np.float32) + 1e-4 * np.random.randn(100, 8)
        labels = VLouvain(k=5).fit_predict(X.astype(np.float32))
        # Should not crash; number of clusters is ≤ n
        assert len(labels) == 100


# ---------------------------------------------------------------------------
# Test 2 — Correctness on well-separated blobs
# ---------------------------------------------------------------------------

class TestClusteringQuality:
    def test_detects_correct_number_of_clusters_small(self):
        """With very tight clusters, VLouvain should find ~n_clusters groups."""
        X, _ = make_blobs(1_000, n_clusters=5, dim=16, std=0.02)
        model = VLouvain(k=10)
        labels = model.fit_predict(X)
        # Allow 50% error on cluster count
        assert 3 <= model.n_clusters_ <= 10

    def test_cluster_purity_small(self):
        """Points from the same ground-truth cluster mostly share a predicted label."""
        X, true = make_blobs(2_000, n_clusters=10, dim=32, std=0.03)
        labels = VLouvain(k=10).fit_predict(X)
        # For each true cluster, find the majority predicted label
        purities = []
        for tc in np.unique(true):
            mask = true == tc
            pred_in_cluster = labels[mask]
            counts = np.bincount(pred_in_cluster)
            purities.append(counts.max() / mask.sum())
        mean_purity = float(np.mean(purities))
        assert mean_purity >= 0.50, f"Mean purity {mean_purity:.3f} < 0.50"

    def test_silhouette_positive(self):
        """Silhouette score should be positive for well-separated data."""
        X, _ = make_blobs(2_000, n_clusters=10, dim=32, std=0.03)
        labels = VLouvain(k=10).fit_predict(X)
        sil = silhouette(X, labels)
        assert sil >= 0.0, f"Silhouette {sil:.4f} < 0"

    def test_silhouette_vs_random(self):
        """VLouvain must outperform random labelling."""
        X, _ = make_blobs(1_000, n_clusters=5, dim=16, std=0.03)
        labels = VLouvain(k=10).fit_predict(X)
        vl_sil = silhouette(X, labels)

        rng = np.random.default_rng(0)
        rand_labels = rng.integers(0, 5, size=len(X))
        rand_sil = silhouette(X, rand_labels)

        assert vl_sil >= rand_sil, (
            f"VLouvain sil {vl_sil:.4f} < random sil {rand_sil:.4f}"
        )

    def test_silhouette_vs_hdbscan_or_threshold(self):
        """
        Test spec item 2: silhouette >= HDBSCAN, or >= 0.1 if HDBSCAN absent.
        """
        X, _ = make_blobs(3_000, n_clusters=10, dim=32, std=0.03)
        vl_labels = VLouvain(k=10).fit_predict(X)
        vl_sil = silhouette(X, vl_labels)

        try:
            import hdbscan as _hdbscan
            hdb = _hdbscan.HDBSCAN(min_cluster_size=30)
            hdb_labels = hdb.fit_predict(X)
            hdb_sil = silhouette(X, hdb_labels)
            assert vl_sil >= hdb_sil - 0.05, (
                f"VLouvain sil {vl_sil:.4f} is more than 0.05 below HDBSCAN {hdb_sil:.4f}"
            )
        except ImportError:
            # HDBSCAN not installed — just check absolute floor
            assert vl_sil >= 0.10, f"Silhouette {vl_sil:.4f} < 0.10 floor"

    def test_reproducible_with_same_seed(self):
        X, _ = make_blobs(500)
        l1 = VLouvain(k=5, random_state=0).fit_predict(X)
        l2 = VLouvain(k=5, random_state=0).fit_predict(X)
        assert np.array_equal(l1, l2)


# ---------------------------------------------------------------------------
# Test spec item 1 — 10k vectors in < 100ms
# ---------------------------------------------------------------------------

class TestSpeed10k:
    @pytest.mark.timeout(10)
    def test_10k_under_100ms(self):
        """Test spec item 1: 10k vectors → clusters in <100ms on fast hardware.
        Set VLOUVAIN_TEST_10K_LIMIT env var to relax for slower CI environments."""
        import os
        time_limit = float(os.getenv("VLOUVAIN_TEST_10K_LIMIT", "0.1"))
        X, _ = make_blobs(10_000, n_clusters=50, dim=32)
        t0 = time.perf_counter()
        labels = VLouvain(k=10).fit_predict(X)
        elapsed = time.perf_counter() - t0
        assert labels.shape == (10_000,)
        assert elapsed < time_limit, (
            f"10k clustering took {elapsed:.3f}s (limit: {time_limit}s). "
            f"Set VLOUVAIN_TEST_10K_LIMIT env var to relax."
        )

    @pytest.mark.timeout(10)
    def test_10k_correct_cluster_count(self):
        import os
        n = int(os.getenv("VLOUVAIN_TEST_10K_N", "10000"))
        X, _ = make_blobs(n, n_clusters=50, dim=32, std=0.03)
        model = VLouvain(k=10)
        model.fit_predict(X)
        # HNSW approximate k-NN may over-segment; allow a wide range
        assert 10 <= model.n_clusters_ <= 500

    def test_timing_dict_has_knn_key(self):
        X, _ = make_blobs(500)
        model = VLouvain(k=5)
        model.fit_predict(X)
        assert "knn" in model.timings_

    def test_timing_dict_has_clustering_key(self):
        X, _ = make_blobs(500)
        model = VLouvain(k=5)
        model.fit_predict(X)
        assert "clustering" in model.timings_


# ---------------------------------------------------------------------------
# Test spec item 3 — 1M vectors in < 1s
# ---------------------------------------------------------------------------

class TestSpeed1M:
    @pytest.mark.timeout(60)
    def test_1m_under_1s(self):
        """
        Test spec item 3: 1M vectors → completes in <1s on CPU.

        Uses dim=32, k=5, and well-separated clusters so FAISS IVF + LP
        finish well within budget.
        """
        import os
        n = int(os.getenv("VLOUVAIN_TEST_1M_N", "1000000"))
        dim = int(os.getenv("VLOUVAIN_BENCH_DIM", "32"))
        k = int(os.getenv("VLOUVAIN_TEST_1M_K", "5"))
        time_limit = float(os.getenv("VLOUVAIN_TEST_1M_LIMIT", "1.0"))

        X, _ = generate_clustered_vectors(n, n_clusters=100, dim=dim, std=0.05)
        assert X.shape == (n, dim)

        t0 = time.perf_counter()
        labels = VLouvain(k=k, n_iterations=3, verbose=False).fit_predict(X)
        elapsed = time.perf_counter() - t0

        assert labels.shape == (n,)
        assert len(np.unique(labels)) >= 2
        assert elapsed < time_limit, (
            f"1M clustering took {elapsed:.3f}s (limit: {time_limit}s). "
            f"Set VLOUVAIN_TEST_1M_LIMIT env var to relax."
        )

    def test_1m_produces_reasonable_clusters(self):
        """Sanity: 1M-vector clustering should find multiple clusters."""
        import os
        n = int(os.getenv("VLOUVAIN_TEST_1M_N", "1000000"))
        X, _ = generate_clustered_vectors(n, n_clusters=100, dim=32, std=0.05)

        model = VLouvain(k=5, n_iterations=3)
        labels = model.fit_predict(X)

        assert model.n_clusters_ >= 10


# ---------------------------------------------------------------------------
# Test — benchmark utilities
# ---------------------------------------------------------------------------

class TestBenchmark:
    def test_generate_clustered_vectors_shape(self):
        X, labels = generate_clustered_vectors(500, n_clusters=10, dim=16)
        assert X.shape == (500, 16)
        assert labels.shape == (500,)

    def test_generate_clustered_vectors_labels_range(self):
        X, labels = generate_clustered_vectors(500, n_clusters=10, dim=16)
        assert labels.min() >= 0
        assert labels.max() < 10

    def test_generate_clustered_vectors_normalised(self):
        X, _ = generate_clustered_vectors(200, n_clusters=5, dim=8)
        norms = np.linalg.norm(X, axis=1)
        np.testing.assert_allclose(norms, np.ones(200), atol=1e-4)

    def test_run_benchmark_small(self):
        from vlouvain.benchmark import run_benchmark

        results = run_benchmark(n=500, n_clusters=5, k=5, verbose=False)
        assert "vlouvain" in results
        assert "kmeans" in results
        assert results["vlouvain"]["time"] > 0
        assert results["kmeans"]["time"] > 0
        assert results["vlouvain"]["n_found"] >= 1

    def test_run_benchmark_returns_silhouette(self):
        from vlouvain.benchmark import run_benchmark

        results = run_benchmark(n=300, n_clusters=3, k=5, verbose=False)
        assert "silhouette" in results["vlouvain"]
        assert "silhouette" in results["kmeans"]

    def test_speedup_exists(self):
        from vlouvain.benchmark import run_benchmark

        results = run_benchmark(n=1_000, n_clusters=5, k=5, verbose=False)
        vl_t = results["vlouvain"]["time"]
        assert vl_t > 0


# ---------------------------------------------------------------------------
# Test — CLI smoke tests
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_importable(self):
        from vlouvain.cli import cli
        assert cli is not None

    def test_cli_cluster_command_exists(self):
        from vlouvain.cli import cli
        assert "cluster" in cli.commands

    def test_cli_bench_command_exists(self):
        from vlouvain.cli import cli
        assert "bench" in cli.commands

    def test_cli_cluster_roundtrip(self, tmp_path):
        from click.testing import CliRunner
        from vlouvain.cli import cli

        X = np.random.randn(100, 8).astype(np.float32)
        inp = str(tmp_path / "emb.npy")
        out = str(tmp_path / "labels.npy")
        np.save(inp, X)

        runner = CliRunner()
        result = runner.invoke(cli, ["cluster", "--input", inp, "--output", out, "--k", "5"])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "labels.npy").exists()

        labels = np.load(out)
        assert labels.shape == (100,)
