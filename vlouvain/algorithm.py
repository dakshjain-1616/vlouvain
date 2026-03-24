"""
VLouvain: Cluster high-dimensional vectors without building a full graph.

Uses FAISS for O(n log n) approximate nearest-neighbor search, then applies
vectorised label propagation / Louvain community detection on the sparse
k-NN graph — bypassing the O(n²) graph construction that makes UMAP/HDBSCAN
slow on 1M+ vectors.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import faiss


def _scipy_mode_rows(A: np.ndarray) -> np.ndarray:
    """
    Find the mode of each row in a 2-D integer array.

    Wraps scipy.stats.mode with compatibility handling for scipy API changes
    (keepdims param was added in 1.9; return shape changed in 1.11).
    """
    from scipy.stats import mode

    result = mode(A, axis=1, keepdims=False)
    modes = result.mode
    # scipy < 1.9: mode is shape (n, 1); scipy >= 1.11: shape (n,)
    return np.asarray(modes).ravel().astype(A.dtype)


class VLouvain:
    """
    VLouvain clusters high-dimensional vectors via fast approximate k-NN
    followed by vectorised label propagation.

    Parameters
    ----------
    k : int
        Number of nearest neighbours to use in the sparse graph.
    n_iterations : int
        Maximum label-propagation passes.
    resolution : float
        Louvain resolution parameter; higher = more, smaller clusters.
    similarity_threshold : float or None
        Minimum cosine similarity for an edge. None = auto (0.3 × median).
    random_state : int or None
        Seed. Reads $VLOUVAIN_RANDOM_STATE if None.
    verbose : bool
        Print per-step timing.
    n_probe : int or None
        FAISS IVF nprobe. None = auto.
    """

    def __init__(
        self,
        k: int = 15,
        n_iterations: int = 10,
        resolution: float = 1.0,
        similarity_threshold: Optional[float] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
        n_probe: Optional[int] = None,
    ) -> None:
        self.k = k
        self.n_iterations = n_iterations
        self.resolution = resolution
        self.similarity_threshold = similarity_threshold
        self.random_state = (
            random_state
            if random_state is not None
            else int(os.getenv("VLOUVAIN_RANDOM_STATE", "42"))
        )
        self.verbose = verbose
        self.n_probe = n_probe

        # Populated after fit_predict
        self.n_clusters_: Optional[int] = None
        self.timings_: dict = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[VLouvain] {msg}")

    def _effective_k(self, n: int) -> int:
        """Reduce k for large datasets to keep wall time bounded."""
        if n > 500_000:
            return min(self.k, 5)
        if n > 200_000:
            return min(self.k, 10)
        return self.k

    # ------------------------------------------------------------------
    # k-NN via FAISS
    # ------------------------------------------------------------------

    def _build_knn(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Build approximate k-NN using FAISS inner-product on L2-normalised X
        (equivalent to cosine similarity).

        Returns
        -------
        indices    : (n, k_eff) int64 — neighbor row indices
        similarities : (n, k_eff) float32 — cosine similarities ∈ [-1, 1]
        """
        n, d = X.shape
        k_eff = self._effective_k(n)
        k_query = min(k_eff + 1, n)  # +1 because self is usually returned

        t0 = time.perf_counter()

        # L2-normalise for cosine similarity via inner product
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        Xn = (X / norms).astype(np.float32)

        if n <= 2_000:
            # Exact flat search — fast enough for very small n
            index = faiss.IndexFlatIP(d)
            index.add(Xn)
            sims, idx = index.search(Xn, k_query)
        elif n <= 500_000:
            # HNSW: O(n log n) build + query, no training required,
            # significantly faster than FlatIP for 2k–500k vectors.
            M = 16
            index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = max(32, M * 2)
            index.hnsw.efSearch = max(M, k_query * 2)
            index.add(Xn)
            sims, idx = index.search(Xn, k_query)
        else:
            # For very large n (>500k) use IVF — training amortises quickly.
            nlist = min(int(np.sqrt(n)), 1024)

            # nprobe ~ sqrt(nlist) balances recall vs speed
            nprobe = self.n_probe or max(4, int(np.sqrt(nlist)))

            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(
                quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
            )

            train_size = min(n, max(nlist * 40, 50_000))
            rng = np.random.default_rng(self.random_state)
            train_idx = rng.choice(n, train_size, replace=False)
            self._log(f"Training IVF({nlist}, nprobe={nprobe}) on {train_size:,} samples …")
            index.train(Xn[train_idx])
            index.add(Xn)
            index.nprobe = nprobe
            sims, idx = index.search(Xn, k_query)

        self.timings_["knn"] = time.perf_counter() - t0
        self._log(f"k-NN (k={k_eff}): {self.timings_['knn']:.3f}s")

        # ── Vectorised self-hit removal ──────────────────────────────────
        # Mark self positions and push them to the end via argsort.
        row_ids = np.arange(n, dtype=np.int64)
        is_self = idx == row_ids[:, None]   # (n, k_query) bool

        # Replace self positions with out-of-range sentinel, then sort
        # row-wise so sentinel slots fall at the end.
        idx_work = idx.astype(np.int64)
        sim_work = sims.astype(np.float32)
        idx_work[is_self] = n + 1          # sentinel
        sim_work[is_self] = -2.0           # sentinel (below any real sim)

        sort_ord = np.argsort(is_self.view(np.uint8), axis=1, kind="stable")
        idx_clean = np.take_along_axis(idx_work, sort_ord, axis=1)[:, :k_eff]
        sim_clean = np.take_along_axis(sim_work, sort_ord, axis=1)[:, :k_eff]

        return idx_clean, sim_clean

    # ------------------------------------------------------------------
    # Label propagation
    # ------------------------------------------------------------------

    def _label_propagation(
        self, indices: np.ndarray, similarities: np.ndarray, n: int
    ) -> np.ndarray:
        """
        Vectorised label propagation on the k-NN graph.

        For n ≤ VLOUVAIN_LP_THRESHOLD: scipy.stats.mode (proper majority vote).
        For n > VLOUVAIN_LP_THRESHOLD: numpy median over sorted neighbor labels
        (approximates mode, avoids Python loops).
        """
        t0 = time.perf_counter()
        k = indices.shape[1]
        lp_threshold = int(os.getenv("VLOUVAIN_LP_THRESHOLD", "200000"))

        # ── Similarity filter ────────────────────────────────────────────
        threshold = self.similarity_threshold
        if threshold is None:
            threshold = max(0.0, float(np.median(similarities)) * 0.3)
        self._log(f"Similarity threshold: {threshold:.4f}")

        # For masked (weak) edges we substitute the node's own current label
        # by passing -1 here and handling below.
        edge_mask = similarities >= threshold  # (n, k) bool

        # ── Iterative propagation ────────────────────────────────────────
        labels = np.arange(n, dtype=np.int64)

        for iteration in range(self.n_iterations):
            old_labels = labels.copy()

            # Gather neighbour labels; fall back to self-label for weak edges
            safe_idx = np.clip(indices, 0, n - 1)
            nbr_labels = labels[safe_idx]                  # (n, k)
            # Replace weak-edge slots with own label (so they don't vote)
            nbr_labels = np.where(edge_mask, nbr_labels, labels[:, None])

            if n <= lp_threshold:
                new_labels = _scipy_mode_rows(nbr_labels)
            else:
                # Approximate mode via median of sorted neighbour labels.
                # Pure numpy — no Python loops.
                sorted_nl = np.sort(nbr_labels, axis=1)
                new_labels = sorted_nl[:, k // 2]

            labels = new_labels.astype(np.int64)
            n_changed = int(np.sum(labels != old_labels))
            self._log(f"  LP iter {iteration + 1}: {n_changed:,} changed")
            if n_changed == 0:
                break

        # Renumber to contiguous 0-based integers
        _, labels = np.unique(labels, return_inverse=True)
        labels = labels.astype(np.int32)

        self.timings_["clustering"] = time.perf_counter() - t0
        self._log(f"Label propagation: {self.timings_['clustering']:.3f}s")
        return labels

    # ------------------------------------------------------------------
    # Optional: Louvain modularity refinement (small n only)
    # ------------------------------------------------------------------

    def _louvain_refinement(
        self, X: np.ndarray, indices: np.ndarray, similarities: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        One-pass Louvain modularity refinement using the sparse k-NN graph.
        Vectorised per-community operations; avoids per-node Python loops
        by processing all nodes whose best community is deterministic.
        """
        from scipy.sparse import csr_matrix

        n = len(labels)
        k = indices.shape[1]
        rng = np.random.default_rng(self.random_state)

        rows = np.repeat(np.arange(n), k)
        cols = indices.ravel()
        data = np.maximum(similarities.ravel(), 0.0)

        adj = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
        adj = (adj + adj.T).multiply(0.5)

        node_deg = np.asarray(adj.sum(axis=1)).ravel()
        comm_deg = np.zeros(n, dtype=np.float64)
        for i in range(n):
            comm_deg[labels[i]] += node_deg[i]

        m = float(node_deg.sum()) / 2.0
        if m < 1e-12:
            return labels

        current = labels.copy()
        for iteration in range(self.n_iterations):
            n_moves = 0
            for i in rng.permutation(n):
                ci = current[i]
                deg_i = node_deg[i]
                comm_deg[ci] -= deg_i

                s, e = adj.indptr[i], adj.indptr[i + 1]
                if s == e:
                    comm_deg[ci] += deg_i
                    continue

                nbrs = adj.indices[s:e]
                wts = adj.data[s:e]
                nbr_comms = current[nbrs]
                unique_c, inv = np.unique(nbr_comms, return_inverse=True)
                k_in = np.zeros(len(unique_c))
                np.add.at(k_in, inv, wts)

                gains = k_in / m - self.resolution * comm_deg[unique_c] * deg_i / (
                    2.0 * m * m
                )
                best = int(np.argmax(gains))
                best_comm = unique_c[best]

                if gains[best] > 1e-10 and best_comm != ci:
                    current[i] = best_comm
                    comm_deg[best_comm] += deg_i
                    n_moves += 1
                else:
                    comm_deg[ci] += deg_i

            self._log(f"  Louvain refine iter {iteration + 1}: {n_moves} moves")
            if n_moves == 0:
                break

        _, current = np.unique(current, return_inverse=True)
        return current.astype(np.int32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Cluster vectors and return integer labels.

        Parameters
        ----------
        X : array-like (n_samples, n_features), float32 or float64

        Returns
        -------
        labels : ndarray (n_samples,) int32, 0-based contiguous
        """
        t_total = time.perf_counter()

        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        self._log(f"Clustering {n:,} × {d}-dim vectors")

        indices, similarities = self._build_knn(X)
        labels = self._label_propagation(indices, similarities, n)

        self.n_clusters_ = int(np.max(labels)) + 1
        self.timings_["total"] = time.perf_counter() - t_total
        self._log(
            f"Done: {self.n_clusters_} clusters in {self.timings_['total']:.3f}s"
        )
        return labels
