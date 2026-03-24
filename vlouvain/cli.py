"""
VLouvain CLI — entry-point for `vlouvain` command.
"""

from __future__ import annotations

import os
import sys
import time

import click
import numpy as np


@click.group()
@click.version_option()
def cli():
    """VLouvain: cluster 1M vectors in under 1 second without building a graph."""


# ---------------------------------------------------------------------------
# vlouvain cluster
# ---------------------------------------------------------------------------

@cli.command("cluster")
@click.option(
    "--input", "-i", "input_path", required=True,
    type=click.Path(exists=True),
    help="Path to input embeddings (.npy or .npz).",
)
@click.option(
    "--output", "-o", "output_path", default=None,
    type=click.Path(),
    help="Path to save cluster labels (.npy).  Default: <input>_labels.npy",
)
@click.option(
    "--k", default=None, type=int,
    help="Number of nearest neighbors (default: $VLOUVAIN_K or 15).",
)
@click.option(
    "--n-iter", default=None, type=int,
    help="Max label-propagation iterations (default: $VLOUVAIN_N_ITER or 10).",
)
@click.option(
    "--resolution", default=None, type=float,
    help="Louvain resolution (default: $VLOUVAIN_RESOLUTION or 1.0).",
)
@click.option(
    "--threshold", default=None, type=float,
    help="Cosine similarity threshold for edges (default: auto).",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False,
    help="Print timing breakdowns.",
)
def cluster_cmd(input_path, output_path, k, n_iter, resolution, threshold, verbose):
    """Cluster embeddings stored in a .npy or .npz file."""
    from vlouvain.algorithm import VLouvain

    k = k or int(os.getenv("VLOUVAIN_K", "15"))
    n_iter = n_iter or int(os.getenv("VLOUVAIN_N_ITER", "10"))
    resolution = resolution or float(os.getenv("VLOUVAIN_RESOLUTION", "1.0"))

    click.echo(f"Loading {input_path} …")
    if input_path.endswith(".npz"):
        data = np.load(input_path)
        key = list(data.keys())[0]
        X = data[key]
    else:
        X = np.load(input_path)

    if X.ndim != 2:
        click.echo(f"ERROR: expected 2-D array, got shape {X.shape}", err=True)
        sys.exit(1)

    n, d = X.shape
    click.echo(f"Loaded {n:,} × {d}-dim embeddings")

    model = VLouvain(
        k=k,
        n_iterations=n_iter,
        resolution=resolution,
        similarity_threshold=threshold,
        verbose=verbose,
    )

    t0 = time.perf_counter()
    labels = model.fit_predict(X)
    elapsed = time.perf_counter() - t0

    n_clusters = int(np.max(labels)) + 1
    click.echo(f"Found {n_clusters} clusters in {elapsed:.3f}s")

    if output_path is None:
        base = input_path.rsplit(".", 1)[0]
        output_path = base + "_labels.npy"

    np.save(output_path, labels)
    click.echo(f"Saved labels → {output_path}")


# ---------------------------------------------------------------------------
# vlouvain bench
# ---------------------------------------------------------------------------

@cli.command("bench")
@click.option(
    "--n", default=None, type=int,
    help="Number of vectors (default: $VLOUVAIN_BENCH_N or 1000000).",
)
@click.option(
    "--clusters", default=None, type=int,
    help="Number of ground-truth clusters (default: $VLOUVAIN_BENCH_CLUSTERS or 100).",
)
@click.option(
    "--k", default=None, type=int,
    help="k for VLouvain (default: 5 for large n).",
)
@click.option(
    "--hdbscan", "include_hdbscan", is_flag=True, default=False,
    help="Also run HDBSCAN (slow; capped at --hdbscan-max-n).",
)
@click.option(
    "--hdbscan-max-n", default=50_000, type=int,
    help="Max n for which HDBSCAN is run (default: 50000).",
)
def bench_cmd(n, clusters, k, include_hdbscan, hdbscan_max_n):
    """Benchmark VLouvain vs MiniBatchKMeans on synthetic vectors."""
    from vlouvain.benchmark import run_benchmark

    n = n or int(os.getenv("VLOUVAIN_BENCH_N", "1000000"))
    clusters = clusters or int(os.getenv("VLOUVAIN_BENCH_CLUSTERS", "100"))
    k = k or int(os.getenv("VLOUVAIN_K", "5"))

    results = run_benchmark(
        n=n,
        n_clusters=clusters,
        k=k,
        include_hdbscan=include_hdbscan,
        hdbscan_max_n=hdbscan_max_n,
        verbose=True,
    )

    vl = results["vlouvain"]
    km = results["kmeans"]
    speedup = km["time"] / max(vl["time"], 1e-6)

    click.echo(
        f"\nResult: VLouvain {vl['time']:.3f}s  |  "
        f"KMeans {km['time']:.3f}s  |  "
        f"speedup {speedup:.1f}x"
    )


def main():
    cli()


if __name__ == "__main__":
    main()
