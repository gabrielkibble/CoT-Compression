"""
DBSCAN-based chain compression.

Idea: within a single chain, some reasoning steps may be internally
redundant -- restating something already established, or hedging/
re-deriving the same quantity twice. If we cluster a chain's OWN chunks
by their residual state (density-based, via DBSCAN -- no need to pick a
cluster count k upfront, unlike k-means), steps that occupy very similar
internal states should land in the same cluster. Keeping only 1-2
representatives per cluster and dropping the rest gives a shortened chain
that (hopefully) still covers every genuinely distinct computational step.

IMPORTANT: DBSCAN-labeled "noise" points (chunks that don't fit any
dense cluster) are the MOST distinctive steps, not junk -- e.g. the
final "#### <answer>" line is very likely to be noise, since nothing else
in the chain looks like it. Noise points are ALWAYS kept, never dropped.

Runs on both layer 16 and the final layer (reusing layer_sweep_residuals.pkl,
no new forward passes needed) so you can compare which gives more
sensible/useful clusters.

Requires: layer_sweep_residuals.pkl, embed_matrix.npy, consensus_routes.pkl
          (to identify each problem's medoid chain)

pip install scikit-learn --break-system-packages   # if not already installed
"""

import pickle
import numpy as np
from sklearn.cluster import DBSCAN

RESIDUALS_CACHE = "layer_sweep_residuals.pkl"
EMBED_MATRIX_PATH = "embed_matrix.npy"
ROUTES_PATH = "consensus_routes.pkl"

LAYERS_TO_TRY = {"final": -1}   # -1 = last available layer index
EPS_VALUES = [0.45, 0.55, 0.65]   # DBSCAN neighborhood radius (cosine distance) --
                                   # try a few, pick by eye based on printed results
MIN_SAMPLES = 2                   # minimum cluster size (2 = smallest possible cluster)
N_REPS_PER_CLUSTER = 1            # how many representatives to keep per cluster


def _correct(residuals, last_token_ids, mean, std, embed_matrix):
    centered = residuals - mean
    tok_embeds = embed_matrix[last_token_ids].astype(np.float32)
    tok_dirs = tok_embeds / (np.linalg.norm(tok_embeds, axis=1, keepdims=True) + 1e-8)
    proj_scale = np.sum(centered * tok_dirs, axis=1, keepdims=True)
    projected = centered - proj_scale * tok_dirs
    return projected / std


def cosine_distance_matrix(a):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    dist = 1.0 - (a_norm @ a_norm.T)
    # Floating-point noise can push near-zero distances slightly negative
    # (e.g. -1e-8 for a vector compared to itself); sklearn's DBSCAN
    # strictly rejects any negative value in a precomputed distance
    # matrix, so clip.
    return np.clip(dist, 0.0, None)


def fit_correction_for_layer(records, layer_idx, embed_matrix):
    """Fit mean/std for one layer, pooled across the WHOLE dataset -- same
    approach as dtw_align_v2.py / layer_sweep.py, just parameterized here."""
    all_res, all_tok = [], []
    for r in records:
        all_res.append(r["chunk_residuals"][:, layer_idx, :])
        all_tok.append(r["last_token_ids"].numpy())
    all_res = np.concatenate(all_res, axis=0)
    all_tok = np.concatenate(all_tok, axis=0)
    mean = all_res.mean(axis=0)
    projected = _correct(all_res, all_tok, mean, np.ones_like(mean), embed_matrix)
    std = projected.std(axis=0) + 1e-6
    return mean, std


def compress_chain(residuals, texts, mean, std, last_token_ids, embed_matrix, eps):
    """
    Cluster this chain's own chunks, keep N_REPS_PER_CLUSTER representative(s)
    per cluster (closest to cluster centroid) plus ALL noise points.
    Returns: kept_indices (sorted, preserving original order), cluster_labels
    """
    corrected = _correct(residuals, last_token_ids, mean, std, embed_matrix)
    dist_matrix = cosine_distance_matrix(corrected)

    db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, metric="precomputed")
    labels = db.fit_predict(dist_matrix)

    kept = set()
    for label in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == label]
        if label == -1:
            kept.update(idxs)  # noise -- always keep, these are the distinctive steps
            continue
        # Representative(s): closest to the cluster's centroid.
        cluster_vecs = corrected[idxs]
        centroid = cluster_vecs.mean(axis=0)
        dists_to_centroid = np.linalg.norm(cluster_vecs - centroid, axis=1)
        order = np.argsort(dists_to_centroid)
        reps = [idxs[i] for i in order[:N_REPS_PER_CLUSTER]]
        kept.update(reps)

    kept_sorted = sorted(kept)
    return kept_sorted, labels


def main():
    with open(RESIDUALS_CACHE, "rb") as f:
        records = pickle.load(f)
    with open(ROUTES_PATH, "rb") as f:
        routes = pickle.load(f)
    embed_matrix = np.load(EMBED_MATRIX_PATH)

    # Index records by (problem_id, chain_idx) for quick medoid lookup.
    by_key = {(r["problem_id"], r["chain_idx"]): r for r in records}
    n_layers_available = records[0]["chunk_residuals"].shape[1]

    for layer_name, layer_idx in LAYERS_TO_TRY.items():
        resolved_idx = layer_idx if layer_idx >= 0 else n_layers_available - 1
        print(f"\n{'#'*70}\n# LAYER: {layer_name} (index {resolved_idx})\n{'#'*70}")

        mean, std = fit_correction_for_layer(records, resolved_idx, embed_matrix)

        # Run on each problem's medoid chain.
        for pid, route in list(routes.items())[:5]:   # first 5 problems -- exploratory
            medoid_chain_idx = route["chain_idx"]
            record = by_key.get((pid, medoid_chain_idx))
            if record is None:
                continue

            residuals = record["chunk_residuals"][:, resolved_idx, :]
            texts = [c["text"] for c in route["chunks"]]
            last_token_ids = record["last_token_ids"].numpy()

            if len(texts) != len(residuals):
                # Guard: chunk counts should match between consensus_routes.pkl
                # and layer_sweep_residuals.pkl since both derive from the same
                # chunked_captures data -- mismatch would indicate a data issue.
                print(f"[{pid}] SKIP -- chunk count mismatch ({len(texts)} vs {len(residuals)})")
                continue

            print(f"\n[{pid}] medoid chain {medoid_chain_idx}, {len(texts)} original chunks")
            for eps in EPS_VALUES:
                kept_idxs, labels = compress_chain(
                    residuals, texts, mean, std, last_token_ids, embed_matrix, eps
                )
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = int(np.sum(labels == -1))
                compressed_text = "\n".join(texts[i] for i in kept_idxs)
                print(f"  eps={eps}: {n_clusters} clusters, {n_noise} noise points, "
                      f"kept {len(kept_idxs)}/{len(texts)} chunks")
                print(f"    kept indices: {kept_idxs}")

            # Print the full compressed text for the LAST (largest) eps tried,
            # for a qualitative read.
            print(f"  --- Compressed text (eps={EPS_VALUES[-1]}) ---")
            print(f"  {compressed_text}")


if __name__ == "__main__":
    main()