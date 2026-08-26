"""
Phase 4: Consensus route construction.

For each problem, we already have (from Phase 3) a DTW distance matrix
between every pair of sampled chains, plus which chains were correct.
Here we use that to pick, per problem, the single correct chain that is
most "typical" of the correct-chain cluster: the MEDOID -- the chain with
the lowest average DTW distance to every other correct chain for that
problem.

Why medoid and not an averaged/synthetic trajectory (e.g. DTW barycenter
averaging): a medoid is a real, coherent chain that the model actually
generated -- guaranteed to be readable, on-distribution, and internally
consistent. A synthetic average of residual vectors risks landing on a
state the model never actually visits (off-manifold), which would be a
bad target to search towards in Phase 5. We can revisit DBA later if the
medoid route turns out too tied to one chain's idiosyncratic phrasing.

For each problem's medoid chain we save the full chunk-level record
(text, corrected residual, probability trace) -- this is the "consensus
route" Phase 5 will score candidate short chains against.

Requires, from earlier phases:
  - phase3_pairwise.pkl        (correct/incorrect labels + dist matrices)
  - anisotropy_correction.npz  (mean, std used to correct residuals)
  - embed_matrix.npy           (token embeddings, for the correction)
  - chunked_captures/<pid>.pt  (the actual chunk data, Phase 2 output)

Output: consensus_routes.pkl
  pid -> {
    "chain_idx": int,               # which sampled chain was the medoid
    "avg_dist_to_other_correct": float,
    "n_correct_chains": int,        # how many correct chains it was chosen among
    "chunks": [
        {"text", "token_start", "token_end",
         "residual_corrected", "prob_last", "prob_mean"},
        ...
    ],
  }
"""

import pickle
import numpy as np
import torch

PAIRWISE_PATH = "phase3_pairwise.pkl"
CORRECTION_PATH = "anisotropy_correction.npz"
EMBED_MATRIX_PATH = "embed_matrix.npy"
CHUNKED_DIR = "chunked_captures"
RESIDUAL_KEY = "residual_last"
OUT_PATH = "consensus_routes.pkl"


def correct_residual(residuals, last_token_ids, mean, std, embed_matrix):
    """Same anisotropy correction as dtw_align_v2.py -- must match exactly."""
    centered = residuals - mean
    tok_embeds = embed_matrix[last_token_ids].astype(np.float32)
    tok_dirs = tok_embeds / (np.linalg.norm(tok_embeds, axis=1, keepdims=True) + 1e-8)
    proj_scale = np.sum(centered * tok_dirs, axis=1, keepdims=True)
    projected = centered - proj_scale * tok_dirs
    return projected / std


def pick_medoid(dist_matrix: np.ndarray, correct_labels: list) -> tuple:
    """
    Among chains where correct_labels[i] is True, find the index (into
    the LOCAL chain list for this problem) with lowest average distance
    to the other correct chains. Returns (local_idx, avg_dist,
    n_correct_chains), or (None, None, n) if fewer than 2 correct chains
    exist (can't compute a meaningful average distance with just one).
    """
    correct_idxs = [i for i, c in enumerate(correct_labels) if c]
    n_correct = len(correct_idxs)

    if n_correct == 0:
        return None, None, 0
    if n_correct == 1:
        # Only one correct chain -- it's the medoid by default, but we
        # flag avg_dist as None since there's nothing to average against.
        return correct_idxs[0], None, 1

    best_idx, best_avg = None, np.inf
    for i in correct_idxs:
        dists = [dist_matrix[i, j] for j in correct_idxs if j != i]
        avg = np.mean(dists)
        if avg < best_avg:
            best_avg = avg
            best_idx = i
    return best_idx, best_avg, n_correct


def main():
    with open(PAIRWISE_PATH, "rb") as f:
        pairwise_store = pickle.load(f)

    correction = np.load(CORRECTION_PATH)
    mean, std = correction["mean"], correction["std"]
    embed_matrix = np.load(EMBED_MATRIX_PATH)

    consensus_routes = {}
    skipped = []

    for pid, data in pairwise_store.items():
        dist_matrix = data["dist_matrix"]
        correct_labels = data["correct"]
        chain_idxs = data["chain_idx"]  # maps local position -> original chain_idx

        local_medoid_idx, avg_dist, n_correct = pick_medoid(dist_matrix, correct_labels)

        if local_medoid_idx is None:
            skipped.append(pid)
            print(f"[{pid}] SKIPPED -- no correct chains found (unexpected given Phase 0 filter)")
            continue

        medoid_chain_idx = chain_idxs[local_medoid_idx]

        # Pull the actual chunk data for this chain from Phase 2's output.
        chunked_data = torch.load(f"{CHUNKED_DIR}/{pid}.pt", weights_only=False)
        medoid_chain = next(
            c for c in chunked_data["chains"] if c["chain_idx"] == medoid_chain_idx
        )

        chunks = medoid_chain["chunks"]
        token_ids = medoid_chain["token_ids"]
        residuals = np.stack([c[RESIDUAL_KEY].numpy() for c in chunks]).astype(np.float32)
        last_token_ids = np.array([token_ids[c["token_end"]].item() for c in chunks])
        corrected = correct_residual(residuals, last_token_ids, mean, std, embed_matrix)

        route_chunks = []
        for i, c in enumerate(chunks):
            route_chunks.append({
                "text": c["text"],
                "token_start": c["token_start"],
                "token_end": c["token_end"],
                "residual_corrected": corrected[i],
                "prob_last": c["prob_last"],
                "prob_mean": c["prob_mean"],
            })

        total_tokens = chunks[-1]["token_end"] + 1
        consensus_routes[pid] = {
            "chain_idx": medoid_chain_idx,
            "avg_dist_to_other_correct": avg_dist,
            "n_correct_chains": n_correct,
            "total_tokens": total_tokens,
            "chunks": route_chunks,
        }

        dist_str = f"{avg_dist:.4f}" if avg_dist is not None else "n/a (only 1 correct chain)"
        print(f"[{pid}] medoid=chain {medoid_chain_idx}  "
              f"n_correct={n_correct}  avg_dist={dist_str}  "
              f"n_chunks={len(route_chunks)}  tokens={total_tokens}")

    with open(OUT_PATH, "wb") as f:
        pickle.dump(consensus_routes, f)

    print(f"\nSaved {len(consensus_routes)} consensus routes -> {OUT_PATH}")
    if skipped:
        print(f"Skipped {len(skipped)} problems with no correct chains: {skipped}")


if __name__ == "__main__":
    main()