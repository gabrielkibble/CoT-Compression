"""
Phase 3 v2: DTW alignment with anisotropy correction.

Same core idea as dtw_align.py, but the residuals are preprocessed first
to remove known confounds before computing cosine distance:

  1. Mean-center: subtract the global mean residual (computed across all
     chunks in the dataset) from every residual. Removes a shared "average
     direction" that would otherwise inflate similarity between ALL pairs
     regardless of problem.
  2. Project out the current-token direction: for the residual at the
     token actually generated at that position, remove the component
     along that token's own embedding direction. Residual streams encode
     "which token is this" strongly (they eventually feed the unembedding
     matrix), so two chunks ending in the same token look artificially
     similar without this.
  3. Standardize each dimension (z-score using per-dimension std computed
     across the whole dataset). Residual stream dimensions have very
     different scales/variances; without this, cosine similarity is
     dominated by a handful of high-variance dimensions.

This changes what gets DTW-aligned (the anisotropy-corrected vectors),
not the DTW algorithm itself.

We ALSO restructure the output: instead of just printing summary stats,
we save one distance MATRIX per problem (chain x chain), plus each
chain's correct/incorrect label. This lets a downstream stats script
recompute group means under label permutations without re-running DTW --
DTW is the expensive part, relabeling is free.

Requires: embed_matrix.npy (run extract_embeddings.py first).

Input:  chunked_captures/<problem_id>.pt
Output: phase3_pairwise.pkl  (per-problem dist matrices + labels)
        phase3_distances.npz (flat group distances, same as v1, for a
                               quick look without needing the stats script)
"""

import os
import glob
import pickle
import random
import itertools
import torch
import numpy as np

IN_DIR = "chunked_captures"
EMBED_MATRIX_PATH = "embed_matrix.npy"
RESIDUAL_KEY = "residual_last"
N_CROSS_PROBLEM_PAIRS = 2000
WORD_OVERLAP_THRESHOLD = 0.5
RANDOM_SEED = 0

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_chains():
    """
    Same as v1, but we also keep chain-level token_ids so we can look up
    the token id at each chunk's last position (needed for the
    current-token-direction projection).
    """
    records = []
    for path in sorted(glob.glob(os.path.join(IN_DIR, "*.pt"))):
        data = torch.load(path, weights_only=False)
        pid = data["problem_id"]
        for chain in data["chains"]:
            chunks = chain["chunks"]
            if len(chunks) < 2:
                continue
            residuals = np.stack([c[RESIDUAL_KEY].numpy() for c in chunks]).astype(np.float32)
            texts = [c["text"] for c in chunks]
            token_ids = chain["token_ids"]
            # Token id actually generated at each chunk's last position --
            # this is what the residual "commits to" at that step.
            last_token_ids = np.array([token_ids[c["token_end"]].item() for c in chunks])
            records.append({
                "problem_id": pid,
                "chain_idx": chain["chain_idx"],
                "correct": chain["correct"],
                "residuals": residuals,        # (n_chunks, hidden), RAW
                "texts": texts,
                "last_token_ids": last_token_ids,  # (n_chunks,)
            })
    return records


# ---------------------------------------------------------------------------
# Anisotropy correction
# ---------------------------------------------------------------------------
def fit_correction(records, embed_matrix):
    """
    Fit the mean vector and per-dimension std from the WHOLE dataset
    (across all chunks, all chains, all problems). This is the background
    corpus the correction is calibrated against -- fitting per-problem
    would be circular (it could remove the very signal we're testing for).
    """
    all_residuals = np.concatenate([r["residuals"] for r in records], axis=0)
    global_mean = all_residuals.mean(axis=0)  # (hidden,)

    # Std is computed AFTER mean-centering and token-direction projection,
    # since std should reflect the corrected space, not the raw one.
    corrected_all = []
    for r in records:
        corrected_all.append(
            _correct(r["residuals"], r["last_token_ids"], global_mean, embed_matrix, std=None)
        )
    corrected_all = np.concatenate(corrected_all, axis=0)
    global_std = corrected_all.std(axis=0) + 1e-6  # avoid divide-by-zero

    return global_mean, global_std


def _correct(residuals, last_token_ids, mean, embed_matrix, std):
    """
    Apply mean-centering + current-token-direction projection + (optional)
    standardization to a (n_chunks, hidden) block of residuals.
    If std is None, standardization is skipped (used when fitting std itself).
    """
    centered = residuals - mean  # (n, hidden)

    # Project out each row's own token-embedding direction.
    tok_embeds = embed_matrix[last_token_ids].astype(np.float32)          # (n, hidden)
    tok_dirs = tok_embeds / (np.linalg.norm(tok_embeds, axis=1, keepdims=True) + 1e-8)
    proj_scale = np.sum(centered * tok_dirs, axis=1, keepdims=True)       # (n, 1)
    projected = centered - proj_scale * tok_dirs                          # (n, hidden)

    if std is None:
        return projected
    return projected / std


def correct_all(records, mean, std, embed_matrix):
    """Apply the fitted correction to every record's residuals in place."""
    for r in records:
        r["residuals_corrected"] = _correct(
            r["residuals"], r["last_token_ids"], mean, embed_matrix, std
        )
    return records


# ---------------------------------------------------------------------------
# DTW (same as v1)
# ---------------------------------------------------------------------------
def cosine_distance_matrix(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return 1.0 - (a_norm @ b_norm.T)


def dtw_align(residuals_a, residuals_b):
    n, m = len(residuals_a), len(residuals_b)
    cost = cosine_distance_matrix(residuals_a, residuals_b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        _, i, j = min(
            [(D[i - 1, j - 1], i - 1, j - 1), (D[i - 1, j], i - 1, j), (D[i, j - 1], i, j - 1)],
            key=lambda x: x[0],
        )
    path.reverse()
    return D[n, m] / len(path), path, cost


def word_overlap(text_a, text_b):
    words_a, words_b = set(text_a.lower().split()), set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def path_avg_cost_filtered(path, cost_matrix, texts_a, texts_b, keep_low_overlap):
    costs = [
        cost_matrix[i, j] for i, j in path
        if (word_overlap(texts_a[i], texts_b[j]) < WORD_OVERLAP_THRESHOLD) == keep_low_overlap
    ]
    return float(np.mean(costs)) if costs else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def summarize(name, values):
    if not values:
        print(f"{name:45s} n=0")
        return
    arr = np.array(values)
    print(f"{name:45s} n={len(arr):5d}  mean={arr.mean():.4f}  "
          f"std={arr.std():.4f}  median={np.median(arr):.4f}")


def main():
    embed_matrix = np.load(EMBED_MATRIX_PATH)
    print(f"Loaded embedding matrix {embed_matrix.shape}")

    records = load_all_chains()
    print(f"Loaded {len(records)} chains total")

    print("Fitting anisotropy correction (mean, std) from full dataset...")
    global_mean, global_std = fit_correction(records, embed_matrix)
    records = correct_all(records, global_mean, global_std, embed_matrix)
    print("Correction applied.\n")

    # Save the fitted correction parameters so downstream scripts (Phase 4
    # consensus routes, Phase 5 search) can reproduce the EXACT same
    # corrected states rather than refitting mean/std from scratch, which
    # could drift if run on a different subset of data.
    np.savez("anisotropy_correction.npz", mean=global_mean, std=global_std)
    print("Saved -> anisotropy_correction.npz (mean, std used for correction)\n")

    by_problem = {}
    for r in records:
        by_problem.setdefault(r["problem_id"], []).append(r)

    # ---- Same-problem pairs: store full pairwise structure per problem ----
    pairwise_store = {}  # pid -> {"chain_idx": [...], "correct": [...], "dist_matrix": (n,n)}
    same_cc, same_ci, same_ii = [], [], []
    same_cc_lo, same_ci_lo, same_ii_lo = [], [], []

    for pid, chains in by_problem.items():
        n = len(chains)
        dist_matrix = np.full((n, n), np.nan)
        for (ia, a), (ib, b) in itertools.combinations(enumerate(chains), 2):
            avg_cost, path, cost_matrix = dtw_align(a["residuals_corrected"], b["residuals_corrected"])
            dist_matrix[ia, ib] = dist_matrix[ib, ia] = avg_cost

            lo_cost = path_avg_cost_filtered(path, cost_matrix, a["texts"], b["texts"], True)
            if a["correct"] and b["correct"]:
                same_cc.append(avg_cost)
                if lo_cost is not None:
                    same_cc_lo.append(lo_cost)
            elif not a["correct"] and not b["correct"]:
                same_ii.append(avg_cost)
                if lo_cost is not None:
                    same_ii_lo.append(lo_cost)
            else:
                same_ci.append(avg_cost)
                if lo_cost is not None:
                    same_ci_lo.append(lo_cost)

        pairwise_store[pid] = {
            "chain_idx": [c["chain_idx"] for c in chains],
            "correct": [c["correct"] for c in chains],
            "dist_matrix": dist_matrix,
        }

    # ---- Cross-problem baseline ----
    cross = []
    problem_ids = list(by_problem.keys())
    attempts = 0
    while len(cross) < N_CROSS_PROBLEM_PAIRS and attempts < N_CROSS_PROBLEM_PAIRS * 5:
        attempts += 1
        pid_a, pid_b = random.sample(problem_ids, 2)
        a = random.choice(by_problem[pid_a])
        b = random.choice(by_problem[pid_b])
        avg_cost, _, _ = dtw_align(a["residuals_corrected"], b["residuals_corrected"])
        cross.append(avg_cost)

    # ---- Report ----
    print("=== DTW distance, ANISOTROPY-CORRECTED (lower = more similar) ===\n")
    summarize("same-problem, correct vs correct", same_cc)
    summarize("same-problem, correct vs incorrect", same_ci)
    summarize("same-problem, incorrect vs incorrect", same_ii)
    summarize("cross-problem (baseline)", cross)

    print("\n=== Same, restricted to LOW word-overlap aligned chunks ===\n")
    summarize("same-problem, correct vs correct (low overlap)", same_cc_lo)
    summarize("same-problem, correct vs incorrect (low overlap)", same_ci_lo)
    summarize("same-problem, incorrect vs incorrect (low overlap)", same_ii_lo)

    np.savez(
        "phase3_distances.npz",
        same_cc=same_cc, same_ci=same_ci, same_ii=same_ii, cross=cross,
        same_cc_lo=same_cc_lo, same_ci_lo=same_ci_lo, same_ii_lo=same_ii_lo,
    )
    with open("phase3_pairwise.pkl", "wb") as f:
        pickle.dump(pairwise_store, f)

    print("\nSaved -> phase3_distances.npz (flat groups, quick look)")
    print("Saved -> phase3_pairwise.pkl (per-problem matrices, for clustered stats)")


if __name__ == "__main__":
    main()