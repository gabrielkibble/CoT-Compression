"""
Phase 3, velocity channel: DTW on STEP-DIFFERENCES of the residual stream.

dtw_align_v2.py compares chains on their internal STATE (the anisotropy-
corrected residual at each chunk). This script compares them on internal
MOTION instead: how much and in what direction the state moved from one
chunk to the next.

  velocity[i] = residual_corrected[i+1] - residual_corrected[i]

Rationale: two chains could sit in similar regions of state-space (similar
"level") while getting there very differently, or vice versa. If the
"shared route" claim is right, correct chains should agree not just on
where they are but on how they're moving -- which step produces which
kind of update. This is a genuinely different signal from the level
channel, not just a transformation of it.

Reuses the SAME anisotropy correction as dtw_align_v2.py (mean-centered,
token-direction-projected, standardized residuals) -- we difference the
corrected states, not the raw ones, since we want the velocity signal to
also be free of the anisotropy confound.

A chain needs >= 3 chunks to produce >= 2 velocity vectors for DTW to
align meaningfully; shorter chains are dropped here (noted in output).

Requires phase3_pairwise.pkl's ingredients to already be validated by
dtw_align_v2.py -- this script duplicates its correction + DTW machinery
rather than importing it, to keep each script runnable standalone.

Input:  chunked_captures/<problem_id>.pt, embed_matrix.npy
Output: phase3_distances_velocity.npz, phase3_pairwise_velocity.pkl
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
MIN_CHUNKS = 3  # need >=3 chunks -> >=2 velocity vectors

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Data loading (same as dtw_align_v2.py)
# ---------------------------------------------------------------------------
def load_all_chains():
    records = []
    for path in sorted(glob.glob(os.path.join(IN_DIR, "*.pt"))):
        data = torch.load(path, weights_only=False)
        pid = data["problem_id"]
        for chain in data["chains"]:
            chunks = chain["chunks"]
            if len(chunks) < MIN_CHUNKS:
                continue
            residuals = np.stack([c[RESIDUAL_KEY].numpy() for c in chunks]).astype(np.float32)
            texts = [c["text"] for c in chunks]
            token_ids = chain["token_ids"]
            last_token_ids = np.array([token_ids[c["token_end"]].item() for c in chunks])
            records.append({
                "problem_id": pid,
                "chain_idx": chain["chain_idx"],
                "correct": chain["correct"],
                "residuals": residuals,
                "texts": texts,
                "last_token_ids": last_token_ids,
            })
    return records


# ---------------------------------------------------------------------------
# Anisotropy correction (identical to dtw_align_v2.py)
# ---------------------------------------------------------------------------
def _correct(residuals, last_token_ids, mean, embed_matrix, std):
    centered = residuals - mean
    tok_embeds = embed_matrix[last_token_ids].astype(np.float32)
    tok_dirs = tok_embeds / (np.linalg.norm(tok_embeds, axis=1, keepdims=True) + 1e-8)
    proj_scale = np.sum(centered * tok_dirs, axis=1, keepdims=True)
    projected = centered - proj_scale * tok_dirs
    if std is None:
        return projected
    return projected / std


def fit_correction(records, embed_matrix):
    all_residuals = np.concatenate([r["residuals"] for r in records], axis=0)
    global_mean = all_residuals.mean(axis=0)
    corrected_all = np.concatenate([
        _correct(r["residuals"], r["last_token_ids"], global_mean, embed_matrix, std=None)
        for r in records
    ], axis=0)
    global_std = corrected_all.std(axis=0) + 1e-6
    return global_mean, global_std


def correct_and_diff_all(records, mean, std, embed_matrix):
    """
    Apply correction, then take consecutive differences to get the
    velocity sequence. Stores under 'velocity' (n_chunks-1, hidden).
    Also keeps chunk-pair texts for the word-overlap control: each
    velocity vector i is associated with the transition from chunk i to
    chunk i+1, so we tag it with chunk i+1's text (the step it arrives at).
    """
    for r in records:
        corrected = _correct(r["residuals"], r["last_token_ids"], mean, embed_matrix, std)
        r["velocity"] = corrected[1:] - corrected[:-1]        # (n-1, hidden)
        r["velocity_texts"] = r["texts"][1:]                  # (n-1,)
    return records


# ---------------------------------------------------------------------------
# DTW (identical machinery to dtw_align_v2.py)
# ---------------------------------------------------------------------------
def cosine_distance_matrix(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return 1.0 - (a_norm @ b_norm.T)


def dtw_align(seq_a, seq_b):
    n, m = len(seq_a), len(seq_b)
    cost = cosine_distance_matrix(seq_a, seq_b)
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
    records = load_all_chains()
    print(f"Loaded {len(records)} chains with >= {MIN_CHUNKS} chunks (velocity needs >=2 steps)")

    global_mean, global_std = fit_correction(records, embed_matrix)
    records = correct_and_diff_all(records, global_mean, global_std, embed_matrix)
    print("Correction + differencing applied.\n")

    by_problem = {}
    for r in records:
        by_problem.setdefault(r["problem_id"], []).append(r)

    pairwise_store = {}
    same_cc, same_ci, same_ii = [], [], []
    same_cc_lo, same_ci_lo, same_ii_lo = [], [], []

    for pid, chains in by_problem.items():
        n = len(chains)
        dist_matrix = np.full((n, n), np.nan)
        for (ia, a), (ib, b) in itertools.combinations(enumerate(chains), 2):
            avg_cost, path, cost_matrix = dtw_align(a["velocity"], b["velocity"])
            dist_matrix[ia, ib] = dist_matrix[ib, ia] = avg_cost

            lo_cost = path_avg_cost_filtered(
                path, cost_matrix, a["velocity_texts"], b["velocity_texts"], True
            )
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

    cross = []
    problem_ids = list(by_problem.keys())
    attempts = 0
    while len(cross) < N_CROSS_PROBLEM_PAIRS and attempts < N_CROSS_PROBLEM_PAIRS * 5:
        attempts += 1
        pid_a, pid_b = random.sample(problem_ids, 2)
        a = random.choice(by_problem[pid_a])
        b = random.choice(by_problem[pid_b])
        avg_cost, _, _ = dtw_align(a["velocity"], b["velocity"])
        cross.append(avg_cost)

    print("=== DTW distance on VELOCITY (step-differences), corrected ===\n")
    summarize("same-problem, correct vs correct", same_cc)
    summarize("same-problem, correct vs incorrect", same_ci)
    summarize("same-problem, incorrect vs incorrect", same_ii)
    summarize("cross-problem (baseline)", cross)

    print("\n=== Same, restricted to LOW word-overlap aligned steps ===\n")
    summarize("same-problem, correct vs correct (low overlap)", same_cc_lo)
    summarize("same-problem, correct vs incorrect (low overlap)", same_ci_lo)
    summarize("same-problem, incorrect vs incorrect (low overlap)", same_ii_lo)

    np.savez(
        "phase3_distances_velocity.npz",
        same_cc=same_cc, same_ci=same_ci, same_ii=same_ii, cross=cross,
        same_cc_lo=same_cc_lo, same_ci_lo=same_ci_lo, same_ii_lo=same_ii_lo,
    )
    with open("phase3_pairwise_velocity.pkl", "wb") as f:
        pickle.dump(pairwise_store, f)

    print("\nSaved -> phase3_distances_velocity.npz")
    print("Saved -> phase3_pairwise_velocity.pkl  (feed into cluster_stats.py by changing PAIRWISE_PATH)")


if __name__ == "__main__":
    main()