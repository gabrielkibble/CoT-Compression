"""
Path A validation: does R_latent (negative DTW distance to the consensus
route) actually correlate with correctness?

This is a cheap sanity check BEFORE building any RL training loop. It
reuses data you already have -- no new generation, no training:

  - chunked_captures/<pid>.pt   (Phase 2: every sampled chain's chunks)
  - consensus_routes.pkl        (Phase 4: each problem's medoid route)
  - anisotropy_correction.npz, embed_matrix.npy   (same correction as
    Phase 3/4, applied identically here for consistency)

For every sampled chain EXCEPT the medoid itself (comparing the medoid to
its own route is trivially distance ~0 and would bias the result), we
compute DTW distance between that chain's corrected residual sequence and
its problem's consensus route. Then we check: do CORRECT chains sit
reliably closer to the route than INCORRECT chains?

This is exactly R_latent from the proposed GRPO reward (up to the minor
normalization difference noted below), just computed offline on chains
you already generated, instead of on live RL rollouts.

Output: printed summary + reward_validation.pkl (per-chain distances, for
further analysis/plotting).
"""

import glob
import pickle
import random
import itertools
import torch
import numpy as np

CHUNKED_DIR = "chunked_captures"
ROUTES_PATH = "consensus_routes.pkl"
CORRECTION_PATH = "anisotropy_correction.npz"
EMBED_MATRIX_PATH = "embed_matrix.npy"
RESIDUAL_KEY = "residual_last"
N_PERMUTATIONS = 5000
N_BOOTSTRAP = 5000
RANDOM_SEED = 0

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Anisotropy correction + DTW (identical to dtw_align_v2.py, duplicated here
# so this script runs standalone)
# ---------------------------------------------------------------------------
def _correct(residuals, last_token_ids, mean, std, embed_matrix):
    centered = residuals - mean
    tok_embeds = embed_matrix[last_token_ids].astype(np.float32)
    tok_dirs = tok_embeds / (np.linalg.norm(tok_embeds, axis=1, keepdims=True) + 1e-8)
    proj_scale = np.sum(centered * tok_dirs, axis=1, keepdims=True)
    projected = centered - proj_scale * tok_dirs
    return projected / std


def cosine_distance_matrix(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return 1.0 - (a_norm @ b_norm.T)


def dtw_align(residuals_a, residuals_b):
    """Same DTW as dtw_align_v2.py -- returns path-length-normalized cost."""
    n, m = len(residuals_a), len(residuals_b)
    cost = cosine_distance_matrix(residuals_a, residuals_b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    path_len = 0
    i, j = n, m
    while i > 0 and j > 0:
        path_len += 1
        _, i, j = min(
            [(D[i - 1, j - 1], i - 1, j - 1), (D[i - 1, j], i - 1, j), (D[i, j - 1], i, j - 1)],
            key=lambda x: x[0],
        )
    return D[n, m] / path_len  # avg_cost, same normalization as Phase 3


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    correction = np.load(CORRECTION_PATH)
    mean, std = correction["mean"], correction["std"]
    embed_matrix = np.load(EMBED_MATRIX_PATH)

    with open(ROUTES_PATH, "rb") as f:
        routes = pickle.load(f)

    records = []  # (problem_id, chain_idx, correct, dist_to_route)

    for path in sorted(glob.glob(f"{CHUNKED_DIR}/*.pt")):
        data = torch.load(path, weights_only=False)
        pid = data["problem_id"]
        if pid not in routes:
            continue

        route = routes[pid]
        medoid_chain_idx = route["chain_idx"]
        route_residuals = np.stack([c["residual_corrected"] for c in route["chunks"]])

        for chain in data["chains"]:
            if chain["chain_idx"] == medoid_chain_idx:
                continue  # skip the chain the route was built FROM -- trivial ~0 distance
            chunks = chain["chunks"]
            if len(chunks) < 2:
                continue

            residuals = np.stack([c[RESIDUAL_KEY].numpy() for c in chunks]).astype(np.float32)
            token_ids = chain["token_ids"]
            last_token_ids = np.array([token_ids[c["token_end"]].item() for c in chunks])
            corrected = _correct(residuals, last_token_ids, mean, std, embed_matrix)

            dist = dtw_align(corrected, route_residuals)
            records.append({
                "problem_id": pid,
                "chain_idx": chain["chain_idx"],
                "correct": chain["correct"],
                "dist_to_route": dist,
            })

    print(f"Scored {len(records)} chains against their problem's consensus route "
          f"(medoid chains excluded)\n")

    # ---- Unclustered quick look ----
    correct_dists = [r["dist_to_route"] for r in records if r["correct"]]
    incorrect_dists = [r["dist_to_route"] for r in records if not r["correct"]]

    def summarize(name, values):
        arr = np.array(values)
        print(f"{name:20s} n={len(arr):4d}  mean={arr.mean():.4f}  "
              f"std={arr.std():.4f}  median={np.median(arr):.4f}")

    summarize("correct", correct_dists)
    summarize("incorrect", incorrect_dists)
    print(f"\nGap (incorrect - correct) = {np.mean(incorrect_dists) - np.mean(correct_dists):.4f}  "
          f"(positive = correct chains ARE closer to the route, as hypothesized)\n")

    # ---- Cluster bootstrap CIs (resample by problem) ----
    by_problem = {}
    for r in records:
        by_problem.setdefault(r["problem_id"], []).append(r)
    problem_ids = list(by_problem.keys())

    def cluster_bootstrap(want_correct):
        boot_means = []
        for _ in range(N_BOOTSTRAP):
            sampled_pids = np.random.choice(problem_ids, size=len(problem_ids), replace=True)
            pooled = []
            for pid in sampled_pids:
                pooled.extend(r["dist_to_route"] for r in by_problem[pid] if r["correct"] == want_correct)
            if pooled:
                boot_means.append(np.mean(pooled))
        return np.array(boot_means)

    boot_correct = cluster_bootstrap(True)
    boot_incorrect = cluster_bootstrap(False)

    def report_ci(name, boot):
        if len(boot) == 0:
            print(f"{name:20s} no data")
            return
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{name:20s} mean={boot.mean():.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

    print("=== Cluster bootstrap 95% CIs (resampled by problem) ===")
    report_ci("correct", boot_correct)
    report_ci("incorrect", boot_incorrect)

    # ---- Permutation test: shuffle correct/incorrect labels WITHIN each
    # problem, recompute the gap, build a null distribution. ----
    def pooled_gap(label_map=None):
        c_all, i_all = [], []
        for pid, rs in by_problem.items():
            labels = label_map[pid] if label_map else {r["chain_idx"]: r["correct"] for r in rs}
            c = [r["dist_to_route"] for r in rs if labels[r["chain_idx"]]]
            i = [r["dist_to_route"] for r in rs if not labels[r["chain_idx"]]]
            if c:
                c_all.append(np.mean(c))
            if i:
                i_all.append(np.mean(i))
        if not c_all or not i_all:
            return None
        return np.mean(i_all) - np.mean(c_all)

    observed_gap = pooled_gap()
    null_gaps = []
    for _ in range(N_PERMUTATIONS):
        shuffled = {}
        for pid, rs in by_problem.items():
            labels = [r["correct"] for r in rs]
            random.shuffle(labels)
            shuffled[pid] = {r["chain_idx"]: lab for r, lab in zip(rs, labels)}
        gap = pooled_gap(shuffled)
        if gap is not None:
            null_gaps.append(gap)
    null_gaps = np.array(null_gaps)
    p_value = np.mean(null_gaps >= observed_gap)

    print(f"\n=== Permutation test (within-problem label shuffle) ===")
    print(f"Observed gap = {observed_gap:.4f}")
    print(f"Null: mean={null_gaps.mean():.4f}  std={null_gaps.std():.4f}")
    print(f"One-sided p-value (P[null >= observed]) = {p_value:.4f}")

    with open("reward_validation.pkl", "wb") as f:
        pickle.dump(records, f)
    print("\nSaved -> reward_validation.pkl")


if __name__ == "__main__":
    main()