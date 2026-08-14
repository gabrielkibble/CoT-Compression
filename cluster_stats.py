"""
Clustered statistics on the Phase 3 pairwise distances.

Why "clustered": every chain-pair distance within a problem is NOT an
independent observation -- pairs sharing a chain are correlated, and all
16-choose-2=120 pairs from one problem share whatever made that problem
easy/hard/well-conditioned for the model. Treating all pairs as i.i.d.
(as a naive mean/std does) understates uncertainty. We fix this two ways:

1. CLUSTER BOOTSTRAP (for confidence intervals on each group's mean):
   resample PROBLEMS with replacement (not individual pairs), recompute
   the group mean from whichever problems got drawn, repeat thousands of
   times -> gives a CI that respects the problem-level clustering.

2. PERMUTATION TEST (for "is correct-correct really tighter than
   incorrect-incorrect?"): within each problem, shuffle which chains are
   labeled correct/incorrect (keeping the same number of each), re-derive
   group means from the SAME already-computed distance matrix (no DTW
   recompute needed -- that's the point of saving the pairwise matrices),
   and see how often a random relabeling produces as extreme a gap as the
   real one. This only reshuffles labels WITHIN a problem, so it respects
   clustering by construction.

Input:  phase3_pairwise.pkl  (from dtw_align_v2.py)
Output: printed CIs and p-values
"""

import pickle
import random
import sys
import numpy as np

# Usage: python3 cluster_stats.py [path_to_pairwise.pkl]
# Defaults to the level-channel output; pass phase3_pairwise_velocity.pkl
# to run the same clustered stats on the velocity channel instead.
PAIRWISE_PATH = sys.argv[1] if len(sys.argv) > 1 else "phase3_pairwise.pkl"
N_BOOTSTRAP = 5000
N_PERMUTATIONS = 5000
RANDOM_SEED = 0

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_pairwise():
    with open(PAIRWISE_PATH, "rb") as f:
        return pickle.load(f)


def group_means_from_labels(dist_matrix, correct_labels):
    """
    Given one problem's distance matrix and a correct/incorrect label per
    chain, return the (cc, ci, ii) mean distances for that problem (each
    may be None if that problem has no pairs of that type -- e.g. all
    chains correct means no ii pairs).
    """
    n = len(correct_labels)
    cc, ci, ii = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_matrix[i, j]
            if np.isnan(d):
                continue
            if correct_labels[i] and correct_labels[j]:
                cc.append(d)
            elif not correct_labels[i] and not correct_labels[j]:
                ii.append(d)
            else:
                ci.append(d)
    return (
        np.mean(cc) if cc else None,
        np.mean(ci) if ci else None,
        np.mean(ii) if ii else None,
    )


# ---------------------------------------------------------------------------
# 1. Cluster bootstrap CIs
# ---------------------------------------------------------------------------
def cluster_bootstrap(pairwise_store, group="cc"):
    """
    Resample problem IDs with replacement N_BOOTSTRAP times. For each
    resample, pool all pairs of the requested group ('cc'/'ci'/'ii') from
    the resampled problems (a problem drawn twice contributes its pairs
    twice) and compute the pooled mean. Returns array of bootstrap means.
    """
    problem_ids = list(pairwise_store.keys())
    idx = {"cc": 0, "ci": 1, "ii": 2}[group]

    # Precompute each problem's (cc, ci, ii) pair-lists once.
    per_problem_pairs = {}
    for pid, data in pairwise_store.items():
        dm, labels = data["dist_matrix"], data["correct"]
        n = len(labels)
        buckets = ([], [], [])
        for i in range(n):
            for j in range(i + 1, n):
                d = dm[i, j]
                if np.isnan(d):
                    continue
                if labels[i] and labels[j]:
                    buckets[0].append(d)
                elif not labels[i] and not labels[j]:
                    buckets[2].append(d)
                else:
                    buckets[1].append(d)
        per_problem_pairs[pid] = buckets

    boot_means = []
    for _ in range(N_BOOTSTRAP):
        sampled_pids = np.random.choice(problem_ids, size=len(problem_ids), replace=True)
        pooled = []
        for pid in sampled_pids:
            pooled.extend(per_problem_pairs[pid][idx])
        if pooled:
            boot_means.append(np.mean(pooled))
    return np.array(boot_means)


def report_ci(name, boot_means):
    if len(boot_means) == 0:
        print(f"{name:50s} no data")
        return
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    print(f"{name:50s} mean={boot_means.mean():.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  (n_boot={len(boot_means)})")


# ---------------------------------------------------------------------------
# 2. Within-problem permutation test
# ---------------------------------------------------------------------------
def permutation_test(pairwise_store, n_permutations=N_PERMUTATIONS):
    """
    Null hypothesis: correctness labels carry no information about which
    chains cluster together internally -- i.e. the observed
    (mean(ii) - mean(cc)) gap is no different from randomly relabeling
    which chains are "correct" within each problem.

    We shuffle labels WITHIN each problem (preserving each problem's count
    of correct/incorrect chains, and never mixing labels across problems),
    recompute the pooled cc/ii means each time, and build a null
    distribution of the gap statistic.
    """
    problem_ids = list(pairwise_store.keys())

    def pooled_cc_ii_gap(label_override=None):
        """label_override: dict pid -> shuffled label list, or None for real labels."""
        cc_all, ii_all = [], []
        for pid in problem_ids:
            data = pairwise_store[pid]
            labels = label_override[pid] if label_override else data["correct"]
            cc, _, ii = group_means_from_labels(data["dist_matrix"], labels)
            if cc is not None:
                cc_all.append(cc)
            if ii is not None:
                ii_all.append(ii)
        if not cc_all or not ii_all:
            return None
        # Gap statistic: incorrect-incorrect distance minus correct-correct
        # distance, pooled across problems (mean of per-problem means, so
        # each problem contributes equally regardless of pair count).
        return np.mean(ii_all) - np.mean(cc_all)

    observed_gap = pooled_cc_ii_gap()
    if observed_gap is None:
        print("Not enough data to compute observed cc/ii gap.")
        return

    null_gaps = []
    for _ in range(n_permutations):
        shuffled = {}
        for pid in problem_ids:
            labels = list(pairwise_store[pid]["correct"])
            random.shuffle(labels)  # shuffle WITHIN this problem only
            shuffled[pid] = labels
        gap = pooled_cc_ii_gap(shuffled)
        if gap is not None:
            null_gaps.append(gap)
    null_gaps = np.array(null_gaps)

    p_value = np.mean(null_gaps >= observed_gap)  # one-sided: real gap should be large & positive

    print(f"Observed gap  mean(ii) - mean(cc) = {observed_gap:.4f}")
    print(f"Null distribution (label-shuffled within problem): "
          f"mean={null_gaps.mean():.4f}  std={null_gaps.std():.4f}")
    print(f"One-sided p-value (P[null >= observed]) = {p_value:.4f}  "
          f"(n_permutations={len(null_gaps)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pairwise_store = load_pairwise()
    print(f"Loaded pairwise data from {PAIRWISE_PATH} -- {len(pairwise_store)} problems\n")

    print("=== Cluster bootstrap 95% CIs (resampled by problem) ===\n")
    for group, label in [("cc", "correct-correct"), ("ci", "correct-incorrect"), ("ii", "incorrect-incorrect")]:
        boot = cluster_bootstrap(pairwise_store, group=group)
        report_ci(f"same-problem, {label}", boot)

    print("\n=== Permutation test: do correct chains genuinely cluster tighter? ===\n")
    permutation_test(pairwise_store)


if __name__ == "__main__":
    main()