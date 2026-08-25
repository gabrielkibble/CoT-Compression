"""
Layer sweep: which layer gives the strongest correct/incorrect internal-
state separation?

LAYER=16 has been used since Phase 1 based on "roughly the middle"
reasoning, never validated. This script checks that empirically.

Key efficiency point: we do NOT need to re-generate any chains. Every
chain's tokens are already fixed (from Phase 1). We just need to re-run
each chain through the model as a single TEACHER-FORCED forward pass
(no sampling, no generation loop) with output_hidden_states=True, which
returns hidden states at EVERY layer in one call. So the expensive part
(one forward pass per chain) happens once regardless of how many
candidate layers we compare afterward.

Steps:
  1. For every chain in chunked_captures/, reconstruct its full
     prompt+completion token sequence and run ONE forward pass, capturing
     the residual at every layer for each chunk's last-token position
     (reusing the exact chunk boundaries already computed in Phase 2 --
     no need to re-chunk).
  2. For each candidate layer, apply the same anisotropy correction and
     same-problem/cross-problem, correct/incorrect DTW comparison as
     dtw_align_v2.py, and record the correct-vs-incorrect gap and its
     permutation-test p-value.
  3. Print a table: layer -> gap, p-value, so you can pick the layer with
     the strongest, most significant separation.

Output: layer_sweep_residuals.pkl (all-layer residuals per chunk, so you
        don't need to re-run the expensive forward-pass step if you want
        to analyze more layers later)
        layer_sweep_results.csv (the comparison table)
"""

import os
import glob
import pickle
import random
import itertools
import csv
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CHUNKED_DIR = "chunked_captures"
EMBED_MATRIX_PATH = "embed_matrix.npy"
RESIDUALS_CACHE = "layer_sweep_residuals.pkl"

# Which layers to compare. Llama-3.1-8B has 32 transformer blocks, so
# output_hidden_states gives indices 0 (embeddings) through 32. Sweeping
# every 4th layer as a reasonable first pass -- narrow this down once you
# see where the signal peaks.
CANDIDATE_LAYERS = [4, 8, 12, 16, 20, 24, 28]

N_CROSS_PROBLEM_PAIRS = 1000   # smaller than dtw_align_v2's 2000 since this
                                # runs once per candidate layer -- keeps
                                # total sweep runtime reasonable
N_PERMUTATIONS = 2000          # likewise reduced per-layer; the full
                                # 5000 from dtw_align_v2.py can be re-run
                                # standalone on whichever layer wins
RANDOM_SEED = 0

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def build_prompt(tokenizer, question: str) -> str:
    """Identical prompt format to every earlier phase, for consistency."""
    messages = [{
        "role": "user",
        "content": (
            f"{question}\n\n"
            "Solve this step by step. On the VERY LAST line of your "
            "response, write ONLY the final numeric answer in exactly "
            "this format (no dollar signs, no extra words):\n"
            "#### <number>"
        ),
    }]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Step 1: extract all-layer residuals via one forward pass per chain
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_all_layers(model, tokenizer, paths, records):
    """
    Extends `records` in place by extracting all-layer residuals for every
    chain in the given `paths` list (NOT a glob -- caller controls exactly
    which problems get processed, so already-cached problems can be
    skipped). Checkpoints the CUMULATIVE records list after each problem,
    so a crash only loses the current problem's chains, not the whole run
    (including anything already in the cache from a previous session).
    """
    for path in paths:
        data = torch.load(path, weights_only=False)
        pid = data["problem_id"]
        question = data["question"]
        prompt_text = build_prompt(tokenizer, question)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids

        for chain in data["chains"]:
            chunks = chain["chunks"]
            if len(chunks) < 2:
                continue
            completion_ids = chain["token_ids"].unsqueeze(0)  # (1, T)
            full_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(model.device)
            prompt_len = prompt_ids.shape[1]

            outputs = model(input_ids=full_ids, output_hidden_states=True)
            n_layers = len(outputs.hidden_states)

            chunk_residuals = []
            for c in chunks:
                abs_pos = prompt_len + c["token_end"]
                per_layer = torch.stack([
                    outputs.hidden_states[l][0, abs_pos, :].float().cpu()
                    for l in range(n_layers)
                ])
                chunk_residuals.append(per_layer)
            chunk_residuals = torch.stack(chunk_residuals)

            records.append({
                "problem_id": pid,
                "chain_idx": chain["chain_idx"],
                "correct": chain["correct"],
                "last_token_ids": chain["token_ids"][[c["token_end"] for c in chunks]],
                "chunk_residuals": chunk_residuals.numpy().astype(np.float32),
            })

        print(f"  [{pid}] done ({len(records)} total chains cached so far)")
        # Atomic write: save to a temp file first, then rename over the
        # real path. Writing directly to RESIDUALS_CACHE would truncate the
        # existing good file immediately on open, BEFORE the new data is
        # fully written -- if the process is interrupted mid-write, that
        # destroys the previously-good checkpoint instead of just losing
        # the current problem's chains.
        #
        # IMPORTANT: if RESIDUALS_CACHE is a symlink (e.g. pointing to
        # scratch storage to avoid a home-directory quota), we resolve to
        # the REAL target path for both the tmp file and the final
        # replace. Otherwise the tmp file would be created as a fresh file
        # in the current directory (ignoring the symlink), and os.replace
        # would overwrite the symlink itself with a real file -- silently
        # undoing the scratch-storage redirection and landing back in the
        # quota-limited home directory.
        real_target = os.path.realpath(RESIDUALS_CACHE) if os.path.exists(RESIDUALS_CACHE) else RESIDUALS_CACHE
        tmp_path = real_target + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(records, f)
        os.replace(tmp_path, real_target)
    return records


# ---------------------------------------------------------------------------
# Step 2: same correction + DTW machinery as dtw_align_v2.py, parameterized
# by which layer to slice out
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
    return D[n, m] / path_len


def analyze_layer(layer_idx, records, embed_matrix):
    """
    Slice out one layer's residuals from every record, fit anisotropy
    correction, run the same-problem correct/incorrect DTW comparison and
    within-problem permutation test. Returns a dict of summary stats.
    """
    layer_records = []
    for r in records:
        residuals = r["chunk_residuals"][:, layer_idx, :]  # (n_chunks, hidden)
        layer_records.append({
            "problem_id": r["problem_id"],
            "chain_idx": r["chain_idx"],
            "correct": r["correct"],
            "residuals": residuals,
            "last_token_ids": r["last_token_ids"].numpy(),
        })

    # Fit correction on this layer's full dataset (mean + projection first,
    # std computed after those two steps, matching dtw_align_v2.py).
    all_residuals = np.concatenate([r["residuals"] for r in layer_records], axis=0)
    mean = all_residuals.mean(axis=0)
    projected_all = np.concatenate([
        _correct(r["residuals"], r["last_token_ids"], mean, np.ones_like(mean), embed_matrix)
        for r in layer_records
    ], axis=0)
    std = projected_all.std(axis=0) + 1e-6

    for r in layer_records:
        r["corrected"] = _correct(r["residuals"], r["last_token_ids"], mean, std, embed_matrix)

    by_problem = {}
    for r in layer_records:
        by_problem.setdefault(r["problem_id"], []).append(r)

    same_cc, same_ci, same_ii = [], [], []
    pairwise_by_problem = {}  # for the permutation test
    for pid, chains in by_problem.items():
        n = len(chains)
        dist_matrix = np.full((n, n), np.nan)
        for (ia, a), (ib, b) in itertools.combinations(enumerate(chains), 2):
            d = dtw_align(a["corrected"], b["corrected"])
            dist_matrix[ia, ib] = dist_matrix[ib, ia] = d
            if a["correct"] and b["correct"]:
                same_cc.append(d)
            elif not a["correct"] and not b["correct"]:
                same_ii.append(d)
            else:
                same_ci.append(d)
        pairwise_by_problem[pid] = {
            "dist_matrix": dist_matrix,
            "correct": [c["correct"] for c in chains],
        }

    # Cross-problem baseline.
    cross = []
    problem_ids = list(by_problem.keys())
    attempts = 0
    while len(cross) < N_CROSS_PROBLEM_PAIRS and attempts < N_CROSS_PROBLEM_PAIRS * 5:
        attempts += 1
        pid_a, pid_b = random.sample(problem_ids, 2)
        a = random.choice(by_problem[pid_a])
        b = random.choice(by_problem[pid_b])
        cross.append(dtw_align(a["corrected"], b["corrected"]))

    # Within-problem permutation test on the cc/ii gap.
    def pooled_gap(label_override=None):
        cc_all, ii_all = [], []
        for pid, data in pairwise_by_problem.items():
            labels = label_override[pid] if label_override else data["correct"]
            dm, n = data["dist_matrix"], len(labels)
            cc, ii = [], []
            for i in range(n):
                for j in range(i + 1, n):
                    if np.isnan(dm[i, j]):
                        continue
                    if labels[i] and labels[j]:
                        cc.append(dm[i, j])
                    elif not labels[i] and not labels[j]:
                        ii.append(dm[i, j])
            if cc:
                cc_all.append(np.mean(cc))
            if ii:
                ii_all.append(np.mean(ii))
        if not cc_all or not ii_all:
            return None
        return np.mean(ii_all) - np.mean(cc_all)

    observed_gap = pooled_gap()
    null_gaps = []
    if observed_gap is not None:
        for _ in range(N_PERMUTATIONS):
            shuffled = {}
            for pid, data in pairwise_by_problem.items():
                labels = list(data["correct"])
                random.shuffle(labels)
                shuffled[pid] = labels
            g = pooled_gap(shuffled)
            if g is not None:
                null_gaps.append(g)
    null_gaps = np.array(null_gaps)
    p_value = np.mean(null_gaps >= observed_gap) if len(null_gaps) else None

    return {
        "layer": layer_idx,
        "mean_cc": np.mean(same_cc) if same_cc else None,
        "mean_ci": np.mean(same_ci) if same_ci else None,
        "mean_ii": np.mean(same_ii) if same_ii else None,
        "mean_cross": np.mean(cross) if cross else None,
        "gap_ii_minus_cc": observed_gap,
        "p_value": p_value,
    }


def main():
    embed_matrix = np.load(EMBED_MATRIX_PATH)

    # Load whatever's already cached (may be from a smaller-scale earlier
    # run) rather than assuming it's either complete or absent -- we EXTEND
    # it with only the problems not yet present.
    try:
        with open(RESIDUALS_CACHE, "rb") as f:
            records = pickle.load(f)
        print(f"Loaded {len(records)} cached chains from {RESIDUALS_CACHE}")
    except FileNotFoundError:
        records = []
        print(f"No existing {RESIDUALS_CACHE} found, starting fresh.")
    except (EOFError, pickle.UnpicklingError) as e:
        # Corrupted/truncated cache (e.g. from an interrupted write before
        # the atomic-write fix above was in place). Can't be partially
        # recovered -- treat as empty and re-extract everything.
        print(f"WARNING: {RESIDUALS_CACHE} exists but is corrupted ({e}). "
              f"Starting fresh -- all problems will be re-extracted.")
        records = []

    done_pids = {r["problem_id"] for r in records}
    all_paths = sorted(glob.glob(f"{CHUNKED_DIR}/*.pt"))
    remaining_paths = [
        p for p in all_paths
        if int(os.path.splitext(os.path.basename(p))[0]) not in done_pids
    ]

    if not remaining_paths:
        print("All problems in chunked_captures/ are already cached -- nothing new to extract.")
    else:
        print(f"{len(remaining_paths)} new problems to extract "
              f"({len(done_pids)} already cached, {len(all_paths)} total). Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        records = extract_all_layers(model, tokenizer, remaining_paths, records)
        print(f"Saved -> {RESIDUALS_CACHE} ({len(records)} total chains)")

    results = []
    for layer in CANDIDATE_LAYERS:
        print(f"\nAnalyzing layer {layer}...")
        stats = analyze_layer(layer, records, embed_matrix)
        results.append(stats)
        p_str = f"{stats['p_value']:.4f}" if stats["p_value"] is not None else "n/a"
        print(f"  cc={stats['mean_cc']:.4f}  ci={stats['mean_ci']:.4f}  "
              f"ii={stats['mean_ii']:.4f}  cross={stats['mean_cross']:.4f}  "
              f"gap={stats['gap_ii_minus_cc']:.4f}  p={p_str}")

    print("\n=== Layer sweep summary (sorted by gap, largest first) ===")
    results_sorted = sorted(results, key=lambda r: r["gap_ii_minus_cc"] or -np.inf, reverse=True)
    for r in results_sorted:
        p_str = f"{r['p_value']:.4f}" if r["p_value"] is not None else "n/a"
        print(f"layer {r['layer']:3d}: gap={r['gap_ii_minus_cc']:.4f}  p={p_str}  "
              f"(cc={r['mean_cc']:.4f}, ii={r['mean_ii']:.4f}, cross={r['mean_cross']:.4f})")

    with open("layer_sweep_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "mean_cc", "mean_ci", "mean_ii",
                                                "mean_cross", "gap_ii_minus_cc", "p_value"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print("\nSaved -> layer_sweep_results.csv")


if __name__ == "__main__":
    main()
