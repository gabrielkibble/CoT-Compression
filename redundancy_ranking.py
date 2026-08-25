"""
Fixed-budget compression via nearest-neighbor redundancy ranking.

Motivation: "Do LLMs Encode Functional Importance of Reasoning Tokens?"
(Singh & Hakkani-Tur, ACL 2026) evaluates compression at a chosen target
KEEP FRACTION rho -- you specify upfront "keep 40% of tokens," and their
greedy pruning (iteratively delete whichever token least hurts model
likelihood) finds the best set under that exact budget. Our DBSCAN
approach (dbscan_compress.py) is threshold-first instead: eps controls a
density cutoff, and the resulting compression ratio EMERGES from the data
rather than being chosen directly -- which makes matched-budget comparison
against their method impossible as originally built.

This script fixes that by replacing DBSCAN's binary keep/drop-per-cluster
with a CONTINUOUS ranking: for each chunk, compute its cosine distance to
its single nearest neighbor within the same chain (same anisotropy-
corrected final-layer space as before). A chunk very close to some other
chunk is redundant -- safe to drop first. A chunk far from everything
(e.g. the final answer line, or a genuinely unique computation) is
important -- kept until very aggressive pruning.

This ranking lets us hit ANY target keep fraction directly: walk down the
ranking from most-unique to most-redundant, keeping chunks (by token
budget) until the target fraction is reached. That's now directly
comparable to the paper's rho-indexed evaluation protocol, while still
being fundamentally a similarity/redundancy-based method (not a
likelihood-greedy search) -- the real methodological difference from
their approach, not just an implementation detail.

NOTE on granularity: this operates at CHUNK level (roughly one reasoning
step), not TOKEN level like the paper. Achieved keep fraction per problem
will not exactly match the target rho -- both target and actual are
reported below.

Requires: layer_sweep_residuals.pkl, embed_matrix.npy, consensus_routes.pkl,
          chunked_captures/
"""

import re
import csv
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
RESIDUALS_CACHE = "layer_sweep_residuals.pkl"
EMBED_MATRIX_PATH = "embed_matrix.npy"
ROUTES_PATH = "consensus_routes.pkl"
CHUNKED_DIR = "chunked_captures"

RHO_GRID = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]  # target keep fractions,
                                                       # matching the paper's
                                                       # rho-indexed evaluation
N_PROBLEMS = 100


# ---------------------------------------------------------------------------
# Shared machinery (same as dbscan_compress.py / eps_scaling_law.py)
# ---------------------------------------------------------------------------
def _correct(residuals, last_token_ids, mean, std, embed_matrix):
    centered = residuals - mean
    tok_embeds = embed_matrix[last_token_ids].astype(np.float32)
    tok_dirs = tok_embeds / (np.linalg.norm(tok_embeds, axis=1, keepdims=True) + 1e-8)
    proj_scale = np.sum(centered * tok_dirs, axis=1, keepdims=True)
    projected = centered - proj_scale * tok_dirs
    return projected / std


def cosine_distance_matrix(a):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    return np.clip(1.0 - (a_norm @ a_norm.T), 0.0, None)


def fit_correction_for_layer(records, layer_idx, embed_matrix):
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


def build_prompt(tokenizer, question: str) -> str:
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


def strip_answer_line(texts):
    return [t for t in texts if not re.match(r"^\s*####", t)]


def extract_answer(text: str):
    match = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    if match:
        return match.group(1).replace(",", "")
    fallback = re.findall(r"(-?[\d,]+\.?\d*)", text)
    return fallback[-1].replace(",", "") if fallback else None


def answers_match(pred, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred.strip() == gold.strip()


@torch.no_grad()
def force_answer(model, tokenizer, prompt_text, reasoning_text):
    full_text = prompt_text + reasoning_text + "\n#### "
    input_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=12,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    completion = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
    return extract_answer(completion)


# ---------------------------------------------------------------------------
# New: nearest-neighbor redundancy ranking
# ---------------------------------------------------------------------------
def redundancy_rank_order(residuals, last_token_ids, mean, std, embed_matrix):
    """
    Returns chunk indices ordered from MOST unique (keep longest) to MOST
    redundant (drop first) -- i.e. descending nearest-neighbor distance.
    """
    corrected = _correct(residuals, last_token_ids, mean, std, embed_matrix)
    dist_matrix = cosine_distance_matrix(corrected)
    np.fill_diagonal(dist_matrix, np.inf)  # exclude self when finding nearest neighbor
    nn_dist = dist_matrix.min(axis=1)
    return list(np.argsort(-nn_dist))  # descending: largest NN-distance (most unique) first


def select_for_budget(rank_order, chunk_token_counts, target_fraction):
    """
    Walk down the rank order (most unique first), accumulating chunks
    until the token budget (target_fraction * total tokens) is reached.
    Returns kept indices in ORIGINAL chronological order (for readable
    output), not rank order.
    """
    total_tokens = sum(chunk_token_counts)
    budget = target_fraction * total_tokens
    kept = set()
    used = 0
    for idx in rank_order:
        if used >= budget and len(kept) > 0:
            break
        kept.add(idx)
        used += chunk_token_counts[idx]
    return sorted(kept)


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    with open(RESIDUALS_CACHE, "rb") as f:
        records = pickle.load(f)
    with open(ROUTES_PATH, "rb") as f:
        routes = pickle.load(f)
    embed_matrix = np.load(EMBED_MATRIX_PATH)

    by_key = {(r["problem_id"], r["chain_idx"]): r for r in records}
    n_layers_available = records[0]["chunk_residuals"].shape[1]
    resolved_layer = n_layers_available - 1  # final layer, matching dbscan_compress.py's finding

    mean, std = fit_correction_for_layer(records, resolved_layer, embed_matrix)

    problems = []
    for pid, route in list(routes.items())[:N_PROBLEMS]:
        medoid_chain_idx = route["chain_idx"]
        record = by_key.get((pid, medoid_chain_idx))
        if record is None:
            continue
        chunked_data = torch.load(f"{CHUNKED_DIR}/{pid}.pt", weights_only=False)
        texts = [c["text"] for c in route["chunks"]]
        residuals = record["chunk_residuals"][:, resolved_layer, :]
        if len(texts) != len(residuals) or len(texts) < 2:
            continue
        problems.append({
            "pid": pid,
            "question": chunked_data["question"],
            "gold_answer": str(chunked_data["gold_answer"]).strip(),
            "prompt_text": build_prompt(tokenizer, chunked_data["question"]),
            "texts": texts,
            "residuals": residuals,
            "last_token_ids": record["last_token_ids"].numpy(),
            "chunk_token_counts": [c["token_end"] - c["token_start"] + 1 for c in route["chunks"]],
        })
    print(f"Prepared {len(problems)} problems. Sweeping {len(RHO_GRID)} target keep fractions...\n")

    # Precompute each problem's ranking ONCE -- doesn't depend on rho.
    for p in problems:
        p["rank_order"] = redundancy_rank_order(
            p["residuals"], p["last_token_ids"], mean, std, embed_matrix
        )

    rows = []
    for rho in RHO_GRID:
        correct_count = 0
        total_kept, total_original = 0, 0
        for p in problems:
            kept_idxs = select_for_budget(p["rank_order"], p["chunk_token_counts"], rho)
            kept_tokens = sum(p["chunk_token_counts"][i] for i in kept_idxs)
            original_tokens = sum(p["chunk_token_counts"])
            kept_texts = strip_answer_line([p["texts"][i] for i in kept_idxs])
            compressed_text = "\n".join(kept_texts)

            pred = force_answer(model, tokenizer, p["prompt_text"], compressed_text)
            if answers_match(pred, p["gold_answer"]):
                correct_count += 1
            total_kept += kept_tokens
            total_original += original_tokens

        accuracy = correct_count / len(problems)
        achieved_keep_fraction = total_kept / total_original
        rows.append({
            "target_rho": rho, "achieved_keep_fraction": achieved_keep_fraction,
            "accuracy": accuracy, "n_correct": correct_count, "n_total": len(problems),
            "tokens_kept": total_kept, "tokens_original": total_original,
        })
        print(f"target_rho={rho:.2f} (achieved={achieved_keep_fraction:.0%} kept): "
              f"{correct_count}/{len(problems)} correct ({accuracy:.0%})")

    with open("redundancy_ranking.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target_rho", "achieved_keep_fraction", "accuracy",
                                                "n_correct", "n_total", "tokens_kept", "tokens_original"])
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved -> redundancy_ranking.csv")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x_vals = [r["achieved_keep_fraction"] * 100 for r in rows]
    ax1.plot(x_vals, [r["accuracy"] * 100 for r in rows], "s-", color="tab:red")
    ax1.set_xlabel("Achieved token keep fraction (%)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(-5, 105)
    fig.suptitle("Redundancy-ranking compression: keep fraction vs. accuracy")
    fig.tight_layout()
    fig.savefig("redundancy_ranking.png", dpi=150)
    print("Saved -> redundancy_ranking.png")


if __name__ == "__main__":
    main()