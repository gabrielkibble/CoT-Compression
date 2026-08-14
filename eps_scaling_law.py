"""
eps scaling law: how does compression (token reduction) and accuracy vary
as a continuous function of DBSCAN's eps parameter?

Earlier checks only compared 2-3 discrete eps values. This sweeps a finer
grid (0.10 to 0.70 in steps of 0.05) across N_PROBLEMS problems, using the
SAME rigorous verification as verify_compressed_chains.py (answer line
stripped, forced completion, real correctness check) -- not just chunk
counts.

Output:
  eps_scaling_law.csv  (eps, avg_token_reduction, accuracy, n_correct, n_total)
  eps_scaling_law.png  (two curves: reduction % and accuracy % vs eps)

This turns the two-point comparison you already have into a real curve --
useful for picking an eps that hits a target token budget, or for finding
the "knee" where accuracy starts dropping off a cliff.
"""

import re
import csv
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
RESIDUALS_CACHE = "layer_sweep_residuals.pkl"
EMBED_MATRIX_PATH = "embed_matrix.npy"
ROUTES_PATH = "consensus_routes.pkl"
CHUNKED_DIR = "chunked_captures"

EPS_GRID = [round(x, 2) for x in np.arange(0.10, 0.71, 0.05)]
MIN_SAMPLES = 2
N_REPS_PER_CLUSTER = 1
N_PROBLEMS = 100


# ---------------------------------------------------------------------------
# Reused machinery (same as dbscan_compress.py / verify_compressed_chains.py)
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


def compress_chain(residuals, mean, std, last_token_ids, embed_matrix, eps):
    corrected = _correct(residuals, last_token_ids, mean, std, embed_matrix)
    dist_matrix = cosine_distance_matrix(corrected)
    db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, metric="precomputed")
    labels = db.fit_predict(dist_matrix)

    kept = set()
    for label in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == label]
        if label == -1:
            kept.update(idxs)
            continue
        cluster_vecs = corrected[idxs]
        centroid = cluster_vecs.mean(axis=0)
        dists = np.linalg.norm(cluster_vecs - centroid, axis=1)
        order = np.argsort(dists)
        kept.update(idxs[i] for i in order[:N_REPS_PER_CLUSTER])
    return sorted(kept)


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
    resolved_layer = n_layers_available - 1

    mean, std = fit_correction_for_layer(records, resolved_layer, embed_matrix)

    # Precompute per-problem fixed data ONCE (prompt, gold, texts, residuals,
    # token counts) -- only the compression + forced-completion step repeats
    # per eps value below.
    problems = []
    for pid, route in list(routes.items())[:N_PROBLEMS]:
        medoid_chain_idx = route["chain_idx"]
        record = by_key.get((pid, medoid_chain_idx))
        if record is None:
            continue
        chunked_data = torch.load(f"{CHUNKED_DIR}/{pid}.pt", weights_only=False)
        texts = [c["text"] for c in route["chunks"]]
        residuals = record["chunk_residuals"][:, resolved_layer, :]
        if len(texts) != len(residuals):
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
    print(f"Prepared {len(problems)} problems. Sweeping {len(EPS_GRID)} eps values...\n")

    rows = []
    for eps in EPS_GRID:
        correct_count = 0
        total_kept, total_original = 0, 0
        for p in problems:
            kept_idxs = compress_chain(
                p["residuals"], mean, std, p["last_token_ids"], embed_matrix, eps
            )
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
        reduction = 1 - (total_kept / total_original)
        rows.append({
            "eps": eps, "accuracy": accuracy, "n_correct": correct_count,
            "n_total": len(problems), "token_reduction": reduction,
            "tokens_kept": total_kept, "tokens_original": total_original,
        })
        print(f"eps={eps:.2f}: {correct_count}/{len(problems)} correct ({accuracy:.0%})  "
              f"token reduction={reduction:.0%}  ({total_kept}/{total_original} tokens)")

    with open("eps_scaling_law.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["eps", "accuracy", "n_correct", "n_total",
                                                "token_reduction", "tokens_kept", "tokens_original"])
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved -> eps_scaling_law.csv")

    # ---- Plot ----
    fig, ax1 = plt.subplots(figsize=(8, 5))
    eps_vals = [r["eps"] for r in rows]
    ax1.plot(eps_vals, [r["token_reduction"] * 100 for r in rows], "o-", color="tab:blue", label="Token reduction (%)")
    ax1.set_xlabel("DBSCAN eps")
    ax1.set_ylabel("Token reduction (%)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(eps_vals, [r["accuracy"] * 100 for r in rows], "s-", color="tab:red", label="Accuracy (%)")
    ax2.set_ylabel("Accuracy (%)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(-5, 105)

    fig.suptitle("DBSCAN compression: eps vs. token reduction and accuracy")
    fig.tight_layout()
    fig.savefig("eps_scaling_law.png", dpi=150)
    print("Saved -> eps_scaling_law.png")


if __name__ == "__main__":
    main()