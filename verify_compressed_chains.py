"""
Verify DBSCAN-compressed chains empirically.

The risk flagged from the eps sweep: a compressed chain could coincidentally
retain the right final answer STRING (since it was part of the original
correct chain) while having dropped the actual derivation -- i.e. "right
answer, broken reasoning." Eyeballing text can't reliably catch this.

Real test: strip out any surviving "#### <answer>" line from the compressed
text (so the model can't just copy it), feed the REMAINING reasoning back
through the model as context, force a short greedy completion asking
directly for the final answer, and check if that's actually correct.

If the compressed reasoning is genuinely sufficient, the model should still
reach the right answer despite never having seen the original literal
answer line. If it doesn't, that's real evidence the compression broke
something load-bearing, not just a coincidence.

Requires: layer_sweep_residuals.pkl, embed_matrix.npy, consensus_routes.pkl,
          chunked_captures/ (for question + gold_answer)
"""

import re
import pickle
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
RESIDUALS_CACHE = "layer_sweep_residuals.pkl"
EMBED_MATRIX_PATH = "embed_matrix.npy"
ROUTES_PATH = "consensus_routes.pkl"
CHUNKED_DIR = "chunked_captures"

FINAL_LAYER_IDX = -1  # resolved to actual index at runtime
EPS_VALUES_TO_CHECK = [0.35, 0.45]
MIN_SAMPLES = 2
N_REPS_PER_CLUSTER = 1
N_PROBLEMS = 100  # scaled up from the 5-problem exploratory pass


# ---------------------------------------------------------------------------
# Reused from dbscan_compress.py
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
    dist = 1.0 - (a_norm @ a_norm.T)
    return np.clip(dist, 0.0, None)


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


# ---------------------------------------------------------------------------
# New: verification via forced completion
# ---------------------------------------------------------------------------
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
    """Remove any chunk that looks like the '#### <answer>' line, so the
    model can't just copy a surviving literal answer -- it has to actually
    use the remaining reasoning to reach one."""
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
    """Feed prompt + (answer-stripped) reasoning, force a short greedy
    completion asking directly for the final number."""
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
    resolved_layer = n_layers_available - 1  # final layer

    mean, std = fit_correction_for_layer(records, resolved_layer, embed_matrix)

    results = []
    for pid, route in list(routes.items())[:N_PROBLEMS]:
        medoid_chain_idx = route["chain_idx"]
        record = by_key.get((pid, medoid_chain_idx))
        if record is None:
            continue

        chunked_data = torch.load(f"{CHUNKED_DIR}/{pid}.pt", weights_only=False)
        question = chunked_data["question"]
        gold_answer = str(chunked_data["gold_answer"]).strip()
        prompt_text = build_prompt(tokenizer, question)

        texts = [c["text"] for c in route["chunks"]]
        residuals = record["chunk_residuals"][:, resolved_layer, :]
        last_token_ids = record["last_token_ids"].numpy()
        chunk_token_counts = [c["token_end"] - c["token_start"] + 1 for c in route["chunks"]]
        original_tokens = sum(chunk_token_counts)

        if len(texts) != len(residuals):
            print(f"[{pid}] SKIP -- chunk count mismatch")
            continue

        # Baseline: full original chain, sanity check (should always be correct).
        full_text_stripped = "\n".join(strip_answer_line(texts))
        baseline_pred = force_answer(model, tokenizer, prompt_text, full_text_stripped)
        baseline_correct = answers_match(baseline_pred, gold_answer)
        print(f"\n[{pid}] FULL CHAIN (answer-stripped): pred={baseline_pred} "
              f"gold={gold_answer} correct={baseline_correct}  "
              f"({len(texts)} chunks, {original_tokens} tokens)")
        results.append({"pid": pid, "eps": "full_chain", "n_kept": len(texts),
                         "n_original": len(texts), "tokens_kept": original_tokens,
                         "tokens_original": original_tokens, "correct": baseline_correct})

        for eps in EPS_VALUES_TO_CHECK:
            kept_idxs = compress_chain(residuals, mean, std, last_token_ids, embed_matrix, eps)
            kept_tokens = sum(chunk_token_counts[i] for i in kept_idxs)
            kept_texts = strip_answer_line([texts[i] for i in kept_idxs])
            compressed_text = "\n".join(kept_texts)

            pred = force_answer(model, tokenizer, prompt_text, compressed_text)
            correct = answers_match(pred, gold_answer)
            reduction = 1 - (kept_tokens / original_tokens)
            print(f"  eps={eps}: kept {len(kept_idxs)}/{len(texts)} chunks, "
                  f"{kept_tokens}/{original_tokens} tokens ({reduction:.0%} reduction) -> "
                  f"pred={pred} gold={gold_answer} correct={correct}")
            results.append({"pid": pid, "eps": eps, "n_kept": len(kept_idxs),
                             "n_original": len(texts), "tokens_kept": kept_tokens,
                             "tokens_original": original_tokens, "correct": correct})

    print("\n=== Summary ===")
    by_condition = {}
    for r in results:
        by_condition.setdefault(r["eps"], []).append(r)
    for eps, group in by_condition.items():
        corrects = [r["correct"] for r in group]
        acc = sum(corrects) / len(corrects)
        total_kept = sum(r["tokens_kept"] for r in group)
        total_original = sum(r["tokens_original"] for r in group)
        avg_reduction = 1 - (total_kept / total_original)
        print(f"  {eps}: {sum(corrects)}/{len(corrects)} correct ({acc:.0%})  "
              f"avg token reduction: {avg_reduction:.0%}  "
              f"(total {total_kept}/{total_original} tokens)")


if __name__ == "__main__":
    main()