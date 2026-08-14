"""
Phase 5: Guided short-chain generation.

For each problem, generate a NEW, hopefully-shorter chain of thought by
searching step by step:

  1. From the current partial chain, sample K candidate next "chunks"
     (roughly one reasoning step / one line each).
  2. Score each candidate using one of three modes:
       --mode prob     (Variant A) score = P(correct answer), via the
                        same logit-lens trick as Phase 1 -- does the
                        candidate move the model closer to the right
                        answer?
       --mode route    (Variant B) score = -distance to the Phase 4
                        consensus route's next state -- does the
                        candidate keep the model's internal trajectory
                        on the "shared route"? Ignores answer probability
                        entirely -- this isolates whether route-proximity
                        alone is doing any real work.
       --mode combined (Variant C) weighted sum of both.
  3. Keep the single best-scoring candidate (best-of-K greedy, not full
     beam search), append it to the running chain, advance the route
     pointer, and repeat until the chain reaches "#### <answer>" or a
     step/token budget runs out.

This directly tests the project's central claim: variant B tells us
whether staying on the internal route -- with NO explicit reward for
getting the answer right -- is enough to actually reach the right answer.

Requires (from earlier phases):
  - consensus_routes.pkl        (Phase 4 -- per-problem target route)
  - anisotropy_correction.npz   (mean, std)
  - embed_matrix.npy
  - chunked_captures/<pid>.pt   (for gold_answer, question text)

Output: guided_chains_<mode>.pkl
  pid -> {mode, text, tokens_used, n_steps, pred_answer, correct,
          route_len, chunks: [...]}

Usage:
  python3 guided_generate.py --mode prob
  python3 guided_generate.py --mode route
  python3 guided_generate.py --mode combined --alpha 0.5 --beta 0.5
"""

import argparse
import glob
import pickle
import re
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 16
CHUNKED_DIR = "chunked_captures"
ROUTES_PATH = "consensus_routes.pkl"
CORRECTION_PATH = "anisotropy_correction.npz"
EMBED_MATRIX_PATH = "embed_matrix.npy"

K_CANDIDATES = 5           # candidates sampled per step
MAX_CANDIDATE_TOKENS = 60  # cap on a single candidate chunk's length
TEMPERATURE = 0.9
TOP_P = 0.95
STEP_SLACK = 5             # allow this many extra steps beyond route length
                            # in case the search needs a bit more room
MAX_TOKEN_MULTIPLIER = 2.0 # default hard stop, overridable via --max_token_frac:
                            # don't exceed this x the route's own token
                            # length, regardless of step count


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def load_everything():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    with open(ROUTES_PATH, "rb") as f:
        routes = pickle.load(f)
    correction = np.load(CORRECTION_PATH)
    embed_matrix = np.load(EMBED_MATRIX_PATH)

    return tokenizer, model, routes, correction["mean"], correction["std"], embed_matrix


def build_prompt(tokenizer, question: str) -> str:
    """Same prompt format as Phase 1, for consistency."""
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


def answer_token_id(tokenizer, answer_str: str):
    """Same fix as Phase 1: skip whitespace-only leading tokens."""
    ids = tokenizer.encode(" " + answer_str, add_special_tokens=False)
    for tid in ids:
        if tokenizer.decode([tid]).strip() != "":
            return tid
    return None


def extract_answer(text: str):
    match = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    if match:
        return match.group(1).replace(",", "")
    fallback = re.findall(r"(-?[\d,]+\.?\d*)", text)
    return fallback[-1].replace(",", "") if fallback else None


def answers_match(pred, gold: str) -> bool:
    """Same numeric-comparison fix as capture_chains.py -- see there for why."""
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred.strip() == gold.strip()


# ---------------------------------------------------------------------------
# Candidate generation + truncation to one chunk
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_candidate(model, tokenizer, prompt_ids: torch.Tensor):
    """
    Sample ONE candidate continuation of up to MAX_CANDIDATE_TOKENS new
    tokens, capturing the LAYER hidden state at every generated token
    (same mechanism as Phase 1's sample_one_chain).
    """
    outputs = model.generate(
        input_ids=prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        max_new_tokens=MAX_CANDIDATE_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        return_dict_in_generate=True,
        output_hidden_states=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    residuals = torch.stack([
        step_hs[LAYER][0, -1, :].float().cpu() for step_hs in outputs.hidden_states
    ])
    gen_ids = outputs.sequences[0, prompt_ids.shape[1]:].cpu()
    return gen_ids, residuals


def truncate_to_one_chunk(tokenizer, gen_ids: torch.Tensor, residuals: torch.Tensor):
    """
    Cut a raw candidate generation down to a single "chunk": up to and
    including the first newline, OR the whole thing if a '####' final
    answer appears (don't cut the answer line short), OR the whole
    candidate if no boundary was found within the token budget.

    Returns: (chunk_text, chunk_token_ids, chunk_residuals, is_final)
    is_final=True means this chunk contains the '#### <answer>' line and
    the search should stop after accepting it.
    """
    full_text = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)

    if "####" in full_text:
        cut_at = full_text.find("####")
        line_end = full_text.find("\n", cut_at)
        end_char = line_end if line_end != -1 else len(full_text)
        chunk_text = full_text[:end_char].strip()
        is_final = True
    else:
        newline_pos = full_text.find("\n")
        if newline_pos == -1:
            chunk_text = full_text.strip()
        else:
            chunk_text = full_text[:newline_pos].strip()
        is_final = False

    if not chunk_text:
        chunk_text = full_text.strip()

    # Map chunk_text back to a token count via cumulative decode length,
    # same technique as Phase 2's build_char_offsets.
    n_tokens = len(gen_ids)
    for i in range(1, len(gen_ids) + 1):
        cursor_text = tokenizer.decode(gen_ids[:i].tolist(), skip_special_tokens=True)
        if len(cursor_text) >= len(chunk_text):
            n_tokens = i
            break

    chunk_token_ids = gen_ids[:n_tokens]
    chunk_residuals = residuals[:n_tokens]
    return chunk_text, chunk_token_ids, chunk_residuals, is_final


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def correct_residual(residual, token_id, mean, std, embed_matrix):
    """Same anisotropy correction as earlier phases, single-vector version."""
    centered = residual - mean
    tok_embed = embed_matrix[token_id].astype(np.float32)
    tok_dir = tok_embed / (np.linalg.norm(tok_embed) + 1e-8)
    proj = np.dot(centered, tok_dir)
    projected = centered - proj * tok_dir
    return projected / std


def cosine_distance(a, b):
    a_n = a / (np.linalg.norm(a) + 1e-8)
    b_n = b / (np.linalg.norm(b) + 1e-8)
    return 1.0 - float(np.dot(a_n, b_n))


def score_candidate(mode, alpha, beta, chunk_residuals, chunk_token_ids,
                     final_norm, lm_head, ans_tok_id, mean, std, embed_matrix,
                     route_target_residual):
    """
    chunk_residuals: (n, hidden) raw residuals at LAYER for this candidate
    chunk_token_ids: (n,) token ids for this candidate
    route_target_residual: (hidden,) corrected consensus state for the
                            CURRENT route pointer, or None if route is
                            exhausted (route-mode score becomes 0 then)
    Returns a scalar score (higher = better).
    """
    last_residual = chunk_residuals[-1].numpy()
    last_token_id = chunk_token_ids[-1].item()

    score_prob = 0.0
    if mode in ("prob", "combined"):
        with torch.no_grad():
            normed = final_norm(
                chunk_residuals[-1].to(final_norm.weight.device, dtype=final_norm.weight.dtype)
            )
            logits = lm_head(normed)
            probs = torch.softmax(logits.float(), dim=-1)
            score_prob = probs[ans_tok_id].item()

    score_route = 0.0
    if mode in ("route", "combined") and route_target_residual is not None:
        corrected = correct_residual(last_residual, last_token_id, mean, std, embed_matrix)
        score_route = -cosine_distance(corrected, route_target_residual)

    if mode == "prob":
        return score_prob
    if mode == "route":
        return score_route
    return alpha * score_prob + beta * score_route  # combined


# ---------------------------------------------------------------------------
# Main search loop, one problem
# ---------------------------------------------------------------------------
@torch.no_grad()
def guided_search(model, tokenizer, question, gold_answer, route, mode, alpha, beta,
                   final_norm, lm_head, mean, std, embed_matrix, max_token_frac):
    ans_tok_id = answer_token_id(tokenizer, gold_answer)
    if ans_tok_id is None:
        return None

    prompt_text = build_prompt(tokenizer, question)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

    route_chunks = route["chunks"]
    max_steps = len(route_chunks) + STEP_SLACK
    max_tokens = int(route["total_tokens"] * max_token_frac)

    accepted_text = ""
    accepted_chunks = []
    route_ptr = 0
    total_new_tokens = 0
    is_final = False

    for step in range(max_steps):
        if total_new_tokens >= max_tokens:
            break

        current_text = prompt_text + accepted_text
        current_ids = tokenizer(
            current_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)

        route_target = None
        if route_ptr < len(route_chunks):
            route_target = route_chunks[route_ptr]["residual_corrected"]

        best_score, best_chunk = -float("inf"), None
        for _ in range(K_CANDIDATES):
            gen_ids, residuals = sample_candidate(model, tokenizer, current_ids)
            if len(gen_ids) == 0:
                continue
            chunk_text, chunk_ids, chunk_res, cand_final = truncate_to_one_chunk(
                tokenizer, gen_ids, residuals
            )
            score = score_candidate(
                mode, alpha, beta, chunk_res, chunk_ids,
                final_norm, lm_head, ans_tok_id, mean, std, embed_matrix, route_target,
            )
            if score > best_score:
                best_score = score
                best_chunk = (chunk_text, chunk_ids, chunk_res, cand_final)

        if best_chunk is None:
            break

        chunk_text, chunk_ids, chunk_res, cand_final = best_chunk
        accepted_text += ("\n" if accepted_text else "") + chunk_text
        accepted_chunks.append({"text": chunk_text, "n_tokens": len(chunk_ids), "score": best_score})
        total_new_tokens += len(chunk_ids)
        route_ptr = min(route_ptr + 1, len(route_chunks) - 1)

        if cand_final:
            is_final = True
            break

    forced_completion = False
    if not is_final:
        # The search loop ended (budget/step cap) WITHOUT the model ever
        # writing "#### <answer>". Rather than let extract_answer's
        # fallback grab some arbitrary intermediate number from the
        # truncated reasoning (which would basically guarantee "wrong"
        # regardless of whether the reasoning itself was on track), force
        # one short, unsampled (greedy) completion asking directly for
        # the final number given whatever reasoning was produced so far.
        # This mirrors how a real short-chain student would be used --
        # always prompted for a final answer, whatever the token budget.
        forced_prompt = prompt_text + accepted_text + "\n#### "
        forced_ids = tokenizer(
            forced_prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)
        forced_out = model.generate(
            input_ids=forced_ids,
            attention_mask=torch.ones_like(forced_ids),
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        forced_text = tokenizer.decode(
            forced_out[0, forced_ids.shape[1]:], skip_special_tokens=True
        )
        accepted_text += "\n#### " + forced_text
        forced_tokens = forced_out.shape[1] - forced_ids.shape[1]
        total_new_tokens += forced_tokens
        forced_completion = True

    if forced_completion:
        # Extract from the FORCED text only -- if the model didn't
        # actually produce a clean number right after being asked
        # directly, that's a real "it didn't know," not license to reach
        # back into the abandoned, truncated reasoning for an unrelated
        # intermediate number.
        pred_answer = extract_answer(forced_text)
    else:
        pred_answer = extract_answer(accepted_text)
    correct = answers_match(pred_answer, gold_answer)

    return {
        "text": accepted_text,
        "tokens_used": total_new_tokens,
        "n_steps": len(accepted_chunks),
        "pred_answer": pred_answer,
        "correct": correct,
        "reached_answer_line": is_final,
        "forced_completion": forced_completion,
        "route_len": len(route_chunks),
        "chunks": accepted_chunks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prob", "route", "combined"], required=True)
    parser.add_argument("--alpha", type=float, default=0.5, help="weight on answer-probability (combined mode)")
    parser.add_argument("--beta", type=float, default=0.5, help="weight on route-proximity (combined mode)")
    parser.add_argument("--max_token_frac", type=float, default=MAX_TOKEN_MULTIPLIER,
                         help="hard token budget as a fraction of the route's own token "
                              "length (e.g. 0.5 = must finish within half the route's "
                              "length). Default 2.0 matches the original loose-budget runs.")
    args = parser.parse_args()

    tokenizer, model, routes, mean, std, embed_matrix = load_everything()
    final_norm = model.model.norm
    lm_head = model.lm_head

    results = {}
    for path in sorted(glob.glob(f"{CHUNKED_DIR}/*.pt")):
        data = torch.load(path, weights_only=False)
        pid = data["problem_id"]
        if pid not in routes:
            print(f"[{pid}] no consensus route found, skipping")
            continue

        result = guided_search(
            model, tokenizer, data["question"], data["gold_answer"], routes[pid],
            args.mode, args.alpha, args.beta, final_norm, lm_head, mean, std, embed_matrix,
            args.max_token_frac,
        )
        if result is None:
            print(f"[{pid}] SKIPPED -- couldn't tokenize gold answer")
            continue

        results[pid] = result
        status = "OK " if result["correct"] else "ERR"
        print(f"[{pid}] [{status}] pred={result['pred_answer']} gold={data['gold_answer']}  "
              f"tokens={result['tokens_used']} (route was {routes[pid]['total_tokens']})  "
              f"steps={result['n_steps']}/{result['route_len']}")

    out_path = f"guided_chains_{args.mode}_frac{args.max_token_frac:.2f}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    n_correct = sum(r["correct"] for r in results.values())
    avg_tokens = np.mean([r["tokens_used"] for r in results.values()]) if results else 0
    print(f"\n{args.mode}: {n_correct}/{len(results)} correct, avg {avg_tokens:.0f} tokens")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()