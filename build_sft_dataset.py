"""
Phase 6, step 1: build SFT training data.

Two training sets, from the SAME train-split problems:
  - train_full.jsonl        -- prompt + FULL medoid chain + answer
  - train_compressed.jsonl  -- prompt + redundancy-ranking-COMPRESSED chain
                                (target keep fraction TARGET_RHO) + answer

A held-out TEST_FRACTION of problems is set aside and NEVER used to build
training data -- test_problems.json is for evaluate_student.py only. This
is the train/test discipline that makes "does the student generalize"
a real question rather than a memorization check.

Unlike the verification scripts (verify_compressed_chains.py etc.), the
answer line is NOT stripped here -- the student needs to learn to
actually output an answer, that's the whole point of training data.

Requires: layer_sweep_residuals.pkl, embed_matrix.npy, consensus_routes.pkl,
          chunked_captures/
"""

import re
import json
import pickle
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
RESIDUALS_CACHE = "layer_sweep_residuals.pkl"
EMBED_MATRIX_PATH = "embed_matrix.npy"
ROUTES_PATH = "consensus_routes.pkl"
CHUNKED_DIR = "chunked_captures"

TARGET_RHO = 0.5       # keep fraction for the compressed training condition --
                        # FIXED per team agreement (Harish/Dilyan/Varniethan/
                        # Siva all use this same value for comparability)
TEST_FRACTION = 0.25   # fraction of problems held out for evaluation only.
                        # IMPORTANT: the test set is NOT filtered based on
                        # whether the source (teacher) model can still answer
                        # after compression -- we're testing whether SFT
                        # generalizes, not re-testing the teacher's own
                        # self-consistency. Every held-out problem stays in
                        # the test set regardless of compression difficulty.
ANS_GUARD_TAU = 0.3    # NOTE: same name/value as the team's token-level ANS
                        # guard (see consensus_v4_ans_guard_tau1.py), adapted
                        # here to chunk-level leave-one-out scoring. A chunk
                        # is "unsafe" to drop if removing it ALONE would drop
                        # log P(gold answer | reasoning) by more than this.
RANDOM_SEED = 0

random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Shared machinery (same as redundancy_ranking.py)
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


def redundancy_rank_order(residuals, last_token_ids, mean, std, embed_matrix):
    corrected = _correct(residuals, last_token_ids, mean, std, embed_matrix)
    dist_matrix = cosine_distance_matrix(corrected)
    np.fill_diagonal(dist_matrix, np.inf)
    nn_dist = dist_matrix.min(axis=1)
    return list(np.argsort(-nn_dist))


def select_for_budget(rank_order, chunk_token_counts, target_fraction):
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


NUMBER_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_numbers(text: str):
    """Numbers as floats, with $ and , formatting stripped."""
    nums = []
    for m in NUMBER_RE.findall(text):
        cleaned = m.replace("$", "").replace(",", "")
        try:
            nums.append(float(cleaned))
        except ValueError:
            continue
    return nums


def chunk_produces_consumes(text: str):
    """
    For one chunk's text: (produced_numbers, consumed_numbers).
    Produced = numbers after the LAST '=' (the chunk's final result).
    Consumed = numbers before the FIRST '=' (its operands). A chunk with
    no '=' (pure narrative) produces/consumes nothing.
    """
    if "=" not in text:
        return [], []
    first_eq = text.index("=")
    last_eq = text.rindex("=")
    consumed = extract_numbers(text[:first_eq])
    produced = extract_numbers(text[last_eq + 1:])
    return produced, consumed


def build_dependency_graph(texts, question_numbers):
    """
    Returns (deps, produced_by):
      deps: {chunk_idx: set(chunk_idx it depends on)}. A chunk depends on
      the most recent EARLIER chunk that produced a number it consumes,
      unless that number was already given in the question.
      produced_by: {number: chunk_idx that most recently produced it} --
      returned separately so the answer line (which has no '=' of its own)
      can be traced back to its actual source chunk. See
      select_for_budget_dependency_aware for why this matters.
    """
    deps = {i: set() for i in range(len(texts))}
    produced_by = {}
    for i, text in enumerate(texts):
        produced, consumed = chunk_produces_consumes(text)
        for num in consumed:
            if num in question_numbers:
                continue
            producer = produced_by.get(num)
            if producer is not None and producer != i:
                deps[i].add(producer)
        for num in produced:
            produced_by[num] = i
    return deps, produced_by


def expand_with_dependencies(kept_idxs, deps):
    """
    Transitive closure: if a chunk is kept, every chunk it depends on
    (recursively) must be kept too. Guarantees no kept chunk ever
    references a number whose derivation was dropped.
    """
    kept = set(kept_idxs)
    frontier = list(kept)
    while frontier:
        idx = frontier.pop()
        for dep in deps.get(idx, ()):
            if dep not in kept:
                kept.add(dep)
                frontier.append(dep)
    return sorted(kept)


def select_for_budget_dependency_aware(rank_order, chunk_token_counts, target_fraction, texts, question_numbers, force_keep_idxs=None):
    """
    Same greedy budget walk as select_for_budget, followed by dependency-
    closure expansion. ACHIEVED keep fraction can end up above
    target_fraction when dependencies pull extra chunks back in -- that's
    expected and correct, it means real derivation steps are preserved
    rather than dropped.

    force_keep_idxs: indices ALWAYS included before the closure runs (e.g.
    the answer chunk).

    IMPORTANT: the "#### N" answer line itself usually contains no '='
    sign, so the equation-based dependency graph above never links it back
    to whichever chunk actually COMPUTED N -- keeping the answer chunk's
    literal text is not the same as keeping its derivation. We explicitly
    trace the answer number to its producer chunk via produced_by and
    force that chunk (and everything IT depends on) to be kept too. This
    is what actually fixes the "answer never derived" failure mode, not
    just the "ungrounded operand" one.
    """
    deps, produced_by = build_dependency_graph(texts, question_numbers)
    initial_kept = set(select_for_budget(rank_order, chunk_token_counts, target_fraction))
    if force_keep_idxs:
        initial_kept.update(force_keep_idxs)

    for idx in list(initial_kept):
        text = texts[idx]
        if re.match(r"^\s*####", text):
            answer_nums = extract_numbers(text)
            if answer_nums:
                producer = produced_by.get(answer_nums[0])
                if producer is not None:
                    initial_kept.add(producer)

    return expand_with_dependencies(sorted(initial_kept), deps)


def user_prompt(question: str) -> str:
    """The user-turn content only -- chat template applied at train/eval time
    by whichever script needs it, so this stays framework-agnostic here."""
    return (
        f"{question}\n\n"
        "Solve this step by step. On the VERY LAST line of your "
        "response, write ONLY the final numeric answer in exactly "
        "this format (no dollar signs, no extra words):\n"
        "#### <number>"
    )


def build_prompt_for_scoring(tokenizer, question: str) -> str:
    """Full chat-templated prompt (matches the format the teacher model was
    always queried with elsewhere in this pipeline)."""
    messages = [{"role": "user", "content": user_prompt(question)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_answer_token_ids(tokenizer, gold_answer: str):
    """Full token id sequence for the gold answer (space-prefixed), skipping
    a leading whitespace-only token if the tokenizer produces one (same fix
    applied throughout this pipeline -- see capture_chains.py)."""
    ids = tokenizer.encode(" " + gold_answer, add_special_tokens=False)
    while ids and tokenizer.decode([ids[0]]).strip() == "":
        ids = ids[1:]
    return ids


@torch.no_grad()
def logp_of_answer(model, tokenizer, prefix_text: str, answer_token_ids) -> float:
    """
    Teacher-forced summed log P(answer_token_ids | prefix_text), via a
    SINGLE forward pass (no generation loop needed, since we already know
    the correct answer tokens -- much cheaper than the greedy-completion
    verification used elsewhere in this pipeline).
    """
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    answer_ids = torch.tensor([answer_token_ids], device=model.device)
    full_ids = torch.cat([prefix_ids, answer_ids], dim=1)

    logits = model(input_ids=full_ids).logits[0]  # (seq_len, vocab)
    prefix_len = prefix_ids.shape[1]
    # Position i's logits predict token i+1. The answer tokens start at
    # prefix_len, so their predicting logits are at prefix_len-1 .. end-2.
    pred_logits = logits[prefix_len - 1: prefix_len - 1 + len(answer_token_ids)]
    log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
    target = torch.tensor(answer_token_ids, device=model.device)
    gold_logp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return float(gold_logp.sum().item())


def compute_unsafe_dropped_chunks(model, tokenizer, prompt_text, texts, non_answer_idxs,
                                   initial_kept_idxs, answer_token_ids, tau):
    """
    ANS-guard safety check, scored against the ALREADY-COMPRESSED candidate
    set (not the original full chain) -- this is what makes it a faithful
    mirror of the team's iterative round-based ANS guard, which checks
    marginal impact against the CURRENT partially-pruned sequence at each
    round, not the original. Testing against the full chain (an earlier
    version of this function) barely ever fired: removing any ONE chunk
    from an otherwise-intact chain leaves plenty of redundant context, so
    the confidence drop is small even for chunks that matter a lot once
    SEVERAL chunks are missing simultaneously -- which is what actually
    happens during real ~50% compression.

    Instead: start from the redundancy-ranking's initial (pre-safety-net)
    compressed set. For each chunk that ranking wants to DROP, check
    whether ADDING IT BACK into the compressed context would meaningfully
    improve answer confidence (delta >= tau). If so, that chunk was unsafe
    to exclude and gets force-protected.

    Returns (unsafe_indices, baseline_logp, deltas). unsafe_indices are
    original chunk indices (not re-mapped local ones, unlike the earlier
    version) -- indices of DROPPED chunks whose re-inclusion would help by
    at least tau.
    """
    kept_set = set(initial_kept_idxs)
    dropped_idxs = [i for i in non_answer_idxs if i not in kept_set]

    def render(idx_set):
        ordered_texts = [texts[i] for i in sorted(idx_set)]
        return ("\n".join(ordered_texts) + "\n#### ") if ordered_texts else "#### "

    baseline_logp = logp_of_answer(model, tokenizer, prompt_text + render(kept_set), answer_token_ids)

    unsafe = set()
    deltas = {}
    for i in dropped_idxs:
        with_i_logp = logp_of_answer(model, tokenizer, prompt_text + render(kept_set | {i}), answer_token_ids)
        delta = with_i_logp - baseline_logp
        deltas[i] = delta
        if delta >= tau:
            unsafe.add(i)
    return unsafe, baseline_logp, deltas


def main():
    with open(RESIDUALS_CACHE, "rb") as f:
        records = pickle.load(f)
    with open(ROUTES_PATH, "rb") as f:
        routes = pickle.load(f)
    embed_matrix = np.load(EMBED_MATRIX_PATH)

    by_key = {(r["problem_id"], r["chain_idx"]): r for r in records}
    n_layers_available = records[0]["chunk_residuals"].shape[1]
    resolved_layer = n_layers_available - 1

    mean, std = fit_correction_for_layer(records, resolved_layer, embed_matrix)

    print(f"Loading teacher model {MODEL_NAME} for ANS-guard leave-one-out scoring...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    all_pids = sorted(routes.keys())
    random.shuffle(all_pids)
    n_test = int(len(all_pids) * TEST_FRACTION)
    test_pids = set(all_pids[:n_test])
    train_pids = [p for p in all_pids if p not in test_pids]
    print(f"{len(train_pids)} train problems, {len(test_pids)} held-out test problems")

    full_examples, compressed_examples = [], []
    test_problems = []

    for pid in all_pids:
        route = routes[pid]
        medoid_chain_idx = route["chain_idx"]
        record = by_key.get((pid, medoid_chain_idx))
        chunked_data = torch.load(f"{CHUNKED_DIR}/{pid}.pt", weights_only=False)
        question = chunked_data["question"]
        gold_answer = str(chunked_data["gold_answer"]).strip()
        texts = [c["text"] for c in route["chunks"]]

        if pid in test_pids:
            test_problems.append({"pid": pid, "question": question, "gold_answer": gold_answer})
            continue  # test problems contribute NOTHING to training data

        if record is None or len(texts) != len(record["chunk_residuals"]):
            continue

        prompt = user_prompt(question)

        # Full chain -- exactly as the medoid produced it (already ends in
        # the real "#### <answer>" line).
        full_completion = "\n".join(texts)
        full_examples.append({"prompt": prompt, "completion": full_completion, "pid": pid})

        # Compressed chain -- redundancy-ranking selection at TARGET_RHO,
        # then DEPENDENCY-CLOSURE EXPANSION so no kept chunk references a
        # number whose derivation got dropped (fixes the "ungrounded
        # number" / "answer never derived" failure modes found by
        # inspection -- see check_ungrounded_numbers.py).
        # ANSWER LINE KEPT (this is training data, not a verification check).
        residuals = record["chunk_residuals"][:, resolved_layer, :]
        last_token_ids = record["last_token_ids"].numpy()
        chunk_token_counts = [c["token_end"] - c["token_start"] + 1 for c in route["chunks"]]
        rank_order = redundancy_rank_order(residuals, last_token_ids, mean, std, embed_matrix)
        question_numbers = set(extract_numbers(question))
        answer_idxs = [i for i, t in enumerate(texts) if re.match(r"^\s*####", t)]

        # ANS GUARD: tested against the redundancy ranking's INITIAL
        # (pre-safety-net) compressed candidate set, not the full chain --
        # see compute_unsafe_dropped_chunks for why this matters. Any
        # chunk the ranking wants to drop, whose re-inclusion would
        # meaningfully help answer confidence given the compressed
        # context, gets force-protected.
        scoring_prompt = build_prompt_for_scoring(tokenizer, question)
        answer_token_ids = get_answer_token_ids(tokenizer, gold_answer)
        non_answer_idxs = [i for i in range(len(texts)) if i not in answer_idxs]
        initial_kept_idxs = select_for_budget(rank_order, chunk_token_counts, TARGET_RHO)
        unsafe_orig_idxs, baseline_logp, deltas = compute_unsafe_dropped_chunks(
            model, tokenizer, scoring_prompt, texts, non_answer_idxs,
            initial_kept_idxs, answer_token_ids, ANS_GUARD_TAU,
        )

        kept_idxs = select_for_budget_dependency_aware(
            rank_order, chunk_token_counts, TARGET_RHO, texts, question_numbers,
            force_keep_idxs=answer_idxs + list(unsafe_orig_idxs),
        )
        kept_texts = [texts[i] for i in kept_idxs]
        # Belt-and-suspenders: force_keep_idxs above should already guarantee
        # the answer line is present, but keep this fallback in case a
        # problem's answer chunk wasn't found by the regex for some reason.
        if not any(re.match(r"^\s*####", t) for t in kept_texts):
            answer_chunks = [t for t in texts if re.match(r"^\s*####", t)]
            if answer_chunks:
                kept_texts.append(answer_chunks[0])
        compressed_completion = "\n".join(kept_texts)
        compressed_examples.append({"prompt": prompt, "completion": compressed_completion, "pid": pid})

        n_unsafe = len(unsafe_orig_idxs)
        print(f"[{pid}] {len(kept_idxs)}/{len(texts)} chunks kept, "
              f"{n_unsafe} chunk(s) ANS-guard-protected (baseline logP={baseline_logp:.2f})")

    with open("train_full.jsonl", "w") as f:
        for ex in full_examples:
            f.write(json.dumps(ex) + "\n")
    with open("train_compressed.jsonl", "w") as f:
        for ex in compressed_examples:
            f.write(json.dumps(ex) + "\n")
    with open("test_problems.json", "w") as f:
        json.dump(test_problems, f, indent=2)

    # Shareable config -- everyone on the team should be able to diff this
    # file against their own to confirm they're using identical settings
    # (target budget, split fraction, split seed).
    shared_config = {
        "target_rho": TARGET_RHO,
        "test_fraction": TEST_FRACTION,
        "random_seed": RANDOM_SEED,
        "ans_guard_tau": ANS_GUARD_TAU,
        "n_train": len(full_examples),
        "n_test": len(test_problems),
        "compression_method": "nearest-neighbor redundancy ranking (final layer) "
                               "+ dependency closure + chunk-level ANS leave-one-out guard",
        "note": "test set is NOT filtered by teacher-model post-compression answerability",
    }
    with open("sft_config.json", "w") as f:
        json.dump(shared_config, f, indent=2)

    print(f"Saved train_full.jsonl ({len(full_examples)} examples)")
    print(f"Saved train_compressed.jsonl ({len(compressed_examples)} examples, "
          f"target_rho={TARGET_RHO})")
    print(f"Saved test_problems.json ({len(test_problems)} problems, held out)")
    print(f"Saved sft_config.json (share this so everyone confirms matching settings)")

    avg_full_tokens = np.mean([len(e["completion"].split()) for e in full_examples])
    avg_compressed_tokens = np.mean([len(e["completion"].split()) for e in compressed_examples])
    print(f"\n(rough word-count check, not real tokenization) "
          f"avg full completion: {avg_full_tokens:.0f} words, "
          f"avg compressed: {avg_compressed_tokens:.0f} words")


if __name__ == "__main__":
    main()