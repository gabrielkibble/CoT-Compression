"""
Stage 4 pilot — ProxySPEX on CoT tokens for a small handful of GSM8K
CoT-necessary examples.

Reuses the model-loading and blank-token logic already validated in the
LiveCodeBench project's stage4_pilot.py (device_map={"": 0} to avoid
silent CPU fallback; Llama 3.1's reserved special token as a semantically
neutral "blank", since it has no true pad/unk token).

Value function ("game"): given a mask over CoT reasoning tokens (1=keep,
0=mask), substitute masked positions with the blank token, run a forward
pass, and return the model's probability assigned to the GOLD answer's
token(s) at the position(s) where the model's own (correct) answer
appears in the original CoT completion — teacher-forced, not regenerated,
matching the "single forward pass, sequence length preserved" design
decision from the LiveCodeBench project.

Run inside your container:
    pip install shapiq[sparse,proxy]
    python3 stage4_proxyspex_pilot.py
"""
import json
import re
import math
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import shapiq

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
INPUT_PATH = "data/gsm8k_cot_necessary.jsonl"
OUTPUT_PATH = "data/stage4_proxyspex_pilot_results.jsonl"

N_PILOT_EXAMPLES = 1
ALPHA = 1  # paper's budget heuristic: budget = alpha * n * log2(n)
BATCH_SIZE = 16  # forward passes per batch when evaluating the game function

# Same extraction patterns as stage3_grade_gsm8k.py, but we need the
# CHARACTER SPAN here too, not just the parsed value.
BOXED_RE = re.compile(r"\\boxed\{(-?[\d,]+(?:\.\d+)?)\}")
ANSWER_PHRASE_RE = re.compile(r"answer(?:\s+is)?\s*:?\s*\$?(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE)
LAST_NUMBER_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    return tokenizer, model


def get_blank_token_id(tokenizer):
    reserved = "<|reserved_special_token_0|>"
    token_id = tokenizer.convert_tokens_to_ids(reserved)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"Could not resolve {reserved} to a real token id.")
    return token_id


def find_answer_span(completion: str):
    """
    Returns (char_start, char_end) of the LAST matched answer expression
    in the completion, using the same priority order as
    stage3_grade_gsm8k.py's extract_predicted_answer: boxed, then
    "answer is X" phrasing, then fallback to the last standalone number.
    """
    boxed_match = None
    for m in BOXED_RE.finditer(completion):
        boxed_match = m
    if boxed_match:
        return boxed_match.start(1), boxed_match.end(1)

    phrase_matches = list(ANSWER_PHRASE_RE.finditer(completion))
    if phrase_matches:
        m = phrase_matches[-1]
        return m.start(1), m.end(1)

    number_matches = list(LAST_NUMBER_RE.finditer(completion))
    if number_matches:
        m = number_matches[-1]
        return m.start(), m.end()

    return None


def build_full_sequence(tokenizer, question: str, cot_completion: str):
    instruction = "Think through this step by step before giving your final numeric answer."
    messages = [{"role": "user", "content": f"{instruction}\n\n{question}"}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    full_text = prompt_text + cot_completion
    encoding = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
    prompt_len_tokens = len(tokenizer(prompt_text)["input_ids"])
    return full_text, prompt_text, encoding, prompt_len_tokens


def char_span_to_token_span(encoding, abs_char_start, abs_char_end):
    offsets = encoding["offset_mapping"][0].tolist()
    token_start = None
    token_end = None
    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if token_start is None and end > abs_char_start:
            token_start = i
        if start < abs_char_end:
            token_end = i
    if token_start is None or token_end is None:
        return None
    return token_start, token_end + 1


@torch.no_grad()
def batched_game_values(model, base_input_ids, reasoning_positions, mask_matrix, blank_token_id,
                         answer_token_start, answer_token_end, device):
    """
    mask_matrix: (num_samples, n_reasoning_tokens) binary array, 1=keep, 0=mask.
    Returns: (num_samples,) array of the model's average probability across
    the gold answer's token positions, under each masked version.

    Vectorized mask application (no nested Python loops over positions),
    since the original per-element Python assignment loop was the likely
    cause of extreme CPU-time blowup on real runs (observed: ~24 hours
    wall-clock, ~5.7 CPU-cores continuously, for what should have been a
    GPU-bound workload).
    """
    num_samples = mask_matrix.shape[0]
    seq_len = base_input_ids.shape[1]
    values = np.zeros(num_samples, dtype=np.float64)

    reasoning_positions_t = torch.tensor(reasoning_positions, dtype=torch.long)

    for batch_start in range(0, num_samples, BATCH_SIZE):
        batch_masks = mask_matrix[batch_start:batch_start + BATCH_SIZE]
        batch_size_actual = batch_masks.shape[0]

        # Start from the base sequence, repeated for the batch.
        batch_input_ids = base_input_ids.repeat(batch_size_actual, 1).clone()

        # Vectorized masking: build a boolean tensor of which (row, seq_pos)
        # entries should become the blank token, then assign in one shot.
        mask_t = torch.tensor(batch_masks, dtype=torch.bool)  # (batch, n_reasoning)
        keep_mask_full = torch.ones((batch_size_actual, seq_len), dtype=torch.bool)
        # Scatter the per-reasoning-token keep/mask flags into full-sequence positions
        keep_mask_full[:, reasoning_positions_t] = mask_t
        batch_input_ids[~keep_mask_full] = blank_token_id

        batch_input_ids = batch_input_ids.to(device)
        outputs = model(batch_input_ids)
        logits = outputs.logits  # (batch, seq_len, vocab)
        probs = torch.softmax(logits, dim=-1)

        # Also vectorize the answer-span probability extraction across the batch.
        answer_positions = list(range(answer_token_start, answer_token_end))
        answer_positions = [p for p in answer_positions if p > 0]
        if answer_positions:
            predicting_positions = torch.tensor([p - 1 for p in answer_positions], dtype=torch.long)
            actual_token_ids = base_input_ids[0, answer_positions]  # (num_answer_tokens,)
            # probs[:, predicting_positions, actual_token_ids] via advanced indexing
            batch_probs = probs[:, predicting_positions, :]  # (batch, num_answer_tokens, vocab)
            gathered = batch_probs[:, torch.arange(len(answer_positions)), actual_token_ids]  # (batch, num_answer_tokens)
            batch_values = gathered.float().mean(dim=1).cpu().numpy()
        else:
            batch_values = np.zeros(batch_size_actual)

        values[batch_start:batch_start + batch_size_actual] = batch_values

    return values


def run_pilot_example(tokenizer, model, rec, blank_token_id, device):
    question = rec["question"]
    cot_completion = rec["cot_completion"]

    span = find_answer_span(cot_completion)
    if span is None:
        print(f"  Could not find answer span — skipping.")
        return None
    char_start, char_end = span

    full_text, prompt_text, encoding, prompt_len_tokens = build_full_sequence(tokenizer, question, cot_completion)
    abs_char_start = len(prompt_text) + char_start
    abs_char_end = len(prompt_text) + char_end

    token_span = char_span_to_token_span(encoding, abs_char_start, abs_char_end)
    if token_span is None:
        print(f"  Could not map answer span to tokens — skipping.")
        return None
    answer_token_start, answer_token_end = token_span

    input_ids = encoding["input_ids"]
    reasoning_positions = list(range(prompt_len_tokens, answer_token_start))
    n = len(reasoning_positions)

    if n < 2:
        print(f"  Too few reasoning tokens ({n}) to run ProxySPEX — skipping.")
        return None

    print(f"  n_reasoning_tokens={n}, answer_span=({answer_token_start},{answer_token_end})")

    call_count = [0]
    cumulative_game_time = [0.0]

    def game(mask_matrix):
        import time as _time
        _start = _time.time()
        mask_matrix = np.asarray(mask_matrix)
        result = batched_game_values(
            model, input_ids, reasoning_positions, mask_matrix, blank_token_id,
            answer_token_start, answer_token_end, device
        )
        _elapsed = _time.time() - _start
        call_count[0] += 1
        cumulative_game_time[0] += _elapsed
        print(f"    [game call {call_count[0]}] {mask_matrix.shape[0]} masks, "
              f"{_elapsed:.2f}s (cumulative game time: {cumulative_game_time[0]:.2f}s)")
        return result

    budget = int(ALPHA * n * math.log2(max(n, 2)))
    print(f"  Running ProxySPEX with budget={budget}...")

    import time as _time
    _proxyspex_start = _time.time()
    approximator = shapiq.ProxySPEX(n=n, index="FBII", max_order=2, hpo=False)
    interaction_values = approximator.approximate(budget=budget, game=game)
    _proxyspex_total = _time.time() - _proxyspex_start

    print(f"  TOTAL ProxySPEX time: {_proxyspex_total:.2f}s | "
          f"game-function time: {cumulative_game_time[0]:.2f}s | "
          f"overhead (LightGBM + recovery, etc.): {_proxyspex_total - cumulative_game_time[0]:.2f}s | "
          f"game calls: {call_count[0]}")

    # get_n_order_values(1) returns the per-token (singleton) attribution
    # array, shape (n,) — this is what we actually need for validating
    # against true knockout and for the scatter plot, rather than just
    # keeping the raw InteractionValues object around unexamined.
    order1_importance = interaction_values.get_n_order_values(1).tolist()
    order2_importance = interaction_values.get_n_order_values(2).tolist()

    return {
        "question": question,
        "n_reasoning_tokens": n,
        "budget_used": budget,
        "answer_token_span": [answer_token_start, answer_token_end],
        "reasoning_positions": reasoning_positions,
        "order1_importance": order1_importance,   # per-token singleton scores
        "order2_importance": order2_importance,   # per-pair interaction scores
        "interaction_values_repr": str(interaction_values),
    }


def main():
    print(f"Loading {INPUT_PATH}...")
    with open(INPUT_PATH) as f:
        records = [json.loads(line) for line in f]
    print(f"Loaded {len(records)} CoT-necessary examples. Using first {N_PILOT_EXAMPLES}.")

    tokenizer, model = load_model()
    blank_token_id = get_blank_token_id(tokenizer)
    device = next(model.parameters()).device
    print(f"Using blank token id {blank_token_id} ({tokenizer.decode([blank_token_id])!r})")

    results = []
    with open(OUTPUT_PATH, "w") as f:
        for rec in records[:N_PILOT_EXAMPLES]:
            print(f"\nProcessing: {rec['question'][:80]}...")
            result = run_pilot_example(tokenizer, model, rec, blank_token_id, device)
            if result is not None:
                results.append(result)
                f.write(json.dumps(result) + "\n")
                f.flush()  # ensure it's actually on disk immediately, not buffered

    print(f"\nSaved {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()