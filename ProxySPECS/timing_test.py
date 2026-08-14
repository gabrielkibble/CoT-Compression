"""
Quick timing test — measures real per-call throughput of the (now
vectorized) game function on a small number of masks, BEFORE running the
full ProxySPEX budget (which can be thousands of calls). Run this first
to catch any remaining performance problems cheaply.

Run inside your container:
    python3 timing_test.py
"""
import json
import time
import numpy as np
from stage4_proxyspex_pilot import (
    load_model, get_blank_token_id, find_answer_span, build_full_sequence,
    char_span_to_token_span, batched_game_values
)

with open("data/gsm8k_cot_necessary.jsonl") as f:
    rec = json.loads(f.readline())

print("Loading model...")
tokenizer, model = load_model()
blank_token_id = get_blank_token_id(tokenizer)
device = next(model.parameters()).device

span = find_answer_span(rec["cot_completion"])
full_text, prompt_text, encoding, prompt_len_tokens = build_full_sequence(
    tokenizer, rec["question"], rec["cot_completion"]
)
abs_start = len(prompt_text) + span[0]
abs_end = len(prompt_text) + span[1]
answer_token_start, answer_token_end = char_span_to_token_span(encoding, abs_start, abs_end)
input_ids = encoding["input_ids"]
reasoning_positions = list(range(prompt_len_tokens, answer_token_start))
n = len(reasoning_positions)
print(f"n_reasoning_tokens={n}")

for num_masks in [16, 64, 256]:
    test_masks = np.random.randint(0, 2, size=(num_masks, n))
    start = time.time()
    values = batched_game_values(
        model, input_ids, reasoning_positions, test_masks, blank_token_id,
        answer_token_start, answer_token_end, device
    )
    elapsed = time.time() - start
    print(f"  {num_masks} masks: {elapsed:.2f}s total, {elapsed/num_masks*1000:.1f}ms/mask")

print("\nExtrapolate to your actual budget (e.g. budget=3049) using the ms/mask figure above "
      "to estimate real total runtime before launching the full pilot.")