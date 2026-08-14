"""
Stage 4 pilot — exhaustive token intervention on a small handful of
examples from the CoT-necessary filtered set.

For each reasoning token position (before the final code block) in each
pilot example, substitutes that one token with either:
  - "blank": the tokenizer's own pad/unk token
  - "filler": the literal text "..."
then does a SINGLE forward pass over the full original-length sequence
(modified reasoning token + everything else unchanged) and measures the
model's teacher-forced probability for the ORIGINAL code-block tokens at
their original positions.

Coarse-grained: did the model's greedy (argmax) prediction at any
code-block position change relative to the unmodified baseline?
Fine-grained: how much did the probability assigned to the original
code-block token drop, at each position, on average?

Uses raw `transformers` rather than vLLM, since we need exact per-token
logits/probabilities for arbitrary target tokens — vLLM's `logprobs=N`
only returns the top-N tokens per position by default, which may not
include our specific target token.

Run inside your container:
    python3 stage4_pilot.py
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from stage3_grade import extract_code

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
INPUT_PATH = "data/lcb_cot_necessary.jsonl"
OUTPUT_PATH = "data/stage4_pilot_results.jsonl"

N_PILOT_EXAMPLES = 3  # small handful, per project decision — exhaustive coverage


def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    return tokenizer, model


def get_blank_token_id(tokenizer):
    """
    Llama 3.1 has no dedicated pad_token or unk_token. Using eos_token
    (<|eot_id|>) as a "blank" would be semantically wrong — it's an
    end-of-turn signal, not a neutral placeholder, and could make the
    model behave as if the turn ended rather than as if content were
    simply missing. Llama 3's reserved special tokens are untrained /
    semantically inert and are a much cleaner stand-in for "blank."
    """
    reserved = "<|reserved_special_token_0|>"
    token_id = tokenizer.convert_tokens_to_ids(reserved)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(
            f"Could not resolve {reserved} to a real token id — "
            "check tokenizer vocab before proceeding."
        )
    return token_id


def build_full_sequence(tokenizer, question_content: str, starter_code: str, cot_completion: str):
    """
    Rebuilds the exact prompt used in Stage 2, then appends the CoT
    completion, tokenizing the whole thing with an offset mapping so we
    can later locate the code block's token span precisely.
    """
    instruction = "Think through this step by step before giving your final code solution."
    content = f"{instruction}\n\n{question_content}"
    if starter_code:
        content += f"\n\nStarter code:\n{starter_code}"
    messages = [{"role": "user", "content": content}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    full_text = prompt_text + cot_completion
    encoding = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
    prompt_len_tokens = len(tokenizer(prompt_text)["input_ids"])
    return full_text, prompt_text, encoding, prompt_len_tokens


def find_code_block_token_span(full_text: str, cot_completion: str, code: str, encoding, prompt_text: str):
    """
    Locates the character span of `code` within `cot_completion` (as
    extracted by extract_code), maps it to the absolute character offset
    within full_text, then to a token index range using the tokenizer's
    offset mapping.
    """
    code_start_in_completion = cot_completion.rfind(code)
    if code_start_in_completion == -1:
        return None
    abs_char_start = len(prompt_text) + code_start_in_completion
    abs_char_end = abs_char_start + len(code)

    offsets = encoding["offset_mapping"][0].tolist()
    token_start = None
    token_end = None
    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue  # special tokens
        if token_start is None and end > abs_char_start:
            token_start = i
        if start < abs_char_end:
            token_end = i
    if token_start is None or token_end is None:
        return None
    return token_start, token_end + 1  # exclusive end


@torch.no_grad()
def get_code_token_probs(model, input_ids, code_token_start, code_token_end):
    """
    Single forward pass; returns, for each code-block token position, the
    model's probability assigned to the ACTUAL token that appears there
    (teacher-forced), plus whether that token was the argmax prediction.
    """
    outputs = model(input_ids)
    logits = outputs.logits[0]  # (seq_len, vocab_size)
    probs = torch.softmax(logits, dim=-1)

    results = []
    for pos in range(code_token_start, code_token_end):
        if pos == 0:
            continue
        predicting_logits_pos = pos - 1  # logits at pos-1 predict token at pos
        actual_token_id = input_ids[0, pos].item()
        prob = probs[predicting_logits_pos, actual_token_id].item()
        argmax_token_id = probs[predicting_logits_pos].argmax().item()
        results.append({
            "position": pos,
            "prob_of_original_token": prob,
            "argmax_matches_original": argmax_token_id == actual_token_id,
        })
    return results


def summarize(results):
    if not results:
        return {"avg_prob": None, "any_argmax_flip": None}
    avg_prob = sum(r["prob_of_original_token"] for r in results) / len(results)
    any_flip = any(not r["argmax_matches_original"] for r in results)
    return {"avg_prob": avg_prob, "any_argmax_flip": any_flip}


def run_pilot_example(tokenizer, model, rec, blank_token_id):
    question_content = rec["question_content"]
    starter_code = rec.get("starter_code", "")
    cot_completion = rec["cot_completion"]

    code = extract_code(cot_completion)
    if code is None:
        print(f"  [{rec['question_id']}] No code block found — skipping.")
        return None

    full_text, prompt_text, encoding, prompt_len_tokens = build_full_sequence(
        tokenizer, question_content, starter_code, cot_completion
    )
    span = find_code_block_token_span(full_text, cot_completion, code, encoding, prompt_text)
    if span is None:
        print(f"  [{rec['question_id']}] Could not locate code block token span — skipping.")
        return None
    code_token_start, code_token_end = span

    device = next(model.parameters()).device
    input_ids = encoding["input_ids"].to(device)
    seq_len = input_ids.shape[1]

    print(f"  [{rec['question_id']}] seq_len={seq_len}, code span=({code_token_start},{code_token_end}), "
          f"reasoning positions to test: {code_token_start - prompt_len_tokens}")

    # Baseline (unmodified) pass
    baseline_results = get_code_token_probs(model, input_ids, code_token_start, code_token_end)
    baseline_summary = summarize(baseline_results)

    per_position_results = []

    # Only intervene on reasoning tokens: from end of prompt to start of code block
    for pos in range(prompt_len_tokens, code_token_start):
        original_token_id = input_ids[0, pos].item()

        for variant in ["blank", "filler"]:
            modified_ids = input_ids.clone()
            if variant == "blank":
                modified_ids[0, pos] = blank_token_id
            else:  # filler
                filler_ids = tokenizer("...", add_special_tokens=False)["input_ids"]
                modified_ids[0, pos] = filler_ids[0] if filler_ids else pad_token_id

            results = get_code_token_probs(model, modified_ids, code_token_start, code_token_end)
            variant_summary = summarize(results)

            per_position_results.append({
                "position": pos,
                "original_token": tokenizer.decode([original_token_id]),
                "variant": variant,
                "avg_prob_of_original_code_tokens": variant_summary["avg_prob"],
                "prob_delta_from_baseline": (
                    variant_summary["avg_prob"] - baseline_summary["avg_prob"]
                    if variant_summary["avg_prob"] is not None and baseline_summary["avg_prob"] is not None
                    else None
                ),
                "any_argmax_flip": variant_summary["any_argmax_flip"],
            })

    return {
        "question_id": rec["question_id"],
        "question_title": rec["question_title"],
        "platform": rec["platform"],
        "baseline_avg_prob": baseline_summary["avg_prob"],
        "baseline_any_argmax_flip": baseline_summary["any_argmax_flip"],
        "num_reasoning_positions_tested": code_token_start - prompt_len_tokens,
        "interventions": per_position_results,
    }


def main():
    print(f"Loading {INPUT_PATH}...")
    with open(INPUT_PATH) as f:
        records = [json.loads(line) for line in f]
    print(f"Loaded {len(records)} CoT-necessary examples. Using first {N_PILOT_EXAMPLES} for exhaustive pilot.")

    tokenizer, model = load_model()
    blank_token_id = get_blank_token_id(tokenizer)
    print(f"Using token id {blank_token_id} ({tokenizer.decode([blank_token_id])!r}) as 'blank'.")

    all_results = []
    for rec in records[:N_PILOT_EXAMPLES]:
        print(f"Processing {rec['question_id']}...")
        result = run_pilot_example(tokenizer, model, rec, blank_token_id)
        if result is not None:
            all_results.append(result)

    with open(OUTPUT_PATH, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(all_results)} pilot example results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()