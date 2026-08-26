"""
Phase 0 filter, GSM8K TRAIN split (not test).

Your existing pipeline (capture_chains.py etc.) was built entirely on
GSM8K's TEST split. Your teammate's production script
(consensus_best3_reasoningweighted_nogate_LLAMA_FULL500.py) requires
parent problems identified by index into GSM8K TRAIN, with the question
text validated against that exact official row. This script rebuilds the
Phase 0 filter (wrong without CoT, right with CoT) on TRAIN instead, so
the two pipelines can eventually be reconciled.

CRITICAL: this preserves the real GSM8K train dataset index for every
qualifying row (train_dataset_idx), separate from the sequential "id"
your existing scripts (capture_chains.py etc.) will assign by file line
order. You need BOTH: the sequential id to drive your own pipeline
unchanged, and train_dataset_idx to satisfy your teammate's schema at
export time.

Output: gsm8k_train_cot_necessary.jsonl
  Fields match what capture_chains.py's load_phase0_problems already
  expects (question, gold_answer, no_cot_correct, cot_correct), PLUS
  train_dataset_idx for later export.
"""

import re
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TARGET_COUNT = 500
OUT_PATH = "gsm8k_train_cot_necessary.jsonl"
DIRECT_MAX_NEW_TOKENS = 20
COT_MAX_NEW_TOKENS = 400

GOLD_RE = re.compile(r"####\s*([^\n]+)")


def gold_from_gsm8k(answer_field: str) -> str:
    m = GOLD_RE.search(answer_field)
    if not m:
        raise ValueError(f"Could not parse gold answer: {answer_field!r}")
    return m.group(1).strip().replace(",", "").replace("$", "")


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


def build_cot_prompt(tokenizer, question: str) -> str:
    """Same style as capture_chains.py's build_prompt, kept consistent so
    'cot_correct' here reflects the same generation setup used downstream."""
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


def build_direct_prompt(tokenizer, question: str) -> str:
    messages = [{
        "role": "user",
        "content": (
            f"{question}\n\n"
            "Answer directly with NO reasoning. Write ONLY the final "
            "numeric answer in exactly this format:\n"
            "#### <number>"
        ),
    }]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tokenizer, prompt_text, max_new_tokens):
    input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    print("Loading GSM8K train split...")
    train_ds = load_dataset("openai/gsm8k", "main", split="train")
    print(f"{len(train_ds)} total train examples available")

    qualifying = []
    for idx in range(len(train_ds)):
        if len(qualifying) >= TARGET_COUNT:
            break
        row = train_ds[idx]
        question = row["question"]
        gold = gold_from_gsm8k(row["answer"])

        direct_text = generate(model, tokenizer, build_direct_prompt(tokenizer, question), DIRECT_MAX_NEW_TOKENS)
        direct_pred = extract_answer(direct_text)
        direct_correct = answers_match(direct_pred, gold)

        if direct_correct:
            continue  # not "necessary" -- model already knows it without CoT

        cot_text = generate(model, tokenizer, build_cot_prompt(tokenizer, question), COT_MAX_NEW_TOKENS)
        cot_pred = extract_answer(cot_text)
        cot_correct = answers_match(cot_pred, gold)

        if not cot_correct:
            continue  # doesn't qualify either way -- model can't solve it at all

        qualifying.append({
            "train_dataset_idx": idx,
            "question": question,
            "gold_answer": gold,
            "no_cot_correct": False,
            "cot_correct": True,
        })

        if len(qualifying) % 25 == 0:
            print(f"  {len(qualifying)}/{TARGET_COUNT} qualifying found (scanned up to idx {idx})")

    with open(OUT_PATH, "w") as f:
        for row in qualifying:
            f.write(json.dumps(row) + "\n")

    print(f"\nFound {len(qualifying)} qualifying problems -> {OUT_PATH}")
    if len(qualifying) < TARGET_COUNT:
        print(f"WARNING: only found {len(qualifying)}, wanted {TARGET_COUNT}. "
              f"Ran out of train examples to scan, or filter is stricter than expected.")


if __name__ == "__main__":
    main()