"""
Stage 2 — generate both No-CoT and CoT completions for every GSM8K test
example, using Llama 3.1 8B-Instruct.

NOTE: field names/answer format below assume GSM8K's well-documented
standard schema (question / answer, with the gold numeric answer after a
"#### " delimiter in the answer field). This is a stable, widely-used
format, but per this project's established discipline (see the
LiveCodeBench project's field-naming surprises), run explore_gsm8k.py
FIRST and confirm this matches before trusting this script blindly.

Requires HF_TOKEN env var set (Llama 3.1 is a gated model).

Run inside your container:
    export HF_TOKEN=<your_token>
    python3 stage2_generate_gsm8k.py
"""
import os
import re
import json
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_PATH = "data/gsm8k_generations.jsonl"
MAX_MODEL_LEN = 4096   # GSM8K questions are short; CoT completions shorter than LiveCodeBench's code
MAX_NEW_TOKENS = 1024

assert os.environ.get("HF_TOKEN"), "Set HF_TOKEN before running — Llama 3.1 is a gated model."

print("Loading GSM8K...")
ds = load_dataset("openai/gsm8k", "main")["test"]
print(f"Loaded {len(ds)} examples.")

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ["HF_TOKEN"])
model = LLM(model=MODEL_NAME, max_model_len=MAX_MODEL_LEN, dtype="bfloat16", gpu_memory_utilization=0.85)

sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)


def build_prompt(question: str, cot: bool) -> str:
    instruction = (
        "Think through this step by step before giving your final numeric answer."
        if cot else
        "Respond with only the final numeric answer, no explanation."
    )
    content = f"{instruction}\n\n{question}"
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_gold_answer(answer_field: str) -> str:
    """GSM8K's standard format: reasoning text ending in '#### <number>'."""
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    if match:
        return match.group(1).replace(",", "")
    return None


print("Building prompts...")
no_cot_prompts = [build_prompt(ex["question"], cot=False) for ex in ds]
cot_prompts = [build_prompt(ex["question"], cot=True) for ex in ds]

print("Generating No-CoT completions...")
no_cot_outputs = model.generate(no_cot_prompts, sampling_params)

print("Generating CoT completions...")
cot_outputs = model.generate(cot_prompts, sampling_params)

print(f"Writing results to {OUTPUT_PATH}...")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
skipped_gold_parse_failures = 0
with open(OUTPUT_PATH, "w") as fout:
    for ex, no_cot_out, cot_out in zip(ds, no_cot_outputs, cot_outputs):
        gold_answer = extract_gold_answer(ex["answer"])
        if gold_answer is None:
            skipped_gold_parse_failures += 1
            continue
        record = {
            "question": ex["question"],
            "raw_answer_field": ex["answer"],
            "gold_answer": gold_answer,
            "no_cot_completion": no_cot_out.outputs[0].text,
            "no_cot_completion_tokens": len(no_cot_out.outputs[0].token_ids),
            "cot_completion": cot_out.outputs[0].text,
            "cot_completion_tokens": len(cot_out.outputs[0].token_ids),
        }
        fout.write(json.dumps(record) + "\n")

print(f"Done. Skipped {skipped_gold_parse_failures} examples where the gold answer couldn't be parsed "
      f"(check extract_gold_answer's regex against explore_gsm8k.py's output if this number is large).")