"""
Stage 2 — generate both No-CoT and CoT completions for every LiveCodeBench
example, using Llama 3.1 8B-Instruct.

Requires HF_TOKEN env var set (Llama 3.1 is a gated model) and access
already approved on your HuggingFace account.

Run inside your container:
    export HF_TOKEN=<your_token>
    python3 stage2_generate.py
"""
import os
import json
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_PATH = "data/lcb_generations.jsonl"
MAX_MODEL_LEN = 8192   # problem text avg ~1437 chars; generous headroom for CoT + code
MAX_NEW_TOKENS = 2048  # adjust upward if CoT completions are getting cut off

assert os.environ.get("HF_TOKEN"), "Set HF_TOKEN before running — Llama 3.1 is a gated model."

print("Loading LiveCodeBench...")
ds = load_dataset("livecodebench/code_generation_lite", version_tag="release_v5", trust_remote_code=True)["test"]
print(f"Loaded {len(ds)} examples.")

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ["HF_TOKEN"])
model = LLM(model=MODEL_NAME, max_model_len=MAX_MODEL_LEN, dtype="bfloat16", gpu_memory_utilization=0.8)

sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)


def build_prompt(problem_text: str, starter_code: str, cot: bool) -> str:
    instruction = (
        "Think through this step by step before giving your final code solution."
        if cot else
        "Respond with only the final code solution, no explanation."
    )
    content = f"{instruction}\n\n{problem_text}"
    if starter_code:
        content += f"\n\nStarter code:\n{starter_code}"
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


print("Building prompts...")
no_cot_prompts = [build_prompt(ex["question_content"], ex["starter_code"], cot=False) for ex in ds]
cot_prompts = [build_prompt(ex["question_content"], ex["starter_code"], cot=True) for ex in ds]

print("Generating No-CoT completions...")
no_cot_outputs = model.generate(no_cot_prompts, sampling_params)

print("Generating CoT completions...")
cot_outputs = model.generate(cot_prompts, sampling_params)

print(f"Writing results to {OUTPUT_PATH}...")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as fout:
    for ex, no_cot_out, cot_out in zip(ds, no_cot_outputs, cot_outputs):
        record = {
            "question_id": ex["question_id"],
            "question_title": ex["question_title"],
            "question_content": ex["question_content"],
            "starter_code": ex["starter_code"],
            "difficulty": ex["difficulty"],
            "platform": ex["platform"],
            "public_test_cases": ex["public_test_cases"],
            "private_test_cases": ex["private_test_cases"],
            "no_cot_completion": no_cot_out.outputs[0].text,
            "no_cot_completion_tokens": len(no_cot_out.outputs[0].token_ids),
            "cot_completion": cot_out.outputs[0].text,
            "cot_completion_tokens": len(cot_out.outputs[0].token_ids),
        }
        fout.write(json.dumps(record) + "\n")

print("Done.")