"""
Phase 6, step 3: evaluate a fine-tuned student on the HELD-OUT test problems.

Usage:
  python3 evaluate_student.py --model student_full
  python3 evaluate_student.py --model student_compressed

Loads test_problems.json (the problems build_sft_dataset.py set aside and
NEVER used to build training data), generates a fresh response from the
fine-tuned student for each, and measures accuracy + average generated
reasoning length. This is the real Phase 6 comparison point -- does the
student trained on compressed reasoning generalize to problems it never
saw a chain for?
"""

import argparse
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TEST_PROBLEMS_PATH = "test_problems.json"
MAX_NEW_TOKENS = 400


def user_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "Solve this step by step. On the VERY LAST line of your "
        "response, write ONLY the final numeric answer in exactly "
        "this format (no dollar signs, no extra words):\n"
        "#### <number>"
    )


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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to the LoRA adapter directory")
    args = parser.parse_args()

    print(f"Loading base model {BASE_MODEL} + adapter {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, args.model)
    model.eval()

    with open(TEST_PROBLEMS_PATH) as f:
        test_problems = json.load(f)
    print(f"Evaluating on {len(test_problems)} held-out test problems...\n")

    n_correct = 0
    token_lengths = []
    per_example = []
    for p in test_problems:
        messages = [{"role": "user", "content": user_prompt(p["question"])}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[0, input_ids.shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = extract_answer(text)
        correct = answers_match(pred, p["gold_answer"])
        n_correct += correct
        token_lengths.append(len(gen_ids))
        # Keep the raw generated text too -- lets you check afterward
        # whether a "wrong" answer was actually a reasoning failure or
        # just a format/extraction failure (e.g. model never emitted a
        # clean "#### N" line at all).
        per_example.append({
            "pid": p["pid"], "pred": pred, "gold": p["gold_answer"],
            "correct": correct, "tokens": len(gen_ids), "generated_text": text,
        })

        status = "OK " if correct else "ERR"
        print(f"[{p['pid']}] [{status}] pred={pred} gold={p['gold_answer']} tokens={len(gen_ids)}")

    accuracy = n_correct / len(test_problems)
    avg_tokens = sum(token_lengths) / len(token_lengths)
    print(f"\n{args.model}: {n_correct}/{len(test_problems)} correct ({accuracy:.0%}), "
          f"avg {avg_tokens:.0f} tokens")

    # Save full per-example predictions for this run -- useful for the
    # format-vs-reasoning-failure check.
    predictions_path = f"{args.model.rstrip('/').replace('/', '_')}_predictions.json"
    with open(predictions_path, "w") as f:
        json.dump(per_example, f, indent=2)
    print(f"Saved per-example predictions -> {predictions_path}")

    # Append this run's summary to a SHARED results file so all four
    # (condition x seed) runs can be compared side by side afterward,
    # instead of hunting through separate terminal outputs.
    summary_row = {
        "model": args.model,
        "n_correct": n_correct,
        "n_total": len(test_problems),
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
    }
    results_path = "evaluation_results.jsonl"
    with open(results_path, "a") as f:
        f.write(json.dumps(summary_row) + "\n")
    print(f"Appended summary -> {results_path}")


if __name__ == "__main__":
    main()