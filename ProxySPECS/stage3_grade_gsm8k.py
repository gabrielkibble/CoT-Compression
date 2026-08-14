"""
Stage 3 — grade GSM8K completions by numeric comparison (much simpler
than LiveCodeBench's test-execution harness — no code to run, just
extract-and-compare a final number), then filter to examples where
No-CoT is wrong AND CoT is right (same criteria as the LiveCodeBench
project, per the locked-in decision to keep the two projects comparable).

Run inside your container:
    python3 stage3_grade_gsm8k.py

Requires data/gsm8k_generations.jsonl from stage2_generate_gsm8k.py.
"""
import json
import re

INPUT_PATH = "data/gsm8k_generations.jsonl"
FILTERED_OUTPUT_PATH = "data/gsm8k_cot_necessary.jsonl"
BREAKDOWN_OUTPUT_PATH = "data/gsm8k_grading_breakdown.json"

# Matches \boxed{...} (common LaTeX-style final-answer marker some models
# use when asked for numeric answers), a number after "answer is"/"answer:"
# phrasing, or falls back to the LAST standalone number in the text.
BOXED_RE = re.compile(r"\\boxed\{(-?[\d,]+(?:\.\d+)?)\}")
ANSWER_PHRASE_RE = re.compile(r"answer(?:\s+is)?\s*:?\s*\$?(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE)
LAST_NUMBER_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def extract_predicted_answer(completion: str):
    boxed_match = BOXED_RE.search(completion)
    if boxed_match:
        return normalize_number(boxed_match.group(1))

    # Search from the end for an "answer is X" style phrase — take the LAST
    # such match, since CoT completions may mention numbers earlier in
    # reasoning before stating a final answer.
    phrase_matches = list(ANSWER_PHRASE_RE.finditer(completion))
    if phrase_matches:
        return normalize_number(phrase_matches[-1].group(1))

    # Fallback: last standalone number anywhere in the completion.
    all_numbers = LAST_NUMBER_RE.findall(completion)
    if all_numbers:
        return normalize_number(all_numbers[-1])

    return None


def normalize_number(s: str):
    s = s.replace(",", "")
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except ValueError:
        return None


def numbers_equal(a: str, b: str) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a == b


def main():
    print(f"Loading {INPUT_PATH}...")
    with open(INPUT_PATH) as f:
        records = [json.loads(line) for line in f]
    print(f"Loaded {len(records)} records.")

    breakdown = {
        "both_correct": 0,
        "both_wrong": 0,
        "cot_only_correct": 0,      # this is the subset we want
        "no_cot_only_correct": 0,
    }
    filtered = []

    for rec in records:
        gold = rec["gold_answer"]
        no_cot_pred = extract_predicted_answer(rec["no_cot_completion"])
        cot_pred = extract_predicted_answer(rec["cot_completion"])

        no_cot_correct = numbers_equal(no_cot_pred, gold)
        cot_correct = numbers_equal(cot_pred, gold)

        rec["no_cot_predicted_answer"] = no_cot_pred
        rec["cot_predicted_answer"] = cot_pred
        rec["no_cot_correct"] = no_cot_correct
        rec["cot_correct"] = cot_correct

        if no_cot_correct and cot_correct:
            breakdown["both_correct"] += 1
        elif not no_cot_correct and not cot_correct:
            breakdown["both_wrong"] += 1
        elif not no_cot_correct and cot_correct:
            breakdown["cot_only_correct"] += 1
            filtered.append(rec)
        else:
            breakdown["no_cot_only_correct"] += 1

    print("\n=== Grading breakdown ===")
    for k, v in breakdown.items():
        print(f"  {k}: {v}")

    with open(BREAKDOWN_OUTPUT_PATH, "w") as f:
        json.dump(breakdown, f, indent=2)
    print(f"\nSaved breakdown to {BREAKDOWN_OUTPUT_PATH}")

    with open(FILTERED_OUTPUT_PATH, "w") as f:
        for rec in filtered:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved {len(filtered)} CoT-necessary examples to {FILTERED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()