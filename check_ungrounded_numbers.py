"""
Scan train_compressed.jsonl for two related but distinct failure modes,
found by eyeballing 3 examples where the compression dropped a step whose
OUTPUT NUMBER is used later, breaking the derivation chain:

  CHECK A -- "ungrounded operand": a line does arithmetic using a number
             that was never established anywhere earlier (not in the
             question, not computed by any prior kept line). Signals a
             dropped intermediate step whose result got reused downstream.

  CHECK B -- "answer never derived": the final "#### <answer>" number
             never appears as the RESULT of any equation anywhere in the
             body of the completion. Signals the actual derivation of the
             final answer was dropped entirely -- the compressed chain
             states intent ("we need to convert X to Y") but never shows
             the arithmetic, and the correct number only appears because
             it was force-appended from the original medoid's answer line.

Both checks are regex heuristics on numbers -- not perfect, but should
give a reasonable estimate of how common this pattern is across the full
training set before deciding whether to fix the compression method.
"""

import re
import json

TRAIN_COMPRESSED_PATH = "train_compressed.jsonl"
REPORT_PATH = "ungrounded_number_report.json"


NUMBER_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_numbers(text: str):
    """Extract numbers as floats, stripping $ and , formatting."""
    nums = []
    for m in NUMBER_RE.findall(text):
        cleaned = m.replace("$", "").replace(",", "")
        try:
            nums.append(float(cleaned))
        except ValueError:
            continue
    return nums


def check_example(prompt: str, completion: str):
    """
    Returns (has_ungrounded_operand, has_undelivered_answer, details_dict).
    """
    # Numbers given in the question itself are always legitimate to use
    # without derivation.
    question_part = prompt.split("Solve this step by step")[0]
    established = set(extract_numbers(question_part))

    lines = [l for l in completion.split("\n") if l.strip()]

    ungrounded_hits = []
    any_equation_results = set()

    answer_line = None
    for line in lines:
        if re.match(r"^\s*####", line):
            answer_line = line
            continue

        if "=" in line:
            first_eq = line.index("=")
            lhs_text = line[:first_eq]
            operands = extract_numbers(lhs_text)
            for op in operands:
                # Small step-numbering digits (1-20ish) at the START of a
                # line are usually "1. Step description" labels, not real
                # operands -- skip a lone leading small integer to reduce
                # false positives.
                if op == operands[0] and op == int(op) and 0 < op <= 20 and lhs_text.strip().startswith(str(int(op))):
                    continue
                if op not in established:
                    ungrounded_hits.append({"line": line.strip(), "ungrounded_number": op})

            # Everything after the LAST "=" is the line's final result(s).
            last_eq = line.rindex("=")
            result_numbers = extract_numbers(line[last_eq + 1:])
            any_equation_results.update(result_numbers)

        established.update(extract_numbers(line))

    has_ungrounded = len(ungrounded_hits) > 0

    has_undelivered_answer = False
    if answer_line:
        answer_nums = extract_numbers(answer_line)
        if answer_nums:
            answer_val = answer_nums[0]
            # Allow small floating point slop (e.g. rounding).
            if not any(abs(answer_val - r) < 1e-6 for r in any_equation_results):
                has_undelivered_answer = True

    return has_ungrounded, has_undelivered_answer, {
        "ungrounded_hits": ungrounded_hits,
        "answer_line": answer_line,
    }


def main():
    examples = []
    with open(TRAIN_COMPRESSED_PATH) as f:
        for line in f:
            examples.append(json.loads(line))

    n_ungrounded = 0
    n_undelivered = 0
    n_either = 0
    flagged = []

    for ex in examples:
        has_ungrounded, has_undelivered, details = check_example(ex["prompt"], ex["completion"])
        if has_ungrounded:
            n_ungrounded += 1
        if has_undelivered:
            n_undelivered += 1
        if has_ungrounded or has_undelivered:
            n_either += 1
            flagged.append({
                "pid": ex["pid"],
                "has_ungrounded_operand": has_ungrounded,
                "has_undelivered_answer": has_undelivered,
                **details,
            })

    total = len(examples)
    print(f"Scanned {total} compressed training examples\n")
    print(f"Ungrounded operand (uses a number never established):  {n_ungrounded}/{total} ({n_ungrounded/total:.0%})")
    print(f"Answer never derived (final number not a computed result): {n_undelivered}/{total} ({n_undelivered/total:.0%})")
    print(f"EITHER failure: {n_either}/{total} ({n_either/total:.0%})")

    with open(REPORT_PATH, "w") as f:
        json.dump(flagged, f, indent=2)
    print(f"\nSaved flagged examples -> {REPORT_PATH}")


if __name__ == "__main__":
    main()