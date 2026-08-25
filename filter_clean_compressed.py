"""
Filter train_compressed.jsonl down to only examples that PASSED both
check_ungrounded_numbers.py checks (no ungrounded operand, answer
properly derived). Used to test whether the accuracy gap between
student_full and student_compressed is driven by the remaining data-
quality issues, or is a more fundamental effect of compression length
itself (less "thinking room" per example).

Requires ungrounded_number_report.json already produced by
check_ungrounded_numbers.py (run that first, on the CURRENT/fixed
train_compressed.jsonl).
"""
import json

with open("ungrounded_number_report.json") as f:
    flagged = json.load(f)
flagged_pids = {row["pid"] for row in flagged}

examples = []
with open("train_compressed.jsonl") as f:
    for line in f:
        examples.append(json.loads(line))

clean = [ex for ex in examples if ex["pid"] not in flagged_pids]

with open("train_compressed_clean.jsonl", "w") as f:
    for ex in clean:
        f.write(json.dumps(ex) + "\n")

print(f"{len(examples)} total compressed examples, {len(flagged_pids)} flagged, "
      f"{len(clean)} clean -> saved train_compressed_clean.jsonl")