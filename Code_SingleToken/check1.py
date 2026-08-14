"""
Cross-tabulates platform (codeforces/leetcode/atcoder) against the four-way
correctness outcome (both_correct / both_wrong / cot_only_correct /
no_cot_only_correct), to check whether no_cot_only_correct skews toward
LeetCode/functional-format problems specifically.

Re-grades everything from scratch (reuses stage3_grade.py's grading
functions directly) — takes a similar amount of time as the original
stage3_grade.py run, since it's the same subprocess-per-test-case cost.

Run inside your container, from the same directory as stage3_grade.py:
    python3 check_platform_outcome_breakdown.py
"""
import json
import collections

from decode_test_cases import decode_private_test_cases
from stage3_grade import grade_completion

INPUT_PATH = "data/lcb_generations.jsonl"

print(f"Loading {INPUT_PATH}...")
with open(INPUT_PATH) as f:
    recs = [json.loads(l) for l in f]
print(f"Loaded {len(recs)} records.")

by_platform_outcome = collections.Counter()

for i, rec in enumerate(recs):
    if (i + 1) % 50 == 0:
        print(f"  Processing {i+1}/{len(recs)}...")

    public = json.loads(rec["public_test_cases"])
    private = decode_private_test_cases(rec["private_test_cases"])
    test_cases = public + private
    starter_code = rec.get("starter_code", "")

    no_cot_correct = grade_completion(rec["no_cot_completion"], test_cases, starter_code)
    cot_correct = grade_completion(rec["cot_completion"], test_cases, starter_code)

    if no_cot_correct and cot_correct:
        outcome = "both_correct"
    elif not no_cot_correct and not cot_correct:
        outcome = "both_wrong"
    elif not no_cot_correct and cot_correct:
        outcome = "cot_only_correct"
    else:
        outcome = "no_cot_only_correct"

    by_platform_outcome[(rec["platform"], outcome)] += 1

print("\n=== Platform x Outcome breakdown ===")
for (platform, outcome), count in sorted(by_platform_outcome.items()):
    print(f"  {platform:12s} {outcome:20s} {count}")

print("\n=== Platform totals (for reference) ===")
platform_totals = collections.Counter()
for (platform, _), count in by_platform_outcome.items():
    platform_totals[platform] += count
for platform, total in platform_totals.items():
    print(f"  {platform:12s} {total}")