import json
import collections
from decode_test_cases import decode_private_test_cases
from stage3_grade import grade_completion, CODE_FENCE_RE

with open('data/lcb_generations.jsonl') as f:
    recs = [json.loads(l) for l in f]

outcome_by_blocks = collections.Counter()

for rec in recs:
    if rec['platform'] != 'leetcode':
        continue

    public = json.loads(rec['public_test_cases'])
    private = decode_private_test_cases(rec['private_test_cases'])
    test_cases = public + private
    starter_code = rec.get('starter_code', '')

    cot_correct = grade_completion(rec['cot_completion'], test_cases, starter_code)
    num_blocks = len(CODE_FENCE_RE.findall(rec['cot_completion']))
    block_bucket = 'multi_block' if num_blocks > 1 else 'single_block'

    outcome_by_blocks[(block_bucket, cot_correct)] += 1

print("=== LeetCode: block count vs CoT correctness ===")
for (bucket, correct), count in sorted(outcome_by_blocks.items()):
    print(f"  {bucket:15s} cot_correct={correct}  count={count}")

for bucket in ['single_block', 'multi_block']:
    total = outcome_by_blocks[(bucket, True)] + outcome_by_blocks[(bucket, False)]
    correct = outcome_by_blocks[(bucket, True)]
    if total:
        print(f"{bucket}: {correct}/{total} = {100*correct/total:.1f}% correct")