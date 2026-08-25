"""
Print a comparison table from evaluation_results.jsonl -- the accumulated
summary rows from every evaluate_student.py run.
"""
import json

with open("evaluation_results.jsonl") as f:
    rows = [json.loads(line) for line in f]

print(f"{'model':30s} {'accuracy':>10s} {'n_correct/total':>16s} {'avg_tokens':>12s}")
print("-" * 72)
for r in rows:
    print(f"{r['model']:30s} {r['accuracy']:>10.1%} "
          f"{r['n_correct']:>7d}/{r['n_total']:<7d} {r['avg_tokens']:>12.1f}")