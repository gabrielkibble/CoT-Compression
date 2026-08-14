"""
Stage 1 — explore LiveCodeBench before committing to a subset size or a
grading harness.

Run inside a container with `datasets` installed:
    python3 explore_livecodebench.py
"""
from datasets import load_dataset

# release_v5 is the latest tag as of the CREST-era lighteval dependency
# (lighteval's lcb/main.py used a similar config). Adjust the version_tag
# if a newer release is preferred once you check what's current.
ds = load_dataset("livecodebench/code_generation_lite", version_tag="release_v5")

print("=== Dataset structure ===")
print(ds)

split = "test" if "test" in ds else list(ds.keys())[0]
print(f"\n=== Using split: {split} ===")
print(f"Number of examples: {len(ds[split])}")

print("\n=== First example's keys ===")
example = ds[split][0]
for k, v in example.items():
    preview = str(v)
    if len(preview) > 200:
        preview = preview[:200] + "... [truncated]"
    print(f"  {k}: {preview}")

print("\n=== Rough length stats for the problem/question field ===")
# Try common field names since LiveCodeBench's schema isn't confirmed yet
candidate_fields = ["question_content", "question", "problem", "prompt"]
field_found = None
for f in candidate_fields:
    if f in example:
        field_found = f
        break

if field_found:
    lengths = [len(ds[split][i][field_found]) for i in range(min(500, len(ds[split])))]
    print(f"Field used: '{field_found}'")
    print(f"  min char length: {min(lengths)}")
    print(f"  max char length: {max(lengths)}")
    print(f"  avg char length: {sum(lengths)/len(lengths):.1f}")
else:
    print(f"None of {candidate_fields} found in example keys — inspect manually:")
    print(list(example.keys()))

print("\n=== Difficulty / tag field check (if present) ===")
for f in ["difficulty", "tags", "platform", "contest_date"]:
    if f in example:
        print(f"  '{f}' present, example value: {example[f]}")