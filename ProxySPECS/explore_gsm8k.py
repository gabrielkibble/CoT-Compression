"""
Stage 1 — explore GSM8K before committing to field names or a generation
script, same discipline as explore_livecodebench.py from the prior
project (don't assume schema, verify it).

Run inside a container with `datasets` installed:
    python3 explore_gsm8k.py
"""
from datasets import load_dataset

# openai/gsm8k is the standard HF dataset id; "main" is the standard config
# (there's also a "socratic" config with a different reasoning style —
# not what we want here).
ds = load_dataset("openai/gsm8k", "main")

print("=== Dataset structure ===")
print(ds)

split = "test" if "test" in ds else list(ds.keys())[0]
print(f"\n=== Using split: {split} ===")
print(f"Number of examples: {len(ds[split])}")

print("\n=== First example, full content ===")
example = ds[split][0]
for k, v in example.items():
    print(f"--- {k} ---")
    print(v)

print("\n=== Rough length stats for the question field ===")
lengths = [len(ds[split][i]["question"]) for i in range(min(500, len(ds[split])))]
print(f"  min char length: {min(lengths)}")
print(f"  max char length: {max(lengths)}")
print(f"  avg char length: {sum(lengths)/len(lengths):.1f}")

print("\n=== Answer field format check (first 5 examples) ===")
for i in range(5):
    print(f"--- example {i} answer ---")
    print(ds[split][i]["answer"])