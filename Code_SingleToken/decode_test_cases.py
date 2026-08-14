"""
Decode LiveCodeBench's `private_test_cases` field, which is encoded as
base64(zlib(pickle(json_string))).

Public test cases are already plain JSON strings and don't need this —
only `private_test_cases` uses this encoding.
"""
import base64
import zlib
import pickle
import json


def decode_private_test_cases(encoded: str):
    """
    Returns a list of {"input": ..., "output": ..., "testtype": ...} dicts,
    same shape as the already-plain public_test_cases field.
    """
    decoded = base64.b64decode(encoded)
    decompressed = zlib.decompress(decoded)
    original = pickle.loads(decompressed)
    # `original` is itself a JSON string in LiveCodeBench's format
    return json.loads(original)


if __name__ == "__main__":
    # Quick smoke test against a real example, run this after Stage 1's
    # explore_livecodebench.py has confirmed the dataset loads correctly.
    from datasets import load_dataset

    ds = load_dataset("livecodebench/code_generation_lite", version_tag="release_v5")
    example = ds["test"][0]

    public = json.loads(example["public_test_cases"])
    private = decode_private_test_cases(example["private_test_cases"])

    print(f"Public test cases: {len(public)}")
    print(f"Private test cases: {len(private)}")
    print("First private test case:", private[0] if private else "(none)")