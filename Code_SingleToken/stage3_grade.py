"""
Stage 3 — grade Stage 2's generations against LiveCodeBench test cases,
then filter to examples where No-CoT is wrong AND CoT is right.

Run inside your container:
    python3 stage3_grade.py

Requires data/lcb_generations.jsonl from stage2_generate.py.

NOTE on functional-test grading fidelity: this implements a from-scratch
harness informed by LiveCodeBench's reference implementation (traced via
https://github.com/mlfoundations/evalchemy/blob/main/eval/chat_benchmarks/LiveCodeBenchv5/livecodebench_utils.py,
itself copied from NovaSky-AI/SkyThought), NOT a byte-for-byte port of it.
That reference harness extracts a callable via
`completion.split("(")[0].split()[-1]` and calls it directly, which assumes
the completion defines a bare function — but LeetCode-sourced completions
in this project's data consistently use the `class Solution:` wrapper
matching `starter_code`. Rather than replicate that mismatched assumption,
this harness instantiates `Solution` and calls the method by name (parsed
from `starter_code` via regex), which is more directly correct for what
these completions actually look like. Input/output parsing (JSON-decode
list-literal inputs into a single positional arg, JSON-decode expected
output, compare by value) follows the reference implementation's approach.
This means results may not be perfectly identical to an official LCB
leaderboard run — worth keeping in mind if comparing numbers externally.
"""
import json
import re
import subprocess
import sys

from decode_test_cases import decode_private_test_cases

INPUT_PATH = "data/lcb_generations.jsonl"
FILTERED_OUTPUT_PATH = "data/lcb_cot_necessary.jsonl"
BREAKDOWN_OUTPUT_PATH = "data/lcb_grading_breakdown.json"

STDIN_TIMEOUT_SECONDS = 6
FUNCTIONAL_TIMEOUT_SECONDS = 6


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------
CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(completion: str) -> str | None:
    """
    Returns the LAST fenced code block in the completion (CoT completions
    often contain an earlier draft + a final revised block — we want the
    one the model actually intended as its final answer). Falls back to
    the raw completion text if no fence is found (covers the case where a
    model ignores formatting instructions entirely).
    """
    matches = CODE_FENCE_RE.findall(completion)
    if matches:
        return matches[-1].strip()
    stripped = completion.strip()
    return stripped if stripped else None


# ---------------------------------------------------------------------------
# stdin-format execution
# ---------------------------------------------------------------------------
def run_against_stdin_test_case(code: str, test_input: str, expected_output: str) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=STDIN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

    actual_lines = [ln.rstrip() for ln in result.stdout.strip().splitlines()]
    expected_lines = [ln.rstrip() for ln in expected_output.strip().splitlines()]
    return actual_lines == expected_lines


# ---------------------------------------------------------------------------
# functional-format execution (class Solution: ... style, LeetCode)
# ---------------------------------------------------------------------------
METHOD_NAME_RE = re.compile(r"def\s+(\w+)\s*\(\s*self")


def extract_method_name(starter_code: str) -> str | None:
    match = METHOD_NAME_RE.search(starter_code)
    return match.group(1) if match else None


def parse_functional_input(test_input: str):
    """
    Mirrors the reference harness's approach: if the raw input string is a
    JSON list literal, decode it as ONE argument and wrap in a single-
    element list (so it gets unpacked as one positional arg). Otherwise,
    fall back to treating each newline-separated, non-empty line as its
    own argument (typed: try int, then float, then raw string/JSON).
    """
    stripped = test_input.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return [json.loads(stripped)]

    args = []
    for line in stripped.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith('"') and line.endswith('"'):
            args.append(line.strip('"'))
            continue
        if line.startswith("[") and line.endswith("]"):
            args.append(json.loads(line))
            continue
        try:
            args.append(int(line))
            continue
        except ValueError:
            pass
        try:
            args.append(float(line))
            continue
        except ValueError:
            pass
        args.append(line)
    return args


def parse_functional_output(test_output: str):
    try:
        return json.loads(test_output.strip())
    except json.JSONDecodeError:
        return test_output.strip()


def _functional_worker(code: str, method_name: str, args: list, conn):
    try:
        namespace = {}
        exec(code, namespace)
        solution_cls = namespace.get("Solution")
        if solution_cls is None:
            conn.send(("error", "No 'Solution' class defined in completion"))
            return
        instance = solution_cls()
        method = getattr(instance, method_name, None)
        if method is None:
            conn.send(("error", f"Solution has no method '{method_name}'"))
            return
        result = method(*args)
        conn.send(("ok", result))
    except Exception as e:
        conn.send(("error", str(e)))
    finally:
        conn.close()


def run_against_functional_test_case(code: str, method_name: str, test_input: str, test_output: str) -> bool:
    import multiprocessing

    try:
        args = parse_functional_input(test_input)
        expected = parse_functional_output(test_output)
    except Exception:
        return False

    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=_functional_worker, args=(code, method_name, args, child_conn))
    p.start()
    p.join(timeout=FUNCTIONAL_TIMEOUT_SECONDS)

    if p.is_alive():
        p.terminate()
        p.join()
        return False

    if not parent_conn.poll():
        return False

    status, value = parent_conn.recv()
    if status != "ok":
        return False
    return value == expected


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def grade_completion(completion: str, test_cases: list, starter_code: str) -> bool:
    code = extract_code(completion)
    if code is None:
        return False

    method_name = extract_method_name(starter_code) if starter_code else None

    for tc in test_cases:
        testtype = tc.get("testtype", "stdin")
        if testtype == "stdin":
            if not run_against_stdin_test_case(code, tc["input"], tc["output"]):
                return False
        elif testtype == "functional":
            if method_name is None:
                return False  # can't grade without knowing which method to call
            if not run_against_functional_test_case(code, method_name, tc["input"], tc["output"]):
                return False
        else:
            return False  # unknown test type — treat as ungradable/fail
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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
        "ungraded_error": 0,
    }

    filtered = []

    for i, rec in enumerate(records):
        if (i + 1) % 50 == 0:
            print(f"  Grading {i+1}/{len(records)}...")

        try:
            public = json.loads(rec["public_test_cases"])
            private = decode_private_test_cases(rec["private_test_cases"])
            test_cases = public + private
        except Exception as e:
            print(f"  [{rec.get('question_id')}] Failed to decode test cases: {e}")
            breakdown["ungraded_error"] += 1
            continue

        starter_code = rec.get("starter_code", "")

        no_cot_correct = grade_completion(rec["no_cot_completion"], test_cases, starter_code)
        cot_correct = grade_completion(rec["cot_completion"], test_cases, starter_code)

        rec["no_cot_correct"] = no_cot_correct
        rec["cot_correct"] = cot_correct

        if no_cot_correct and cot_correct:
            breakdown["both_correct"] += 1
        elif not no_cot_correct and not cot_correct:
            breakdown["both_wrong"] += 1
        elif not no_cot_correct and cot_correct:
            breakdown["cot_only_correct"] += 1
            filtered.append(rec)
        elif no_cot_correct and not cot_correct:
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