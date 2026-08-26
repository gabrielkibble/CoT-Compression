# Running `consensus_staged4.py`

This is the "best-3 consensus, continuous reasoning-weighted, no gate"
pruning experiment, extended with a new `*_staged4` variant that splits
each parent's reasoning into 4 contiguous token-count stages and runs the
existing pruning logic independently on each (see the `prune_one_staged4`
docstring in the file for the full design rationale and caveats).

**This is unverified against the real training environment** — I don't
have GPU access to this codebase, so everything below is traced from
reading the code, not confirmed by running it. Test on a small slice
before trusting a full run. Points marked "NEEDS TEAM CONFIRMATION" are
places the code's intent is ambiguous from reading alone and should be
checked with whoever wrote the original file (Harish/Dilyan) before
relying on them.

## 1. Environment

```bash
pip install rich torch transformers accelerate datasets --break-system-packages
```

Needs GPU access to `meta-llama/Llama-3.1-8B-Instruct` (gated — request
access at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct if you
haven't already), and:

```bash
export HF_TOKEN=hf_your_actual_token_here
```

## 2. Config constants to edit at the top of the file

These are hardcoded to a teammate's own machine and paths — change before
running anything:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "2"   # -> whichever GPU is actually free on your machine
HF_ROOT = "/mnt/fast0/dag83/huggingface"    # -> your own HF cache location
WORKDIR = "/mnt/fast0/dag83/results_experiment"  # -> your own scratch/results directory
PARENTS_FILE = None                          # see step 3 below
```

## 3. The parent-CoT manifest (the critical blocker)

This script requires exactly 500 parent problems, each identified by
**index into GSM8K's *train* split** (not test), with the question
validated against that official row. This is a fundamentally different
data source than a test-split-based pipeline — you can't substitute
problems from a different split.

**If your team already has a shared, prepared workdir** (i.e. someone has
already run `cmd_prepare` once and `protocol.json` + `parents.jsonl`
already exist under a shared `WORKDIR`), just point `WORKDIR` at that
location and skip straight to step 5 — you don't need to rebuild anything.

**If you need to build your own 500-parent manifest** (e.g. to
independently verify the pipeline, or because no shared workdir exists
yet), the required schema per row is:

```json
{"dataset_idx": <GSM8K train index>, "question": "...", "gold": "...", "reasoning_text": "..."}
```

See the separate `filter_gsm8k_train.py` / pipeline rebuild /
`export_parents_manifest.py` scripts for how to produce a
`parents_500.jsonl` file in this exact schema from scratch. Once you
have it:

```python
PARENTS_FILE = "/path/to/your/parents_500.jsonl"
```

## 4. Running `cmd_prepare` (NEEDS TEAM CONFIRMATION)

Reading the code: `main()` (the actual entry point run when you execute
this file directly) does **not** call `cmd_prepare` itself — it calls
`load_protocol(paths)` directly and raises `FileNotFoundError` with the
message `"... missing. Run 'prepare' first."` if the workdir isn't
already set up. There's a `run_selected_job()` dispatcher with a
`"prepare"` job defined, but it's never actually called anywhere in this
file (dead code, likely left over from an earlier version) — so there is
**no built-in way to trigger prepare by just running this file**.

If `protocol.json`/`parents.jsonl` don't already exist at your `WORKDIR`,
you need to trigger `cmd_prepare` manually, once, before running `main()`:

```bash
python3 -c "from consensus_staged4 import cmd_prepare, make_prepare_config; cmd_prepare(make_prepare_config())"
```

**Confirm this with your team first** — if there's a separate, earlier
script in this pipeline family that's meant to be the actual "run prepare"
entry point, use that instead rather than improvising the above.

## 5. Running the staged4 experiment

The file's `main()` hardcodes which method it runs:

```python
method = "consensus_best3_reasoningweighted_nogate"
```

To run the **new staged4 variant** instead of the original single-pass
method, change this one line to:

```python
method = "consensus_best3_reasoningweighted_nogate_staged4"
```

Then run the whole pipeline (prune -> SFT -> eval -> report) with:

```bash
python3 consensus_staged4.py
```

**If you only want to test the staged4 pruning step in isolation** (much
cheaper -- skips SFT/eval, useful for a quick sanity check before
committing to the full run), call it directly instead:

```bash
python3 -c "
from consensus_staged4 import cmd_prune, make_prune_config
cmd_prune(make_prune_config('consensus_best3_reasoningweighted_nogate_staged4'))
"
```

## 6. Before trusting a full run

- Test on a handful of examples first if there's an easy way to limit the
  parent set (e.g. a small `--num_shards`/`--shard_id` slice) -- this is
  new, unexecuted code.
- Expect **roughly 4x the pruning wall-clock time per example** versus the
  original single-pass method, since each of the 4 stages independently
  re-optimizes the 4-direction consensus bank (64 steps) from scratch.
- Check the printed per-stage breakdown (`stages` field in each result
  row) to confirm stage lengths and retention percentages look sane before
  assuming the run is producing something worth training on.