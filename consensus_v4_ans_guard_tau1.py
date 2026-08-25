from __future__ import annotations

# ======================================================================
# CONSENSUS-V2 + ANS SAFETY GUARD, tau = 1.0
#
# Hybrid pruning rule:
#   geometry_i = median_r cosine(d_i^-, d_r^*)
#   deltaANS_i = logP(A | R_-i) - logP(A | R)
#   safe_i     = deltaANS_i >= -1.0
#
# If enough safe candidates exist: top-k SAFE by geometry.
# Otherwise: all safe + least answer-damaging unsafe candidates as fallback.
#
# Runs ONLY this new pruning variant, its Llama SFT, dev eval, and report.
# Existing outputs are never overwritten.
# ======================================================================


# ======================================================================
# FOCUSED CONSENSUS-V2 ABLATION
# Reuses existing parents/baselines. Runs ONLY:
#   consensus_v2 pruning -> Llama SFT -> Llama eval -> focused report
#
# V2 score:
#   score_i = median_r cosine(d_i^-, d_r^*)
# where r indexes the 4 independently optimised directions.
# ======================================================================


"""
Harish-500 one-command production pipeline.

Primary comparison:
  ACL JOINT-50 vs Consensus-50 on the SAME 500 filtered parent CoTs.

Harish's latest auxiliary requests are retained:
  Gradient-50 and ANS-50.

Protocol:
  - GSM8K train: first 500 examples (or Gabriel/Siva manifest) for which the
    source model is wrong without CoT and right with CoT.
  - GSM8K dev: test indices 0..499, unfiltered.
  - Round-based pruning: delete 20% of CURRENT remaining rationale tokens per
    round, clipped to exactly 50% retained.
  - Consensus: 4 independently optimised directions, 64 optimisation steps.
  - One source-model compressed-rationale accuracy sanity check per method.
  - Single fixed seed; no path-agreement, warm-start, or seed-sweep diagnostics.
  - SFT/evaluation on Llama and Qwen students from the same base checkpoints.
  - Physical CUDA device 2; Hugging Face caches under /mnt/fast0/dag83.
  - One invocation runs the entire pipeline; per-example pruning and Trainer
    checkpoints make interrupted work resumable.

Speed engineering that does NOT change the scoring objective:
  - batched candidate evaluation with persistent OOM backoff;
  - backbone-only hidden-state scoring for ANS/JOINT;
  - chunked LM-head evaluation only at scored target positions;
  - SDPA/TF32;
  - memory-aware SFT microbatch while keeping effective batch exactly 28.
"""


import contextlib
import dataclasses
from dataclasses import dataclass, asdict
from decimal import Decimal, InvalidOperation
import gc
import inspect
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import time
from typing import Any, Iterable, Optional
from types import SimpleNamespace

import os

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Keep ALL Hugging Face caches on fast0
HF_ROOT = "/mnt/fast0/dag83/huggingface"

os.environ["HF_HOME"] = HF_ROOT
os.environ["HF_HUB_CACHE"] = f"{HF_ROOT}/hub"
os.environ["HF_DATASETS_CACHE"] = f"{HF_ROOT}/datasets"
os.environ["HF_XET_CACHE"] = f"{HF_ROOT}/xet"
os.environ["HF_ASSETS_CACHE"] = f"{HF_ROOT}/assets"

# Authentication
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

# Because Xet has been crashing on your server:
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Lightweight dependencies only at import time. ML stack is lazy-imported.
try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        MofNCompleteColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.panel import Panel
except Exception as exc:
    raise SystemExit(
        "Missing 'rich'. Install requirements.txt first. "
        f"Original error: {exc}"
    )

console = Console()

# --------------------------------------------------------------------------------------
# Fixed protocol
# --------------------------------------------------------------------------------------

MODEL_ALIASES = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    # Explicit instruction-tuned ~8B-class Qwen used by the prior project.
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    # Optional exact 8B Qwen family checkpoint.
    "qwen3": "Qwen/Qwen3-8B",
}

N_TRAIN = 500
N_DEV = 500
RETENTION = 0.50

# Harish: ~15-20% remaining per round. Freeze one value; no speed experiment.
DEFAULT_ROUND_DELETE_FRACTION = 0.20
MIN_ROUND_DELETE_FRACTION = 0.15
MAX_ROUND_DELETE_FRACTION = 0.20

CONSENSUS_DIRECTIONS = 4
CONSENSUS_STEPS = 64

# Primary scientific comparison requested here.
PRIMARY_METHODS = ("joint", "consensus")
CONSENSUS_V2_NAME = "consensus_v2"
CONSENSUS_GUARD_NAME = "consensus_guard_t1"
ANS_GUARD_TAU = 1.0

# Harish's latest requested auxiliary conditions are retained as well.
ALL_PRUNING_METHODS = ("gradient", "ans", "joint", "consensus")

# Reconstructed project values retained from the last notebook.
VECTOR_NORM = 0.519343
VECTOR_LR = 0.05
NEAR_OPTIMAL_LOGP_GAP = 0.05

# Single seed only. No seed sweep.
DEFAULT_SEED = 13

# ======================================================================================
# USER CONFIGURATION — EDIT THIS BLOCK ONLY
# ======================================================================================
#
# There are NO command-line arguments in this version.
# Run:
#
#     python gsm8k_harish500_noargs.py
#
# Change JOB below between runs.
#
# Harish workflow:
#   1) JOB = "all_sequential"  # informational only; main() below always runs every stage
#   2) JOB = "prune_fast"       -> Gradient-50 + ANS-50
#   3) JOB = "sft_gradient"     -> start SFT as soon as Gradient-50 is ready
#   4) JOB = "prune_consensus"  -> run in parallel on another CUDA device 2
#   5) JOB = "sft_remaining"
#   6) JOB = "eval"
#   7) JOB = "report"
#
# Or use JOB = "all_sequential" if you deliberately want everything serially.
#
JOB = "all_sequential"  # informational; main() directly runs the whole pipeline

# Persistent directory shared by ALL jobs.
WORKDIR = "/mnt/fast0/dag83/results_experiment"

# Source/pruning model. "llama" is the project default.
SOURCE_MODEL = "llama"


# Hugging Face token.
# Add it to your shell environment before running:
#     export HF_TOKEN="hf_..."
# The script reads it here and passes it to Hugging Face loaders.
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

# Preferred: exact 500 parent-CoT JSONL from Gabriel/Siva.
# Leave None only if their manifest is unavailable. In that case PREPARE takes
# the first 500 GSM8K-train examples satisfying Harish's filter.
PARENTS_FILE = None
# Example:
# PARENTS_FILE = "/data/gabriel_siva_exact_500.jsonl"

# SFT/evaluation students.
STUDENT_MODELS = "llama,qwen"

# Harish: 15%-20% of CURRENT remaining rationale tokens per pruning round.
# We freeze 20%; this is not a speed experiment.
ROUND_DELETE_FRACTION = 0.20

# One fixed seed only. No seed sweep.
SEED = DEFAULT_SEED

# Parent construction fallback only. These matter ONLY when PARENTS_FILE=None.
# One greedy CoT follows Harish's filter literally and avoids changing the
# eligible population by sampling multiple rationales.
DIRECT_MAX_NEW_TOKENS = 32
COT_MAX_NEW_TOKENS = 512
COT_SAMPLES = 1
COT_TEMPERATURE = 0.0
COT_TOP_P = 1.0

# Candidate-forward batch for pruning.
# The runtime permanently backs this off after an OOM, so one oversized batch does
# not cause repeated OOMs on every pruning round.
CANDIDATE_BATCH = 64

# JOINT/ANS speed optimisation: apply the LM head only to the target positions,
# and do so in small time chunks instead of materialising full-sequence vocabulary logits.
LM_HEAD_TIME_CHUNK = 16

# Sharding, mainly useful for Consensus across several servers/GPUs.
NUM_SHARDS = 1
SHARD_ID = 0

# Evaluation generation settings.
EVAL_BATCH_SIZE = 16
EVAL_MAX_NEW_TOKENS = 384


# Always run the cheap pure protocol/unit checks before the selected job.
RUN_SELF_TEST_FIRST = True

# ======================================================================================
# END USER CONFIGURATION
# ======================================================================================

# Parent construction prompts. These are saved into protocol.json.
DIRECT_PROMPT = """Answer the following maths problem without showing your reasoning.
Return only one final line in exactly this form:
The answer is <number>

Problem:
{question}"""

COT_PROMPT = """Solve the following maths problem step by step.
At the end, write one final line in exactly this form:
The answer is <number>

Problem:
{question}"""

ANSWER_CUE = "\nThe answer is "

# SFT settings inherited from the previous production notebook's single-run setup.
# Harish did not request a hyperparameter sweep, so these remain fixed.
SFT_MAX_STEPS = 1200
SFT_LR = 5e-6
SFT_SCHEDULER = "cosine"
SFT_EFFECTIVE_BATCH = 28
SFT_PER_DEVICE_BATCH = 4
SFT_GRAD_ACCUM = SFT_EFFECTIVE_BATCH // SFT_PER_DEVICE_BATCH  # 7
SFT_LORA_R = 16
SFT_LORA_ALPHA = 32
SFT_LORA_DROPOUT = 0.05
SFT_MAX_LENGTH = 1024

GOLD_RE = re.compile(r"####\s*([^\n]+)")
ANSWER_MARKER_RE = re.compile(
    r"(?i)(?:the\s+answer\s+is|answer\s*:)\s*([-+]?\$?[\d,]+(?:\.\d+)?)"
)
NUMBER_RE = re.compile(r"[-+]?\$?[\d,]+(?:\.\d+)?")

# Lazy ML globals
torch = None
np = None
pd = None
plt = None
load_dataset = None
HFDataset = None
AutoTokenizer = None
AutoModelForCausalLM = None
Trainer = None
TrainingArguments = None
TrainerCallback = None
set_seed = None
get_last_checkpoint = None
LoraConfig = None
get_peft_model = None
PeftModel = None


def require_ml_stack(include_plotting: bool = False) -> None:
    """Import heavy dependencies only for GPU/data commands."""
    global torch, np, pd, plt, load_dataset, HFDataset
    global AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
    global TrainerCallback, set_seed, get_last_checkpoint
    global LoraConfig, get_peft_model, PeftModel

    if torch is not None:
        if include_plotting and plt is None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            plt = _plt
        return

    try:
        import torch as _torch
        import numpy as _np
        import pandas as _pd
        from datasets import load_dataset as _load_dataset, Dataset as _Dataset
        from transformers import (
            AutoTokenizer as _AutoTokenizer,
            AutoModelForCausalLM as _AutoModelForCausalLM,
            Trainer as _Trainer,
            TrainingArguments as _TrainingArguments,
            TrainerCallback as _TrainerCallback,
            set_seed as _set_seed,
        )
        from transformers.trainer_utils import get_last_checkpoint as _get_last_checkpoint
        from peft import (
            LoraConfig as _LoraConfig,
            get_peft_model as _get_peft_model,
            PeftModel as _PeftModel,
        )
    except Exception as exc:
        raise SystemExit(
            "ML dependencies are missing. Run:\n"
            "  pip install -r requirements.txt\n"
            f"Original error: {exc}"
        )

    torch = _torch
    np = _np
    pd = _pd
    load_dataset = _load_dataset
    HFDataset = _Dataset
    AutoTokenizer = _AutoTokenizer
    AutoModelForCausalLM = _AutoModelForCausalLM
    Trainer = _Trainer
    TrainingArguments = _TrainingArguments
    TrainerCallback = _TrainerCallback
    set_seed = _set_seed
    get_last_checkpoint = _get_last_checkpoint
    LoraConfig = _LoraConfig
    get_peft_model = _get_peft_model
    PeftModel = _PeftModel

    if include_plotting:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        plt = _plt


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------

def rich_progress(*, transient: bool = False) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
        refresh_per_second=5,
    )


def resolve_model(name_or_id: str) -> str:
    return MODEL_ALIASES.get(name_or_id, name_or_id)


def safe_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_id)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def rewrite_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def normalize_number(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    s = s.replace("$", "").replace(",", "").strip()
    # Remove a trailing full stop when it is punctuation, not decimal syntax.
    if s.endswith(".") and s.count(".") == 1:
        s = s[:-1]
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    if d == d.to_integral_value():
        return str(d.quantize(Decimal("1")))
    # Canonical non-scientific decimal.
    return format(d.normalize(), "f")


def gold_from_gsm8k(answer_field: str) -> str:
    m = GOLD_RE.search(answer_field)
    if not m:
        raise ValueError(f"Could not parse GSM8K gold answer: {answer_field!r}")
    gold = normalize_number(m.group(1))
    if gold is None:
        raise ValueError(f"Could not normalize GSM8K gold: {m.group(1)!r}")
    return gold


def extract_prediction(text: str) -> Optional[str]:
    matches = list(ANSWER_MARKER_RE.finditer(text))
    if matches:
        return normalize_number(matches[-1].group(1))
    nums = NUMBER_RE.findall(text)
    if not nums:
        return None
    return normalize_number(nums[-1])


def split_cot_text(text: str) -> tuple[Optional[str], Optional[str]]:
    """Require an explicit final answer marker; return (reasoning, prediction)."""
    matches = list(ANSWER_MARKER_RE.finditer(text))
    if not matches:
        return None, extract_prediction(text)
    m = matches[-1]
    reasoning = text[:m.start()].strip()
    pred = normalize_number(m.group(1))
    return reasoning, pred


def strip_token_subsequence(ids: list[int], needle: list[int]) -> list[int]:
    if not needle:
        return ids[:]
    out = ids[:]
    i = 0
    while i <= len(out) - len(needle):
        if out[i:i + len(needle)] == needle:
            del out[i:i + len(needle)]
        else:
            i += 1
    return out


def first_present(row: dict[str, Any], names: list[str], default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


@dataclass
class RunPaths:
    workdir: Path

    @property
    def protocol(self) -> Path:
        return self.workdir / "protocol.json"

    @property
    def parents(self) -> Path:
        return self.workdir / "parents_500.jsonl"

    @property
    def filter_audit(self) -> Path:
        return self.workdir / "filter_audit.jsonl"

    @property
    def dev(self) -> Path:
        return self.workdir / "dev_first500.jsonl"

    def prune_dir(self, method: str) -> Path:
        return ensure_dir(self.workdir / "pruned" / method)

    def prune_shard(self, method: str, shard_id: int, num_shards: int) -> Path:
        return self.prune_dir(method) / f"shard_{shard_id:02d}_of_{num_shards:02d}.jsonl"

    def adapter_dir(self, student_id: str, condition: str) -> Path:
        return ensure_dir(
            self.workdir / "adapters" / safe_slug(student_id) / condition
        )

    def eval_file(self, student_id: str, condition: str) -> Path:
        return ensure_dir(
            self.workdir / "eval" / safe_slug(student_id)
        ) / f"{condition}.json"

    @property
    def report_dir(self) -> Path:
        return ensure_dir(self.workdir / "reports")


# --------------------------------------------------------------------------------------
# GPU/model helpers
# --------------------------------------------------------------------------------------

def check_gpu() -> dict[str, Any]:
    require_ml_stack()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required. This script is configured to use physical CUDA device 2 "
            "via CUDA_VISIBLE_DEVICES=2."
        )

    # Because CUDA_VISIBLE_DEVICES=2 is set before torch import, this is cuda:0
    # inside the process but maps to physical GPU 2 on the server.
    name = torch.cuda.get_device_name(0)
    mem_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    bf16 = torch.cuda.is_bf16_supported()

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass

    info = {
        "physical_cuda_device": 2,
        "visible_cuda_device": 0,
        "name": name,
        "memory_gib": mem_gib,
        "bf16": bf16,
    }

    console.print(
        Panel.fit(
            f"Physical CUDA device: 2\n"
            f"PyTorch-visible device: cuda:0\n"
            f"GPU: {name}\n"
            f"VRAM: {mem_gib:.1f} GiB\n"
            f"BF16: {bf16}",
            title="CUDA",
        )
    )
    return info


def hf_token() -> Optional[str]:
    return HF_TOKEN or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def load_tokenizer(model_id: str):
    require_ml_stack()
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token())
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_causal_lm(model_id: str, *, train: bool = False):
    require_ml_stack()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = dict(
        pretrained_model_name_or_path=model_id,
        token=hf_token(),
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    # Current Transformers uses dtype; older releases use torch_dtype.
    try:
        model = AutoModelForCausalLM.from_pretrained(dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(torch_dtype=dtype, **kwargs)

    model.config.use_cache = not train
    if not train:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def clear_gpu(*objs: Any) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# Chat/generation helpers
# --------------------------------------------------------------------------------------

# Transformers compatibility note:
# apply_chat_template(tokenize=True) changed return behaviour across versions.
# All tokenised chat-template calls explicitly request return_dict=False and
# are additionally normalised by _chat_template_input_ids().


def _chat_template_input_ids(value) -> list[int]:
    """
    Normalise apply_chat_template output across Transformers versions.

    Transformers 5.x may return a dict/BatchEncoding by default when
    tokenize=True, while older versions commonly returned a plain list.
    """
    if isinstance(value, dict) or hasattr(value, "keys"):
        if "input_ids" not in value:
            raise TypeError(
                "apply_chat_template returned a mapping without 'input_ids': "
                f"{type(value)!r}"
            )
        value = value["input_ids"]

    if hasattr(value, "tolist"):
        value = value.tolist()

    if (
        isinstance(value, (list, tuple))
        and value
        and isinstance(value[0], (list, tuple))
    ):
        if len(value) != 1:
            raise ValueError(
                "Expected one chat example, but apply_chat_template returned "
                f"a batch of {len(value)}."
            )
        value = value[0]

    if not isinstance(value, (list, tuple)):
        raise TypeError(
            "Unexpected apply_chat_template output type: "
            f"{type(value)!r}"
        )

    return [int(x) for x in value]


def render_generation_ids(tokenizer, user_text: str) -> list[int]:
    messages = [{"role": "user", "content": user_text}]
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    return _chat_template_input_ids(ids)


def generate_from_prompt(
    model,
    tokenizer,
    user_text: str,
    *,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
    seed: Optional[int] = None,
) -> list[str]:
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    ids = render_generation_ids(tokenizer, user_text)
    x = torch.tensor([ids], device=model.device, dtype=torch.long)

    kwargs = dict(
        input_ids=x,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        kwargs.update(temperature=float(temperature), top_p=float(top_p))

    with torch.inference_mode():
        out = model.generate(**kwargs)

    prefix_len = x.shape[1]
    return [
        tokenizer.decode(seq[prefix_len:], skip_special_tokens=True)
        for seq in out
    ]


# --------------------------------------------------------------------------------------
# Prepare: dev set and 500 filtered parents
# --------------------------------------------------------------------------------------

def protocol_dict(args, source_id: str) -> dict[str, Any]:
    return {
        "source_model": source_id,
        "train_size": N_TRAIN,
        "dev_size": N_DEV,
        "train_rule": "wrong_without_cot_and_right_with_cot",
        "dev_rule": "gsm8k_test_indices_0_through_499_unfiltered",
        "retention": RETENTION,
        "round_delete_fraction": args.round_delete_fraction,
        "consensus_directions": CONSENSUS_DIRECTIONS,
        "consensus_steps": CONSENSUS_STEPS,
        "vector_norm": VECTOR_NORM,
        "vector_lr": VECTOR_LR,
        "near_optimal_logp_gap": NEAR_OPTIMAL_LOGP_GAP,
        "pruning_methods": list(ALL_PRUNING_METHODS),
        "primary_comparison": list(PRIMARY_METHODS),
        "joint_objective": "sum log P(retained_reasoning + answer_completion | question)",
        "joint_scored_completion_includes_answer_cue": True,
        "round_based_pruning": True,
        "direct_prompt": DIRECT_PROMPT,
        "cot_prompt": COT_PROMPT,
        "answer_cue": ANSWER_CUE,
        "parent_generation": {
            "cot_samples": args.cot_samples,
            "cot_temperature": args.cot_temperature,
            "cot_top_p": args.cot_top_p,
            "cot_max_new_tokens": args.cot_max_new_tokens,
            "direct_max_new_tokens": args.direct_max_new_tokens,
        },
        "seed": args.seed,
        "sft": {
            "single_seed_only": True,
            "max_steps": SFT_MAX_STEPS,
            "learning_rate": SFT_LR,
            "scheduler": SFT_SCHEDULER,
            "effective_batch": SFT_EFFECTIVE_BATCH,
            "lora_r": SFT_LORA_R,
            "lora_alpha": SFT_LORA_ALPHA,
            "lora_dropout": SFT_LORA_DROPOUT,
        },
    }


def prepare_dev(paths: RunPaths) -> None:
    require_ml_stack()
    ds = load_dataset("openai/gsm8k", "main", split="test")
    assert len(ds) >= N_DEV
    rows = []
    for i in range(N_DEV):
        row = ds[i]
        rows.append(
            {
                "dataset_idx": i,
                "question": row["question"],
                "gold": gold_from_gsm8k(row["answer"]),
            }
        )
    # Hard guard: first 500, no filtering.
    assert [r["dataset_idx"] for r in rows] == list(range(500))
    rewrite_jsonl(paths.dev, rows)
    console.print(
        f"[green]Dev manifest:[/green] {paths.dev} "
        f"({len(rows)} unfiltered GSM8K test examples, indices 0..499)"
    )


def normalize_external_parent(
    row: dict[str, Any],
    train_ds,
    tokenizer,
) -> dict[str, Any]:
    idx = int(first_present(row, ["dataset_idx", "idx"]))
    if not 0 <= idx < len(train_ds):
        raise ValueError(f"Parent dataset_idx {idx} outside GSM8K train range.")

    official = train_ds[idx]
    question = str(first_present(row, ["question", "q"], official["question"]))
    if question.strip() != official["question"].strip():
        raise ValueError(f"Question mismatch for GSM8K train idx={idx}.")

    gold = normalize_number(
        str(first_present(row, ["gold", "answer"], gold_from_gsm8k(official["answer"])))
    )
    official_gold = gold_from_gsm8k(official["answer"])
    if gold != official_gold:
        raise ValueError(f"Gold mismatch for idx={idx}: {gold} vs {official_gold}")

    reasoning_text = first_present(
        row,
        ["reasoning_text", "cot", "rationale", "teacher_reasoning"],
    )
    reasoning_ids = first_present(row, ["reasoning_ids", "cot_ids"])

    if reasoning_ids is None and reasoning_text is None:
        raise ValueError(
            f"External parent idx={idx} needs reasoning_text/cot/rationale "
            "or reasoning_ids/cot_ids."
        )

    if reasoning_ids is None:
        reasoning_ids = tokenizer.encode(
            str(reasoning_text), add_special_tokens=False
        )
    reasoning_ids = [int(x) for x in reasoning_ids]

    gold_ids = tokenizer.encode(str(gold), add_special_tokens=False)
    reasoning_ids = strip_token_subsequence(reasoning_ids, gold_ids)
    reasoning_text = tokenizer.decode(reasoning_ids, skip_special_tokens=True).strip()

    if len(reasoning_ids) < 10:
        raise ValueError(f"External parent idx={idx} has <10 reasoning tokens.")

    prefix_ids = render_generation_ids(tokenizer, COT_PROMPT.format(question=question))
    cue_ids = tokenizer.encode(ANSWER_CUE, add_special_tokens=False)
    answer_ids = tokenizer.encode(str(gold), add_special_tokens=False)

    return {
        "dataset_idx": idx,
        "question": question,
        "gold": gold,
        "reasoning_text": reasoning_text,
        "reasoning_ids": reasoning_ids,
        "prefix_ids": prefix_ids,
        "cue_ids": cue_ids,
        "answer_ids": answer_ids,
        "parent_tokens": len(reasoning_ids),
        "parent_source": "external_gabriel_siva_manifest",
    }


def import_parent_manifest(paths: RunPaths, parents_file: Path, source_id: str) -> None:
    require_ml_stack()
    tokenizer = load_tokenizer(source_id)
    train_ds = load_dataset("openai/gsm8k", "main", split="train")

    raw = read_jsonl(parents_file)
    if len(raw) != N_TRAIN:
        raise ValueError(
            f"Expected exactly {N_TRAIN} parent rows from Gabriel/Siva; found {len(raw)}."
        )
    rows = [normalize_external_parent(r, train_ds, tokenizer) for r in raw]
    ids = [r["dataset_idx"] for r in rows]
    if len(set(ids)) != N_TRAIN:
        raise ValueError("External parent manifest contains duplicate dataset indices.")

    rewrite_jsonl(paths.parents, rows)
    console.print(
        f"[green]Imported exact parent manifest:[/green] {paths.parents} "
        f"({N_TRAIN} rows)"
    )


def build_filtered_parents(paths: RunPaths, source_id: str, args) -> None:
    require_ml_stack()
    if paths.parents.exists() and len(read_jsonl(paths.parents)) == N_TRAIN:
        console.print(f"[green]Reusing complete parents:[/green] {paths.parents}")
        return

    tokenizer = load_tokenizer(source_id)
    model = load_causal_lm(source_id, train=False)
    train_ds = load_dataset("openai/gsm8k", "main", split="train")

    existing_parents = read_jsonl(paths.parents)
    parent_by_idx = {int(r["dataset_idx"]): r for r in existing_parents}
    audit = read_jsonl(paths.filter_audit)
    processed = {int(r["dataset_idx"]) for r in audit}

    # Resume from the first unprocessed dataset index; audit includes discarded rows.
    start_idx = 0
    while start_idx in processed:
        start_idx += 1

    console.print(
        Panel.fit(
            f"Need {N_TRAIN} training parents.\n"
            "KEEP only: direct answer WRONG + CoT answer RIGHT.\n"
            f"Already kept: {len(parent_by_idx)} | already scanned: {len(processed)}\n"
            "Fallback construction uses the FIRST 500 qualifying train examples "
            "in GSM8K dataset order.",
            title="Harish training filter",
        )
    )

    total_to_scan = len(train_ds)
    with rich_progress() as progress:
        task = progress.add_task(
            f"Filtering GSM8K train | kept {len(parent_by_idx)}/{N_TRAIN}",
            total=total_to_scan,
            completed=start_idx,
        )

        for idx in range(start_idx, len(train_ds)):
            if len(parent_by_idx) >= N_TRAIN:
                break
            row = train_ds[idx]
            question = row["question"]
            gold = gold_from_gsm8k(row["answer"])

            direct_text = generate_from_prompt(
                model,
                tokenizer,
                DIRECT_PROMPT.format(question=question),
                max_new_tokens=args.direct_max_new_tokens,
                do_sample=False,
                num_return_sequences=1,
                seed=args.seed + idx,
            )[0]
            direct_pred = extract_prediction(direct_text)
            direct_correct = direct_pred == gold

            chosen_reasoning = None
            chosen_cot_text = None
            cot_pred = None
            n_correct_cot = 0

            if not direct_correct:
                # Literal default is one deterministic CoT generation.
                # More than one sample changes the eligible population; use only
                # if that is what Gabriel/Siva used.
                do_sample = args.cot_samples > 1 or args.cot_temperature > 0
                cot_texts = generate_from_prompt(
                    model,
                    tokenizer,
                    COT_PROMPT.format(question=question),
                    max_new_tokens=args.cot_max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.cot_temperature,
                    top_p=args.cot_top_p,
                    num_return_sequences=args.cot_samples,
                    seed=args.seed * 1_000_003 + idx,
                )

                correct_candidates = []
                for t in cot_texts:
                    reasoning, pred = split_cot_text(t)
                    if reasoning is not None and pred == gold:
                        correct_candidates.append((reasoning, t, pred))

                n_correct_cot = len(correct_candidates)
                if correct_candidates:
                    chosen_reasoning, chosen_cot_text, cot_pred = correct_candidates[0]

            eligible = (not direct_correct) and chosen_reasoning is not None

            audit_row = {
                "dataset_idx": idx,
                "gold": gold,
                "direct_pred": direct_pred,
                "direct_correct": direct_correct,
                "cot_correct": bool(chosen_reasoning is not None),
                "n_correct_cot_samples": n_correct_cot,
                "eligible": eligible,
            }
            append_jsonl(paths.filter_audit, audit_row)

            if eligible:
                reasoning_ids = tokenizer.encode(
                    chosen_reasoning, add_special_tokens=False
                )
                # Preserve the project's anti-answer-leakage convention.
                gold_ids = tokenizer.encode(str(gold), add_special_tokens=False)
                reasoning_ids = strip_token_subsequence(reasoning_ids, gold_ids)

                if len(reasoning_ids) >= 10:
                    reasoning_text = tokenizer.decode(
                        reasoning_ids, skip_special_tokens=True
                    ).strip()
                    prefix_ids = render_generation_ids(
                        tokenizer, COT_PROMPT.format(question=question)
                    )
                    cue_ids = tokenizer.encode(ANSWER_CUE, add_special_tokens=False)
                    answer_ids = tokenizer.encode(str(gold), add_special_tokens=False)

                    parent = {
                        "dataset_idx": idx,
                        "question": question,
                        "gold": gold,
                        "reasoning_text": reasoning_text,
                        "reasoning_ids": [int(x) for x in reasoning_ids],
                        "prefix_ids": [int(x) for x in prefix_ids],
                        "cue_ids": [int(x) for x in cue_ids],
                        "answer_ids": [int(x) for x in answer_ids],
                        "parent_tokens": len(reasoning_ids),
                        "direct_pred": direct_pred,
                        "direct_correct": False,
                        "cot_pred": cot_pred,
                        "cot_correct": True,
                        "parent_source": "deterministic_harish_filter",
                    }
                    append_jsonl(paths.parents, parent)
                    parent_by_idx[idx] = parent

            progress.update(
                task,
                advance=1,
                description=(
                    f"Filtering GSM8K train | kept {len(parent_by_idx)}/{N_TRAIN}"
                ),
            )

    clear_gpu(model)

    rows = read_jsonl(paths.parents)
    if len(rows) != N_TRAIN:
        raise RuntimeError(
            f"Scanned GSM8K train but found only {len(rows)} qualifying examples "
            f"under the current direct/CoT generation protocol. Do NOT silently "
            "change the filter. Obtain Gabriel/Siva's exact parent manifest or "
            "confirm the generation protocol with Harish."
        )

    # First 500 eligible in order, exactly.
    rows = rows[:N_TRAIN]
    rewrite_jsonl(paths.parents, rows)
    console.print(f"[green]Parents complete:[/green] {paths.parents}")


def cmd_prepare(args) -> None:
    require_ml_stack()
    source_id = resolve_model(args.source_model)
    paths = RunPaths(Path(args.workdir))
    ensure_dir(paths.workdir)

    # Exact Gabriel/Siva manifest import only needs tokenizer/dataset access.
    # Fallback parent construction performs generation and must use the configured CUDA device 2.
    if not args.parents_file:
        check_gpu()

    if not (MIN_ROUND_DELETE_FRACTION <= args.round_delete_fraction <= MAX_ROUND_DELETE_FRACTION):
        raise ValueError(
            f"--round-delete-fraction must stay within Harish's 0.15-0.20 range; "
            f"got {args.round_delete_fraction}."
        )

    write_json(paths.protocol, protocol_dict(args, source_id))
    prepare_dev(paths)

    if args.parents_file:
        import_parent_manifest(paths, Path(args.parents_file), source_id)
    else:
        build_filtered_parents(paths, source_id, args)

    parents = read_jsonl(paths.parents)
    assert len(parents) == N_TRAIN
    assert len({int(r["dataset_idx"]) for r in parents}) == N_TRAIN

    console.print(
        Panel.fit(
            f"TRAIN parents: {len(parents)}\n"
            f"DEV: {len(read_jsonl(paths.dev))} first test examples, unfiltered\n"
            f"Source model: {source_id}",
            title="Prepare complete",
        )
    )


# --------------------------------------------------------------------------------------
# Pruning core
# --------------------------------------------------------------------------------------

class PruningEngine:
    def __init__(self, source_id: str, round_delete_fraction: float, candidate_batch: int):
        require_ml_stack()
        self.source_id = source_id
        self.round_delete_fraction = float(round_delete_fraction)
        self.candidate_batch = int(candidate_batch)
        self.tokenizer = load_tokenizer(source_id)
        self.model = load_causal_lm(source_id, train=False)
        self.backbone = getattr(self.model, "model", None)
        if self.backbone is None:
            raise RuntimeError(
                f"{source_id} does not expose model.model; update backbone selection."
            )
        self.lm_head = self.model.get_output_embeddings()
        self.device = next(self.model.parameters()).device

    def case_parts(self, parent: dict[str, Any], reasoning_ids: list[int]):
        return (
            parent["prefix_ids"],
            reasoning_ids,
            parent["cue_ids"],
            parent["answer_ids"],
        )

    def hard_answer_state(self, parent, reasoning_ids):
        q_ids, r_ids, cue_ids, ans_ids = self.case_parts(parent, reasoning_ids)
        seq = q_ids + r_ids + cue_ids + ans_ids
        x = torch.tensor([seq], device=self.device, dtype=torch.long)
        with torch.inference_mode():
            hidden = self.backbone(
                input_ids=x,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        answer_start = len(q_ids) + len(r_ids) + len(cue_ids)
        A = len(ans_ids)
        return hidden[:, answer_start - 1:answer_start - 1 + A, :][0].detach().float()

    def _candidate_chunk(self, parent, chunk):
        q_ids, _, cue_ids, ans_ids = self.case_parts(parent, chunk[0])
        seqs = [q_ids + list(r) + cue_ids + ans_ids for r in chunk]
        x = torch.tensor(seqs, device=self.device, dtype=torch.long)
        with torch.inference_mode():
            hidden = self.backbone(
                input_ids=x,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        rlen = len(chunk[0])
        answer_start = len(q_ids) + rlen + len(cue_ids)
        A = len(ans_ids)
        return hidden[:, answer_start - 1:answer_start - 1 + A, :].detach().float()

    def _run_adaptive_batches(self, items, fn, *, label: str):
        """
        Run equal-shape candidate items in batches.

        Engineering-only speed/memory optimisation:
        - start from configured candidate_batch;
        - if CUDA OOM occurs, halve the batch and retry;
        - remember the smaller working batch for later rounds/methods.

        This does not change candidate scores, only how many are evaluated together.
        """
        if not items:
            raise ValueError(f"{label}: empty candidate list")

        outputs = []
        pos = 0
        batch = max(1, min(int(self.candidate_batch), len(items)))

        while pos < len(items):
            chunk = items[pos:pos + batch]
            try:
                outputs.append(fn(chunk))
                pos += len(chunk)
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if batch <= 1:
                    raise
                new_batch = max(1, batch // 2)
                self.candidate_batch = min(int(self.candidate_batch), new_batch)
                batch = new_batch
                console.print(
                    f"[yellow]{label}: CUDA OOM; reducing candidate batch "
                    f"permanently to {batch} and retrying.[/yellow]"
                )

        return torch.cat(outputs, dim=0)

    def candidate_displacements(self, parent, current_ids, H_base):
        candidates = [
            current_ids[:i] + current_ids[i + 1:]
            for i in range(len(current_ids))
        ]
        H_cands = self._run_adaptive_batches(
            candidates,
            lambda chunk: self._candidate_chunk(parent, chunk),
            label="vector candidate states",
        )
        return (H_cands - H_base.unsqueeze(0)).reshape(len(candidates), -1)

    def soft_answer_forward(self, parent, reasoning_ids, E_float32):
        q_ids, r_ids, cue_ids, ans_ids = self.case_parts(parent, reasoning_ids)
        context_ids = q_ids + r_ids + cue_ids

        context = torch.tensor(context_ids, device=self.device, dtype=torch.long)
        answer = torch.tensor(ans_ids, device=self.device, dtype=torch.long)

        R = E_float32.shape[0]
        embed = self.model.get_input_embeddings()
        ctx_emb = embed(context).unsqueeze(0).expand(R, -1, -1)
        ans_emb = embed(answer).unsqueeze(0).expand(R, -1, -1)
        e_model = E_float32.to(dtype=ctx_emb.dtype).unsqueeze(1)
        inputs_embeds = torch.cat([ctx_emb, e_model, ans_emb], dim=1)

        hidden = self.backbone(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

        answer_start = len(context_ids) + 1
        A = len(ans_ids)
        pred_h = hidden[:, answer_start - 1:answer_start - 1 + A, :]

        logits = self.lm_head(
            pred_h.to(dtype=self.lm_head.weight.dtype)
        ).float()
        target = answer.view(1, A).expand(R, -1)
        gold_logits = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        seq_logp = (
            gold_logits - torch.logsumexp(logits, dim=-1)
        ).sum(dim=-1)
        return seq_logp, pred_h.float()

    def fresh_bank(self, seed: int, bank_size: int = CONSENSUS_DIRECTIONS):
        d_model = self.model.get_input_embeddings().embedding_dim
        rows = []
        for r in range(bank_size):
            g = torch.Generator(device=self.device)
            g.manual_seed(int(seed + 1009 * r))
            e = torch.randn(
                d_model,
                generator=g,
                device=self.device,
                dtype=torch.float32,
            )
            e = VECTOR_NORM * e / e.norm().clamp_min(1e-12)
            rows.append(e)
        return torch.stack(rows, dim=0)

    def optimise_consensus_bank(self, parent, reasoning_ids, seed):
        E = self.fresh_bank(seed, CONSENSUS_DIRECTIONS)
        best_E = E.detach().clone()
        best_logp = torch.full(
            (CONSENSUS_DIRECTIONS,),
            -float("inf"),
            device=self.device,
            dtype=torch.float32,
        )

        for _ in range(CONSENSUS_STEPS):
            E = E.detach().requires_grad_(True)
            logp, _ = self.soft_answer_forward(parent, reasoning_ids, E)

            with torch.no_grad():
                improved = logp > best_logp
                best_logp = torch.where(improved, logp.detach(), best_logp)
                best_E[improved] = E.detach()[improved]

            grad = torch.autograd.grad(logp.sum(), E, only_inputs=True)[0]

            with torch.no_grad():
                E = E + VECTOR_LR * grad.float()
                E = VECTOR_NORM * E / E.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)

        with torch.no_grad():
            final_logp, _ = self.soft_answer_forward(parent, reasoning_ids, E)
            improved = final_logp > best_logp
            best_logp = torch.where(improved, final_logp, best_logp)
            best_E[improved] = E.detach()[improved]

        return best_E.detach()

    def consensus_direction(self, parent, reasoning_ids, H_base, seed):
        bank_E = self.optimise_consensus_bank(parent, reasoning_ids, seed)
        with torch.no_grad():
            logp, hidden = self.soft_answer_forward(parent, reasoning_ids, bank_E)

        best_logp = float(logp.max().detach().cpu())
        near_idx = torch.where(
            logp >= best_logp - NEAR_OPTIMAL_LOGP_GAP
        )[0].detach().cpu().tolist()

        dirs = []
        for r in near_idx:
            d = (hidden[r] - H_base).reshape(-1)
            dirs.append(d / d.norm().clamp_min(1e-12))
        return torch.stack(dirs, dim=0).mean(dim=0).detach()

    def consensus_v2_directions(self, parent, reasoning_ids, H_base, seed):
        """
        Improved Consensus variant.

        The original Consensus averaged the independently optimised hidden-state
        directions first and then projected raw deletion displacements onto that
        average. That can (a) cancel distinct useful directions and (b) reward
        large-magnitude deletion effects.

        V2 keeps the 4 independently optimised directions separate. Each direction
        is converted into a unit hidden-state displacement. Candidate deletions are
        also normalised, then scored by cosine similarity to EACH direction. The
        final score is the median across the 4 similarities.

        Everything else is unchanged:
          - same 4 independent optimisation starts
          - same 64 optimisation steps
          - same vector norm/LR
          - same 20%-of-current round deletion
          - same final 50% retention
          - same source-model sanity check
        """
        bank_E = self.optimise_consensus_bank(parent, reasoning_ids, seed)
        with torch.no_grad():
            _, hidden = self.soft_answer_forward(parent, reasoning_ids, bank_E)

        dirs = (hidden - H_base.unsqueeze(0)).reshape(
            CONSENSUS_DIRECTIONS, -1
        ).float()
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return dirs.detach()

    @staticmethod
    def consensus_v2_scores(d_flat, directions):
        """
        Robust directional agreement:
            score_i = median_r cosine(d_i^-, d_r^*)

        Higher score => the deletion moves the answer-state representation in a
        direction consistently aligned with the independently optimised
        answer-improving interventions.
        """
        d = d_flat.float()
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        dirs = directions.to(device=d.device, dtype=d.dtype)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        cosine = d @ dirs.T  # [num_candidates, 4]
        return cosine.median(dim=1).values

    def gradient_direction(self, parent, reasoning_ids, H_base):
        d_model = self.model.get_input_embeddings().embedding_dim
        E0 = torch.zeros(1, d_model, device=self.device, dtype=torch.float32)
        E0 = E0.detach().requires_grad_(True)

        logp0, _ = self.soft_answer_forward(parent, reasoning_ids, E0)
        grad = torch.autograd.grad(logp0.sum(), E0, only_inputs=True)[0].detach().float()
        if float(grad.norm().detach().cpu()) < 1e-12:
            raise RuntimeError("Gradient norm is effectively zero.")

        E_grad = VECTOR_NORM * grad / grad.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

        with torch.no_grad():
            _, hidden_grad = self.soft_answer_forward(parent, reasoning_ids, E_grad)

        d = (hidden_grad[0] - H_base).reshape(-1)
        return (d / d.norm().clamp_min(1e-12)).detach()

    @staticmethod
    def scores_from_displacements(d_flat, direction):
        direction = direction.reshape(-1).to(
            device=d_flat.device, dtype=d_flat.dtype
        )
        denom = direction.pow(2).sum().clamp_min(1e-12)
        return (d_flat @ direction) / denom

    def _gold_logp_from_pred_hidden(self, pred_h, target_ids):
        """
        Exact teacher-forced summed gold log-probability, but without materialising
        vocabulary logits for irrelevant sequence positions.

        pred_h: [B, T, H]
        target_ids: [B, T]
        """
        B, T, _ = pred_h.shape
        sums = torch.zeros(B, device=pred_h.device, dtype=torch.float32)
        head_dtype = self.lm_head.weight.dtype

        for s in range(0, T, LM_HEAD_TIME_CHUNK):
            e = min(T, s + LM_HEAD_TIME_CHUNK)
            logits = self.lm_head(
                pred_h[:, s:e, :].to(dtype=head_dtype)
            ).float()
            gold = target_ids[:, s:e]
            gold_logits = logits.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
            sums += (
                gold_logits - torch.logsumexp(logits, dim=-1)
            ).sum(dim=-1)

        return sums

    def _ans_scores_chunk(self, parent, candidates):
        """
        ACL ANS baseline:
            log P(answer | question, retained reasoning)

        Only answer prediction states are sent through the LM head.
        """
        q_ids = parent["prefix_ids"]
        cue_ids = parent["cue_ids"]
        ans_ids = parent["answer_ids"]

        seqs = [q_ids + list(r) + cue_ids + ans_ids for r in candidates]
        x = torch.tensor(seqs, device=self.device, dtype=torch.long)

        with torch.inference_mode():
            hidden = self.backbone(
                input_ids=x,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state

        answer_start = len(q_ids) + len(candidates[0]) + len(cue_ids)
        A = len(ans_ids)
        pred_h = hidden[:, answer_start - 1:answer_start - 1 + A, :]
        target = torch.tensor(
            ans_ids, device=self.device, dtype=torch.long
        ).view(1, -1).expand(len(candidates), -1)

        return self._gold_logp_from_pred_hidden(pred_h, target).detach()

    def ans_scores(self, parent, current_ids):
        candidates = [
            current_ids[:i] + current_ids[i + 1:]
            for i in range(len(current_ids))
        ]
        return self._run_adaptive_batches(
            candidates,
            lambda chunk: self._ans_scores_chunk(parent, chunk),
            label="ANS candidate scoring",
        )


    def current_answer_logp(self, parent, current_ids) -> torch.Tensor:
        """
        Teacher-forced gold-answer log probability for the CURRENT rationale,
        using the same ANS objective as the candidate deletion scores.
        """
        return self._ans_scores_chunk(parent, [current_ids])[0].detach()

    def _joint_scores_chunk(self, parent, candidates):
        """
        ACL JOINT objective adapted to the script's answer formatting:

            sum log P(retained_reasoning + answer_completion | question)

        The answer completion includes the fixed ANSWER_CUE plus the gold answer.
        Thus the scored target is:
            retained reasoning + "\\nThe answer is " + gold

        All candidates in one pruning round have equal target length, so summed
        teacher-forced log-probability gives a valid within-round ranking.
        """
        q_ids = parent["prefix_ids"]
        cue_ids = parent["cue_ids"]
        ans_ids = parent["answer_ids"]

        targets = [list(r) + cue_ids + ans_ids for r in candidates]
        seqs = [q_ids + target for target in targets]

        x = torch.tensor(seqs, device=self.device, dtype=torch.long)
        target = torch.tensor(targets, device=self.device, dtype=torch.long)

        with torch.inference_mode():
            hidden = self.backbone(
                input_ids=x,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state

        p = len(q_ids)
        T = target.shape[1]
        pred_h = hidden[:, p - 1:p - 1 + T, :]

        return self._gold_logp_from_pred_hidden(pred_h, target).detach()

    def joint_scores(self, parent, current_ids):
        candidates = [
            current_ids[:i] + current_ids[i + 1:]
            for i in range(len(current_ids))
        ]
        return self._run_adaptive_batches(
            candidates,
            lambda chunk: self._joint_scores_chunk(parent, chunk),
            label="JOINT candidate scoring",
        )

    def source_compressed_correct(self, parent, retained_ids) -> tuple[bool, Optional[str], str]:
        prefix = (
            parent["prefix_ids"]
            + list(retained_ids)
            + parent["cue_ids"]
        )
        x = torch.tensor([prefix], device=self.device, dtype=torch.long)
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=x,
                do_sample=False,
                max_new_tokens=24,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            out[0, x.shape[1]:], skip_special_tokens=True
        )
        pred = extract_prediction(text)
        return pred == parent["gold"], pred, text

    def prune_one(self, parent: dict[str, Any], method: str, seed: int) -> dict[str, Any]:
        assert method in {"gradient", "consensus", "consensus_v2", "consensus_guard_t1", "ans", "joint"}

        current_ids = [int(x) for x in parent["reasoning_ids"]]
        current_orig = list(range(len(current_ids)))
        parent_n = len(current_ids)
        target_n = max(1, math.ceil(parent_n * RETENTION))

        rounds = []
        round_id = 0
        t0 = time.perf_counter()

        while len(current_ids) > target_n:
            n_before = len(current_ids)
            chosen = None
            round_extra = {}
            k = min(
                n_before - target_n,
                max(1, math.ceil(self.round_delete_fraction * n_before)),
            )

            round_t0 = time.perf_counter()

            if method == "ans":
                scores = self.ans_scores(parent, current_ids)
            elif method == "joint":
                # Harish speed setting: rank all one-token deletions by the ACL
                # JOINT objective, then delete the top ~20% together this round.
                # This is the round-based scalable approximation, not exact
                # one-token-at-a-time greedy recomputation.
                scores = self.joint_scores(parent, current_ids)
            else:
                H_base = self.hard_answer_state(parent, current_ids)

                direction_seed = (
                    seed
                    + int(parent["dataset_idx"]) * 100_003
                    + round_id * 101
                )

                if method == "gradient":
                    direction = self.gradient_direction(
                        parent, current_ids, H_base
                    )
                    d_flat = self.candidate_displacements(
                        parent, current_ids, H_base
                    )
                    scores = self.scores_from_displacements(
                        d_flat, direction
                    )
                elif method == "consensus":
                    direction = self.consensus_direction(
                        parent,
                        current_ids,
                        H_base,
                        seed=direction_seed,
                    )
                    d_flat = self.candidate_displacements(
                        parent, current_ids, H_base
                    )
                    scores = self.scores_from_displacements(
                        d_flat, direction
                    )
                elif method == "consensus_v2":
                    directions = self.consensus_v2_directions(
                        parent,
                        current_ids,
                        H_base,
                        seed=direction_seed,
                    )
                    d_flat = self.candidate_displacements(
                        parent, current_ids, H_base
                    )
                    scores = self.consensus_v2_scores(
                        d_flat, directions
                    )

                else:  # consensus_guard_t1
                    # Geometry is EXACTLY Consensus-v2.
                    directions = self.consensus_v2_directions(
                        parent,
                        current_ids,
                        H_base,
                        seed=direction_seed,
                    )
                    d_flat = self.candidate_displacements(
                        parent, current_ids, H_base
                    )
                    geom_scores = self.consensus_v2_scores(
                        d_flat, directions
                    )

                    # ANS is used ONLY as a deletion-safety guard.
                    candidate_ans = self.ans_scores(parent, current_ids)
                    base_ans = self.current_answer_logp(parent, current_ids)
                    delta_ans = candidate_ans - base_ans

                    safe = delta_ans >= (-ANS_GUARD_TAU)
                    safe_idx = torch.nonzero(
                        safe, as_tuple=False
                    ).flatten()
                    n_safe = int(safe_idx.numel())

                    if n_safe >= k:
                        # Enough safe choices: ranking remains purely Consensus.
                        guarded_scores = geom_scores.masked_fill(
                            ~safe, float("-inf")
                        )
                        chosen = torch.topk(
                            guarded_scores, k=k
                        ).indices.detach().cpu().tolist()
                        fallback_count = 0
                    else:
                        # Preserve exact round budget. Keep every safe candidate,
                        # then fill only the unavoidable remainder with the
                        # least answer-damaging unsafe deletions.
                        chosen_tensors = []
                        if n_safe > 0:
                            # All safe candidates must be taken when n_safe < k.
                            chosen_tensors.append(safe_idx)

                        need = k - n_safe
                        unsafe_idx = torch.nonzero(
                            ~safe, as_tuple=False
                        ).flatten()

                        if need > 0:
                            unsafe_ans = candidate_ans[unsafe_idx]
                            best_unsafe_local = torch.topk(
                                unsafe_ans, k=need
                            ).indices
                            fill_idx = unsafe_idx[best_unsafe_local]
                            chosen_tensors.append(fill_idx)

                        chosen = torch.cat(
                            chosen_tensors
                        ).detach().cpu().tolist()
                        fallback_count = need

                    # Record diagnostics, but these do NOT alter SFT filtering.
                    round_extra = {
                        "ans_guard_tau": float(ANS_GUARD_TAU),
                        "base_answer_logp": float(base_ans.detach().cpu()),
                        "safe_candidates": n_safe,
                        "candidate_count": int(len(current_ids)),
                        "guard_fallback_deletions": int(fallback_count),
                        "mean_candidate_delta_ans": float(
                            delta_ans.float().mean().detach().cpu()
                        ),
                        "min_candidate_delta_ans": float(
                            delta_ans.float().min().detach().cpu()
                        ),
                        "max_candidate_delta_ans": float(
                            delta_ans.float().max().detach().cpu()
                        ),
                    }

            if chosen is None:
                chosen = torch.topk(
                    scores, k=k
                ).indices.detach().cpu().tolist()
            drop = set(int(x) for x in chosen)
            deleted_orig = [current_orig[i] for i in chosen]

            current_ids = [
                tok for i, tok in enumerate(current_ids)
                if i not in drop
            ]
            current_orig = [
                pos for i, pos in enumerate(current_orig)
                if i not in drop
            ]

            rounds.append(
                {
                    "round": round_id,
                    "tokens_before": n_before,
                    "deleted": k,
                    "tokens_after": len(current_ids),
                    "retention_after": len(current_ids) / parent_n,
                    "round_seconds": time.perf_counter() - round_t0,
                    "deleted_original_positions": sorted(int(x) for x in deleted_orig),
                    **round_extra,
                }
            )
            round_id += 1

        # Harish's ONE requested sanity check. It does not filter results.
        correct, pred, raw = self.source_compressed_correct(parent, current_ids)

        return {
            "dataset_idx": int(parent["dataset_idx"]),
            "question": parent["question"],
            "gold": parent["gold"],
            "method": method,
            "parent_tokens": parent_n,
            "target_tokens": target_n,
            "retained_tokens": len(current_ids),
            "retention": len(current_ids) / parent_n,
            "token_reduction_pct": 100.0 * (1.0 - len(current_ids) / parent_n),
            "retained_ids": [int(x) for x in current_ids],
            "retained_positions": [int(x) for x in current_orig],
            "retained_text": self.tokenizer.decode(
                current_ids, skip_special_tokens=True
            ).strip(),
            "round_delete_fraction": self.round_delete_fraction,
            "rounds": rounds,
            "n_rounds": len(rounds),
            "prune_seconds": time.perf_counter() - t0,
            "source_compressed_correct": bool(correct),
            "source_compressed_pred": pred,
            "source_compressed_raw": raw,
        }


def load_protocol(paths: RunPaths) -> dict[str, Any]:
    if not paths.protocol.exists():
        raise FileNotFoundError(
            f"{paths.protocol} missing. Run 'prepare' first."
        )
    return json.loads(paths.protocol.read_text(encoding="utf-8"))


def validate_parent_ids(paths: RunPaths) -> list[dict[str, Any]]:
    parents = read_jsonl(paths.parents)
    if len(parents) != N_TRAIN:
        raise RuntimeError(
            f"Need exactly {N_TRAIN} fixed parents; found {len(parents)}."
        )
    ids = [int(r["dataset_idx"]) for r in parents]
    if len(set(ids)) != N_TRAIN:
        raise RuntimeError("Parent manifest has duplicates.")
    return parents


def cmd_prune(args) -> None:
    require_ml_stack()
    check_gpu()

    paths = RunPaths(Path(args.workdir))
    protocol = load_protocol(paths)
    source_id = resolve_model(args.source_model)

    if protocol["source_model"] != source_id:
        raise RuntimeError(
            f"Workdir was prepared with source_model={protocol['source_model']}, "
            f"but prune requested {source_id}."
        )

    round_frac = float(protocol["round_delete_fraction"])
    if not MIN_ROUND_DELETE_FRACTION <= round_frac <= MAX_ROUND_DELETE_FRACTION:
        raise RuntimeError("Saved round fraction violates Harish protocol.")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = {"gradient", "ans", "joint", "consensus", "consensus_v2", "consensus_guard_t1"}
    bad = set(methods) - allowed
    if bad:
        raise ValueError(f"Unknown methods: {sorted(bad)}")

    parents = validate_parent_ids(paths)
    assigned = [
        p for j, p in enumerate(parents)
        if j % args.num_shards == args.shard_id
    ]

    engine = PruningEngine(
        source_id=source_id,
        round_delete_fraction=round_frac,
        candidate_batch=args.candidate_batch,
    )

    for method in methods:
        out_path = paths.prune_shard(method, args.shard_id, args.num_shards)
        existing = read_jsonl(out_path)
        done = {int(r["dataset_idx"]) for r in existing}

        todo = [p for p in assigned if int(p["dataset_idx"]) not in done]
        console.print(
            Panel.fit(
                f"Method: {method}\nAssigned: {len(assigned)}\n"
                f"Already complete: {len(done)}\nRemaining: {len(todo)}\n"
                f"Retention: 50%\nRound deletion: {round_frac:.0%} of remaining\n"
                + (
                    "Consensus-v2 + ANS guard: median cosine geometry; reject deletions with ΔlogP(answer) < -1.0 when the round budget allows"
                    if method == "consensus_guard_t1"
                    else "Consensus-v2: median cosine agreement across 4 independently optimised directions × 64 steps"
                    if method == "consensus_v2"
                    else "Consensus: 4 independently optimised directions × 64 steps"
                    if method == "consensus"
                    else "Gradient-only: raw gradient, zero optimisation steps"
                    if method == "gradient"
                    else "ACL JOINT: retained CoT + answer-completion likelihood"
                    if method == "joint"
                    else "ANS: post-deletion P(gold answer)"
                ),
                title="Pruning",
            )
        )

        with rich_progress() as progress:
            task = progress.add_task(
                f"{method} | examples",
                total=len(todo),
            )
            for parent in todo:
                idx = int(parent["dataset_idx"])
                progress.update(task, description=f"{method} | GSM8K train idx={idx}")
                result = engine.prune_one(parent, method, args.seed)
                append_jsonl(out_path, result)
                progress.advance(task)

        console.print(f"[green]{method} shard complete:[/green] {out_path}")

    clear_gpu(engine.model)


def load_complete_pruned(paths: RunPaths, method: str) -> list[dict[str, Any]]:
    files = sorted(paths.prune_dir(method).glob("shard_*_of_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No pruning shards found for {method}.")
    merged = {}
    for f in files:
        for row in read_jsonl(f):
            idx = int(row["dataset_idx"])
            if idx in merged:
                # Exact duplicate from an accidental rerun is okay only if same method.
                continue
            merged[idx] = row

    parents = validate_parent_ids(paths)
    parent_ids = {int(r["dataset_idx"]) for r in parents}
    if set(merged) != parent_ids:
        missing = sorted(parent_ids - set(merged))
        extra = sorted(set(merged) - parent_ids)
        raise RuntimeError(
            f"{method} is not a complete paired 500-set. "
            f"Missing={len(missing)} extra={len(extra)}. "
            f"First missing={missing[:10]}"
        )

    # Preserve exact parent order.
    return [merged[int(p["dataset_idx"])] for p in parents]


# --------------------------------------------------------------------------------------
# SFT
# --------------------------------------------------------------------------------------

class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        m = self.pad_to_multiple_of
        max_len = int(math.ceil(max_len / m) * m)

        batch_ids, batch_mask, batch_labels = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            batch_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            batch_mask.append(f["attention_mask"] + [0] * pad)
            batch_labels.append(f["labels"] + [-100] * pad)

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def tokenise_sft_row(tokenizer, row: dict[str, Any]) -> dict[str, Any]:
    user = COT_PROMPT.format(question=row["question"])
    assistant = row["retained_text"].strip() + ANSWER_CUE + str(row["gold"])

    prompt_messages = [{"role": "user", "content": user}]
    full_messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
    )
    prompt_ids = _chat_template_input_ids(prompt_ids)
    full_ids = _chat_template_input_ids(full_ids)

    boundary = common_prefix_len(prompt_ids, full_ids)
    drift = len(prompt_ids) - boundary
    if boundary == 0 or drift > 6:
        raise RuntimeError(
            f"Unexpected chat-template prompt boundary for idx={row['dataset_idx']} "
            f"(boundary={boundary}, drift={drift})."
        )
    if len(full_ids) > SFT_MAX_LENGTH:
        raise RuntimeError(
            f"SFT example idx={row['dataset_idx']} has {len(full_ids)} tokens > "
            f"SFT_MAX_LENGTH={SFT_MAX_LENGTH}. Raise the explicit limit rather "
            "than silently truncate."
        )

    labels = [-100] * boundary + [int(x) for x in full_ids[boundary:]]
    return {
        "input_ids": [int(x) for x in full_ids],
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def _sft_runtime_batching() -> tuple[int, int, bool]:
    """
    Throughput-only tuning. Effective batch remains exactly 28.

    Uses larger microbatches when memory permits, otherwise falls back safely.
    Gradient checkpointing is disabled only when there is ample memory because
    it costs speed; the scientific SFT settings remain unchanged.
    """
    mem_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    if mem_gib >= 70:
        micro = 7
        gradient_checkpointing = False
    elif mem_gib >= 40:
        micro = 4
        gradient_checkpointing = True
    elif mem_gib >= 24:
        micro = 2
        gradient_checkpointing = True
    else:
        micro = 1
        gradient_checkpointing = True

    accum = SFT_EFFECTIVE_BATCH // micro
    assert micro * accum == SFT_EFFECTIVE_BATCH
    return micro, accum, gradient_checkpointing


def make_training_args(output_dir: Path, seed: int):
    micro, accum, use_gc = _sft_runtime_batching()
    kwargs = dict(
        output_dir=str(output_dir),
        seed=int(seed),
        data_seed=int(seed),
        max_steps=SFT_MAX_STEPS,
        learning_rate=SFT_LR,
        lr_scheduler_type=SFT_SCHEDULER,
        per_device_train_batch_size=micro,
        gradient_accumulation_steps=accum,
        optim="adamw_torch_fused",
        weight_decay=0.0,
        warmup_steps=0,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=bool(torch.cuda.is_bf16_supported()),
        fp16=not bool(torch.cuda.is_bf16_supported()),
        tf32=True,
        gradient_checkpointing=use_gc,
        remove_unused_columns=False,
        report_to="none",
        disable_tqdm=True,  # Rich callback below replaces tqdm.
    )
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "no"
    return TrainingArguments(**kwargs)


def make_rich_trainer_callback(total_steps: int):
    class RichTrainingCallback(TrainerCallback):
        def __init__(self):
            self.progress = None
            self.task = None

        def on_train_begin(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                self.progress = rich_progress()
                self.progress.start()
                self.task = self.progress.add_task(
                    "SFT training",
                    total=total_steps,
                    completed=int(state.global_step),
                )

        def on_step_end(self, args, state, control, **kwargs):
            if self.progress is not None and self.task is not None:
                self.progress.update(
                    self.task,
                    completed=int(state.global_step),
                    description=f"SFT training | step {state.global_step}/{total_steps}",
                )

        def on_train_end(self, args, state, control, **kwargs):
            if self.progress is not None:
                self.progress.stop()

    return RichTrainingCallback()


def train_one_adapter(
    student_id: str,
    condition: str,
    rows: list[dict[str, Any]],
    paths: RunPaths,
    seed: int,
) -> Path:
    require_ml_stack()
    tokenizer = load_tokenizer(student_id)
    tokenised = [tokenise_sft_row(tokenizer, r) for r in rows]
    ds = HFDataset.from_list(tokenised)
    collator = CompletionOnlyCollator(tokenizer.pad_token_id)

    out_dir = paths.adapter_dir(student_id, condition)
    final_marker = out_dir / "FINAL_COMPLETE.txt"
    if final_marker.exists() and (out_dir / "adapter_config.json").exists():
        console.print(f"[green]Reusing completed adapter:[/green] {out_dir}")
        return out_dir

    set_seed(seed)
    model = load_causal_lm(student_id, train=True)
    model.config.use_cache = False

    lora = LoraConfig(
        r=SFT_LORA_R,
        lora_alpha=SFT_LORA_ALPHA,
        lora_dropout=SFT_LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora)

    micro, accum, use_gc = _sft_runtime_batching()
    console.print(
        f"[cyan]SFT throughput:[/cyan] microbatch={micro}, accumulation={accum}, "
        f"effective_batch={micro * accum}, gradient_checkpointing={use_gc}"
    )

    if use_gc:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    elif hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    args = make_training_args(out_dir, seed)
    callback = make_rich_trainer_callback(SFT_MAX_STEPS)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        callbacks=[callback],
    )

    last_ckpt = get_last_checkpoint(str(out_dir))
    if last_ckpt:
        console.print(f"[yellow]Resuming checkpoint:[/yellow] {last_ckpt}")

    trainer.train(resume_from_checkpoint=last_ckpt if last_ckpt else None)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    final_marker.write_text("complete\n", encoding="utf-8")

    del trainer, model
    clear_gpu()
    return out_dir


def cmd_sft(args) -> None:
    require_ml_stack()
    check_gpu()

    paths = RunPaths(Path(args.workdir))
    load_protocol(paths)
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    students = [resolve_model(x.strip()) for x in args.student_models.split(",") if x.strip()]

    for condition in conditions:
        if condition not in {"gradient", "ans", "joint", "consensus", "consensus_v2", "consensus_guard_t1"}:
            raise ValueError(condition)
        rows = load_complete_pruned(paths, condition)
        assert len(rows) == N_TRAIN

        for student_id in students:
            console.print(
                Panel.fit(
                    f"Student: {student_id}\nCondition: {condition}\n"
                    f"Training examples: {len(rows)}\n"
                    f"Single seed: {args.seed}",
                    title="SFT",
                )
            )
            train_one_adapter(student_id, condition, rows, paths, args.seed)


# --------------------------------------------------------------------------------------
# Evaluation: first 500 test examples, unfiltered
# --------------------------------------------------------------------------------------

def batch_generate_eval(
    model,
    tokenizer,
    dev_rows: list[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    tokenizer.padding_side = "left"
    results = []

    with rich_progress() as progress:
        task = progress.add_task("Dev evaluation", total=len(dev_rows))

        for start in range(0, len(dev_rows), batch_size):
            chunk = dev_rows[start:start + batch_size]
            rendered = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": COT_PROMPT.format(question=r["question"])}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for r in chunk
            ]
            enc = tokenizer(
                rendered,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            input_width = enc["input_ids"].shape[1]

            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            for i, row in enumerate(chunk):
                text = tokenizer.decode(
                    out[i, input_width:], skip_special_tokens=True
                )
                pred = extract_prediction(text)
                results.append(
                    {
                        "dataset_idx": int(row["dataset_idx"]),
                        "gold": row["gold"],
                        "pred": pred,
                        "correct": pred == row["gold"],
                        "raw": text,
                    }
                )

            progress.update(task, advance=len(chunk))

    return results


def eval_one(
    student_id: str,
    condition: str,
    paths: RunPaths,
    dev_rows: list[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    require_ml_stack()
    out_file = paths.eval_file(student_id, condition)
    if out_file.exists():
        return json.loads(out_file.read_text(encoding="utf-8"))

    tokenizer = load_tokenizer(student_id)
    base = load_causal_lm(student_id, train=False)

    if condition == "base":
        model = base
        adapter_path = None
    else:
        adapter_path = paths.adapter_dir(student_id, condition)
        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Adapter missing for {student_id} / {condition}: {adapter_path}"
            )
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()

    rows = batch_generate_eval(
        model,
        tokenizer,
        dev_rows,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    acc = sum(bool(r["correct"]) for r in rows) / len(rows)
    result = {
        "student_model": student_id,
        "condition": condition,
        "n": len(rows),
        "accuracy": acc,
        "correct": int(sum(bool(r["correct"]) for r in rows)),
        "dev_rule": "first_500_gsm8k_test_unfiltered",
        "predictions": rows,
    }
    write_json(out_file, result)
    clear_gpu(model, base)
    return result


def cmd_eval(args) -> None:
    require_ml_stack()
    check_gpu()

    paths = RunPaths(Path(args.workdir))
    dev_rows = read_jsonl(paths.dev)
    if len(dev_rows) != N_DEV or [int(r["dataset_idx"]) for r in dev_rows] != list(range(500)):
        raise RuntimeError(
            "Dev manifest must be exactly the first 500 GSM8K test examples, unfiltered."
        )

    students = [resolve_model(x.strip()) for x in args.student_models.split(",") if x.strip()]
    conditions = ["base"] + [
        x.strip() for x in args.conditions.split(",") if x.strip()
    ]

    for student_id in students:
        for condition in conditions:
            console.print(
                Panel.fit(
                    f"Student: {student_id}\nCondition: {condition}\n"
                    "Dev: GSM8K test 0..499, unfiltered",
                    title="Evaluation",
                )
            )
            result = eval_one(
                student_id,
                condition,
                paths,
                dev_rows,
                batch_size=args.eval_batch_size,
                max_new_tokens=args.eval_max_new_tokens,
            )
            console.print(
                f"[bold]{student_id} | {condition}[/bold] "
                f"accuracy = {result['accuracy']:.3%} "
                f"({result['correct']}/{result['n']})"
            )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------

def aggregate_pruning(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    return {
        "n_train": n,
        "mean_parent_tokens": sum(float(r["parent_tokens"]) for r in rows) / n,
        "mean_retained_tokens": sum(float(r["retained_tokens"]) for r in rows) / n,
        "mean_retention_pct": 100.0 * sum(float(r["retention"]) for r in rows) / n,
        "mean_token_reduction_pct": sum(float(r["token_reduction_pct"]) for r in rows) / n,
        "source_compressed_accuracy": sum(
            bool(r["source_compressed_correct"]) for r in rows
        ) / n,
        "mean_prune_seconds": sum(float(r["prune_seconds"]) for r in rows) / n,
        "median_prune_seconds": float(
            sorted(float(r["prune_seconds"]) for r in rows)[n // 2]
        ),
        "mean_rounds": sum(float(r["n_rounds"]) for r in rows) / n,
    }


def maybe_eval_result(paths: RunPaths, student_id: str, condition: str) -> Optional[dict[str, Any]]:
    p = paths.eval_file(student_id, condition)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def cmd_report(args) -> None:
    require_ml_stack(include_plotting=True)
    paths = RunPaths(Path(args.workdir))
    protocol = load_protocol(paths)
    report_dir = paths.report_dir

    methods = ["joint", "consensus", "gradient", "ans"]
    pruning = {}
    for m in methods:
        try:
            rows = load_complete_pruned(paths, m)
        except Exception as exc:
            console.print(f"[yellow]Skipping incomplete {m}: {exc}[/yellow]")
            continue
        pruning[m] = aggregate_pruning(rows)

    student_ids = set()
    eval_root = paths.workdir / "eval"
    if eval_root.exists():
        for d in eval_root.iterdir():
            if d.is_dir():
                for jf in d.glob("*.json"):
                    try:
                        obj = json.loads(jf.read_text(encoding="utf-8"))
                        student_ids.add(obj["student_model"])
                    except Exception:
                        pass

    comparison_rows = []
    for student_id in sorted(student_ids):
        base = maybe_eval_result(paths, student_id, "base")
        base_acc = base["accuracy"] if base else None

        for method in methods:
            if method not in pruning:
                continue
            ev = maybe_eval_result(paths, student_id, method)
            row = {
                "source_model": protocol["source_model"],
                "student_model": student_id,
                "condition": method,
                **pruning[method],
                "base_dev_accuracy": base_acc,
                "dev_accuracy": ev["accuracy"] if ev else None,
                "dev_delta_vs_base": (
                    ev["accuracy"] - base_acc
                    if ev and base_acc is not None
                    else None
                ),
            }
            comparison_rows.append(row)

    # If eval is not ready, still produce source/pruning rows.
    if not comparison_rows:
        for method in methods:
            if method in pruning:
                comparison_rows.append(
                    {
                        "source_model": protocol["source_model"],
                        "student_model": None,
                        "condition": method,
                        **pruning[method],
                        "base_dev_accuracy": None,
                        "dev_accuracy": None,
                        "dev_delta_vs_base": None,
                    }
                )

    df = pd.DataFrame(comparison_rows)
    csv_path = report_dir / "comparison.csv"
    df.to_csv(csv_path, index=False)

    md_path = report_dir / "comparison.md"
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")

    # Primary requested comparison: ACL JOINT-50 vs Consensus-50 only.
    primary_df = df[df["condition"].isin(PRIMARY_METHODS)].copy()
    primary_csv = report_dir / "primary_joint_vs_consensus.csv"
    primary_md = report_dir / "primary_joint_vs_consensus.md"
    primary_df.to_csv(primary_csv, index=False)
    primary_md.write_text(primary_df.to_markdown(index=False), encoding="utf-8")

    # Rich console table.
    table = Table(title="Harish-500 comparison")
    columns = [
        ("Student", "student_model"),
        ("Condition", "condition"),
        ("Retained", "mean_retention_pct"),
        ("Reduction", "mean_token_reduction_pct"),
        ("Source acc", "source_compressed_accuracy"),
        ("Dev acc", "dev_accuracy"),
        ("Δ vs base", "dev_delta_vs_base"),
        ("Prune sec/ex", "mean_prune_seconds"),
    ]
    for title, _ in columns:
        table.add_column(title)

    for _, r in df.iterrows():
        def fmt(key, pct=False):
            v = r.get(key)
            if pd.isna(v):
                return "—"
            if pct:
                return f"{100.0 * float(v):.1f}%"
            return f"{float(v):.2f}"

        table.add_row(
            str(r.get("student_model") or "—"),
            str(r["condition"]),
            f"{float(r['mean_retention_pct']):.1f}%",
            f"{float(r['mean_token_reduction_pct']):.1f}%",
            fmt("source_compressed_accuracy", pct=True),
            fmt("dev_accuracy", pct=True),
            fmt("dev_delta_vs_base", pct=True),
            f"{float(r['mean_prune_seconds']):.1f}",
        )
    console.print(table)

    # ---------------------------
    # Plots
    # ---------------------------
    if pruning:
        p_df = pd.DataFrame(
            [{"condition": k, **v} for k, v in pruning.items()]
        )

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.bar(p_df["condition"], p_df["source_compressed_accuracy"] * 100.0)
        ax.set_ylabel("Source-model accuracy after compression (%)")
        ax.set_xlabel("Pruning condition")
        ax.set_ylim(0, 100)
        ax.set_title("Compressed-500 source-model sanity check")
        fig.tight_layout()
        fig.savefig(report_dir / "source_accuracy_by_condition.png", dpi=180)
        plt.close(fig)

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.bar(p_df["condition"], p_df["mean_prune_seconds"])
        ax.set_ylabel("Mean pruning seconds per example")
        ax.set_xlabel("Pruning condition")
        ax.set_title("Pruning runtime")
        fig.tight_layout()
        fig.savefig(report_dir / "pruning_runtime.png", dpi=180)
        plt.close(fig)

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        x = range(len(p_df))
        width = 0.36
        ax.bar(
            [i - width / 2 for i in x],
            p_df["mean_parent_tokens"],
            width=width,
            label="Parent",
        )
        ax.bar(
            [i + width / 2 for i in x],
            p_df["mean_retained_tokens"],
            width=width,
            label="Retained",
        )
        ax.set_xticks(list(x), p_df["condition"].tolist())
        ax.set_ylabel("Mean source-model rationale tokens")
        ax.set_title("Parent vs 50%-retained rationale length")
        ax.legend()
        fig.tight_layout()
        fig.savefig(report_dir / "tokens_parent_vs_retained.png", dpi=180)
        plt.close(fig)

    eval_df = df.dropna(subset=["dev_accuracy"]).copy()
    if len(eval_df):
        # Accuracy by condition/student.
        pivot = eval_df.pivot(
            index="condition", columns="student_model", values="dev_accuracy"
        ) * 100.0
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Dev accuracy (%)")
        ax.set_xlabel("Training condition")
        ax.set_title("SFT accuracy on unfiltered GSM8K test 0..499")
        ax.legend(title="Student model", fontsize=8)
        fig.tight_layout()
        fig.savefig(report_dir / "dev_accuracy_by_condition.png", dpi=180)
        plt.close(fig)

        # Requested reduction-vs-accuracy view.
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        for _, r in eval_df.iterrows():
            ax.scatter(
                float(r["mean_token_reduction_pct"]),
                100.0 * float(r["dev_accuracy"]),
                s=70,
            )
            ax.annotate(
                f"{r['condition']} | {safe_slug(str(r['student_model'])).split('__')[-1]}",
                (
                    float(r["mean_token_reduction_pct"]),
                    100.0 * float(r["dev_accuracy"]),
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xlabel("Mean training-rationale token reduction (%)")
        ax.set_ylabel("Dev accuracy (%)")
        ax.set_title("Token reduction vs SFT accuracy")
        fig.tight_layout()
        fig.savefig(report_dir / "token_reduction_vs_dev_accuracy.png", dpi=180)
        plt.close(fig)

    summary = {
        "protocol": protocol,
        "comparison_csv": str(csv_path),
        "comparison_markdown": str(md_path),
        "primary_joint_vs_consensus_csv": str(primary_csv),
        "primary_joint_vs_consensus_markdown": str(primary_md),
        "plots": sorted(str(p) for p in report_dir.glob("*.png")),
    }
    write_json(report_dir / "report_manifest.json", summary)

    console.print(
        Panel.fit(
            f"All conditions: {csv_path}\n"
            f"Primary JOINT vs Consensus: {primary_csv}\n"
            f"Plots: {len(summary['plots'])}\nManifest: {report_dir / 'report_manifest.json'}",
            title="Report written",
        )
    )



def cmd_report_consensus_guard(args) -> None:
    """
    Write a separate comparison for the guarded Consensus variant.
    Existing reports and evaluations are untouched.
    """
    require_ml_stack(include_plotting=False)
    paths = RunPaths(Path(args.workdir))
    protocol = load_protocol(paths)
    report_dir = paths.report_dir

    student_id = resolve_model("llama")
    methods = [
        "gradient",
        "ans",
        "joint",
        "consensus",
        "consensus_v2",
        "consensus_v3",
        "consensus_guard_t1",
    ]

    base = maybe_eval_result(paths, student_id, "base")
    base_acc = base["accuracy"] if base else None

    rows_out = []
    for method in methods:
        try:
            pruned = load_complete_pruned(paths, method)
        except Exception:
            continue

        agg = aggregate_pruning(pruned)
        ev = maybe_eval_result(paths, student_id, method)

        # Guard-specific diagnostics from pruning rounds.
        guard_rounds = [
            rr
            for row in pruned
            for rr in row.get("rounds", [])
            if "ans_guard_tau" in rr
        ]
        if guard_rounds:
            total_deleted = sum(int(rr["deleted"]) for rr in guard_rounds)
            total_fallback = sum(
                int(rr.get("guard_fallback_deletions", 0))
                for rr in guard_rounds
            )
            fallback_pct = (
                100.0 * total_fallback / total_deleted
                if total_deleted
                else 0.0
            )
            mean_safe_fraction = sum(
                float(rr["safe_candidates"]) / float(rr["candidate_count"])
                for rr in guard_rounds
            ) / len(guard_rounds)
        else:
            fallback_pct = None
            mean_safe_fraction = None

        rows_out.append(
            {
                "source_model": protocol["source_model"],
                "student_model": student_id,
                "condition": method,
                **agg,
                "base_dev_accuracy": base_acc,
                "dev_accuracy": ev["accuracy"] if ev else None,
                "dev_delta_vs_base": (
                    ev["accuracy"] - base_acc
                    if ev and base_acc is not None
                    else None
                ),
                "ans_guard_tau": (
                    float(ANS_GUARD_TAU)
                    if method == "consensus_guard_t1"
                    else None
                ),
                "guard_fallback_deletion_pct": fallback_pct,
                "mean_safe_candidate_fraction": mean_safe_fraction,
            }
        )

    df = pd.DataFrame(rows_out)
    csv_path = report_dir / "consensus_guard_tau1_vs_existing.csv"
    md_path = report_dir / "consensus_guard_tau1_vs_existing.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")

    table = Table(title="Consensus-v2 + ANS guard (tau=1.0)")
    table.add_column("Condition")
    table.add_column("Source acc")
    table.add_column("Dev acc")
    table.add_column("Δ vs base")
    table.add_column("Prune sec/ex")
    table.add_column("Guard fallback")

    for _, r in df.iterrows():
        def pct(v):
            if pd.isna(v):
                return "—"
            return f"{100.0 * float(v):.1f}%"

        fallback = (
            "—"
            if pd.isna(r["guard_fallback_deletion_pct"])
            else f"{float(r['guard_fallback_deletion_pct']):.1f}%"
        )

        table.add_row(
            str(r["condition"]),
            pct(r["source_compressed_accuracy"]),
            pct(r["dev_accuracy"]),
            pct(r["dev_delta_vs_base"]),
            f"{float(r['mean_prune_seconds']):.1f}",
            fallback,
        )

    console.print(table)
    console.print(
        Panel.fit(
            f"CSV: {csv_path}\nMarkdown: {md_path}",
            title="Guarded Consensus report written",
        )
    )


# --------------------------------------------------------------------------------------
# Validation / self-test
# --------------------------------------------------------------------------------------

def cmd_self_test(args) -> None:
    failures = []

    def check(name, cond):
        if cond:
            console.print(f"[green]PASS[/green] {name}")
        else:
            failures.append(name)
            console.print(f"[red]FAIL[/red] {name}")

    check("Train size fixed at 500", N_TRAIN == 500)
    check("Dev size fixed at 500", N_DEV == 500)
    check("Retention fixed at 50%", RETENTION == 0.50)
    check("Consensus directions fixed at 4", CONSENSUS_DIRECTIONS == 4)
    check("Consensus steps fixed at 64", CONSENSUS_STEPS == 64)
    check("Primary comparison is JOINT vs Consensus", PRIMARY_METHODS == ("joint", "consensus"))
    check("Harish auxiliary Gradient baseline retained", "gradient" in ALL_PRUNING_METHODS)
    check("Harish auxiliary ANS baseline retained", "ans" in ALL_PRUNING_METHODS)
    check(
        "Default round deletion inside Harish 15-20% range",
        MIN_ROUND_DELETE_FRACTION
        <= DEFAULT_ROUND_DELETE_FRACTION
        <= MAX_ROUND_DELETE_FRACTION,
    )
    check("Single default seed", isinstance(DEFAULT_SEED, int))
    check("Numeric normalisation", normalize_number("$1,234.00") == "1234")
    check("GSM8K gold parser", gold_from_gsm8k("reasoning\n#### 42") == "42")
    check("Prediction parser", extract_prediction("The answer is 42") == "42")
    check(
        "Token subsequence stripping",
        strip_token_subsequence([1, 2, 3, 2, 3, 4], [2, 3]) == [1, 4],
    )
    check(
        "Qwen explicit instruct alias exists",
        MODEL_ALIASES["qwen"] == "Qwen/Qwen2.5-7B-Instruct",
    )
    check(
        "Llama canonical alias exists",
        MODEL_ALIASES["llama"] == "meta-llama/Llama-3.1-8B-Instruct",
    )

    if failures:
        raise SystemExit("Self-test failures: " + ", ".join(failures))
    console.print(Panel.fit("All protocol/unit checks passed.", title="Self-test"))


# --------------------------------------------------------------------------------------
# No-argument runner
# --------------------------------------------------------------------------------------

VALID_JOBS = {
    "prepare",
    "prune_fast",
    "prune_consensus",
    "sft_gradient",
    "sft_remaining",
    "eval",
    "report",
    "all_sequential",
}


def _common_namespace() -> dict[str, Any]:
    return {
        "workdir": WORKDIR,
        "source_model": SOURCE_MODEL,
        "seed": int(SEED),
    }


def make_prepare_config() -> SimpleNamespace:
    d = _common_namespace()
    d.update(
        {
            "parents_file": PARENTS_FILE,
            "round_delete_fraction": float(ROUND_DELETE_FRACTION),
            "direct_max_new_tokens": int(DIRECT_MAX_NEW_TOKENS),
            "cot_max_new_tokens": int(COT_MAX_NEW_TOKENS),
            "cot_samples": int(COT_SAMPLES),
            "cot_temperature": float(COT_TEMPERATURE),
            "cot_top_p": float(COT_TOP_P),
        }
    )
    return SimpleNamespace(**d)


def make_prune_config(methods: str) -> SimpleNamespace:
    d = _common_namespace()
    d.update(
        {
            "methods": methods,
            "candidate_batch": int(CANDIDATE_BATCH),
            "num_shards": int(NUM_SHARDS),
            "shard_id": int(SHARD_ID),
        }
    )
    return SimpleNamespace(**d)


def make_sft_config(conditions: str) -> SimpleNamespace:
    d = _common_namespace()
    d.update(
        {
            "conditions": conditions,
            "student_models": STUDENT_MODELS,
        }
    )
    return SimpleNamespace(**d)


def make_eval_config() -> SimpleNamespace:
    d = _common_namespace()
    d.update(
        {
            "conditions": "gradient,ans,joint,consensus",
            "student_models": STUDENT_MODELS,
            "eval_batch_size": int(EVAL_BATCH_SIZE),
            "eval_max_new_tokens": int(EVAL_MAX_NEW_TOKENS),
        }
    )
    return SimpleNamespace(**d)


def validate_user_configuration() -> None:
    if JOB not in VALID_JOBS:
        raise ValueError(
            f"JOB={JOB!r} is invalid. Choose one of: {sorted(VALID_JOBS)}"
        )

    if not str(WORKDIR).strip():
        raise ValueError("WORKDIR must not be empty.")

    if not HF_TOKEN:
        console.print(
            "[yellow]WARNING:[/yellow] HF_TOKEN is empty. "
            "Set it in the environment with: export HF_TOKEN=\"hf_...\" "
            "before loading gated models."
        )


    if not (
        MIN_ROUND_DELETE_FRACTION
        <= float(ROUND_DELETE_FRACTION)
        <= MAX_ROUND_DELETE_FRACTION
    ):
        raise ValueError(
            "ROUND_DELETE_FRACTION must stay inside Harish's 0.15-0.20 range."
        )

    if int(COT_SAMPLES) < 1:
        raise ValueError("COT_SAMPLES must be >=1.")

    if int(COT_SAMPLES) > 1 and float(COT_TEMPERATURE) <= 0.0:
        raise ValueError(
            "COT_SAMPLES > 1 requires COT_TEMPERATURE > 0. "
            "Changing this also changes the eligible training population."
        )

    if int(NUM_SHARDS) < 1:
        raise ValueError("NUM_SHARDS must be >=1.")

    if not 0 <= int(SHARD_ID) < int(NUM_SHARDS):
        raise ValueError("Need 0 <= SHARD_ID < NUM_SHARDS.")


def print_run_configuration() -> None:
    table = Table(title="Harish-500 full JOINT + Consensus pipeline")
    table.add_column("Setting")
    table.add_column("Value")
    rows = [
        ("PIPELINE", "RUN EVERYTHING"),
        ("WORKDIR", WORKDIR),
        ("SOURCE_MODEL", resolve_model(SOURCE_MODEL)),
        ("CUDA DEVICE", "physical cuda:2"),
        ("HF_TOKEN", "set" if HF_TOKEN else "NOT SET"),
        ("PARENTS_FILE", str(PARENTS_FILE) if PARENTS_FILE else "None (fallback filtering)"),
        ("STUDENT_MODELS", STUDENT_MODELS),
        ("TRAIN", "500 filtered GSM8K-train parents"),
        ("DEV", "GSM8K test indices 0..499, UNFILTERED"),
        ("RETENTION", "50%"),
        ("ROUND_DELETE_FRACTION", f"{ROUND_DELETE_FRACTION:.0%} of current remaining"),
        ("PRIMARY", "ACL JOINT-50 vs Consensus-50"),
        ("AUXILIARY", "Gradient-50 + ANS-50 (Harish requested)"),
        ("JOINT", "retained CoT + answer completion likelihood"),
        ("CONSENSUS", "4 directions × 64 steps"),
        ("LM_HEAD_TIME_CHUNK", str(LM_HEAD_TIME_CHUNK)),
        ("SEED", str(SEED)),
        ("SHARD", f"{SHARD_ID}/{NUM_SHARDS}"),
    ]
    for k, v in rows:
        table.add_row(str(k), str(v))
    console.print(table)


def run_selected_job(job: str) -> None:
    if job == "prepare":
        cmd_prepare(make_prepare_config())

    elif job == "prune_fast":
        cmd_prune(make_prune_config("gradient,ans,joint"))

    elif job == "prune_consensus":
        cmd_prune(make_prune_config("consensus"))

    elif job == "sft_gradient":
        cmd_sft(make_sft_config("gradient"))

    elif job == "sft_remaining":
        cmd_sft(make_sft_config("ans,joint,consensus"))

    elif job == "eval":
        cmd_eval(make_eval_config())

    elif job == "report":
        cmd_report(SimpleNamespace(workdir=WORKDIR))

    elif job == "all_sequential":
        # Safe/resumable serial run. Harish prefers Consensus in parallel with
        # Gradient SFT, so use the individual JOB values above when multiple GPUs
        # are available.
        cmd_prepare(make_prepare_config())
        cmd_prune(make_prune_config("gradient,ans"))
        cmd_sft(make_sft_config("gradient"))
        cmd_prune(make_prune_config("consensus"))
        cmd_sft(make_sft_config("ans,consensus"))
        cmd_eval(make_eval_config())
        cmd_report(SimpleNamespace(workdir=WORKDIR))

    else:
        raise AssertionError(job)


def main() -> None:
    require_ml_stack()
    check_gpu()

    paths = RunPaths(Path(WORKDIR))
    protocol = load_protocol(paths)

    parents = validate_parent_ids(paths)
    if len(parents) != 500:
        raise RuntimeError("Expected the already-fixed 500 parent CoTs.")

    if float(protocol["retention"]) != 0.5:
        raise RuntimeError("Saved protocol is not the expected 50% retention run.")
    if float(protocol["round_delete_fraction"]) != 0.2:
        raise RuntimeError(
            "Saved protocol is not the expected 20%-per-round setting."
        )
    if int(protocol["consensus_directions"]) != 4:
        raise RuntimeError("Saved protocol does not use 4 directions.")
    if int(protocol["consensus_steps"]) != 64:
        raise RuntimeError("Saved protocol does not use 64 consensus steps.")

    console.print(
        Panel.fit(
            "[bold]CONSENSUS-V2 + ANS SAFETY GUARD[/bold]\n\n"
            "Reuses the existing 500 parent CoTs and all existing baselines.\n"
            "Does NOT rerun filtering, ANS, JOINT, Gradient, Consensus-v1/v2/v3,\n"
            "or their existing adapters/evaluations.\n\n"
            "Guarded score:\n"
            "  1. Geometry = Consensus-v2 median cosine across 4 independently\n"
            "     optimised answer-directed directions.\n"
            "  2. For each candidate deletion compute\n"
            "       ΔANS = logP(A | R_-i) - logP(A | R).\n"
            "  3. A candidate is safe when ΔANS >= -1.0.\n"
            "  4. If >= k safe candidates exist, choose top-k ONLY by Consensus\n"
            "     geometry among those safe candidates.\n"
            "  5. If fewer than k are safe, keep all safe candidates and fill\n"
            "     only the unavoidable remainder using the least damaging ANS\n"
            "     deletions, preserving the exact 20% round budget.\n\n"
            "This is a HYBRID method: Consensus supplies the ranking, ANS supplies\n"
            "an answer-preservation safety constraint.\n\n"
            "Everything else stays fixed: 4 directions, 64 steps, 20% per round,\n"
            "50% final retention, same seed, same Llama SFT settings, same dev set.",
            title="Focused guarded-Consensus experiment",
        )
    )

    console.rule(
        "[bold cyan]STAGE 1/4: PRUNE CONSENSUS-GUARD-TAU1[/bold cyan]"
    )
    cmd_prune(make_prune_config("consensus_guard_t1"))

    console.rule(
        "[bold cyan]STAGE 2/4: SFT LLAMA ON GUARDED CONSENSUS[/bold cyan]"
    )
    sft_cfg = make_sft_config("consensus_guard_t1")
    sft_cfg.student_models = "llama"
    cmd_sft(sft_cfg)

    console.rule(
        "[bold cyan]STAGE 3/4: EVALUATE GUARDED CONSENSUS[/bold cyan]"
    )
    eval_cfg = make_eval_config()
    eval_cfg.conditions = "consensus_guard_t1"
    eval_cfg.student_models = "llama"
    cmd_eval(eval_cfg)

    console.rule(
        "[bold cyan]STAGE 4/4: WRITE GUARDED CONSENSUS REPORT[/bold cyan]"
    )
    cmd_report_consensus_guard(SimpleNamespace(workdir=WORKDIR))

    console.print(
        Panel.fit(
            "[bold green]GUARDED CONSENSUS RUN COMPLETE[/bold green]\n"
            f"Pruning: {Path(WORKDIR) / 'pruned' / 'consensus_guard_t1'}\n"
            f"Adapter: {Path(WORKDIR) / 'adapters' / safe_slug(resolve_model('llama')) / 'consensus_guard_t1'}\n"
            f"Eval: {Path(WORKDIR) / 'eval' / safe_slug(resolve_model('llama')) / 'consensus_guard_t1.json'}\n"
            f"Comparison: {Path(WORKDIR) / 'reports' / 'consensus_guard_tau1_vs_existing.csv'}",
            title="DONE",
        )
    )


if __name__ == "__main__":
    main()