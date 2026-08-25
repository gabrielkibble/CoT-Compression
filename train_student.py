"""
Phase 6, step 2: train a student model via LoRA SFT.

Usage:
  python3 train_student.py --dataset train_full.jsonl --output_dir student_full --seed 0
  python3 train_student.py --dataset train_compressed.jsonl --output_dir student_compressed --seed 0

Trains meta-llama/Llama-3.2-1B-Instruct (a genuinely weaker model than the
Llama-3.1-8B-Instruct teacher used everywhere else in this pipeline) via
LoRA, on whichever dataset you point it at.

TRL >=1.x NOTE: modern SFTTrainer natively supports datasets with exactly
"prompt" and "completion" columns, and masks the prompt from the loss by
default (completion_only_loss=True) -- no separate DataCollator needed,
and passing a custom formatting_func actively CONFLICTS with this (raises
an error), so we do NOT pre-join prompt+completion into one "text" field.

IMPORTANT CORRECTNESS POINT: the "prompt" field must already be wrapped in
the model's chat template (ending in the assistant generation prompt) --
otherwise the student trains on plain, un-templated text while
evaluate_student.py queries it WITH the proper chat template at inference
time, a train/inference mismatch that would silently degrade the trained
model. The raw JSONL (shared with the team) stores the plain question
text; we apply the chat template here, at load time, rather than storing
templated text in the shared file -- keeps the shared dataset
model-agnostic in case a different student model is used later.

Requires: pip install trl peft transformers accelerate datasets --break-system-packages
"""

import argparse
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

STUDENT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def load_dataset(path, tokenizer):
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            templated_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            rows.append({"prompt": templated_prompt, "completion": row["completion"]})
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, required=True,
                         help="REQUIRED and explicit (not defaulted) -- team agreement is "
                              "two seeds minimum per condition, so this must be set "
                              "deliberately each run, not silently reused.")
    args = parser.parse_args()

    import random as _random
    import numpy as _np
    from transformers import set_seed
    _random.seed(args.seed)
    _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    args.output_dir = f"{args.output_dir}_seed{args.seed}"

    print(f"Loading student model {STUDENT_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL)

    dataset = load_dataset(args.dataset, tokenizer)
    print(f"Loaded {len(dataset)} training examples from {args.dataset}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        seed=args.seed,
        report_to="none",
        completion_only_loss=True,
        max_length=1024,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print(f"Training for {args.epochs} epochs...")
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    run_config = {
        "student_model": STUDENT_MODEL,
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "lora_target_modules": lora_config.target_modules,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
    }
    with open(f"{args.output_dir}/run_config.json", "w") as f:
        json.dump(run_config, f, indent=2, default=list)

    print(f"Saved -> {args.output_dir} (seed={args.seed}, config saved to run_config.json)")


if __name__ == "__main__":
    main()