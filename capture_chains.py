"""
Phase 1: Chain sampling + instrumentation.

For each GSM8K problem (already filtered in Phase 0 to "wrong without CoT,
right with CoT"), we:
  1. Sample N chain-of-thought completions.
  2. For every generated token, record the residual stream (hidden state)
     at a chosen middle layer.
  3. Convert that middle-layer state into a "running answer probability"
     using the logit-lens trick: apply the model's own final norm + LM
     head to the middle-layer state, as if it were the last layer, and
     read off the probability mass on the correct answer token.
  4. Extract the final numeric answer from the chain and compare to gold
     to label the chain correct / incorrect.

Output: one .pt file per problem containing all its sampled chains, ready
for the DTW alignment step in Phase 3.

Keep this script simple — it's an instrumentation pass, not the final
pipeline. Chunking (Phase 2) and DTW (Phase 3) happen downstream, on the
saved data.
"""

import json
import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config — edit these for your setup
# ---------------------------------------------------------------------------
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 16                 # middle layer to probe (model has 32 layers total;
                            # start here, sweep later if the signal looks off)
N_CHAINS_PER_PROBLEM = 16  # samples per problem
MAX_NEW_TOKENS = 400
TEMPERATURE = 0.8
TOP_P = 0.95

PROBLEMS_PATH = "ProxySPECS/data/gsm8k_cot_necessary.jsonl"   # your Phase 0 output (jsonl)
OUT_DIR = "chain_captures"
MAX_PROBLEMS = 100           # cap for a quick pipeline sanity check; set to
                            # None to run the full filtered set

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# Llama's decoder layers + final norm + lm_head. This path is specific to
# the Llama architecture in HF transformers; check model.model if you swap
# to a different model family.
final_norm = model.model.norm
lm_head = model.lm_head


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_prompt(question: str) -> str:
    """Simple CoT prompt using the model's chat template."""
    messages = [
        {
            "role": "user",
            "content": (
                f"{question}\n\n"
                "Solve this step by step. On the VERY LAST line of your "
                "response, write ONLY the final numeric answer in exactly "
                "this format (no dollar signs, no extra words):\n"
                "#### <number>"
            ),
        }
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_answer(text: str) -> str | None:
    """
    Pull the final numeric answer out of a generated chain.
    Primary format: '#### <number>' (what we asked for in the prompt).
    Fallback: instruction-tuned models don't always obey the literal
    '####' format, so if that's missing we fall back to the LAST
    standalone number in the text, which is usually the final answer in
    a "So the answer is X" / "**X**" style completion.
    """
    match = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: last standalone number anywhere in the text.
    fallback_matches = re.findall(r"(-?[\d,]+\.?\d*)", text)
    if fallback_matches:
        return fallback_matches[-1].replace(",", "")
    return None


def answers_match(pred: str | None, gold: str) -> bool:
    """
    Compare extracted vs. gold answers NUMERICALLY, not as strings.
    A pure string comparison wrongly marks "18." != "18" (or "18.0" !=
    "18") as incorrect even though they're the same number -- this
    happens often because the fallback regex above allows a trailing
    bare '.' with no digits after it. Falls back to a stripped string
    comparison only if either side isn't parseable as a float (shouldn't
    normally happen for GSM8K answers).
    """
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred.strip() == gold.strip()


def answer_token_id(answer_str: str) -> int | None:
    """
    Get a single token id representing the correct answer, for the
    logit-lens probability readout. GSM8K answers are numbers, which may
    tokenize into multiple pieces — we use the FIRST *content* token of
    the answer as a simple proxy for "the model is starting to commit to
    this answer".

    NOTE on a real gotcha: encoding " " + answer_str can produce a leading
    token that is JUST the whitespace (decodes to ''), with the actual
    digits starting at index 1, not 0. Verified empirically for this
    tokenizer, e.g. " 18" -> [' ', '18']. We explicitly skip any
    whitespace-only tokens rather than assuming index 0 is the real one.

    Multi-digit answers (e.g. "70000" -> ["700", "00"]) still only get
    scored on their first chunk — this distinguishes "70000" from "54000"
    but not "70000" from "70500". Good enough as a first-pass signal for
    DTW; if the probability trace looks too noisy/uninformative later,
    consider scoring the full multi-token sequence instead (requires
    rolling generation forward from each captured hidden state, not just
    a single logit-lens readout).
    """
    ids = tokenizer.encode(" " + answer_str, add_special_tokens=False)
    for tid in ids:
        if tokenizer.decode([tid]).strip() != "":
            return tid
    return None


@torch.no_grad()
def sample_one_chain(prompt: str):
    """
    Generate one CoT chain and capture, per generated token:
      - the residual stream at LAYER
      - the generated token id
    Returns (generated_text, token_ids, residuals [n_tokens, hidden]).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        return_dict_in_generate=True,
        output_hidden_states=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # outputs.hidden_states is a tuple over generation steps.
    # Each step is a tuple over layers: (embeddings, layer1, layer2, ..., layerN).
    # For step 0 the sequence dim covers the whole prompt; for later steps
    # it's just the 1 new token (because of KV caching). We always want the
    # hidden state of the newly generated token, i.e. index -1 along seq dim.
    residuals = []
    for step_hidden_states in outputs.hidden_states:
        layer_state = step_hidden_states[LAYER]      # (batch, seq, hidden)
        residuals.append(layer_state[0, -1, :].float().cpu())
    residuals = torch.stack(residuals)  # (n_generated_tokens, hidden)

    gen_token_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(gen_token_ids, skip_special_tokens=True)

    return generated_text, gen_token_ids.cpu(), residuals


@torch.no_grad()
def running_answer_probability(residuals: torch.Tensor, ans_token_id: int) -> torch.Tensor:
    """
    Logit-lens: treat each middle-layer residual as if it were the final
    hidden state, apply the model's own final norm + LM head, and read off
    the probability assigned to the correct answer's first token.

    residuals: (n_tokens, hidden)
    returns: (n_tokens,) probability trace
    """
    residuals = residuals.to(model.device, dtype=model.dtype)
    normed = final_norm(residuals)                 # (n_tokens, hidden)
    logits = lm_head(normed)                        # (n_tokens, vocab)
    probs = torch.softmax(logits.float(), dim=-1)
    return probs[:, ans_token_id].cpu()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def load_phase0_problems(path: str):
    """
    Load the Phase 0 output jsonl. Each line already has question,
    gold_answer, and the no_cot/cot correctness flags baked in. We assign
    a stable integer id from line order (== row index) and re-check the
    no_cot_correct/cot_correct filter here as a cheap safety net in case
    this file ever gets concatenated with something unfiltered.
    """
    problems = []
    with open(path) as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            if row.get("no_cot_correct", True) or not row.get("cot_correct", False):
                # Doesn't match the "wrong without CoT, right with CoT"
                # criterion -- skip, this run should only use qualifying
                # problems.
                continue
            problems.append({
                "id": idx,
                "question": row["question"],
                "answer": row["gold_answer"],
            })
    return problems


def main():
    problems = load_phase0_problems(PROBLEMS_PATH)
    if MAX_PROBLEMS is not None:
        problems = problems[:MAX_PROBLEMS]
    print(f"Loaded {len(problems)} qualifying problems from {PROBLEMS_PATH}"
          + (f" (capped at MAX_PROBLEMS={MAX_PROBLEMS})" if MAX_PROBLEMS else ""))

    for problem in problems:
        pid = problem["id"]
        question = problem["question"]
        gold_answer = str(problem["answer"]).strip()
        ans_tok_id = answer_token_id(gold_answer)

        if ans_tok_id is None:
            print(f"[skip] problem {pid}: couldn't tokenize gold answer")
            continue

        prompt = build_prompt(question)
        chains = []

        for chain_idx in range(N_CHAINS_PER_PROBLEM):
            text, token_ids, residuals = sample_one_chain(prompt)
            pred_answer = extract_answer(text)
            is_correct = answers_match(pred_answer, gold_answer)
            prob_trace = running_answer_probability(residuals, ans_tok_id)

            if pred_answer is None and chain_idx == 0:
                # Diagnostic: if extraction is failing, print the raw text
                # once per problem so we can see what format the model
                # actually used instead of guessing blind.
                print(f"---- [{pid}] chain 0 RAW TEXT (extract_answer failed) ----")
                print(text)
                print("---- end raw text ----")

            chains.append({
                "chain_idx": chain_idx,
                "text": text,
                "token_ids": token_ids,       # (n_tokens,)
                "residuals": residuals,       # (n_tokens, hidden) at LAYER
                "prob_trace": prob_trace,     # (n_tokens,) running P(answer)
                "pred_answer": pred_answer,
                "correct": is_correct,
            })

            status = "OK " if is_correct else "ERR"
            print(f"[{pid}] chain {chain_idx:2d} [{status}] "
                  f"pred={pred_answer} gold={gold_answer}")

        out_path = os.path.join(OUT_DIR, f"{pid}.pt")
        torch.save({
            "problem_id": pid,
            "question": question,
            "gold_answer": gold_answer,
            "layer": LAYER,
            "chains": chains,
        }, out_path)
        print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()