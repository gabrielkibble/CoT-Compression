"""
Direct residual steering experiment.

Rather than the logit-lens shortcut used elsewhere in this project (which
applies the model's final norm + unembedding DIRECTLY to a middle-layer
state, skipping the remaining layers entirely), this does real activation
patching: we FORCE the hidden state at layer 16 to sit near the Phase 4
consensus route's direction, let the model's actual remaining layers
(17-32) process that patched state, and read off what token genuinely
falls out the other end.

This tests something the logit-lens can't: does the "corrected" residual
direction we've been using for all our similarity metrics correspond to
something the network can actually be steered toward and act on
coherently, or is it a purely descriptive/statistical construct that
produces gibberish when forced?

IMPORTANT DESIGN CHOICE: consensus route states are stored anisotropy-
CORRECTED (mean-centered, token-direction-projected, standardized). To
inject them we invert the mean/std part to get back to raw residual
scale:
    raw_target = residual_corrected * std + mean
We deliberately do NOT re-add the removed token-direction component. That
component encodes "how much this state points toward the medoid chain's
own specific next token" -- re-adding it would just force the model to
trivially reproduce the medoid's exact token (we already know the
logit-lens recovers that token with high probability, that's not new
information). Leaving it out means we're injecting the GENERIC direction
shared across correct reasoning, and asking what the network naturally
wants to say from there.

Method, per chunk of a problem's consensus route:
  1. Build the prefix as the MEDOID's own actual prior chunks (so we're
     on-route right up to the injection point, isolating the effect of
     patching just this one step).
  2. Run one forward pass with a hook that overwrites the layer-16 output
     at the prefix's last token position with raw_target.
  3. Read the REAL final logits (after the patched state has passed
     through the rest of the network) and decode the top-k candidate next
     tokens.
  4. Optionally continue generation a few more (unpatched) tokens to see
     a short natural-language snippet.
  5. Compare against what the medoid chain actually said next.

This is exploratory/qualitative -- run on a handful of problems, not the
full dataset.

Requires: consensus_routes.pkl, anisotropy_correction.npz
"""

import pickle
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 16
ROUTES_PATH = "consensus_routes.pkl"
CORRECTION_PATH = "anisotropy_correction.npz"
N_PROBLEMS_TO_TEST = 5     # keep small -- this is qualitative/exploratory
TOP_K = 5
CONTINUATION_TOKENS = 15   # how many extra tokens to generate after the
                           # patched step, to see a short natural snippet


class Steerer:
    """
    Forward hook that overwrites the LAST-position hidden state of one
    specific decoder layer with a target vector, but only for the NEXT
    forward call it sees -- then deactivates itself. This means during
    model.generate(), only the very first forward call (which processes
    the injected prefix) gets patched; every subsequent token generates
    completely naturally, letting us see how the model runs with the
    injected state once and no further interference.
    """
    def __init__(self):
        self.active = False
        self.target = None

    def hook(self, module, inputs, output):
        if not self.active:
            return output
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        hidden = hidden.clone()
        hidden[:, -1, :] = self.target.to(hidden.dtype).to(hidden.device)
        self.active = False  # fire once only
        return (hidden,) + output[1:] if is_tuple else hidden


def build_prompt(tokenizer, question: str) -> str:
    messages = [{
        "role": "user",
        "content": (
            f"{question}\n\n"
            "Solve this step by step. On the VERY LAST line of your "
            "response, write ONLY the final numeric answer in exactly "
            "this format (no dollar signs, no extra words):\n"
            "#### <number>"
        ),
    }]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def steer_at_chunk(model, tokenizer, steerer, prefix_text, raw_target):
    """
    Run one forward pass with the steering hook armed, patching the last
    token of prefix_text at layer 16, and return the top-k REAL next-token
    candidates (after full remaining-layer computation).
    """
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    steerer.active = True
    steerer.target = torch.from_numpy(raw_target)
    outputs = model(input_ids=prefix_ids)
    logits = outputs.logits[0, -1, :]
    topk = torch.topk(logits, TOP_K)
    decoded = [(tokenizer.decode([tid]), float(p)) for tid, p in
               zip(topk.indices.tolist(), torch.softmax(logits, dim=-1)[topk.indices].tolist())]
    return decoded, prefix_ids


@torch.no_grad()
def steered_continuation(model, tokenizer, steerer, prefix_ids, raw_target):
    """
    Generate CONTINUATION_TOKENS tokens starting from the patched state --
    only the first forward call (processing prefix_ids) gets patched,
    everything after generates naturally. Gives a short readable snippet.
    """
    steerer.active = True
    steerer.target = torch.from_numpy(raw_target)
    out = model.generate(
        input_ids=prefix_ids,
        attention_mask=torch.ones_like(prefix_ids),
        max_new_tokens=CONTINUATION_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, prefix_ids.shape[1]:], skip_special_tokens=True)


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    with open(ROUTES_PATH, "rb") as f:
        routes = pickle.load(f)
    correction = np.load(CORRECTION_PATH)
    mean, std = correction["mean"], correction["std"]

    steerer = Steerer()
    model.model.layers[LAYER - 1].register_forward_hook(steerer.hook)

    tested = 0
    for pid, route in routes.items():
        if tested >= N_PROBLEMS_TO_TEST:
            break
        tested += 1

        # Need the question text -- reload from chunked_captures since
        # consensus_routes.pkl doesn't store it directly.
        import torch as _torch  # local alias, avoids shadowing issues
        chunked_data = _torch.load(f"chunked_captures/{pid}.pt", weights_only=False)
        question = chunked_data["question"]
        prompt_text = build_prompt(tokenizer, question)

        chunks = route["chunks"]
        print(f"\n{'='*70}\n[Problem {pid}] {question[:80]}...\n{'='*70}")

        running_prefix = prompt_text
        for i, chunk in enumerate(chunks[:-1]):  # skip last chunk (the #### answer line)
            raw_target = chunk["residual_corrected"] * std + mean

            decoded_topk, prefix_ids = steer_at_chunk(model, tokenizer, steerer, running_prefix, raw_target)
            actual_next = chunks[i + 1]["text"]

            print(f"\n--- Step {i} ---")
            print(f"Prefix ends with: ...{running_prefix[-60:]!r}")
            print(f"Steered top-{TOP_K} next tokens: {decoded_topk}")
            print(f"Actual medoid's next chunk was: {actual_next[:80]!r}")

            snippet = steered_continuation(model, tokenizer, steerer, prefix_ids, raw_target)
            print(f"Steered continuation ({CONTINUATION_TOKENS} tokens): {snippet!r}")

            # Advance the prefix using the MEDOID's real text (staying
            # on-route for context), not the steered output -- isolates
            # each step's injection rather than compounding steering
            # errors across steps.
            running_prefix += ("\n" if running_prefix != prompt_text else "") + chunk["text"]


if __name__ == "__main__":
    main()