import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def load_model():
    print(f"Loading {MODEL}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )

    model.eval()

    return tokenizer, model


@torch.no_grad()
def generate(model, tokenizer, input_ids, max_new_tokens=1024):

    attention_mask = torch.ones_like(input_ids)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(
        output[0][input_ids.shape[1]:],
        skip_special_tokens=False
    )


def main():

    tokenizer, model = load_model()

    with open("data/lcb_cot_necessary.jsonl") as f:
        rec = json.loads(next(f))

    question = rec["question_content"]
    cot = rec["cot_completion"]


    messages = [
        {
            "role": "user",
            "content":
                "Think through this step by step before giving your final code solution.\n\n"
                + question
        }
    ]


    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids[0]


    cot_tokens = tokenizer(
        cot,
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids[0]


    prompt_len = len(prompt_ids)


    # -------------------------------
    # Choose one reasoning token to mask
    # -------------------------------

    cot_mask_position = 5   # sixth CoT token

    original_token = cot_tokens[cot_mask_position].item()

    print(
        "Masking token:",
        repr(tokenizer.decode([original_token])),
        "at CoT position",
        cot_mask_position
    )


    # -------------------------------
    # Build prefix for generation
    # -------------------------------

    prefix_cot = cot_tokens[:cot_mask_position + 1]


    input_ids = torch.cat(
        [
            prompt_ids,
            prefix_cot
        ]
    ).unsqueeze(0).cuda()


    # -------------------------------
    # Baseline generation
    # -------------------------------

    baseline = generate(
        model,
        tokenizer,
        input_ids
    )


    # -------------------------------
    # Masked generation
    # -------------------------------

    masked_ids = input_ids.clone()


    blank_token_id = tokenizer.convert_tokens_to_ids(
        "<|reserved_special_token_0|>"
    )


    # replace the selected CoT token
    masked_ids[0, prompt_len + cot_mask_position] = blank_token_id


    masked = generate(
        model,
        tokenizer,
        masked_ids
    )


    print("\n===== BASELINE =====")
    print(baseline)


    print("\n===== MASKED =====")
    print(masked)


if __name__ == "__main__":
    main()