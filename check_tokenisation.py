from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

for ans in ["18", "3", "70000", "540", "20", "9", "1234", "0.5"]:
    ids = tok.encode(" " + ans, add_special_tokens=False)
    pieces = [tok.decode([i]) for i in ids]
    print(f"{ans!r:>8} -> {pieces}")