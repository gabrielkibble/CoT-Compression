import torch
data = torch.load("chunked_captures/2.pt", weights_only=False)
chain = data["chains"][1]
for i, c in enumerate(chain["chunks"]):
    print(f"{i:2d}: {c['text']!r}")