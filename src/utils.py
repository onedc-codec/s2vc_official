import torch
from safetensors import safe_open


def load_safetensor(path, map_location="cpu", force_safetensor=False):
    if force_safetensor or path.endswith("safetensors"):
        tensors = {}
        with safe_open(path, framework="pt", device=map_location) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        return tensors
    else:
        return torch.load(path, map_location=map_location)
