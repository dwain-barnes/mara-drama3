import json, sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, "D:/drama3/third_party/llama.cpp")
sys.path.insert(1, "D:/drama3/third_party/llama.cpp/gguf-py")
import gguf
from conversion import get_model_class

src = Path("D:/models/drama-3")
names = [f"model.language_model.layers.{l}.mlp.experts.{e}.{p}_proj.weight" for l, e, p in [(0, 0, "gate"), (0, 0, "down"), (39, 255, "up")]]
index = json.load(open(src / "model.safetensors.index.json"))["weight_map"]


def ref(name):
    with safe_open(src / index[name], "pt") as f:
        w = f.get_tensor(name)
    with safe_open(src / index[name + "_scale_inv"], "pt") as f:
        s = f.get_tensor(name + "_scale_inv")
    rows, cols = w.shape
    rb, cb = s.shape
    blocks = w.view(rb, 128, cb, 128)
    return (blocks.float() * s[:, None, :, None]).view(rows, cols), w, s


cls = get_model_class("FishOmniModel")
model = cls(src, gguf.LlamaFileType.MOSTLY_BF16, Path("D:/tmp/unused.gguf"), eager=True, dry_run=True)
model.dequant_model()
for name in names:
    conv_name = name.replace("language_model.", "")
    got = model.model_tensors[conv_name]().float()
    want, w, s = ref(name)
    print(name, tuple(w.shape), w.dtype, tuple(s.shape), "scale range", s.min().item(), s.max().item())
    print("  max abs err f32:", (got - want).abs().max().item(),
          " bf16 vs ref->bf16:", (got.bfloat16().float() - want.bfloat16().float()).abs().max().item(),
          " max |w|:", want.abs().max().item())
print("scale_inv left in tensor list:", sum(k.endswith("_scale_inv") for k in model.model_tensors))
print("audio_decoder tensors:", sum("audio_decoder" in k for k in model.model_tensors), "total tensors:", len(model.model_tensors))
