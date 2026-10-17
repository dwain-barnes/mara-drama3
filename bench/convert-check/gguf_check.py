import json, os, sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "D:/drama3/third_party/llama.cpp/gguf-py")
from gguf import GGUFReader

path = "D:/models/drama-3-own/llamacpp/drama-3-bf16.gguf"
r = GGUFReader(path)
f = r.fields
for k in ("general.architecture", "general.file_type", "qwen35moe.block_count", "qwen35moe.expert_count",
          "tokenizer.ggml.pre", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id", "tokenizer.ggml.bos_token_id"):
    print(k, f[k].contents() if k in f else None)
sup = f["tokenizer.ggml.suppress_tokens"].contents()
print("n_vocab", len(f["tokenizer.ggml.tokens"].data), "suppress_tokens", len(sup),
      "eos in suppress", 248046 in set(sup), "semantic in suppress", any(248099 <= i <= 252194 for i in sup),
      "min/max", min(sup), max(sup))
print("nextn key present", any("nextn" in k for k in f))
print("tensor_count", len(r.tensors))
print("audio tensors", sum("audio" in t.name for t in r.tensors))
n = sum(int(np.prod(t.shape)) for t in r.tensors)
print("params", n)
exps = [t for t in r.tensors if "_exps" in t.name]
print("exps tensors", len(exps), {t.name: [int(x) for x in t.shape] for t in exps[:3]}, set(t.tensor_type.name for t in exps))
print("file size bytes", os.path.getsize(path))

src = Path("D:/models/drama-3")
idx = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
t = next(t for t in r.tensors if t.name == "blk.0.ffn_gate_exps.weight")
got = torch.from_numpy(np.asarray(t.data).view(np.int16).reshape(256, 512, 2048)[0].copy()).view(torch.bfloat16).float()
name = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
with safe_open(src / idx[name], "pt") as sf:
    w = sf.get_tensor(name)
with safe_open(src / idx[name + "_scale_inv"], "pt") as sf:
    s = sf.get_tensor(name + "_scale_inv")
want = (w.view(4, 128, 16, 128).float() * s[:, None, :, None]).view(512, 2048)
print("gguf vs fish-speech dequant: max abs err", (got - want).abs().max().item(),
      "vs ref->bf16", (got - want.bfloat16().float()).abs().max().item())
