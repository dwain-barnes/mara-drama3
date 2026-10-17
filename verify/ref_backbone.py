"""Reference last-position logits from the Drama 3 PyTorch backbone, BF16 on CPU."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FISH_DRAMA3_FP8_EXPERTS"] = "0"

import numpy as np
import psutil
import torch
from safetensors import safe_open

TEXT = "Hello there, how are you today?"
EOS = 248046


def peak_gb():
    mem = psutil.Process().memory_info()
    return getattr(mem, "peak_wset", mem.rss) / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--fish-speech", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()

    sys.path.insert(0, str(args.fish_speech))
    from tokenizers import Tokenizer

    from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone, load_backbone_weights
    from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
    from fish_speech.models.text2semantic.fish_omni.generate import (
        DRAMA3_GENERATION_START,
        DRAMA3_MESSAGE_END,
        DRAMA3_TTS_INSTRUCTION,
        build_drama3_prompt,
        with_speaker_tag,
    )
    from fish_speech.models.text2semantic.fish_omni.model import AUDIO_DECODER_PREFIXES, iter_checkpoint_tensors
    from fish_speech.models.text2semantic.fish_omni.moe import dequantize_fp8_blocks

    torch.set_grad_enabled(False)
    with open(args.model / "config.json", encoding="utf-8") as f:
        cfg = FishOmniConfig.from_dict(json.load(f))
    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))

    rows = build_drama3_prompt(TEXT, tok, cfg)
    segments = [
        "<|im_start|>system\n",
        DRAMA3_TTS_INSTRUCTION,
        DRAMA3_MESSAGE_END,
        "<|im_start|>user\n",
        with_speaker_tag(TEXT),
        DRAMA3_MESSAGE_END,
        DRAMA3_GENERATION_START,
    ]
    ids = [i for s in segments for i in tok.encode(s).ids]
    assert rows[:, 0].tolist() == ids, "prompt builder disagrees with the segment list"
    assert not rows[:, 1:].any() and max(ids) < cfg.semantic_start_token_id, "expected text rows only"

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "prompt_tokens.txt").write_text("".join(f"{i}\n" for i in ids))
    with open(args.out / "prompt_tokens.json", "w", encoding="utf-8") as f:
        json.dump({"text": TEXT, "segments": segments, "ids": ids}, f, ensure_ascii=False, indent=1)
    print(f"prompt: {len(ids)} tokens")

    t0 = time.perf_counter()
    with torch.device("meta"):
        model = FishOmniBackbone(cfg.text, fp8_experts=False, dtype=torch.bfloat16)
    model.materialize(["cpu"])
    load_backbone_weights(model, iter_checkpoint_tensors(args.model, skip_prefixes=AUDIO_DECODER_PREFIXES))
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"load: {load_s:.1f}s, peak {peak_gb():.1f} GiB")

    # Spot-check one dequantized expert against the raw FP8 tensors.
    experts = model.model.layers[0].mlp.experts
    assert experts.w13_weight.dtype == torch.bfloat16
    with open(args.model / "model.safetensors.index.json", encoding="utf-8") as f:
        wmap = json.load(f)["weight_map"]
    key = "model.language_model.layers.0.mlp.experts.7.up_proj.weight"
    with safe_open(str(args.model / wmap[key]), framework="pt", device="cpu") as f:
        w, s = f.get_tensor(key), f.get_tensor(key + "_scale_inv")
    ref = dequantize_fp8_blocks(w, s, torch.float32).to(torch.bfloat16)
    rows_up = experts.w13_weight.shape[1] // 2
    print(f"expert check: raw {w.dtype}, loaded {experts.w13_weight.dtype}, "
          f"match {torch.equal(experts.w13_weight[7, rows_up:], ref)}")

    n = len(ids)
    with torch.inference_mode():
        t0 = time.perf_counter()
        model.setup_caches(max_batch_size=1, max_seq_len=n)
        x = model.embed(torch.tensor(ids))
        hidden = model(inputs_embeds=x[None], input_pos=torch.arange(n))[0]
        last = hidden[-1:]
        logits = model.compute_logits(last)[0].float().numpy()
        logits32 = (last.float() @ model.lm_head.weight.float().T)[0].numpy()
        fwd_s = time.perf_counter() - t0
    print(f"forward: {fwd_s:.1f}s, peak {peak_gb():.1f} GiB")

    np.save(args.out / "ref_logits.npy", logits.astype(np.float32))
    np.save(args.out / "ref_logits_fp32.npy", logits32.astype(np.float32))
    np.save(args.out / "ref_hidden.npy", last[0].float().numpy())

    def line(i):
        return f"{i:7d} {logits[i]:10.4f} {tok.decode([int(i)], skip_special_tokens=False)!r}"

    order = np.argsort(-logits, kind="stable")
    rank = int((logits > logits[EOS]).sum()) + 1
    allowed = np.array([EOS, *range(cfg.semantic_start_token_id, cfg.semantic_end_token_id + 1)])
    sub = allowed[np.argsort(-logits[allowed], kind="stable")[:10]]
    out = ["top 20 over the full vocab (id, logit, token)"]
    out += [line(i) for i in order[:20]]
    out += ["", f"eos {EOS} logit {logits[EOS]:.4f} rank {rank} of {len(logits)}", ""]
    out += [f"top 10 over eos + semantic {cfg.semantic_start_token_id}..{cfg.semantic_end_token_id}"]
    out += [line(i) for i in sub]
    out += ["", f"load {load_s:.1f}s, forward {fwd_s:.1f}s, peak {peak_gb():.1f} GiB"]
    text = "\n".join(out) + "\n"
    (args.out / "ref_top.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
