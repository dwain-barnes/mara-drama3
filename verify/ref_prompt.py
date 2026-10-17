"""Reference-prompt variant of ref_tower.py: build a prompt with reference and history audio, run it, dump the tower.

No frames are generated. The prompt's own audio rows give the teacher-forced tower something to predict.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ref_tower sets CUDA_VISIBLE_DEVICES and the fp8 switch, so it goes before torch
from ref_tower import TEXT, peak_gb

import torch


def read_codes(path):
    codes = np.fromfile(path, dtype="<i4").reshape(-1, 10)
    return torch.from_numpy(codes.astype(np.int64)).T, Path(path).with_suffix(".txt").read_text(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--fish-speech", type=Path, required=True)
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--history", action="append", default=[])
    ap.add_argument("--out-prefix", default="ref_prompt")
    ap.add_argument("--ids-only", action="store_true")
    args = ap.parse_args()
    out = Path(__file__).parent / args.out_prefix

    sys.path.insert(0, str(args.fish_speech))
    from tokenizers import Tokenizer

    from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone, load_backbone_weights
    from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
    from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt
    from fish_speech.models.text2semantic.fish_omni.model import AUDIO_DECODER_PREFIXES, iter_checkpoint_tensors
    from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder, load_audio_decoder_weights

    torch.set_grad_enabled(False)
    with open(args.model / "config.json", encoding="utf-8") as f:
        cfg = FishOmniConfig.from_dict(json.load(f))
    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))

    refs = [read_codes(p) for p in args.ref]
    hist = [read_codes(p) for p in args.history]
    frames = build_drama3_prompt(
        args.text,
        tok,
        cfg,
        prompt_texts=[t for _, t in refs],
        prompt_tokens=[c for c, _ in refs],
        history=[(t, c) for c, t in hist],
    )
    P = len(frames)
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads, P={P}", flush=True)

    meta = {
        "P": P,
        "text": args.text,
        "refs": args.ref,
        "history": args.history,
        "prompt_ids": frames[:, 0].tolist(),
    }
    np.save(f"{out}_frames.npy", frames.numpy())
    if args.ids_only:
        Path(f"{out}_meta.json").write_text(json.dumps(meta, indent=1))
        return

    t0 = time.perf_counter()
    with torch.device("meta"):
        backbone = FishOmniBackbone(cfg.text, fp8_experts=False, dtype=torch.bfloat16)
        tower = FishOmniAudioDecoder(cfg, dtype=torch.bfloat16)
    backbone.materialize(["cpu"])
    load_backbone_weights(backbone, iter_checkpoint_tensors(args.model, skip_prefixes=AUDIO_DECODER_PREFIXES))
    tower.to_empty(device="cpu")
    load_audio_decoder_weights(
        tower, iter_checkpoint_tensors(args.model, skip_prefixes=("model.language_model.", "lm_head.weight"))
    )
    backbone.eval()
    tower.eval()
    load_s = time.perf_counter() - t0
    print(f"load: {load_s:.1f}s, peak {peak_gb():.1f} GiB", flush=True)

    band = torch.arange(cfg.semantic_start_token_id, cfg.semantic_end_token_id + 1)
    cand = torch.cat((torch.tensor([cfg.eos_token_id]), band))
    pos = torch.arange(P)

    with torch.inference_mode():
        t0 = time.perf_counter()
        backbone.setup_caches(max_batch_size=1, max_seq_len=P)
        tower.setup_caches(P)
        emb = tower.embed_backbone(frames, backbone.embed(frames[:, 0]))
        hidden = backbone(inputs_embeds=emb[None], input_pos=pos)[0]
        bb_s = time.perf_counter() - t0
        print(f"backbone: {bb_s:.1f}s, peak {peak_gb():.1f} GiB", flush=True)

        t0 = time.perf_counter()
        last = backbone.compute_logits(hidden[-1:])[0, cand].float()
        top = last.topk(5)

        R = P - 1
        h = hidden[:R]
        cur, nxt = frames[:R], frames[1:P]
        sem = tower.is_semantic(nxt[:, 0])
        unseen = tower.embedding_sum(cur, 1 + tower.config.vq_backbone_codebooks, tower.config.num_codebooks)
        unseen = torch.where(tower.is_semantic(cur[:, 0])[:, None], unseen, 0.0)
        layers, logits = [], []
        for i, layer in enumerate(tower.shared_layers):
            h = layer(h, nxt[:, 1], sem, pos[:R], unseen if i == 0 else None)
            layers.append(h)
        for i, (layer, head) in enumerate(zip(tower.residual_layers, tower.codebook_heads)):
            h = layer(h, nxt[:, i + 1], sem, pos[:R])
            layers.append(h)
            logits.append(head(h))
        tf_s = time.perf_counter() - t0
    print(f"teacher-forced pass: {tf_s:.1f}s, peak {peak_gb():.1f} GiB", flush=True)

    logits = torch.stack(logits, 1)
    ok = (logits.argmax(-1) == nxt[:, 2:11])[sem]
    per_head = ok.float().mean(0).tolist()
    print(f"audio rows {int(sem.sum())} of {R}; tower heads agree {int(ok.sum())}/{ok.numel()}, per head "
          + " ".join(f"{a:.3f}" for a in per_head), flush=True)

    def f32(t):
        return t.float().numpy()

    np.save(f"{out}_hidden.npy", f32(hidden))
    np.save(f"{out}_layers.npy", f32(torch.stack(layers)))
    np.save(f"{out}_logits.npy", f32(logits))
    np.save(f"{out}_semantic.npy", sem.numpy())
    meta.update(
        {
            "top5_ids": cand[top.indices].tolist(),
            "top5_values": top.values.tolist(),
            "audio_rows": int(sem.sum()),
            "tower_agree": f"{int(ok.sum())}/{ok.numel()}",
            "tower_agree_per_head": per_head,
            "load_s": load_s,
            "backbone_s": bb_s,
            "teacher_forced_s": tf_s,
            "peak_gib": peak_gb(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
        }
    )
    Path(f"{out}_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
