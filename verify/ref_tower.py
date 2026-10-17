"""Reference codebook tower dump for Drama 3, BF16 on CPU, greedy and deterministic."""

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

TEXT = "Hello there, how are you today?"


def peak_gb():
    mem = psutil.Process().memory_info()
    return getattr(mem, "peak_wset", mem.rss) / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--fish-speech", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()
    K = args.frames

    sys.path.insert(0, str(args.fish_speech))
    from tokenizers import Tokenizer

    from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone, load_backbone_weights
    from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
    from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt
    from fish_speech.models.text2semantic.fish_omni.model import AUDIO_DECODER_PREFIXES, iter_checkpoint_tensors
    from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder, load_audio_decoder_weights

    torch.set_grad_enabled(False)
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "ref_tower.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    with open(args.model / "config.json", encoding="utf-8") as f:
        cfg = FishOmniConfig.from_dict(json.load(f))
    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    prompt = build_drama3_prompt(TEXT, tok, cfg)
    assert not prompt[:, 1:].any(), "expected text rows only"
    n0 = len(prompt)
    P = n0 + K
    say(f"torch {torch.__version__}, {torch.get_num_threads()} threads, prompt {n0} rows, {K} frames, P={P}")

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
    say(f"load: {load_s:.1f}s, peak {peak_gb():.1f} GiB")

    def embed(rows):
        return tower.embed_backbone(rows, backbone.embed(rows[:, 0]))

    band = torch.arange(cfg.semantic_start_token_id, cfg.semantic_end_token_id + 1)
    cand = torch.cat((torch.tensor([cfg.eos_token_id]), band))

    frames = torch.zeros((P, 11), dtype=torch.long)
    frames[:n0] = prompt
    pos = torch.arange(P)
    codes_out = []
    eos_first = []

    with torch.inference_mode():
        t0 = time.perf_counter()
        backbone.setup_caches(max_batch_size=1, max_seq_len=P)
        tower.setup_caches(P)
        hidden = backbone(inputs_embeds=embed(frames[:n0])[None], input_pos=pos[:n0])[0]
        tower.prefill(hidden[: n0 - 1], pos[: n0 - 1], frames[: n0 - 1], frames[1:n0])
        hidden = hidden[-1:]
        for step in range(K):
            p = n0 - 1 + step
            logits = backbone.compute_logits(hidden)
            best = cand[logits[:, cand].argmax(-1)]
            if best.item() == cfg.eos_token_id:
                eos_first.append(step)
                best = band[logits[:, band].argmax(-1)]
            c0 = best - cfg.semantic_start_token_id
            sem = torch.ones(1, dtype=torch.bool)
            h = tower.run_shared(hidden, c0, sem, pos[p : p + 1], frames[p : p + 1])
            codes = [c0]
            for layer, head in zip(tower.residual_layers, tower.codebook_heads):
                h = layer(h, codes[-1], sem, pos[p : p + 1])
                codes.append(head(h).argmax(-1))
            row = torch.stack((best, *codes), -1)
            frames[p + 1] = row[0]
            codes_out.append([int(c) for c in row[0, 1:]])
            if step + 1 < K:
                hidden = backbone(inputs_embeds=embed(row)[None], input_pos=pos[p + 1 : p + 2])[0]
        gen_s = time.perf_counter() - t0
    say(f"generation: {gen_s:.1f}s, peak {peak_gb():.1f} GiB")
    for i, c in enumerate(codes_out):
        say(f"frame {i}: {c}")
    if eos_first:
        say(f"eos was the top candidate at steps {eos_first}; best semantic id used instead")

    with torch.inference_mode():
        t0 = time.perf_counter()
        backbone.setup_caches(max_batch_size=1, max_seq_len=P)
        tower.setup_caches(P)
        hidden = backbone(inputs_embeds=embed(frames)[None], input_pos=pos)[0]
        bb_logits = backbone.compute_logits(hidden[n0 - 1 : P - 1])

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
    say(f"teacher-forced pass: {tf_s:.1f}s, peak {peak_gb():.1f} GiB")

    logits = torch.stack(logits, 1)
    pred = logits.argmax(-1)
    rows = np.arange(n0 - 1, P - 1)
    tower_ok = (pred[rows] == frames[rows + 1, 2:11]).numpy()
    main_ok = (cand[bb_logits[:, cand].argmax(-1)] == frames[n0:P, 0]).numpy()
    say(f"tower heads agree: {int(tower_ok.sum())}/{tower_ok.size} codes, per row {tower_ok.sum(1).tolist()}")
    say(f"backbone main token agrees: {int(main_ok.sum())}/{main_ok.size}")
    if not (tower_ok.all() and main_ok.all()):
        say("mismatch between the incremental and the teacher-forced pass")

    def f32(t):
        return t.float().numpy()

    np.save(args.out / "ref_tower_frames.npy", frames.numpy())
    np.save(args.out / "ref_tower_hidden.npy", f32(hidden))
    np.save(args.out / "ref_tower_layers.npy", f32(torch.stack(layers)))
    np.save(args.out / "ref_tower_logits.npy", f32(logits))
    np.save(args.out / "ref_tower_unseen.npy", f32(unseen))
    np.save(args.out / "ref_tower_semantic.npy", sem.numpy())
    meta = {
        "P": P,
        "K": K,
        "text": TEXT,
        "prompt_ids": prompt[:, 0].tolist(),
        "codes": codes_out,
        "load_s": load_s,
        "generation_s": gen_s,
        "teacher_forced_s": tf_s,
        "peak_gib": peak_gb(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "tower_agree": f"{int(tower_ok.sum())}/{tower_ok.size}",
        "main_agree": f"{int(main_ok.sum())}/{main_ok.size}",
    }
    (args.out / "ref_tower_meta.json").write_text(json.dumps(meta, indent=1))
    say(f"load {load_s:.1f}s, generation {gen_s:.1f}s, teacher-forced {tf_s:.1f}s, peak {peak_gb():.1f} GiB")


if __name__ == "__main__":
    main()
