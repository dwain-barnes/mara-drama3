"""Compare tower.exe dumps against the fish-speech BF16 tower reference.

usage: python compare_tower.py outdir   (run from D:\\drama3\\verify)
"""
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
ref_layers = np.load("ref_tower_layers.npy")  # [13, 34, 2048]
ref_logits = np.load("ref_tower_logits.npy")  # [34, 9, 4096]
n_layer, R, n_embd = ref_layers.shape
n_heads = ref_logits.shape[1]
# the rows whose next row is audio, the others predict text rows
audio = np.flatnonzero(np.load("ref_tower_semantic.npy"))
K = len(audio)


def snr(ref, got):
    err = np.sum((got - ref) ** 2)
    return 10 * np.log10(np.sum(ref**2) / err) if err > 0 else float("inf")


def load(run):
    layers = []
    for il in range(n_layer):
        x = np.fromfile(out / run / f"layer_{il:02d}.bin", dtype=np.float32)
        layers.append(x.reshape(R, n_embd))
    logits = np.fromfile(out / f"logits_{run}.bin", dtype=np.float32).reshape(ref_logits.shape)
    return np.stack(layers), logits


runs = {r: load(r) for r in ("a", "b")}

for run, (layers, logits) in runs.items():
    print(f"run {run}: layers vs reference")
    print("  layer   snr_all  maxabs_all  snr_audio  maxabs_audio")
    for il in range(n_layer):
        r, g = ref_layers[il], layers[il]
        print(f"  {il:5d}  {snr(r, g):8.2f}  {np.abs(g - r).max():10.4f}  {snr(r[audio], g[audio]):10.2f}  {np.abs(g[audio] - r[audio]).max():12.4f}")

    print(f"run {run}: logits on the {K} rows before an audio row ({audio[0]}..{audio[-1]}) vs reference")
    print("  head      snr    maxabs  top1  top5")
    t1 = t5 = 0
    for h in range(n_heads):
        r, g = ref_logits[audio, h], logits[audio, h]
        top1 = int((r.argmax(-1) == g.argmax(-1)).sum())
        rk = np.argsort(-r, -1)[:, :5]
        gk = np.argsort(-g, -1)[:, :5]
        top5 = sum(len(set(a) & set(b)) for a, b in zip(rk, gk))
        t1 += top1
        t5 += top5
        print(f"  {h:4d}  {snr(r, g):7.2f}  {np.abs(g - r).max():8.4f}  {top1:2d}/{K}  {top5:2d}/{5 * K}")
    print(f"  all heads: top1 {t1}/{n_heads * K}, top5 overlap {t5}/{5 * n_heads * K}")

    # frame rows are [main, c0..c9], the heads predict c1..c9
    ref_codes = np.load("ref_tower_frames.npy")[audio + 1, 2:2 + n_heads]
    got_codes = logits[audio].argmax(-1)
    print(f"  greedy codes vs the incremental reference frames: {int((got_codes == ref_codes).sum())}/{n_heads * K}")

print("a vs b: max abs difference")
la, ga = runs["a"]
lb, gb = runs["b"]
print("  layers " + " ".join(f"{np.abs(la[il] - lb[il]).max():.2e}" for il in range(n_layer)))
print(f"  logits {np.abs(ga - gb).max():.2e}")

rewind = out / "logits_rewind.bin"
if rewind.exists():
    lr = np.fromfile(rewind, dtype=np.float32).reshape(-1, *ref_logits.shape[1:])
    n_keep = R - len(lr)
    layers_r = [np.fromfile(out / "rewind" / f"layer_{il:02d}.bin", dtype=np.float32).reshape(R - n_keep, n_embd) for il in range(n_layer)]
    same_logits = np.array_equal(lr, gb[n_keep:])
    same_layers = all(np.array_equal(layers_r[il], lb[il][n_keep:]) for il in range(n_layer))
    print(f"rewind to {n_keep}, rows {n_keep}..{R - 1} again vs run b: logits {'identical' if same_logits else 'differ'}, "
          f"layers {'identical' if same_layers else 'differ'} (max abs {np.abs(lr - gb[n_keep:]).max():.2e})")
