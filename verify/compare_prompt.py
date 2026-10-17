r"""Compare the prefill head logits of pipeline.exe against ref_prompt.py, on the rows whose next row is audio.

usage: python compare_prompt.py logits.bin [--prefix ref_prompt] [--hidden hidden.bin]   (run from D:\drama3\verify)
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("logits")
ap.add_argument("--prefix", default="ref_prompt")
ap.add_argument("--hidden")
args = ap.parse_args()

ref = np.load(f"{args.prefix}_logits.npy")  # [P-1, 9, 4096]
sem = np.load(f"{args.prefix}_semantic.npy")
frames = np.load(f"{args.prefix}_frames.npy")  # [P, 11], rows [main, c0..c9]
n_rows, n_heads, n_vocab = ref.shape

got = np.fromfile(args.logits, dtype=np.float32).reshape(-1, n_heads, n_vocab)
if len(got) != n_rows:
    raise SystemExit(f"{args.logits}: {len(got)} rows, the reference has {n_rows}")

rows = np.flatnonzero(sem)
r, g = ref[rows], got[rows]
# head i predicts c_{i+1} of the next row
want = np.stack([frames[rows + 1, i + 2] for i in range(n_heads)], axis=1)

print(f"{args.logits}: {len(rows)} rows where the next row is audio, of {n_rows}")
print("  head   snr_db  top1        top5         ref_hit  our_hit")
t1 = t5 = ref_hit = our_hit = 0
snrs = []
for h in range(n_heads):
    x, y = r[:, h], g[:, h]
    snr = 20 * np.log10(np.linalg.norm(x) / np.linalg.norm(x - y))
    top1 = int((x.argmax(-1) == y.argmax(-1)).sum())
    k5 = np.sort(np.argpartition(-x, 5, axis=-1)[:, :5], -1)
    g5 = np.sort(np.argpartition(-y, 5, axis=-1)[:, :5], -1)
    top5 = sum(len(set(a) & set(b)) for a, b in zip(k5, g5))
    rh = int((x.argmax(-1) == want[:, h]).sum())
    oh = int((y.argmax(-1) == want[:, h]).sum())
    n = len(rows)
    print(f"  {h:4d}  {snr:7.2f}  {top1:3d} {top1 / n:5.3f}  {top5:4d}/{5 * n} {top5 / (5 * n):5.3f}  {rh:4d}     {oh:4d}")
    snrs.append(snr)
    t1 += top1
    t5 += top5
    ref_hit += rh
    our_hit += oh
n = len(rows) * n_heads
print(f"  mean  {np.mean(snrs):7.2f}  {t1:4d} {t1 / n:5.3f}  {t5:4d}/{5 * n} {t5 / n / 5:5.3f}  {ref_hit:4d}     {our_hit:4d}   (of {n})")

if args.hidden:
    ref_h = np.load(f"{args.prefix}_hidden.npy")
    h = np.fromfile(args.hidden, dtype=np.float32).reshape(ref_h.shape)
    cos = (ref_h * h).sum(-1) / (np.linalg.norm(ref_h, axis=-1) * np.linalg.norm(h, axis=-1))
    audio = frames[:, 1:].any(1)
    a = np.flatnonzero(audio)
    a0, a1 = a[0], a[-1]
    print(f"hidden {args.hidden}: cosine over {len(cos)} rows, audio rows {a0}..{a1}")
    print(f"  min {cos.min():.5f} at row {cos.argmin()}; min over the audio rows {cos[audio].min():.5f}; mean {cos.mean():.5f}")
    print(f"  first 3 audio rows: {' '.join(f'{c:.5f}' for c in cos[a0:a0 + 3])}")
    print(f"  first 3 rows after the audio: {' '.join(f'{c:.5f}' for c in cos[a1 + 1:a1 + 4])}")
    print(f"  last row: {cos[-1]:.5f}")
    print(f"  rows below 0.99: {int((cos < 0.99).sum())}")
