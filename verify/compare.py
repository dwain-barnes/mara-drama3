"""Compare a llama.cpp logits dump against the fish-speech BF16 reference."""
import sys
import numpy as np

ref = np.load("ref_logits.npy")
got = np.fromfile(sys.argv[1], dtype=np.float32)
assert got.shape == ref.shape, (got.shape, ref.shape)

eos = 248046
sem = np.arange(248099, 252195)
allowed = np.concatenate([[eos], sem])

def top(x, ids, k=10):
    o = ids[np.argsort(-x[ids])[:k]]
    return list(zip(o.tolist(), np.round(x[o], 3).tolist()))

def summary(name, ids):
    d = got[ids] - ref[ids]
    r = np.corrcoef(got[ids], ref[ids])[0, 1]
    pr = np.exp(ref[ids] - ref[ids].max()); pr /= pr.sum()
    pg = np.exp(got[ids] - got[ids].max()); pg /= pg.sum()
    kl = float((pr * (np.log(pr + 1e-12) - np.log(pg + 1e-12))).sum())
    print(f"{name}: n={len(ids)} max|d|={np.abs(d).max():.4f} mean|d|={np.abs(d).mean():.4f} "
          f"corr={r:.6f} kl(ref||got)={kl:.5f} argmax ref={ids[ref[ids].argmax()]} got={ids[got[ids].argmax()]}")
    tr = [i for i, _ in top(ref, ids)]
    tg = [i for i, _ in top(got, ids)]
    print(f"  top10 overlap {len(set(tr) & set(tg))}/10, top1 same: {tr[0] == tg[0]}")

summary("full vocab", np.arange(len(ref)))
summary("eos+semantic", allowed)
print("ref top", top(ref, allowed))
print("got top", top(got, allowed))
print(f"eos logit ref {ref[eos]:.4f} got {got[eos]:.4f}; eos rank in allowed ref {int((ref[allowed] > ref[eos]).sum())} got {int((got[allowed] > got[eos]).sum())}")
