"""Compare codec.exe output against the fp32 PyTorch DAC reference.

usage: python compare_codec.py outdir   (run from D:\\drama3\\verify)
"""
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
MIN_SNR = 40.0
MAX_STREAM_ERR = 1e-3


def snr(ref, got):
    err = np.sum((got - ref) ** 2)
    return 10 * np.log10(np.sum(ref**2) / err) if err > 0 else float("inf")


fail = False
print(f"thresholds: offline snr >= {MIN_SNR} dB, streamed vs offline max abs <= {MAX_STREAM_ERR}")
for name in ("hello", "ref6"):
    ref = np.load(f"out/codec/{name}_ref.npy").astype(np.float32).reshape(-1)
    off = np.fromfile(out / f"{name}_offline.bin", dtype=np.float32)
    st = np.fromfile(out / f"{name}_stream.bin", dtype=np.float32)
    same = len(ref) == len(off) == len(st)
    print(f"{name}: samples reference {len(ref)}, offline {len(off)}, streamed {len(st)}"
          f" ({'ok' if same else 'LENGTH MISMATCH'})")
    if not same:
        fail = True
        continue
    s = snr(ref, off)
    d = np.abs(st - off).max()
    print(f"  offline vs reference: snr {s:.2f} dB, max abs {np.abs(off - ref).max():.2e}")
    print(f"  streamed vs offline:  snr {snr(off, st):.2f} dB, max abs {d:.2e}")
    fail |= s < MIN_SNR or d > MAX_STREAM_ERR

sys.exit(1 if fail else 0)
