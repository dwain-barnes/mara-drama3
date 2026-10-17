import argparse
import os
import struct
import subprocess
import sys
import time

import numpy as np

SAY = r"D:\drama3\demo\build\say.exe"
MMPROJ = r"D:\models\drama-3-own\llamacpp\drama-3-mmproj-q8_0.gguf"
REF = r"D:\drama3\verify\refs\p283.codes"
WHISPER_DIR = r"D:\tmp\whisper_verify"

SENTENCE = "I really did not expect that, and I have to say, it changes everything we planned for tomorrow."

VARIANTS = [
    ("plain", SENTENCE),
    ("whisper_tag", "<whisper>" + SENTENCE + "</whisper>"),
    ("whisper_dir", "[whispering] " + SENTENCE),
    ("excited", "[excited] " + SENTENCE),
    ("shout", "[angry, shouting] " + SENTENCE),
    ("serious", "[serious, measured] " + SENTENCE),
    ("sad", "[sad, quiet] " + SENTENCE),
    ("laugh", SENTENCE.replace("that,", "that, [laughs]", 1)),
]


def read_wav(path):
    b = open(path, "rb").read()
    i = 12
    while i < len(b):
        cid = b[i:i + 4]
        n = struct.unpack("<I", b[i + 4:i + 8])[0]
        if cid == b"fmt ":
            tag, ch, sr, _, _, bits = struct.unpack("<HHIIHH", b[i + 8:i + 24])
        elif cid == b"data":
            raw = b[i + 8:i + 8 + n]
            break
        i += 8 + n + (n & 1)
    if bits == 16:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif bits == 32 and tag in (3, 0xFFFE):
        x = np.frombuffer(raw, "<f4")
    else:
        sys.exit(f"{path}: unsupported wav format tag={tag} bits={bits}")
    return x.reshape(-1, ch).mean(1), sr


def db(v):
    return 20 * np.log10(max(v, 1e-9))


def f0_stats(x, sr):
    win = int(0.04 * sr)
    hop = int(0.02 * sr)
    lo, hi = int(sr / 400), int(sr / 60)
    f0 = []
    for s in range(0, len(x) - win, hop):
        w = x[s:s + win]
        if db(np.sqrt(np.mean(w ** 2))) < -35:
            continue
        w = w - w.mean()
        ac = np.correlate(w, w, "full")[win - 1:]
        if ac[0] <= 0:
            continue
        k = lo + np.argmax(ac[lo:hi + 1])
        if ac[k] / ac[0] < 0.3:
            continue
        f0.append(sr / k)
    if not f0:
        return float("nan"), float("nan")
    q1, med, q3 = np.percentile(f0, [25, 50, 75])
    return med, q3 - q1


def measure(x, sr):
    rms = np.sqrt(np.mean(x ** 2))
    n = int(0.5 * sr)
    if len(x) > n:
        loud = max(np.sqrt(np.mean(x[s:s + n] ** 2)) for s in range(0, len(x) - n, n // 5))
    else:
        loud = rms
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    hi_share = spec[freqs > 4000].sum() / spec.sum()
    med, iqr = f0_stats(x, sr)
    return dict(dur=len(x) / sr, rms=db(rms), peak=db(np.abs(x).max()), loud=db(loud),
                hi=hi_share, f0=med, iqr=iqr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone")
    ap.add_argument("outdir")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mmproj", default=MMPROJ)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    env = dict(os.environ, TEMP=r"D:\tmp")
    if a.cpu:
        env["CUDA_VISIBLE_DEVICES"] = "-1"

    log = open(os.path.join(a.outdir, "say.log"), "wb")
    cmd = [SAY, a.backbone, a.mmproj, "--ref", a.ref, "--seed", str(a.seed)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env)

    def ask(line):
        p.stdin.write(line.encode() + b"\n")
        p.stdin.flush()
        out = p.stdout.readline()
        if not out:
            sys.exit(f"say.exe exited (see {log.name})")
        return out.decode().rstrip("\r\n")

    t0 = time.time()
    first = p.stdout.readline().decode().strip()
    if first != "ready":
        sys.exit(f"say.exe said {first!r} (see {log.name})")
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)

    rows = []
    for name, text in VARIANTS:
        path = os.path.join(a.outdir, name + ".wav")
        ask("reset")
        t0 = time.time()
        res = ask(f"say\t{path}\t{text}").split("\t")
        wall = time.time() - t0
        if res[0] != "ok":
            print(f"{name}: {res}", flush=True)
            continue
        m = measure(*read_wav(path))
        m.update(name=name, frames=int(res[2]), secs=int(res[3]) / 1000, text=text)
        rows.append(m)
        print(f"{name}: {m['frames']} frames, {m['secs']:.0f} s (wall {wall:.0f} s)", flush=True)
    p.stdin.write(b"quit\n")
    p.stdin.flush()
    p.wait()

    head = f"{'variant':<12}{'frames':>7}{'dur_s':>7}{'rms':>8}{'peak':>8}{'loud0.5':>9}{'hi>4k':>8}{'f0_med':>8}{'f0_iqr':>8}{'render_s':>10}"
    lines = [head]
    for r in rows:
        lines.append(f"{r['name']:<12}{r['frames']:>7}{r['dur']:>7.2f}{r['rms']:>8.1f}{r['peak']:>8.1f}"
                     f"{r['loud']:>9.1f}{r['hi']:>8.3f}{r['f0']:>8.1f}{r['iqr']:>8.1f}{r['secs']:>10.0f}")
    table = "\n".join(lines)
    print(table, flush=True)

    wavs = [os.path.join(a.outdir, r["name"] + ".wav") for r in rows]
    wenv = dict(os.environ, HF_HOME="D:/hf-home", UV_CACHE_DIR="D:/uv/cache", HF_HUB_OFFLINE="1",
                UV_PYTHON_INSTALL_DIR="D:/uv/python", CUDA_VISIBLE_DEVICES="-1")
    tr = subprocess.run(["uv", "run", "--offline", "--python", "3.12", "--with", "faster-whisper",
                         "python", "-I", "tr.py"] + wavs,
                        cwd=WHISPER_DIR, env=wenv, capture_output=True, text=True)
    out = tr.stdout + tr.stderr if tr.returncode else tr.stdout

    with open(os.path.join(a.outdir, "table.txt"), "w") as f:
        f.write(table + "\n\ntranscripts\n" + out)
    print(out)


if __name__ == "__main__":
    main()
