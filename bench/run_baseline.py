import argparse
import array
import csv
import re
import statistics
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

FRAMES_PER_S = 21.5
SILENCE = 100  # int16 amplitude counted as silent
FIELDS = ["id", "layers", "seed", "ttfa_ms", "frames", "wall_ms", "proc_wall_ms", "fps",
          "rtf", "audio_s", "lead_sil", "trail_sil", "peak_vram_mib", "exit_code", "parts", "note"]

kv_re = re.compile(r"([A-Za-z_][\w.]*)(?:\s*[=:]\s*|\s+)(-?\d+(?:\.\d+)?)")


def poll_vram(gpu, stop, peak):
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", gpu],
                capture_output=True, text=True, timeout=5).stdout
            peak[0] = max(peak[0], int(out.split()[0]))
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            pass
        stop.wait(0.25)


def first_audio(stderr):
    for line in stderr.splitlines():
        if "[drama3.first_audio]" in line:
            return {k: float(v) for k, v in kv_re.findall(line)}
    return {}


def read_log(path):
    keys = {}
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return keys
    for line in text.splitlines():
        for k, v in kv_re.findall(line):
            if k.startswith("drama3.") or k == "session.wall_ms":
                keys.setdefault(k, []).append(float(v))
    return keys


def wav_stats(path):
    try:
        with wave.open(str(path), "rb") as w:
            rate, n, width = w.getframerate(), w.getnframes(), w.getsampwidth()
            ch = w.getnchannels()
            raw = w.readframes(n)
    except (OSError, wave.Error, EOFError):
        return None
    dur = n / rate if rate else 0.0
    if width != 2 or n == 0:
        return dur, None, None
    s = array.array("h")
    s.frombytes(raw)
    if sys.byteorder == "big":
        s.byteswap()
    s = s[::ch]
    lead = 0
    while lead < len(s) and abs(s[lead]) < SILENCE:
        lead += 1
    trail = 0
    while trail < len(s) - lead and abs(s[-1 - trail]) < SILENCE:
        trail += 1
    return dur, lead / len(s), trail / len(s)


def run_one(args, gpu, row_id, text, layers, out_dir):
    tag = f"{row_id}_L{layers}"
    wav = out_dir / f"{tag}.wav"
    log = out_dir / f"{tag}.log"
    for p in (wav, log):
        p.unlink(missing_ok=True)
    cmd = [args.cli, "--task", "tts", "--family", "fish_drama3", "--backend", "cuda",
           "--device", str(args.device), "--mode", args.mode, "--model", args.model,
           "--text", text, "--seed", str(args.seed), "--max-tokens", "600",
           "--session-option", f"fish_drama3.cpu_expert_layers={layers}",
           "--session-option", "fish_drama3.context_length=2048",
           "--session-option", "fish_drama3.profile_first_audio=true",
           "--log-file", str(log), "--out", str(wav)]
    if args.voice_ref:
        cmd += ["--voice-ref", args.voice_ref, "--reference-text", args.reference_text]

    peak = [0]
    stop = threading.Event()
    t = threading.Thread(target=poll_vram, args=(gpu, stop, peak), daemon=True)
    t.start()
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        code, err = p.returncode, p.stderr
    except OSError as e:
        code, err = -1, str(e)
    proc_ms = (time.perf_counter() - t0) * 1000
    stop.set()
    t.join()
    (out_dir / f"{tag}.stderr.txt").write_text(err)

    fa = first_audio(err)
    keys = read_log(log)
    note = []
    frames = int(sum(keys.get("drama3.generated_frames", []))) or None
    sw = keys.get("session.wall_ms")
    if sw:
        wall = sw[-1]
    else:
        wall = proc_ms
        note.append("wall_ms is process time, includes model load")
    stats = wav_stats(wav)
    if not wav.exists() or stats is None:
        note.append("missing or unreadable wav")
    if code != 0:
        note.append("non-zero exit")
    if "max_tokens reached" in err:
        note.append("hit max_tokens")

    row = {"id": row_id, "layers": layers, "seed": args.seed, "ttfa_ms": fa.get("total_ms"),
           "frames": frames, "wall_ms": round(wall, 1), "proc_wall_ms": round(proc_ms, 1),
           "peak_vram_mib": peak[0] or None, "exit_code": code}
    if frames:
        row["fps"] = round(frames / (wall / 1000), 2)
        row["rtf"] = round(wall / (frames / FRAMES_PER_S) / 1000, 3)
    if stats:
        row["audio_s"] = round(stats[0], 3)
        if stats[1] is not None:
            row["lead_sil"], row["trail_sil"] = round(stats[1], 4), round(stats[2], 4)
    parts = {k: v for k, v in fa.items() if k.endswith("_ms")}
    parts.update({k: sum(v) for k, v in keys.items() if k.endswith("_ms") and k != "session.wall_ms"})
    row["parts"] = ";".join(f"{k}={v:g}" for k, v in parts.items())
    row["note"] = "; ".join(note)
    return row


def med(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return f"{statistics.median(v):.2f}" if v else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--layers", default="4,8,12")
    ap.add_argument("--lines", default=r"D:\drama3\bench\lines.tsv")
    ap.add_argument("--out-dir", default=r"D:\drama3\bench\out")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--voice-ref")
    ap.add_argument("--reference-text")
    ap.add_argument("--gpu-uuid")
    ap.add_argument("--mode", default="streaming")
    args = ap.parse_args()
    if args.voice_ref and not args.reference_text:
        ap.error("--voice-ref needs --reference-text")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu = f"--id={args.gpu_uuid}" if args.gpu_uuid else f"-i{args.device}"
    with open(args.lines, newline="", encoding="utf-8") as f:
        lines = [(r["id"], r["text"]) for r in csv.DictReader(f, delimiter="\t")]

    rows = []
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for layers in args.layers.split(","):
            for row_id, text in lines:
                for _ in range(args.repeat):
                    row = run_one(args, gpu, row_id, text, int(layers), out_dir)
                    rows.append(row)
                    w.writerow(row)
                    f.flush()
                    print(f"L{layers} {row_id}: exit={row['exit_code']} ttfa={row['ttfa_ms']} "
                          f"fps={row.get('fps')} {row['note']}", flush=True)

    print()
    print(f"{'layers':>6} {'runs':>4} {'ttfa_ms':>9} {'fps':>7} {'rtf':>7} {'audio_s':>8} {'vram_mib':>9}")
    for layers in args.layers.split(","):
        ok = [r for r in rows if r["layers"] == int(layers) and r["exit_code"] == 0]
        print(f"{layers:>6} {len(ok):>4} {med(ok, 'ttfa_ms'):>9} {med(ok, 'fps'):>7} "
              f"{med(ok, 'rtf'):>7} {med(ok, 'audio_s'):>8} {med(ok, 'peak_vram_mib'):>9}")


if __name__ == "__main__":
    main()
