"""Encode a reference wav to a .codes file with fish-speech's DAC codec, on the CPU, as fish does.

usage: python encode_reference.py MODEL_DIR in.wav out.codes [--text "what is said"]
out.codes is int32 rows of 10 codes (the format llama-tts reads); --text writes out.txt beside it.
Run from the verify directory.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from decode_codes import load_codec, read_wav


def main(args):
    text = None
    if "--text" in args:
        i = args.index("--text")
        text = args[i + 1]
        del args[i : i + 2]
    if len(args) != 3:
        sys.exit(__doc__)
    model, wav, out = Path(args[0]), args[1], Path(args[2])

    codec = load_codec(model / "codec.pth")
    x, rate = read_wav(wav)
    if rate != codec.sample_rate:
        import torchaudio

        x = torchaudio.functional.resample(torch.from_numpy(x), rate, codec.sample_rate).numpy()

    audio = torch.from_numpy(x)[None, None]
    with torch.inference_mode():
        codes, _ = codec.encode(audio, torch.tensor([audio.shape[-1]]))
    codes = codes[0].T.numpy().astype("<i4")
    out.parent.mkdir(parents=True, exist_ok=True)
    codes.tofile(out)
    if text is not None:
        out.with_suffix(".txt").write_text(text, encoding="utf-8")
    print(f"{out}: {len(codes)} frames, {len(codes) * 2048 / codec.sample_rate:.2f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
