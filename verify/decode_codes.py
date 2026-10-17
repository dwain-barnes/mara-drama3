"""Decode Drama 3 codes to a wav with fish-speech's DAC codec, on the CPU.

usage: python decode_codes.py MODEL_DIR file.codes ...      int32 rows of 10 codes (llama-tts, MTMD_DRAMA3_CODES_OUT)
       python decode_codes.py MODEL_DIR --reference         the greedy frames in ref_tower_frames.npy
       python decode_codes.py MODEL_DIR --roundtrip x.wav   encode and decode a wav, as a check of the codec itself
MODEL_DIR holds codec.pth (D:\\models\\drama-3).
Wavs go next to the input (.dac.wav, or .roundtrip.wav), the reference one to out/ref_frames.wav.
Run from D:\\drama3\\verify.

The venv has no descript-audio-codec or audiotools, so the few pieces of them that modded_dac.py and rvq.py
import are defined here. They follow descript's dac.nn.layers and dac.nn.quantize.
"""
import os

# before torch is imported: the codec runs on the CPU only, and temporary files stay on D: with the project
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TEMP"] = r"D:\tmp"
os.environ["TMP"] = r"D:\tmp"

import functools
import importlib
import json
import sys
import types
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

FISH = Path(r"D:\drama3\third_party\fish-speech")


def WNConv1d(*args, **kwargs):
    return nn.utils.weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return nn.utils.weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)
        return x.reshape(shape)


class VectorQuantize(nn.Module):
    def __init__(self, input_dim, codebook_size, codebook_dim):
        super().__init__()
        self.codebook_size = codebook_size
        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def decode_code(self, ids):
        return F.embedding(ids, self.codebook.weight).transpose(1, 2)

    def forward(self, z):
        z_e = self.in_proj(z)
        enc = F.normalize(z_e.transpose(1, 2).reshape(-1, z_e.shape[1]))
        book = F.normalize(self.codebook.weight)
        ids = (enc @ book.t()).argmax(1).reshape(z_e.shape[0], -1)
        z_q = self.out_proj(self.decode_code(ids))
        zero = torch.zeros(())
        return z_q, zero, zero, ids, z_e


class ResidualVectorQuantize(nn.Module):
    def __init__(self, input_dim=512, n_codebooks=9, codebook_size=1024, codebook_dim=8, quantizer_dropout=0.0):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.quantizers = nn.ModuleList(
            VectorQuantize(input_dim, codebook_size, codebook_dim) for _ in range(n_codebooks)
        )

    def forward(self, z, n_quantizers=None):
        z_q, residual, codes, latents = 0, z, [], []
        for q in self.quantizers:
            z_q_i, _, _, ids, z_e = q(residual)
            z_q = z_q + z_q_i
            residual = residual - z_q_i
            codes.append(ids)
            latents.append(z_e)
        zero = torch.zeros(())
        return z_q, torch.stack(codes, 1), torch.cat(latents, 1), zero, zero

    def from_codes(self, codes):
        z_q, z_p = 0, []
        for i in range(codes.shape[1]):
            p = self.quantizers[i].decode_code(codes[:, i])
            z_p.append(p)
            z_q = z_q + self.quantizers[i].out_proj(p)
        return z_q, torch.cat(z_p, 1), codes


class CodecMixin:
    def get_delay(self):
        return 0


def stub(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod


stub("audiotools", AudioSignal=None)
stub("audiotools.ml", BaseModel=nn.Module)
stub("dac")
stub("dac.model")
stub("dac.model.base", CodecMixin=CodecMixin)
stub("dac.nn")
stub("dac.nn.layers", Snake1d=Snake1d, WNConv1d=WNConv1d, WNConvTranspose1d=WNConvTranspose1d)
stub("dac.nn.quantize", ResidualVectorQuantize=ResidualVectorQuantize)
sys.path.insert(0, str(FISH))


# hydra's instantiate, as much of it as modded_dac_vq.yaml uses
def instantiate(node):
    if isinstance(node, list):
        return [instantiate(v) for v in node]
    if isinstance(node, str):
        # PyYAML reads 1e-5 as a string, OmegaConf as a float
        try:
            return float(node)
        except ValueError:
            return node
    if not isinstance(node, dict):
        return node
    kwargs = {k: instantiate(v) for k, v in node.items() if not k.startswith("_")}
    if "_target_" not in node:
        return kwargs
    module, name = node["_target_"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module), name)
    return functools.partial(cls, **kwargs) if node.get("_partial_") else cls(**kwargs)


# load_codec_model in fish_speech/models/text2semantic/inference.py, kept in float32
@torch.inference_mode()
def load_codec(path):
    from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

    cfg = yaml.safe_load((FISH / "fish_speech/configs/modded_dac_vq.yaml").read_text())
    codec = instantiate(cfg)
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    rebuilt = {
        f"{name}.{buf}" if name else buf
        for name, module in codec.named_modules()
        for buf in module._non_persistent_buffers_set
    }
    codec.load_state_dict({k: v for k, v in sd.items() if k not in rebuilt}, strict=True)
    codec.eval()
    for module in list(codec.modules()):
        if is_parametrized(module, "weight"):
            remove_parametrizations(module, "weight", leave_parametrized=True)
        elif hasattr(module, "weight_g"):
            nn.utils.remove_weight_norm(module)
    return codec.float()


def write_wav(path, audio, rate):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2, "16-bit wav only"
        rate, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    return x.reshape(-1, ch).mean(1), rate


@torch.inference_mode()
def decode(codec, codes, path):
    # codes: [frames, 10], c0 in [0, 4096), c1..c9 in [0, 1024)
    audio = codec.from_indices(torch.from_numpy(codes.T.astype(np.int64))[None])[0, 0].numpy()
    write_wav(path, audio, codec.sample_rate)
    print(f"{path}: {len(codes)} frames, {len(audio)} samples, {len(audio) / codec.sample_rate:.2f} s, "
          f"peak {np.abs(audio).max():.3f}, rms {np.sqrt(np.mean(audio**2)):.4f}")
    return audio


def log_spectrum(a, n):
    frames = a[: n // 2048 * 2048].reshape(-1, 2048) * np.hanning(2048)
    return np.log(np.abs(np.fft.rfft(frames, axis=1)) + 1e-4)


def main(args):
    if len(args) < 2 or (args[1] == "--roundtrip" and len(args) < 3):
        sys.exit(__doc__)
    torch.set_num_threads(os.cpu_count())
    codec = load_codec(Path(args[0]) / "codec.pth")
    args = args[1:]
    if args[0] == "--reference":
        # the last K rows are the generated frames, [main, c0..c9]
        meta = json.loads(Path("ref_tower_meta.json").read_text())
        frames = np.load("ref_tower_frames.npy")
        decode(codec, frames[meta["P"] - meta["K"] : meta["P"], 1:11], Path("out/ref_frames.wav"))
    elif args[0] == "--roundtrip":
        x, rate = read_wav(args[1])
        assert rate == codec.sample_rate, f"expected {codec.sample_rate} Hz, got {rate}"
        with torch.inference_mode():
            codes, n = codec.encode(torch.from_numpy(x)[None, None])
        codes = codes[0, :, : n[0]].T.numpy()
        y = decode(codec, codes, Path(args[1]).with_suffix(".roundtrip.wav"))
        m = min(len(x), len(y))
        sx, sy = log_spectrum(x, m), log_spectrum(y, m)
        print(f"log-spectrum correlation with the input: {np.corrcoef(sx.ravel(), sy.ravel())[0, 1]:.3f}")
    else:
        for p in map(Path, args):
            n = json.loads(Path(str(p) + ".json").read_text())["frames"]
            codes = np.fromfile(p, dtype="<i4").reshape(n, 10)
            decode(codec, codes, p.with_suffix(".dac.wav"))


if __name__ == "__main__":
    main(sys.argv[1:])
