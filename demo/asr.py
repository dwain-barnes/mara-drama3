import os
import sys
import time
import wave

import numpy as np
import sherpa_onnx

RATE = 16000
FRAME = 512


def resample(x, src, dst=RATE):
    if src == dst:
        return x
    n = int(round(len(x) * dst / src))
    spec = np.fft.rfft(x)
    out = np.zeros(n // 2 + 1, dtype=spec.dtype)
    m = min(len(out), len(spec))
    out[:m] = spec[:m]
    return (np.fft.irfft(out, n) * (n / len(x))).astype(np.float32)


def read_wav(path):
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1)
    return resample(pcm.astype(np.float32) / 32768, rate)


class Recognizer:
    def __init__(self, model_dir, threads=4):
        t0 = time.perf_counter()
        path = lambda name: os.path.join(model_dir, name)
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=path("encoder.int8.onnx"),
            decoder=path("decoder.int8.onnx"),
            joiner=path("joiner.int8.onnx"),
            tokens=path("tokens.txt"),
            num_threads=threads,
            model_type="nemo_transducer",
        )
        self.load_s = time.perf_counter() - t0
        self.last_s = 0.0

    def transcribe(self, samples):
        t0 = time.perf_counter()
        stream = self.rec.create_stream()
        stream.accept_waveform(RATE, samples)
        self.rec.decode_stream(stream)
        self.last_s = time.perf_counter() - t0
        return stream.result.text.strip()


class Vad:
    def __init__(self, model, threshold=0.5, min_silence=0.5, min_speech=0.25):
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = model
        cfg.silero_vad.threshold = threshold
        cfg.silero_vad.min_silence_duration = min_silence
        cfg.silero_vad.min_speech_duration = min_speech
        cfg.silero_vad.window_size = FRAME
        cfg.sample_rate = RATE
        self.vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)

    def feed(self, frame):
        self.vad.accept_waveform(frame)
        while not self.vad.empty():
            yield np.array(self.vad.front.samples, dtype=np.float32)
            self.vad.pop()

    def flush(self):
        self.vad.flush()
        while not self.vad.empty():
            yield np.array(self.vad.front.samples, dtype=np.float32)
            self.vad.pop()

    def reset(self):
        self.vad.reset()


def main():
    asr_dir = r"D:\models\parakeet-tdt-0.6b-v3"
    vad_path = r"D:\models\silero-vad\silero_vad.onnx"
    clips = sys.argv[1:] or [
        r"D:\drama3\demo\bank\neutral.wav",
        r"D:\realtime_ai_avatar\p283_023_enhanced.wav",
        r"D:\drama3\demo\out\tag_ab_q4\whisper_tag.wav",
    ]
    rec = Recognizer(asr_dir)
    print("load %.2f s" % rec.load_s)
    for path in clips:
        x = read_wav(path)
        text = rec.transcribe(x)
        print("%s (%.1f s audio, %.2f s): %s" % (os.path.basename(path), len(x) / RATE, rec.last_s, text))

    vad = Vad(vad_path)
    x = np.concatenate([read_wav(clips[min(1, len(clips) - 1)]), np.zeros(RATE, dtype=np.float32)])
    segs = []
    for i in range(0, len(x) - FRAME + 1, FRAME):
        segs.extend(vad.feed(x[i:i + FRAME]))
    print("vad: %d segments, lengths %s s" % (len(segs), [round(len(s) / RATE, 2) for s in segs]))


if __name__ == "__main__":
    main()
