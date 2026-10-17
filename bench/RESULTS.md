# Baseline: Drama 3 Q4 on one RTX 3090 Ti, 2026-10-16

All runs used our own conversion of fishaudio/drama-3 (revision e0a9218a): Q4_K routed experts, Q8_0
backbone and embedding, FP16 tower and codec, 22.9 GiB. Engine: audio.cpp main at commit 624cf918,
built for sm86. Only the 3090 Ti was visible to the process. No voice reference was used.

## Memory

| CPU expert layers | peak VRAM |
|---|---|
| 4 | 21.8 GiB (serving), 21.75 GiB (CLI) |
| 6 | 20.9 GiB |
| 12 | 18.3 GiB |

Each expert layer moved to the CPU frees about 430 MiB. With a 1.7B to 4B dialogue LLM on the same
card, 8 to 10 CPU layers will be needed.

## Speed

Steady-state decode, per 46.4 ms frame: backbone 13 to 15 ms, tower 5 to 6 ms, codec 4 to 6 ms,
about 22 to 24 ms in total (real-time factor 0.5). Moving layers from 12 to 4 on the CPU changed the
frame time by about 1 ms, so the CPU layers are not the bottleneck at this count.

Time to first audio:

| condition | prompt rows | first audio |
|---|---|---|
| cold process, CLI, 16 lines, 12 CPU layers | 24 to 60 | 650 to 920 ms (median 753) |
| cold process, CLI, 4 CPU layers | 26 to 50 | 590 to 770 ms |
| warm server, first request | 26 | 2.5 s (one-off warm-up) |
| warm server, short text | 26 | 136 ms |
| warm server, long text | 50 | 208 ms |

Warm prefill is about 4 ms per prompt row; cold it is about 11 ms per row plus graph capture.
Everything after prefill (tower prefill, first codec window) is about 10 ms warm.

## Quality

All 28 CLI clips transcribe word-perfect with Whisper small.en, and no tag text was spoken aloud.
One of 29 renders stopped early: "Right, I had a look at the logs..." produced "Right." and then
end-of-speech after 24 frames. The GGUF README for the community conversion reports the same failure.

Loudness differed by direction: `[excited]` -14 dBFS, `[warm]` and `[serious]` about -15, neutral
-18, the `[sighs]` line -27. The `<whisper>` line was -18 dBFS, the same as neutral, so the paired
whisper tag may not have taken. These need listening; the clips are in `out/L12`.

The same seed gave a different take every time in server mode, and "Yeah, that works." ranged from
1.0 to 3.0 s of audio across six requests.

## Against the design's kill criteria

- Frame p50 under 35 ms: passes, 22 to 24 ms.
- First audio p50 under 300 ms with a warm model: passes, 136 to 208 ms, and this is without the
  saved-prefix trick the design relies on.
- Fits in 24 GiB with room for a small LLM: passes with 8 or more expert layers on the CPU.
- Q4 keeps the expression: not settled. Words are intact; delivery needs a blind listen against
  the hosted model.

## Open points for the llama.cpp port

- Forbid end-of-speech until a frame budget proportional to the text length is met; the early stop
  after a short first clause would hurt a clause-chunked pipeline most.
- Cold prefill at 11 ms per row is audio.cpp's cost. llama.cpp's batched prefill on the same card
  should be measured early.
- Test whether `<whisper>` and the other paired tags do anything at Q4, and whether a voice reference
  carries affect, before building the reference bank.

Files: `out/L12` (16 lines, 12 CPU layers), `out/sweep` (6 lines at 4 and 6 layers), `out/smoke`,
`server/` (warm test, RESULTS.md, WAVs), `out/transcripts.txt`.
