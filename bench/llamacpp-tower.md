# Drama 3 codebook tower in llama.cpp

2026-10-16. Second llama.cpp piece: the tower (the 13-layer codebook decoder) as an mtmd gen-audio graph
with a persistent KV cache over time, checked against fish-speech. Same llama.cpp tree and build as
`llamacpp-backbone.md`, nothing committed.

## What was built

The converter writes the tower into its own mmproj GGUF (`conversion/drama3.py`, projector `drama3_gen`,
no vision or audio encoder in the file). Norm weights are stored with the +1 of the zero-centred RMS norm
already added. Files: `drama-3-tower-bf16.gguf` (2.28 GiB) and `drama-3-tower-q8_0.gguf` (1.21 GiB) in
`D:\models\drama-3-own\llamacpp\`.

On the llama.cpp side the graph is `tools/mtmd/models/drama3-gen.cpp`, with the plumbing in `clip.cpp`,
`mtmd.cpp` and `mtmd.h`. One call takes n consecutive timeline rows of backbone hidden state and runs all
13 layers over them at once. It writes K and V into a per-layer F16 cache with `set_rows`, attends over the
first `n_kv = min(n_ctx, pad(n_past + n, 256))` rows with a causal mask by absolute position, and returns
the 9 heads' logits for every row. Positions, cache rows, mask, codes and flags are graph inputs, so the
graph only changes shape with n and the 256-row bucket. It is teacher-forced for now: the caller passes
the codes of the frame each row predicts, and nothing is sampled on the graph.

The cache is an opaque `mtmd_gen_state` (`mtmd_gen_state_init(ctx, n_ctx)`, 26 KiB a row), passed in
`mtmd_gen_inp` next to the new per-row fields. `mtmd_gen_state_set_n_past` rewinds it. A gen-only mmproj
did not load before (mtmd required a vision or audio encoder); the guard now accepts it.

## Rerun

    cd D:\drama3\verify
    build\tower.exe <tower.gguf> D:\models\drama-3-own\llamacpp\drama-3-q4k-experts-q8-backbone.gguf out\cpu_bf16 [--gpu]
    .venv\Scripts\python.exe compare_tower.py out\cpu_bf16

`tower.cpp` loads the backbone only for the n_embd check, which takes a few seconds. It feeds rows 0..33
of the reference (`ref_tower.py`: the 29-row text prompt plus 6 greedy frames, fish-speech BF16 on the
CPU). Run a does all 34 rows in one call. Run b does rows 0..19 in one call, then one row per call. Then
run b's state is rewound to 25 and rows 25..33 are run again. `MTMD_DRAMA3_DUMP_DIR` makes the graph write
each layer's output. `--bench` is the timing mode below.

## Against fish-speech

Rows 28..33 are the ones whose next frame is audio, so only their logits mean anything (6 rows x 9 heads
= 54 codes). The layer SNR covers all 34 rows and these 6 rows separately. All numbers are from run a.

Layer output SNR, dB (all rows / rows 28..33):

| layer | CPU bf16 | CPU q8_0 | GPU bf16 | GPU q8_0 |
|---|---|---|---|---|
| 0 | 48.2 / 47.4 | 41.7 / 39.3 | 48.3 / 47.5 | 41.8 / 39.4 |
| 1 | 45.5 / 43.6 | 38.3 / 35.5 | 45.5 / 43.6 | 38.3 / 35.5 |
| 2 | 43.0 / 42.0 | 36.6 / 34.2 | 43.1 / 42.3 | 36.7 / 34.4 |
| 3 | 44.8 / 41.9 | 37.3 / 34.2 | 44.9 / 42.1 | 37.5 / 34.5 |
| 4 | 45.1 / 41.3 | 37.7 / 33.4 | 45.0 / 41.4 | 37.5 / 33.5 |
| 5 | 43.4 / 41.3 | 36.5 / 33.6 | 43.7 / 41.4 | 36.5 / 33.9 |
| 6 | 41.7 / 41.5 | 35.2 / 33.5 | 42.6 / 41.5 | 34.7 / 34.1 |
| 7 | 44.1 / 41.0 | 37.1 / 32.9 | 44.6 / 41.0 | 36.8 / 33.7 |
| 8 | 44.1 / 40.7 | 36.4 / 32.6 | 44.0 / 40.5 | 36.4 / 33.4 |
| 9 | 44.1 / 40.5 | 36.5 / 32.4 | 44.2 / 40.3 | 36.9 / 33.0 |
| 10 | 44.2 / 41.2 | 36.7 / 33.3 | 44.8 / 41.4 | 36.8 / 33.6 |
| 11 | 44.6 / 39.2 | 36.5 / 30.8 | 44.8 / 39.3 | 36.7 / 31.0 |
| 12 | 39.1 / 43.6 | 31.3 / 34.7 | 39.4 / 43.3 | 31.1 / 34.7 |

Head logits SNR on rows 28..33, dB:

| head (predicts) | CPU bf16 | CPU q8_0 | GPU bf16 | GPU q8_0 |
|---|---|---|---|---|
| 0 (c1) | 48.6 | 40.8 | 48.5 | 40.9 |
| 1 (c2) | 48.5 | 41.4 | 48.6 | 41.8 |
| 2 (c3) | 49.6 | 41.2 | 49.7 | 42.9 |
| 3 (c4) | 49.2 | 42.1 | 49.1 | 43.2 |
| 4 (c5) | 47.8 | 42.4 | 47.8 | 42.9 |
| 5 (c6) | 48.8 | 41.0 | 48.5 | 42.0 |
| 6 (c7) | 50.9 | 43.6 | 51.2 | 43.7 |
| 7 (c8) | 49.8 | 43.6 | 50.0 | 43.7 |
| 8 (c9) | 48.9 | 40.7 | 47.5 | 39.6 |

Argmax and top 5 against the reference's teacher-forced logits, rows 28..33:

| | top 1, run a | top 1, run b | top 5 shared, run a |
|---|---|---|---|
| CPU bf16 | 51/54 | 52/54 | 265/270 |
| CPU q8_0 | 45/54 | 45/54 | 257/270 |
| GPU bf16 | 52/54 | 52/54 | 264/270 |
| GPU q8_0 | 50/54 | 51/54 | 256/270 |

The reference does not agree with itself at this level: its batched and incremental passes disagree on
3 of the 54 codes (`ref_tower_meta.json`, tower_agree 51/54). Every CPU bf16 miss is an exact tie in the
reference's BF16 logits, or one BF16 step (0.0625). In one of those (row 31, head 3), the llama.cpp pick is
the code the incremental reference chose. Every CPU q8_0 miss is the reference's second choice, at most
0.19 below its first.

## Batching and rewind

Run a and run b run the same graph with different batching. Rows 0..19 come out bit-identical in every
configuration. For the rows run one at a time, CPU q8_0 is bit-identical as well. CPU bf16 differs by up to
2e-3 at layer 0, growing to 0.09 at layer 12, with 0.13 in the logits; the n = 1 BF16 matmul rounds
differently from the batched one. On the GPU the difference is 0.11 in the logits (bf16) and 0.38 (q8_0).

The rewind check: run b's state is set back to n_past 25, and rows 25..33 are run again one row per call.
The logits and all 13 layer outputs are bit-identical to run b's in all four configurations.

## Timings

Per call, the 34-row reference run, n_ctx 512 (so n_kv 256). GPU is the 3090 Ti; CPU is the 64-thread
host, default mmproj threads.

| | 34 rows, one call | rows 0..19, one call | one row, rows 22..33 | first one-row call |
|---|---|---|---|---|
| CPU bf16 | 121 ms | 57 ms | 23.5 to 28.3 ms | 26 ms |
| CPU q8_0 | 90 ms | 38 ms | 15.9 to 17.1 ms | 17 ms |
| GPU bf16 | 144 ms | 14 ms | 6.1 to 6.7 ms | 30 ms |
| GPU q8_0 | 49 ms | 11 ms | 4.3 to 5.7 ms | 34 ms |

The 34-row GPU call is the first call of the context and includes its allocation and flash-attention probe.
These runs also read back and write the 13 layer dumps after every call; the deep bench below does not.
audio.cpp's tower takes 4.7 ms a frame warm on the 3090 Ti, as 13 single-layer submissions.

### Deep bench

`tower --bench --gpu`: n_ctx 2048, a 300-row prefill from n_past 0, 200 single rows at depth 300..499
(n_kv 512 throughout), then a 130-row prefill at n_past 500. Each prefill is run once and then five more
times from the same n_past; the first call of a new shape also pays for the allocation. No dump.

Measured on the plain RTX 3090 at 0000:41:00.0, not the Ti: the Ti dropped off the bus during the first
300-row call of this bench and is lost until a reboot. The 34-row q8_0 run on the 3090 gives the same
run a result as on the Ti (top 1 50/54, top 5 256/270) and the same bit-identical rewind.

| | CUDA graphs | prefill 300, first | prefill 300, median | single p50 | p90 | p99 | max | prefill 130, first | prefill 130, median |
|---|---|---|---|---|---|---|---|---|---|
| q8_0 | on | 67.4 ms | 21.9 ms | 2.80 ms | 3.07 ms | 5.16 ms | 24.7 ms | 14.9 ms | 12.1 ms |
| q8_0 | off | 63.4 ms | 22.6 ms | 4.85 ms | 5.36 ms | 6.58 ms | 25.7 ms | 14.6 ms | 13.2 ms |
| bf16 | on | 165.2 ms | 32.1 ms | 3.77 ms | 3.93 ms | 6.86 ms | 25.5 ms | 23.6 ms | 21.1 ms |
| bf16 | off | 176.2 ms | 33.9 ms | 4.54 ms | 4.72 ms | 5.54 ms | 22.9 ms | 24.0 ms | 22.9 ms |

"Off" is `GGML_CUDA_DISABLE_GRAPHS=1`. The single-row p50 drops from 4.85 to 2.80 ms (q8_0) and from 4.54 to
3.77 ms (bf16) with graphs on, so CUDA graphs are reused across calls even though clip_encode rebuilds the
graph each time.

Memory as allocated by mtmd (from its log): weights 1239 MiB (q8_0) or 2333 MiB (bf16), the 2048-row state
52 MiB, compute buffer 125 MiB on the GPU and 16 MiB on the host at the 300-row reserve. nvidia-smi gives
no per-process figure on this Windows driver (N/A), so there is no measured peak.

## Sampling codes 1..9 on the graph

Second session, same day. A one-row call without `next_codes` now decodes: the caller passes c0 (the
backbone token minus 248099) in `code0`, and after residual layer i the graph samples c_{i+1} from head i
and feeds it to layer i+1's embedding lookup, nine dependent steps in one graph. Codes 1..9 come back in
`mtmd_gen_out.codes`, and `out.embd` is the book 1 embedding of the sampled c1. The sampler follows
fish-speech's `sample_top_k_top_p`: probabilities from the whole head at the temperature, at most top_k
kept, every token whose cumulative probability passes top_p dropped, the top one always kept. It takes
one uniform draw per head from the mmproj context's RNG. Temperature 0 or top_k 1 builds an argmax graph
instead. Defaults for drama3 are fish-speech's: temperature 0.7, top_k 30, top_p 0.7. Prefill is unchanged.

Check: `tower --decode`, in `verify/tower.cpp`. Rows 0..27 are prefilled teacher-forced from the
reference hidden states, then rows 28..33 are decoded one at a time with the reference's c0 and, for the
codes of the row itself, the reference's previous frame.

| | CPU bf16 | GPU q8_0 | GPU bf16 |
|---|---|---|---|
| greedy codes equal to the reference frames | 38/54 | 32/54 | 38/54 |
| same rows teacher-forced with the reference codes, argmax equal to the frames | 51/54 | 47/54 | 51/54 |
| same rows teacher-forced with the decoded codes, argmax equal to the decoded codes | 54/54 | 54/54 | 54/54 |

The first line is lower than the second because a miss carries over: within the row, the next layer
embeds the code that was picked, and later rows attend over the cached rows that were built from it. The
first miss in each row is always one of the teacher-forced misses. In bf16 those are row 28 c5, an exact
tie in the reference logits, and row 29 c2 and row 32 c9, where the reference's own teacher-forced logits
prefer the llama.cpp pick by 0.125 and 0.0625 (the reference's incremental pass chose differently; it
disagrees with its own batched pass on 3 of the 54). The q8_0 misses are ties or at most 0.125 apart.
The third line shows the decode graph computes exactly what the teacher-forced graph does.

Sampled, temperature 0.7, top_k 30, top_p 0.7: seeds 1 and 2 agree on 8 of 54 codes (10 at q8_0), every
code is in range, c1 is within the reference head's top 30 in all rows, and seed 1 run twice gives the
same codes. Row 31's c1 over 1000 seeds against the rule applied to the same row's own head 0 logits:
chi-square 5.40 on 5 degrees of freedom (CPU bf16), 2.43 on 6 (GPU q8_0), nothing drawn outside the
kept set.

Per decode call on the 3090, n_kv 256, p50 over the 6 rows after two warmup calls:

| | greedy | sampled | sampled, `GGML_CUDA_DISABLE_GRAPHS=1` |
|---|---|---|---|
| GPU q8_0 | 2.62 ms | 3.38 ms | 6.68 ms |
| GPU bf16 | 3.56 ms | 4.31 ms | |
| CPU bf16 | 20.4 ms | 21.8 ms | |

The sampler costs about 0.75 ms a frame on the GPU, nine small chains of softmax, sort and cumsum. A later
simplification (one argsort of the raw logits instead of top-k plus a re-sort) gives the same codes and the
same q8_0 timing; CPU bf16 sampled is 23.4 ms with it.

## llama-tts end to end

`mtmd-helper-gen.cpp` has a drama3 pipeline next to the Qwen3-TTS one. It builds the prompt the way
`build_drama3_prompt` does for plain TTS (each literal tokenized on its own, `<|speaker:0|> ` added), decodes
it as embedding rows with every row an output, and prefills the tower over rows 0..P-2. Each `step_gen`
draws the main token, runs one decode call of the tower at the next row, and decodes
(token embedding + codebook 1 embedding of c1) / sqrt(2) as the next backbone row. It stops on eos, or when
the tower state is full (fish-speech's finish reason "length"). There is no codec in llama.cpp yet, so the
codes go to `$MTMD_DRAMA3_CODES_OUT` (int32 rows of c0..c9, with a `.json` sidecar holding the frame count
and, for verification, the prompt ids) and the WAV that llama-tts writes is empty.

The main token is drawn in the pipeline, from the backbone's last logits, and the token llama-tts's
common_sampler picked is ignored, as the Pocket TTS pipeline does. The draw is fish-speech's rule over eos
and the 4096 semantic ids only: temperature first, at most top_k, the token that crosses top_p dropped, the
top one kept. So the GGUF's `suppress_tokens` and the `--samplers` order no longer matter for Drama 3.
Repetition-aware sampling follows the same way: a semantic draw that repeats one of the last 16 main
tokens is redrawn at temperature 1.0, top_k 30, top_p 0.95 (fish-speech's RAS_* constants), and that draw
may be eos. It is needed. The first run without it went 445 frames without an eos, most of them one token
repeated. The pipeline's RNG and the tower's are both seeded once from the request's seed, so a fixed
seed repeats a run. llama-tts now passes `--temp`, `--top-k` and `--top-p` to the pipeline only when they
are given, otherwise the pipeline's defaults apply (for Drama 3 fish-speech's 0.7 / 30 / 0.7).

    set CUDA_VISIBLE_DEVICES=GPU-dcb6b5a1-ac97-a0f2-4fb2-15932bf0fe68
    set MTMD_DRAMA3_CODES_OUT=D:\drama3\verify\out\tts\hello.codes
    build\bin\llama-tts.exe -m drama-3-q4k-experts-q8-backbone.gguf -mm drama-3-tower-q8_0.gguf -ngl 99 -c 2048 ^
        -p "Hello there, how are you today?" --temp 0.7 --top-k 30 --top-p 0.7 ^
        --samplers "temperature;top_p;top_k" --seed 42 -lv 4 -o hello.wav

Plain RTX 3090, whole backbone on the card. 45 frames then eos (2.09 s of audio). Two runs give the same
codes.

| | time |
|---|---|
| load, backbone and mmproj | about 10.3 s |
| prompt, 29 rows with the tower prefill | 0.23 s (first call, includes the allocations) |
| generation, 45 frames | 0.62 s, 13.8 ms a frame |
| wall time of the process | 13.3 s |

An earlier build, with the main token from common_sampler, timed the parts of a frame: tower decode
3.71 ms, the backbone row 6.92 ms (decode and the wait for its embeddings), and about 2.7 ms in
common_sampler over the 252288 logits. llama-tts still runs that sampler every frame though the pipeline
now ignores its pick.

`verify/pipeline.cpp` checks the pipeline through the helper API. The prompt ids in the sidecar equal
`verify/prompt_tokens.txt`. The backbone hidden state of the `<|voice|>` row has cosine 0.9957 to the
reference (Q4 experts against BF16). The reference's first main token has rank 2 there, the known
988 / 1385 swap, so a greedy run parts from the reference frames at frame 0 (1/60 codes equal). Seed 42 twice
gives identical codes, seed 43 none of the same.

## Codec, on the CPU

`verify/decode_codes.py` loads `codec.pth` the way fish-speech's `load_codec_model` does (modded_dac_vq.yaml,
strict load, weight norm folded), in float32 on the CPU, and calls `from_indices`. The venv has no
descript-audio-codec or audiotools, so the script defines the few layers fish-speech imports from them.
As a check of those, a 1.44 s Drama 3 clip from the audio.cpp bench encoded and decoded again keeps a
log-spectrum correlation of 0.936 with the original.

    .venv\Scripts\python.exe decode_codes.py D:\models\drama-3 out\tts\hello.codes

| codes | wav | frames | length |
|---|---|---|---|
| llama-tts run above | `verify\out\tts\hello.dac.wav` | 45 | 2.09 s |
| reference greedy frames | `verify\out\ref_frames.wav` | 6 | 0.28 s |

Whisper small.en (faster-whisper on the CPU, `D:\tmp\whisper_verify\tr.py`, command in
`verify\out\tts\transcript.txt`) hears the llama-tts clip as "Hello there. How are you today?". The 36-frame
clip of the earlier build was transcribed word for word as well.

## Not done

The codec in llama.cpp. A reference-audio prompt and history turns in the pipeline. llama-tts still runs
its own sampler each frame for nothing. The deep bench on the 3090 Ti once it is back.
