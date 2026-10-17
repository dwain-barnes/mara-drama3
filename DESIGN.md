# Drama 3 in llama.cpp for Crosstalk

## What this is

A llama.cpp branch that runs Fish Audio Drama 3 entirely in ggml, plus a thin agent loop that uses it as the voice of Crosstalk, the tick-based duplex agent in `D:\Voice_agent_duplex`. The Drama 3 backbone, tower and codec and a small dialogue LLM all share one 24 GB RTX 3090 Ti. There is no PyTorch at run time.

The branch adds Drama 3 to the TTS path llama.cpp already has for Qwen3-TTS and Pocket TTS (PRs #26254 and #26871). The agent loop is Crosstalk's `loop.py`. The LLM picks an affect per clause, and the TTS renders it from recordings of one voice in different moods, each clause continuing from the agent's last few seconds of speech.

## The model

Drama 3 (`fishaudio/drama-3`, 2026-10-13) is a 35B MoE with 3B active. The backbone is Qwen3.5-35B-A3B with a larger vocabulary (252288), no MTP layer and no vision tower: 40 layers (30 linear attention, 10 full attention), 256 experts of width 512 with 8 active, hidden size 2048. It predicts codebook 0 as one of 4096 `<|semantic:N|>` tokens. A 1.22B-parameter tower of 13 layers then predicts codebooks 1 to 9 in order, and each tower layer attends over all earlier frames. The codec is S2-Pro's DAC: 44.1 kHz, 2048 samples per frame, 21.5 frames/s. Routed experts are FP8 in 128x128 blocks, the rest BF16. Fish says 48 GB minimum.

The prompt is a chat template: a system turn with an instruction and each reference as `[Reference i] Text: ... Speech: <codes>`, earlier chunks as completed user-text and assistant-audio turns, the target text as the last user turn, and an assistant turn opening with `<|voice|>`. `<|speaker:N|>` (0 to 15) is a real token. Everything else is plain text the model learned: point tags like `[laughs]`, bracketed directions, paired tags like `<whisper>`. A tag at the very start of a line is reported to break. Back-channel tokens exist in the vocabulary but no prompt code uses them.

The weights, and any GGUF made from them, are under the Fish Audio Community License 1.0 with "Built with Fish Audio" attribution. This repo ships code, not weights.

## Architecture

```
 mic 16k --> ASR: Nemotron 0.6B, NeMo-Speech.cpp, CPU   (:9100)
                  | partial words every 160 ms
                  v
          agent loop (loop.py)
            tick every 0.3-0.6 s  ------------>  /completion   small LLM
            <listen> | <label> words  <--------
            clause buffer, label -> bank clip + direction
            clause  -------------------------->  /tts          Drama 3
            PCM 44.1k, streamed  <-------------
                  | 44.1k -> 24k, cut() on barge-in
                  v
          agent track (wall-clock mixer)

 llama-server from this branch, one process, 3090 Ti
   small LLM    Q4_K_M, its own context
   Drama 3      backbone GGUF: qwen35moe, embedding input, first N expert layers on CPU
                mmproj GGUF:   codebook tables, tower graph, DAC codec graph
   one loop     renders audio ahead of playback, fits ticks into the slack
```

### Files

The converter is one module, `conversion/drama3.py`, next to `qwen3tts.py`, and writes two GGUFs as Qwen3-TTS does.

The backbone uses the existing `qwen35moe` arch, whose converter already matches the tensor names. The subclass registers `FishOmniModel`, skips `audio_decoder*`, handles the missing `mtp_num_hidden_layers`, sets EOS to 248046, and aliases the `bsgemm-moe` quant method to the existing `fp8` path, which multiplies each block by `weight_scale_inv`. It writes `suppress_tokens` for every id except EOS and the 4096 semantic ids, so libllama's sampler can only draw a frame or the end. Experts are quantised to Q4_K and everything else to Q8_0 by tensor type, not as a Q4_K_M mix.

Only 4097 rows of the output head are ever sampled, so the head is trimmed to them and the logits padded back to vocab width, as Qwen3-TTS does for its `codec_head`. That needs a thin arch alias over `qwen35moe`, the way `qwen3tts` sits on `qwen3vl`. Early conversions keep the full head so stock llama.cpp loads them.

The mmproj carries the 10 codebook tables, a copy of the 4096 semantic rows of the token embedding, the tower at Q8_0, and the codec from `codec.pth` at F16 with weight norm folded.

### Graphs

The backbone runs in libllama with rows passed through `llama_batch.embd`, as the Qwen3-TTS helper does. A text row is `embed_tokens[id]` from a host copy of the table. An audio row is `(embed_tokens[c0 + 248099] + codebook_embeddings[4096 + c1]) / sqrt(2)`. libllama samples c0, and the helper redraws once if it repeats one of the last 16, as fish-speech does. The post-norm hidden state from `llama_get_embeddings_ith` goes to the tower.

The tower graph follows `tools/mtmd/models/qwen3tts-gen.cpp`. It takes the hidden state and c0, runs 4 shared and 9 residual layers, and samples c1 to c9 on the graph with top-k and top-p. That is one graph and one host call per frame, and it returns the ten codes and the next backbone row.

The codec uses the parts Qwen3-TTS's code2wav and Pocket TTS's decoder already use: `ggml_conv_1d` through im2col, transposed conv as `mul_mat` plus `ggml_col2im_1d`, and Snake as the five-op sequence the CUDA backend fuses into one kernel. Streaming uses the same explicit state: each causal conv keeps its left context, each transposed conv its overlap tail, each windowed attention block its K/V and RoPE position, passed in and out of every call. It decodes a first window after 4 frames, then 8 at a time.

### The tower's cache

This is the one piece llama.cpp does not have, and the main engineering risk. Qwen3-TTS's code predictor works inside one frame, with a zeroed 16-row cache thrown away after each call. Drama 3's tower attends over time: each of its 13 layers keeps K and V for every earlier frame, prompt rows included.

The cache is a device buffer owned by the mmproj context, allocated once: 13 layers, K and V of 512 values per row, F16, 2048 rows, about 55 MB. Each frame writes one row per layer with `ggml_set_rows`, in place, as libllama's KV cache does. Attention reads the first n rows, n padded to 256 and the tail masked, so the graph keeps its shape and the CUDA graph is reused. A counter feeds RoPE, which in the tower is interleaved, unlike the backbone's.

Prefill runs all 13 layers over every prompt row in one teacher-forced graph. That needs the backbone's hidden state for every prompt row, so backbone prefill asks for outputs on all rows, which the trimmed head makes cheap. Text rows get a zeroed code embedding, and shared layer 0 adds the summed embeddings of c2 to c9, which only prompt rows have.

Rewinding is a counter reset to the restored prefix length. A snapshot is the backbone state from `llama_state_seq_get_data` (about 70 MB) plus the tower rows for the same prefix (about 4 MB).

audio.cpp (`0xShug0/audio.cpp`, Apache-2.0) already runs all of Drama 3 in ggml on CUDA, in about 5.3k lines on a forked ggml 0.12 with its own `ggml_snake_1d`. Its tower keeps the same kind of timeline cache but samples on the host, 9 launches per frame. It proves the model runs in ggml. The branch reads it and credits it where its ideas are used, but does not paste it. s2.cpp is not used, because it is under the Fish Audio Research License.

### One card, two models

llama-server runs as one process with two contexts: the small LLM as its model, and Drama 3 loaded alongside, the way a draft model is, behind `POST /tts` from the open PR #26603, extended to stream PCM. One process, because each Windows process pays its own CUDA context, a few hundred MiB on a card with a GiB to spare, and because WDDM time-slices two processes with no priority, so a tick could land in the middle of a frame. In one process the server's task loop orders the work. The cost is that one crash takes both down.

The loop renders ahead of playback into a buffer of 300 to 500 ms. While the buffer holds more than 200 ms a tick runs when due. Below that, frames go first and the tick waits a few frames. Ticks are delayed, never dropped. On barge-in, `spk.cut()` drops the buffer and sends the last sample played, and the server trims its ring buffer to it, so the next turn continues from what the user heard.

Two processes from the same build are the fallback, and the first end-to-end test uses them because they need no server work.

## Expression

### Affect labels from the dialogue LLM, rendered by a reference bank

The LLM gets about ten labels: neutral, warm, amused, laughing, excited, sympathetic, serious, hesitant, apologetic, emphatic. On a speaking tick it emits one before its words, e.g. `<warm> oh that's lovely,`. Each label is one token, an unused reserved token or a rare single-token word, so it costs one decode step. Listen ticks are unchanged.

The label is trained into the small dialogue LLM through the axolotl SFT on the phase2 tick data. The tick SFT has to be redone on the small model anyway, since run2 is a 35B fine-tune. If the source dialogues have audio, labels come from an arousal and valence model and a laughter detector run over each agent clause. Otherwise an offline LLM pass labels clauses from text with two turns of context.

Each label maps to a render entry:

```
warm:      clip=bank/warm_2.wav    direction=none
amused:    clip=bank/amused_1.wav  direction="[amused]"
laughing:  clip=bank/amused_1.wav  point="[laughs]" after the first word
hesitant:  clip=bank/neutral.wav   direction="[unsure, slowing down]"
```

The bank is 8 to 12 clips of the agent's one voice, 5 to 10 s each, one affect per clip, made by cloning a neutral recording, rendering lines per affect with directions, and keeping clips a speaker-verification model scores as the same voice. The bank carries most of the affect, since a reference with strong prosody tends to beat a direction.

The Speaker keeps the label of the clause's first tick, and switches clip only if the new label held for the whole clause, or is laughing or apologetic, so the voice does not flicker at every comma. If the model spoke while `p_listen` was between 0.3 and 0.5, the turn's first clause gets the hesitant entry. The request carries the clip and `[direction] clause`, and PCM streams back. The fallback is the neutral clip with no direction. If labels hurt turn-taking, a small CPU classifier produces them from the last two turns instead.

### Rolling history

Every prompt carries the last 3 to 5 s the agent actually played as one history turn, its words as user text and its codes as assistant audio. fish-speech uses the same mechanism between its own chunks, so the model has seen it. The Drama 3 pipeline in the branch writes it from codes it sampled, cut at the last frame the user heard, keeping all ten codebooks because the tower's prefill reads them. It is dropped on a turn's first clause if the agent last spoke more than 2 s ago.

## Why not the alternatives

**Tags in the system prompt.** Crosstalk's LLM is trained, not prompted, and free-text directions cost 5 to 10 tokens a tick, GPU time the TTS needs.

**A side classifier from dialogue text.** It cannot see what the LLM is about to do, and needs the same labels. It stays as the fallback.

**User-tone mirroring from mic audio.** A hand-written mirroring policy is often wrong: mirroring anger escalates. An arousal value could later go into the tick input, not drive the voice.

**Back-channels rendered live.** A "mm-hm" is better pre-rendered from the bank and played from memory.

**Speaker tags for two registers.** A trick. The bank covers register changes.

**Hedging from token entropy.** Entropy mostly reflects tokenisation, not doubt. The `p_listen` rule uses the one signal with a clear meaning.

**The tower and codec in PyTorch.** Eager, the tower takes 30 to 45 ms a frame under WDDM, `torch.compile` needs Triton, which Windows lacks, and PyTorch's own CUDA context costs memory this card does not have.

**audio.cpp as the engine.** The LLM would sit in a second engine on the same card, with nothing to schedule the two together.

## Hardware

Drama 3 gets one card, the 3090 Ti at 2B:00. The other three cards are in use on another project. The ASR moves to the Threadripper 5995WX, and its 160 ms step has to be measured again there, on cores kept apart from llama.cpp's threads.

Memory on the Ti in GiB. The token embedding table stays on the CPU (llama.cpp's default) and the head is trimmed to 4097 rows, about 1 GiB saved between them. The LLM is a Qwen at Q4_K_M with 8k context and q8_0 KV. N is the expert layers kept on the CPU with `--n-cpu-moe N`, 432 MiB each at Q4_K.

| On the Ti | 1.7B LLM, N=4 | 4B LLM, N=7 |
|---|---|---|
| Backbone non-expert weights, Q8_0 | 1.42 | 1.42 |
| Routed experts, Q4_K, 40-N layers | 15.19 | 13.92 |
| Tower, Q8_0 (2.28 at F16) | 1.21 | 1.21 |
| Codec, F16 | 0.88 | 0.88 |
| Drama 3 KV and state, tower cache, compute buffers | 1.5 | 1.5 |
| Dialogue LLM weights, KV, compute buffer | 1.9 | 3.25 |
| CUDA context | 0.4 | 0.4 |
| Total | 22.5 | 22.6 |

The card has 24.0 GiB and no display. With every expert layer on the GPU the backbone alone is 19.3 GiB at Q4_K and 18.4 at IQ4_XS, too much once the rest is added. There is no step up: Q5_K experts alone are 20.6 GiB.

Per frame, each CPU layer reads 8 experts x 3 matrices x 2048 x 512 at 0.5625 bytes, 14.2 MB. At 150 to 200 GB/s of DDR4 that is 0.3 to 0.4 ms for N=4 and 0.5 to 0.7 ms for N=7. llama.cpp's overhead is larger: each CPU layer is a graph split with copies down and back, a sync and a thread-pool wake-up, about 0.3 to 0.8 ms under WDDM. So 1.5 to 3.5 ms per frame at N=4, 3 to 6 ms at N=7. `--moe-cache-mib` keeps a GPU cache of CPU-held experts, but its behaviour is unverified, so it is not in the plan.

Measured. Drama 3 alone in audio.cpp peaked at 21.8 GiB with 4 CPU expert layers, 20.9 at 6 and 18.3 at 12, so each layer moved frees about 430 MiB, as estimated. The table's 22.5 GiB at N=4 was too low: the LLM takes 1.9 to 3.25 GiB on top of 21.8. Next to a small LLM, 8 to 10 CPU layers are needed, which is at or past the stop line below. Measured again in the llama.cpp port on a plain 3090 (2026-10-17, `demo/say.cpp --n-cpu-moe`, Q4_K experts, Q8_0 tower and codec, 4096-row context): 21.8 GiB with every layer on the card, 432 MiB back per layer moved, 16.6 GiB at N=12, where Gemma 4 E4B (UD-Q4_K_XL, 8k context, 3.7 GiB) fits beside it with 4.3 GiB free; N=8 also fits with 2.5 GiB free. Per frame 11.9 ms at N=0, 15.5 at N=8, 17.4 at N=12, 19.4 at N=16. The larger cost is the prompt phase: with two reference clips (about 420 rows) the first audio moves from 0.43 s at N=0 to 1.2 s at N=8 and 1.6 s at N=12, and 0.4 s more while the LLM is answering on the same card; the LLM drops from 118 to 91 tokens/s during a render.

Pin by UUID with `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Pocket TTS stays as the fallback on the CPU.

## Latency

Per frame on the Ti, against 46.4 ms of audio:

| Step | Estimate |
|---|---|
| Backbone row, GPU expert layers, Q4_K | 10-15 ms |
| Expert layers on the CPU, N=4 to 7 | 2-6 ms |
| Sample c0, hidden down, next row up | about 1 ms |
| Tower graph, one launch | 4-8 ms |
| Codec, amortised | 2-5 ms |
| Total | 19-35 ms, RTF 0.41-0.75 |
| Slack | 11-27 ms |

Measured in audio.cpp, not the llama.cpp port: backbone 13 to 15 ms, tower 5 to 6, codec 4 to 6, total 22 to 24 ms a frame. First audio on a warm server was 136 ms at 26 prompt rows and 208 ms at 50 rows, with no saved prefix. Warm prefill was about 4 ms a row. A cold process took 590 to 920 ms, and the first request after start 2.5 s.

A tick on the small model is prompt processing of 20 to 40 new tokens, 8 to 20 ms, then 4 to 8 ms per token from 1.7B to 4B. A listen tick costs 20 to 45 ms, a speaking tick of a label and 16 tokens 75 to 155 ms. At 0.3 s ticks the slack adds up to 70 to 180 ms per tick. A listen tick always fits. A 150 ms speaking tick fits only with frames under about 23 ms, and at 35 ms frames back-to-back speaking ticks empty the buffer in about 1.5 s. The planned one word per 0.3 s tick, needed anyway for speech lag, makes speaking ticks as cheap as listen ticks. Failing that, ticks go to 0.6 s while the agent speaks.

First audio. A 6 s reference, 4 s of history and a clause make about 290 rows. The reference prefix is saved once per bank clip and restored per clause, leaving about 130 rows.

| Step | Estimate |
|---|---|
| Restore backbone state (70 MB) and tower rows | 5-10 ms |
| Backbone prefill, 130 rows, GPU layers | 45-90 ms |
| Same prefill, N CPU expert layers | 20-60 ms |
| Tower prefill, one graph | 10-20 ms |
| First 4 frames and codec window | 80-150 ms |
| Total | 160-330 ms |

130 rows touch nearly every expert, so each CPU layer reads all 432 MiB. llama.cpp may instead copy those weights to the GPU over PCIe, about 20 ms a layer, so both are measured. Without the restore, first audio is 230 to 430 ms.

End of user speech to first agent audio, 0.3 s ticks:

| Step | Today (35B LLM, Pocket TTS) | This design |
|---|---|---|
| ASR emits last words | 160-320 ms | same, if the CPU holds 160 ms |
| Wait for next tick | 0-300 ms | same |
| LLM tick | 130-220 ms | 40-150 ms |
| TTS first audio | 42-80 ms | 160-330 ms |
| Resample and mixer | 20-40 ms | same |
| Total, typical | 0.5-0.9 s | 0.55-1.0 s |

A turn starts 50 to 200 ms later than today, since the faster tick pays back part of the slower first audio. The history should shrink the 80 to 440 ms join silences.

## What is uncertain until tested

1. The converter. Answered 2026-10-16: `conversion/drama3.py` converts the backbone with the stock `qwen35moe` arch. The `bsgemm-moe` experts dequantize bit-exactly against fish-speech's `dequantize_fp8_blocks`, the tokenizer hashes to `qwen35` and gives the same 29 ids as fish-speech for the TTS prompt, the MTP key is skipped with `no_mtp`, EOS is 248046 and `suppress_tokens` has 248191 entries. BF16 GGUF logits match the PyTorch BF16 backbone on the CPU (correlation 0.9999 over the vocab, KL 0.002); Q4_K experts with Q8_0 elsewhere give correlation 0.996 and KL 0.022 over EOS plus the semantic ids, same top three. Numbers in `bench/llamacpp-backbone.md`. The tokenizer on reference prompts: answered 2026-10-16, a one-reference prompt (245 rows, 167 of them audio) and a reference-plus-history prompt (286 rows) give exactly fish-speech's ids row for row, with every literal tokenized on its own as `build_drama3_prompt` does (`bench/llamacpp-references.md`). Still open: the trimmed head.
2. The tower and its persistent cache. Answered 2026-10-16: `conversion/drama3.py` also writes the tower as a gen-only mmproj (projector `drama3_gen`, 13 blocks, norm weights stored with the +1 added), and `tools/mtmd/models/drama3-gen.cpp` runs all 13 layers over n timeline rows in one graph, writing K and V into a per-layer F16 cache with `set_rows` and attending over the first `pad(n_past + n, 256)` rows. The cache is an opaque `mtmd_gen_state` with `set_n_past` for rewind. Teacher-forced against fish-speech BF16 on the CPU over the 29-row prompt plus 6 greedy frames: per-layer SNR 39 to 48 dB with the bf16 mmproj, head logits 48 to 51 dB, argmax 51 to 52 of 54 codes, where the reference's own batched and incremental passes already disagree on 3 of 54 and the rest are exact ties. Q8_0 gives 31 to 42 dB per layer and 45 to 51 of 54, every miss the reference's second choice. Batched prefill, one row at a time and a rewind all give the same rows bit for bit. Timed at depth on a plain 3090 (the Ti dropped off the bus during the first attempt and is lost until a reboot): a one-row call at rows 300 to 500 takes 2.8 ms p50 and 5.2 ms p99 at Q8_0, 3.8 ms p50 at bf16, so CUDA graphs are reused even though mtmd rebuilds the graph every call (4.9 ms p50 with them off); a 300-row prefill takes 22 ms warm, a 130-row prefill 12 ms. audio.cpp's tower is 4.7 ms a frame on the Ti. The tower holds 1.24 GiB of weights at Q8_0, 52 MiB of cache at 2048 rows and 125 MiB of compute buffer. Numbers in `bench/llamacpp-tower.md`. c1 to c9 are sampled on the graph (fish-speech's top-k, top-p and temperature rule, one draw per head, 0.75 ms extra a frame; greedy mode reproduces the reference's codes up to its own ties), and a Drama 3 pipeline in `mtmd-helper-gen.cpp` runs the whole chain from `llama-tts`: prompt rows, tower prefill, the main token drawn in the pipeline with the RAS redraw, the mixed next row. With the Q4 backbone and the Q8_0 tower on a plain 3090, "Hello there, how are you today?" gives 45 frames at 13.8 ms a frame, and fish-speech's DAC on the CPU (`verify/decode_codes.py`) turns the codes into audio that Whisper transcribes word for word. Still open: the codec inside llama.cpp, reference prompts and history turns in the pipeline, and the Ti figures once the card is back.
3. Whether backbone prefill returns hidden states for all rows cheaply, and whether the padded head gives the same c0 distribution.
4. Whether the linear-attention state behaves with embedding input and after a restore. Embedding input answered 2026-10-16: the pipeline feeds a 167-frame reference clip as mixed rows (token embedding plus codebook 1 embedding over sqrt 2) through the stock `qwen35moe` graph, and every row's post-norm hidden state matches fish-speech BF16: median cosine 0.998 with the Q4 backbone and 0.9993 with the BF16 GGUF, the audio rows at 0.983 or better, and the first rows after the audio block, the first through the linear-attention layers after embedding input, at 0.9999 with BF16. One text row before the audio comes out different in every implementation (an expert routing near a tie), the rest of the spread is Q4 noise. The tower prefilled with the real codes gives head logits within the margins of the text-only check, and the words come through with the reference and with a history turn (`bench/llamacpp-references.md`). Still open: the state after a restore.
5. Streamed codec windows against an offline decode. Answered 2026-10-16: `conversion/drama3.py` writes the S2-Pro DAC decode path from `codec.pth` into the same mmproj as the tower (`drama-3-mmproj-bf16.gguf` and `-q8_0.gguf`, 184 M parameters, F16 in both, weight norm folded, Qwen3-TTS's `a.gen.wav.*` names), and `build_wav()` in `tools/mtmd/models/drama3-gen.cpp` decodes any number of frames per call with Qwen3-TTS's code2wav parts, carrying the transformer's K/V, the conv left contexts and the transposed-conv tails through the host (8.9 MB). Against fish-speech's codec in fp32, an offline decode gives 60.1 dB on the 45-frame clip and 67.8 dB on 6 reference frames on the CPU, 60.0 and 66.3 dB on the GPU; F16 in PyTorch gives 50 to 57 dB and fish-speech's BF16 33 dB. Streamed as a 4-frame first window and then 8 at a time, the output equals one offline call: bit for bit on the CPU except the last, near-silent frame (5.7e-5), within 5e-4 on the GPU, where CUDA picks other kernels by batch size. The per-layer K/V is what makes this hold; a stateless window gives 25 dB, since the transformer's stacked receptive field is 8 x 127 frames. On a plain 3090 an 8-frame call takes 13.4 ms p50 (1.7 ms a frame) and a 4-frame call 10.3 ms; audio.cpp's codec takes 4.2 to 5 ms a frame on the Ti. llama-tts now writes the audio itself: "Hello there, how are you today?" gives the same 45 frames as before with 163 ms of codec in all, 15.3 ms a frame overall, 59.3 dB against the PyTorch decode, and Whisper transcribes it word for word. Numbers in `bench/llamacpp-codec.md`. Still open: whether the 4-frame first window sounds clean by ear (numerically there is no seam), and the Ti figures.
6. Whether Q4 experts keep the expression. Mostly answered 2026-10-16 in llama.cpp with a cloned voice (`bench/llamacpp-references.md`, tag A/B): the same sentence under `<whisper>` comes out 13 to 16 dB quieter with half its energy above 4 kHz (breath, not voicing) at Q4 and at BF16 alike, `[excited]` and `[angry, shouting]` louder with a higher and wider pitch, `[serious, measured]` and `[sad, quiet]` with a narrower one, and every word survives. The audio.cpp bench's "no loudness change" for `<whisper>` does not hold here. A bank of ten same-voice clips rendered from one neutral recording with directions spans 12 dB of level between whisper and excited, and the typed chat demo switches clip by a label the LLM puts before each sentence. Dwain's ear says it sounds right. Still to do: compare blind against the hosted model, and IQ4_XS.
7. What the CPU expert layers cost under Windows, at decode and prefill, by thread count and pinning.
8. Whether the ASR holds its 160 ms step on the CPU.
9. How often LLM ticks empty the buffer, over replayed sessions at 0.3 and 0.6 s ticks.
10. Whether a 1.7B to 4B model learns the tick protocol as well as run2, with the label token. Compare listen and interrupt rates in `phase0/replay.py`.
11. Whether history gives continuity without repeated words, whether a bank clip carries affect against a direction, and whether a direction at the start of a short clause breaks it.
12. Whether the phase2 data has source audio, and whether the back-channel tokens do anything.
13. An experiment, not the plan: the backbone is Qwen3.5-35B-A3B and may still hold a conversation. If it can, one model with two sequences would replace the small LLM and give its 2 to 3 GiB back to expert layers.
14. End of speech after a short first clause: "Right, I had a look at the logs..." came out as "Right." and stopped after 24 frames. Clause chunking would hit this most. The fix to try is forbidding EOS in our own sampler until a frame budget proportional to the text length is met.

## First thing to build

The audio.cpp baseline is done, on our own conversion of the Q4-expert, Q8_0-backbone model. A frame takes 22 to 24 ms, warm first audio 136 to 208 ms, and peak VRAM is 21.8 GiB at 4 CPU layers. Numbers and files are in `bench/RESULTS.md`. The blind listen against the hosted model is still owed.

Then the converter and the tower graph with its persistent cache, checked on a short prompt against fish-speech in BF16 on the CPU. Then the codec, checked against fish-speech's decode of the same codes. Then `llama-tts` renders one clause on the Ti from a restored snapshot plus history, with the small LLM loaded alongside in llama-server ticking from a replayed session. Log per-frame ms per stage, TTFA, peak VRAM and buffer underruns.

Stop if a frame is over 35 ms at p50 or 46 ms at p99, if first audio is over 300 ms at p50 with the reference restored, if the LLM at 8k needs more than 8 expert layers on the CPU to fit or peak VRAM passes 23.2 GiB, or if Q4 experts lose to the hosted model blind, since no larger quant fits.

If it passes, run the offline render test: an 8-clip bank, 20 agent turns from Crosstalk logs chunked as `loop.py` does, labelled by hand, rendered three ways (neutral clip, a direction per clause, bank plus history). Measure join silences and listen blind in pairs. If bank plus history wins, build the one-process server, add `--tts drama` to `loop.py`, and start the small LLM's tick SFT with the label token.
