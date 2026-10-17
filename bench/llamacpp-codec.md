# Drama 3 codec in llama.cpp

2026-10-16. Third llama.cpp piece: the S2-Pro DAC decoder, which turns the ten codes of a frame into 2048
samples at 44.1 kHz, as a streaming mtmd graph in the same mmproj as the tower, checked against
fish-speech's decode in fp32. Same llama.cpp tree and build as `llamacpp-backbone.md` and
`llamacpp-tower.md` (master f3363593), nothing committed.

## What was built

`Drama3TowerModel` in `conversion/drama3.py` gained `generate_extra_tensors`, which reads `codec.pth` and
writes the decode path into the mmproj next to the tower. Weight norm is folded (w = v * (g / ||v||), per
output channel, per input channel for ConvTranspose1d), wqkv is split into q, k and v, and each Snake is
stored as alpha and 1 / (alpha + 1e-9) so the graph can use Qwen3-TTS's Snake op sequence. The encoder side
and the freqs_cis and causal_mask buffers are skipped. Tensor names reuse Qwen3-TTS's `a.gen.wav.*` kinds
(`quant.first` and `quant.rest` with an out_proj per codebook, `tfm.blk.*`, `up.blk.*`, `dac.*`). There are
no new GGUF keys; the codec hparams are set in `clip.cpp`'s drama3 case. Conv and transposed-conv kernels and
all matrices are F16 whatever `--outtype` says, the quantizer and the 1-D tensors F32.

The codec adds 235 tensors, 369 MB, 184.4 M parameters. Files: `drama-3-mmproj-bf16.gguf` (2.62 GiB) and
`drama-3-mmproj-q8_0.gguf` (1.55 GiB) in `D:\models\drama-3-own\llamacpp\`, about 20 s each to convert. Their
tower tensors are byte-identical to the tower-only files. Those still load; without a codec GEN_WAV returns
an error, and llama-tts warns and writes only the codes file.

The codec as fish-speech builds it, measured on `codec.pth`: ten codebook lookups (semantic 4096 x 8, nine
residual 1024 x 8), each through its own 1x1 out_proj to 1024, summed. Then an 8-layer causal transformer at
the code frame rate: dim 1024, 16 heads of 64, SwiGLU 3072, RMSNorm 1e-5, LayerScale, RoPE on adjacent pairs
with base 10000, a 128-frame window. Then 2 upsample blocks, each a causal ConvTranspose k2 s2 and a ConvNeXt
block (dwconv k7, LayerNorm, 1024-4096-1024 with exact GELU, gamma). Then the DAC decoder: a causal conv k7
from 1024 to 1536, 4 blocks (Snake, ConvTranspose with k = 2s and strides 8, 8, 4, 2, three residual units
with dilations 1, 3, 9) through 1536, 768, 384, 192 and 96 channels, a last Snake, a conv k7 to one channel
and tanh. All of it is causal, exactly 2048 samples a frame, with no delay.

The graph is `build_wav()` in `tools/mtmd/models/drama3-gen.cpp`, a GEN_WAV branch of the `drama3_gen`
projector. It builds Qwen3-TTS's code2wav helper object and uses its causal conv (im2col), transposed conv
(`mul_mat` plus `ggml_col2im_1d`), Snake, ConvNeXt, residual unit and windowed transformer layer. The
drama3 parts are the per-codebook quantizer, RoPE NORMAL, erf GELU, the tanh, and F32 accumulation forced on
every matmul: without it CUDA's cuBLAS path accumulates in F16 and the GPU decode was 44.5 dB. One call takes
any number of frames. About 1206 nodes.

The state goes through the host as F32 slots, listed by `list_gen_state_slots`: `tfm_pos`, K and V of the
last 127 positions per transformer layer (`tfm_k`, `tfm_v`, [1024 x 127] each), the left context of each
causal conv and the overlap tail of each transposed conv. That is 8,889,092 bytes, 8.3 MB of it K/V. The
K/V has to be carried. The transformer's stacked receptive field is 8 x 127 frames, so a stateless window
does not reproduce the offline decode (25 dB); carrying K/V per layer does.

`drama3_gen_audio_pipeline` in `mtmd-helper-gen.cpp` decodes the first window after 4 frames and then every
8, flushes the rest in `get_output` and returns a real WAV. `MTMD_DRAMA3_CODES_OUT` still writes the codes.
Around it: mtmd.cpp no longer rejects GEN_WAV for drama3, the Qwen-only audio trim is guarded by projector
type, and clip.cpp's precheck, input and output paths know about GEN_WAV. `qwen3tts-gen.cpp` changed in three
small ways: RoPE type and GELU keyed on the projector, the state-slot list keyed on `dac.entry` with
`pre_conv` optional, and F32 accumulation on the transposed conv's `mul_mat`.

## Rerun

    cd D:\drama3\third_party\llama.cpp
    D:\drama3\.venv-convert\Scripts\python.exe convert_hf_to_gguf.py D:\models\drama-3 --mmproj --outtype q8_0 ^
        --outfile D:\models\drama-3-own\llamacpp\drama-3-mmproj-q8_0.gguf

    cd D:\drama3\verify
    build\codec.exe <mmproj> D:\models\drama-3-own\llamacpp\drama-3-q4k-experts-q8-backbone.gguf out\codec_cpu_bf16 [--gpu] [--bench]
    .venv\Scripts\python.exe compare_codec.py out\codec_cpu_bf16

`codec.cpp` (target `codec` in the verify CMakeLists) loads the backbone on the CPU only for n_embd, as
`tower.exe` does. It decodes `out\tts\hello.codes` (45 frames) and rows 29..34 of `ref_tower_frames.npy`
(6 frames, "ref6") twice: offline in one call, and streamed as 4 frames, then 8 at a time, then the
remainder. The streamed hello is decoded a second time to check that it repeats. `compare_codec.py`
compares both with fish-speech's codec in fp32 on the CPU (the one in `decode_codes.py`), whose output
`D:\tmp\drama3\inspect_codec.py` saved as `out\codec\hello_ref.npy` and `ref6_ref.npy` along with the
intermediates. `--bench` is the timing mode below.

## Against fish-speech

SNR against the fp32 decode, and the largest absolute sample difference:

| | hello offline | ref6 offline | hello, streamed vs offline | ref6, streamed vs offline |
|---|---|---|---|---|
| CPU bf16 | 60.08 dB, 5.2e-4 | 67.79 dB, 6.1e-4 | 87.9 dB, 5.7e-5 | identical |
| GPU bf16 and q8_0 | 60.03 dB, 8.5e-4 | 66.29 dB, 1.24e-3 | 61.7 dB, 4.7e-4 | 69.5 dB, 5.1e-4 |

The bf16 and q8_0 files give byte-identical audio on the GPU, since the codec is F16 in both. For scale, in
PyTorch against the same fp32 decode: F16 weights with F16-rounded matmul inputs give 57.2 dB, the whole
model in F16 50 dB, and BF16, fish-speech's own default on the GPU, 33 dB.

On the CPU the streamed hello equals the offline one except in the last frame, which is near silent; for
ref6 they are bit-identical. On the GPU they differ by up to 4.7e-4 because CUDA picks different matmul
kernels for 4, 8 and 45 rows. A rerun of the streamed decode is bit-identical on both.

The transposed-conv kernels are F16 in the file, where Qwen3-TTS stores F32. On CUDA an F32 kernel with 16
columns or fewer goes to the mmf kernel, whose TF32 path cost the upsample 12 dB at 4 to 8 frames: GPU
streamed was 51 dB with F32 kernels and 62 dB with F16. On the CPU the change costs 60.8 to 60.1 dB on hello.

## Timings

`codec --bench`: 3 warmup calls, then 20 calls of the same 8 or 4 frames of hello with the state carried
from call to call. Plain RTX 3090 at 0000:41:00.0 (the Ti is still off the bus), q8_0 mmproj.

| | CUDA graphs | 8 frames, p50 | p99 | 4 frames, p50 | p99 |
|---|---|---|---|---|---|
| GPU q8_0 | on | 13.40 ms | 14.05 ms | 10.33 ms | 11.21 ms |
| GPU q8_0 | off | 16.79 ms | 18.04 ms | 13.14 ms | 13.91 ms |

An 8-frame call holds 372 ms of audio and costs 1.68 ms a frame at p50; a 4-frame call holds 186 ms and
costs 2.58 ms a frame. "Off" is `GGML_CUDA_DISABLE_GRAPHS=1`.

Per call in the verification runs:

| | 4 frames | 8 frames | 45 frames, one call |
|---|---|---|---|
| GPU q8_0 | 10.3 ms (bench p50) | 13.4 ms (bench p50) | 162 to 166 ms |
| CPU bf16 | 100 to 104 ms | 164 to 170 ms | 1068 ms |

The first call of a new shape takes about 33 to 36 ms on the GPU. The 45-frame call is 3.6 ms a frame,
more than an 8-frame call; the compute buffer grows with n (about 125 MiB at 8 frames, 670 MiB at 45, and
the scheduler keeps the largest it has seen). The CPU takes 20.9 ms a frame in 8-frame calls. audio.cpp's
codec takes 4.2 to 5 ms a frame warm on the 3090 Ti, one frame per call.

Memory as mtmd logs it: the clip model is 1591.66 MiB with the q8_0 file and 2684.78 MiB with the bf16 one,
352 MiB of either the codec. The tower state is 52 MiB at 2047 rows. The codec's compute buffer is not logged.

## llama-tts end to end

    set CUDA_VISIBLE_DEVICES=GPU-dcb6b5a1-ac97-a0f2-4fb2-15932bf0fe68
    set MTMD_DRAMA3_CODES_OUT=D:\drama3\verify\out\tts\hello_codec.codes
    build\bin\llama-tts.exe -m drama-3-q4k-experts-q8-backbone.gguf -mm drama-3-mmproj-q8_0.gguf -ngl 99 -c 2048 ^
        -p "Hello there, how are you today?" --seed 42 -lv 4 -o hello_codec.wav

Plain RTX 3090, Q4 backbone. The same 45 frames as in `llamacpp-tower.md`: `hello_codec.codes` is
byte-identical to `hello.codes`. The mtmd INFO lines, the codec total among them, show at `-lv 4`, the time
of each window at `-lv 5`.

| window | frames | time |
|---|---|---|
| first | 4 | 53.2 ms |
| 2 to 6 | 8 each | 23.7, 17.0, 16.5, 17.5, 16.6 ms |
| flush in `get_output` | 1 | 17.1 ms |
| codec in total | 45 | 163.5 ms |

Generation takes 0.69 s for 2.09 s of audio: 15.3 ms a frame with the codec, 12.1 ms without, 2.47x real
time overall. The WAV is 44100 Hz mono PCM16, 92160 samples, 59.30 dB against the PyTorch decode (streamed
on the GPU, plus the int16 rounding). Whisper small.en on the CPU hears "Hello there. How are you today?"
(command and entry in `verify\out\tts\transcript.txt`).

## The tower on the combined file

CPU, `drama-3-mmproj-bf16.gguf`: top 1 51/54 in one call and 52/54 row by row, rewind bit-identical, layer
SNR 39 to 48 dB. The same as with the tower-only file.

## Open

The state blob (8.9 MB) crosses PCIe twice a call; keeping the K/V on the device, as the tower does, would
save that. Each call transposes the transposed-conv kernels again (about 50 MB of F16, mostly DAC block 0),
which is negligible on the GPU and about 10 ms on the CPU; storing them pre-transposed would remove it.
Streamed and offline differ by 5e-4 on the GPU rather than being bit-identical, from the kernel choice by
batch size. The codec compute buffer is sized by the largest n seen. In the pipeline the first 4-frame
window costs 53 ms (graph build and allocation for a new shape) instead of the 10 ms of a warm call; a
warmup call when the pipeline starts would hide it. Whether a 4-frame first window sounds clean is not yet
judged by ear, though streamed equals offline numerically, so there is no seam to hear. The timings on the
3090 Ti once it is back.
