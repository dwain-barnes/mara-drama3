# Drama 3 backbone in llama.cpp

2026-10-16. First llama.cpp piece: the converter, a GGUF of the backbone, and a check against
fish-speech. llama.cpp master f3363593 built with CUDA for sm86 (`third_party/scripts/build-llamacpp.ps1`,
binaries in `third_party/llama.cpp/build/bin`).

## Converter

`conversion/drama3.py` plus one line in `conversion/__init__.py`, patch in `patches/drama3-converter.diff`.
It registers `FishOmniModel` as a subclass of the Qwen3.5 MoE text model, so the GGUF is plain `qwen35moe`
and stock llama.cpp loads it. The subclass sets `no_mtp`, maps the `bsgemm-moe` quant method onto the
existing fp8 block dequant, builds the tokenizer from `tokenizer.json` alone (the checkpoint's
`tokenizer_config.json` is in the transformers 5 format), and writes `suppress_tokens` for every id except
EOS 248046 and the 4096 semantic ids. The `audio_decoder*` tensors are dropped by the base class filter.

Checks: expert dequant matches fish-speech's `dequantize_fp8_blocks` with max abs error 0.0 (three
experts across layers 0 and 39). The pre-tokenizer hash is already known as `qwen35`. Header: 733 tensors,
vocab 252288 (72 pad rows), eos 248046, pad 248044, no bos, 248191 suppressed ids, 34.68B params.

    python convert_hf_to_gguf.py D:\models\drama-3 --outtype bf16 --outfile drama-3-bf16.gguf   # 594 s, 64.6 GiB
    llama-quantize --tensor-type ffn_gate_exps=q4_k --tensor-type ffn_up_exps=q4_k \
        --tensor-type ffn_down_exps=q4_k drama-3-bf16.gguf drama-3-q4k-experts-q8-backbone.gguf Q8_0 32   # 188 s

Files in `D:\models\drama-3-own\llamacpp\`. The quantised file is 19.37 GiB (4.80 bpw): 120 expert tensors
Q4_K, 312 tensors Q8_0 including token_embd, output and the shared experts, norms and SSM small tensors F32.
It is 3.5 GiB smaller than the audio.cpp package because the tower and codec are not in it.

## Logits against fish-speech

Prompt: the Drama 3 ChatML layout for "Hello there, how are you today?" with speaker 0, no reference, no
history. fish-speech's `build_drama3_prompt` and `llama-tokenize` both give the same 29 ids
(`verify/prompt_tokens.txt`). Reference: fish-speech 54156693, backbone only, BF16 on the CPU, experts
dequantized at load, `verify/ref_backbone.py`. llama.cpp side: `verify/logits.cpp` decodes the ids in one
batch and dumps the last-row logits; `verify/compare.py` does the comparison.

| GGUF | where | full vocab corr | KL | eos+semantic corr | KL | top 10 shared | top 1 |
|---|---|---|---|---|---|---|---|
| BF16 | CPU | 0.99990 | 0.0019 | 0.99900 | 0.0054 | 10/10 | 988 vs 1385 |
| Q4_K experts, Q8_0 rest | 3090 Ti | 0.99928 | 0.0090 | 0.99619 | 0.0217 | 9/10 | 988 vs 1385 |

KL is KL(reference || llama.cpp) over the softmax of the named set. The reference's top two semantic
tokens (988 at 12.75, 1385 at 12.69) swap in both GGUFs; the gap is inside BF16 noise. Over the whole
vocab the argmax is the text token `[` in all three, 4.6 logits above the best semantic token, which is
why the GGUF needs `suppress_tokens`: fish-speech never samples outside EOS plus semantic. EOS sits at
rank 3862 of 4097 in the reference and 3616 at Q4.

One prompt, text rows only. Audio rows (mixed embeddings), the trimmed head and longer prompts are not
checked yet.

## llama-bench, 3090 Ti

`-ngl 99 -p 512 -n 128 -r 3`, default threads, GPU pinned by UUID. Raw output in `llamacpp-bench-*.txt`.

| n_cpu_moe | pp512 t/s | tg128 t/s | ms per token |
|---|---|---|---|
| 0 | 4703 | 160.8 | 6.2 |
| 4 | 2014 | 133.3 | 7.5 |
| 8 | 1230 | 116.2 | 8.6 |

The whole quantised backbone fits on the card at n_cpu_moe 0: 19.3 GiB of weights plus a 0.5 GiB
compute buffer at a 512 context, token_embd stays on the host. With the tower, codec, KV and a small
LLM to add, 8 CPU layers is the realistic point, and its 8.6 ms a token leaves room against the
13 to 15 ms backbone share measured in audio.cpp. Prompt processing at 1230 t/s is 0.8 ms a row
against audio.cpp's 11 ms a row cold.

Same build, Qwen3.6-35B-A3B UD-Q4_K_M fully on the card, for scale: pp128 2250, tg32 142.
