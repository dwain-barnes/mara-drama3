# Drama 3 reference prompts and history turns in llama.cpp

2026-10-16. Fourth llama.cpp piece: the drama3 pipeline in `mtmd-helper-gen.cpp` now takes reference clips
(the voice to clone) and earlier turns (the last few seconds the agent spoke) and lays the prompt out the way
fish-speech's `build_drama3_prompt` does. Same tree and build as `llamacpp-tower.md`, nothing committed.

## What was built

References and history arrive as codes, not audio. A bank clip is encoded once offline with
`verify/encode_reference.py` (fish-speech's DAC encoder on the CPU, a few seconds a clip) into the format the
pipeline already writes to `$MTMD_DRAMA3_CODES_OUT`: int32 rows of c0..c9, one per frame, with the transcript in
the `.txt` of the same name. The codes file of one render is therefore a history turn for the next without any
conversion. The helper API takes them in memory (`mtmd_helper_gen_audio_codes`: text, codes, frame count; `refs`
and `history` on `mtmd_helper_gen_audio_inp`), and `llama-tts` fills those from `MTMD_DRAMA3_REFS` and
`MTMD_DRAMA3_HISTORY`, `;`-separated lists of codes files. The audio reference `--tts-speaker-file` is still ignored
with a warning.

The prompt is fish-speech's, literal by literal (the exact segments are in `drama3-tower-spec.md`): the system
turn switches to "convert the provided text to speech reference to the following:" and carries one
"[Reference i] Text: ... Speech: " block per clip, followed directly by the clip's audio rows; each history turn
is a completed user text turn and an assistant `<|voice|>` turn of audio rows; then the target text and the open
assistant `<|voice|>`. An audio row is `[248099 + c0, c0..c9]`. Its backbone input is
(token embedding + codebook 1 embedding of c1) / sqrt(2), the same mix `step_gen` already fed back for generated
frames; the codebook rows come from the mmproj's table through a new accessor, `mtmd_gen_get_code_embd`, which
reads and dequantizes rows straight out of the GPU buffer. The tower prefill now gets the real codes of each
audio row (its own c2..c9 as `cur_idx`, the next row's c0..c8 as `next_codes`, the flags per row), and the
repetition window is seeded with the prompt's last 16 main tokens as fish-speech's eager path does. The text-only
prompt gives byte-identical codes to the run before this change (seed 42, 45 frames).

## Rerun

    cd D:\drama3\verify
    .venv\Scripts\python.exe encode_reference.py D:\models\drama-3 D:\drama3\bench\out\L12\neutral_long_L12.wav refs\neutral_long.codes --text "..."
    .venv\Scripts\python.exe ref_prompt.py D:\models\drama-3 --fish-speech D:\drama3\third_party\fish-speech --ref refs\neutral_long.codes
    build\pipeline.exe <backbone.gguf> <tower.gguf> out\pipeline_ref --prefix ref_prompt --ref refs\neutral_long.codes
    .venv\Scripts\python.exe compare_prompt.py out\pipeline_ref\prefill.logits --hidden out\pipeline_ref\hidden.bin

`ref_prompt.py` builds the same prompt through `build_drama3_prompt`, runs the BF16 backbone and the
teacher-forced tower on the CPU (69 GiB, 55 s to load, 15 s for the 245 rows) and dumps every row's hidden state
and head logits; `--ids-only` writes just the prompt ids in 3 s. `pipeline.cpp` in reference mode collects every
row's hidden state chunk by chunk from `step_prompt`, and `$MTMD_DRAMA3_LOGITS_OUT` makes the pipeline append the
prefill head logits. The tower on its own, fed the reference hidden states, is `tower.exe` run from
`out\tower_ref`, where the `ref_prompt_*` files sit under the `ref_tower_*` names with K = 0
(`compare_tower.py` now takes the scored rows from `ref_tower_semantic.npy` instead of P and K).

## Against fish-speech

The test prompt is "Hello there, how are you today?" with one reference, `neutral_long_L12.wav` from the
audio.cpp bench (Drama 3's own voice, 7.76 s, 167 frames). 245 rows: 58 text rows, the 167 audio rows (58..224),
20 text rows after them. With a history turn on top (the 26-frame clause 1 render below) it is 286 rows.

Prompt ids: identical to fish-speech's on all 245 rows, and on all 286 of the history prompt. That closes the
tokenizer question for reference prompts: the literals around the audio rows ("\n\n[Reference 1]\nText: ",
"\nSpeech: " with its trailing space, the `<|im_end|>\n` straight after the last audio row) tokenize the same in
llama.cpp as in `tokenizers`.

Backbone hidden state per row, cosine to the BF16 reference (Q4_K experts, Q8_0 elsewhere, on the GPU; and the
BF16 GGUF on the CPU, no GPU at all):

| | Q4 GPU | BF16 CPU |
|---|---|---|
| median | 0.99826 | 0.99927 |
| rows below 0.99 | 12 of 245 | 2 of 245 |
| audio rows, min | 0.98267 (row 178) | 0.98985 (row 192) |
| first 3 audio rows 58..60 | 0.99826 0.99816 0.99874 | 0.99795 0.99648 0.99836 |
| first 3 rows after the audio 225..227 | 0.99681 0.99924 0.99950 | 0.99993 0.99990 0.99995 |
| last row 244 (`<|voice|>`) | 0.99250 | 0.99202 |
| reference's top-1 main token, rank in the last row's logits | 1 of 4097 | 1 of 4097 |

Ten of the twelve low Q4 rows are above 0.99 with the BF16 backbone, most above 0.995, so they are quantisation
noise. The rows right after the audio block, the first ones through the linear-attention layers after embedding
input, are the best rows of the prompt at BF16 (0.9999). Row 56 is the odd one: the ":" of "Speech:", a text row
before any audio, at 0.947 (Q4) and 0.749 (BF16) to the reference, and 0.88 between the two llama.cpp runs,
while rows 54, 55 and 57 agree to 0.993 or better and nothing after it is disturbed. Three implementations
giving three answers for one row whose neighbours agree is an expert routing near a tie, not a prompt error; the
same token at rows 13 and 22 is at 0.999. `pipeline.cpp` therefore passes on the ids, the first audio row, the
first row after the audio, the last row and determinism, and prints the count of low rows.

Tower head logits on the 167 rows whose next row is audio (1503 codes), with the tower fed by the llama.cpp
backbone (the pipeline as it runs) and, separately, by the reference's own hidden states (the tower alone, as
in `llamacpp-tower.md`):

| | SNR, dB | top-1 | top-5 shared |
|---|---|---|---|
| pipeline, Q4 backbone + Q8_0 tower, GPU | 34.0 to 39.4, mean 36.9 | 1316/1503 (0.876) | 6884/7515 (0.916) |
| pipeline, BF16 backbone + Q8_0 tower, CPU | 33.7 to 39.2, mean 36.8 | 1323/1503 (0.880) | 6877/7515 (0.915) |
| tower alone, Q8_0, GPU | 40.6 to 43.1, mean 41.6 | 1379/1503 (0.917) | 7070/7515 (0.941) |
| tower alone, bf16, GPU | 47.8 to 49.9, mean 48.8 | 1417/1503 (0.943) | 7247/7515 (0.964) |

The tower alone on real audio rows sits where the six greedy rows of the text-only check put it (Q8_0 40 to 44 dB,
bf16 48 to 51 dB). Per-layer SNR on these rows is 31 to 40 dB at Q8_0 and 40 to 48 dB at bf16, and the rewind to
row 25 is bit-identical again. The 5 dB and 4 points the pipeline loses on top come from the backbone's hidden
states, and the BF16 backbone on the CPU loses the same, so it is the hidden states' own sensitivity (the ~0.998
cosine rows) rather than the Q4 quantisation. How often the top choice is the clip's actual next code is 159 of
1503 for the reference and 152 to 156 for every llama.cpp configuration: the tower does not reproduce a real
recording's residual codes from the semantic code, and llama.cpp does so exactly as often as fish-speech.

## llama-tts end to end

The plain 3090 (the Ti is still off the bus), Q4 backbone, Q8_0 tower, seed 42, codes decoded with fish-speech's
DAC on the CPU and transcribed with faster-whisper small.en. Every run stopped on eos.

| run | prompt rows | frames | audio | prompt eval | generation | transcript |
|---|---|---|---|---|---|---|
| "Hello there, how are you today?", reference | 245 | 48 | 2.23 s | 0.35 s | 0.54 s | Hello there. How are you today? |
| same, seed 7 | 245 | 48 | 2.23 s | 0.48 s | 0.55 s | Hello there. How are you today? |
| "I checked the config twice, and the port was wrong the whole time.", reference | 252 | 93 | 4.32 s | 0.49 s | 0.99 s | I checked the config twice, and the port was wrong the whole time. |
| clause 1 "So the thing is,", reference | 242 | 26 | 1.21 s | 0.38 s | 0.33 s | So the thing is... |
| clause 2 "I checked the config twice,", reference + clause 1 as history | 286 | 42 | 1.95 s | 0.44 s | 0.48 s | I check the config twice. |
| clause 1 + clause 2 joined | | 68 | 3.16 s | | | So the thing is, I checked the config twice. |

About 11 ms a frame once the prompt is in, and 0.35 to 0.5 s for a 245 to 286 row prompt (one backbone batch plus
two tower prefill calls). The wavs are in `verify\out\refs\` (`hello_ref.dac.wav`, `hello_ref_s7.dac.wav`,
`long_ref.dac.wav`, `clause12.wav`); whether they are in the reference's voice is for the ear, Whisper only
confirms the words. Clause 2 on its own is heard as "check" by Whisper and as "checked" in the joined file.

## Prompt cap

fish-speech caps the prompt at 8192 rows and, when a chunk's prompt is longer, retries with less history before
giving up. The pipeline does the same: the limit is the smaller of 8192 and the llama context less 257 rows (a
prompt that fills the context would stop on "length" at the first frame; 256 rows is about 12 s of speech). Over
the limit it drops history turns oldest first, one at a time with a warning, and refuses with an error once the
prompt has no history left. References are never dropped. fish-speech's own drop order keeps the first chunk of a
long text pinned and drops the turns after it first; the agent's history is a rolling window with nothing pinned,
so oldest first is the only order here. Checked with the clause 2 prompt (286 rows with its history turn, 243
without): at `-c 512` the turn is dropped and 45 frames come out; at `-c 256` set_input fails with
"the prompt has 243 rows without history, at most 0 fit".

## Still open

The codec in llama.cpp (another session is adding it); `--tts-speaker-file` as audio would need that codec's
encoder; the Ti figures.

## Do the tags survive Q4? (same day, after the chat demo)

The audio.cpp bench had noted that the paired `<whisper>` tag changed nothing, and nobody had judged the Q4
delivery by ear (DESIGN item 6). The demo in `D:\drama3\demo` made the question testable: `say.exe` keeps the
models loaded and renders a line per stdin request, and `tag_ab.py` renders one sentence ("I really did not
expect that, and I have to say, it changes everything we planned for tomorrow.") under eight variants with the
same reference (`verify\refs\p283.codes`, a VCTK speaker at 32 kHz resampled to 44.1 kHz, 252 frames) and seed,
then measures each wav: rms and peak, the loudest 0.5 s, the share of energy above 4 kHz, and a crude median f0
with its interquartile range over voiced 40 ms windows. Q4 backbone on the GPU; the BF16 GGUF on the CPU alone
(about a minute a render) as the quantisation-free control. The two are different draws, so compare the direction
and size of each effect, not the values.

| variant | Q4: s, rms, loud 0.5 s, >4 kHz, f0 med / iqr | BF16: s, rms, loud 0.5 s, >4 kHz, f0 med / iqr |
|---|---|---|
| plain | 5.8, -17.7, -13.3, 3%, 213 / 35 | 5.5, -19.8, -16.2, 2%, 204 / 29 |
| `<whisper>` ... `</whisper>` | 10.0, -30.7, -23.6, 60%, unvoiced | 5.3, -35.7, -31.8, 55%, unvoiced |
| `[whispering]` | 5.9, -33.9, -29.4, 44%, unvoiced | 6.0, -35.7, -31.7, 41%, unvoiced |
| `[excited]` | 5.3, -12.9, -9.9, 1%, 241 / 67 | 5.9, -18.2, -14.6, 3%, 230 / 40 |
| `[angry, shouting]` | 7.5, -15.9, -11.7, 4%, 226 / 43 | 5.8, -13.2, -10.1, 2%, 269 / 56 |
| `[serious, measured]` | 5.4, -18.8, -15.3, 3%, 209 / 25 | 6.0, -19.2, -14.0, 4%, 198 / 27 |
| `[sad, quiet]` | 5.9, -20.1, -14.1, 3%, 206 / 19 | 6.2, -19.4, -14.7, 3%, 213 / 20 |
| `[laughs]` after "that," | 6.4, -19.5, -15.9, 4%, 207 / 17 | 6.4, -17.7, -12.1, 2%, 206 / 24 |

Whisper is a whisper at both precisions: 13 to 16 dB quieter and half the energy above 4 kHz, which is breath
instead of voicing (the f0 tracker finds nothing to lock on to). Excited and shouting are louder with a higher,
wider pitch; serious and sad narrow it. Q4 and BF16 move the same way on every tag, so the experts' quantisation
is not what decides this, the reference and the direction are. faster-whisper gets every word in all 16 clips
(the one with the `<whisper>` tag at Q4 is slow, 10 s, the only outlier). The wavs are in `demo\out\tag_ab_q4`
and `demo\out\tag_ab_bf16`.

The bank the chat demo uses (`demo\bank`, ten clips rendered from the p283 reference with one direction each,
`make_bank.py`) spans the same range: whisper -24.9 dBFS, neutral -17.5, excited -13.0, and all ten transcribe.
In the demo the LLM (Gemma 4 E4B by default, Qwen3.6-35B-A3B as the alternative) starts every sentence with a
label from ten, chat.py switches the reference clip to that affect's bank entry and keeps the direction in the
text, and say.exe renders it in 0.9 to 1.6 s. Dwain's ear says it sounds right.
