



# mara-drama3

Fish Audio's Drama 3, a 35B text-to-speech model that whispers, laughs and shouts
on cue, running in llama.cpp on one 24 GB card next to a small chat LLM, with a
web page you talk to.


https://github.com/user-attachments/assets/b57dcbbc-5e08-47e1-b16b-fd2076bd1280


[demo/demo.mp4](demo/demo.mp4) is a short clip of it in use.

## Run it

You need Docker Desktop with GPU support and an NVIDIA card with 24 GB
(RTX 3090, 4090 or 5090). From `demo`:

    .\start.ps1

The first run downloads about 27 GB of models and builds the image, then
prints "open http://localhost:8000". Press the round button and talk. Details,
settings and the terminal version are in [demo/README.md](demo/README.md).

## What is inside

- `third_party/llama.cpp`, a fork with the Drama 3 pieces: the converter to
  GGUF, the codebook tower as an mtmd graph with a persistent cache, the DAC
  codec, and a pipeline that takes reference clips and earlier turns as codes
  and streams audio as it decodes.
- `demo/say.cpp`, a small process that keeps the models loaded and renders a
  sentence per request, with a rolling history of what it last said so
  sentences join.
- `demo/web`, the page and its server: Parakeet TDT 0.6B v3 and Silero VAD on
  the CPU for hearing, Gemma 4 E4B for the answers, Drama 3 for the voice. The
  LLM puts a mood label in front of every sentence; the server turns it into a
  direction and picks a same-voice reference clip for it.
- `DESIGN.md`, the plan this follows, with what was measured along the way;
  `bench/`, the numbers; `verify/`, the checks against fish-speech.

The models themselves are not in the repo. The Drama 3 GGUFs are at
[EryriLabs/drama-3-llamacpp-GGUF](https://huggingface.co/EryriLabs/drama-3-llamacpp-GGUF);
the download script fetches those, Gemma 4, Parakeet and Silero.

## Mara's voice

Mara speaks with the voice of speaker p283 from the CSTR VCTK Corpus
(University of Edinburgh), one sentence of which is the reference clip in
`verify/refs/p283.codes`; the mood clips in `demo/bank` were rendered from it.
VCTK is released under the Creative Commons Attribution 4.0 licence:
Yamagishi, Veaux and MacDonald, "CSTR VCTK Corpus: English Multi-speaker
Corpus for CSTR Voice Cloning Toolkit (version 0.92)", 2019,
https://doi.org/10.7488/ds/2645. If you ship the voice, keep the credit.

## Thanks

Fish Audio for Drama 3 and fish-speech, which was the ground truth for every
check here; the llama.cpp project; NVIDIA for Parakeet and the sherpa-onnx
team for its ONNX export; Unsloth for the Gemma 4 quants; the CSTR at
Edinburgh for VCTK.

## Licence

The code here is MIT. Drama 3 and the converted GGUFs are under the Fish Audio
Community License, which allows research and non-commercial use freely and
commercial use below a revenue threshold; read it before shipping anything.
