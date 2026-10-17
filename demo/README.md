# One command

With Docker Desktop (GPU support on) and an Nvidia card, from this folder:

    .\start.ps1        (Windows)
    ./start.sh         (Linux)

The first run downloads about 28 GB of models and builds the image, which
takes 20 to 40 minutes on top of the download. After that it starts in about
a minute. When the log says ready, open http://localhost:8000. Use
localhost, the microphone does not work on other addresses without https.

The models are not in the image. They go in `D:/models` on Windows and
`./models` on Linux; set `MODELS` (in `.env` or the environment) to use
another folder. `get-models.ps1` / `get-models.sh` fills it, and `start`
runs it every time, so it only downloads what is missing and picks up a
half-finished file where it stopped. The layout is the one in `Dockerfile`.
The files come from:

- Drama 3: https://huggingface.co/EryriLabs/drama-3-llamacpp-GGUF, Fish
  Audio's Drama 3 converted for llama.cpp (`DRAMA3_REPO` at the top of the
  two scripts)
- Gemma 4 E4B: unsloth/gemma-4-E4B-it-GGUF on Hugging Face
- Parakeet and Silero VAD: the sherpa-onnx asr-models release on GitHub

`N_CPU_MOE` is how many layers of Drama 3's experts stay on the CPU. The
default of 12 fits Drama 3 and Gemma 4 on one 24 GB card with room to spare.
With nothing else on the card, 8 is about 0.4 s quicker to first audio:

    $env:N_CPU_MOE = 8; .\start.ps1

`GPU_ID` picks the card when there is more than one, as an index or a UUID
from `nvidia-smi -L`; the default is 0.

The image is built for RTX 30, 40 and 50 series cards (CUDA architectures
86, 89 and 120). For an older card, change `CMAKE_CUDA_ARCHITECTURES` in the
Dockerfile.

# Chat with a spoken answer

chat.py listens on the microphone (or takes typed lines), gets an answer from
a local Gemma 4 model (llama-server) and has Drama 3 say it through say.exe,
one sentence at a time, so speech starts before the answer is finished.
say.exe streams the audio back and chat.py plays it as it arrives, so
sentences run into each other without a gap.

    .venv\Scripts\python.exe chat.py

The venv needs numpy, sounddevice and sherpa-onnx:

    uv venv .venv --python 3.12
    uv pip install --python .venv/Scripts/python.exe numpy sounddevice sherpa-onnx

Speech is recognised by asr.py: NVIDIA parakeet-tdt-0.6b-v3 through
sherpa-onnx (int8 export from the sherpa-onnx asr-models release, in
`--asr-dir`, default D:\models\parakeet-tdt-0.6b-v3) and Silero VAD
(`--vad`, silero_vad.onnx from the same release). Both run on the CPU. A
segment ends after 0.5 s of silence, so a long pause in the middle of a
sentence starts a new turn. While Mara is speaking, and for 0.4 s after, the
mic is ignored (`[mic muted]` / `[mic open]` on stderr), so she doesn't hear
herself. Typing works at the same time; spoken and typed turns are handled
one after the other. `--no-mic` turns listening off.

Wait for "ready" (about a minute, the model loads on GPU 0 and Drama 3 on
GPU 2). Talk or type. `/reset` forgets the chat and Drama 3's history,
Ctrl-C or end of input quits and stops both processes. If llama-server is
already running on the port, add `--no-llm-start`.

Paths and ids are the defaults at the top of the file and in `--help`:
the Drama 3 models and reference under D:\models\drama-3-own and
D:\drama3\verify\refs, the Gemma gguf under D:\models, and the
CUDA_VISIBLE_DEVICES ids set in start_say and start_llm. TEMP is set to
D:\tmp for say.exe. Logs go to the out folder; `--save-wavs` also writes
each sentence to out\speech.
Build say.exe from say.cpp with the CMakeLists.txt here.

The LLM starts every sentence with a label such as `[warm]`. chat.py strips
it, puts the matching direction in front of the text, e.g.
`[sympathetic, gentle] ...`, and gives say.exe two references: the p283
recording first, then the affect's clip from `bank`. The clips (one per
label: neutral, warm, amused, excited, sympathetic, serious, hesitant,
apologetic, emphatic, whisper) are rendered from the recording by
`python make_bank.py` (with the Drama 3 card and TEMP set as above) and
encoded to `<affect>.codes` and `<affect>.txt`. They carry the mood but
drift toward Drama 3's own accent, which is why the recording comes first.
`[laughs]`, `[sighs]`, `<emphasis>` and `<whisper>` stay in the text. A
missing or unknown label counts as neutral, which uses the recording alone.
`--no-bank` drops the clips and leaves the mood to the direction text
(milder, about 0.4 s quicker a sentence). `--bank` points at another folder.

Qwen3.6-35B-A3B is the alternative LLM: pass
`--llm-model D:\models\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
It also follows the labels, but its first audio came about 3 s later in the
test.

## In the browser

web/server.py runs the same loop behind a web page, so anyone can talk to
Mara without the terminal. It needs aiohttp in the venv as well:

    uv pip install --python .venv/Scripts/python.exe aiohttp
    .venv\Scripts\python.exe web\server.py

and then open http://localhost:8000. The page says "waking up" until the
models are loaded. Press the round button and allow the microphone: what
you say goes to the server, and Mara answers out loud in the page, with
her words and the label of each sentence written underneath. The bars show
your voice while she listens and hers while she speaks. Press the button
again to stop listening. You can also type to her in the box at the
bottom, and "start over" forgets the conversation. One visitor at a time;
a second one is told she is busy. Reloading the page starts a new chat.

The flags match chat.py, except that `--port` is the page's port and
`--llm-port` is llama-server's (8089). Every path and setting can also come
from the environment (BACKBONE, MMPROJ, REF, BANK, LLM_MODEL, LLM_BIN,
SAY_BIN, ASR_DIR, VAD, PORT, N_CPU_MOE, SAY_GPU, LLM_GPU, plus LLM_PORT and
HOST); a flag beats the environment. `--n-cpu-moe` goes to say.exe. Unlike
chat.py, the server doesn't set TEMP for say.exe, so set it to D:\tmp
first if that matters on this machine. Logs go to out\web-say.log and
out\web-llama-server.log. The microphone only works on localhost or over
https, which browsers require for getUserMedia.
