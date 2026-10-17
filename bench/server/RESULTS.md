# Drama 3 warm time-to-first-audio, audio.cpp HTTP server

Machine: RTX 3090 Ti only (CUDA_VISIBLE_DEVICES set to its UUID, so it shows up as device 0), TEMP and TMP on D:\tmp.
Model: drama-3-q4k-experts-q8-backbone.gguf, cpu_expert_layers=4, context_length=2048, streaming mode,
8 threads, max_tokens 500. The server was built from the tree in third_party\audio.cpp.

The server was started with lazy_load false, so the weights load at startup. The process was up and answering
/health 26.7 s after launch. VRAM was 21,410 MiB after the load and peaked at 21,812 MiB while serving. After the
process was killed the Ti was back at 0 MiB.

Each request is a streamed POST to /v1/audio/speech asking for raw 16-bit 44.1 kHz mono PCM. Client time is from
sending the request to the first non-empty body chunk. Server time is total_ms from the `[drama3.first_audio]`
line. Short text is "Yeah, that works." (26 prompt rows), long text is the 125 character logs sentence (50 rows).
Six requests each, alternating, seed 123, no voice reference.

| text | request | client TTFA ms | server total_ms | backbone prefill | tower prefill | codec first frame |
|---|---|---|---|---|---|---|
| short | 1 (cold) | 2464 | 2845 | 1453 | 240 | 291 |
| short | 2-6 median | 136 | 133 | 106 | 4.8 | 5.2 |
| long | 1 | 249 | 232 | 197 | 11 | 5.0 |
| long | 2-6 median | 208 | 205 | 176 | 5.4 | 5.1 |

Per request client TTFA in ms, short: 2464, 158, 121, 122, 147, 136. Long: 249, 219, 186, 219, 208, 189.

The first request pays a one-off warm-up (CUDA kernels, graph setup), about 2.5 s. After that the whole first-audio
path is about 130 ms for the short text and about 205 ms for the long one, and nearly all of it is backbone
prefill. Tower prefill and the first codec frame are about 5 ms each once warm, against about 80 ms and 70 ms in
the cold CLI runs. The long request was fast even as request 2 overall because the warm-up in request 1 had
already covered the kernels. Steady state decode was about 22 ms per frame (long, 153 frames in 3.3 s).

`drama3.prefill_reused_rows` was 0 on all 12 requests. The two texts alternate and share no prompt prefix, and
there was no voice reference, so there was nothing to reuse. Reuse was not tested with a repeated text or a fixed
reference.

Side note: the same seed did not give identical audio. All 12 WAVs have different hashes and the short text varied
from 1.0 s to 3.0 s of audio. The repo's benchmark asserts identical PCM, but that uses a voice reference and a
different offload setup. Not investigated further. Also in request 1 the client saw audio about 380 ms before the
server's total_ms; the cause is not known.

No errors in the server log, and no max_tokens warnings.

## Commands

Server (start_server.ps1, run with powershell):

    $env:CUDA_VISIBLE_DEVICES = 'GPU-a21cb855-8208-face-ba36-1b52373420a0'
    $env:TEMP = 'D:\tmp'; $env:TMP = 'D:\tmp'
    cd D:\drama3\bench\server
    audiocpp_server.exe --config drama3-server.json --log --log-file server.log
      (stdout and stderr redirected to server.out.txt and server.err.txt; the first_audio lines are in server.err.txt)

Client, from cmd with the same three environment variables set:

    python warm_ttfa.py

VRAM was sampled with `nvidia-smi --id=GPU-a21cb855-8208-face-ba36-1b52373420a0 --query-gpu=memory.used` every 0.5 s
into vram.log. The server was stopped with taskkill on the PID in server.pid.

Files: drama3-server.json, start_server.ps1, warm_ttfa.py, client_rows.json, server.log, server.err.txt, vram.log,
and short-N.wav / long-N.wav for each request.
