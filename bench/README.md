# Drama 3 benchmark

`run_baseline.py` runs `audiocpp_cli.exe` once per test line and per `fish_drama3.cpu_expert_layers` value, in streaming mode, and writes one row per run to `results.csv`.

Per run it records time to first audio (`total_ms` from the `[drama3.first_audio]` stderr line), generated frames, frames per second, real-time factor, output duration, leading and trailing silence, and peak VRAM sampled from `nvidia-smi` every 0.25 s. The `parts` column holds the per-phase `*_ms` values.

The CLI loads the model on every call. `wall_ms` is `session.wall_ms` from the log when present, which leaves out loading. Otherwise it is process time and the note column says so. `proc_wall_ms` is always process time. You need an audio.cpp build that includes the `fish_drama3` backend (v0.9.2 does not).

    python run_baseline.py --cli C:\audio.cpp\build\windows-cuda-release\bin\audiocpp_cli.exe --model D:\models\drama-3-backbone-8bit-experts-Q2_K.gguf --layers 4,8,12 --repeat 3

Pass `--gpu-uuid GPU-...` if nvidia-smi numbers the cards differently from CUDA.
