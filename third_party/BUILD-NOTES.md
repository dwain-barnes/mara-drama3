# audio.cpp build notes

Built on 2026-10-16 from audio.cpp main, commit 624cf918e51263d39c2a0d28a6c5e3d55990b79f
("Optimize Drama 3 output head for speech token candidates (#931)"). Source is in
`D:\drama3\third_party\audio.cpp`.

## What is installed

C: had about 8 GB free, so nothing big went there.

- Visual Studio Build Tools 2022 17.14.41 with the C++ workload (MSVC 14.44.35207), installed to
  `D:\VS\BuildTools` (cache in `D:\VS\cache`). It includes VS's own CMake 3.31.6 and Ninja.
- CMake 4.4.4 from winget (`C:\Program Files\CMake`). The build script prefers the VS copy.
- CUDA toolkit 13.4.1 (nvcc 13.4.59), unpacked by hand to `D:\CUDA\v13.4`. The winget installer needs
  several GB on C: and a UAC prompt, so I took the official installer (SHA256 matched the winget
  manifest), pulled the nvcc, cudart, crt, cccl, nvrtc, cublas, nvvm, nvptxcompiler, nvfatbin,
  nvjitlink, cufft and curand parts out of it, and merged them into one tree. There is no Visual Studio
  integration and nothing is registered; the build finds it through `CUDA_PATH`. The installer is kept
  in `D:\drama3\third_party\dl`.

## Build command

`scripts\build.ps1` in this folder sets TEMP to `D:\tmp`, points CUDA_PATH and CUDAToolkit_ROOT at
`D:\CUDA\v13.4`, then runs:

    .\scripts\build_windows.ps1 -Preset windows-cuda-release -ModelSet custom -Models fish_drama3 `
      -Target <target> -Jobs 16 -CudaArchitectures 86 -VsInstall D:\VS\BuildTools

Run it as `powershell -NoProfile -ExecutionPolicy Bypass -File D:\drama3\third_party\scripts\build.ps1 -Target audiocpp_cli`.
Targets built: `audiocpp_cli`, `audiocpp_gguf`, `audiocpp_server`. The cli took about 5 minutes.

## Binaries

`D:\drama3\third_party\audio.cpp\build\windows-cuda-release\bin`: audiocpp_cli.exe, audiocpp_server.exe,
audiocpp_gguf.exe, plus cublas64_13.dll and cublasLt64_13.dll copied from `D:\CUDA\v13.4\bin\x64`.
cudart is linked statically. Model specs are embedded in the exe, there is no model_specs folder next
to it. Run the exe from PowerShell or cmd; from Git Bash it fails to find the CRT DLLs.

`audiocpp_cli.exe --help` does not list families. `--list-loaders` shows
`fish_drama3: tts (offline|streaming), clon (offline|streaming)`.

## Converting the original checkpoint

Script: `tools\community_models\convert_fish_drama3.py`. It is a uv inline-script, so
`uv run --python 3.12 <script> ...` installs its dependencies itself: Python >= 3.10, gguf >= 0.19,
numpy >= 1.26, safetensors >= 0.5, and CPU-only torch >= 2.6 from the pytorch-cpu index. It never loads
weights on a GPU. The checkpoint dir needs `config.json` (model_type fish_omni),
`model.safetensors.index.json`, the shards, `codec.pth` and the tokenizer files. Docs pin revision
dd64a85cd82e9521f7202610ede97b367fc83946.

Disk: the FP16 intermediate GGUF (`--output`) is about 68.2 GiB and is kept after conversion; the
quantized package is about 22.9 GiB for Q4_K experts / Q8_0 backbone. Put both on D:. Set
`UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` to D: before running, because C: is nearly full and the
torch wheel and Python download land there by default.

    uv run --python 3.12 tools\community_models\convert_fish_drama3.py `
      --checkpoint D:\models\drama-3 `
      --output D:\models\drama3-f16.gguf `
      --quantized-output D:\models\drama-3-q4k-experts-q8-backbone.gguf `
      --gguf-tool D:\drama3\third_party\audio.cpp\build\windows-cuda-release\bin\audiocpp_gguf.exe `
      --expert-type q4_k --backbone-type q8_0 --embedding-type q8_0 --threads 16

Run it from the repo root (or pass full paths); it refuses to overwrite existing outputs. To redo only
the quantization step, use `--input-gguf <f16.gguf>` instead of `--output`.

# llama.cpp build notes

Built on 2026-10-16 from llama.cpp master, commit f3363593f457169cdfbdfe22ea399b1cc139ba18, in
`D:\drama3\third_party\llama.cpp`. Same toolchain as above. `scripts\build-llamacpp.ps1` sets TEMP, the CUDA
paths and the vcvars64 environment, then runs

    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
          -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
    cmake --build build -j 24

The whole thing took about 4 minutes. Binaries in `build\bin`, with the two cublas DLLs copied in. The log
in `logs\build_llamacpp.log` is UTF-16. Run the exes from cmd or with a full path; `./x.exe` from a Git Bash
shell whose cwd has moved fails quietly.

The converter needs a Python env: `D:\drama3\.venv-convert` (uv, Python 3.12, CPU torch, from
`requirements\requirements-convert_hf_to_gguf.txt`). Set `UV_CACHE_DIR=D:\uv\cache` and
`UV_PYTHON_INSTALL_DIR=D:\uv\python` first. The fish-speech reference env is `D:\drama3\verify\.venv`.
