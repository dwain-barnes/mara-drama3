FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY third_party/llama.cpp /src/llama.cpp
WORKDIR /src/llama.cpp
RUN ln -s libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LDFLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89;120" \
        -DGGML_NATIVE=OFF -DGGML_CUDA_NCCL=OFF \
        -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    && cmake --build build --target llama-server mtmd

COPY demo/say.cpp demo/CMakeLists.txt /src/say/
RUN cmake -S /src/say -B /src/say/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_DIR=/src/llama.cpp -DLLAMA_BUILD=/src/llama.cpp/build \
    && cmake --build /src/say/build \
    && mkdir /out && cp -a /src/llama.cpp/build/bin/llama-server /src/llama.cpp/build/bin/*.so* /src/say/build/say /out/


FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --break-system-packages numpy sherpa-onnx aiohttp

WORKDIR /app
COPY --from=build /out/ /app/bin/
COPY verify/refs/p283.codes verify/refs/p283.txt /app/refs/

ENV LD_LIBRARY_PATH=/app/bin \
    BACKBONE=/models/drama-3-own/llamacpp/drama-3-q4k-experts-q8-backbone.gguf \
    MMPROJ=/models/drama-3-own/llamacpp/drama-3-mmproj-q8_0.gguf \
    LLM_MODEL=/models/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-UD-Q4_K_XL.gguf \
    ASR_DIR=/models/parakeet-tdt-0.6b-v3 \
    VAD=/models/silero-vad/silero_vad.onnx \
    REF=/app/refs/p283.codes \
    BANK=/app/bank \
    LLM_BIN=/app/bin/llama-server \
    SAY_BIN=/app/bin/say \
    SAY_GPU=0 \
    LLM_GPU=0 \
    N_CPU_MOE=12 \
    HOST=0.0.0.0 \
    PORT=8000

# last, so editing the demo code does not rebuild anything above
COPY demo/ /app/
EXPOSE 8000
CMD ["python3", "web/server.py"]
