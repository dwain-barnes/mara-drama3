#!/bin/sh
# usage: ./get-models.sh [models folder]   (default $MODELS, else ./models)
DRAMA3_REPO=EryriLabs/drama-3-llamacpp-GGUF

MODELS=${1:-${MODELS:-./models}}
HF=https://huggingface.co
SHERPA=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models

get() {
    if [ -s "$1" ]; then
        echo "have $(du -h "$1" | cut -f1)	$1"
        return
    fi
    mkdir -p "$(dirname "$1")"
    echo "downloading $1"
    curl -L -C - --fail --progress-bar -o "$1.part" "$2" || exit 1
    mv "$1.part" "$1"
    echo "done $(du -h "$1" | cut -f1)	$1"
}

get "$MODELS/drama-3-own/llamacpp/drama-3-q4k-experts-q8-backbone.gguf" \
    "$HF/$DRAMA3_REPO/resolve/main/drama-3-q4k-experts-q8-backbone.gguf"
get "$MODELS/drama-3-own/llamacpp/drama-3-mmproj-q8_0.gguf" \
    "$HF/$DRAMA3_REPO/resolve/main/drama-3-mmproj-q8_0.gguf"
get "$MODELS/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-UD-Q4_K_XL.gguf" \
    "$HF/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-UD-Q4_K_XL.gguf"
get "$MODELS/silero-vad/silero_vad.onnx" "$SHERPA/silero_vad.onnx"

asr=$MODELS/parakeet-tdt-0.6b-v3
files="encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt"
need=
for f in $files; do
    [ -s "$asr/$f" ] || need=1
done
if [ -n "$need" ]; then
    tgz=$MODELS/parakeet.tar.bz2
    get "$tgz" "$SHERPA/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
    dir=sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8
    members=
    for f in $files; do members="$members $dir/$f"; done
    mkdir -p "$asr"
    tar -xjf "$tgz" -C "$asr" --strip-components=1 $members || exit 1
    rm "$tgz"
fi
for f in $files; do
    echo "have $(du -h "$asr/$f" | cut -f1)	$asr/$f"
done
