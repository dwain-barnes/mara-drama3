$DRAMA3_REPO = "EryriLabs/drama-3-llamacpp-GGUF"
$src = "D:\models\drama-3-own\llamacpp"
$license = "D:\models\drama-3\LICENSE"
$card = "D:\tmp\drama3\hf-readme.md"

$hf = "$PSScriptRoot\.venv\Scripts\hf.exe"

$text = @'
---
license: other
license_name: fish-audio-community-license
license_link: LICENSE
base_model: fishaudio/drama-3
tags:
  - gguf
  - llama.cpp
  - text-to-speech
---

# Drama 3 for llama.cpp

[Fish Audio's Drama 3](https://huggingface.co/fishaudio/drama-3) converted to
GGUF for llama.cpp. Thank you to Fish Audio for releasing the model. Built
with Fish Audio.

| File | What it is | Size |
| --- | --- | --- |
| drama-3-q4k-experts-q8-backbone.gguf | the backbone, routed experts in Q4_K and everything else in Q8_0 | 20.8 GB |
| drama-3-mmproj-q8_0.gguf | the codebook tower and the DAC codec, Q8_0 | 1.7 GB |

Both were built from fishaudio/drama-3 with the converter in my llama.cpp
branch. To run them, see https://github.com/dwain-barnes/mara-drama3, which has
a Docker setup that downloads these files and starts a voice chat demo.

The licence is the original model's, the Fish Audio Community License 1.0. A
copy is in this repo as LICENSE.
'@
New-Item -ItemType Directory -Force (Split-Path $card) | Out-Null
[IO.File]::WriteAllText($card, $text)

& $hf auth whoami *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "run hf auth login first"
    exit 1
}

function Hf { & $hf @args; if ($LASTEXITCODE -ne 0) { exit 1 } }

Hf repos create $DRAMA3_REPO --type model --public --exist-ok
Hf upload $DRAMA3_REPO "$src\drama-3-q4k-experts-q8-backbone.gguf" drama-3-q4k-experts-q8-backbone.gguf
Hf upload $DRAMA3_REPO "$src\drama-3-mmproj-q8_0.gguf" drama-3-mmproj-q8_0.gguf
Hf upload $DRAMA3_REPO $license LICENSE
Hf upload $DRAMA3_REPO $card README.md
