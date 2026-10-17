param([string]$Models)

$DRAMA3_REPO = "EryriLabs/drama-3-llamacpp-GGUF"

if (-not $Models) { $Models = $env:MODELS }
if (-not $Models -and (Test-Path "$PSScriptRoot\.env")) {
    $m = Select-String -Path "$PSScriptRoot\.env" -Pattern '^MODELS=(.+)$'
    if ($m) { $Models = $m.Matches[0].Groups[1].Value.Trim() }
}
if (-not $Models) { $Models = "./models" }

$hf = "https://huggingface.co"
$sherpa = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

function Get-File($path, $url) {
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Force (Split-Path $path) | Out-Null
        Write-Host "downloading $path"
        curl.exe -L -C - --fail --progress-bar -o "$path.part" $url
        if ($LASTEXITCODE -ne 0) { Write-Host "download failed: $url"; exit 1 }
        Move-Item "$path.part" $path
    }
    Write-Host ("{0,8:N0} MB  {1}" -f ((Get-Item $path).Length / 1MB), $path)
}

Get-File "$Models/drama-3-own/llamacpp/drama-3-q4k-experts-q8-backbone.gguf" `
    "$hf/$DRAMA3_REPO/resolve/main/drama-3-q4k-experts-q8-backbone.gguf"
Get-File "$Models/drama-3-own/llamacpp/drama-3-mmproj-q8_0.gguf" `
    "$hf/$DRAMA3_REPO/resolve/main/drama-3-mmproj-q8_0.gguf"
Get-File "$Models/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-UD-Q4_K_XL.gguf" `
    "$hf/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-UD-Q4_K_XL.gguf"
Get-File "$Models/silero-vad/silero_vad.onnx" "$sherpa/silero_vad.onnx"

$asr = "$Models/parakeet-tdt-0.6b-v3"
$files = "encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"
if ($files | Where-Object { -not (Test-Path "$asr/$_") }) {
    $tgz = "$Models/parakeet.tar.bz2"
    Get-File $tgz "$sherpa/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
    $dir = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    New-Item -ItemType Directory -Force $asr | Out-Null
    & "$env:SystemRoot\System32\tar.exe" -xjf $tgz -C $asr --strip-components=1 ($files | ForEach-Object { "$dir/$_" })
    if ($LASTEXITCODE -ne 0) { Write-Host "could not unpack $tgz"; exit 1 }
    Remove-Item $tgz
}
foreach ($f in $files) { Get-File "$asr/$f" "" }
