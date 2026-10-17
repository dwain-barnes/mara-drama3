Set-Location $PSScriptRoot

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker is not running. Start Docker Desktop and try again."
    exit 1
}

& .\get-models.ps1
if ($LASTEXITCODE -ne 0) { exit 1 }

docker image inspect drama3-mara *> $null
if ($LASTEXITCODE -eq 0) {
    Write-Host "open http://localhost:8000"
    docker compose up
} else {
    Write-Host "First run: building the image, this takes a while."
    Write-Host "open http://localhost:8000 once it says ready"
    docker compose up --build
}
