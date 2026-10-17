#!/bin/sh
cd "$(dirname "$0")" || exit 1

if ! docker info >/dev/null 2>&1; then
    echo "Docker is not running. Start it and try again."
    exit 1
fi

MODELS=${MODELS:-./models}
export MODELS
./get-models.sh || exit 1

if docker image inspect drama3-mara >/dev/null 2>&1; then
    echo "open http://localhost:8000"
    docker compose up
else
    echo "First run: building the image, this takes a while."
    echo "open http://localhost:8000 once it says ready"
    docker compose up --build
fi
