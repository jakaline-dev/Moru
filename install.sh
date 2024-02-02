#!/bin/bash

cd "$(cd "$(dirname "$0")" && pwd)"

export MAMBA_ROOT_PREFIX="/mnt/$(pwd)/.micromamba/linux"
RELEASE_URL="https://micro.mamba.pm/api/micromamba/linux-64/latest"

curl -Ls RELEASE_URL | tar -xvj --strip-components=1 -C $MAMBA_ROOT_PREFIX bin/micromamba
eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook -s posix)"
micromamba create -f env-linux.yml -y
micromamba clean -a -f -y
micromamba activate Moru
pip cache purge