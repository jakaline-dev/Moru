#!/bin/bash

#cd "$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export MAMBA_ROOT_PREFIX="$HOME/.micromamba"
RELEASE_URL="https://micro.mamba.pm/api/micromamba/linux-64/latest"

check(){
	if [ ! -d "$MAMBA_ROOT_PREFIX" ]; then
		install
	else
		eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook -s posix)" && micromamba activate Moru || install
	fi
}

install(){
	mkdir -p "$MAMBA_ROOT_PREFIX"
	curl -Ls $RELEASE_URL | tar -xvj --strip-components=1 -C $MAMBA_ROOT_PREFIX bin/micromamba
	eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook -s posix)"
	micromamba create -f env-linux.yml -y
	micromamba clean -a -f -y
	micromamba activate Moru
	pip cache purge
}

check
eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook -s posix)"
micromamba activate Moru
clear

while true
do
	options=("Start Trainer" "Update Moru" "Open CMD" "Exit")
	select command in "${options[@]}"
	do
		case $command in
			"Start Trainer")
				cd "$SCRIPT_DIR/trainer"
				python train_SDXL.py
				;;
			"Update Moru")
				echo "2"
				micromamba update -f env-win.yml -y
				micromamba clean -a -f -y
				pip cache purge
				;;
			"Open CMD")
				echo "3"
				;;
			"Exit")
				echo "4"
				break;;
			*)
				echo "Ooops";;
		esac
	done
done