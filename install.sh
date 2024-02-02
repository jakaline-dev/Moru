#!/bin/bash

cd "$(cd "$(dirname "$0")" && pwd)"

export MAMBA_ROOT_PREFIX="$(pwd)/.micromamba/linux"
RELEASE_URL="https://micro.mamba.pm/api/micromamba/linux-64/latest"

check(){
	if [ ! -d "$MAMBA_ROOT_PREFIX" ]; then
		echo "?????"
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

options=("Start Trainer" "Update Moru" "Open CMD" "Exit")
select command in "${options[@]}"
do
	case $command in
		"Start Trainer")
			echo "1"
			;;
		"Update Moru")
			echo "2"
			micromamba activate Moru
			micromamba
			;;
		"Open CMD")
			#exec bash && eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook -s posix)" && micromamba activate Moru
			;;
		"Exit")
			echo "4"
			break;;
		*)
			echo "Ooops";;
	esac
done