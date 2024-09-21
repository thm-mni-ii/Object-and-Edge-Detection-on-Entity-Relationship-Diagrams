#!/bin/bash
VENV_PATH=".venv"

# Script Variables
OK='\033[0;32m' # Green
HINT='\033[1;33m' # Yellow
NC='\033[0m' # No Color

if [[ "$0" == "$BASH_SOURCE" ]]; then
  echo -e "${HINT}Please source this script!${NC}\n  source $0"
  exit
fi

# always go to location of script
OLD_DIR="$(pwd)"
cd "$(dirname '$BASH_SOURCE')"

# download weights
test -d weights || mkdir weights
cd weights
test -f yolo.pt || wget -O yolo.pt 'https://mndthmde-my.sharepoint.com/:u:/g/personal/maximilian_huber_mnd_thm_de/ERuYDWjopadGmWYJ4NuZqbIBZT8v7sd2ideJJ9ZzmLQcsw?e=yVKJhT&download=1'
test -f sam_vit_h_4b8939.pth || wget -O sam_vit_h_4b8939.pth 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
test -f b-low-lr.ckpt || wget -O b-low-lr.ckpt 'https://drive.usercontent.google.com/download?id=1t2MjHlfbkZOdNotmmQb-XeJyHOMRSKpH&export=download&confirm=t&uuid=c807f72d-572f-423a-903c-d198b3e8aa4e'
cd ..


# setup venv with poetry
test -d "$VENV_PATH" || {
    python3 -m venv "$VENV_PATH"
    "${VENV_PATH}/bin/pip" install -U pip setuptools
    "${VENV_PATH}/bin/pip" install poetry
    source "${VENV_PATH}/bin/activate"
    poetry install
};

# activate environment
pip -V | fgrep "$(pwd)/${VENV_PATH}" > /dev/null || source "${VENV_PATH}/bin/activate"

echo -e "${OK}Successfully set up project${NC}"
cd "$OLD_DIR"