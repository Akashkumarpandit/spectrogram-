#!/usr/bin/env bash
# Quick script to run training on combined ASVspoof 2019 + 2021
set -e

CONFIG=${1:-"configs/config.yaml"}
EPOCHS=${2:-50}

echo "=========================================================="
echo "🚀 Training AASIST on Combined ASVspoof 2019 & 2021"
echo "Config : ${CONFIG}"
echo "Epochs : ${EPOCHS}"
echo "=========================================================="

python3 -m src.train --config ${CONFIG} --epochs ${EPOCHS}
