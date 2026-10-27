#!/usr/bin/env bash
# Quick startup script for AASIST Deepfake Detection Web Application
set -e

PORT=${1:-8080}
echo "=========================================================="
echo "🎙️ Starting SpectraGuard Deepfake Audio Detection Web UI"
echo "🌐 URL: http://localhost:${PORT}"
echo "=========================================================="

python3 -m web.app --port ${PORT}
