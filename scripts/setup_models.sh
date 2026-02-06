#!/usr/bin/env bash
set -euo pipefail

echo "=============================="
echo " VoiceRad Model Setup"
echo "=============================="

MODEL_DIR="${1:-./models}"
mkdir -p "$MODEL_DIR"

# ── MedGemma 1.5 4B (multimodal) ────────────────────────
echo ""
echo "[1/3] Downloading MedGemma 1.5 4B ..."
if [ ! -d "$MODEL_DIR/medgemma-1.5-4b-it" ]; then
  git lfs install
  git clone https://huggingface.co/google/medgemma-1.5-4b-it "$MODEL_DIR/medgemma-1.5-4b-it"
else
  echo "  -> already present, skipping"
fi

# ── MedASR ───────────────────────────────────────────────
echo ""
echo "[2/3] Downloading MedASR ..."
if [ ! -d "$MODEL_DIR/medASR" ]; then
  git clone https://huggingface.co/google/medASR "$MODEL_DIR/medASR"
else
  echo "  -> already present, skipping"
fi

# ── MedSigLIP (optional) ────────────────────────────────
echo ""
echo "[3/3] Downloading MedSigLIP (optional) ..."
if [ ! -d "$MODEL_DIR/med-sigclip" ]; then
  git clone https://huggingface.co/google/med-sigclip "$MODEL_DIR/med-sigclip"
else
  echo "  -> already present, skipping"
fi

echo ""
echo "=============================="
echo " All models downloaded!"
echo ""
echo " MedGemma : $MODEL_DIR/medgemma-1.5-4b-it"
echo " MedASR   : $MODEL_DIR/medASR"
echo " MedSigLIP: $MODEL_DIR/med-sigclip"
echo "=============================="
echo ""
echo "Next:  cd backend && python app.py"
