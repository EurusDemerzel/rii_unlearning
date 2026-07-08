#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  Machine Unlearning Environment Setup"
echo "  MacBook Pro (M5 Pro) - MPS Acceleration"
echo "============================================"
echo ""

# --- Step 1: Create virtual environment ---
VENV_DIR="$(cd "$(dirname "$0")" && pwd)/venv"

if [ -d "$VENV_DIR" ]; then
    echo "[INFO] Virtual environment already exists at $VENV_DIR"
    echo "[INFO] To recreate, run: rm -rf $VENV_DIR && bash setup.sh"
else
    echo "[STEP 1/4] Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "[OK] Virtual environment created."
fi

# --- Step 2: Activate and upgrade pip ---
echo "[STEP 2/4] Activating venv and upgrading pip..."
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip --quiet
echo "[OK] pip upgraded."

# --- Step 3: Install dependencies ---
echo "[STEP 3/4] Installing dependencies from requirements.txt..."
pip install -r "$(dirname "$0")/requirements.txt" --quiet
echo "[OK] Dependencies installed."

# --- Step 4: Verify installation ---
echo "[STEP 4/4] Verifying installation..."
python -c "
import sys
import torch
import torchvision
import numpy
import matplotlib
import sklearn
import pandas
import tqdm

print(f'  Python:     {sys.version.split()[0]}')
print(f'  PyTorch:    {torch.__version__}')
print(f'  torchvision:{torchvision.__version__}')
print(f'  NumPy:      {numpy.__version__}')
print(f'  Matplotlib: {matplotlib.__version__}')
print(f'  scikit-learn:{sklearn.__version__}')
print(f'  pandas:     {pandas.__version__}')
print(f'  tqdm:       {tqdm.__version__}')

mps_built  = torch.backends.mps.is_built()
mps_avail  = torch.backends.mps.is_available()
device_str = 'mps' if mps_avail else ('cpu (MPS not available on this hardware)' if not mps_built else 'cpu (MPS built but not available — check macOS version >= 12.3)')
print(f'  Device:     {device_str}')
"
echo "[OK] All dependencies verified."
echo ""

echo "============================================"
echo "  Setup complete!"
echo "  To activate the environment, run:"
echo "    source venv/bin/activate"
echo "  To run the experiment:"
echo "    python machine_unlearning.py"
echo "============================================"
