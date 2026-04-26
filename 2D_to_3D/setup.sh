#!/bin/bash
# ============================================================
# Sketch-to-3D Project Setup Script
# For macOS Apple Silicon (M1/M2/M3/M4) — no NVIDIA GPU needed
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================"
echo " Sketch-to-3D: Project Setup"
echo " Target: macOS Apple Silicon (MPS backend)"
echo "============================================"
echo ""

# ---- 1. Create virtual environment ----
if [ ! -d "venv" ]; then
    echo "[1/5] Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "[1/5] Virtual environment already exists, skipping."
fi

source venv/bin/activate
echo "  → Python: $(python3 --version)"
echo "  → Path:   $(which python3)"
echo ""

# ---- 2. Upgrade pip and build tools ----
echo "[2/5] Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel cmake ninja
echo ""

# ---- 3. Install project dependencies ----
echo "[3/5] Installing project dependencies..."
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip install -r requirements.txt
echo ""

# ---- 4. Clone and install TripoSR ----
echo "[4/5] Setting up TripoSR..."
if [ ! -d "TripoSR" ]; then
    git clone https://github.com/VAST-AI-Research/TripoSR.git
else
    echo "  → TripoSR directory already exists, skipping clone."
fi

cd TripoSR

# Install torchmcubes (marching cubes for mesh extraction)
echo "  → Installing torchmcubes..."
TORCH_CMAKE=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
CMAKE_PREFIX_PATH="$TORCH_CMAKE" CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip install git+https://github.com/tatsy/torchmcubes.git

# Install TripoSR's own requirements (but we already have most)
# We skip re-installing to avoid version conflicts
echo "  → TripoSR setup complete."
cd "$PROJECT_DIR"
echo ""

# ---- 5. Verify installation ----
echo "[5/5] Verifying installation..."
python3 -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  MPS:      {torch.backends.mps.is_available()}')
print(f'  Device:   {\"mps\" if torch.backends.mps.is_available() else \"cpu\"}')

import trimesh
print(f'  Trimesh:  {trimesh.__version__}')

import transformers
print(f'  Transformers: {transformers.__version__}')

import lpips
print(f'  LPIPS:    OK')

import open_clip
print(f'  OpenCLIP: OK')

print()
print('  ✅ All dependencies installed successfully!')
"

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " Activate the environment with:"
echo "   source venv/bin/activate"
echo ""
echo " Then run:"
echo "   python run_single.py <image_path>"
echo "============================================"
