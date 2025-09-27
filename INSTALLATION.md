# Installation Guide - NLP Course 2025

Complete setup instructions for the NLP Course 2025. Follow these steps to get your environment ready for all 12 weeks of labs.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Python Installation](#python-installation)
3. [Package Installation](#package-installation)
4. [GPU Setup (Optional)](#gpu-setup-optional)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS:** Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python:** 3.8 or higher
- **RAM:** 8GB (16GB recommended)
- **Storage:** 5GB free space
- **CPU:** Multi-core processor (4+ cores recommended)

### Recommended for GPU Acceleration
- **GPU:** NVIDIA GPU with 6GB+ VRAM (for Weeks 5+)
- **CUDA:** 11.8 or 12.1
- **RAM:** 16GB+
- **Storage:** 10GB+ free space

### Week-Specific Requirements
- **Weeks 1-4:** CPU sufficient, 8GB RAM
- **Weeks 5-7:** GPU recommended, 12GB RAM
- **Weeks 8-12:** GPU optional, depends on model size

## Python Installation

### Option 1: Anaconda (Recommended for Beginners)

1. Download Anaconda from [anaconda.com](https://www.anaconda.com/download)
2. Install following the default options
3. Verify installation:
```bash
conda --version
python --version
```

### Option 2: Python.org (Lightweight)

1. Download Python 3.10 from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Verify installation:
```bash
python --version
pip --version
```

## Package Installation

### Method 1: pip (Quick Start)

**Step 1: Clone the repository**
```bash
git clone https://github.com/josterri/2025_NLP_Lectures.git
cd 2025_NLP_Lectures
```

**Step 2: Create virtual environment (recommended)**
```bash
# Windows
python -m venv nlp2025
nlp2025\Scripts\activate

# macOS/Linux
python3 -m venv nlp2025
source nlp2025/bin/activate
```

**Step 3: Install packages**
```bash
# CPU-only installation (works for Weeks 1-4)
pip install -r requirements.txt

# For GPU support, see GPU Setup section below
```

**Expected installation time:** 10-15 minutes

### Method 2: conda (Isolated Environment)

**Step 1: Clone repository**
```bash
git clone https://github.com/josterri/2025_NLP_Lectures.git
cd 2025_NLP_Lectures
```

**Step 2: Create environment**
```bash
# CPU-only
conda env create -f environment.yml

# Activate environment
conda activate nlp2025
```

**Expected installation time:** 15-20 minutes

## GPU Setup (Optional)

GPU acceleration significantly speeds up training in Weeks 5-12. Skip this section if you don't have an NVIDIA GPU.

### Check GPU Availability

**Windows:**
```bash
nvidia-smi
```

**Linux:**
```bash
lspci | grep -i nvidia
```

### Install CUDA-enabled PyTorch

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU-only (fallback):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Verify GPU Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Expected output with GPU:**
```
PyTorch version: 2.1.0
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 3080
```

## Verification

### Verify Installation

**1. Check Python packages:**
```bash
python -c "import torch, transformers, numpy, matplotlib; print('All packages imported successfully!')"
```

**2. Run verification script:**
```bash
python verify_installation.py
```

**3. Test a simple notebook:**
```bash
jupyter lab
# Open: NLP_slides/week02_neural_lm/lab/week02_word_embeddings_lab.ipynb
# Run the first few cells
```

### Expected Package Versions
```
torch>=2.0.0
transformers>=4.30.0
numpy>=1.21.0
matplotlib>=3.5.0
jupyter>=1.0.0
```

## Troubleshooting

### Common Issues

#### Issue 1: "No module named 'torch'"

**Solution:**
```bash
pip install torch
```

#### Issue 2: CUDA out of memory

**Solutions:**
1. Reduce batch size in notebooks
2. Use CPU instead: Add to notebook
```python
device = 'cpu'
```
3. Restart Jupyter kernel to free memory

#### Issue 3: Jupyter not found

**Solution:**
```bash
pip install jupyter jupyterlab
# Then restart terminal
```

#### Issue 4: SSL Certificate errors during installation

**Solution:**
```bash
# Windows
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Or update certificates
pip install --upgrade certifi
```

#### Issue 5: Permission denied (macOS/Linux)

**Solution:**
```bash
# Use --user flag
pip install --user -r requirements.txt

# Or use sudo (not recommended)
sudo pip install -r requirements.txt
```

#### Issue 6: Slow installation

**Solutions:**
1. Use a faster mirror:
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. Install without dependencies first:
```bash
pip install --no-deps -r requirements.txt
```

### Platform-Specific Issues

#### Windows

**Issue:** Long path names cause errors

**Solution:** Enable long paths:
```powershell
# Run PowerShell as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

#### macOS

**Issue:** SSL errors with Python 3.x

**Solution:**
```bash
# Install certificates
/Applications/Python\ 3.x/Install\ Certificates.command
```

#### Linux

**Issue:** Missing system dependencies

**Solution (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip build-essential
```

## Advanced Configuration

### Using Different Python Versions

```bash
# Install specific Python version with conda
conda create -n nlp2025 python=3.10
conda activate nlp2025
pip install -r requirements.txt
```

### Minimal Installation (Core Packages Only)

For Weeks 1-4 only:
```bash
pip install torch numpy matplotlib jupyter notebook
```

For Weeks 5-12 add:
```bash
pip install transformers datasets tokenizers
```

### Development Installation

For contributors and advanced users:
```bash
pip install -r requirements.txt
pip install pytest black flake8 mypy  # Development tools
```

## Testing Your Setup

### Quick Test
```bash
python test_notebooks.py
```

This tests all 12 lab notebooks and generates a report.

### Manual Test
```python
# Save as test_setup.py
import sys

def test_imports():
    try:
        import torch
        import transformers
        import numpy
        import matplotlib
        import jupyter
        print("✓ All core packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_gpu():
    import torch
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ GPU not available (CPU-only mode)")

if __name__ == "__main__":
    success = test_imports()
    test_gpu()
    sys.exit(0 if success else 1)
```

Run with:
```bash
python test_setup.py
```

## Next Steps

Once installation is complete:

1. **Start with Week 2:** Open the word embeddings lab
```bash
jupyter lab NLP_slides/week02_neural_lm/lab/week02_word_embeddings_lab.ipynb
```

2. **Try the Neural Network Primer:** If you're new to neural networks
```bash
cd NLP_slides/nn_primer/presentations
# Review the handouts or presentations
```

3. **Check the course index:** For week-by-week navigation
```bash
cat COURSE_INDEX.md
```

## Getting Help

If you encounter issues not covered here:

1. Check [GitHub Issues](https://github.com/josterri/2025_NLP_Lectures/issues)
2. Review the [PyTorch installation guide](https://pytorch.org/get-started/locally/)
3. Consult the [Transformers installation docs](https://huggingface.co/docs/transformers/installation)

## Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Repository cloned
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] GPU setup (if applicable)
- [ ] Installation verified (imports work)
- [ ] Jupyter Lab launches successfully
- [ ] First notebook opens and runs

**Ready to go?** Head back to [README.md](README.md) to start the course!