"""
NLP Course 2025 - Installation Verification Script

This script quickly verifies that all required packages are installed
and working correctly. Run this after installing dependencies.

Usage:
    python verify_installation.py

Expected time: 30 seconds
"""

import sys
import importlib

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    print(f"\n{'='*70}")
    print("Python Version Check")
    print(f"{'='*70}")

    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 8:
        print("✓ Python version is compatible (3.8+)")
        return True
    else:
        print("✗ Python version is incompatible (need 3.8+)")
        return False

def check_package(package_name, import_name=None):
    """Check if a package can be imported"""
    if import_name is None:
        import_name = package_name

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name:20s} (version: {version})")
        return True
    except ImportError as e:
        print(f"✗ {package_name:20s} - NOT INSTALLED")
        print(f"  Error: {str(e)[:60]}")
        return False
    except Exception as e:
        print(f"⚠ {package_name:20s} - WARNING: {str(e)[:60]}")
        return True

def check_torch_details():
    """Check PyTorch installation details"""
    print(f"\n{'='*70}")
    print("PyTorch Details")
    print(f"{'='*70}")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print("\n✓ GPU acceleration available!")
        else:
            print("\nℹ GPU not available (CPU-only mode)")
            print("  This is fine for Weeks 1-4.")
            print("  For Weeks 5+, consider using Google Colab for GPU access.")

        return True
    except Exception as e:
        print(f"✗ Error checking PyTorch: {str(e)}")
        return False

def check_core_packages():
    """Check all core packages"""
    print(f"\n{'='*70}")
    print("Core Packages")
    print(f"{'='*70}")

    packages = [
        ('numpy', None),
        ('scipy', None),
        ('pandas', None),
        ('matplotlib', None),
        ('seaborn', None),
        ('sklearn', 'scikit-learn'),
    ]

    results = []
    for import_name, display_name in packages:
        if display_name is None:
            display_name = import_name
        results.append(check_package(display_name, import_name))

    return all(results)

def check_deep_learning_packages():
    """Check deep learning packages"""
    print(f"\n{'='*70}")
    print("Deep Learning Packages")
    print(f"{'='*70}")

    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', None),
        ('torchaudio', None),
        ('transformers', None),
        ('tokenizers', None),
        ('datasets', None),
    ]

    results = []
    for import_name, display_name in packages:
        if display_name is None:
            display_name = import_name
        results.append(check_package(display_name, import_name))

    return all(results)

def check_nlp_packages():
    """Check NLP-specific packages"""
    print(f"\n{'='*70}")
    print("NLP Packages")
    print(f"{'='*70}")

    packages = [
        ('nltk', None),
        ('spacy', None),
        ('gensim', None),
        ('sentencepiece', None),
    ]

    results = []
    for import_name, display_name in packages:
        if display_name is None:
            display_name = import_name
        results.append(check_package(display_name, import_name))

    return all(results)

def check_jupyter():
    """Check Jupyter installation"""
    print(f"\n{'='*70}")
    print("Jupyter Environment")
    print(f"{'='*70}")

    packages = [
        ('jupyter', None),
        ('notebook', None),
        ('jupyterlab', 'JupyterLab'),
        ('ipywidgets', None),
    ]

    results = []
    for import_name, display_name in packages:
        if display_name is None:
            display_name = import_name
        results.append(check_package(display_name, import_name))

    return all(results)

def check_utilities():
    """Check utility packages"""
    print(f"\n{'='*70}")
    print("Utility Packages")
    print(f"{'='*70}")

    packages = [
        ('tqdm', None),
        ('requests', None),
        ('plotly', None),
    ]

    results = []
    for import_name, display_name in packages:
        if display_name is None:
            display_name = import_name
        results.append(check_package(display_name, import_name))

    return all(results)

def print_summary(results):
    """Print summary of all checks"""
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}\n")

    categories = [
        "Python Version",
        "Core Packages",
        "Deep Learning",
        "NLP Packages",
        "Jupyter",
        "Utilities",
        "PyTorch Details"
    ]

    for category, result in zip(categories, results):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{category:20s}: {status}")

    print(f"\n{'='*70}")

    if all(results):
        print("\n🎉 SUCCESS! All packages installed correctly.")
        print("\nYou're ready to start the course!")
        print("Next step: jupyter lab")
        return 0
    else:
        print("\n⚠ ISSUES FOUND")
        print("\nSome packages are missing or incompatible.")
        print("Please check the errors above and reinstall:")
        print("  pip install -r requirements.txt")
        print("\nFor help, see INSTALLATION.md")
        return 1

def main():
    """Run all verification checks"""
    print("\nNLP Course 2025 - Installation Verification")
    print("Checking your environment...")

    results = [
        check_python_version(),
        check_core_packages(),
        check_deep_learning_packages(),
        check_nlp_packages(),
        check_jupyter(),
        check_utilities(),
        check_torch_details()
    ]

    return print_summary(results)

if __name__ == "__main__":
    sys.exit(main())