"""
Regenerate All 32 Figures Used in Presentation
==============================================
Intelligently finds and runs the correct script for each figure.
"""

import subprocess
import sys
from pathlib import Path

# Read figures from tex
figures_file = Path("FIGURES_IN_TEX.txt")
with open(figures_file, 'r') as f:
    figures_needed = [line.strip() for line in f if line.strip()]

generators_dir = Path("python/standalone_generators")

# Build mapping of figure -> script by scanning directory
available_scripts = {}
for script in generators_dir.glob("generate_*.py"):
    # Figure out what PDF this generates
    base = script.stem.replace('generate_', '')
    # Try with and without .pdf
    available_scripts[base + '.pdf'] = script.name
    available_scripts[base] = script.name

print("=" * 70)
print(f"Regenerating {len(figures_needed)} Figures for Presentation")
print("=" * 70)

success = 0
skipped = 0

for fig in figures_needed:
    # Try to find matching script
    script_name = None

    if fig in available_scripts:
        script_name = available_scripts[fig]
    else:
        # Try without _bsc suffix
        fig_no_bsc = fig.replace('_bsc.pdf', '.pdf')
        if fig_no_bsc in available_scripts:
            script_name = available_scripts[fig_no_bsc]

    if not script_name:
        print(f"  [SKIP] {fig} - no script found")
        skipped += 1
        continue

    # Run script
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=generators_dir,
            capture_output=True,
            text=True,
            timeout=90
        )

        # Check result
        pdf_path = generators_dir / fig
        if pdf_path.exists():
            print(f"  [OK] {fig} <- {script_name}")
            success += 1
        else:
            print(f"  [FAIL] {fig}")
            skipped += 1

    except subprocess.TimeoutExpired:
        # Check if PDF exists anyway
        pdf_path = generators_dir / fig
        if pdf_path.exists():
            print(f"  [OK] {fig} <- {script_name} (timeout but PDF exists)")
            success += 1
        else:
            print(f"  [TIMEOUT] {fig}")
            skipped += 1
    except Exception as e:
        print(f"  [ERROR] {fig}: {str(e)[:40]}")
        skipped += 1

print("\n" + "=" * 70)
print(f"Generated: {success}/{len(figures_needed)} figures")
if skipped:
    print(f"Skipped: {skipped} (scripts not found or errors)")
print("=" * 70)

if success == len(figures_needed):
    print("\nSUCCESS: All figures generated!")
