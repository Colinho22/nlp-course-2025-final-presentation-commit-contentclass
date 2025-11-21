"""
Generate All Figures Used in NLP_Decoding_Strategies.tex
=========================================================
Runs only the Python scripts needed to generate the 32 figures used in the presentation.
"""

import subprocess
import sys
from pathlib import Path

# Read the figures list
figures_file = Path("FIGURES_IN_TEX.txt")
if not figures_file.exists():
    print("ERROR: Run extract_tex_figures.py first")
    sys.exit(1)

with open(figures_file, 'r') as f:
    figures_needed = [line.strip() for line in f if line.strip()]

# Map figure to script name (remove .pdf, add .py)
figure_to_script = {}
for fig in figures_needed:
    # Convert figure name to script name
    base_name = fig.replace('.pdf', '')
    script_name = f"generate_{base_name}.py"
    figure_to_script[fig] = script_name

generators_dir = Path("python/standalone_generators")

print("=" * 70)
print(f"Generating {len(figures_needed)} Figures for Week 9 Presentation")
print("=" * 70)

success = 0
failed = 0
errors = []

for fig in figures_needed:
    script_name = figure_to_script[fig]
    script_path = generators_dir / script_name

    if not script_path.exists():
        print(f"  [SKIP] {fig} - Script not found: {script_name}")
        failed += 1
        errors.append((fig, "Script not found"))
        continue

    # Run the script
    try:
        result = subprocess.run(
            [sys.executable, script_path.name],
            cwd=generators_dir,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Check if PDF was created
        pdf_path = generators_dir / fig
        if pdf_path.exists():
            print(f"  [OK] {fig}")
            success += 1
        else:
            print(f"  [FAIL] {fig}")
            if result.stderr:
                print(f"        {result.stderr[:100]}")
            failed += 1
            errors.append((fig, "No PDF generated"))

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {fig} (>60s)")
        # Check if PDF exists anyway
        pdf_path = generators_dir / fig
        if pdf_path.exists():
            success += 1
        else:
            failed += 1
            errors.append((fig, "Timeout"))
    except Exception as e:
        print(f"  [ERROR] {fig}: {str(e)[:50]}")
        failed += 1
        errors.append((fig, str(e)))

print("\n" + "=" * 70)
print(f"Results: {success}/{len(figures_needed)} figures generated successfully")
print("=" * 70)

if errors:
    print(f"\nIssues ({len(errors)}):")
    for fig, error in errors:
        print(f"  - {fig}: {error}")

if success == len(figures_needed):
    print("\nSUCCESS: All presentation figures generated!")
    print(f"Location: {generators_dir.absolute()}")
else:
    print(f"\nPartial success: {failed} figures need attention")

sys.exit(0 if failed == 0 else 1)
