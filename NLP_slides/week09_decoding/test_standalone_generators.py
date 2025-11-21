"""
Test All Standalone Chart Generators
=====================================
Runs each script in standalone_generators/ and verifies PDF generation.
"""

import subprocess
import sys
from pathlib import Path

generators_dir = Path("python/standalone_generators")

if not generators_dir.exists():
    print(f"ERROR: {generators_dir} not found")
    sys.exit(1)

scripts = sorted(generators_dir.glob('generate_*.py'))

print("=" * 70)
print(f"Testing {len(scripts)} Standalone Chart Generators")
print("=" * 70)

passed = 0
failed = 0
errors = []

for script in scripts:
    script_name = script.name
    expected_pdf = script.stem.replace('generate_', '') + '.pdf'

    # Clean any existing PDF
    pdf_path = generators_dir / expected_pdf
    if pdf_path.exists():
        pdf_path.unlink()

    # Run script from its directory
    try:
        result = subprocess.run(
            [sys.executable, script.name],
            cwd=generators_dir,
            capture_output=True,
            text=True,
            timeout=30
        )

        # Check if PDF was created
        if pdf_path.exists():
            print(f"  [PASS] {script_name} -> {expected_pdf}")
            passed += 1
        else:
            print(f"  [FAIL] {script_name} - No PDF generated")
            if result.stderr:
                print(f"         Error: {result.stderr[:100]}")
            failed += 1
            errors.append((script_name, "No PDF generated"))

    except subprocess.TimeoutExpired:
        print(f"  [FAIL] {script_name} - Timeout")
        failed += 1
        errors.append((script_name, "Timeout"))
    except Exception as e:
        print(f"  [FAIL] {script_name} - {str(e)[:50]}")
        failed += 1
        errors.append((script_name, str(e)))

print("\n" + "=" * 70)
print(f"Results: {passed} passed, {failed} failed")
print("=" * 70)

if errors:
    print("\nErrors:")
    for script, error in errors:
        print(f"  - {script}: {error}")

if passed == len(scripts):
    print("\nSUCCESS: All chart generators work!")
else:
    print(f"\nWARNING: {failed} scripts need fixing")

sys.exit(0 if failed == 0 else 1)
