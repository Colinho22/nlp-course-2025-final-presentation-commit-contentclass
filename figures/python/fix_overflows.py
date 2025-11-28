"""
Fix all overflow issues in Beamer LaTeX presentation.
Reduces figure widths to prevent slides from overflowing.
"""

import re
import os
from datetime import datetime

TEX_FILE = r"D:\Joerg\Research\slides\NLPFinalLecture\presentations\20251127_1000_final_lecture.tex"
OUTPUT_DIR = os.path.dirname(TEX_FILE)
PREVIOUS_DIR = os.path.join(OUTPUT_DIR, "previous")

# Target widths based on current width
WIDTH_MAPPING = {
    0.95: 0.70,  # Very large -> standard
    0.9: 0.70,   # Large -> standard
    0.85: 0.70,  # Medium-large -> standard
    0.8: 0.65,   # Medium -> smaller
    0.75: 0.60,  # Already reduced -> even smaller
}

def fix_overflows(tex_path):
    """Fix all figure width overflows."""

    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes = []

    # Find all figure includes with width
    pattern = r'\\includegraphics\[width=([0-9.]+)\\textwidth\]'

    def replace_width(match):
        old_width = float(match.group(1))

        # Determine new width
        if old_width >= 0.95:
            new_width = 0.70
        elif old_width >= 0.9:
            new_width = 0.70
        elif old_width >= 0.85:
            new_width = 0.68
        elif old_width >= 0.8:
            new_width = 0.65
        elif old_width >= 0.75:
            new_width = 0.62
        elif old_width > 0.7:
            new_width = 0.65
        else:
            return match.group(0)  # No change needed

        changes.append(f"{old_width} -> {new_width}")
        return f'\\includegraphics[width={new_width}\\textwidth]'

    new_content = re.sub(pattern, replace_width, content)

    return new_content, changes, original_content


def main():
    print("=" * 70)
    print("FIXING ALL OVERFLOW FIGURES")
    print("=" * 70)

    # Ensure previous directory exists
    if not os.path.exists(PREVIOUS_DIR):
        os.makedirs(PREVIOUS_DIR)
        print(f"Created directory: {PREVIOUS_DIR}")

    # Read and fix
    new_content, changes, original = fix_overflows(TEX_FILE)

    if not changes:
        print("\nNo figures need width reduction.")
        return

    print(f"\nApplying {len(changes)} width reductions:\n")
    for i, change in enumerate(changes, 1):
        print(f"  {i:2d}. {change}")

    # Backup original
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    backup_name = f"{timestamp}_final_lecture_backup.tex"
    backup_path = os.path.join(PREVIOUS_DIR, backup_name)

    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original)
    print(f"\nBackup saved to: {backup_path}")

    # Write new version with timestamp
    new_filename = f"{timestamp}_final_lecture.tex"
    new_path = os.path.join(OUTPUT_DIR, new_filename)

    with open(new_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"New file written: {new_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total figures modified: {len(changes)}")
    print(f"  New file: {new_filename}")
    print(f"  Backup: {backup_name}")
    print("\nNext step: Compile the new .tex file to verify fixes.")


if __name__ == "__main__":
    main()
