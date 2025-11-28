"""
Fix remaining vbox overflow issues by further reducing figure widths
and adjusting content on specific slides.
"""

import re
import os
from datetime import datetime

TEX_FILE = r"D:\Joerg\Research\slides\NLPFinalLecture\presentations\20251127_1321_final_lecture.tex"
OUTPUT_DIR = os.path.dirname(TEX_FILE)
PREVIOUS_DIR = os.path.join(OUTPUT_DIR, "previous")

# Specific fixes for overflow slides
FIGURE_FIXES = {
    # (filename pattern, old_width, new_width)
    'embedding_space_2d.pdf': ('0.7', '0.55'),
    'hybrid_search_flow.pdf': ('0.6', '0.50'),
    'rag_failures_flowchart.pdf': ('0.7', '0.55'),
    'self_consistency_voting.pdf': ('0.7', '0.55'),
    'test_time_scaling.pdf': ('0.7', '0.55'),
    'inference_scaling_curve.pdf': ('0.7', '0.55'),
    'rlhf_detailed_pipeline.pdf': ('0.7', '0.55'),
}


def fix_specific_figures(content):
    """Fix specific figure widths."""
    changes = []

    for filename, (old_w, new_w) in FIGURE_FIXES.items():
        pattern = rf'(\\includegraphics\[width=){old_w}(\\textwidth\]\{{[^}}]*{filename}\}})'
        replacement = rf'\g<1>{new_w}\g<2>'

        new_content, count = re.subn(pattern, replacement, content)
        if count > 0:
            changes.append(f"{filename}: {old_w} -> {new_w}")
            content = new_content

    return content, changes


def fix_resources_slide(content):
    """Fix the Resources for Continued Learning slide by using smaller font."""
    # Find the Resources frame and add \small
    old_pattern = r'(\\begin\{frame\}\[t\]\{Resources for Continued Learning\}\n\\begin\{columns\}\[T\])'
    new_pattern = r'\\begin{frame}[t]{Resources for Continued Learning}\n\\small\n\\begin{columns}[T]'

    new_content, count = re.subn(old_pattern, new_pattern, content)

    return new_content, count > 0


def fix_final_message_slide(content):
    """Fix the Final Message slide by reducing vertical spaces."""
    # Reduce vspace in the final message slide
    old_pattern = r'(\\begin\{frame\}\[plain\]\n\\vspace\{)2cm(\})'
    new_pattern = r'\g<1>1cm\g<2>'

    new_content, count = re.subn(old_pattern, new_pattern, content)

    # Also reduce other vertical spaces in that slide
    # [0.5cm] -> [0.3cm] for small spaces
    new_content = re.sub(r'\[0\.5cm\]', '[0.3cm]', new_content)
    # [1.5cm] -> [1cm]
    new_content = re.sub(r'\[1\.5cm\]', '[1cm]', new_content)
    # [2cm] -> [1.2cm]
    new_content = re.sub(r'\[2cm\]', '[1.2cm]', new_content)

    return new_content, count > 0


def main():
    print("=" * 70)
    print("FIXING REMAINING VBOX OVERFLOWS")
    print("=" * 70)

    # Ensure previous directory exists
    if not os.path.exists(PREVIOUS_DIR):
        os.makedirs(PREVIOUS_DIR)

    with open(TEX_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    all_changes = []

    # Fix specific figures
    content, fig_changes = fix_specific_figures(content)
    all_changes.extend(fig_changes)

    # Fix resources slide
    content, resources_fixed = fix_resources_slide(content)
    if resources_fixed:
        all_changes.append("Resources slide: added \\small for smaller font")

    # Fix final message slide
    content, final_fixed = fix_final_message_slide(content)
    if final_fixed:
        all_changes.append("Final message slide: reduced vertical spacing")

    if not all_changes:
        print("\nNo changes needed.")
        return

    print(f"\nApplied {len(all_changes)} fixes:\n")
    for i, change in enumerate(all_changes, 1):
        print(f"  {i}. {change}")

    # Backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    backup_path = os.path.join(PREVIOUS_DIR, f"{timestamp}_backup_before_vbox_fix.tex")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original)
    print(f"\nBackup: {backup_path}")

    # Write new file
    new_filename = f"{timestamp}_final_lecture.tex"
    new_path = os.path.join(OUTPUT_DIR, new_filename)
    with open(new_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"New file: {new_path}")


if __name__ == "__main__":
    main()
