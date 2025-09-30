"""
Fix overfull vbox warnings by adjusting chart sizes in Week 5 presentation
Analyzes compilation log and reduces chart widths proportionally
"""

import re
import os

# Parse log file to find overfull vbox warnings with line numbers
log_file = '../presentations/week05_test.log'
tex_file = '../presentations/20250928_1648_week05_transformers_speed_revolution.tex'

print("Analyzing overfull vbox warnings...")

overfull_lines = []
with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        match = re.search(r'Overfull \\vbox \((\d+\.\d+)pt too high\) detected at line (\d+)', line)
        if match:
            overflow_pt = float(match.group(1))
            line_num = int(match.group(2))
            overfull_lines.append((line_num, overflow_pt))

print(f"Found {len(overfull_lines)} overfull vbox warnings")

# Read LaTeX file
with open(tex_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find chart includes and their current widths
chart_adjustments = []
for line_num, overflow_pt in overfull_lines:
    # Search backwards from overflow line to find nearest \includegraphics
    for i in range(max(0, line_num - 20), min(len(lines), line_num + 5)):
        if 'includegraphics' in lines[i] and 'sr_' in lines[i]:
            match = re.search(r'width=(0\.\d+)\\textwidth', lines[i])
            if match:
                current_width = float(match.group(1))
                chart_name = re.search(r'sr_\d+_[\w_]+\.pdf', lines[i]).group(0)

                # Calculate reduction needed (rough heuristic)
                # Beamer slide height ~7 inches, overflow in points (1pt = 1/72 inch)
                overflow_inches = overflow_pt / 72
                slide_height_inches = 7.0
                overflow_fraction = overflow_inches / slide_height_inches

                # Reduce width to compensate (charts scale proportionally)
                reduction_factor = max(0.5, 1 - (overflow_fraction * 0.5))  # Don't go below 50%
                new_width = round(current_width * reduction_factor, 2)

                chart_adjustments.append({
                    'line_num': i,
                    'chart': chart_name,
                    'old_width': current_width,
                    'new_width': new_width,
                    'overflow_pt': overflow_pt,
                    'line_content': lines[i]
                })
                break

# Remove duplicates (same chart found multiple times)
seen_lines = set()
unique_adjustments = []
for adj in chart_adjustments:
    if adj['line_num'] not in seen_lines:
        seen_lines.add(adj['line_num'])
        unique_adjustments.append(adj)

print(f"\nChart adjustments needed: {len(unique_adjustments)}")
print("\n" + "="*80)

# Apply adjustments
for adj in sorted(unique_adjustments, key=lambda x: x['line_num'], reverse=True):
    print(f"Line {adj['line_num']}: {adj['chart']}")
    print(f"  Overflow: {adj['overflow_pt']:.1f}pt")
    print(f"  Width: {adj['old_width']:.2f} -> {adj['new_width']:.2f}")

    # Replace width in line
    old_line = lines[adj['line_num']]
    new_line = re.sub(
        r'width=0\.\d+\\textwidth',
        f"width={adj['new_width']:.2f}\\\\textwidth",
        old_line
    )
    lines[adj['line_num']] = new_line

# Write updated file
output_file = '../presentations/20250928_1648_week05_transformers_speed_revolution_fixed.tex'
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("="*80)
print(f"\nFixed LaTeX written to: {output_file}")
print("\nSummary of changes:")
print(f"  Total adjustments: {len(unique_adjustments)}")
print(f"  Average width reduction: {sum(adj['old_width'] - adj['new_width'] for adj in unique_adjustments) / len(unique_adjustments):.2f}")
print("\nNext steps:")
print(f"  1. cd ../presentations")
print(f"  2. pdflatex -jobname=week05_fixed 20250928_1648_week05_transformers_speed_revolution_fixed.tex")
print(f"  3. Check week05_fixed.pdf for improved layout")