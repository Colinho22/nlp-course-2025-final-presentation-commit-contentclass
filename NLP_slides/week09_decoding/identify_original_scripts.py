"""
Identify Which Python Script Created Each Figure in figures/
============================================================
Compares file sizes and creation dates to find the source script.
"""

from pathlib import Path
import subprocess
import os
from datetime import datetime

figures_dir = Path("figures")
python_dir = Path("python")

# Get all figures with metadata
figures_info = {}
for pdf in figures_dir.glob("*.pdf"):
    stat = pdf.stat()
    figures_info[pdf.name] = {
        'size': stat.st_size,
        'modified': datetime.fromtimestamp(stat.st_mtime),
        'source_script': 'UNKNOWN'
    }

print("=" * 70)
print(f"Analyzing {len(figures_info)} Figures in figures/")
print("=" * 70)

# Check which scripts generate which figures by searching for savefig/render calls
all_scripts = list(python_dir.glob("*.py"))

for script in all_scripts:
    try:
        with open(script, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find all savefig and render calls
        import re
        # Pattern for savefig
        savefig_pattern = r"savefig\s*\(\s*['\"]\.\.\/figures\/([^'\"]+\.pdf)"
        # Pattern for graphviz render
        render_pattern = r"render\s*\(\s*['\"]\.\.\/figures\/([^'\"]+)['\"]"

        matches = re.findall(savefig_pattern, content)
        for match in matches:
            if match in figures_info:
                if figures_info[match]['source_script'] == 'UNKNOWN':
                    figures_info[match]['source_script'] = script.name
                else:
                    # Multiple scripts - note both
                    figures_info[match]['source_script'] += f" OR {script.name}"

        matches = re.findall(render_pattern, content)
        for match in matches:
            pdf_name = match if match.endswith('.pdf') else match + '.pdf'
            if pdf_name in figures_info:
                if figures_info[pdf_name]['source_script'] == 'UNKNOWN':
                    figures_info[pdf_name]['source_script'] = script.name
                else:
                    figures_info[pdf_name]['source_script'] += f" OR {script.name}"

    except Exception as e:
        pass

# Read FIGURES_IN_TEX.txt to know which are actually used
used_figures = set()
tex_list = Path("FIGURES_IN_TEX.txt")
if tex_list.exists():
    with open(tex_list, 'r') as f:
        used_figures = {line.strip() for line in f if line.strip()}

# Print report
print("\n## Figures in ../figures/ Directory\n")
print("| Figure | Used in .tex | Source Script | Size |")
print("|--------|--------------|---------------|------|")

for fig in sorted(figures_info.keys()):
    info = figures_info[fig]
    used = "YES" if fig in used_figures else "no"
    size_kb = info['size'] // 1024
    script = info['source_script']

    # Highlight if unknown
    if script == 'UNKNOWN':
        script = "**UNKNOWN**"

    print(f"| `{fig}` | {used} | `{script}` | {size_kb}KB |")

# Summary
unknown_count = sum(1 for info in figures_info.values() if info['source_script'] == 'UNKNOWN')
known_count = len(figures_info) - unknown_count

print(f"\n**Summary**: {known_count} figures have identified scripts, {unknown_count} unknown")

# Save report
output_file = Path("FIGURE_SOURCES.md")
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("# Week 9 Figure Sources\n\n")
    f.write("## Figures with Identified Source Scripts\n\n")

    for fig in sorted(figures_info.keys()):
        info = figures_info[fig]
        if info['source_script'] != 'UNKNOWN':
            f.write(f"- `{fig}` <- `{info['source_script']}`\n")

    f.write(f"\n## Unknown Sources ({unknown_count} figures)\n\n")
    for fig in sorted(figures_info.keys()):
        info = figures_info[fig]
        if info['source_script'] == 'UNKNOWN':
            used = " (USED IN TEX)" if fig in used_figures else ""
            f.write(f"- `{fig}`{used}\n")

print(f"\nReport saved to: {output_file}")
