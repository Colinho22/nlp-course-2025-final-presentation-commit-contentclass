"""
Map Charts in .tex to Python Generation Scripts
================================================
Creates a definitive mapping between:
- Figures used in NLP_Decoding_Strategies.tex
- Python scripts/functions that generate them

Output: CHART_MAPPING.md with complete mapping
"""

import re
from pathlib import Path
from collections import defaultdict

# Read the .tex file
tex_file = Path("presentations/NLP_Decoding_Strategies.tex")
with open(tex_file, 'r', encoding='utf-8') as f:
    tex_content = f.read()

# Extract all \includegraphics
pattern = r'\\includegraphics\[.*?\]\{\.\.\/figures\/(.*?)\}'
figures_in_tex = re.findall(pattern, tex_content)

# Remove duplicates but keep order
seen = set()
figures_unique = []
for fig in figures_in_tex:
    if fig not in seen:
        seen.add(fig)
        figures_unique.append(fig)

print(f"Found {len(figures_unique)} unique figures in presentation\n")

# Check which figures exist
figures_dir = Path("figures")
existing_figures = {f.name for f in figures_dir.glob('*.pdf')}

# Now scan all Python scripts to find which generates which
python_dir = Path("python")
figure_to_script = {}

for py_file in python_dir.glob('generate_*.py'):
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find savefig and render calls
    savefig_pattern = r"(?:savefig|render)\s*\(\s*['\"]\.\.\/figures\/([^'\"]+)"
    matches = re.findall(savefig_pattern, content)

    for match in matches:
        # Clean up (remove cleanup=True etc)
        fig_name = match.split("'")[0].strip()
        if not fig_name.endswith('.pdf'):
            fig_name += '.pdf'

        if fig_name in figures_unique:
            if fig_name not in figure_to_script:
                figure_to_script[fig_name] = []
            figure_to_script[fig_name].append(py_file.name)

# Group by script
script_to_figures = defaultdict(list)
for fig, scripts in figure_to_script.items():
    for script in scripts:
        script_to_figures[script].append(fig)

# Create markdown report
report = []
report.append("# Week 9 Chart Mapping: .tex Figures -> Python Scripts\n")
report.append(f"**Total figures in presentation**: {len(figures_unique)}")
report.append(f"**Figures with known scripts**: {len(figure_to_script)}")
report.append(f"**Figures missing scripts**: {len(figures_unique) - len(figure_to_script)}\n")

report.append("## Figures by Python Script\n")
for script in sorted(script_to_figures.keys()):
    figs = script_to_figures[script]
    report.append(f"### `{script}` ({len(figs)} charts)\n")
    for fig in sorted(figs):
        exists = "Y" if fig in existing_figures else "N"
        report.append(f"- [{exists}] `{fig}`")
    report.append("")

# Missing figures
missing = [f for f in figures_unique if f not in figure_to_script]
if missing:
    report.append(f"## Missing Generation Scripts ({len(missing)} figures)\n")
    for fig in missing:
        exists = "Y" if fig in existing_figures else "N"
        report.append(f"- [{exists}] `{fig}`")

# Write report
output_file = Path("CHART_MAPPING.md")
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print('\n'.join(report))
print(f"\n\nSaved to: {output_file.absolute()}")
