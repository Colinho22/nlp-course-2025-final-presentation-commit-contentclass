"""
Extract all figure filenames from NLP_Decoding_Strategies.tex
"""

import re
from pathlib import Path

tex_file = Path("presentations/NLP_Decoding_Strategies.tex")

with open(tex_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Extract all \includegraphics commands
pattern = r'\\includegraphics\[.*?\]\{\.\.\/figures\/(.*?)\}'
figures = re.findall(pattern, content)

# Remove duplicates while preserving order
seen = set()
unique_figures = []
for fig in figures:
    if fig not in seen:
        seen.add(fig)
        unique_figures.append(fig)

print(f"Found {len(unique_figures)} unique figures in presentation:\n")
for i, fig in enumerate(unique_figures, 1):
    print(f"{i:2d}. {fig}")

# Save to file
output_file = Path("FIGURES_IN_TEX.txt")
with open(output_file, 'w', encoding='utf-8') as f:
    for fig in unique_figures:
        f.write(fig + '\n')

print(f"\nSaved to: {output_file.absolute()}")
print(f"\nTotal: {len(unique_figures)} figures needed")
