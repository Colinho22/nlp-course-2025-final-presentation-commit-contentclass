"""
Update .tex File to Import from Individual Chart Folders
=========================================================
Changes paths from:
  \includegraphics[...]{../figures/chart_name.pdf}
To:
  \includegraphics[...]{../python/charts_individual/chart_name/chart_name.pdf}
"""

import re
from pathlib import Path

tex_file = Path("presentations/NLP_Decoding_Strategies.tex")

# Read original
with open(tex_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern to match \includegraphics with ../figures/
pattern = r'(\\includegraphics\[.*?\])\{\.\.\/figures\/(.*?\.pdf)\}'

def replace_path(match):
    """Replace ../figures/chart.pdf with ../python/charts_individual/chart_name/chart.pdf"""
    prefix = match.group(1)  # \includegraphics[...]
    filename = match.group(2)  # chart_name.pdf

    # Get folder name (remove .pdf)
    folder_name = filename.replace('.pdf', '')

    # New path
    new_path = f"../python/charts_individual/{folder_name}/{filename}"

    return f"{prefix}{{{new_path}}}"

# Replace all paths
updated_content = re.sub(pattern, replace_path, content)

# Count replacements
original_count = len(re.findall(pattern, content))
new_pattern = r'\\includegraphics\[.*?\]\{\.\.\/python\/charts_individual\/'
new_count = len(re.findall(new_pattern, updated_content))

print("=" * 70)
print("Update .tex Figure Paths")
print("=" * 70)
print(f"Original paths (../figures/): {original_count}")
print(f"Updated paths (../python/charts_individual/): {new_count}")

# Save backup
backup_file = Path("presentations/NLP_Decoding_Strategies_BACKUP.tex")
with open(backup_file, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"\nBackup saved: {backup_file}")

# Write updated file
with open(tex_file, 'w', encoding='utf-8') as f:
    f.write(updated_content)

print(f"Updated: {tex_file}")
print("\nExample changes:")
print("  OLD: ../figures/vocabulary_probability_bsc.pdf")
print("  NEW: ../python/charts_individual/vocabulary_probability_bsc/vocabulary_probability_bsc.pdf")
print("=" * 70)
