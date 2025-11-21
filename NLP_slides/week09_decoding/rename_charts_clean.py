"""
Rename Charts to Clean, Descriptive Names
==========================================
Removes: "generate_", "_bsc" suffixes
Keeps: Descriptive content-based names

Old: generate_vocabulary_probability_bsc.py -> vocabulary_probability_bsc.pdf
New: vocabulary_probability.py -> vocabulary_probability.pdf

Also:
1. Adds URL metadata to top of each script
2. Renames folders in charts_individual/
3. Creates mapping file for .tex updates
"""

import re
from pathlib import Path
import shutil

def create_clean_name(old_name):
    """Remove generate_ prefix and _bsc suffix."""
    name = old_name.replace('generate_', '')
    name = name.replace('_bsc', '')
    # Keep graphviz suffix for clarity
    return name


# Scan standalone generators
generators_dir = Path("python/standalone_generators")
scripts = sorted(generators_dir.glob("generate_*.py"))

print("=" * 70)
print(f"Creating Clean Names for {len(scripts)} Scripts")
print("=" * 70)

# Build rename mapping
rename_map = {}
for script in scripts:
    old_name = script.stem
    new_name = create_clean_name(old_name)
    old_pdf = old_name.replace('generate_', '') + '.pdf'
    new_pdf = new_name + '.pdf'

    rename_map[old_name] = {
        'new_script': new_name,
        'old_pdf': old_pdf,
        'new_pdf': new_pdf,
        'quantlet_url': f"https://quantlet.com/NLPDecoding_{new_name.replace('_', '').title()}"
    }

# Show preview
print("\nRename Preview (first 10):")
print("-" * 70)
for i, (old, info) in enumerate(list(rename_map.items())[:10], 1):
    print(f"{i}. {old}.py")
    print(f"   -> {info['new_script']}.py")
    print(f"   PDF: {info['old_pdf']} -> {info['new_pdf']}")
    print(f"   URL: {info['quantlet_url']}")
    print()

# Save complete mapping
mapping_file = Path("RENAME_MAPPING.txt")
with open(mapping_file, 'w', encoding='utf-8') as f:
    f.write("# Chart Rename Mapping\n")
    f.write("# Format: old_script.py | new_script.py | old_pdf | new_pdf | url\n\n")
    for old, info in rename_map.items():
        f.write(f"{old}.py | {info['new_script']}.py | {info['old_pdf']} | {info['new_pdf']} | {info['quantlet_url']}\n")

print(f"Saved complete mapping: {mapping_file}")
print(f"\nTotal: {len(rename_map)} files to rename")
print("\nNext: Review mapping, then run rename operation")
