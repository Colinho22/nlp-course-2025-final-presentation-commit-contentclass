"""
Apply Clean Rename to All Charts
==================================
1. Renames Python scripts (removes generate_, _bsc)
2. Adds QUANTLET_URL metadata to top of each script
3. Renames folders in charts_individual/
4. Updates .tex file with new names
5. Copies to figures/ with new names
"""

import re
from pathlib import Path
import shutil

# Read mapping
mapping_file = Path("RENAME_MAPPING.txt")
if not mapping_file.exists():
    print("ERROR: Run rename_charts_clean.py first")
    exit(1)

rename_map = {}
with open(mapping_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('#') or '|' not in line:
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 5:
            old_script = parts[0].replace('.py', '')
            new_script = parts[1].replace('.py', '')
            old_pdf = parts[2]
            new_pdf = parts[3]
            url = parts[4]
            rename_map[old_script] = {
                'new_script': new_script,
                'old_pdf': old_pdf,
                'new_pdf': new_pdf,
                'url': url
            }

print("=" * 70)
print(f"Applying Clean Rename to {len(rename_map)} Charts")
print("=" * 70)

generators_dir = Path("python/standalone_generators")
charts_dir = Path("python/charts_individual")
figures_dir = Path("figures")

# Step 1: Rename scripts and add URL metadata
print("\nStep 1: Renaming scripts and adding URL metadata...")
renamed_scripts = 0

for old_script, info in rename_map.items():
    old_path = generators_dir / f"{old_script}.py"
    new_path = generators_dir / f"{info['new_script']}.py"

    if not old_path.exists():
        continue

    # Read script
    with open(old_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Add URL metadata at top (after docstring)
    docstring_end = content.find('"""', 10)  # Find end of docstring
    if docstring_end > 0:
        insert_pos = docstring_end + 3
        url_metadata = f"\n\n# QuantLet Metadata\nQUANTLET_URL = \"{info['url']}\"\n"
        content = content[:insert_pos] + url_metadata + content[insert_pos:]

    # Write to new name
    with open(new_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Remove old
    old_path.unlink()
    print(f"  {old_script}.py -> {info['new_script']}.py")
    renamed_scripts += 1

print(f"Renamed: {renamed_scripts} scripts")

# Step 2: Rename folders
print("\nStep 2: Renaming folders in charts_individual/...")
renamed_folders = 0

for old_script, info in rename_map.items():
    old_folder_name = old_script.replace('generate_', '')
    new_folder_name = info['new_script']

    old_folder = charts_dir / old_folder_name
    new_folder = charts_dir / new_folder_name

    if old_folder.exists() and old_folder != new_folder:
        shutil.move(str(old_folder), str(new_folder))
        print(f"  {old_folder_name}/ -> {new_folder_name}/")
        renamed_folders += 1

print(f"Renamed: {renamed_folders} folders")

# Step 3: Update .tex file
print("\nStep 3: Updating .tex file...")
tex_file = Path("presentations/NLP_Decoding_Strategies.tex")

with open(tex_file, 'r', encoding='utf-8') as f:
    tex_content = f.read()

updates = 0
for old_script, info in rename_map.items():
    old_pdf = info['old_pdf']
    new_pdf = info['new_pdf']

    if old_pdf != new_pdf:
        # Update references
        tex_content = tex_content.replace(f"/{old_pdf}", f"/{new_pdf}")
        if f"/{old_pdf}" in tex_content:
            updates += 1

# Save updated .tex
with open(tex_file, 'w', encoding='utf-8') as f:
    f.write(tex_content)

print(f"Updated: {updates} figure references in .tex")

# Step 4: Copy renamed PDFs to figures/
print("\nStep 4: Copying renamed PDFs to figures/...")
copied = 0

for old_script, info in rename_map.items():
    old_pdf_path = figures_dir / info['old_pdf']
    new_pdf_path = figures_dir / info['new_pdf']

    if old_pdf_path.exists() and old_pdf_path != new_pdf_path:
        shutil.copy2(old_pdf_path, new_pdf_path)
        print(f"  {info['old_pdf']} -> {info['new_pdf']}")
        copied += 1

print(f"Copied: {copied} PDFs with new names")

print("\n" + "=" * 70)
print("COMPLETE: Clean rename applied")
print("=" * 70)
print(f"Scripts: {renamed_scripts} renamed")
print(f"Folders: {renamed_folders} renamed")
print(f".tex: {updates} references updated")
print(f"PDFs: {copied} copied with new names")
print("=" * 70)
