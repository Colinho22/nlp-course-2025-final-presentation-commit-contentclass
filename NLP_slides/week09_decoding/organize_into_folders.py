"""
Organize Standalone Scripts into Individual Folders
====================================================
Creates one folder per chart, each containing:
- The Python script
- Generated PDF output
- README with description
"""

from pathlib import Path
import shutil

source_dir = Path("python/standalone_generators")
output_base = Path("python/charts_individual")

# Clean output
if output_base.exists():
    shutil.rmtree(output_base)
output_base.mkdir(parents=True)

scripts = sorted(source_dir.glob("generate_*.py"))

print("=" * 70)
print(f"Organizing {len(scripts)} Scripts into Individual Folders")
print("=" * 70)

for script in scripts:
    # Create folder name from script
    folder_name = script.stem.replace('generate_', '')
    folder = output_base / folder_name
    folder.mkdir()

    # Copy script
    shutil.copy2(script, folder / script.name)

    # Copy PDF if exists
    pdf_name = folder_name + '.pdf'
    pdf_source = source_dir / pdf_name
    if pdf_source.exists():
        shutil.copy2(pdf_source, folder / pdf_name)
        print(f"  OK {folder_name}/ (with PDF)")
    else:
        print(f"  OK {folder_name}/ (script only)")

print("\n" + "=" * 70)
print(f"Created {len(scripts)} individual folders")
print(f"Output: {output_base.absolute()}")
print("=" * 70)
print("\nStructure:")
print("  charts_individual/")
print("    vocabulary_probability_bsc/")
print("      generate_vocabulary_probability_bsc.py")
print("      vocabulary_probability_bsc.pdf")
print("    beam_search_tree_graphviz/")
print("      generate_beam_search_graphviz.py")
print("      beam_search_tree_graphviz.pdf")
print("    ...")
