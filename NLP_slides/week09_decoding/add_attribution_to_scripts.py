"""
Add QuantLet Attribution to Chart Generation Scripts
====================================================
Modifies all Python scripts in standalone_generators/ to add:
- Text annotation in bottom-right corner
- "Code: quantlet.com/[chart-name]" or GitHub URL
- Small, unobtrusive, gray text

This ensures all regenerated charts will have the attribution.
"""

import re
from pathlib import Path

def add_attribution_to_matplotlib_script(script_path, quantlet_name):
    """Add attribution text to matplotlib chart script before savefig."""

    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find savefig line
    pattern = r'\n([ \t]*)(plt\.savefig.*)'

    match = re.search(pattern, content)
    if not match:
        return False, "No savefig found"

    indent = match.group(1)

    # Attribution code with proper indentation
    attribution_lines = [
        "",
        "# Add QuantLet attribution",
        f"ax.text(0.98, 0.02, 'Code: quantlet.com/{quantlet_name}',",
        "        transform=ax.transAxes,",
        "        ha='right', va='bottom',",
        "        fontsize=7, color='#888888',",
        "        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',",
        "                  edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))",
    ]

    attribution_code = '\n'.join(indent + line if line else '' for line in attribution_lines)

    # Insert before savefig
    modified = content.replace(match.group(0), attribution_code + match.group(0))

    return modified, None


def add_attribution_to_graphviz_script(script_path, quantlet_name):
    """Add attribution to graphviz script by adding a label node."""

    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find render line
    pattern = r'\n([ \t]*)(dot\.render.*)'

    match = re.search(pattern, content)
    if not match:
        return False, "No render found"

    indent = match.group(1)

    # Attribution node with proper indentation
    attribution_lines = [
        "",
        "# Add QuantLet attribution",
        f"dot.node('quantlet_attr', 'Code: quantlet.com/{quantlet_name}',",
        "         shape='plaintext', fontsize='8', fontcolor='#888888')",
    ]

    attribution_code = '\n'.join(indent + line if line else '' for line in attribution_lines)

    # Insert before render
    modified = content.replace(match.group(0), attribution_code + match.group(0))

    return modified, None


def process_script(script_path):
    """Add attribution to a chart generation script."""

    # Determine chart name from filename
    chart_name = script_path.stem.replace('generate_', '')
    quantlet_name = f"NLPDecoding_{chart_name.replace('_bsc', '').replace('_graphviz', '').title().replace(' ', '')}"

    # Check if it's matplotlib or graphviz
    with open(script_path, 'r', encoding='utf-8') as f:
        preview = f.read(500)

    is_graphviz = 'graphviz' in preview.lower()

    # Add attribution
    if is_graphviz:
        result, error = add_attribution_to_graphviz_script(script_path, quantlet_name)
    else:
        result, error = add_attribution_to_matplotlib_script(script_path, quantlet_name)

    if error:
        return False, error

    # Write back
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(result)

    return True, quantlet_name


# Main execution
generators_dir = Path("python/standalone_generators")

if not generators_dir.exists():
    print("ERROR: standalone_generators/ not found")
    exit(1)

scripts = sorted(generators_dir.glob("generate_*.py"))

print("=" * 70)
print(f"Adding QuantLet Attribution to {len(scripts)} Scripts")
print("=" * 70)
print("\nWill add bottom-right annotation:")
print('  "Code: quantlet.com/NLPDecoding_ChartName"')
print("\nNote: Charts must be regenerated for attribution to appear")
print("=" * 70)

modified = 0
skipped = 0

for script in scripts:
    success, info = process_script(script)
    if success:
        print(f"  OK {script.name} -> {info}")
        modified += 1
    else:
        print(f"  SKIP {script.name} - {info}")
        skipped += 1

print("\n" + "=" * 70)
print(f"Modified: {modified} scripts")
print(f"Skipped: {skipped} scripts")
print("=" * 70)
print("\nNext step: Regenerate all charts")
print("  cd python/standalone_generators")
print("  python ../regenerate_all_tex_figures.py")
