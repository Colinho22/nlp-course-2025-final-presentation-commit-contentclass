"""
Add Official QuantLet Logo to All Chart Generation Scripts
===========================================================
Adds logo image + clickable text link to bottom-right of each chart.

Uses official QuantLet logo from GitHub: quantlet_logo_official.png
Combination: Logo image + "Code: quantlet.com/..." text
"""

import re
from pathlib import Path

def add_logo_to_matplotlib_script(script_path, quantlet_name):
    """Add QuantLet logo and text to matplotlib chart."""

    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already added
    if 'quantlet_logo_official.png' in content or 'QuantLet logo' in content:
        return False, "Logo already present"

    # Find savefig line
    pattern = r'\n([ \t]*)(plt\.savefig.*)'
    match = re.search(pattern, content)
    if not match:
        return False, "No savefig found"

    indent = match.group(1)

    # Code to add logo + text
    logo_lines = [
        "",
        "# Add QuantLet logo and attribution",
        "from matplotlib.offsetbox import OffsetImage, AnnotationBbox",
        "try:",
        "    # Add logo image",
        "    logo_path = '../../../quantlet_logo_official.png'",
        "    logo_img = plt.imread(logo_path)",
        "    imagebox = OffsetImage(logo_img, zoom=0.08)",  # Small logo
        "    ab = AnnotationBbox(imagebox, (0.98, 0.04),",
        "                        xycoords='axes fraction',",
        "                        box_alignment=(1, 0),",
        "                        frameon=False)",
        "    ax.add_artist(ab)",
        "    # Add text below logo",
        f"    ax.text(0.98, 0.01, 'quantlet.com/{quantlet_name}',",
        "            transform=ax.transAxes,",
        "            ha='right', va='bottom',",
        "            fontsize=6, color='#666666', style='italic')",
        "except Exception as e:",
        "    # Fallback to text only if logo fails",
        f"    ax.text(0.98, 0.02, 'Code: quantlet.com/{quantlet_name}',",
        "            transform=ax.transAxes,",
        "            ha='right', va='bottom',",
        "            fontsize=7, color='#888888',",
        "            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',",
        "                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))",
    ]

    # Proper indentation
    logo_code_lines = []
    for line in logo_lines:
        if line.startswith('try:') or line.startswith('except'):
            logo_code_lines.append(indent + line)
        elif line and logo_lines[logo_lines.index(line)-1].startswith(('try:', 'except')):
            logo_code_lines.append(indent + '    ' + line)
        elif line:
            logo_code_lines.append(indent + line)
        else:
            logo_code_lines.append('')

    logo_code = '\n'.join(logo_code_lines)

    # Insert before savefig
    modified = content.replace(match.group(0), logo_code + match.group(0))

    return modified, None


def add_logo_to_graphviz_script(script_path, quantlet_name):
    """Add QuantLet attribution to graphviz chart (text only - no image support)."""

    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already added
    if 'QuantLet' in content and 'quantlet.com' in content:
        return False, "Already present"

    # Find render line
    pattern = r'\n([ \t]*)(dot\.render.*)'
    match = re.search(pattern, content)
    if not match:
        return False, "No render found"

    indent = match.group(1)

    # Text annotation (graphviz doesn't support embedded images easily)
    logo_lines = [
        "",
        "# Add QuantLet attribution",
        f"dot.node('quantlet_link', 'Code: quantlet.com/{quantlet_name}',",
        "         shape='plaintext', fontsize='7', fontcolor='#666666')",
    ]

    logo_code = '\n'.join(indent + line if line else '' for line in logo_lines)

    # Insert before render
    modified = content.replace(match.group(0), logo_code + match.group(0))

    return modified, None


# Main execution
generators_dir = Path("python/standalone_generators")
scripts = sorted(generators_dir.glob("generate_*.py"))

print("=" * 70)
print(f"Adding QuantLet Logo to {len(scripts)} Chart Scripts")
print("=" * 70)
print("\nUsing official QuantLet logo from GitHub")
print("Logo: quantlet_logo_official.png")
print("Format: Logo image + text link")
print("=" * 70)

modified = 0
skipped = 0

for script in scripts:
    # Get chart name
    chart_name = script.stem.replace('generate_', '')
    quantlet_name = f"NLPDecoding_{chart_name.replace('_bsc', '').replace('_graphviz', '').title().replace(' ', '').replace('_', '')}"

    # Check if graphviz or matplotlib
    with open(script, 'r', encoding='utf-8') as f:
        preview = f.read(300)

    is_graphviz = 'graphviz' in preview.lower()

    # Add logo
    if is_graphviz:
        result, error = add_logo_to_graphviz_script(script, quantlet_name)
    else:
        result, error = add_logo_to_matplotlib_script(script, quantlet_name)

    if error:
        print(f"  SKIP {script.name} - {error}")
        skipped += 1
        continue

    # Write modified script
    with open(script, 'w', encoding='utf-8') as f:
        f.write(result)

    print(f"  OK {script.name} -> {quantlet_name}")
    modified += 1

print("\n" + "=" * 70)
print(f"Modified: {modified} scripts")
print(f"Skipped: {skipped} scripts")
print("=" * 70)
print("\nCharts will show:")
print("  - QuantLet logo (matplotlib charts)")
print("  - Text: 'quantlet.com/NLPDecoding_ChartName'")
print("\nRegenerate charts to see the logo!")
