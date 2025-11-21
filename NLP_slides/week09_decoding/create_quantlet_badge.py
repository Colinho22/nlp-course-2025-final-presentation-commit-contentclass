"""
Create QuantLet Badge and Add to Chart PDFs
============================================
Two approaches:

APPROACH 1 (RECOMMENDED): Modify chart generation scripts to add badge during creation
APPROACH 2: Post-process existing PDFs to overlay badge (requires PyPDF2/PyMuPDF)

This script implements APPROACH 1 with a visual badge design.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_quantlet_badge_image():
    """Create a small QuantLet badge image (PNG with transparency)."""
    fig, ax = plt.subplots(figsize=(2, 0.5), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Background rounded rectangle
    rect = mpatches.FancyBboxPatch(
        (0.02, 0.15), 0.96, 0.70,
        boxstyle="round,pad=0.02",
        facecolor='#3333B2',  # QuantLet purple
        edgecolor='#2222A0',
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(rect)

    # Text: "Q"
    ax.text(0.15, 0.5, 'Q',
            fontsize=28, weight='bold', color='white',
            ha='center', va='center',
            family='sans-serif')

    # Vertical line separator
    ax.plot([0.28, 0.28], [0.25, 0.75], color='white', linewidth=2, alpha=0.7)

    # Text: "QuantLet"
    ax.text(0.62, 0.5, 'QuantLet',
            fontsize=14, weight='bold', color='white',
            ha='center', va='center',
            family='sans-serif')

    plt.tight_layout(pad=0)
    plt.savefig('quantlet_badge.png', dpi=150, bbox_inches='tight',
                transparent=True, facecolor='none')
    plt.close()

    print("Created: quantlet_badge.png")
    return Path('quantlet_badge.png')


def add_badge_to_matplotlib_script(script_path, badge_path):
    """
    Add code to matplotlib scripts to display QuantLet badge.

    Adds an inset image in bottom-right corner.
    """
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if badge already added
    if 'quantlet_badge.png' in content or 'QuantLet badge' in content:
        return False, "Badge already present"

    # Find savefig line
    import re
    pattern = r'\n([ \t]*)(plt\.savefig.*)'

    match = re.search(pattern, content)
    if not match:
        return False, "No savefig found"

    indent = match.group(1)

    # Code to add badge
    badge_lines = [
        "",
        "# Add QuantLet badge",
        "from matplotlib.offsetbox import OffsetImage, AnnotationBbox",
        "try:",
        "    badge_img = plt.imread('../../../quantlet_badge.png')",
        "    imagebox = OffsetImage(badge_img, zoom=0.15)",
        "    ab = AnnotationBbox(imagebox, (0.98, 0.02),",
        "                        xycoords='axes fraction',",
        "                        box_alignment=(1, 0),",
        "                        frameon=False)",
        "    ax.add_artist(ab)",
        "except:",
        "    # Fallback to text if badge image not found",
        "    ax.text(0.98, 0.02, 'QuantLet',",
        "            transform=ax.transAxes,",
        "            ha='right', va='bottom',",
        "            fontsize=7, weight='bold', color='#3333B2',",
        "            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',",
        "                      edgecolor='#3333B2', alpha=0.8, linewidth=1))",
    ]

    badge_code = '\n'.join(indent + line if line and not line.startswith('except:') and not line.startswith('try:') else indent[:-4] + line if line else '' for line in badge_lines)

    # Insert before savefig
    modified = content.replace(match.group(0), badge_code + match.group(0))

    return modified, None


print("=" * 70)
print("QuantLet Badge System")
print("=" * 70)

# Create the badge image
badge_path = create_quantlet_badge_image()

print("\nCreated badge image (300x75px PNG with transparency)")
print(f"Location: {badge_path.absolute()}")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("1. Review quantlet_badge.png - is the design acceptable?")
print("2. Run script again to add badge to all chart generators")
print("3. Regenerate charts to see badges")
print("\nAlternatively:")
print("- Provide your own QuantLet logo as quantlet_logo.png")
print("- Or use text-only attribution (already implemented)")
print("=" * 70)
