"""
Generate HNSW Cities Example - Visual representation of the simple example.
Shows 3 layers with cities and the search path from Berlin query.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import os

# Quantlet metadata for branding
CHART_METADATA = {
    'name': '03_hnsw_cities_example',
    'url': 'https://github.com/Digital-AI-Finance/Natural-Language-Processing/tree/main/FinalLecture/03_hnsw_cities_example'
}


# Output directory
OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Color scheme
mlblue = '#0066CC'
mlpurple = '#3333B2'
mlorange = '#FF7F0E'
mlgreen = '#2CA02C'
mlred = '#D62728'
mlgray = '#7F7F7F'

def create_hnsw_cities_example():
    """Create visual HNSW example with cities."""

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'HNSW Search: Finding the Nearest City to Berlin',
            fontsize=16, fontweight='bold', ha='center', color=mlpurple)

    # City positions (approximate geographic layout)
    cities = {
        # name: (x, y)
        'Paris': (3, 4.5),
        'London': (2.5, 5.5),
        'Berlin': (4.5, 5.2),  # Query
        'Amsterdam': (3.2, 5.8),
        'Rome': (4.5, 2.5),
        'Madrid': (1, 2.5),
        'Tokyo': (12, 4),
        'Sydney': (12, 1.5),
    }

    # Layer definitions
    layer2_cities = ['Paris', 'Tokyo']
    layer1_cities = ['Paris', 'Tokyo', 'London', 'Sydney']
    layer0_cities = list(cities.keys())
    layer0_cities.remove('Berlin')  # Berlin is the query

    # Draw layer backgrounds
    layer_colors = [mlblue, mlorange, mlgreen]
    layer_labels = ['Layer 0 (All 8 cities)', 'Layer 1 (4 cities)', 'Layer 2 (2 entry points)']
    layer_y_positions = [1.2, 4.2, 7.2]
    layer_heights = [2.5, 2.5, 1.8]

    for i, (y, h, label, color) in enumerate(zip(layer_y_positions, layer_heights, layer_labels, layer_colors)):
        rect = FancyBboxPatch((0.3, y - 0.3), 13.4, h, boxstyle='round,pad=0.05',
                              facecolor=color, alpha=0.08, edgecolor=color,
                              linestyle='--', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(0.5, y + h - 0.4, label, fontsize=10, fontweight='bold',
                color=color, va='top')

    # ========== LAYER 2 ==========
    l2_y = 8.0
    l2_positions = {'Paris': (4, l2_y), 'Tokyo': (10, l2_y)}

    for city, (x, y) in l2_positions.items():
        circle = Circle((x, y), 0.4, facecolor=mlgreen, edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, city[:3], fontsize=8, ha='center', va='center',
                fontweight='bold', color='white')

    # Connection in layer 2
    ax.plot([4.4, 9.6], [l2_y, l2_y], color=mlgreen, linewidth=1.5, alpha=0.5)

    # Query arrives - Berlin
    ax.scatter([7], [l2_y + 0.8], s=200, c=mlred, marker='*', zorder=10)
    ax.text(7, l2_y + 1.2, 'Query: Berlin', fontsize=10, ha='center',
            fontweight='bold', color=mlred)

    # Arrow from query to Paris (closer)
    ax.annotate('', xy=(4.3, l2_y + 0.3), xytext=(6.8, l2_y + 0.6),
                arrowprops=dict(arrowstyle='->', color=mlred, lw=2.5))
    ax.text(5.5, l2_y + 0.8, '1', fontsize=11, fontweight='bold', color=mlred,
            bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', edgecolor=mlred))

    # ========== LAYER 1 ==========
    l1_y = 5.2
    l1_positions = {'Paris': (3, l1_y), 'Tokyo': (11, l1_y), 'London': (5, l1_y), 'Sydney': (9, l1_y)}

    for city, (x, y) in l1_positions.items():
        circle = Circle((x, y), 0.4, facecolor=mlorange, edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, city[:3], fontsize=8, ha='center', va='center',
                fontweight='bold', color='white')

    # Connections in layer 1
    ax.plot([3.4, 4.6], [l1_y, l1_y], color=mlorange, linewidth=1.5, alpha=0.5)
    ax.plot([5.4, 8.6], [l1_y, l1_y], color=mlorange, linewidth=1.5, alpha=0.5)
    ax.plot([9.4, 10.6], [l1_y, l1_y], color=mlorange, linewidth=1.5, alpha=0.5)

    # Vertical connection from Layer 2 Paris to Layer 1 Paris
    ax.plot([4, 3], [l2_y - 0.4, l1_y + 0.5], color=mlgray, linewidth=1, linestyle=':', alpha=0.5)

    # Arrow from Paris to London (closer to Berlin)
    ax.annotate('', xy=(4.6, l1_y), xytext=(3.4, l1_y),
                arrowprops=dict(arrowstyle='->', color=mlred, lw=2.5))
    ax.text(4, l1_y + 0.6, '2', fontsize=11, fontweight='bold', color=mlred,
            bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', edgecolor=mlred))

    # ========== LAYER 0 ==========
    l0_y = 2.2
    l0_positions = {
        'Paris': (2, l0_y),
        'London': (3.5, l0_y + 0.8),
        'Amsterdam': (5, l0_y + 0.5),
        'Rome': (6.5, l0_y - 0.5),
        'Madrid': (1, l0_y - 0.3),
        'Tokyo': (10, l0_y),
        'Sydney': (11.5, l0_y - 0.5),
    }

    for city, (x, y) in l0_positions.items():
        if city == 'Amsterdam':
            # Highlight Amsterdam as the result
            circle = Circle((x, y), 0.45, facecolor=mlgreen, edgecolor=mlred, linewidth=3)
        else:
            circle = Circle((x, y), 0.4, facecolor=mlblue, edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, city[:3], fontsize=7, ha='center', va='center',
                fontweight='bold', color='white')

    # Connections in layer 0 (some nearby connections)
    connections = [
        ('Paris', 'London'), ('London', 'Amsterdam'), ('Paris', 'Madrid'),
        ('Amsterdam', 'Rome'), ('Rome', 'Paris'), ('Tokyo', 'Sydney')
    ]
    for c1, c2 in connections:
        x1, y1 = l0_positions[c1]
        x2, y2 = l0_positions[c2]
        ax.plot([x1, x2], [y1, y2], color=mlblue, linewidth=1, alpha=0.3)

    # Vertical connection from Layer 1 London to Layer 0 London
    ax.plot([5, 3.5], [l1_y - 0.4, l0_y + 1.2], color=mlgray, linewidth=1, linestyle=':', alpha=0.5)

    # Arrow from London to Amsterdam (found!)
    ax.annotate('', xy=(4.6, l0_y + 0.5), xytext=(3.8, l0_y + 0.7),
                arrowprops=dict(arrowstyle='->', color=mlred, lw=2.5))
    ax.text(4.2, l0_y + 1.2, '3', fontsize=11, fontweight='bold', color=mlred,
            bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', edgecolor=mlred))

    # Result annotation
    ax.annotate('FOUND!', xy=(5, l0_y + 0.5), xytext=(6.5, l0_y + 1.5),
                fontsize=11, fontweight='bold', color=mlgreen,
                arrowprops=dict(arrowstyle='->', color=mlgreen, lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor=mlgreen))

    # ========== SUMMARY BOX ==========
    summary_box = FancyBboxPatch((8.5, 0.3), 5, 3.2, boxstyle='round,pad=0.1',
                                  facecolor='white', edgecolor=mlpurple, linewidth=2)
    ax.add_patch(summary_box)

    ax.text(11, 3.2, 'Search Path', fontsize=12, fontweight='bold',
            ha='center', color=mlpurple)

    summary_text = (
        "1  Check entry points\n"
        "    Paris closer than Tokyo\n\n"
        "2  Descend, check neighbors\n"
        "    London closer than Paris\n\n"
        "3  Bottom layer search\n"
        "    Amsterdam is nearest!"
    )
    ax.text(9, 2.8, summary_text, fontsize=9, va='top', family='monospace',
            linespacing=1.3)

    # Total comparisons
    ax.text(11, 0.5, 'Total: 9 comparisons (not 8!)', fontsize=9,
            ha='center', fontweight='bold', color=mlpurple)

    plt.tight_layout()

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'hnsw_cities_example.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Chart saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_hnsw_cities_example()
