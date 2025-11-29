"""
Generate comprehensive ANN explanation chart with mathematical and conceptual content.
Shows: The Problem, Exact Solution, ANN Solution, and Key Trade-off.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np
import os

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Color scheme
mlblue = '#0066CC'
mlpurple = '#3333B2'
mlorange = '#FF7F0E'
mlgreen = '#2CA02C'
mlred = '#D62728'
mlgray = '#7F7F7F'

def create_ann_math_concept():
    """Create comprehensive ANN explanation with math and concept."""

    fig = plt.figure(figsize=(14, 8))

    # Title
    fig.suptitle('Approximate Nearest Neighbor (ANN): The Core Idea',
                 fontsize=16, fontweight='bold', color=mlpurple, y=0.98)

    # Create 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25,
                          left=0.05, right=0.95, top=0.90, bottom=0.08)

    # ========== TOP LEFT: The Problem ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')

    # Title box
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.3, boxstyle='round,pad=0.1',
                                facecolor=mlpurple, alpha=0.15, edgecolor=mlpurple)
    ax1.add_patch(title_box)
    ax1.text(5, 9.2, 'The Problem', fontsize=13, fontweight='bold',
             ha='center', va='center', color=mlpurple)

    # Problem description
    problem_text = (
        "Given: Database of n vectors\n"
        "       D = {d_1, d_2, ..., d_n}\n\n"
        "Query: Find k vectors closest to q\n\n"
        "Exact solution requires:\n"
        "  - Compute distance to ALL n vectors\n"
        "  - Sort and return top-k\n"
        "  - Time: O(n) per query"
    )
    ax1.text(0.8, 4.5, problem_text, fontsize=10, va='center',
             family='monospace', linespacing=1.5)

    # Warning box
    warn_box = FancyBboxPatch((0.5, 0.5), 9, 1.5, boxstyle='round,pad=0.1',
                               facecolor=mlred, alpha=0.1, edgecolor=mlred)
    ax1.add_patch(warn_box)
    ax1.text(5, 1.25, 'n = 1 billion? That is 1 billion distance calculations per query!',
             fontsize=9, ha='center', va='center', color=mlred, style='italic')

    # ========== TOP RIGHT: The Math ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    # Title box
    title_box2 = FancyBboxPatch((0.5, 8.5), 9, 1.3, boxstyle='round,pad=0.1',
                                 facecolor=mlblue, alpha=0.15, edgecolor=mlblue)
    ax2.add_patch(title_box2)
    ax2.text(5, 9.2, 'The Mathematics', fontsize=13, fontweight='bold',
             ha='center', va='center', color=mlblue)

    # Math content
    math_text = (
        "Exact k-NN:\n"
        "  N_k(q) = argmin |S|=k  max  ||q - d||\n"
        "                       d in S\n\n"
        "c-Approximate k-NN:\n"
        "  For all d in ANN_k(q):\n"
        "    ||q - d|| <= c * ||q - d*||\n\n"
        "  where d* is the true k-th neighbor\n"
        "  and c >= 1 is the approximation factor"
    )
    ax2.text(0.8, 4.5, math_text, fontsize=10, va='center',
             family='monospace', linespacing=1.5)

    # Key insight box
    insight_box = FancyBboxPatch((0.5, 0.5), 9, 1.5, boxstyle='round,pad=0.1',
                                  facecolor=mlgreen, alpha=0.1, edgecolor=mlgreen)
    ax2.add_patch(insight_box)
    ax2.text(5, 1.25, 'c = 1.05 means we accept neighbors at most 5% farther than optimal',
             fontsize=9, ha='center', va='center', color=mlgreen, style='italic')

    # ========== BOTTOM LEFT: Visual Intuition ==========
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')

    # Title box
    title_box3 = FancyBboxPatch((0.5, 8.5), 9, 1.3, boxstyle='round,pad=0.1',
                                 facecolor=mlorange, alpha=0.15, edgecolor=mlorange)
    ax3.add_patch(title_box3)
    ax3.text(5, 9.2, 'Visual Intuition', fontsize=13, fontweight='bold',
             ha='center', va='center', color=mlorange)

    # Draw embedding space
    np.random.seed(42)
    n_points = 30
    px = np.random.uniform(1, 9, n_points)
    py = np.random.uniform(1, 7.5, n_points)

    # Query point
    qx, qy = 5, 4

    # True nearest neighbors (3 closest)
    distances = np.sqrt((px - qx)**2 + (py - qy)**2)
    nearest_idx = np.argsort(distances)[:3]

    # Draw all points
    ax3.scatter(px, py, s=40, c=mlblue, alpha=0.5, zorder=2)

    # Highlight true neighbors
    ax3.scatter(px[nearest_idx], py[nearest_idx], s=80, c=mlgreen,
                edgecolors='white', linewidths=1.5, zorder=3)

    # Draw exact radius circle
    exact_radius = distances[nearest_idx[-1]]
    circle_exact = Circle((qx, qy), exact_radius, fill=False,
                           edgecolor=mlgreen, linewidth=2, linestyle='-', zorder=1)
    ax3.add_patch(circle_exact)

    # Draw approximate radius circle (1.2x)
    approx_radius = exact_radius * 1.2
    circle_approx = Circle((qx, qy), approx_radius, fill=False,
                            edgecolor=mlorange, linewidth=2, linestyle='--', zorder=1)
    ax3.add_patch(circle_approx)

    # Query point
    ax3.scatter([qx], [qy], s=150, c=mlred, marker='*', zorder=4,
                edgecolors='white', linewidths=1.5)
    ax3.annotate('Query', (qx, qy), xytext=(qx+0.5, qy+0.6),
                 fontsize=9, fontweight='bold', color=mlred)

    # Legend
    ax3.text(1, 0.3, 'Exact boundary', fontsize=9, color=mlgreen)
    ax3.plot([0.5, 0.9], [0.3, 0.3], color=mlgreen, linewidth=2)
    ax3.text(5.5, 0.3, 'Approx boundary (c=1.2)', fontsize=9, color=mlorange)
    ax3.plot([5, 5.4], [0.3, 0.3], color=mlorange, linewidth=2, linestyle='--')

    # ========== BOTTOM RIGHT: The Trade-off ==========
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')

    # Title box
    title_box4 = FancyBboxPatch((0.5, 8.5), 9, 1.3, boxstyle='round,pad=0.1',
                                 facecolor=mlgreen, alpha=0.15, edgecolor=mlgreen)
    ax4.add_patch(title_box4)
    ax4.text(5, 9.2, 'The Trade-off', fontsize=13, fontweight='bold',
             ha='center', va='center', color=mlgreen)

    # Trade-off table
    table_data = [
        ['Method', 'Time', 'Recall', 'Use Case'],
        ['Exact (brute)', 'O(n)', '100%', 'Small datasets'],
        ['IVF', 'O(sqrt(n))', '~95%', 'Medium scale'],
        ['HNSW', 'O(log n)', '~99%', 'Production'],
        ['LSH', 'O(1)*', '~90%', 'Massive scale'],
    ]

    # Draw table
    row_height = 1.1
    col_widths = [2.2, 1.8, 1.5, 2.8]
    start_x = 0.8
    start_y = 7.5

    for i, row in enumerate(table_data):
        y = start_y - i * row_height
        x = start_x
        for j, cell in enumerate(row):
            # Header row styling
            if i == 0:
                weight = 'bold'
                bg_color = mlpurple
                alpha = 0.2
            else:
                weight = 'normal'
                bg_color = mlgray
                alpha = 0.05

            # Cell background
            cell_box = FancyBboxPatch((x, y - 0.4), col_widths[j] - 0.1, row_height - 0.1,
                                       boxstyle='round,pad=0.02',
                                       facecolor=bg_color, alpha=alpha, edgecolor=mlgray,
                                       linewidth=0.5)
            ax4.add_patch(cell_box)

            # Cell text
            ax4.text(x + col_widths[j]/2 - 0.05, y + 0.15, cell,
                     fontsize=9, ha='center', va='center', fontweight=weight)
            x += col_widths[j]

    # Footer note
    ax4.text(5, 0.6, '* LSH: O(1) query but O(n) space for hash tables',
             fontsize=8, ha='center', color=mlgray, style='italic')

    # Key message
    ax4.text(5, 1.5, 'Key: Accept 1-5% accuracy loss for 100-1000x speedup',
             fontsize=10, ha='center', fontweight='bold', color=mlpurple,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8E8F8',
                       edgecolor=mlpurple, alpha=0.5))

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'ann_math_concept.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Chart saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_ann_math_concept()
