"""
Generate HNSW Explanation chart comparing Exact Search vs HNSW.
Visual comparison showing brute force vs hierarchical graph approach.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import numpy as np
import os

# Quantlet metadata for branding
CHART_METADATA = {
    'name': '02_hnsw_explanation',
    'url': 'https://github.com/Digital-AI-Finance/Natural-Language-Processing/tree/main/FinalLecture/02_hnsw_explanation'
}


# Output directory
OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Color scheme (matching existing charts)
mlblue = '#0066CC'
mlpurple = '#3333B2'
mlorange = '#FF7F0E'
mlgreen = '#2CA02C'
mlred = '#D62728'
mlgray = '#7F7F7F'

def create_hnsw_explanation():
    """Create side-by-side comparison of Exact Search vs HNSW."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Exact Search vs HNSW: Why Approximate is Faster',
                 fontsize=14, fontweight='bold', color=mlpurple, y=0.98)

    # ========== LEFT PANEL: Exact Search (Brute Force) ==========
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Exact Search (Brute Force)', fontsize=12, fontweight='bold',
                  color=mlred, pad=10)

    # Draw document points
    np.random.seed(42)
    n_docs = 20
    doc_x = np.random.uniform(1, 9, n_docs)
    doc_y = np.random.uniform(1, 8, n_docs)

    # Query point
    query_x, query_y = 5, 4.5

    # Draw ALL connections from query to documents (brute force)
    for i in range(n_docs):
        ax1.plot([query_x, doc_x[i]], [query_y, doc_y[i]],
                 color=mlgray, alpha=0.4, linewidth=0.8, linestyle='-')

    # Draw document points
    ax1.scatter(doc_x, doc_y, s=80, c=mlblue, alpha=0.7, zorder=3, edgecolors='white')

    # Draw query point
    ax1.scatter([query_x], [query_y], s=150, c=mlred, marker='*', zorder=4,
                edgecolors='white', linewidths=1.5)
    ax1.annotate('Query', (query_x, query_y), xytext=(query_x+0.5, query_y+0.8),
                 fontsize=10, fontweight='bold', color=mlred)

    # Complexity annotation
    ax1.text(5, 0.3, 'Complexity: O(n)', fontsize=11, ha='center',
             fontweight='bold', color=mlred,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor=mlred))

    # Description
    ax1.text(5, 9.3, 'Compare query to ALL documents', fontsize=10, ha='center',
             style='italic', color=mlgray)

    # ========== RIGHT PANEL: HNSW ==========
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('HNSW (Hierarchical Navigable Small World)', fontsize=12,
                  fontweight='bold', color=mlgreen, pad=10)

    # Layer labels on left
    layer_y = [1.5, 4.5, 7.5]
    layer_labels = ['Layer 0\n(All nodes)', 'Layer 1\n(Subset)', 'Layer 2\n(Entry)']
    layer_colors = [mlblue, mlorange, mlgreen]

    for i, (y, label, color) in enumerate(zip(layer_y, layer_labels, layer_colors)):
        ax2.text(-0.3, y, label, fontsize=8, ha='right', va='center', color=color,
                 fontweight='bold')
        # Layer background
        rect = FancyBboxPatch((0.5, y-1.2), 9, 2.2, boxstyle='round,pad=0.1',
                              facecolor=color, alpha=0.08, edgecolor=color,
                              linestyle='--', linewidth=1)
        ax2.add_patch(rect)

    # Layer 0: Dense connections (many nodes)
    l0_x = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    l0_y = np.array([1.5, 1.8, 1.3, 1.6, 1.4, 1.7, 1.5, 1.6])
    ax2.scatter(l0_x, l0_y, s=60, c=mlblue, alpha=0.8, zorder=3, edgecolors='white')

    # Layer 0 connections (dense)
    for i in range(len(l0_x)-1):
        ax2.plot([l0_x[i], l0_x[i+1]], [l0_y[i], l0_y[i+1]],
                 color=mlblue, alpha=0.3, linewidth=1)

    # Layer 1: Fewer nodes
    l1_x = np.array([2, 4.5, 7])
    l1_y = np.array([4.5, 4.8, 4.4])
    ax2.scatter(l1_x, l1_y, s=80, c=mlorange, alpha=0.9, zorder=3, edgecolors='white')

    # Layer 1 connections
    for i in range(len(l1_x)-1):
        ax2.plot([l1_x[i], l1_x[i+1]], [l1_y[i], l1_y[i+1]],
                 color=mlorange, alpha=0.5, linewidth=1.5)

    # Vertical connections Layer 1 -> Layer 0
    ax2.plot([2, 2.5], [4.5, 1.8], color=mlgray, alpha=0.3, linewidth=1, linestyle=':')
    ax2.plot([4.5, 4.5], [4.8, 1.6], color=mlgray, alpha=0.3, linewidth=1, linestyle=':')
    ax2.plot([7, 7.5], [4.4, 1.5], color=mlgray, alpha=0.3, linewidth=1, linestyle=':')

    # Layer 2: Entry point
    l2_x, l2_y = 5, 7.5
    ax2.scatter([l2_x], [l2_y], s=120, c=mlgreen, alpha=1, zorder=4,
                edgecolors='white', linewidths=2)
    ax2.annotate('Entry', (l2_x, l2_y), xytext=(l2_x+0.8, l2_y+0.3),
                 fontsize=9, fontweight='bold', color=mlgreen)

    # Vertical connection Layer 2 -> Layer 1
    ax2.plot([5, 4.5], [7.5, 4.8], color=mlgray, alpha=0.3, linewidth=1, linestyle=':')

    # Search path (highlighted)
    # Entry -> Layer 1 -> Layer 0 -> Target
    search_path = [(5, 7.5), (4.5, 4.8), (4.5, 1.6)]
    target = (4.5, 1.6)

    for i in range(len(search_path)-1):
        ax2.annotate('', xy=search_path[i+1], xytext=search_path[i],
                     arrowprops=dict(arrowstyle='->', color=mlred, lw=2.5))

    # Highlight target
    ax2.scatter([target[0]], [target[1]], s=150, c=mlred, marker='*', zorder=5,
                edgecolors='white', linewidths=1.5)
    ax2.annotate('Found!', (target[0], target[1]), xytext=(target[0]+0.6, target[1]-0.5),
                 fontsize=9, fontweight='bold', color=mlred)

    # Complexity annotation
    ax2.text(5, 0.3, 'Complexity: O(log n)', fontsize=11, ha='center',
             fontweight='bold', color=mlgreen,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor=mlgreen))

    # Description
    ax2.text(5, 9.3, 'Navigate graph: sparse top -> dense bottom', fontsize=10,
             ha='center', style='italic', color=mlgray)

    # ========== Bottom annotation ==========
    fig.text(0.5, 0.02,
             'Trade-off: HNSW achieves 95-99% recall with 100-1000x speedup over exact search',
             ha='center', fontsize=11, style='italic', color=mlpurple,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8E8F8', edgecolor=mlpurple, alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'hnsw_explanation.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Chart saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_hnsw_explanation()
