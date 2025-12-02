"""
Generate Vector Database Architecture visualization.
Shows: (1) How vector DB works internally, (2) Vector DB in RAG pipeline.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import os

# Quantlet metadata for branding
CHART_METADATA = {
    'name': '08_vector_db_architecture',
    'url': 'https://github.com/Digital-AI-Finance/Natural-Language-Processing/tree/main/FinalLecture/08_vector_db_architecture'
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
mlteal = '#17BECF'
mlpink = '#E377C2'

def draw_arrow(ax, start, end, color=mlgray, lw=2, style='->', connectionstyle='arc3,rad=0'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle=connectionstyle))

def draw_box(ax, x, y, width, height, text, color, alpha=0.3, fontsize=8,
             text_color='black', bold=True):
    """Draw a rounded box with text."""
    box = FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0.02',
                          facecolor=color, alpha=alpha, edgecolor=color, linewidth=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color=text_color)

def create_vector_db_architecture():
    """Create two-panel visualization of Vector DB architecture."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Vector Database: Architecture and Role in RAG',
                 fontsize=14, fontweight='bold', color=mlpurple, y=0.98)

    # ========== Panel 1: How Vector DB Works Internally ==========
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('How Vector DB Works Internally', fontsize=11, fontweight='bold',
                  color=mlpurple, pad=10)

    # Stage 1: Documents/Data Input (left side)
    draw_box(ax1, 0.2, 7.5, 2, 1.8, 'Documents\n(text, images)', mlblue, fontsize=9)

    # Stage 2: Embedding Model
    draw_box(ax1, 0.2, 5, 2, 1.8, 'Embedding\nModel', mlorange, fontsize=9)
    draw_arrow(ax1, (1.2, 7.5), (1.2, 6.9), mlgray, 2)

    # Stage 3: Vectors
    draw_box(ax1, 0.2, 2.5, 2, 1.8, 'Vectors\n[0.2, -0.5, ...]', mlgreen, fontsize=8)
    draw_arrow(ax1, (1.2, 5), (1.2, 4.4), mlgray, 2)

    # Vector DB Internal Components (center-right)
    # Main DB box outline
    db_box = FancyBboxPatch((3, 1.5), 6.5, 7.5, boxstyle='round,pad=0.05',
                             facecolor='#F0F0FF', alpha=0.5, edgecolor=mlpurple,
                             linewidth=2.5, linestyle='--')
    ax1.add_patch(db_box)
    ax1.text(6.25, 8.7, 'Vector Database', ha='center', fontsize=10,
             fontweight='bold', color=mlpurple)

    # Inside DB: Index Structure
    draw_box(ax1, 3.5, 6.5, 2.5, 1.8, 'Index Structure\n(HNSW/IVF)', mlred, fontsize=8)

    # Inside DB: Vector Storage
    draw_box(ax1, 6.5, 6.5, 2.5, 1.8, 'Vector Storage\n(compressed)', mlteal, fontsize=8)

    # Inside DB: Metadata Storage
    draw_box(ax1, 3.5, 4, 2.5, 1.8, 'Metadata\nStorage', mlpink, fontsize=8)

    # Inside DB: ANN Search Engine
    draw_box(ax1, 6.5, 4, 2.5, 1.8, 'ANN Search\nEngine', mlgreen, fontsize=8)

    # Inside DB: Query Processor
    draw_box(ax1, 5, 1.8, 3, 1.5, 'Query Processor', mlorange, fontsize=9)

    # Arrows inside DB
    draw_arrow(ax1, (4.75, 6.5), (4.75, 5.9), mlgray, 1.5)  # Index to Metadata
    draw_arrow(ax1, (7.75, 6.5), (7.75, 5.9), mlgray, 1.5)  # Storage to ANN
    draw_arrow(ax1, (6, 4.9), (6.5, 4.9), mlgray, 1.5)  # Metadata to ANN
    draw_arrow(ax1, (7.75, 4), (7, 3.4), mlgray, 1.5)  # ANN to Query

    # Arrow from vectors to DB
    draw_arrow(ax1, (2.2, 3.4), (3.5, 5), mlgray, 2)
    ax1.text(2.5, 4.5, 'Insert', fontsize=8, color=mlgray, style='italic', rotation=35)

    # Query input arrow
    ax1.annotate('', xy=(5, 1.8), xytext=(2, 0.8),
                arrowprops=dict(arrowstyle='->', color=mlblue, lw=2))
    ax1.text(1, 0.5, 'Query\nVector', ha='center', fontsize=8, color=mlblue, fontweight='bold')

    # Output arrow
    ax1.annotate('', xy=(9.5, 2.5), xytext=(8, 2.5),
                arrowprops=dict(arrowstyle='->', color=mlgreen, lw=2))
    ax1.text(9.5, 2.5, 'Top-k\nResults', ha='left', fontsize=8, color=mlgreen, fontweight='bold')

    # ========== Panel 2: Vector DB in RAG Pipeline ==========
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Vector DB in RAG Pipeline', fontsize=11, fontweight='bold',
                  color=mlpurple, pad=10)

    # User Query (top)
    draw_box(ax2, 3.5, 8.5, 3, 1.2, 'User Query', mlblue, fontsize=10)
    ax2.text(5, 8.2, '"What is the capital\nof France?"', ha='center', fontsize=7,
             style='italic', color=mlgray)

    # Embedding
    draw_box(ax2, 3.5, 6.5, 3, 1.2, 'Embed Query', mlorange, fontsize=9)
    draw_arrow(ax2, (5, 8.5), (5, 7.8), mlgray, 2)

    # Vector DB (highlighted)
    db_box2 = FancyBboxPatch((2, 4), 6, 2, boxstyle='round,pad=0.05',
                              facecolor=mlpurple, alpha=0.15, edgecolor=mlpurple,
                              linewidth=3)
    ax2.add_patch(db_box2)
    ax2.text(5, 5.3, 'Vector Database', ha='center', fontsize=11,
             fontweight='bold', color=mlpurple)
    ax2.text(5, 4.5, 'ANN Search: find similar docs', ha='center', fontsize=8,
             color=mlgray, style='italic')
    draw_arrow(ax2, (5, 6.5), (5, 6.1), mlgray, 2)

    # Retrieved Documents
    doc_y = 2.2
    ax2.text(5, 3.5, 'Retrieved Documents (top-k)', ha='center', fontsize=9,
             fontweight='bold', color=mlgreen)
    draw_arrow(ax2, (5, 4), (5, 3.6), mlgray, 2)

    # Document boxes
    for i, (name, desc) in enumerate([('$z_1$', 'Paris is capital...'),
                                       ('$z_2$', 'France in Europe...'),
                                       ('$z_3$', 'Eiffel Tower...')]):
        x_pos = 1.5 + i * 2.7
        box = FancyBboxPatch((x_pos, doc_y), 2.3, 0.9, boxstyle='round,pad=0.02',
                              facecolor=mlgreen, alpha=0.3, edgecolor=mlgreen, linewidth=1)
        ax2.add_patch(box)
        ax2.text(x_pos + 1.15, doc_y + 0.55, name, ha='center', fontsize=8, fontweight='bold')
        ax2.text(x_pos + 1.15, doc_y + 0.25, desc, ha='center', fontsize=6, color=mlgray)

    # LLM
    draw_box(ax2, 3.5, 0.3, 3, 1.3, 'LLM Generator', mlred, fontsize=9)
    ax2.text(5, -0.1, '$p(y|x, z_1, z_2, z_3)$', ha='center', fontsize=8, color=mlred)

    # Arrows from docs to LLM
    for i in range(3):
        x_pos = 2.65 + i * 2.7
        draw_arrow(ax2, (x_pos, doc_y), (5, 1.7), mlgray, 1.5)

    # Also connect query to LLM
    ax2.annotate('', xy=(3.5, 1), xytext=(1, 8.5),
                arrowprops=dict(arrowstyle='->', color=mlblue, lw=1.5,
                               connectionstyle='arc3,rad=0.3', linestyle='--'))
    ax2.text(0.8, 4.5, 'Query\ncontext', fontsize=7, color=mlblue, style='italic')

    # Key insight box
    insight_box = FancyBboxPatch((0.3, 0.3), 2.5, 1.3, boxstyle='round,pad=0.05',
                                  facecolor='#FFFACD', alpha=0.8, edgecolor=mlorange,
                                  linewidth=1.5)
    ax2.add_patch(insight_box)
    ax2.text(1.55, 1.1, 'Key Role:', ha='center', fontsize=8, fontweight='bold', color=mlorange)
    ax2.text(1.55, 0.65, 'Fast retrieval of\nrelevant context', ha='center', fontsize=7, color=mlgray)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'vector_db_architecture.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Chart saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_vector_db_architecture()
