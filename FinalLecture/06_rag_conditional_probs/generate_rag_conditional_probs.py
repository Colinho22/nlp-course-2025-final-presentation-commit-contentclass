"""
Generate Venn diagram visualization for RAG conditional probabilities.
Shows p(y|x), p(z|x), and p(y|x,z) relationships for the France capital example.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Quantlet metadata for branding
CHART_METADATA = {
    'name': '06_rag_conditional_probs',
    'url': 'https://github.com/Digital-AI-Finance/Natural-Language-Processing/tree/main/FinalLecture/06_rag_conditional_probs'
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

def create_rag_conditional_probs():
    """Create Venn diagram showing conditional probability relationships in RAG."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('RAG Conditional Probabilities: Visual Intuition',
                 fontsize=14, fontweight='bold', color=mlpurple, y=0.98)

    # Data for our example
    docs = [
        {'name': '$z_1$', 'desc': 'Paris is capital...', 'sim': 0.92,
         'p_z': 0.52, 'p_y_given_z': 0.95, 'color': mlgreen},
        {'name': '$z_2$', 'desc': 'France is in Europe...', 'sim': 0.71,
         'p_z': 0.27, 'p_y_given_z': 0.40, 'color': mlorange},
        {'name': '$z_3$', 'desc': 'Eiffel Tower in Paris...', 'sim': 0.65,
         'p_z': 0.21, 'p_y_given_z': 0.70, 'color': mlblue},
    ]

    # ========== Panel 1: Retrieval Probability p(z|x) ==========
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('$p(z_i|x)$: Retrieval Probability', fontsize=11, fontweight='bold',
                  color=mlpurple, pad=10)

    # Draw query circle
    query_circle = Circle((5, 7), 1.5, facecolor=mlred, alpha=0.3, edgecolor=mlred, linewidth=2)
    ax1.add_patch(query_circle)
    ax1.text(5, 7, 'Query $x$', ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.text(5, 5.2, '"Capital of France?"', ha='center', va='center', fontsize=8, style='italic')

    # Draw document circles with sizes proportional to p(z|x)
    doc_positions = [(2, 2.5), (5, 2), (8, 2.5)]
    for i, (doc, pos) in enumerate(zip(docs, doc_positions)):
        radius = 0.5 + doc['p_z'] * 1.5  # Scale radius by probability
        circle = Circle(pos, radius, facecolor=doc['color'], alpha=0.4,
                       edgecolor=doc['color'], linewidth=2)
        ax1.add_patch(circle)
        ax1.text(pos[0], pos[1], f"{doc['name']}\n{doc['p_z']:.2f}",
                ha='center', va='center', fontsize=9, fontweight='bold')

        # Arrow from query to doc
        ax1.annotate('', xy=(pos[0], pos[1] + radius + 0.2),
                    xytext=(5, 7 - 1.7),
                    arrowprops=dict(arrowstyle='->', color=doc['color'],
                                   lw=1.5 + doc['p_z']*3, alpha=0.7))

    # Explanation
    ax1.text(5, 0.3, 'Size/arrow = retrieval probability\n(how relevant is doc to query?)',
             ha='center', va='center', fontsize=8, style='italic', color=mlgray)

    # ========== Panel 2: Generation Probability p(y|x,z) ==========
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('$p(y|x,z_i)$: Generation Probability', fontsize=11, fontweight='bold',
                  color=mlpurple, pad=10)

    # For each document, show conditional probability of generating "Paris"
    y_positions = [7.5, 5, 2.5]

    for i, (doc, y_pos) in enumerate(zip(docs, y_positions)):
        # Document box
        doc_box = FancyBboxPatch((0.5, y_pos - 0.8), 4, 1.6, boxstyle='round,pad=0.1',
                                  facecolor=doc['color'], alpha=0.2, edgecolor=doc['color'])
        ax2.add_patch(doc_box)
        ax2.text(2.5, y_pos, f"{doc['name']}: {doc['desc']}", ha='center', va='center',
                fontsize=8, fontweight='bold')

        # Arrow to answer
        arrow_width = doc['p_y_given_z'] * 3
        ax2.annotate('', xy=(8.5, y_pos), xytext=(4.7, y_pos),
                    arrowprops=dict(arrowstyle='->', color=doc['color'],
                                   lw=arrow_width, alpha=0.8))

        # Probability label on arrow
        ax2.text(6.5, y_pos + 0.4, f"$p(y|x,z_{i+1})={doc['p_y_given_z']:.2f}$",
                ha='center', va='bottom', fontsize=8, color=doc['color'], fontweight='bold')

        # Answer box
        answer_box = FancyBboxPatch((8.3, y_pos - 0.5), 1.4, 1, boxstyle='round,pad=0.1',
                                     facecolor=mlred, alpha=0.15 + doc['p_y_given_z']*0.3,
                                     edgecolor=mlred, linewidth=1)
        ax2.add_patch(answer_box)
        ax2.text(9, y_pos, '"Paris"', ha='center', va='center', fontsize=8,
                fontweight='bold', color=mlred)

    # Explanation
    ax2.text(5, 0.3, 'Arrow thickness = generation probability\n(given doc, how likely is answer?)',
             ha='center', va='center', fontsize=8, style='italic', color=mlgray)

    # ========== Panel 3: Marginalization p(y|x) ==========
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('$p(y|x) = \\sum_i p(z_i|x) \\cdot p(y|x,z_i)$', fontsize=11,
                  fontweight='bold', color=mlpurple, pad=10)

    # Draw stacked bar showing contributions
    bar_x = 3
    bar_width = 4
    cumulative = 0
    contributions = []

    for doc in docs:
        contrib = doc['p_z'] * doc['p_y_given_z']
        contributions.append(contrib)

    total = sum(contributions)

    # Draw bars
    y_start = 2
    bar_height = 5

    for i, (doc, contrib) in enumerate(zip(docs, contributions)):
        height = (contrib / total) * bar_height * total  # Scale to show actual probability
        rect = FancyBboxPatch((bar_x, y_start + cumulative), bar_width, height,
                               boxstyle='round,pad=0.02', facecolor=doc['color'],
                               alpha=0.6, edgecolor=doc['color'], linewidth=1.5)
        ax3.add_patch(rect)

        # Label inside bar
        if height > 0.4:
            ax3.text(bar_x + bar_width/2, y_start + cumulative + height/2,
                    f"{doc['name']}: {doc['p_z']:.2f} x {doc['p_y_given_z']:.2f}\n= {contrib:.3f}",
                    ha='center', va='center', fontsize=8, fontweight='bold')

        cumulative += height

    # Total bar outline
    rect_outline = FancyBboxPatch((bar_x, y_start), bar_width, cumulative,
                                   boxstyle='round,pad=0.02', facecolor='none',
                                   edgecolor=mlpurple, linewidth=2)
    ax3.add_patch(rect_outline)

    # Total label
    ax3.text(bar_x + bar_width/2, y_start + cumulative + 0.5,
            f'$p(y|x) = {total:.2f}$', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=mlpurple,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8E8F8', edgecolor=mlpurple))

    # Formula breakdown on right
    ax3.text(8.5, 7, '$= 0.52 \\times 0.95$', ha='left', fontsize=9, color=mlgreen)
    ax3.text(8.5, 6.2, '$+ 0.27 \\times 0.40$', ha='left', fontsize=9, color=mlorange)
    ax3.text(8.5, 5.4, '$+ 0.21 \\times 0.70$', ha='left', fontsize=9, color=mlblue)
    ax3.plot([8.3, 9.7], [4.9, 4.9], color=mlpurple, linewidth=1)
    ax3.text(8.5, 4.4, '$= 0.75$', ha='left', fontsize=11, fontweight='bold', color=mlpurple)

    # Explanation
    ax3.text(5, 0.3, 'Each document contributes to final answer\nweighted by its retrieval probability',
             ha='center', va='center', fontsize=8, style='italic', color=mlgray)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'rag_conditional_probs.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Chart saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_rag_conditional_probs()
