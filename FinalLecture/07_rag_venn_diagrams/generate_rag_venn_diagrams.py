"""
Generate Venn diagram visualization for RAG conditional probabilities.
Shows p(y|x), p(z|x), and p(y|x,z) relationships using proper Venn/set diagrams.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Wedge
import numpy as np
import os

# Quantlet metadata for branding
CHART_METADATA = {
    'name': '07_rag_venn_diagrams',
    'url': 'https://github.com/Digital-AI-Finance/Natural-Language-Processing/tree/main/FinalLecture/07_rag_venn_diagrams'
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

def create_rag_venn_diagrams():
    """Create Venn diagrams showing conditional probability relationships in RAG."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('RAG Probabilities: Venn Diagram Interpretation',
                 fontsize=14, fontweight='bold', color=mlpurple, y=0.98)

    # ========== Panel 1: Sample Space and Events ==========
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Sample Space: All Possible Outcomes', fontsize=10, fontweight='bold',
                  color=mlpurple, pad=10)

    # Universe rectangle (sample space)
    universe = FancyBboxPatch((0.5, 1), 9, 7.5, boxstyle='round,pad=0.02',
                               facecolor='#F5F5F5', edgecolor=mlgray, linewidth=2)
    ax1.add_patch(universe)
    ax1.text(5, 8.8, 'Sample Space $\\Omega$', ha='center', fontsize=9, color=mlgray)

    # Event X (query matches) - left circle
    circle_x = Circle((3.5, 5), 2.2, facecolor=mlblue, alpha=0.25,
                       edgecolor=mlblue, linewidth=2)
    ax1.add_patch(circle_x)
    ax1.text(2.3, 6.8, '$X$: Query', ha='center', fontsize=9, fontweight='bold', color=mlblue)
    ax1.text(2.3, 6.3, 'relevant', ha='center', fontsize=8, color=mlblue)

    # Event Z (document retrieved) - middle circle
    circle_z = Circle((5, 4.5), 2.2, facecolor=mlgreen, alpha=0.25,
                       edgecolor=mlgreen, linewidth=2)
    ax1.add_patch(circle_z)
    ax1.text(6.5, 2, '$Z$: Doc', ha='center', fontsize=9, fontweight='bold', color=mlgreen)
    ax1.text(6.5, 1.5, 'retrieved', ha='center', fontsize=8, color=mlgreen)

    # Event Y (correct answer) - right circle
    circle_y = Circle((6.5, 5), 2.2, facecolor=mlred, alpha=0.25,
                       edgecolor=mlred, linewidth=2)
    ax1.add_patch(circle_y)
    ax1.text(7.7, 6.8, '$Y$: Correct', ha='center', fontsize=9, fontweight='bold', color=mlred)
    ax1.text(7.7, 6.3, 'answer', ha='center', fontsize=8, color=mlred)

    # Intersection labels
    ax1.text(5, 5.2, '$X \\cap Z \\cap Y$', ha='center', fontsize=7, fontweight='bold')

    # Legend
    ax1.text(5, 0.3, '$X$ = query context | $Z$ = retrieved doc | $Y$ = correct answer "Paris"',
             ha='center', fontsize=8, style='italic', color=mlgray)

    # ========== Panel 2: Conditional Probability p(Z|X) ==========
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('$p(Z|X)$: Retrieval Probability', fontsize=10, fontweight='bold',
                  color=mlpurple, pad=10)

    # Given X (shaded as universe)
    given_x = Circle((5, 5), 3.5, facecolor=mlblue, alpha=0.15,
                      edgecolor=mlblue, linewidth=3)
    ax2.add_patch(given_x)
    ax2.text(5, 8.8, 'Given: Query $X$', ha='center', fontsize=10, fontweight='bold', color=mlblue)

    # Z within X (retrieval success)
    z_given_x = Circle((5.5, 5.2), 2, facecolor=mlgreen, alpha=0.5,
                        edgecolor=mlgreen, linewidth=2)
    ax2.add_patch(z_given_x)
    ax2.text(5.5, 5.2, '$Z|X$', ha='center', fontsize=11, fontweight='bold', color='white')

    # Formula
    ax2.text(5, 1.5, '$p(Z|X) = \\frac{p(Z \\cap X)}{p(X)}$', ha='center', fontsize=10)

    # Interpretation
    ax2.text(5, 0.5, 'How likely is doc $Z$ retrieved given query $X$?',
             ha='center', fontsize=8, style='italic', color=mlgray)

    # Example values
    ax2.text(8.5, 7, '$z_1$: 0.52', ha='left', fontsize=8, color=mlgreen, fontweight='bold')
    ax2.text(8.5, 6.3, '$z_2$: 0.27', ha='left', fontsize=8, color=mlgreen)
    ax2.text(8.5, 5.6, '$z_3$: 0.21', ha='left', fontsize=8, color=mlgreen)

    # ========== Panel 3: Conditional Probability p(Y|X,Z) ==========
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('$p(Y|X,Z)$: Generation Probability', fontsize=10, fontweight='bold',
                  color=mlpurple, pad=10)

    # Given X and Z (intersection as universe)
    # Draw X circle
    x_circle = Circle((4, 5), 3, facecolor=mlblue, alpha=0.15,
                       edgecolor=mlblue, linewidth=2, linestyle='--')
    ax3.add_patch(x_circle)

    # Draw Z circle overlapping
    z_circle = Circle((6, 5), 3, facecolor=mlgreen, alpha=0.15,
                       edgecolor=mlgreen, linewidth=2, linestyle='--')
    ax3.add_patch(z_circle)

    # Intersection X AND Z (darker)
    # Approximate with an ellipse in the intersection region
    intersection = Ellipse((5, 5), 2.5, 4, facecolor='#8B4513', alpha=0.2,
                            edgecolor='#8B4513', linewidth=2)
    ax3.add_patch(intersection)

    ax3.text(5, 8.8, 'Given: Query $X$ AND Doc $Z$', ha='center', fontsize=10,
             fontweight='bold', color='#8B4513')

    # Y within intersection (generation success)
    y_given_xz = Circle((5, 5), 1.3, facecolor=mlred, alpha=0.6,
                         edgecolor=mlred, linewidth=2)
    ax3.add_patch(y_given_xz)
    ax3.text(5, 5, '$Y|X,Z$', ha='center', fontsize=10, fontweight='bold', color='white')

    # Labels
    ax3.text(2.5, 7, '$X$', fontsize=10, color=mlblue, fontweight='bold')
    ax3.text(7.5, 7, '$Z$', fontsize=10, color=mlgreen, fontweight='bold')

    # Formula
    ax3.text(5, 1.5, '$p(Y|X,Z) = \\frac{p(Y \\cap X \\cap Z)}{p(X \\cap Z)}$',
             ha='center', fontsize=10)

    # Interpretation
    ax3.text(5, 0.5, 'Given query AND doc, how likely is correct answer?',
             ha='center', fontsize=8, style='italic', color=mlgray)

    # Example values
    ax3.text(0.3, 3, '$p(Y|X,z_1)$=0.95', ha='left', fontsize=8, color=mlred, fontweight='bold')
    ax3.text(0.3, 2.3, '$p(Y|X,z_2)$=0.40', ha='left', fontsize=8, color=mlred)
    ax3.text(0.3, 1.6, '$p(Y|X,z_3)$=0.70', ha='left', fontsize=8, color=mlred)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'rag_venn_diagrams.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Chart saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_rag_venn_diagrams()
