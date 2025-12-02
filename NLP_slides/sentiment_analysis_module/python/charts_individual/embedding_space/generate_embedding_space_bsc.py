"""
Embedding Space t-SNE Visualization for Sentiment Analysis

PEDAGOGICAL PURPOSE:
- Show how BERT [CLS] embeddings separate positive vs negative reviews
- Visualize the learned representation space
- Make abstract "embedding" concept concrete

GENUINELY NEEDS VISUALIZATION: Yes - 2D clustering cannot be expressed in text
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# BSc Discovery Color Scheme
COLOR_MAIN = '#404040'
COLOR_ACCENT = '#3333B2'
COLOR_GREEN = '#2CA02C'
COLOR_RED = '#D62728'
COLOR_LIGHT = '#F0F0F0'

# BSc Discovery Font Standard
FONTSIZE_TITLE = 24
FONTSIZE_LABEL = 20
FONTSIZE_TICK = 16
FONTSIZE_ANNOTATION = 18

def create_chart():
    np.random.seed(42)

    # Generate clustered data (simulating t-SNE output)
    n_samples = 150

    # Positive reviews cluster (top-right)
    pos_center = np.array([3, 2])
    pos_points = pos_center + np.random.randn(n_samples, 2) * 0.8

    # Negative reviews cluster (bottom-left)
    neg_center = np.array([-2.5, -1.5])
    neg_points = neg_center + np.random.randn(n_samples, 2) * 0.9

    # Some ambiguous points in between (sarcasm, mixed sentiment)
    mixed_center = np.array([0, 0.5])
    mixed_pos = mixed_center + np.random.randn(20, 2) * 0.6
    mixed_neg = mixed_center + np.random.randn(20, 2) * 0.6 + np.array([0.5, -0.5])

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot clusters
    ax.scatter(pos_points[:, 0], pos_points[:, 1], c=COLOR_GREEN, alpha=0.6,
               s=80, label='Positive Reviews', edgecolors='white', linewidth=0.5)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], c=COLOR_RED, alpha=0.6,
               s=80, label='Negative Reviews', edgecolors='white', linewidth=0.5)

    # Ambiguous points
    ax.scatter(mixed_pos[:, 0], mixed_pos[:, 1], c=COLOR_GREEN, alpha=0.4,
               s=60, marker='s', edgecolors='gray', linewidth=1)
    ax.scatter(mixed_neg[:, 0], mixed_neg[:, 1], c=COLOR_RED, alpha=0.4,
               s=60, marker='s', edgecolors='gray', linewidth=1)

    # Add cluster centers with labels
    ax.scatter([pos_center[0]], [pos_center[1]], c=COLOR_GREEN, s=300,
               marker='*', edgecolors='black', linewidth=2, zorder=5)
    ax.scatter([neg_center[0]], [neg_center[1]], c=COLOR_RED, s=300,
               marker='*', edgecolors='black', linewidth=2, zorder=5)

    # Add example review annotations
    ax.annotate('"Absolutely loved it!"', xy=(4.2, 3.2), fontsize=FONTSIZE_TICK,
                color=COLOR_GREEN, fontweight='bold', style='italic')
    ax.annotate('"Terrible movie"', xy=(-4.5, -2.8), fontsize=FONTSIZE_TICK,
                color=COLOR_RED, fontweight='bold', style='italic')
    ax.annotate('"Great, another\nboring movie"', xy=(0.2, 1.5), fontsize=FONTSIZE_TICK-2,
                color=COLOR_ACCENT, fontweight='bold', style='italic',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # Decision boundary (approximate)
    x_boundary = np.linspace(-5, 6, 100)
    y_boundary = 0.6 * x_boundary + 0.3
    ax.plot(x_boundary, y_boundary, '--', color=COLOR_ACCENT, linewidth=2,
            label='Decision Boundary', alpha=0.7)

    # Styling
    ax.set_xlabel('t-SNE Dimension 1', fontsize=FONTSIZE_LABEL, color=COLOR_MAIN)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=FONTSIZE_LABEL, color=COLOR_MAIN)
    ax.set_title('BERT [CLS] Embedding Space: Sentiment Clusters',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', color=COLOR_ACCENT, pad=20)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_ACCENT)
    ax.spines['bottom'].set_color(COLOR_ACCENT)
    ax.tick_params(labelsize=FONTSIZE_TICK, colors=COLOR_MAIN)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=1)
    ax.set_facecolor('white')

    ax.legend(loc='lower right', fontsize=FONTSIZE_ANNOTATION-2)

    ax.set_xlim(-6, 7)
    ax.set_ylim(-5, 5)

    plt.tight_layout()
    plt.savefig('../../../figures/embedding_space_tsne_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print("[OK] Generated: embedding_space_tsne_bsc.pdf")
    print("     Pedagogical role: Shows BERT learns to SEPARATE positive from negative")

if __name__ == '__main__':
    create_chart()
