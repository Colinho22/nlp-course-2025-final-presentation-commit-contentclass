"""
Loss Landscape 3D Visualization for Sentiment Analysis

PEDAGOGICAL PURPOSE:
- Visualize how cross-entropy loss varies with model parameters
- Show why gradient descent finds good solutions
- Make abstract optimization concept concrete

GENUINELY NEEDS VISUALIZATION: Yes - 3D surface cannot be expressed in text
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# BSc Discovery Color Scheme
COLOR_MAIN = '#404040'
COLOR_ACCENT = '#3333B2'
COLOR_GREEN = '#2CA02C'
COLOR_RED = '#D62728'

# BSc Discovery Font Standard
FONTSIZE_TITLE = 24
FONTSIZE_LABEL = 20
FONTSIZE_TICK = 16
FONTSIZE_ANNOTATION = 18

def cross_entropy_surface(w1, w2):
    """Simulated cross-entropy loss landscape with a clear minimum."""
    # Create a bowl-shaped surface with some local minima
    base = 0.5 * (w1**2 + w2**2)  # Quadratic bowl
    noise = 0.1 * np.sin(3*w1) * np.cos(3*w2)  # Small undulations
    return base + noise + 0.5  # Shift up for positive loss

def create_chart():
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create mesh grid
    w1 = np.linspace(-2, 2, 100)
    w2 = np.linspace(-2, 2, 100)
    W1, W2 = np.meshgrid(w1, w2)
    Loss = cross_entropy_surface(W1, W2)

    # Plot surface
    surf = ax.plot_surface(W1, W2, Loss, cmap='viridis', alpha=0.8,
                           linewidth=0, antialiased=True)

    # Mark the global minimum
    min_idx = np.unravel_index(np.argmin(Loss), Loss.shape)
    min_w1, min_w2 = W1[min_idx], W2[min_idx]
    min_loss = Loss[min_idx]

    ax.scatter([min_w1], [min_w2], [min_loss], color=COLOR_GREEN, s=200,
               marker='*', label='Global Minimum', zorder=5)

    # Add gradient descent path (simulated)
    path_w1 = np.array([1.8, 1.4, 1.0, 0.6, 0.3, 0.1, 0.0])
    path_w2 = np.array([1.5, 1.2, 0.9, 0.6, 0.3, 0.1, 0.0])
    path_loss = cross_entropy_surface(path_w1, path_w2)

    ax.plot(path_w1, path_w2, path_loss, color=COLOR_RED, linewidth=3,
            marker='o', markersize=8, label='Gradient Descent Path')

    # Labels
    ax.set_xlabel('Weight 1 Perturbation', fontsize=FONTSIZE_LABEL,
                  color=COLOR_MAIN, labelpad=10)
    ax.set_ylabel('Weight 2 Perturbation', fontsize=FONTSIZE_LABEL,
                  color=COLOR_MAIN, labelpad=10)
    ax.set_zlabel('Cross-Entropy Loss', fontsize=FONTSIZE_LABEL,
                  color=COLOR_MAIN, labelpad=10)

    ax.set_title('Loss Landscape: Fine-Tuning BERT for Sentiment',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', color=COLOR_ACCENT, pad=20)

    # Tick sizes
    ax.tick_params(labelsize=FONTSIZE_TICK)

    # Legend
    ax.legend(loc='upper right', fontsize=FONTSIZE_ANNOTATION-2)

    # View angle
    ax.view_init(elev=25, azim=45)

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Loss Value', fontsize=FONTSIZE_ANNOTATION, color=COLOR_MAIN)
    cbar.ax.tick_params(labelsize=FONTSIZE_TICK)

    plt.tight_layout()
    plt.savefig('../../../figures/loss_landscape_3d_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print("[OK] Generated: loss_landscape_3d_bsc.pdf")
    print("     Pedagogical role: Shows WHY gradient descent finds good solutions")

if __name__ == '__main__':
    create_chart()
