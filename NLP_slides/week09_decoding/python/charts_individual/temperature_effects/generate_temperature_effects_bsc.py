"""
Temperature effects - FULLY REDESIGNED with minimal fonts

Ultra-conservative font sizes (8-12pt) for proper slide readability.
Larger figure size to compensate.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# BSc Discovery colors
COLOR_MAIN = '#404040'
COLOR_ACCENT = '#3333B2'
COLOR_GRAY = '#B4B4B4'
COLOR_GREEN = '#2CA02C'

plt.style.use('seaborn-v0_8-whitegrid')

def generate_temperature_effects():
    """Clean 3-panel comparison with minimal fonts"""

    # Larger figure to make small fonts readable
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Data
    tokens = ['the', 'a', 'on', 'at', 'in']
    logits = np.array([2.0, 1.0, 0.5, 0.3, 0.1])

    temperatures = [0.5, 1.0, 2.0]
    labels = ['T=0.5 (Focused)', 'T=1.0 (Original)', 'T=2.0 (Random)']

    for ax, temp, label in zip(axes, temperatures, labels):
        # Apply temperature
        scaled_logits = logits / temp
        probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))

        # Bars
        bars = ax.bar(tokens, probs, color=COLOR_ACCENT, alpha=0.7,
                     edgecolor=COLOR_MAIN, linewidth=1.5)

        # Highlight max
        max_idx = np.argmax(probs)
        bars[max_idx].set_color(COLOR_GREEN)
        bars[max_idx].set_alpha(0.9)

        # MINIMAL FONTS
        ax.set_title(label, fontsize=12, fontweight='bold', pad=6)
        ax.set_ylim(0, 1.0)

        if ax == axes[0]:
            ax.set_ylabel('Probability', fontsize=10)

        # Probability values - very small
        for i, prob in enumerate(probs):
            ax.text(i, prob + 0.03, f'{prob:.2f}',
                   ha='center', fontsize=9,
                   color=COLOR_MAIN)

        # Minimal styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLOR_GRAY)
        ax.spines['bottom'].set_color(COLOR_GRAY)
        ax.tick_params(labelsize=9, colors=COLOR_MAIN)
        ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.8)
        ax.set_facecolor('white')

    plt.tight_layout()
    plt.savefig('../../figures/temperature_effects_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: temperature_effects_bsc.pdf (minimal fonts)")

if __name__ == "__main__":
    generate_temperature_effects()
