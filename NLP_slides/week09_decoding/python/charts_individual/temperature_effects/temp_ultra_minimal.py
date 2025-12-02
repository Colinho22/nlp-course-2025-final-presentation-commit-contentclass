"""
Temperature Effects - ULTRA MINIMAL DESIGN
Fonts: 6-8pt only (truly minimal for slide)
Figure: Large (18x6) to compensate for tiny fonts
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def create_chart():
    # LARGE figure with TINY fonts
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Data
    words = ['the', 'a', 'on', 'at', 'in']
    logits = np.array([2.0, 1.0, 0.5, 0.3, 0.1])

    temps = [0.5, 1.0, 2.0]
    titles = ['T=0.5', 'T=1.0', 'T=2.0']

    for ax, temp, title in zip([ax1, ax2, ax3], temps, titles):
        # Calculate probabilities
        scaled = logits / temp
        probs = np.exp(scaled) / np.sum(np.exp(scaled))

        # Bars
        colors = ['#2CA02C' if i == np.argmax(probs) else '#3333B2' for i in range(5)]
        ax.bar(words, probs, color=colors, alpha=0.8, edgecolor='#404040', linewidth=1)

        # TINY fonts
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.set_ylim(0, 1.0)

        if ax == ax1:
            ax.set_ylabel('P', fontsize=7)

        # Probability values - TINY
        for i, p in enumerate(probs):
            ax.text(i, p + 0.02, f'{p:.2f}', ha='center', fontsize=6, color='#404040')

        # Clean
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#B4B4B4')
        ax.spines['bottom'].set_color('#B4B4B4')
        ax.tick_params(labelsize=7, colors='#404040')
        ax.grid(True, alpha=0.1, linestyle='--', linewidth=0.5)
        ax.set_facecolor('white')

    plt.tight_layout()
    plt.savefig('../../figures/temperature_effects_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated with fonts: title=8pt, ylabel=7pt, values=6pt, ticks=7pt")

if __name__ == '__main__':
    create_chart()
