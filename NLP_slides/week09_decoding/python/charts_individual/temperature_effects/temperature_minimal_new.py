"""
Temperature Effects Chart - Brand New Design

Ultra-minimal fonts (8-10pt) for slide readability.
Simple bar chart comparison across three temperature values.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def create_temperature_chart():
    """Generate temperature effects visualization with minimal fonts."""

    # Create figure - MUCH larger size with same minimal fonts
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Sample data
    words = ['the', 'a', 'on', 'at', 'in']
    base_logits = np.array([2.0, 1.0, 0.5, 0.3, 0.1])

    # Three temperature settings
    temps = [0.5, 1.0, 2.0]
    titles = ['T=0.5', 'T=1.0', 'T=2.0']
    subtitles = ['(Focused)', '(Original)', '(Random)']

    for idx, (ax, temp, title, subtitle) in enumerate(zip([ax1, ax2, ax3], temps, titles, subtitles)):
        # Calculate probabilities with temperature
        scaled = base_logits / temp
        probs = np.exp(scaled) / np.sum(np.exp(scaled))

        # Create bars
        colors = ['#3333B2' if i != np.argmax(probs) else '#2CA02C' for i in range(len(words))]
        bars = ax.bar(words, probs, color=colors, alpha=0.8, edgecolor='#404040', linewidth=1.2)

        # Add value labels - SMALL
        for i, p in enumerate(probs):
            ax.text(i, p + 0.02, f'{p:.2f}', ha='center', va='bottom',
                   fontsize=8, color='#404040')

        # Title - SMALL
        ax.text(0.5, 1.05, title, transform=ax.transAxes,
               fontsize=10, fontweight='bold', ha='center', color='#3333B2')
        ax.text(0.5, 0.98, subtitle, transform=ax.transAxes,
               fontsize=8, ha='center', color='#666666', style='italic')

        # Axes
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Token', fontsize=9, color='#404040')

        if idx == 0:
            ax.set_ylabel('Probability', fontsize=9, color='#404040')

        # Clean style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#B4B4B4')
        ax.spines['bottom'].set_color('#B4B4B4')
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(labelsize=8, colors='#404040', width=1, length=4)
        ax.grid(True, alpha=0.12, linestyle='--', linewidth=0.6, color='#CCCCCC')
        ax.set_facecolor('white')

    plt.tight_layout()

    # Save to main figures directory
    output_path = '../../figures/temperature_effects_bsc.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Generated: {output_path}")
    print("Font sizes: titles=10pt, labels=9pt, values=8pt, ticks=8pt")

if __name__ == '__main__':
    create_temperature_chart()
