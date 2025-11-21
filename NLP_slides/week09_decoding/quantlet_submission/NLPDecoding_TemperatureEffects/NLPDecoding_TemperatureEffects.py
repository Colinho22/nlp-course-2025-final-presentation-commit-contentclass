"""
Generate temperature effect visualization showing how probability distribution changes
Shows 5 temperature settings side-by-side with entropy indicators
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_temperature_animation():
    """Create 5-panel visualization of temperature effects on probability distribution."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))

    # Base logits (unnormalized scores)
    words = ['nice', 'beautiful', 'perfect', 'gorgeous', 'mild', 'sunny', 'awful']
    base_logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, -1.0])

    temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]
    titles = ['T=0.1\n(Very Focused)', 'T=0.5\n(Focused)',
              'T=1.0\n(Original)', 'T=1.5\n(Creative)', 'T=2.0\n(Very Random)']

    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#17becf', '#9467bd']

    for ax, temp, title, color in zip(axes, temperatures, titles, colors):
        # Apply temperature
        scaled_logits = base_logits / temp
        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # numerical stability
        probs = exp_logits / np.sum(exp_logits)

        # Create bars
        bars = ax.bar(range(len(words)), probs, color=color, alpha=0.7,
                     edgecolor='black', linewidth=1.5)

        # Add probability values on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            if prob > 0.02:  # Only show if visible
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.3f}', ha='center', va='bottom', fontsize=8,
                       fontweight='bold')

        # Calculate entropy (measure of randomness)
        # H = -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Styling
        ax.set_ylim(0, max(1.0, max(probs) * 1.15))
        ax.set_ylabel('Probability', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', color=color)
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Add entropy indicator
        entropy_text = f'Entropy: {entropy:.2f}'
        ax.text(0.5, 0.95, entropy_text, transform=ax.transAxes,
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3,
                         edgecolor='black', linewidth=1.5))

        # Add variance indicator (standard deviation)
        std_dev = np.std(probs)
        if temp == 0.1:
            ax.text(0.5, 0.85, 'Sharp\nPeak', transform=ax.transAxes,
                   ha='center', fontsize=8, style='italic')
        elif temp == 2.0:
            ax.text(0.5, 0.85, 'Flat\nDistribution', transform=ax.transAxes,
                   ha='center', fontsize=8, style='italic')

    # Overall title
    fig.suptitle('Temperature Controls Probability Distribution Shape',
                fontsize=16, fontweight='bold', y=1.02)

    # Add explanation at bottom
    fig.text(0.5, -0.05,
            'Low T: Model is confident (picks top word) | High T: Model is uncertain (spreads probability)',
            ha='center', fontsize=11, style='italic')

    # Add entropy explanation
    fig.text(0.5, -0.10,
            'Entropy measures randomness: Low entropy = predictable, High entropy = random',
            ha='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig('./probability_distribution_animation.pdf',
                dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print("Generated probability_distribution_animation.pdf")


if __name__ == "__main__":
    create_temperature_animation()
