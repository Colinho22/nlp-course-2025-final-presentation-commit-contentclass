"""
Softmax Distribution Visualization for Sentiment Classification

PEDAGOGICAL PURPOSE:
- Show how logits transform to probabilities via softmax
- Compare before vs after fine-tuning confidence
- Make classification output concrete

GENUINELY NEEDS VISUALIZATION: Yes - probability distributions need bar charts
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
COLOR_ORANGE = '#FF7F0E'
COLOR_GRAY = '#B4B4B4'
COLOR_LIGHT = '#F0F0F0'

# BSc Discovery Font Standard
FONTSIZE_TITLE = 24
FONTSIZE_LABEL = 20
FONTSIZE_TICK = 16
FONTSIZE_ANNOTATION = 18

def softmax(logits):
    """Compute softmax probabilities."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def create_chart():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Example review: "Great, another boring movie" (sarcastic - actually NEGATIVE)
    labels = ['Negative', 'Positive']
    x_pos = np.arange(len(labels))
    bar_width = 0.5

    # ===============================
    # Top Left: Before fine-tuning (random)
    # ===============================
    ax1 = axes[0, 0]
    logits_before = np.array([0.1, 0.3])  # Almost uniform
    probs_before = softmax(logits_before)

    bars1 = ax1.bar(x_pos, probs_before, bar_width, color=[COLOR_RED, COLOR_GREEN], alpha=0.7,
                    edgecolor=COLOR_MAIN, linewidth=2)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=FONTSIZE_ANNOTATION)
    ax1.set_ylabel('Probability', fontsize=FONTSIZE_LABEL, color=COLOR_MAIN)
    ax1.set_title('Before Fine-Tuning\n(Random Classifier)', fontsize=FONTSIZE_LABEL,
                  fontweight='bold', color=COLOR_GRAY, pad=10)
    ax1.set_ylim(0, 1.0)

    # Add probability values
    for bar, prob in zip(bars1, probs_before):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{prob:.2f}', ha='center', fontsize=FONTSIZE_ANNOTATION, fontweight='bold')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(labelsize=FONTSIZE_TICK)
    ax1.axhline(y=0.5, color=COLOR_GRAY, linestyle='--', alpha=0.5)

    # ===============================
    # Top Right: After fine-tuning (confident NEGATIVE)
    # ===============================
    ax2 = axes[0, 1]
    logits_after = np.array([2.1, -0.5])  # Confident negative
    probs_after = softmax(logits_after)

    bars2 = ax2.bar(x_pos, probs_after, bar_width, color=[COLOR_RED, COLOR_GREEN], alpha=0.9,
                    edgecolor=COLOR_MAIN, linewidth=2)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=FONTSIZE_ANNOTATION)
    ax2.set_ylabel('Probability', fontsize=FONTSIZE_LABEL, color=COLOR_MAIN)
    ax2.set_title('After Fine-Tuning\n(Confident Classifier)', fontsize=FONTSIZE_LABEL,
                  fontweight='bold', color=COLOR_ACCENT, pad=10)
    ax2.set_ylim(0, 1.0)

    for bar, prob in zip(bars2, probs_after):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{prob:.2f}', ha='center', fontsize=FONTSIZE_ANNOTATION, fontweight='bold')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(labelsize=FONTSIZE_TICK)
    ax2.axhline(y=0.5, color=COLOR_GRAY, linestyle='--', alpha=0.5)

    # Add "CORRECT!" annotation
    ax2.annotate('CORRECT!', xy=(0, probs_after[0]), xytext=(0.3, 0.6),
                 fontsize=FONTSIZE_ANNOTATION, color=COLOR_GREEN, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=2))

    # ===============================
    # Bottom: Logits to Softmax calculation
    # ===============================
    ax3 = axes[1, 0]
    ax3.axis('off')

    # Show the calculation step-by-step
    calc_text = """
    STEP 1: Raw Logits from BERT
    $z = [z_{neg}, z_{pos}] = [2.1, -0.5]$

    STEP 2: Apply Softmax
    $P(neg) = \\frac{e^{2.1}}{e^{2.1} + e^{-0.5}} = \\frac{8.17}{8.17 + 0.61} = 0.93$

    $P(pos) = \\frac{e^{-0.5}}{e^{2.1} + e^{-0.5}} = \\frac{0.61}{8.17 + 0.61} = 0.07$

    STEP 3: Prediction
    argmax([0.93, 0.07]) = NEGATIVE
    """

    ax3.text(0.1, 0.9, 'Softmax Calculation (Worked Example)', fontsize=FONTSIZE_LABEL,
             fontweight='bold', color=COLOR_ACCENT, transform=ax3.transAxes, va='top')
    ax3.text(0.1, 0.75, 'Input: "Great, another boring movie"', fontsize=FONTSIZE_ANNOTATION,
             style='italic', color=COLOR_MAIN, transform=ax3.transAxes, va='top')

    # Calculation box
    calc_box = """Logits: [2.1, -0.5]

exp(2.1) = 8.17
exp(-0.5) = 0.61
Sum = 8.78

P(Negative) = 8.17 / 8.78 = 0.93
P(Positive) = 0.61 / 8.78 = 0.07

Prediction: NEGATIVE (93% confidence)"""

    ax3.text(0.1, 0.6, calc_box, fontsize=FONTSIZE_TICK, color=COLOR_MAIN,
             transform=ax3.transAxes, va='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT))

    # ===============================
    # Bottom Right: Cross-entropy loss calculation
    # ===============================
    ax4 = axes[1, 1]
    ax4.axis('off')

    ax4.text(0.1, 0.9, 'Cross-Entropy Loss (Worked Example)', fontsize=FONTSIZE_LABEL,
             fontweight='bold', color=COLOR_ACCENT, transform=ax4.transAxes, va='top')

    loss_box = """Ground Truth: y = [1, 0]  (Negative)
Predicted:    p = [0.93, 0.07]

Loss = -sum(y * log(p))
     = -(1 * log(0.93) + 0 * log(0.07))
     = -(-0.073 + 0)
     = 0.073

This is LOW loss (good prediction!)

If prediction was wrong:
p = [0.07, 0.93]  (predicted Positive)
Loss = -log(0.07) = 2.66  (HIGH loss!)"""

    ax4.text(0.1, 0.75, loss_box, fontsize=FONTSIZE_TICK, color=COLOR_MAIN,
             transform=ax4.transAxes, va='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT))

    # Add title for whole figure
    fig.suptitle('From Logits to Predictions: Softmax & Cross-Entropy',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', color=COLOR_ACCENT, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('../../../figures/softmax_distribution_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print("[OK] Generated: softmax_distribution_bsc.pdf")
    print("     Pedagogical role: Makes softmax/cross-entropy CONCRETE with real numbers")

if __name__ == '__main__':
    create_chart()
