"""
Stage 1 Pre-training: MLM and NSP Objectives

PEDAGOGICAL PURPOSE:
- Show how BERT learns language understanding from raw text
- Visualize MLM (Masked Language Model) training objective
- Make pre-training concrete with worked example

GENUINELY NEEDS VISUALIZATION: Yes - multi-step process requires visual flow
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# BSc Discovery Color Scheme
COLOR_MAIN = '#404040'
COLOR_ACCENT = '#3333B2'
COLOR_GRAY = '#B4B4B4'
COLOR_GREEN = '#2CA02C'
COLOR_RED = '#D62728'
COLOR_ORANGE = '#FF7F0E'
COLOR_LIGHT = '#F0F0F0'
COLOR_BLUE = '#0066CC'

# BSc Discovery Font Standard
FONTSIZE_TITLE = 24
FONTSIZE_LABEL = 20
FONTSIZE_TICK = 16
FONTSIZE_ANNOTATION = 18
FONTSIZE_TEXT = 20
FONTSIZE_SMALL = 18

def create_chart():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ===============================
    # Panel 1: Input with Masked Tokens
    # ===============================
    ax1 = axes[0]
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Step 1: Mask 15% of Tokens', fontsize=FONTSIZE_LABEL,
                  fontweight='bold', color=COLOR_ACCENT, pad=15)

    # Original sentence
    ax1.text(0.5, 0.85, 'Original Sentence:', ha='center', va='top',
             fontsize=FONTSIZE_ANNOTATION, color=COLOR_GRAY)

    original_box = FancyBboxPatch((0.05, 0.70), 0.9, 0.10, boxstyle="round,pad=0.01",
                                  edgecolor=COLOR_GRAY, facecolor=COLOR_LIGHT, linewidth=2)
    ax1.add_patch(original_box)
    ax1.text(0.5, 0.75, '"The movie was fantastic"', ha='center', va='center',
             fontsize=FONTSIZE_ANNOTATION, color=COLOR_MAIN, style='italic')

    # Arrow down
    ax1.annotate('', xy=(0.5, 0.68), xytext=(0.5, 0.58),
                arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=3))

    # Masked sentence
    ax1.text(0.5, 0.52, 'Masked Input (15% masked):', ha='center', va='top',
             fontsize=FONTSIZE_ANNOTATION, color=COLOR_ACCENT, fontweight='bold')

    masked_box = FancyBboxPatch((0.05, 0.35), 0.9, 0.12, boxstyle="round,pad=0.01",
                                edgecolor=COLOR_ACCENT, facecolor='#E6E6FA', linewidth=2)
    ax1.add_patch(masked_box)

    # Show tokens with one masked
    tokens = ['The', 'movie', '[MASK]', 'fantastic']
    x_positions = [0.18, 0.35, 0.52, 0.72]
    for i, (token, x) in enumerate(zip(tokens, x_positions)):
        if token == '[MASK]':
            color = COLOR_RED
            weight = 'bold'
            box_color = '#FFE6E6'
        else:
            color = COLOR_MAIN
            weight = 'normal'
            box_color = COLOR_LIGHT

        token_box = FancyBboxPatch((x-0.06, 0.37), 0.12, 0.08, boxstyle="round,pad=0.005",
                                   edgecolor=color, facecolor=box_color, linewidth=1.5)
        ax1.add_patch(token_box)
        ax1.text(x, 0.41, token, ha='center', va='center',
                fontsize=FONTSIZE_TICK, color=color, fontweight=weight)

    # Bottom note
    ax1.text(0.5, 0.15, 'BERT must predict\nmasked words from context',
             ha='center', va='center', fontsize=FONTSIZE_ANNOTATION,
             color=COLOR_ACCENT, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT))

    # ===============================
    # Panel 2: BERT Processing
    # ===============================
    ax2 = axes[1]
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Step 2: BERT Processes Input', fontsize=FONTSIZE_LABEL,
                  fontweight='bold', color=COLOR_ACCENT, pad=15)

    # Input layer
    input_box = FancyBboxPatch((0.1, 0.80), 0.8, 0.10, boxstyle="round,pad=0.01",
                               edgecolor=COLOR_ACCENT, facecolor=COLOR_LIGHT, linewidth=2)
    ax2.add_patch(input_box)
    ax2.text(0.5, 0.85, 'Token Embeddings', ha='center', va='center',
            fontsize=FONTSIZE_ANNOTATION, color=COLOR_ACCENT)

    # Transformer layers
    for i, y in enumerate([0.65, 0.50, 0.35]):
        layer_box = FancyBboxPatch((0.1, y-0.05), 0.8, 0.10, boxstyle="round,pad=0.01",
                                   edgecolor=COLOR_ACCENT, facecolor='#E6E6FA', linewidth=2)
        ax2.add_patch(layer_box)
        ax2.text(0.5, y, f'Transformer Layer {i+1}', ha='center', va='center',
                fontsize=FONTSIZE_ANNOTATION-2, color=COLOR_ACCENT)

        # Arrow to next layer
        if y > 0.4:
            ax2.annotate('', xy=(0.5, y-0.07), xytext=(0.5, y-0.13),
                        arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=2))

    # Output layer
    output_box = FancyBboxPatch((0.1, 0.15), 0.8, 0.10, boxstyle="round,pad=0.01",
                                edgecolor=COLOR_GREEN, facecolor='#E6FFE6', linewidth=2)
    ax2.add_patch(output_box)
    ax2.text(0.5, 0.20, 'Output Embeddings', ha='center', va='center',
            fontsize=FONTSIZE_ANNOTATION, color=COLOR_GREEN, fontweight='bold')

    # Side annotation
    ax2.text(0.02, 0.5, 'Bidirectional\nContext', ha='left', va='center',
            fontsize=FONTSIZE_TICK, color=COLOR_ACCENT, rotation=90,
            fontweight='bold')

    # ===============================
    # Panel 3: Predictions
    # ===============================
    ax3 = axes[2]
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Step 3: Predict Masked Token', fontsize=FONTSIZE_LABEL,
                  fontweight='bold', color=COLOR_ACCENT, pad=15)

    # Prediction header
    ax3.text(0.5, 0.85, 'Top Predictions for [MASK]:', ha='center', va='top',
             fontsize=FONTSIZE_ANNOTATION, color=COLOR_ACCENT, fontweight='bold')

    # Prediction bars
    predictions = [
        ('was', 0.85, COLOR_GREEN),
        ('is', 0.08, COLOR_ORANGE),
        ('seems', 0.04, COLOR_GRAY),
        ('looked', 0.02, COLOR_GRAY),
        ('other', 0.01, COLOR_GRAY)
    ]

    y_start = 0.70
    for i, (word, prob, color) in enumerate(predictions):
        y = y_start - i * 0.12

        # Probability bar
        bar_width = prob * 0.7
        bar = FancyBboxPatch((0.15, y-0.03), bar_width, 0.06, boxstyle="round,pad=0.002",
                            edgecolor=color, facecolor=color, alpha=0.7, linewidth=1)
        ax3.add_patch(bar)

        # Word label
        ax3.text(0.12, y, word, ha='right', va='center',
                fontsize=FONTSIZE_ANNOTATION, color=COLOR_MAIN, fontweight='bold')

        # Probability value
        ax3.text(0.88, y, f'{prob:.2f}', ha='left', va='center',
                fontsize=FONTSIZE_ANNOTATION, color=color, fontweight='bold')

        # Checkmark for correct prediction
        if i == 0:
            ax3.text(0.95, y, '✓', ha='center', va='center',
                    fontsize=FONTSIZE_TITLE, color=COLOR_GREEN, fontweight='bold')

    # Bottom note
    ax3.text(0.5, 0.08, 'Loss = -log(0.85) = 0.16\n(Low loss = good prediction)',
             ha='center', va='center', fontsize=FONTSIZE_ANNOTATION,
             color=COLOR_GREEN, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#E6FFE6', edgecolor=COLOR_GREEN))

    # Overall title
    fig.suptitle('Stage 1: Pre-training with Masked Language Model (MLM)',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', color=COLOR_ACCENT, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('../../../figures/pretrained_mlm_objectives_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print("[OK] Generated: pretrained_mlm_objectives_bsc.pdf")
    print("     Pedagogical role: Shows MLM training objective with concrete example")

if __name__ == '__main__':
    create_chart()
