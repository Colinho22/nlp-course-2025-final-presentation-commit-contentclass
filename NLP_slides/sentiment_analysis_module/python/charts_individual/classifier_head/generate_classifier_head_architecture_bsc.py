"""
Stage 2 Classifier Head Architecture

PEDAGOGICAL PURPOSE:
- Show how [CLS] token embedding becomes sentiment prediction
- Visualize linear layer dimensions and weight matrix
- Make classifier head concrete with tensor shapes

GENUINELY NEEDS VISUALIZATION: Yes - architecture requires visual flow diagram
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
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
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], width_ratios=[1, 1])

    # ===============================
    # Top: Main Architecture Flow
    # ===============================
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.axis('off')

    # Stage 1: Input tokens
    ax_main.text(0.08, 0.90, 'Input Tokens', ha='center', va='top',
                 fontsize=FONTSIZE_ANNOTATION, color=COLOR_GRAY)

    tokens = ['[CLS]', 'Great', 'movie']
    for i, token in enumerate(tokens):
        y = 0.78 - i * 0.18
        token_box = FancyBboxPatch((0.02, y-0.05), 0.12, 0.10, boxstyle="round,pad=0.005",
                                   edgecolor=COLOR_GRAY, facecolor=COLOR_LIGHT, linewidth=1.5)
        ax_main.add_patch(token_box)
        ax_main.text(0.08, y, token, ha='center', va='center',
                    fontsize=FONTSIZE_ANNOTATION-2, color=COLOR_MAIN)

    # Arrow to BERT
    ax_main.annotate('', xy=(0.16, 0.55), xytext=(0.14, 0.55),
                    arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=3))

    # Stage 2: BERT
    bert_box = FancyBboxPatch((0.16, 0.35), 0.15, 0.40, boxstyle="round,pad=0.01",
                             edgecolor=COLOR_ACCENT, facecolor='#E6E6FA', linewidth=3)
    ax_main.add_patch(bert_box)
    ax_main.text(0.235, 0.55, 'BERT\nEncoder', ha='center', va='center',
                fontsize=FONTSIZE_ANNOTATION, color=COLOR_ACCENT, fontweight='bold')

    # Stage 3: [CLS] output
    ax_main.text(0.38, 0.90, '[CLS] Embedding', ha='center', va='top',
                 fontsize=FONTSIZE_ANNOTATION, color=COLOR_ACCENT, fontweight='bold')

    cls_box = FancyBboxPatch((0.33, 0.40), 0.10, 0.35, boxstyle="round,pad=0.01",
                            edgecolor=COLOR_ACCENT, facecolor='#E6FFE6', linewidth=2)
    ax_main.add_patch(cls_box)

    # Show vector elements
    for i in range(5):
        y = 0.70 - i * 0.06
        ax_main.text(0.38, y, f'h[{i}]', ha='center', va='center',
                    fontsize=FONTSIZE_TICK-2, color=COLOR_ACCENT, family='monospace')
    ax_main.text(0.38, 0.40, '...', ha='center', va='center',
                fontsize=FONTSIZE_TICK, color=COLOR_ACCENT)

    # Dimension annotation
    ax_main.text(0.38, 0.32, 'Shape: [768]', ha='center', va='center',
                fontsize=FONTSIZE_TICK, color=COLOR_ACCENT, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT))

    # Arrow from BERT to [CLS]
    ax_main.annotate('', xy=(0.33, 0.55), xytext=(0.31, 0.55),
                    arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=3))

    # Stage 4: Linear Layer (Weight Matrix)
    ax_main.text(0.56, 0.90, 'Linear Layer', ha='center', va='top',
                 fontsize=FONTSIZE_ANNOTATION, color=COLOR_ORANGE, fontweight='bold')

    # Weight matrix visualization
    matrix_box = FancyBboxPatch((0.48, 0.40), 0.16, 0.35, boxstyle="round,pad=0.01",
                               edgecolor=COLOR_ORANGE, facecolor='#FFE6CC', linewidth=2)
    ax_main.add_patch(matrix_box)

    # Show weight matrix grid (simplified)
    for i in range(4):
        for j in range(6):
            cell = Rectangle((0.50 + j*0.02, 0.68 - i*0.06), 0.018, 0.05,
                           edgecolor=COLOR_ORANGE, facecolor='white', linewidth=0.5)
            ax_main.add_patch(cell)

    ax_main.text(0.56, 0.45, 'W', ha='center', va='center',
                fontsize=FONTSIZE_TITLE, color=COLOR_ORANGE, fontweight='bold', style='italic')

    # Dimension annotation
    ax_main.text(0.56, 0.32, 'Shape: [768, 2]', ha='center', va='center',
                fontsize=FONTSIZE_TICK, color=COLOR_ORANGE, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ORANGE))

    # Arrow to linear layer
    ax_main.annotate('', xy=(0.48, 0.55), xytext=(0.43, 0.55),
                    arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=3))

    # Stage 5: Logits
    ax_main.text(0.74, 0.90, 'Logits', ha='center', va='top',
                 fontsize=FONTSIZE_ANNOTATION, color=COLOR_BLUE, fontweight='bold')

    logits_box = FancyBboxPatch((0.69, 0.50), 0.10, 0.25, boxstyle="round,pad=0.01",
                               edgecolor=COLOR_BLUE, facecolor='#E6F2FF', linewidth=2)
    ax_main.add_patch(logits_box)

    ax_main.text(0.74, 0.67, 'z[0] = 2.1', ha='center', va='center',
                fontsize=FONTSIZE_ANNOTATION-2, color=COLOR_RED, family='monospace', fontweight='bold')
    ax_main.text(0.74, 0.57, 'z[1] = -0.5', ha='center', va='center',
                fontsize=FONTSIZE_ANNOTATION-2, color=COLOR_GREEN, family='monospace', fontweight='bold')

    ax_main.text(0.74, 0.42, 'Shape: [2]', ha='center', va='center',
                fontsize=FONTSIZE_TICK, color=COLOR_BLUE, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_BLUE))

    # Arrow to logits
    ax_main.annotate('', xy=(0.69, 0.60), xytext=(0.64, 0.60),
                    arrowprops=dict(arrowstyle='->', color=COLOR_BLUE, lw=3))

    # Stage 6: Softmax
    ax_main.text(0.88, 0.90, 'Softmax', ha='center', va='top',
                 fontsize=FONTSIZE_ANNOTATION, color=COLOR_GREEN, fontweight='bold')

    probs_box = FancyBboxPatch((0.83, 0.50), 0.10, 0.25, boxstyle="round,pad=0.01",
                              edgecolor=COLOR_GREEN, facecolor='#E6FFE6', linewidth=2)
    ax_main.add_patch(probs_box)

    ax_main.text(0.88, 0.67, 'p[0] = 0.93', ha='center', va='center',
                fontsize=FONTSIZE_ANNOTATION-2, color=COLOR_RED, family='monospace', fontweight='bold')
    ax_main.text(0.88, 0.57, 'p[1] = 0.07', ha='center', va='center',
                fontsize=FONTSIZE_ANNOTATION-2, color=COLOR_GREEN, family='monospace', fontweight='bold')

    ax_main.text(0.88, 0.42, 'Prediction:\nNEGATIVE', ha='center', va='center',
                fontsize=FONTSIZE_TICK, color=COLOR_RED, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#FFE6E6', edgecolor=COLOR_RED))

    # Arrow to softmax
    ax_main.annotate('', xy=(0.83, 0.60), xytext=(0.79, 0.60),
                    arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=3))

    # ===============================
    # Bottom Left: Matrix Multiplication Math
    # ===============================
    ax_math = fig.add_subplot(gs[1, 0])
    ax_math.set_xlim(0, 1)
    ax_math.set_ylim(0, 1)
    ax_math.axis('off')
    ax_math.set_title('Matrix Multiplication Details', fontsize=FONTSIZE_LABEL,
                      fontweight='bold', color=COLOR_ACCENT, pad=10)

    math_text = """Linear Layer Computation:

z = W^T · h + b

Where:
  h = [CLS] embedding (768 dimensions)
  W = Weight matrix (768 × 2)
  b = Bias vector (2 dimensions)
  z = Output logits (2 dimensions)

Example (simplified to 3D):
  h = [0.5, -0.2, 0.8]^T
  W = [[0.3, -0.1],
       [0.2,  0.4],
       [-0.1, 0.5]]
  b = [0.1, -0.05]

  z = [0.3*0.5 + 0.2*(-0.2) + (-0.1)*0.8,
       -0.1*0.5 + 0.4*(-0.2) + 0.5*0.8] + b
    = [0.03, 0.27] + [0.1, -0.05]
    = [0.13, 0.22]
"""

    ax_math.text(0.05, 0.95, math_text, ha='left', va='top',
                fontsize=FONTSIZE_TICK-2, color=COLOR_MAIN, family='monospace',
                bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT))

    # ===============================
    # Bottom Right: Initialization Details
    # ===============================
    ax_init = fig.add_subplot(gs[1, 1])
    ax_init.set_xlim(0, 1)
    ax_init.set_ylim(0, 1)
    ax_init.axis('off')
    ax_init.set_title('Initialization Strategy', fontsize=FONTSIZE_LABEL,
                      fontweight='bold', color=COLOR_ORANGE, pad=10)

    init_text = """Classifier Head Initialization:

BERT Layers (Stage 1):
  - Load pre-trained weights
  - Already optimized on Wikipedia
  - Frozen or fine-tuned slowly

Linear Layer (Stage 2):
  - Random initialization
  - Xavier/Glorot uniform:
      W ~ U(-√(6/(768+2)), √(6/(768+2)))
  - Bias initialized to zeros: b = [0, 0]

Why Random Init?:
  - No prior knowledge of task
  - Fine-tuning will adapt to sentiment
  - Fast convergence (3-5 epochs)
"""

    ax_init.text(0.05, 0.95, init_text, ha='left', va='top',
                fontsize=FONTSIZE_TICK-2, color=COLOR_MAIN, family='monospace',
                bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ORANGE))

    # Overall title
    fig.suptitle('Stage 2: Adding Classifier Head to Pre-trained BERT',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', color=COLOR_ACCENT, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('../../../figures/classifier_head_architecture_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print("[OK] Generated: classifier_head_architecture_bsc.pdf")
    print("     Pedagogical role: Shows [CLS] -> Linear -> Softmax with tensor shapes")

if __name__ == '__main__':
    create_chart()
