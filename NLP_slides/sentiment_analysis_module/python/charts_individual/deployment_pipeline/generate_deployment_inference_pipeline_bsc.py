"""
Stage 4 Deployment: Inference Pipeline

PEDAGOGICAL PURPOSE:
- Show production inference flow from text to prediction
- Visualize batching and GPU optimization
- Make deployment concrete with timing and code

GENUINELY NEEDS VISUALIZATION: Yes - multi-step pipeline requires visual flow
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
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.2, 0.8], width_ratios=[1, 1, 1])

    # ===============================
    # Top Row: Pipeline Flow (3 panels)
    # ===============================

    # Panel 1: Input Processing
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Step 1: Input Processing', fontsize=FONTSIZE_LABEL,
                  fontweight='bold', color=COLOR_ACCENT, pad=10)

    # Raw text input
    text_box = FancyBboxPatch((0.05, 0.65), 0.9, 0.25, boxstyle="round,pad=0.01",
                             edgecolor=COLOR_GRAY, facecolor=COLOR_LIGHT, linewidth=2)
    ax1.add_patch(text_box)
    ax1.text(0.5, 0.77, 'Raw Review Text:', ha='center', va='center',
            fontsize=FONTSIZE_TICK, color=COLOR_GRAY)
    ax1.text(0.5, 0.70, '"Great movie!"', ha='center', va='center',
            fontsize=FONTSIZE_ANNOTATION, color=COLOR_MAIN, style='italic')

    # Arrow
    ax1.annotate('', xy=(0.5, 0.63), xytext=(0.5, 0.53),
                arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=2))

    # Tokenization
    token_box = FancyBboxPatch((0.05, 0.30), 0.9, 0.20, boxstyle="round,pad=0.01",
                              edgecolor=COLOR_ACCENT, facecolor='#E6E6FA', linewidth=2)
    ax1.add_patch(token_box)
    ax1.text(0.5, 0.45, 'Tokenization:', ha='center', va='center',
            fontsize=FONTSIZE_TICK, color=COLOR_ACCENT, fontweight='bold')
    ax1.text(0.5, 0.38, '[CLS] Great movie ! [SEP]', ha='center', va='center',
            fontsize=FONTSIZE_TICK-2, color=COLOR_MAIN, family='monospace')
    ax1.text(0.5, 0.32, 'IDs: [101, 2624, 3185, 999, 102]', ha='center', va='center',
            fontsize=FONTSIZE_TICK-2, color=COLOR_ACCENT, family='monospace')

    # Timing
    ax1.text(0.5, 0.12, 'Timing: ~1ms', ha='center', va='center',
            fontsize=FONTSIZE_TICK, color=COLOR_GREEN, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#E6FFE6', edgecolor=COLOR_GREEN))

    # Panel 2: Model Inference
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Step 2: Model Inference', fontsize=FONTSIZE_LABEL,
                  fontweight='bold', color=COLOR_ACCENT, pad=10)

    # BERT processing
    bert_box = FancyBboxPatch((0.15, 0.40), 0.7, 0.45, boxstyle="round,pad=0.01",
                             edgecolor=COLOR_ACCENT, facecolor='#E6E6FA', linewidth=3)
    ax2.add_patch(bert_box)

    ax2.text(0.5, 0.75, 'Fine-tuned BERT', ha='center', va='center',
            fontsize=FONTSIZE_ANNOTATION, color=COLOR_ACCENT, fontweight='bold')
    ax2.text(0.5, 0.65, '+ Classifier Head', ha='center', va='center',
            fontsize=FONTSIZE_TICK, color=COLOR_ORANGE)

    # Processing steps
    steps = ['Embeddings', 'Layers 1-12', 'Pooling', 'Linear']
    for i, step in enumerate(steps):
        y = 0.58 - i * 0.08
        ax2.text(0.5, y, step, ha='center', va='center',
                fontsize=FONTSIZE_TICK-2, color=COLOR_ACCENT)

    # GPU note
    ax2.text(0.5, 0.30, 'GPU Accelerated', ha='center', va='center',
            fontsize=FONTSIZE_TICK, color=COLOR_ORANGE, fontweight='bold')

    # Timing
    ax2.text(0.5, 0.12, 'Timing: ~10-50ms', ha='center', va='center',
            fontsize=FONTSIZE_TICK, color=COLOR_ORANGE, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFE6CC', edgecolor=COLOR_ORANGE))

    # Panel 3: Output Processing
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Step 3: Output Processing', fontsize=FONTSIZE_LABEL,
                  fontweight='bold', color=COLOR_ACCENT, pad=10)

    # Logits
    logit_box = FancyBboxPatch((0.05, 0.70), 0.9, 0.18, boxstyle="round,pad=0.01",
                              edgecolor=COLOR_BLUE, facecolor='#E6F2FF', linewidth=2)
    ax3.add_patch(logit_box)
    ax3.text(0.5, 0.83, 'Raw Logits:', ha='center', va='center',
            fontsize=FONTSIZE_TICK, color=COLOR_BLUE)
    ax3.text(0.5, 0.75, '[-0.8, 2.3]', ha='center', va='center',
            fontsize=FONTSIZE_ANNOTATION, color=COLOR_BLUE, family='monospace', fontweight='bold')

    # Arrow
    ax3.annotate('', xy=(0.5, 0.68), xytext=(0.5, 0.60),
                arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=2))

    # Softmax
    prob_box = FancyBboxPatch((0.05, 0.40), 0.9, 0.18, boxstyle="round,pad=0.01",
                             edgecolor=COLOR_GREEN, facecolor='#E6FFE6', linewidth=2)
    ax3.add_patch(prob_box)
    ax3.text(0.5, 0.53, 'Probabilities:', ha='center', va='center',
            fontsize=FONTSIZE_TICK, color=COLOR_GREEN)
    ax3.text(0.5, 0.45, '[0.09, 0.91]', ha='center', va='center',
            fontsize=FONTSIZE_ANNOTATION, color=COLOR_GREEN, family='monospace', fontweight='bold')

    # Final prediction
    pred_box = FancyBboxPatch((0.1, 0.20), 0.8, 0.15, boxstyle="round,pad=0.01",
                             edgecolor=COLOR_GREEN, facecolor='#CCFFCC', linewidth=2)
    ax3.add_patch(pred_box)
    ax3.text(0.5, 0.30, 'Prediction: POSITIVE', ha='center', va='center',
            fontsize=FONTSIZE_ANNOTATION, color=COLOR_GREEN, fontweight='bold')
    ax3.text(0.5, 0.24, 'Confidence: 91%', ha='center', va='center',
            fontsize=FONTSIZE_TICK, color=COLOR_GREEN)

    # Timing
    ax3.text(0.5, 0.05, 'Timing: <1ms', ha='center', va='center',
            fontsize=FONTSIZE_TICK, color=COLOR_GREEN, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#E6FFE6', edgecolor=COLOR_GREEN))

    # ===============================
    # Middle Row: Batching Optimization
    # ===============================
    ax_batch = fig.add_subplot(gs[1, :])
    ax_batch.set_xlim(0, 1)
    ax_batch.set_ylim(0, 1)
    ax_batch.axis('off')
    ax_batch.set_title('Production Optimization: Batch Processing',
                       fontsize=FONTSIZE_LABEL, fontweight='bold', color=COLOR_ORANGE, pad=10)

    # Sequential processing (inefficient)
    ax_batch.text(0.15, 0.90, 'Sequential (Slow):', ha='center', va='top',
                  fontsize=FONTSIZE_ANNOTATION, color=COLOR_RED, fontweight='bold')

    for i in range(3):
        y = 0.80 - i * 0.18
        seq_box = FancyBboxPatch((0.05, y-0.06), 0.20, 0.12, boxstyle="round,pad=0.005",
                                edgecolor=COLOR_RED, facecolor=COLOR_LIGHT, linewidth=1.5)
        ax_batch.add_patch(seq_box)
        ax_batch.text(0.15, y, f'Review {i+1}', ha='center', va='center',
                     fontsize=FONTSIZE_TICK, color=COLOR_MAIN)

        # Arrow to model
        ax_batch.annotate('', xy=(0.27, y), xytext=(0.25, y),
                         arrowprops=dict(arrowstyle='->', color=COLOR_RED, lw=1.5))

        # Model box
        model_box = FancyBboxPatch((0.27, y-0.04), 0.08, 0.08, boxstyle="round,pad=0.003",
                                  edgecolor=COLOR_RED, facecolor='#FFE6E6', linewidth=1)
        ax_batch.add_patch(model_box)
        ax_batch.text(0.31, y, 'BERT', ha='center', va='center',
                     fontsize=FONTSIZE_TICK-4, color=COLOR_RED)

        # Timing
        ax_batch.text(0.38, y, f'~50ms', ha='left', va='center',
                     fontsize=FONTSIZE_TICK-2, color=COLOR_RED, family='monospace')

    ax_batch.text(0.15, 0.25, 'Total: ~150ms', ha='center', va='center',
                 fontsize=FONTSIZE_ANNOTATION, color=COLOR_RED, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='#FFE6E6', edgecolor=COLOR_RED))

    # Batch processing (efficient)
    ax_batch.text(0.70, 0.90, 'Batch Processing (Fast):', ha='center', va='top',
                  fontsize=FONTSIZE_ANNOTATION, color=COLOR_GREEN, fontweight='bold')

    # Three reviews stacked
    for i in range(3):
        y = 0.80 - i * 0.10
        batch_box = FancyBboxPatch((0.52, y-0.03), 0.20, 0.06, boxstyle="round,pad=0.003",
                                  edgecolor=COLOR_GREEN, facecolor=COLOR_LIGHT, linewidth=1)
        ax_batch.add_patch(batch_box)
        ax_batch.text(0.62, y, f'Review {i+1}', ha='center', va='center',
                     fontsize=FONTSIZE_TICK-2, color=COLOR_MAIN)

    # Batch arrow
    ax_batch.annotate('', xy=(0.75, 0.65), xytext=(0.72, 0.65),
                     arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=3))

    # Single model box
    large_model = FancyBboxPatch((0.75, 0.53), 0.12, 0.24, boxstyle="round,pad=0.005",
                                edgecolor=COLOR_GREEN, facecolor='#E6FFE6', linewidth=2)
    ax_batch.add_patch(large_model)
    ax_batch.text(0.81, 0.65, 'BERT\n(Batch=3)', ha='center', va='center',
                 fontsize=FONTSIZE_TICK, color=COLOR_GREEN, fontweight='bold')

    # Timing
    ax_batch.text(0.90, 0.65, '~60ms', ha='left', va='center',
                 fontsize=FONTSIZE_ANNOTATION, color=COLOR_GREEN, family='monospace', fontweight='bold')

    ax_batch.text(0.70, 0.25, 'Total: ~60ms (2.5x faster!)', ha='center', va='center',
                 fontsize=FONTSIZE_ANNOTATION, color=COLOR_GREEN, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='#E6FFE6', edgecolor=COLOR_GREEN))

    # ===============================
    # Bottom Row: Code Examples
    # ===============================
    ax_code = fig.add_subplot(gs[2, :])
    ax_code.set_xlim(0, 1)
    ax_code.set_ylim(0, 1)
    ax_code.axis('off')
    ax_code.set_title('Production Code Example (Hugging Face Transformers)',
                      fontsize=FONTSIZE_LABEL, fontweight='bold', color=COLOR_ACCENT, pad=10)

    code_text = """from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("my-sentiment-model")
model = AutoModelForSequenceClassification.from_pretrained("my-sentiment-model")
model.eval()  # Set to inference mode
model.to("cuda")  # Move to GPU

# Batch inference
reviews = ["Great movie!", "Terrible acting", "Loved it!"]
inputs = tokenizer(reviews, padding=True, truncation=True, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}  # Move to GPU

with torch.no_grad():  # Disable gradient computation
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1)
    labels = torch.argmax(predictions, dim=1)

# Results: labels = [1, 0, 1] (POSITIVE, NEGATIVE, POSITIVE)
"""

    ax_code.text(0.02, 0.95, code_text, ha='left', va='top',
                fontsize=FONTSIZE_TICK-4, color=COLOR_MAIN, family='monospace',
                bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT))

    # Overall title
    fig.suptitle('Stage 4: Production Deployment and Inference',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', color=COLOR_ACCENT, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('../../../figures/deployment_inference_pipeline_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print("[OK] Generated: deployment_inference_pipeline_bsc.pdf")
    print("     Pedagogical role: Shows production inference flow with batching and code")

if __name__ == '__main__':
    create_chart()
