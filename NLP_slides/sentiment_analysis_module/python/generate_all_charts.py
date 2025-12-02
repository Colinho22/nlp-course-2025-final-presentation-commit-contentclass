"""
Generate all sentiment analysis charts with pedagogical documentation.

This master script creates all 5 charts for the sentiment analysis module.
Each chart has a specific pedagogical role in the Mystery/Investigation narrative.
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

def chart2_architecture_comparison():
    """
    Chart 2: Architecture Comparison (The Dead End)

    PEDAGOGICAL ROLE: Show structural limitation of traditional methods
    PREVENTS: "Just use TF-IDF" resistance
    SCAFFOLDS: From understanding problem to need for new architecture
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Dead End: Traditional Methods Can't Solve Our Puzzle",
                 fontsize=FONTSIZE_TITLE, fontweight='bold', color=COLOR_RED)

    # Left: Traditional approach
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Traditional: BOW + SVM', fontsize=FONTSIZE_LABEL,
                 fontweight='bold', color=COLOR_GRAY, pad=20)

    # Pipeline stages
    stages = [
        ('Input:\\n"Great, another\\nboring movie"', 0.85, COLOR_MAIN),
        ('Word Counts:\\nGreat=1, another=1,\\nboring=1, movie=1', 0.65, COLOR_GRAY),
        ('Feature Vector:\\n[1, 1, 1, 1, ...]', 0.45, COLOR_GRAY),
        ('SVM Classifier', 0.25, COLOR_ORANGE),
        ('Output: POSITIVE\\n(WRONG!)', 0.05, COLOR_RED)
    ]

    for text, y, color in stages:
        box = FancyBboxPatch((0.1, y-0.06), 0.8, 0.12, boxstyle="round,pad=0.01",
                            edgecolor=color, facecolor=COLOR_LIGHT, linewidth=2)
        ax1.add_patch(box)
        ax1.text(0.5, y, text, ha='center', va='center',
                fontsize=FONTSIZE_ANNOTATION-2, color=color)
        if y > 0.1:
            ax1.annotate('', xy=(0.5, y-0.08), xytext=(0.5, y-0.18),
                        arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=2))

    # Problem annotation
    ax1.text(0.5, -0.05, 'No word order, no context!', ha='center', va='top',
            fontsize=FONTSIZE_ANNOTATION, color=COLOR_RED, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFE6E6', edgecolor=COLOR_RED))

    # Right: BERT approach
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Breakthrough: BERT', fontsize=FONTSIZE_LABEL,
                 fontweight='bold', color=COLOR_GREEN, pad=20)

    stages_bert = [
        ('Input:\\n"Great, another\\nboring movie"', 0.85, COLOR_MAIN),
        ('Tokenization +\\n[CLS] token', 0.65, COLOR_ACCENT),
        ('Transformer Layers\\n(Bidirectional)', 0.45, COLOR_ACCENT),
        ('[CLS] = Sentence\\nRepresentation', 0.25, COLOR_ACCENT),
        ('Output: NEGATIVE\\n(CORRECT!)', 0.05, COLOR_GREEN)
    ]

    for text, y, color in stages_bert:
        box = FancyBboxPatch((0.1, y-0.06), 0.8, 0.12, boxstyle="round,pad=0.01",
                            edgecolor=color, facecolor=COLOR_LIGHT, linewidth=2)
        ax2.add_patch(box)
        ax2.text(0.5, y, text, ha='center', va='center',
                fontsize=FONTSIZE_ANNOTATION-2, color=color)
        if y > 0.1:
            ax2.annotate('', xy=(0.5, y-0.08), xytext=(0.5, y-0.18),
                        arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=2))

    # Solution annotation
    ax2.text(0.5, -0.05, 'Understands context!', ha='center', va='top',
            fontsize=FONTSIZE_ANNOTATION, color=COLOR_GREEN, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#E6FFE6', edgecolor=COLOR_GREEN))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('../figures/architecture_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Chart 2: architecture_comparison_bsc.pdf")

def chart3_finetuning_pipeline():
    """
    Chart 3: Fine-Tuning Pipeline (The Solution Process)

    PEDAGOGICAL ROLE: Show how to implement breakthrough
    PREVENTS: "Train BERT from scratch" misconception
    SCAFFOLDS: From architecture to training process
    """
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    fig.suptitle('The Solution: BERT Fine-Tuning Pipeline',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', color=COLOR_ACCENT)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Four stages
    stages = [
        {
            'x': 0.12,
            'title': 'Stage 1\\nPre-trained BERT',
            'content': 'Wikipedia\\n+ Books\\n(General Language)',
            'color': COLOR_GRAY
        },
        {
            'x': 0.33,
            'title': 'Stage 2\\nAdd Classifier',
            'content': 'Linear Layer\\n(Random Init)\\n2 outputs',
            'color': COLOR_ORANGE
        },
        {
            'x': 0.54,
            'title': 'Stage 3\\nFine-Tune',
            'content': 'IMDb Reviews\\n(Labeled Data)\\n1000s examples',
            'color': COLOR_ACCENT
        },
        {
            'x': 0.75,
            'title': 'Stage 4\\nDeploy',
            'content': 'New Reviews\\nPredictions\\nProduction',
            'color': COLOR_GREEN
        }
    ]

    for i, stage in enumerate(stages):
        # Box
        box = FancyBboxPatch((stage['x']-0.09, 0.3), 0.18, 0.4, boxstyle="round,pad=0.02",
                            edgecolor=stage['color'], facecolor=COLOR_LIGHT, linewidth=3)
        ax.add_patch(box)

        # Title
        ax.text(stage['x'], 0.65, stage['title'], ha='center', va='top',
               fontsize=FONTSIZE_ANNOTATION, fontweight='bold', color=stage['color'])

        # Content
        ax.text(stage['x'], 0.50, stage['content'], ha='center', va='center',
               fontsize=FONTSIZE_TICK, color=COLOR_MAIN)

        # Arrow to next stage
        if i < 3:
            ax.annotate('', xy=(stage['x']+0.12, 0.50), xytext=(stages[i+1]['x']-0.12, 0.50),
                       arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=3))

    # Bottom annotation
    ax.text(0.5, 0.15, 'Pre-training gives general understanding, fine-tuning adapts to sentiment task',
           ha='center', va='center', fontsize=FONTSIZE_ANNOTATION, color=COLOR_ACCENT,
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('../figures/finetuning_pipeline_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Chart 3: finetuning_pipeline_bsc.pdf")

def chart4_performance_comparison():
    """
    Chart 4: Performance Comparison (The Proof)

    PEDAGOGICAL ROLE: Provide empirical validation
    PREVENTS: Skepticism about transformers
    SCAFFOLDS: From theoretical solution to quantitative evidence
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    methods = ['BOW\\n+SVM', 'TF-IDF\\n+Logistic', 'LSTM', 'BERT-\\nbase', 'BERT-\\nlarge']
    # Qualitative relative scores (baseline to best)
    rel_scores = [1, 1.2, 1.8, 2.5, 2.8]
    colors = [COLOR_GRAY, COLOR_GRAY, COLOR_ORANGE, COLOR_GREEN, COLOR_GREEN]
    labels = ['Baseline', 'Similar', 'Better', 'Much Better', 'Best']

    bars = ax.bar(methods, rel_scores, color=colors, alpha=0.8, edgecolor=COLOR_MAIN, linewidth=2)

    # Add qualitative labels
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
               label, ha='center', va='bottom',
               fontsize=FONTSIZE_TICK, fontweight='bold', color=COLOR_MAIN)

    ax.set_ylabel('Relative Performance', fontsize=FONTSIZE_LABEL, color=COLOR_MAIN)
    ax.set_xlabel('Method', fontsize=FONTSIZE_LABEL, color=COLOR_MAIN)
    ax.set_title('Proof: BERT Solves the Puzzle',
                fontsize=FONTSIZE_TITLE, fontweight='bold', color=COLOR_ACCENT, pad=20)
    ax.set_ylim(0, 3.5)

    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_ACCENT)
    ax.spines['bottom'].set_color(COLOR_ACCENT)
    ax.tick_params(labelsize=FONTSIZE_TICK, colors=COLOR_MAIN)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=1, axis='y')
    ax.set_facecolor('white')

    # Annotation
    ax.text(0.5, 0.02, 'BERT substantially outperforms traditional methods',
           transform=ax.transAxes, ha='center', va='bottom',
           fontsize=FONTSIZE_ANNOTATION, color=COLOR_GREEN, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#E6FFE6', edgecolor=COLOR_GREEN))

    plt.tight_layout()
    plt.savefig('../figures/performance_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Chart 4: performance_comparison_bsc.pdf")

def chart5_attention_heatmap():
    """
    Chart 5: Attention Heatmap (Understanding the Magic)

    PEDAGOGICAL ROLE: Demystify transformers
    PREVENTS: "Black box" fear
    SCAFFOLDS: From performance to mechanistic understanding
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Example sentence
    tokens = ['[CLS]', 'The', 'movie', 'was', 'not', 'very', 'good', 'but', 'the', 'acting', 'was', 'excellent']

    # Simulated attention weights (focusing on "not", "good", "excellent")
    attention = np.array([
        [0.05, 0.05, 0.05, 0.05, 0.15, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.15],  # CLS attends to key words
        [0.08, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08],  # The
        [0.07, 0.07, 0.12, 0.07, 0.07, 0.07, 0.09, 0.07, 0.07, 0.07, 0.07, 0.09],  # movie
        [0.07, 0.07, 0.07, 0.10, 0.15, 0.07, 0.12, 0.07, 0.07, 0.07, 0.07, 0.08],  # was
        [0.06, 0.06, 0.06, 0.10, 0.20, 0.12, 0.18, 0.06, 0.06, 0.06, 0.06, 0.08],  # not (HIGH)
        [0.07, 0.07, 0.07, 0.09, 0.15, 0.11, 0.14, 0.07, 0.07, 0.07, 0.07, 0.09],  # very
        [0.06, 0.06, 0.07, 0.08, 0.16, 0.10, 0.18, 0.06, 0.06, 0.06, 0.06, 0.11],  # good (HIGH)
        [0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.08, 0.09, 0.10, 0.10, 0.09, 0.09],  # but
        [0.07, 0.10, 0.07, 0.07, 0.07, 0.07, 0.07, 0.08, 0.11, 0.09, 0.07, 0.09],  # the
        [0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.07, 0.07, 0.10, 0.14, 0.10, 0.16],  # acting
        [0.07, 0.07, 0.07, 0.09, 0.08, 0.08, 0.08, 0.08, 0.08, 0.11, 0.10, 0.09],  # was
        [0.05, 0.05, 0.06, 0.05, 0.08, 0.06, 0.10, 0.05, 0.06, 0.14, 0.08, 0.22],  # excellent (HIGH)
    ])

    # Create heatmap
    im = ax.imshow(attention, cmap='Blues', aspect='auto', vmin=0, vmax=0.25)

    # Set ticks
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=FONTSIZE_ANNOTATION-2, rotation=45, ha='right')
    ax.set_yticklabels(tokens, fontsize=FONTSIZE_ANNOTATION-2)

    # Labels
    ax.set_xlabel('Tokens (attending to)', fontsize=FONTSIZE_LABEL, color=COLOR_MAIN)
    ax.set_ylabel('Tokens (from)', fontsize=FONTSIZE_LABEL, color=COLOR_MAIN)
    ax.set_title('Understanding the Magic: BERT Attention Heatmap',
                fontsize=FONTSIZE_TITLE, fontweight='bold', color=COLOR_ACCENT, pad=20)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=FONTSIZE_ANNOTATION, color=COLOR_MAIN)
    cbar.ax.tick_params(labelsize=FONTSIZE_TICK)

    # Annotation
    ax.text(0.5, -0.25, 'BERT focuses on "not", "good", "excellent" - understands negation and contrast',
           transform=ax.transAxes, ha='center', va='top',
           fontsize=FONTSIZE_ANNOTATION, color=COLOR_ACCENT, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT))

    plt.tight_layout()
    plt.savefig('../figures/attention_heatmap_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Chart 5: attention_heatmap_bsc.pdf")

if __name__ == '__main__':
    print("Generating all sentiment analysis charts...")
    print("=" * 60)

    chart2_architecture_comparison()
    chart3_finetuning_pipeline()
    chart4_performance_comparison()
    chart5_attention_heatmap()

    print("=" * 60)
    print("All charts generated successfully!")
    print("Charts saved to: ../figures/")
    print("Font standard: BSc Discovery (16-24pt)")
