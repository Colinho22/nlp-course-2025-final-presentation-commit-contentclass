#!/usr/bin/env python3
"""
Generate minimalist charts for Week 4: Sequence-to-Sequence Models
Following strict monochromatic gray palette
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrow
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches

# Configure matplotlib for minimalist design
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.2

# Monochromatic color scheme
COLOR_MAIN = '#404040'       # RGB(64,64,64) - Main text
COLOR_ANNOTATION = '#B4B4B4'  # RGB(180,180,180) - Annotations
COLOR_BACKGROUND = '#F0F0F0'  # RGB(240,240,240) - Backgrounds
COLOR_DARK = '#333333'        # Darker gray for emphasis
COLOR_MID = '#666666'         # Mid gray
COLOR_LIGHT = '#999999'       # Light gray
COLOR_VLIGHT = '#CCCCCC'      # Very light gray

def create_seq2seq_architecture():
    """Create encoder-decoder architecture visualization"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Encoder section
    encoder_x = np.linspace(1, 3.5, 4)
    for i, x in enumerate(encoder_x):
        rect = FancyBboxPatch((x-0.3, 2), 0.6, 0.8,
                               boxstyle="round,pad=0.05",
                               facecolor=COLOR_LIGHT, edgecolor=COLOR_DARK, linewidth=1)
        ax.add_patch(rect)
        ax.text(x, 2.4, f'$h_{i+1}$', ha='center', va='center', fontsize=9, color=COLOR_MAIN)

        # Input tokens
        ax.text(x, 1.2, f'$x_{i+1}$', ha='center', va='center', fontsize=9, color=COLOR_MID)
        ax.arrow(x, 1.5, 0, 0.4, head_width=0.1, head_length=0.1, fc=COLOR_LIGHT, ec=COLOR_LIGHT)

        # Connections between states
        if i < len(encoder_x) - 1:
            ax.arrow(x+0.3, 2.4, encoder_x[i+1]-x-0.6, 0,
                    head_width=0.1, head_length=0.1, fc=COLOR_LIGHT, ec=COLOR_LIGHT)

    # Context vector
    context_rect = FancyBboxPatch((4.2, 2), 1.6, 0.8,
                                  boxstyle="round,pad=0.05",
                                  facecolor=COLOR_MID, edgecolor=COLOR_DARK, linewidth=2)
    ax.add_patch(context_rect)
    ax.text(5, 2.4, 'Context $c$', ha='center', va='center', fontsize=10,
           fontweight='bold', color='white')

    # Arrow from encoder to context
    ax.arrow(3.8, 2.4, 0.3, 0, head_width=0.1, head_length=0.1,
            fc=COLOR_DARK, ec=COLOR_DARK, linewidth=1.5)

    # Decoder section
    decoder_x = np.linspace(6.5, 9, 4)
    for i, x in enumerate(decoder_x):
        rect = FancyBboxPatch((x-0.3, 2), 0.6, 0.8,
                               boxstyle="round,pad=0.05",
                               facecolor=COLOR_LIGHT, edgecolor=COLOR_DARK, linewidth=1)
        ax.add_patch(rect)
        ax.text(x, 2.4, f'$s_{i+1}$', ha='center', va='center', fontsize=9, color=COLOR_MAIN)

        # Output tokens
        ax.text(x, 3.5, f'$y_{i+1}$', ha='center', va='center', fontsize=9, color=COLOR_MID)
        ax.arrow(x, 2.9, 0, 0.4, head_width=0.1, head_length=0.1, fc=COLOR_LIGHT, ec=COLOR_LIGHT)

        # Context connections
        ax.arrow(5, 2.8, x-5, -0.3, head_width=0.05, head_length=0.05,
                fc=COLOR_VLIGHT, ec=COLOR_VLIGHT, alpha=0.5, linestyle='--')

        # Connections between states
        if i < len(decoder_x) - 1:
            ax.arrow(x+0.3, 2.4, decoder_x[i+1]-x-0.6, 0,
                    head_width=0.1, head_length=0.1, fc=COLOR_LIGHT, ec=COLOR_LIGHT)

    # Arrow from context to decoder
    ax.arrow(5.8, 2.4, 0.3, 0, head_width=0.1, head_length=0.1,
            fc=COLOR_DARK, ec=COLOR_DARK, linewidth=1.5)

    # Labels
    ax.text(2.25, 0.5, 'Encoder', ha='center', fontsize=12, fontweight='bold', color=COLOR_DARK)
    ax.text(7.75, 0.5, 'Decoder', ha='center', fontsize=12, fontweight='bold', color=COLOR_DARK)
    ax.text(5, 4.5, 'Sequence-to-Sequence Architecture', ha='center', fontsize=14,
           fontweight='bold', color=COLOR_MAIN)

    # Annotations
    ax.text(2.25, 5.2, 'Variable input length', ha='center', fontsize=8,
           style='italic', color=COLOR_ANNOTATION)
    ax.text(7.75, 5.2, 'Variable output length', ha='center', fontsize=8,
           style='italic', color=COLOR_ANNOTATION)
    ax.text(5, 1.2, 'Fixed-size representation', ha='center', fontsize=8,
           style='italic', color=COLOR_ANNOTATION)

    plt.tight_layout()
    plt.savefig('../figures/week4_seq2seq_architecture_minimalist.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_attention_mechanism():
    """Create attention mechanism visualization"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Source sequence (encoder states)
    source_x = np.linspace(1.5, 4.5, 4)
    encoder_states = []
    for i, x in enumerate(source_x):
        rect = FancyBboxPatch((x-0.3, 1), 0.6, 0.8,
                               boxstyle="round,pad=0.05",
                               facecolor=COLOR_VLIGHT, edgecolor=COLOR_MID, linewidth=1)
        ax.add_patch(rect)
        ax.text(x, 1.4, f'$h_{i+1}$', ha='center', va='center', fontsize=9, color=COLOR_MAIN)
        encoder_states.append((x, 1.8))

    # Current decoder state
    decoder_rect = FancyBboxPatch((6, 3), 0.8, 0.8,
                                  boxstyle="round,pad=0.05",
                                  facecolor=COLOR_MID, edgecolor=COLOR_DARK, linewidth=2)
    ax.add_patch(decoder_rect)
    ax.text(6.4, 3.4, '$s_t$', ha='center', va='center', fontsize=10,
           fontweight='bold', color='white')

    # Attention weights (different intensities)
    attention_weights = [0.1, 0.2, 0.6, 0.1]  # Example weights
    for i, (x, y) in enumerate(encoder_states):
        alpha = attention_weights[i]
        # Draw attention connection with varying thickness
        ax.plot([x, 6.4], [y, 3], color=COLOR_MID, alpha=alpha, linewidth=3*alpha+0.5)

        # Attention weight label
        mid_x = (x + 6.4) / 2
        mid_y = (y + 3) / 2
        if alpha > 0.3:
            ax.text(mid_x, mid_y + 0.2, f'$\\alpha_{i+1}$={alpha:.1f}',
                   ha='center', fontsize=8, color=COLOR_DARK)

    # Context vector
    context_rect = FancyBboxPatch((7.5, 3), 0.8, 0.8,
                                  boxstyle="round,pad=0.05",
                                  facecolor=COLOR_DARK, edgecolor=COLOR_DARK, linewidth=2)
    ax.add_patch(context_rect)
    ax.text(7.9, 3.4, '$c_t$', ha='center', va='center', fontsize=10,
           fontweight='bold', color='white')

    # Arrow from decoder to context
    ax.arrow(6.9, 3.4, 0.5, 0, head_width=0.1, head_length=0.1,
            fc=COLOR_DARK, ec=COLOR_DARK, linewidth=1.5)

    # Labels
    ax.text(3, 0.3, 'Encoder Hidden States', ha='center', fontsize=11,
           fontweight='bold', color=COLOR_DARK)
    ax.text(6.4, 4.5, 'Decoder State', ha='center', fontsize=11,
           fontweight='bold', color=COLOR_DARK)
    ax.text(7.9, 4.5, 'Context', ha='center', fontsize=11,
           fontweight='bold', color=COLOR_DARK)

    # Title
    ax.text(5, 5.3, 'Attention Mechanism', ha='center', fontsize=14,
           fontweight='bold', color=COLOR_MAIN)

    # Formula
    ax.text(5, 2.2, r'$c_t = \sum_{i=1}^{T} \alpha_i h_i$', ha='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_BACKGROUND, edgecolor=COLOR_MID))

    # Annotation
    ax.text(5, 0.3, 'Weighted sum of encoder states based on relevance',
           ha='center', fontsize=8, style='italic', color=COLOR_ANNOTATION)

    plt.tight_layout()
    plt.savefig('../figures/week4_attention_mechanism_minimalist.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_bleu_comparison():
    """Create BLEU score comparison chart"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

    models = ['RNN\nBaseline', 'LSTM\nBaseline', 'Seq2Seq\nBasic',
              'Seq2Seq\n+Attention', 'Transformer']
    bleu_scores = [15.2, 18.7, 24.3, 31.4, 35.8]

    x_pos = np.arange(len(models))

    # Create bars with gradient effect
    bars = ax.bar(x_pos, bleu_scores, color=[COLOR_VLIGHT, COLOR_LIGHT, COLOR_MID, COLOR_DARK, COLOR_MAIN])

    # Add value labels on bars
    for bar, score in zip(bars, bleu_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{score:.1f}', ha='center', va='bottom', fontsize=9, color=COLOR_DARK)

    # Styling
    ax.set_xlabel('Model Architecture', fontsize=11, color=COLOR_MAIN)
    ax.set_ylabel('BLEU Score', fontsize=11, color=COLOR_MAIN)
    ax.set_title('Translation Quality Evolution (WMT14 EN-DE)', fontsize=13,
                fontweight='bold', color=COLOR_MAIN, pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=9, color=COLOR_MID)
    ax.set_ylim(0, 40)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Subtle grid
    ax.yaxis.grid(True, alpha=0.2, linestyle='--', color=COLOR_VLIGHT)
    ax.set_axisbelow(True)

    # Annotation
    ax.text(len(models)/2 - 0.5, -7, 'Higher is better. Human performance ~40',
           ha='center', fontsize=8, style='italic', color=COLOR_ANNOTATION)

    plt.tight_layout()
    plt.savefig('../figures/week4_bleu_comparison_minimalist.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_training_process():
    """Create training process visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')

    # Teacher Forcing (left)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 5)
    ax1.axis('off')
    ax1.set_title('Teacher Forcing', fontsize=12, fontweight='bold', color=COLOR_MAIN)

    # Timeline
    steps = ['START', 'Je', "t'", 'aime', 'END']
    y_pos = 3

    for i, step in enumerate(steps[:-1]):
        # Current input
        rect = FancyBboxPatch((i+0.2, y_pos-0.3), 0.6, 0.6,
                               boxstyle="round,pad=0.05",
                               facecolor=COLOR_LIGHT, edgecolor=COLOR_DARK)
        ax1.add_patch(rect)
        ax1.text(i+0.5, y_pos, step, ha='center', va='center', fontsize=9, color=COLOR_MAIN)

        # Arrow to next
        if i < len(steps) - 2:
            ax1.arrow(i+0.85, y_pos, 0.3, 0, head_width=0.1, head_length=0.05,
                     fc=COLOR_MID, ec=COLOR_MID)

        # Prediction above
        ax1.text(i+0.5, y_pos+0.8, steps[i+1], ha='center', va='center',
                fontsize=8, color=COLOR_DARK, style='italic')

    ax1.text(2.5, 1.5, 'Ground truth used as input', ha='center', fontsize=9,
            style='italic', color=COLOR_ANNOTATION)

    # Inference Mode (right)
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 5)
    ax2.axis('off')
    ax2.set_title('Inference Mode', fontsize=12, fontweight='bold', color=COLOR_MAIN)

    for i, step in enumerate(steps[:-1]):
        # Current input (predicted from previous)
        rect = FancyBboxPatch((i+0.2, y_pos-0.3), 0.6, 0.6,
                               boxstyle="round,pad=0.05",
                               facecolor=COLOR_VLIGHT if i > 0 else COLOR_LIGHT,
                               edgecolor=COLOR_MID, linestyle='--' if i > 0 else '-')
        ax2.add_patch(rect)
        ax2.text(i+0.5, y_pos, step if i == 0 else '?', ha='center', va='center',
                fontsize=9, color=COLOR_MID if i > 0 else COLOR_MAIN)

        # Arrow to next
        if i < len(steps) - 2:
            ax2.arrow(i+0.85, y_pos, 0.3, 0, head_width=0.1, head_length=0.05,
                     fc=COLOR_VLIGHT, ec=COLOR_VLIGHT, linestyle='--')

        # Prediction above
        if i < len(steps) - 1:
            ax2.text(i+0.5, y_pos+0.8, '?', ha='center', va='center',
                    fontsize=8, color=COLOR_MID, style='italic')

    ax2.text(2.5, 1.5, 'Previous predictions used as input', ha='center', fontsize=9,
            style='italic', color=COLOR_ANNOTATION)

    plt.tight_layout()
    plt.savefig('../figures/week4_training_process_minimalist.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_applications_timeline():
    """Create applications timeline"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

    years = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
    applications = [
        'Neural Machine\nTranslation',
        'Image\nCaptioning',
        'Speech\nRecognition',
        'Text\nSummarization',
        'Dialogue\nSystems',
        'Code\nGeneration',
        'Multimodal\nTranslation'
    ]
    impact_scores = [85, 70, 75, 80, 90, 95, 100]

    # Create timeline
    ax.plot(years, impact_scores, 'o-', color=COLOR_DARK, linewidth=2, markersize=8)

    # Fill area under curve
    ax.fill_between(years, 0, impact_scores, alpha=0.1, color=COLOR_MID)

    # Add labels for each milestone
    for year, app, score in zip(years, applications, impact_scores):
        ax.annotate(app, xy=(year, score), xytext=(0, 15),
                   textcoords='offset points', ha='center', fontsize=8,
                   color=COLOR_MAIN,
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor=COLOR_BACKGROUND, edgecolor=COLOR_MID, linewidth=0.5))

    # Styling
    ax.set_xlabel('Year', fontsize=11, color=COLOR_MAIN)
    ax.set_ylabel('Industry Impact Score', fontsize=11, color=COLOR_MAIN)
    ax.set_title('Seq2Seq Applications Timeline', fontsize=13,
                fontweight='bold', color=COLOR_MAIN, pad=20)
    ax.set_xlim(2013.5, 2020.5)
    ax.set_ylim(0, 110)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Grid
    ax.grid(True, alpha=0.1, linestyle='--', color=COLOR_VLIGHT)
    ax.set_axisbelow(True)

    # Annotation
    ax.text(2017, 10, 'From research to production deployment',
           ha='center', fontsize=8, style='italic', color=COLOR_ANNOTATION)

    plt.tight_layout()
    plt.savefig('../figures/week4_applications_timeline_minimalist.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all charts for Week 4"""
    print("Generating Week 4 minimalist charts...")

    create_seq2seq_architecture()
    print("  - Seq2Seq architecture diagram created")

    create_attention_mechanism()
    print("  - Attention mechanism visualization created")

    create_bleu_comparison()
    print("  - BLEU score comparison chart created")

    create_training_process()
    print("  - Training process visualization created")

    create_applications_timeline()
    print("  - Applications timeline created")

    print("\nAll Week 4 minimalist charts generated successfully!")

if __name__ == "__main__":
    main()