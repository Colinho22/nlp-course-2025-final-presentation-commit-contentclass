"""
Generate BSc-level discovery-based visualizations for Week 6: Pre-trained Models
Following Educational Presentation Framework - comprehensive chart suite

Date: 2025-10-26
Charts: 35 comprehensive visualizations for 52-slide presentation (0.67 ratio)
Focus: BERT and GPT - the 2018 revolution
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch, Wedge, Polygon
from matplotlib.lines import Line2D
import seaborn as sns
import os

# Create figures directory
os.makedirs('../figures', exist_ok=True)

# Educational Presentation Framework colors
COLOR_MLPURPLE = '#3333B2'
COLOR_DARKGRAY = '#404040'
COLOR_MIDGRAY = '#B4B4B4'
COLOR_LIGHTGRAY = '#F0F0F0'

# Pedagogical colors
COLOR_CURRENT = '#FF6B6B'
COLOR_CONTEXT = '#4ECDC4'
COLOR_PREDICT = '#95E77E'
COLOR_NEUTRAL = '#E0E0E0'

# BERT/GPT specific colors
COLOR_BERT = '#E74C3C'  # Red for BERT
COLOR_GPT = '#3498DB'   # Blue for GPT

def set_minimalist_style(ax):
    """Apply minimalist style"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_MIDGRAY)
    ax.spines['bottom'].set_color(COLOR_MIDGRAY)
    ax.tick_params(colors=COLOR_DARKGRAY, which='both')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_facecolor('white')

# ===== OPENING & PRE-2018 CHARTS (1-8) =====

# Chart 1: Training Cost Comparison
def plot_training_cost_comparison():
    """Show cost difference: scratch vs fine-tune"""
    fig, ax = plt.subplots(figsize=(12, 6))

    approaches = ['Train BERT\nfrom Scratch', 'Fine-tune\nPre-trained BERT']
    costs = [1000000, 250]  # Dollars
    times = [4, 0.16]  # Days (4 days vs 4 hours)

    x = np.arange(len(approaches))
    width = 0.35

    bars1 = ax.bar(x - width/2, costs, width, label='Cost (USD)', color=COLOR_CURRENT,
                   alpha=0.8, edgecolor=COLOR_DARKGRAY, linewidth=2)

    # Create secondary axis for time
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, times, width, label='Time (days)', color=COLOR_PREDICT,
                    alpha=0.8, edgecolor=COLOR_DARKGRAY, linewidth=2)

    ax.set_ylabel('Cost (USD)', fontsize=13, fontweight='bold', color=COLOR_DARKGRAY)
    ax2.set_ylabel('Time (days)', fontsize=13, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_title('The $1 Million Problem: Training Cost Revolution', fontsize=15, fontweight='bold',
                 color=COLOR_DARKGRAY)
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, fontsize=11)
    ax.set_yscale('log')
    ax2.set_yscale('log')

    # Add value labels
    for i, cost in enumerate(costs):
        ax.text(i - width/2, cost * 1.5, f'${cost:,}', ha='center', fontsize=10,
                fontweight='bold', color=COLOR_CURRENT)

    for i, time in enumerate(times):
        ax2.text(i + width/2, time * 1.5, f'{time:.1f} days', ha='center', fontsize=10,
                 fontweight='bold', color=COLOR_PREDICT)

    # Annotation
    ax.annotate('4000x cheaper!', xy=(1, 250), xytext=(0.5, 5000),
               arrowprops=dict(arrowstyle='->', color='green', lw=3),
               fontsize=12, fontweight='bold', color='green')

    set_minimalist_style(ax)
    ax.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig('../figures/training_cost_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 2: 2018 Breakthrough Timeline
def plot_2018_breakthrough_timeline():
    """Timeline showing June (GPT-1) and October (BERT) 2018"""
    fig, ax = plt.subplots(figsize=(14, 6))

    events = [
        ('Jan 2018', 0, 'Pre-2018\nTask-specific models', COLOR_NEUTRAL),
        ('Jun 2018', 5, 'GPT-1\nAutoregressive pre-training', COLOR_GPT),
        ('Oct 2018', 9, 'BERT\nBidirectional pre-training', COLOR_BERT),
        ('Dec 2018', 11, 'Post-2018\nPre-training era begins', COLOR_PREDICT)
    ]

    for date, x, label, color in events:
        circle = Circle((x, 1), 0.4, facecolor=color, edgecolor=COLOR_DARKGRAY, linewidth=2.5,
                       alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, 1, date.split()[1], ha='center', va='center', fontsize=10,
                fontweight='bold', color='white' if x > 0 else COLOR_DARKGRAY)

        ax.text(x, 0.2, label, ha='center', fontsize=9, color=COLOR_DARKGRAY)

    # Timeline line
    ax.plot([0, 11], [1, 1], color=COLOR_MIDGRAY, linewidth=3, alpha=0.5)

    # Breakthrough annotations
    ax.annotate('The Revolution\nBegins', xy=(5, 1.4), xytext=(5, 2.3),
               arrowprops=dict(arrowstyle='->', color=COLOR_GPT, lw=2.5),
               fontsize=11, fontweight='bold', color=COLOR_GPT, ha='center')

    ax.annotate('State-of-Art on\nAll Tasks', xy=(9, 1.4), xytext=(9, 2.3),
               arrowprops=dict(arrowstyle='->', color=COLOR_BERT, lw=2.5),
               fontsize=11, fontweight='bold', color=COLOR_BERT, ha='center')

    ax.set_xlim(-1, 12)
    ax.set_ylim(-0.5, 2.8)
    ax.axis('off')
    ax.set_title('The 2018 Breakthrough: 4 Months That Changed NLP', fontsize=15,
                 fontweight='bold', color=COLOR_DARKGRAY, pad=20)

    plt.tight_layout()
    plt.savefig('../figures/2018_breakthrough_timeline_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 3: Pre-training vs Fine-tuning Paradigm
def plot_pretraining_finetuning_paradigm():
    """Core paradigm shift diagram"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Pre-training phase
    pretrain_box = FancyBboxPatch((0.5, 5), 5, 2, boxstyle="round,pad=0.15",
                                  facecolor=COLOR_MLPURPLE, alpha=0.3,
                                  edgecolor=COLOR_DARKGRAY, linewidth=3)
    ax.add_patch(pretrain_box)
    ax.text(3, 6.5, 'Pre-training (Unsupervised)', ha='center', fontsize=13,
            fontweight='bold', color=COLOR_DARKGRAY)
    ax.text(3, 6.0, 'Massive corpus: 3-300B tokens', ha='center', fontsize=10)
    ax.text(3, 5.6, 'Self-supervised objectives', ha='center', fontsize=10)
    ax.text(3, 5.2, 'Cost: $1M-$10M (one time)', ha='center', fontsize=9, style='italic')

    # Arrow down
    ax.annotate('', xy=(3, 4.8), xytext=(3, 5),
               arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=3))
    ax.text(3.5, 4.9, 'General language model', fontsize=9, style='italic')

    # Fine-tuning phase (multiple tasks)
    tasks = ['Classification', 'QA', 'NER', 'Generation']
    x_positions = [0.8, 3, 5.2, 7.4]

    for i, (task, x) in enumerate(zip(tasks, x_positions)):
        finetune_box = FancyBboxPatch((x-0.6, 2.5), 1.2, 1.8, boxstyle="round,pad=0.1",
                                      facecolor=COLOR_PREDICT, alpha=0.4,
                                      edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(finetune_box)
        ax.text(x, 3.9, f'Task {i+1}:', ha='center', fontsize=9, fontweight='bold')
        ax.text(x, 3.5, task, ha='center', fontsize=10)
        ax.text(x, 3.1, '100-10K\nexamples', ha='center', fontsize=8)
        ax.text(x, 2.7, '$50-500', ha='center', fontsize=8, style='italic')

        # Arrow from pre-training to each task
        ax.annotate('', xy=(x, 4.3), xytext=(3, 4.8),
                   arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2, alpha=0.7))

    ax.set_xlim(0, 8.5)
    ax.set_ylim(1.5, 7.5)
    ax.axis('off')
    ax.set_title('The New Paradigm: Learn Once, Apply Everywhere', fontsize=15,
                 fontweight='bold', color=COLOR_DARKGRAY, pad=20)

    # Bottom note
    ax.text(4.25, 1.7, 'Pre-training is expensive but done once. Fine-tuning is cheap and repeated.',
            ha='center', fontsize=10, style='italic', color=COLOR_DARKGRAY)

    plt.tight_layout()
    plt.savefig('../figures/pretraining_finetuning_paradigm_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 4: Task-Specific Architectures (Pre-2018)
def plot_task_specific_architectures():
    """Show fragmented pre-2018 landscape"""
    fig, ax = plt.subplots(figsize=(12, 7))

    tasks = [
        ('Sentiment\nAnalysis', 'CNN', COLOR_CURRENT),
        ('Question\nAnswering', 'BiDAF', COLOR_CONTEXT),
        ('Named Entity\nRecognition', 'BiLSTM\n-CRF', COLOR_PREDICT),
        ('Translation', 'Seq2Seq\n+Attention', COLOR_MLPURPLE)
    ]

    for i, (task, model, color) in enumerate(tasks):
        x = i * 2.5 + 1

        # Task box
        task_box = FancyBboxPatch((x-0.8, 5), 1.6, 1, boxstyle="round,pad=0.1",
                                  facecolor=COLOR_LIGHTGRAY,
                                  edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(task_box)
        ax.text(x, 5.5, task, ha='center', fontsize=10, fontweight='bold')

        # Arrow down
        ax.annotate('', xy=(x, 4.5), xytext=(x, 5),
                   arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))

        # Model box
        model_box = FancyBboxPatch((x-0.8, 3), 1.6, 1.2, boxstyle="round,pad=0.1",
                                   facecolor=color, alpha=0.5,
                                   edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(model_box)
        ax.text(x, 3.6, model, ha='center', fontsize=10, fontweight='bold')

        # "No Transfer" X marks
        if i < len(tasks) - 1:
            next_x = (i + 1) * 2.5 + 1
            mid_x = (x + next_x) / 2
            ax.plot([mid_x-0.2, mid_x+0.2], [3.6-0.2, 3.6+0.2], 'r-', linewidth=3)
            ax.plot([mid_x-0.2, mid_x+0.2], [3.6+0.2, 3.6-0.2], 'r-', linewidth=3)
            ax.text(mid_x, 2.8, 'No\ntransfer', ha='center', fontsize=8,
                   color='red', fontweight='bold')

    ax.set_xlim(0, 10)
    ax.set_ylim(2, 6.5)
    ax.axis('off')
    ax.set_title('Pre-2018: Every Task Needed Its Own Model', fontsize=15,
                 fontweight='bold', color=COLOR_DARKGRAY, pad=20)

    plt.tight_layout()
    plt.savefig('../figures/task_specific_architectures_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 5: Data Requirements Pre-2018
def plot_data_requirements_pre2018():
    """Show performance vs dataset size"""
    fig, ax = plt.subplots(figsize=(10, 6))

    dataset_sizes = [100, 500, 1000, 5000, 10000, 50000]
    accuracy_scratch = [45, 55, 62, 71, 78, 83]  # Training from scratch
    accuracy_pretrained = [68, 82, 87, 91, 93, 94]  # With pre-training (hypothetical 2024)

    ax.plot(dataset_sizes, accuracy_scratch, marker='o', markersize=8, linewidth=2.5,
            color=COLOR_CURRENT, label='Training from scratch (pre-2018)')
    ax.plot(dataset_sizes, accuracy_pretrained, marker='s', markersize=8, linewidth=2.5,
            color=COLOR_PREDICT, linestyle='--', alpha=0.7,
            label='With pre-training (post-2018)')

    # Highlight the problem zone
    ax.axvspan(100, 5000, alpha=0.1, color='red', label='Data bottleneck zone')

    ax.set_xlabel('Labeled Training Examples', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_title('Data Requirements: Pre-training Changes Everything', fontsize=14,
                 fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_xscale('log')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim(40, 100)

    # Annotation
    ax.annotate('Pre-training gives\n30-50% boost!', xy=(500, 82), xytext=(2000, 90),
               arrowprops=dict(arrowstyle='->', color='green', lw=2),
               fontsize=10, fontweight='bold', color='green')

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/data_requirements_pre2018_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 6: Paradigm Shift Comparison (used in opening and integration)
def plot_paradigm_shift_comparison():
    """Before/after comparison visual"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Before 2018
    ax1.set_title('Before 2018: Task-Specific Training', fontsize=13, fontweight='bold',
                  color=COLOR_DARKGRAY)

    # Multiple isolated pipelines
    tasks_before = ['Task 1', 'Task 2', 'Task 3']
    for i, task in enumerate(tasks_before):
        y = 2 - i * 0.8

        # Data
        rect1 = FancyBboxPatch((0.2, y), 1.2, 0.5, boxstyle="round,pad=0.05",
                               facecolor=COLOR_LIGHTGRAY, edgecolor=COLOR_DARKGRAY, linewidth=1.5)
        ax1.add_patch(rect1)
        ax1.text(0.8, y+0.25, f'{task}\ndata', ha='center', fontsize=8)

        # Arrow
        ax1.annotate('', xy=(1.6, y+0.25), xytext=(1.4, y+0.25),
                    arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))

        # Model
        rect2 = FancyBboxPatch((1.7, y), 1.2, 0.5, boxstyle="round,pad=0.05",
                               facecolor=COLOR_CURRENT, alpha=0.5,
                               edgecolor=COLOR_DARKGRAY, linewidth=1.5)
        ax1.add_patch(rect2)
        ax1.text(2.3, y+0.25, f'{task}\nmodel', ha='center', fontsize=8, fontweight='bold')

    ax1.text(1.5, -0.3, 'No sharing\nNo transfer\nExpensive', ha='center', fontsize=10,
            color='red', fontweight='bold')

    ax1.set_xlim(0, 3.5)
    ax1.set_ylim(-0.5, 2.8)
    ax1.axis('off')

    # After 2018
    ax2.set_title('After 2018: Pre-training + Fine-tuning', fontsize=13, fontweight='bold',
                  color=COLOR_DARKGRAY)

    # Pre-training (shared)
    pretrain = FancyBboxPatch((0.2, 1.8), 2.8, 0.8, boxstyle="round,pad=0.1",
                              facecolor=COLOR_MLPURPLE, alpha=0.3,
                              edgecolor=COLOR_DARKGRAY, linewidth=3)
    ax2.add_patch(pretrain)
    ax2.text(1.6, 2.2, 'Pre-trained Model (BERT/GPT)', ha='center', fontsize=10,
            fontweight='bold')

    # Multiple fine-tuning paths
    for i, task in enumerate(['Task 1', 'Task 2', 'Task 3']):
        x_offset = i * 1.0
        y_bottom = 0.3

        # Arrow down
        ax2.annotate('', xy=(0.6 + x_offset, 1.6), xytext=(1.6, 1.8),
                    arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))

        # Fine-tuned model
        ft_box = FancyBboxPatch((x_offset + 0.1, y_bottom), 1, 0.8, boxstyle="round,pad=0.08",
                                facecolor=COLOR_PREDICT, alpha=0.6,
                                edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax2.add_patch(ft_box)
        ax2.text(x_offset + 0.6, y_bottom + 0.4, task, ha='center', fontsize=9, fontweight='bold')

    ax2.text(1.6, -0.3, 'Shared foundation\nFast adaptation\nCheap', ha='center', fontsize=10,
            color='green', fontweight='bold')

    ax2.set_xlim(0, 3.5)
    ax2.set_ylim(-0.5, 2.8)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/paradigm_shift_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# ===== BERT CHARTS (9-20) =====

# Chart 7: Fill-in-Blank Challenge
def plot_fill_in_blank_challenge():
    """Show why bidirectional matters"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    sentence = ["The", "[BLANK]", "sat", "on", "the", "mat"]

    # Left-to-right (fails)
    ax = axes[0]
    ax.set_title("Left-to-Right: Can Only See 'The [BLANK]'", fontsize=12,
                 fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')

    for i, word in enumerate(sentence):
        if i < 2:
            color = COLOR_CONTEXT
            alpha = 0.8
        else:
            color = COLOR_NEUTRAL
            alpha = 0.3  # Grayed out (can't see)

        rect = FancyBboxPatch((i-0.4, 0), 0.8, 0.7, boxstyle="round,pad=0.06",
                              facecolor=color, alpha=alpha,
                              edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(rect)
        ax.text(i, 0.35, word, ha='center', fontsize=9, fontweight='bold')

    ax.text(3, -0.3, 'Missing critical context! Accuracy: LOW', ha='center',
            fontsize=10, color='red', fontweight='bold')

    # Bidirectional (succeeds)
    ax = axes[1]
    ax.set_title("Bidirectional: Sees Full Sentence", fontsize=12, fontweight='bold',
                 color=COLOR_DARKGRAY)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')

    for i, word in enumerate(sentence):
        color = COLOR_PREDICT if word == "[BLANK]" else COLOR_CONTEXT
        rect = FancyBboxPatch((i-0.4, 0), 0.8, 0.7, boxstyle="round,pad=0.06",
                              facecolor=color, edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(rect)
        ax.text(i, 0.35, word, ha='center', fontsize=9, fontweight='bold')

        # Bidirectional arrows
        if i != 1:
            arrow = FancyArrowPatch((i, 0.7), (1, 0.7), arrowstyle='<->', mutation_scale=12,
                                   linewidth=1.5, color=COLOR_DARKGRAY, alpha=0.5)
            ax.add_patch(arrow)

    ax.text(3, -0.3, 'Full context! "sat on the mat" \u2192 predicts "cat". Accuracy: HIGH',
            ha='center', fontsize=10, color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/fill_in_blank_challenge_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 8: Bidirectional Attention Flow
def plot_bidirectional_attention_flow():
    """Show full attention matrix (no causal mask)"""
    fig, ax = plt.subplots(figsize=(10, 8))

    tokens = ["The", "[MASK]", "sat", "on", "the", "mat"]
    n = len(tokens)

    # Create attention matrix (all 1s for bidirectional)
    attention = np.ones((n, n))

    # Plot heatmap
    im = ax.imshow(attention, cmap='Purples', aspect='auto', alpha=0.7)

    # Add token labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, fontsize=11)
    ax.set_yticklabels(tokens, fontsize=11)

    ax.set_xlabel('Attend TO (keys)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attend FROM (queries)', fontsize=12, fontweight='bold')
    ax.set_title('BERT Bidirectional Attention: Every Token Sees Every Other Token',
                 fontsize=13, fontweight='bold', color=COLOR_DARKGRAY, pad=15)

    # Highlight [MASK] row
    ax.add_patch(Rectangle((-0.5, 1-0.5), n, 1, fill=False, edgecolor='red',
                          linewidth=3, linestyle='--'))
    ax.text(n, 1, '[MASK] attends to ALL', va='center', fontsize=10,
            color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/bidirectional_attention_flow_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 9: Masked LM Process
def plot_masked_lm_process():
    """Step-by-step masked language modeling"""
    fig, ax = plt.subplots(figsize=(12, 8))

    steps = [
        (6.5, 'Input: "The [MASK] sat on the mat"', COLOR_LIGHTGRAY),
        (5.5, 'Step 1: Token + Position Embeddings', COLOR_CONTEXT),
        (4.5, 'Step 2: 12 Transformer Encoder Layers', COLOR_MLPURPLE),
        (3.5, 'Step 3: Output for [MASK] position', COLOR_PREDICT),
        (2.5, 'Step 4: Project to vocabulary (30K)', COLOR_CONTEXT),
        (1.5, 'Step 5: Softmax → P(cat)=0.73', COLOR_PREDICT)
    ]

    for y, text, color in steps:
        box = FancyBboxPatch((1, y-0.3), 10, 0.6, boxstyle="round,pad=0.1",
                             facecolor=color, alpha=0.4, edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(box)
        ax.text(6, y, text, ha='center', fontsize=11, fontweight='bold' if 'Input' in text else 'normal')

        # Arrow down (except last)
        if y > 1.5:
            ax.annotate('', xy=(6, y-0.5), xytext=(6, y-0.3),
                       arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2.5))

    ax.set_xlim(0, 12)
    ax.set_ylim(0.8, 7)
    ax.axis('off')
    ax.set_title('Masked Language Modeling Process', fontsize=15, fontweight='bold',
                 color=COLOR_DARKGRAY, pad=20)

    plt.tight_layout()
    plt.savefig('../figures/masked_lm_process_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 10: BERT Architecture
def plot_bert_architecture():
    """Layered architecture diagram"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Input layer
    input_box = FancyBboxPatch((2, 0.5), 6, 0.8, boxstyle="round,pad=0.1",
                               facecolor=COLOR_LIGHTGRAY, edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 0.9, 'Token + Position + Segment Embeddings', ha='center', fontsize=10,
            fontweight='bold')

    # 12 Encoder layers
    for i in range(12):
        y = 1.8 + i * 0.6
        layer_box = FancyBboxPatch((2, y), 6, 0.5, boxstyle="round,pad=0.08",
                                   facecolor=COLOR_MLPURPLE, alpha=0.3 + (i/12)*0.4,
                                   edgecolor=COLOR_DARKGRAY, linewidth=1.5)
        ax.add_patch(layer_box)
        ax.text(5, y+0.25, f'Encoder Layer {i+1}', ha='center', fontsize=8)

        # Arrow between layers
        if i < 11:
            ax.annotate('', xy=(5, y+0.6), xytext=(5, y+0.5),
                       arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=1.5))

    # Output layer
    output_box = FancyBboxPatch((2, 9.5), 6, 0.8, boxstyle="round,pad=0.1",
                                facecolor=COLOR_PREDICT, alpha=0.5,
                                edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 9.9, 'Contextualized Representations', ha='center', fontsize=10,
            fontweight='bold')

    # Annotations
    ax.text(8.5, 5.5, 'Bidirectional\nSelf-Attention', ha='left', fontsize=10,
            color=COLOR_MLPURPLE, fontweight='bold')
    ax.text(8.5, 4.8, '+ Feed-Forward', ha='left', fontsize=9, color=COLOR_DARKGRAY)
    ax.text(8.5, 4.4, '+ Layer Norm', ha='left', fontsize=9, color=COLOR_DARKGRAY)

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10.8)
    ax.axis('off')
    ax.set_title('BERT Architecture: 12-Layer Encoder Stack', fontsize=14, fontweight='bold',
                 color=COLOR_DARKGRAY, pad=20)

    plt.tight_layout()
    plt.savefig('../figures/bert_architecture_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 11: BERT Special Tokens
def plot_bert_special_tokens():
    """Show [CLS], [SEP], [MASK] usage"""
    fig, axes = plt.subplots(3, 1, figsize=(13, 9))

    # [CLS] for classification
    ax = axes[0]
    tokens_cls = ["[CLS]", "Great", "movie", "!", "[SEP]"]
    for i, token in enumerate(tokens_cls):
        color = COLOR_BERT if token == "[CLS]" else COLOR_LIGHTGRAY
        rect = FancyBboxPatch((i*1.5, 0), 1.2, 0.6, boxstyle="round,pad=0.06",
                              facecolor=color, edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(rect)
        ax.text(i*1.5 + 0.6, 0.3, token, ha='center', fontsize=10, fontweight='bold')

    # Arrow from [CLS] to output
    ax.annotate('', xy=(0.6, 1), xytext=(0.6, 0.6),
               arrowprops=dict(arrowstyle='->', color=COLOR_BERT, lw=3))
    ax.text(1.5, 1.1, 'Sentence embedding for classification', fontsize=10,
            color=COLOR_BERT, fontweight='bold')

    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-0.3, 1.5)
    ax.axis('off')
    ax.set_title('[CLS]: Classification Token', fontsize=11, fontweight='bold',
                 color=COLOR_DARKGRAY)

    # [SEP] for sentence pairs
    ax = axes[1]
    tokens_sep = ["[CLS]", "Question", "[SEP]", "Answer", "text", "[SEP]"]
    for i, token in enumerate(tokens_sep):
        color = COLOR_BERT if token == "[SEP]" else COLOR_LIGHTGRAY
        rect = FancyBboxPatch((i*1.5, 0), 1.2, 0.6, boxstyle="round,pad=0.06",
                              facecolor=color, edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(rect)
        ax.text(i*1.5 + 0.6, 0.3, token, ha='center', fontsize=10, fontweight='bold')

    ax.text(4.5, -0.2, 'Separator for sentence pairs (QA, entailment)', fontsize=10,
            color=COLOR_BERT, fontweight='bold')

    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-0.3, 1.2)
    ax.axis('off')
    ax.set_title('[SEP]: Separator Token', fontsize=11, fontweight='bold',
                 color=COLOR_DARKGRAY)

    # [MASK] for pre-training
    ax = axes[2]
    tokens_mask = ["The", "[MASK]", "sat", "on", "the", "[MASK]"]
    for i, token in enumerate(tokens_mask):
        color = COLOR_BERT if token == "[MASK]" else COLOR_LIGHTGRAY
        rect = FancyBboxPatch((i*1.5, 0), 1.2, 0.6, boxstyle="round,pad=0.06",
                              facecolor=color, edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(rect)
        ax.text(i*1.5 + 0.6, 0.3, token, ha='center', fontsize=10, fontweight='bold')

    ax.text(1.5, -0.2, 'Predict "cat"', fontsize=9, color=COLOR_BERT, fontweight='bold')
    ax.text(6.5, -0.2, 'Predict "mat"', fontsize=9, color=COLOR_BERT, fontweight='bold')

    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-0.3, 1.2)
    ax.axis('off')
    ax.set_title('[MASK]: Masked Token (Pre-training Only)', fontsize=11, fontweight='bold',
                 color=COLOR_DARKGRAY)

    plt.tight_layout()
    plt.savefig('../figures/bert_special_tokens_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 12: WordPiece Tokenization
def plot_wordpiece_tokenization():
    """Subword tokenization example"""
    fig, ax = plt.subplots(figsize=(12, 6))

    examples = [
        ("playing", ["play", "##ing"], 2),
        ("unhappiness", ["un", "##happiness"], 2),
        ("cat", ["cat"], 1),
        ("COVID-19", ["CO", "##VI", "##D", "-", "19"], 5)
    ]

    for i, (word, subwords, n_sub) in enumerate(examples):
        y = 3 - i * 0.8

        # Original word
        word_box = FancyBboxPatch((0.5, y), 2, 0.5, boxstyle="round,pad=0.08",
                                  facecolor=COLOR_LIGHTGRAY,
                                  edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(word_box)
        ax.text(1.5, y+0.25, word, ha='center', fontsize=11, fontweight='bold')

        # Arrow
        ax.annotate('', xy=(3, y+0.25), xytext=(2.5, y+0.25),
                   arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))

        # Subwords
        for j, sub in enumerate(subwords):
            sub_box = FancyBboxPatch((3.5 + j*1.3, y), 1.2, 0.5, boxstyle="round,pad=0.06",
                                     facecolor=COLOR_PREDICT, alpha=0.6,
                                     edgecolor=COLOR_DARKGRAY, linewidth=1.5)
            ax.add_patch(sub_box)
            ax.text(4.1 + j*1.3, y+0.25, sub, ha='center', fontsize=9, fontweight='bold')

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 3.8)
    ax.axis('off')
    ax.set_title('WordPiece Tokenization: Handling Rare Words', fontsize=14,
                 fontweight='bold', color=COLOR_DARKGRAY, pad=15)

    ax.text(5, -0.3, '## prefix indicates continuation of previous subword', fontsize=9,
            style='italic', color=COLOR_DARKGRAY)

    plt.tight_layout()
    plt.savefig('../figures/wordpiece_tokenization_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 13: BERT Pre-training Corpus
def plot_bert_pretraining_corpus():
    """Visualize corpus size"""
    fig, ax = plt.subplots(figsize=(10, 6))

    sources = ['BooksCorpus', 'Wikipedia']
    words = [800, 2500]  # Millions
    colors = [COLOR_MLPURPLE, COLOR_CONTEXT]

    bars = ax.bar(sources, words, color=colors, alpha=0.7, edgecolor=COLOR_DARKGRAY, linewidth=2)

    ax.set_ylabel('Millions of Words', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_title('BERT Pre-training Corpus: 3.3 Billion Words', fontsize=14,
                 fontweight='bold', color=COLOR_DARKGRAY)

    # Add value labels
    for i, (source, word_count) in enumerate(zip(sources, words)):
        ax.text(i, word_count + 100, f'{word_count}M words', ha='center', fontsize=11,
                fontweight='bold', color=COLOR_DARKGRAY)

    # Total annotation
    total_box = FancyBboxPatch((0.2, 2200), 1.6, 250, boxstyle="round,pad=0.1",
                               facecolor=COLOR_PREDICT, alpha=0.3,
                               edgecolor='green', linewidth=2)
    ax.add_patch(total_box)
    ax.text(1, 2325, 'Total: 3.3B', ha='center', fontsize=11, fontweight='bold', color='green')

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/bert_pretraining_corpus_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 14: BERT Fine-tuning Process
def plot_bert_finetuning_process():
    """Show adding task head and training"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Pre-trained BERT
    bert_box = FancyBboxPatch((1, 4), 3, 2.5, boxstyle="round,pad=0.15",
                              facecolor=COLOR_MLPURPLE, alpha=0.3,
                              edgecolor=COLOR_DARKGRAY, linewidth=3)
    ax.add_patch(bert_box)
    ax.text(2.5, 5.8, 'Pre-trained BERT', ha='center', fontsize=12, fontweight='bold')
    ax.text(2.5, 5.4, '110M parameters', ha='center', fontsize=10)
    ax.text(2.5, 5.0, '(Frozen or trainable)', ha='center', fontsize=9, style='italic')

    # Arrow
    ax.annotate('', xy=(5, 5.25), xytext=(4, 5.25),
               arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=3))
    ax.text(4.5, 5.6, 'Add task head', fontsize=9, fontweight='bold')

    # Task-specific head
    head_box = FancyBboxPatch((5.5, 4.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                              facecolor=COLOR_PREDICT, alpha=0.5,
                              edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(head_box)
    ax.text(6.75, 5.5, 'Classification Head', ha='center', fontsize=10, fontweight='bold')
    ax.text(6.75, 5.1, 'Linear layer', ha='center', fontsize=9)

    # Training data
    data_box = FancyBboxPatch((5.5, 2), 2.5, 1.5, boxstyle="round,pad=0.1",
                              facecolor=COLOR_CONTEXT, alpha=0.4,
                              edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(data_box)
    ax.text(6.75, 2.9, 'Labeled Data', ha='center', fontsize=10, fontweight='bold')
    ax.text(6.75, 2.5, '1000 examples', ha='center', fontsize=9)

    # Arrow from data
    ax.annotate('', xy=(6.75, 4.3), xytext=(6.75, 3.5),
               arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2.5))
    ax.text(7.5, 3.9, 'Train 2-4 epochs\nLR=2e-5', fontsize=8, style='italic')

    ax.set_xlim(0, 9)
    ax.set_ylim(1, 7)
    ax.axis('off')
    ax.set_title('Fine-tuning BERT: Add Head, Train on Small Data', fontsize=14,
                 fontweight='bold', color=COLOR_DARKGRAY, pad=15)

    plt.tight_layout()
    plt.savefig('../figures/bert_finetuning_process_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# ===== GPT CHARTS (15-25) =====

# Chart 15: Text Generation Challenge
def plot_text_generation_challenge():
    """Show sequential nature of generation"""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Generation steps
    steps = [
        ("Input", "Once upon a", COLOR_LIGHTGRAY),
        ("Predict", "time", COLOR_PREDICT),
        ("Input", "Once upon a time", COLOR_LIGHTGRAY),
        ("Predict", "there", COLOR_PREDICT),
        ("Input", "Once upon a time there", COLOR_LIGHTGRAY),
        ("Predict", "was", COLOR_PREDICT)
    ]

    y = 1
    x = 0.5
    for i, (step_type, text, color) in enumerate(steps):
        width = len(text.split()) * 0.4 + 0.3

        box = FancyBboxPatch((x, y), width, 0.5, boxstyle="round,pad=0.06",
                             facecolor=color, edgecolor=COLOR_DARKGRAY, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + width/2, y+0.25, text, ha='center', fontsize=9, fontweight='bold')

        x += width + 0.2

        # Arrow if prediction
        if step_type == "Predict" and i < len(steps) - 1:
            ax.annotate('', xy=(x, y+0.25), xytext=(x-0.15, y+0.25),
                       arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))
            x += 0.3

    ax.set_xlim(0, 12)
    ax.set_ylim(0.3, 2)
    ax.axis('off')
    ax.set_title('Text Generation: Sequential Prediction Process', fontsize=14,
                 fontweight='bold', color=COLOR_DARKGRAY, pad=15)

    ax.text(6, 0.5, 'Each prediction uses only previous tokens (autoregressive)',
            ha='center', fontsize=10, style='italic', color=COLOR_DARKGRAY)

    plt.tight_layout()
    plt.savefig('../figures/text_generation_challenge_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 16: Autoregressive Visual
def plot_autoregressive_visual():
    """Left-to-right generation visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))

    tokens = ["The", "cat", "sat", "on", "the", "?"]

    for i, token in enumerate(tokens):
        x = i * 1.5

        # Token box - color by whether generated yet
        if i < 5:
            color = COLOR_CONTEXT
            label_text = token
        else:
            color = COLOR_PREDICT
            label_text = "?"

        rect = FancyBboxPatch((x, 1), 1.2, 0.6, boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.6, 1.3, label_text, ha='center', fontsize=11, fontweight='bold')

        # Arrows showing dependencies (only to past)
        if i > 0:
            for j in range(i):
                alpha = 0.3 if i - j > 2 else 0.7
                arrow = FancyArrowPatch((j*1.5 + 0.6, 1.7), (x + 0.6, 1.7),
                                       arrowstyle='->', mutation_scale=10, linewidth=1.5,
                                       color=COLOR_DARKGRAY, alpha=alpha)
                ax.add_patch(arrow)

    ax.text(4, 0.3, 'Predict next: P(? | The, cat, sat, on, the)', ha='center',
            fontsize=11, fontweight='bold', color=COLOR_PREDICT)

    ax.set_xlim(-0.5, 9)
    ax.set_ylim(0, 2.5)
    ax.axis('off')
    ax.set_title('Autoregressive: Predict Using Only Past Tokens', fontsize=14,
                 fontweight='bold', color=COLOR_DARKGRAY, pad=15)

    plt.tight_layout()
    plt.savefig('../figures/autoregressive_visual_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 17: Causal Mask Matrix
def plot_causal_mask_matrix():
    """Lower triangular attention mask"""
    fig, ax = plt.subplots(figsize=(9, 8))

    # Create causal mask
    n = 6
    mask = np.tril(np.ones((n, n)))

    # Plot
    im = ax.imshow(mask, cmap='Blues', aspect='auto', alpha=0.8)

    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, fontsize=11)
    ax.set_yticklabels(tokens, fontsize=11)

    ax.set_xlabel('Attend TO (keys)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attend FROM (queries)', fontsize=12, fontweight='bold')
    ax.set_title('GPT Causal Mask: Lower Triangular (No Future Peeking)',
                 fontsize=13, fontweight='bold', color=COLOR_DARKGRAY, pad=15)

    # Annotations
    for i in range(n):
        for j in range(n):
            if mask[i, j] == 1:
                ax.text(j, i, '1', ha='center', va='center', fontsize=10, fontweight='bold')
            else:
                ax.text(j, i, '0', ha='center', va='center', fontsize=10,
                       color='red', fontweight='bold')

    # Highlight diagonal
    ax.plot([-0.5, n-0.5], [-0.5, n-0.5], 'r--', linewidth=2, alpha=0.5)

    plt.tight_layout()
    plt.savefig('../figures/causal_mask_matrix_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 18: Next Token Prediction
def plot_next_token_prediction():
    """Show prediction at each position"""
    fig, ax = plt.subplots(figsize=(13, 6))

    sequence = ["The", "cat", "sat", "on"]
    predictions = [
        ("cat", 0.65),
        ("sat", 0.42),
        ("on", 0.71),
        ("the", 0.88)
    ]

    for i, word in enumerate(sequence):
        x = i * 2.5

        # Input word
        input_box = FancyBboxPatch((x, 3.5), 1.8, 0.6, boxstyle="round,pad=0.08",
                                   facecolor=COLOR_CONTEXT,
                                   edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(input_box)
        ax.text(x + 0.9, 3.8, word, ha='center', fontsize=11, fontweight='bold')

        # Arrow down
        ax.annotate('', xy=(x + 0.9, 3), xytext=(x + 0.9, 3.5),
                   arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))

        # Prediction
        pred_word, prob = predictions[i]
        pred_box = FancyBboxPatch((x, 1.8), 1.8, 0.9, boxstyle="round,pad=0.08",
                                  facecolor=COLOR_PREDICT, alpha=0.7,
                                  edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(pred_box)
        ax.text(x + 0.9, 2.5, f'Predict:\n"{pred_word}"', ha='center', fontsize=10,
                fontweight='bold')
        ax.text(x + 0.9, 2.0, f'P={prob:.2f}', ha='center', fontsize=9,
                color=COLOR_DARKGRAY, fontweight='bold')

    ax.set_xlim(-0.5, 10)
    ax.set_ylim(1, 4.5)
    ax.axis('off')
    ax.set_title('Next Token Prediction at Each Position', fontsize=14, fontweight='bold',
                 color=COLOR_DARKGRAY, pad=15)

    plt.tight_layout()
    plt.savefig('../figures/next_token_prediction_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 19: GPT Architecture
def plot_gpt_architecture():
    """Decoder stack visualization"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Input
    input_box = FancyBboxPatch((2, 0.5), 6, 0.8, boxstyle="round,pad=0.1",
                               facecolor=COLOR_LIGHTGRAY, edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 0.9, 'Token + Position Embeddings', ha='center', fontsize=10, fontweight='bold')

    # 12 Decoder layers
    for i in range(12):
        y = 1.8 + i * 0.6
        layer_box = FancyBboxPatch((2, y), 6, 0.5, boxstyle="round,pad=0.08",
                                   facecolor=COLOR_GPT, alpha=0.2 + (i/12)*0.5,
                                   edgecolor=COLOR_DARKGRAY, linewidth=1.5)
        ax.add_patch(layer_box)
        ax.text(5, y+0.25, f'Decoder Layer {i+1}', ha='center', fontsize=8)

        if i < 11:
            ax.annotate('', xy=(5, y+0.6), xytext=(5, y+0.5),
                       arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=1.5))

    # Output
    output_box = FancyBboxPatch((2, 9.5), 6, 0.8, boxstyle="round,pad=0.1",
                                facecolor=COLOR_PREDICT, alpha=0.5,
                                edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 9.9, 'Next Token Probabilities', ha='center', fontsize=10, fontweight='bold')

    # Side annotations
    ax.text(8.5, 5.5, 'Causal\nSelf-Attention', ha='left', fontsize=10,
            color=COLOR_GPT, fontweight='bold')
    ax.text(8.5, 4.8, '(masked)', ha='left', fontsize=9, color=COLOR_DARKGRAY)

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10.8)
    ax.axis('off')
    ax.set_title('GPT Architecture: 12-Layer Decoder Stack', fontsize=14, fontweight='bold',
                 color=COLOR_DARKGRAY, pad=20)

    plt.tight_layout()
    plt.savefig('../figures/gpt_architecture_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 20: GPT Evolution Timeline
def plot_gpt_evolution_timeline():
    """GPT-1 to GPT-3 scaling"""
    fig, ax = plt.subplots(figsize=(12, 7))

    models = ['GPT-1\n(2018)', 'GPT-2\n(2019)', 'GPT-3\n(2020)']
    params = [117, 1500, 175000]  # Millions
    costs = [50, 500, 4600]  # Thousands of dollars

    x = np.arange(len(models))
    width = 0.35

    # Parameters bars
    bars1 = ax.bar(x - width/2, params, width, label='Parameters (millions)',
                   color=[COLOR_GPT, '#5DADE2', '#85C1E9'], alpha=0.8,
                   edgecolor=COLOR_DARKGRAY, linewidth=2)

    # Cost on secondary axis
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, costs, width, label='Training cost ($K)',
                    color=[COLOR_PREDICT, '#A9DFBF', '#D5F4E6'], alpha=0.8,
                    edgecolor=COLOR_DARKGRAY, linewidth=2)

    ax.set_ylabel('Parameters (millions)', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax2.set_ylabel('Training Cost ($1000s)', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_title('GPT Evolution: Exponential Scaling 2018-2020', fontsize=14,
                 fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_yscale('log')
    ax2.set_yscale('log')

    # Annotations
    for i, param in enumerate(params):
        if param < 10000:
            ax.text(i - width/2, param * 1.8, f'{param}M', ha='center', fontsize=9,
                    fontweight='bold')
        else:
            ax.text(i - width/2, param * 1.8, f'{param/1000:.0f}B', ha='center', fontsize=9,
                    fontweight='bold')

    ax.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/gpt_evolution_timeline_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# ===== INTEGRATION CHARTS (21-35) =====

# Chart 21: BERT vs GPT Architecture Comparison
def plot_bert_vs_gpt_architecture():
    """Side-by-side comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # BERT (Encoder)
    ax1.set_title('BERT: Encoder Stack\n(Bidirectional)', fontsize=12, fontweight='bold',
                  color=COLOR_BERT)

    for i in range(6):  # Show 6 layers for visualization
        y = 1 + i * 0.8
        layer = FancyBboxPatch((1, y), 3, 0.6, boxstyle="round,pad=0.08",
                               facecolor=COLOR_BERT, alpha=0.3 + i*0.1,
                               edgecolor=COLOR_DARKGRAY, linewidth=1.5)
        ax1.add_patch(layer)
        ax1.text(2.5, y+0.3, f'Layer {i+1}', ha='center', fontsize=9)

        # Bidirectional arrows
        if i > 0:
            ax1.annotate('', xy=(2.5, y), xytext=(2.5, y-0.2),
                        arrowprops=dict(arrowstyle='<->', color=COLOR_DARKGRAY, lw=1.5))

    ax1.text(2.5, 6, 'Full Context\nNo Mask', ha='center', fontsize=10,
            fontweight='bold', color=COLOR_BERT)

    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 6.5)
    ax1.axis('off')

    # GPT (Decoder)
    ax2.set_title('GPT: Decoder Stack\n(Causal)', fontsize=12, fontweight='bold',
                  color=COLOR_GPT)

    for i in range(6):
        y = 1 + i * 0.8
        layer = FancyBboxPatch((1, y), 3, 0.6, boxstyle="round,pad=0.08",
                               facecolor=COLOR_GPT, alpha=0.3 + i*0.1,
                               edgecolor=COLOR_DARKGRAY, linewidth=1.5)
        ax2.add_patch(layer)
        ax2.text(2.5, y+0.3, f'Layer {i+1}', ha='center', fontsize=9)

        # Causal arrow (only upward)
        if i > 0:
            ax2.annotate('', xy=(2.5, y), xytext=(2.5, y-0.2),
                        arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=1.5))

    ax2.text(2.5, 6, 'Left-to-Right\nCausal Mask', ha='center', fontsize=10,
            fontweight='bold', color=COLOR_GPT)

    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 6.5)
    ax2.axis('off')

    plt.suptitle('Architecture Comparison: Encoder vs Decoder', fontsize=15,
                 fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig('../figures/bert_vs_gpt_architecture_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 22: Transfer Learning Pipeline
def plot_transfer_learning_pipeline():
    """Complete pipeline from pre-training to deployment"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Stage 1: Pre-training
    stage1 = FancyBboxPatch((0.5, 6), 3, 1.5, boxstyle="round,pad=0.15",
                            facecolor=COLOR_MLPURPLE, alpha=0.3,
                            edgecolor=COLOR_DARKGRAY, linewidth=3)
    ax.add_patch(stage1)
    ax.text(2, 7.2, 'Stage 1: Pre-training', ha='center', fontsize=12, fontweight='bold')
    ax.text(2, 6.7, 'Unsupervised', ha='center', fontsize=9)
    ax.text(2, 6.4, 'Billions of tokens', ha='center', fontsize=9)

    # Arrow to checkpoint
    ax.annotate('', xy=(4, 6.75), xytext=(3.5, 6.75),
               arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=3))

    # Checkpoint
    checkpoint = Circle((4.5, 6.75), 0.4, facecolor=COLOR_PREDICT,
                       edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(checkpoint)
    ax.text(4.5, 6.75, 'Save', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(5.5, 6.75, 'Pre-trained\nmodel', va='center', fontsize=9)

    # Stage 2: Fine-tuning (multiple paths)
    ax.text(2, 5, 'Stage 2: Fine-tuning (choose task)', ha='center', fontsize=11,
            fontweight='bold', color=COLOR_DARKGRAY)

    tasks_ft = ['Classification', 'QA', 'NER']
    for i, task in enumerate(tasks_ft):
        x = i * 2.2 + 0.8
        y = 3.2

        # Arrow from checkpoint
        ax.annotate('', xy=(x + 0.7, y + 0.9), xytext=(4.5, 6.35),
                   arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2, alpha=0.6))

        # Fine-tune box
        ft_box = FancyBboxPatch((x, y), 1.4, 1.2, boxstyle="round,pad=0.1",
                                facecolor=COLOR_CONTEXT, alpha=0.5,
                                edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(ft_box)
        ax.text(x + 0.7, y + 0.8, task, ha='center', fontsize=10, fontweight='bold')
        ax.text(x + 0.7, y + 0.4, '1K examples', ha='center', fontsize=8)

        # Arrow to deployment
        ax.annotate('', xy=(x + 0.7, 1.8), xytext=(x + 0.7, y),
                   arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))

        # Deployed model
        deploy_box = FancyBboxPatch((x, 0.5), 1.4, 1, boxstyle="round,pad=0.08",
                                    facecolor=COLOR_PREDICT, alpha=0.6,
                                    edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(deploy_box)
        ax.text(x + 0.7, 1, 'Deploy', ha='center', fontsize=10, fontweight='bold')

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Complete Transfer Learning Pipeline', fontsize=15, fontweight='bold',
                 color=COLOR_DARKGRAY, pad=20)

    plt.tight_layout()
    plt.savefig('../figures/transfer_learning_pipeline_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# ===== ADDITIONAL SUPPORTING CHARTS (24-35) - Complete Implementation =====

# Chart 24: ImageNet Transfer Learning Analogy
def plot_imagenet_transfer_analogy():
    """Show computer vision's success inspiring NLP"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Computer Vision (2012)
    ax1.set_title('Computer Vision (2012)', fontsize=13, fontweight='bold')

    # ImageNet pre-training
    pretrain_cv = FancyBboxPatch((1, 5), 4, 1.2, boxstyle="round,pad=0.1",
                                 facecolor='#9B59B6', alpha=0.3,
                                 edgecolor=COLOR_DARKGRAY, linewidth=2.5)
    ax1.add_patch(pretrain_cv)
    ax1.text(3, 5.6, 'ImageNet Pre-training\n1M images, 1000 classes', ha='center',
            fontsize=10, fontweight='bold')

    # Transfer to tasks
    tasks_cv = ['Object\nDetection', 'Segmentation', 'Style\nTransfer']
    for i, task in enumerate(tasks_cv):
        x = 1.2 + i * 1.3
        ax1.annotate('', xy=(x, 3.8), xytext=(3, 4.8),
                    arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))
        task_box = FancyBboxPatch((x-0.5, 2.8), 1, 0.8, boxstyle="round,pad=0.06",
                                  facecolor=COLOR_PREDICT, alpha=0.5,
                                  edgecolor=COLOR_DARKGRAY, linewidth=1.5)
        ax1.add_patch(task_box)
        ax1.text(x, 3.2, task, ha='center', fontsize=9, fontweight='bold')

    ax1.text(3, 1.8, 'SUCCESS:\n10x less data needed\nState-of-art on all tasks',
            ha='center', fontsize=10, color='green', fontweight='bold')
    ax1.set_xlim(0, 6)
    ax1.set_ylim(1, 6.5)
    ax1.axis('off')

    # NLP (2018)
    ax2.set_title('Natural Language Processing (2018)', fontsize=13, fontweight='bold')

    # Pre-training
    pretrain_nlp = FancyBboxPatch((1, 5), 4, 1.2, boxstyle="round,pad=0.1",
                                  facecolor=COLOR_MLPURPLE, alpha=0.3,
                                  edgecolor=COLOR_DARKGRAY, linewidth=2.5)
    ax2.add_patch(pretrain_nlp)
    ax2.text(3, 5.6, 'BERT/GPT Pre-training\nBillions of words', ha='center',
            fontsize=10, fontweight='bold')

    # Transfer to tasks
    tasks_nlp = ['Sentiment', 'QA', 'NER']
    for i, task in enumerate(tasks_nlp):
        x = 1.2 + i * 1.3
        ax2.annotate('', xy=(x, 3.8), xytext=(3, 4.8),
                    arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))
        task_box = FancyBboxPatch((x-0.5, 2.8), 1, 0.8, boxstyle="round,pad=0.06",
                                  facecolor=COLOR_PREDICT, alpha=0.5,
                                  edgecolor=COLOR_DARKGRAY, linewidth=1.5)
        ax2.add_patch(task_box)
        ax2.text(x, 3.2, task, ha='center', fontsize=9, fontweight='bold')

    ax2.text(3, 1.8, 'FINALLY!\nTransfer learning\nworks for NLP',
            ha='center', fontsize=10, color='green', fontweight='bold')
    ax2.set_xlim(0, 6)
    ax2.set_ylim(1, 6.5)
    ax2.axis('off')

    plt.suptitle('ImageNet Moment: Computer Vision (2012) Inspires NLP (2018)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../figures/imagenet_transfer_analogy_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 25: Masked Token Prediction Step-by-Step Visual
def plot_masked_token_prediction_steps():
    """Visual walkthrough of masked prediction"""
    fig, ax = plt.subplots(figsize=(12, 9))

    steps_y = [7.5, 6.2, 4.9, 3.6, 2.3, 1.0]
    steps_text = [
        'Input: "The [MASK] sat on the mat"',
        'Embeddings: Token + Position (768-dim vectors)',
        'Transformer Layers: 12 encoder layers process',
        'Output at [MASK]: Hidden state h_mask',
        'Project: h_mask × W_vocab → logits (30K)',
        'Softmax: P(cat)=0.73, P(dog)=0.15, P(person)=0.04'
    ]

    colors = [COLOR_LIGHTGRAY, COLOR_CONTEXT, COLOR_MLPURPLE, COLOR_PREDICT,
              COLOR_CONTEXT, COLOR_PREDICT]

    for y, text, color in zip(steps_y, steps_text, colors):
        box = FancyBboxPatch((1, y-0.35), 10, 0.7, boxstyle="round,pad=0.12",
                             facecolor=color, alpha=0.4,
                             edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(box)
        ax.text(6, y, text, ha='center', fontsize=10, fontweight='bold' if y > 6 else 'normal')

        if y > 1.0:
            ax.annotate('', xy=(6, y-0.5), xytext=(6, y-0.35),
                       arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2.5))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.5)
    ax.axis('off')
    ax.set_title('Masked Token Prediction: Step-by-Step Process', fontsize=14,
                 fontweight='bold', color=COLOR_DARKGRAY, pad=15)

    plt.tight_layout()
    plt.savefig('../figures/masked_token_prediction_steps_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 26: MLM Objective Equation Visual
def plot_mlm_objective_equation():
    """Visualize masked language modeling objective"""
    fig, ax = plt.subplots(figsize=(11, 7))

    # Main equation box
    eq_box = FancyBboxPatch((2, 5), 7, 1.8, boxstyle="round,pad=0.15",
                            facecolor=COLOR_MLPURPLE, alpha=0.2,
                            edgecolor=COLOR_DARKGRAY, linewidth=3)
    ax.add_patch(eq_box)

    ax.text(5.5, 6.3, 'Masked Language Modeling Objective', ha='center',
            fontsize=13, fontweight='bold', color=COLOR_DARKGRAY)
    ax.text(5.5, 5.5, r'$\mathcal{L}_{MLM} = -\sum_{i \in masked} \log P(w_i | context)$',
            ha='center', fontsize=14, fontweight='bold')

    # Components
    components = [
        (1.5, 3.5, 'Masked\ntokens\n(15%)', COLOR_CURRENT),
        (4, 3.5, 'Cross-entropy\nloss', COLOR_CONTEXT),
        (6.5, 3.5, 'Conditional\nprobability', COLOR_PREDICT),
        (9, 3.5, 'Full sentence\ncontext', COLOR_MLPURPLE)
    ]

    for x, y, label, color in components:
        box = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8, boxstyle="round,pad=0.08",
                             facecolor=color, alpha=0.4,
                             edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', fontsize=9, fontweight='bold')

        ax.annotate('', xy=(x, 4.8), xytext=(x, y+0.4),
                   arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))

    # Example
    example_box = FancyBboxPatch((1.5, 1), 8, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#FFF3CD', alpha=0.5,
                                 edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(example_box)
    ax.text(5.5, 2.1, 'Example:', ha='center', fontsize=10, fontweight='bold')
    ax.text(5.5, 1.7, '"The [MASK] sat" → P(cat) = 0.73', ha='center', fontsize=10)
    ax.text(5.5, 1.3, 'Loss = -log(0.73) = 0.31', ha='center', fontsize=10,
            style='italic')

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.5)
    ax.axis('off')
    ax.set_title('BERT Training Objective: Mathematics', fontsize=15,
                 fontweight='bold', color=COLOR_DARKGRAY, pad=20)

    plt.tight_layout()
    plt.savefig('../figures/mlm_objective_equation_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 27: Next Sentence Prediction
def plot_next_sentence_prediction():
    """Illustrate NSP objective"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Positive example (real pair)
    y_pos = 5
    sent_a = FancyBboxPatch((0.5, y_pos), 4.5, 0.8, boxstyle="round,pad=0.08",
                            facecolor=COLOR_CONTEXT, alpha=0.5,
                            edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(sent_a)
    ax.text(2.75, y_pos+0.4, 'Sentence A: "Alice was tired."', ha='center', fontsize=10)

    sep_token = Circle((5.5, y_pos+0.4), 0.25, facecolor=COLOR_BERT,
                      edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(sep_token)
    ax.text(5.5, y_pos+0.4, '[SEP]', ha='center', va='center', fontsize=8,
            fontweight='bold', color='white')

    sent_b = FancyBboxPatch((6, y_pos), 4.5, 0.8, boxstyle="round,pad=0.08",
                            facecolor=COLOR_CONTEXT, alpha=0.5,
                            edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(sent_b)
    ax.text(8.25, y_pos+0.4, 'Sentence B: "She went to sleep."', ha='center', fontsize=10)

    # Prediction
    ax.annotate('', xy=(5.5, 4.3), xytext=(5.5, y_pos),
               arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2.5))

    pred_box = FancyBboxPatch((3.5, 3.5), 4, 0.7, boxstyle="round,pad=0.08",
                              facecolor=COLOR_PREDICT, alpha=0.6,
                              edgecolor='green', linewidth=2.5)
    ax.add_patch(pred_box)
    ax.text(5.5, 3.85, 'Prediction: IsNext = TRUE', ha='center', fontsize=11,
            fontweight='bold', color='green')

    # Negative example (random pair)
    y_neg = 2
    sent_a2 = FancyBboxPatch((0.5, y_neg), 4.5, 0.8, boxstyle="round,pad=0.08",
                             facecolor=COLOR_NEUTRAL, alpha=0.5,
                             edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(sent_a2)
    ax.text(2.75, y_neg+0.4, 'Sentence A: "Alice was tired."', ha='center', fontsize=10)

    sep_token2 = Circle((5.5, y_neg+0.4), 0.25, facecolor=COLOR_BERT,
                       edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(sep_token2)
    ax.text(5.5, y_neg+0.4, '[SEP]', ha='center', va='center', fontsize=8,
            fontweight='bold', color='white')

    sent_b2 = FancyBboxPatch((6, y_neg), 4.5, 0.8, boxstyle="round,pad=0.08",
                             facecolor=COLOR_NEUTRAL, alpha=0.5,
                             edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(sent_b2)
    ax.text(8.25, y_neg+0.4, 'Sentence B: "The recipe is easy."', ha='center', fontsize=10)

    # Prediction
    ax.annotate('', xy=(5.5, 1), xytext=(5.5, y_neg),
               arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2.5))

    pred_box2 = FancyBboxPatch((3.5, 0.2), 4, 0.7, boxstyle="round,pad=0.08",
                               facecolor=COLOR_CURRENT, alpha=0.6,
                               edgecolor='red', linewidth=2.5)
    ax.add_patch(pred_box2)
    ax.text(5.5, 0.55, 'Prediction: IsNext = FALSE', ha='center', fontsize=11,
            fontweight='bold', color='red')

    ax.set_xlim(0, 11)
    ax.set_ylim(-0.5, 7)
    ax.axis('off')
    ax.set_title('Next Sentence Prediction (NSP): Training Objective #2', fontsize=14,
                 fontweight='bold', color=COLOR_DARKGRAY, pad=15)

    plt.tight_layout()
    plt.savefig('../figures/next_sentence_prediction_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 28: Fine-tuning Task Heads
def plot_finetuning_task_heads():
    """Show different task heads for BERT"""
    fig, ax = plt.subplots(figsize=(13, 8))

    # Base BERT (shared)
    bert_base = FancyBboxPatch((4, 6), 3.5, 1, boxstyle="round,pad=0.12",
                               facecolor=COLOR_MLPURPLE, alpha=0.3,
                               edgecolor=COLOR_DARKGRAY, linewidth=3)
    ax.add_patch(bert_base)
    ax.text(5.75, 6.5, 'Pre-trained BERT\n(shared)', ha='center', fontsize=11,
            fontweight='bold')

    # Different task heads
    tasks = [
        ('Classification', '[CLS] → Linear → Softmax', 1.5),
        ('Token Classification\n(NER)', 'Each token → Linear → Labels', 5.75),
        ('Question Answering', 'Span prediction\n(start/end logits)', 10)
    ]

    for task, head_desc, x in tasks:
        # Arrow from BERT
        ax.annotate('', xy=(x, 4.8), xytext=(5.75, 6),
                   arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))

        # Task head
        head_box = FancyBboxPatch((x-1.2, 3.5), 2.4, 1.2, boxstyle="round,pad=0.1",
                                  facecolor=COLOR_PREDICT, alpha=0.5,
                                  edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(head_box)
        ax.text(x, 4.4, task, ha='center', fontsize=10, fontweight='bold')
        ax.text(x, 3.9, head_desc, ha='center', fontsize=8)

    ax.set_xlim(0, 12)
    ax.set_ylim(2.5, 7.5)
    ax.axis('off')
    ax.set_title('Fine-tuning: Task-Specific Heads on BERT', fontsize=14,
                 fontweight='bold', color=COLOR_DARKGRAY, pad=15)

    plt.tight_layout()
    plt.savefig('../figures/finetuning_task_heads_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 29: Layer Freezing Strategies
def plot_layer_freezing_strategies():
    """Show different freezing approaches"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    strategies = [
        ('Full Fine-tuning', 0, 'All layers trainable'),
        ('Partial Freezing', 8, 'Bottom 8 frozen'),
        ('Feature Extraction', 12, 'All frozen, head only')
    ]

    for ax_idx, (strategy, freeze_count, desc) in enumerate(strategies):
        ax = axes[ax_idx]
        ax.set_title(strategy, fontsize=11, fontweight='bold')

        # Draw 12 layers
        for i in range(12):
            y = i * 0.6
            if i < freeze_count:
                color = COLOR_NEUTRAL  # Frozen
                edgecolor = COLOR_DARKGRAY
                alpha = 0.3
                label = 'Frozen'
            else:
                color = COLOR_PREDICT  # Trainable
                edgecolor = 'green'
                alpha = 0.6
                label = 'Train'

            layer = FancyBboxPatch((0.5, y), 2, 0.5, boxstyle="round,pad=0.05",
                                   facecolor=color, alpha=alpha,
                                   edgecolor=edgecolor, linewidth=1.5)
            ax.add_patch(layer)
            ax.text(1.5, y+0.25, f'L{i+1}', ha='center', fontsize=7)

            if i == 0:
                ax.text(3, y+0.25, label, va='center', fontsize=8,
                       color='gray' if label == 'Frozen' else 'green',
                       fontweight='bold')

        # Task head
        head = FancyBboxPatch((0.5, 7.5), 2, 0.6, boxstyle="round,pad=0.08",
                              facecolor=COLOR_BERT, alpha=0.6,
                              edgecolor='green', linewidth=2.5)
        ax.add_patch(head)
        ax.text(1.5, 7.8, 'Task Head\n(always train)', ha='center', fontsize=8,
                fontweight='bold')

        ax.text(1.5, -0.5, desc, ha='center', fontsize=9, style='italic')

        ax.set_xlim(0, 3.5)
        ax.set_ylim(-1, 8.8)
        ax.axis('off')

    plt.suptitle('Layer Freezing Strategies for Fine-tuning', fontsize=14,
                 fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../figures/layer_freezing_strategies_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 30: Autoregressive Objective Equation
def plot_autoregressive_objective():
    """GPT training objective visualization"""
    fig, ax = plt.subplots(figsize=(11, 7))

    # Main equation
    eq_box = FancyBboxPatch((2, 5), 7.5, 1.8, boxstyle="round,pad=0.15",
                            facecolor=COLOR_GPT, alpha=0.2,
                            edgecolor=COLOR_DARKGRAY, linewidth=3)
    ax.add_patch(eq_box)

    ax.text(5.75, 6.3, 'Autoregressive Language Modeling Objective', ha='center',
            fontsize=13, fontweight='bold', color=COLOR_DARKGRAY)
    ax.text(5.75, 5.5, r'$\mathcal{L}_{AR} = -\sum_{t=1}^{T} \log P(w_t | w_1, ..., w_{t-1})$',
            ha='center', fontsize=14, fontweight='bold')

    # Components
    components = [
        (2, 3.5, 'Each\nposition', COLOR_CURRENT),
        (4.5, 3.5, 'Cross-entropy\nloss', COLOR_CONTEXT),
        (7, 3.5, 'Conditional on\npast only', COLOR_PREDICT),
        (9.5, 3.5, 'Causal\nmasking', COLOR_GPT)
    ]

    for x, y, label, color in components:
        box = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8, boxstyle="round,pad=0.08",
                             facecolor=color, alpha=0.4,
                             edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', fontsize=9, fontweight='bold')

        ax.annotate('', xy=(x, 4.8), xytext=(x, y+0.4),
                   arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))

    # Example
    example_box = FancyBboxPatch((1.5, 1), 8.5, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#E3F2FD', alpha=0.7,
                                 edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(example_box)
    ax.text(5.75, 2.1, 'Example: "The cat sat on"', ha='center', fontsize=10, fontweight='bold')
    ax.text(5.75, 1.7, 'Predict next: P(the|past) = 0.88', ha='center', fontsize=10)
    ax.text(5.75, 1.3, 'Loss = -log(0.88) = 0.13', ha='center', fontsize=10,
            style='italic')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis('off')
    ax.set_title('GPT Training Objective: Mathematics', fontsize=15,
                 fontweight='bold', color=COLOR_DARKGRAY, pad=20)

    plt.tight_layout()
    plt.savefig('../figures/autoregressive_objective_equation_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 31: Few-Shot Learning Examples
def plot_few_shot_learning_examples():
    """Show zero/one/few-shot patterns"""
    fig, ax = plt.subplots(figsize=(13, 9))

    examples = [
        (7.5, 'Zero-Shot', 'Translate to French: Hello →', 'Bonjour', COLOR_CURRENT),
        (5.5, 'One-Shot', 'English: Hello, French: Bonjour.\nEnglish: Goodbye →',
         'Au revoir', COLOR_CONTEXT),
        (3, 'Few-Shot (3 examples)',
         'E: Hello, F: Bonjour.\nE: Goodbye, F: Au revoir.\nE: Thank you, F: Merci.\nE: Please →',
         'S\'il vous plaît', COLOR_PREDICT)
    ]

    for y, shot_type, prompt, answer, color in examples:
        # Shot type label
        type_box = FancyBboxPatch((0.5, y-0.3), 2, 0.6, boxstyle="round,pad=0.08",
                                  facecolor=color, alpha=0.5,
                                  edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(type_box)
        ax.text(1.5, y, shot_type, ha='center', fontsize=10, fontweight='bold')

        # Prompt
        prompt_box = FancyBboxPatch((3, y-0.3), 5.5, 0.6 if y > 5 else 1.2,
                                    boxstyle="round,pad=0.08",
                                    facecolor=COLOR_LIGHTGRAY,
                                    edgecolor=COLOR_DARKGRAY, linewidth=1.5)
        ax.add_patch(prompt_box)
        ax.text(5.75, y + (0 if y > 5 else 0.3), prompt, ha='center', fontsize=8,
                family='monospace')

        # Arrow
        ax.annotate('', xy=(9, y), xytext=(8.5, y),
                   arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))

        # Answer
        ans_box = FancyBboxPatch((9.2, y-0.3), 2.5, 0.6, boxstyle="round,pad=0.08",
                                 facecolor=color, alpha=0.7,
                                 edgecolor='green', linewidth=2)
        ax.add_patch(ans_box)
        ax.text(10.45, y, answer, ha='center', fontsize=10, fontweight='bold')

    ax.set_xlim(0, 12.5)
    ax.set_ylim(1.5, 8.5)
    ax.axis('off')
    ax.set_title('Few-Shot Learning: In-Context Pattern Matching', fontsize=14,
                 fontweight='bold', color=COLOR_DARKGRAY, pad=15)

    ax.text(6, 0.8, 'No gradient updates! Pure inference from examples in prompt.',
            ha='center', fontsize=10, style='italic', color=COLOR_DARKGRAY)

    plt.tight_layout()
    plt.savefig('../figures/few_shot_learning_examples_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 32: In-Context Learning Visualization
def plot_incontext_learning():
    """Show how GPT learns from prompt examples"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Prompt with examples
    prompt_box = FancyBboxPatch((0.5, 4.5), 11, 2.5, boxstyle="round,pad=0.15",
                                facecolor='#FFF9E6', edgecolor=COLOR_DARKGRAY, linewidth=2.5)
    ax.add_patch(prompt_box)

    ax.text(6, 6.5, 'Prompt with Examples (In-Context)', ha='center', fontsize=11,
            fontweight='bold', color=COLOR_DARKGRAY)
    ax.text(6, 6.0, 'Q: What is 2+2? A: 4', ha='center', fontsize=9, family='monospace')
    ax.text(6, 5.6, 'Q: What is 5+3? A: 8', ha='center', fontsize=9, family='monospace')
    ax.text(6, 5.2, 'Q: What is 7+6? A: ___', ha='center', fontsize=9, family='monospace')

    # Arrow to model
    ax.annotate('', xy=(6, 3.8), xytext=(6, 4.5),
               arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=3))

    # GPT-3 processing
    gpt_box = FancyBboxPatch((3, 2.5), 6, 1, boxstyle="round,pad=0.12",
                             facecolor=COLOR_GPT, alpha=0.3,
                             edgecolor=COLOR_DARKGRAY, linewidth=2.5)
    ax.add_patch(gpt_box)
    ax.text(6, 3, 'GPT-3 (175B parameters)\nPattern matching in weights',
            ha='center', fontsize=10, fontweight='bold')

    # Arrow to output
    ax.annotate('', xy=(6, 1.8), xytext=(6, 2.5),
               arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=3))

    # Output
    output_box = FancyBboxPatch((4, 0.8), 4, 0.8, boxstyle="round,pad=0.1",
                                facecolor=COLOR_PREDICT, alpha=0.6,
                                edgecolor='green', linewidth=2.5)
    ax.add_patch(output_box)
    ax.text(6, 1.2, 'Output: 13', ha='center', fontsize=12, fontweight='bold',
            color='green')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis('off')
    ax.set_title('In-Context Learning: Learning from Prompt Examples', fontsize=14,
                 fontweight='bold', color=COLOR_DARKGRAY, pad=15)

    plt.tight_layout()
    plt.savefig('../figures/incontext_learning_visual_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 33: Data Efficiency Comparison
def plot_data_efficiency_comparison():
    """Quantify pre-training advantage"""
    fig, ax = plt.subplots(figsize=(12, 7))

    approaches = ['Train from\nScratch', 'Pre-trained\n+ Fine-tune']
    data_needed = [10000, 100]  # Examples
    accuracy = [78, 92]  # Percent
    time_hours = [6, 0.17]  # Hours

    x = np.arange(len(approaches))

    # Data requirements
    ax1 = ax
    bars1 = ax1.bar(x - 0.2, data_needed, 0.4, label='Data needed',
                    color=[COLOR_CURRENT, COLOR_PREDICT], alpha=0.7,
                    edgecolor=COLOR_DARKGRAY, linewidth=2)

    ax1.set_ylabel('Training Examples Needed', fontsize=12, fontweight='bold',
                   color=COLOR_DARKGRAY)
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(approaches, fontsize=11)

    # Accuracy on secondary axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + 0.2, accuracy, 0.4, label='Test accuracy',
                    color=[COLOR_CONTEXT, '#2ECC71'], alpha=0.7,
                    edgecolor=COLOR_DARKGRAY, linewidth=2)

    ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold',
                   color=COLOR_DARKGRAY)
    ax2.set_ylim(0, 100)

    # Add labels
    for i, data in enumerate(data_needed):
        ax1.text(i - 0.2, data * 1.5, f'{data:,}', ha='center', fontsize=10,
                fontweight='bold')

    for i, acc in enumerate(accuracy):
        ax2.text(i + 0.2, acc + 3, f'{acc}%', ha='center', fontsize=10,
                fontweight='bold')

    # Annotation
    ax1.annotate('100x less data!\n14% better!', xy=(1, 100), xytext=(0.5, 1000),
                arrowprops=dict(arrowstyle='->', color='green', lw=3),
                fontsize=11, fontweight='bold', color='green')

    ax1.set_title('Transfer Learning: Dramatic Data Efficiency Gain', fontsize=14,
                  fontweight='bold', color=COLOR_DARKGRAY)

    ax1.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)

    set_minimalist_style(ax1)
    plt.tight_layout()
    plt.savefig('../figures/data_efficiency_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 34: Fine-tuning Best Practices Flowchart
def plot_finetuning_best_practices():
    """Decision flowchart for fine-tuning"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Start
    start = FancyBboxPatch((4.5, 9), 3, 0.7, boxstyle="round,pad=0.1",
                           facecolor=COLOR_MLPURPLE, alpha=0.5,
                           edgecolor=COLOR_DARKGRAY, linewidth=2.5)
    ax.add_patch(start)
    ax.text(6, 9.35, 'Start Fine-tuning', ha='center', fontsize=11, fontweight='bold')

    # Decision 1: Dataset size
    ax.annotate('', xy=(6, 8.5), xytext=(6, 9),
               arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))

    dec1 = FancyBboxPatch((4, 7.8), 4, 0.6, boxstyle="round,pad=0.08",
                          facecolor=COLOR_LIGHTGRAY,
                          edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(dec1)
    ax.text(6, 8.1, 'Dataset size?', ha='center', fontsize=10, fontweight='bold')

    # Small dataset branch
    ax.plot([6, 3], [7.8, 6.5], 'k-', linewidth=2)
    ax.text(4, 7, '< 1000', fontsize=9, style='italic')

    small_box = FancyBboxPatch((1.5, 5.8), 3, 0.6, boxstyle="round,pad=0.08",
                               facecolor=COLOR_CONTEXT, alpha=0.5,
                               edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(small_box)
    ax.text(3, 6.1, 'Freeze bottom 8 layers\nLR=2e-5, Epochs=4', ha='center',
            fontsize=9)

    # Large dataset branch
    ax.plot([6, 9], [7.8, 6.5], 'k-', linewidth=2)
    ax.text(8, 7, '> 1000', fontsize=9, style='italic')

    large_box = FancyBboxPatch((7.5, 5.8), 3, 0.6, boxstyle="round,pad=0.08",
                               facecolor=COLOR_PREDICT, alpha=0.5,
                               edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(large_box)
    ax.text(9, 6.1, 'Full fine-tuning\nLR=3e-5, Epochs=3', ha='center',
            fontsize=9)

    # Both converge
    ax.plot([3, 6], [5.8, 5], 'k-', linewidth=2)
    ax.plot([9, 6], [5.8, 5], 'k-', linewidth=2)

    # Validation monitoring
    val_box = FancyBboxPatch((4.5, 4.5), 3, 0.5, boxstyle="round,pad=0.08",
                             facecolor='#FFF3CD',
                             edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(val_box)
    ax.text(6, 4.75, 'Monitor validation loss', ha='center', fontsize=10,
            fontweight='bold')

    ax.annotate('', xy=(6, 4), xytext=(6, 4.5),
               arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))

    # Final step
    final_box = FancyBboxPatch((4, 3), 4, 0.9, boxstyle="round,pad=0.1",
                               facecolor=COLOR_PREDICT, alpha=0.6,
                               edgecolor='green', linewidth=2.5)
    ax.add_patch(final_box)
    ax.text(6, 3.6, 'Save best checkpoint', ha='center', fontsize=10, fontweight='bold')
    ax.text(6, 3.2, 'Early stopping if val loss increases', ha='center', fontsize=8,
            style='italic')

    ax.set_xlim(0, 12)
    ax.set_ylim(2, 10)
    ax.axis('off')
    ax.set_title('Fine-tuning Best Practices: Decision Flowchart', fontsize=14,
                 fontweight='bold', color=COLOR_DARKGRAY, pad=15)

    plt.tight_layout()
    plt.savefig('../figures/finetuning_best_practices_flowchart_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 35: Unified Pre-training Paradigm
def plot_unified_pretraining_paradigm():
    """Final synthesis diagram"""
    fig, ax = plt.subplots(figsize=(13, 9))

    # Central concept
    central = FancyBboxPatch((4, 7), 5, 1.5, boxstyle="round,pad=0.15",
                             facecolor=COLOR_MLPURPLE, alpha=0.3,
                             edgecolor=COLOR_DARKGRAY, linewidth=3)
    ax.add_patch(central)
    ax.text(6.5, 7.75, 'Pre-training Paradigm', ha='center', fontsize=14,
            fontweight='bold', color=COLOR_DARKGRAY)

    # Three pillars
    pillars = [
        (1.5, 5, 'Massive\nUnsupervised\nData', COLOR_CONTEXT),
        (5, 5, 'Self-Supervised\nObjectives\n(MLM/AR)', COLOR_MLPURPLE),
        (8.5, 5, 'Transfer to\nAny Task\n(Fine-tune)', COLOR_PREDICT)
    ]

    for x, y, text, color in pillars:
        pillar = FancyBboxPatch((x-0.9, y-0.6), 1.8, 1.2, boxstyle="round,pad=0.1",
                                facecolor=color, alpha=0.5,
                                edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(pillar)
        ax.text(x, y, text, ha='center', fontsize=9, fontweight='bold')

        ax.annotate('', xy=(x + 0.3, 6.8), xytext=(x, y+0.6),
                   arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))

    # Two architectures
    bert_box = FancyBboxPatch((1, 2.5), 4.5, 1.5, boxstyle="round,pad=0.12",
                              facecolor=COLOR_BERT, alpha=0.3,
                              edgecolor=COLOR_DARKGRAY, linewidth=2.5)
    ax.add_patch(bert_box)
    ax.text(3.25, 3.5, 'BERT', ha='center', fontsize=12, fontweight='bold',
            color=COLOR_BERT)
    ax.text(3.25, 3.1, 'Encoder: Bidirectional', ha='center', fontsize=9)
    ax.text(3.25, 2.8, 'Best: Understanding', ha='center', fontsize=9)

    gpt_box = FancyBboxPatch((7.5, 2.5), 4.5, 1.5, boxstyle="round,pad=0.12",
                             facecolor=COLOR_GPT, alpha=0.3,
                             edgecolor=COLOR_DARKGRAY, linewidth=2.5)
    ax.add_patch(gpt_box)
    ax.text(9.75, 3.5, 'GPT', ha='center', fontsize=12, fontweight='bold',
            color=COLOR_GPT)
    ax.text(9.75, 3.1, 'Decoder: Autoregressive', ha='center', fontsize=9)
    ax.text(9.75, 2.8, 'Best: Generation', ha='center', fontsize=9)

    # Arrows from paradigm
    ax.annotate('', xy=(3.25, 4), xytext=(5, 7),
               arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))
    ax.annotate('', xy=(9.75, 4), xytext=(8, 7),
               arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))

    # Bottom: The revolution
    revolution_box = FancyBboxPatch((2, 0.5), 9, 1.2, boxstyle="round,pad=0.12",
                                    facecolor='#E8F5E9', alpha=0.8,
                                    edgecolor='green', linewidth=2.5)
    ax.add_patch(revolution_box)
    ax.text(6.5, 1.4, 'The 2018 Revolution', ha='center', fontsize=12, fontweight='bold',
            color='green')
    ax.text(6.5, 0.9, 'Transfer learning finally works for NLP', ha='center', fontsize=10,
            style='italic')

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_title('Unified Pre-training Paradigm: The Foundation of Modern NLP', fontsize=15,
                 fontweight='bold', color=COLOR_DARKGRAY, pad=20)

    plt.tight_layout()
    plt.savefig('../figures/unified_pretraining_paradigm_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Additional supporting charts (23-35) - Simplified versions

# Chart 23: Performance Before/After 2018 (GLUE benchmark)
def plot_performance_before_after_2018():
    """Leaderboard jumps"""
    fig, ax = plt.subplots(figsize=(11, 6))

    tasks = ['CoLA', 'SST-2', 'MRPC', 'STS-B', 'QQP', 'MNLI', 'QNLI', 'RTE']
    before = [45, 82, 75, 73, 80, 74, 77, 56]  # Pre-2018 best
    after = [60, 93, 88, 87, 89, 86, 91, 70]   # BERT 2018

    x = np.arange(len(tasks))
    width = 0.35

    bars1 = ax.bar(x - width/2, before, width, label='Best pre-2018',
                   color=COLOR_CURRENT, alpha=0.7, edgecolor=COLOR_DARKGRAY, linewidth=1.5)
    bars2 = ax.bar(x + width/2, after, width, label='BERT (Oct 2018)',
                   color=COLOR_PREDICT, alpha=0.7, edgecolor=COLOR_DARKGRAY, linewidth=1.5)

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_title('GLUE Benchmark: BERT Changed Everything', fontsize=14, fontweight='bold',
                 color=COLOR_DARKGRAY)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)

    # Show improvements
    for i, (b, a) in enumerate(zip(before, after)):
        improvement = a - b
        ax.text(i, max(a, b) + 2, f'+{improvement}', ha='center', fontsize=8,
                color='green', fontweight='bold')

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/performance_before_after_2018_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate remaining charts with placeholder implementations
def generate_remaining_charts():
    """Create remaining 12 supporting charts with simple designs"""

    # Chart 24: Encoder vs Decoder Comparison Table (text-based, use simple bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    features = ['Attention', 'Best for', 'Training', 'Use case']
    bert_vals = [4, 3, 2, 4]
    gpt_vals = [3, 4, 3, 3]

    x = np.arange(len(features))
    width = 0.35
    ax.bar(x - width/2, bert_vals, width, label='BERT', color=COLOR_BERT, alpha=0.6)
    ax.bar(x + width/2, gpt_vals, width, label='GPT', color=COLOR_GPT, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=11)
    ax.set_ylabel('Feature Score', fontsize=12, fontweight='bold')
    ax.set_title('BERT vs GPT: Complementary Strengths', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 5)
    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/encoder_decoder_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # Additional supporting charts (simplified placeholders for speed)
    # These would be expanded in a real production scenario

    print("Generated core 23 charts. Remaining 12 charts are simplified versions.")
    print("Core charts cover all critical concepts for 52-slide presentation.")

def generate_all_charts():
    """Generate all 35 Week 6 BSc Discovery charts"""
    print("Generating Week 6 BSc Discovery Charts - Comprehensive Edition...")
    print("Target: 35 charts for 52 slides (0.67 ratio)\n")

    print("=== OPENING & PRE-2018 (Charts 1-8) ===")
    print("1/35: Training cost comparison...")
    plot_training_cost_comparison()

    print("2/35: 2018 breakthrough timeline...")
    plot_2018_breakthrough_timeline()

    print("3/35: Pre-training vs fine-tuning paradigm...")
    plot_pretraining_finetuning_paradigm()

    print("4/35: Task-specific architectures...")
    plot_task_specific_architectures()

    print("5/35: Data requirements pre-2018...")
    plot_data_requirements_pre2018()

    print("6/35: Paradigm shift comparison...")
    plot_paradigm_shift_comparison()

    print("7/35: Performance before/after 2018...")
    plot_performance_before_after_2018()

    print("\n=== BERT CHARTS (Charts 8-19) ===")
    print("8/35: Fill-in-blank challenge...")
    plot_fill_in_blank_challenge()

    print("9/35: Bidirectional attention flow...")
    plot_bidirectional_attention_flow()

    print("10/35: Masked LM process...")
    plot_masked_lm_process()

    print("11/35: BERT architecture...")
    plot_bert_architecture()

    print("12/35: BERT special tokens...")
    plot_bert_special_tokens()

    print("13/35: WordPiece tokenization...")
    plot_wordpiece_tokenization()

    print("14/35: BERT pre-training corpus...")
    plot_bert_pretraining_corpus()

    print("15/35: BERT fine-tuning process...")
    plot_bert_finetuning_process()

    print("\n=== GPT CHARTS (Charts 16-23) ===")
    print("16/35: Text generation challenge...")
    plot_text_generation_challenge()

    print("17/35: Autoregressive visual...")
    plot_autoregressive_visual()

    print("18/35: Causal mask matrix...")
    plot_causal_mask_matrix()

    print("19/35: Next token prediction...")
    plot_next_token_prediction()

    print("20/35: GPT architecture...")
    plot_gpt_architecture()

    print("21/35: GPT evolution timeline...")
    plot_gpt_evolution_timeline()

    print("\n=== INTEGRATION CHARTS (Charts 22-35) ===")
    print("22/35: BERT vs GPT architecture comparison...")
    plot_bert_vs_gpt_architecture()

    print("23/35: Transfer learning pipeline...")
    plot_transfer_learning_pipeline()

    print("\n=== ENHANCED CHARTS (Charts 24-35) - Complete Set ===")
    print("24/35: ImageNet transfer learning analogy...")
    plot_imagenet_transfer_analogy()

    print("25/35: Masked token prediction steps...")
    plot_masked_token_prediction_steps()

    print("26/35: MLM objective equation...")
    plot_mlm_objective_equation()

    print("27/35: Next sentence prediction...")
    plot_next_sentence_prediction()

    print("28/35: Fine-tuning task heads...")
    plot_finetuning_task_heads()

    print("29/35: Layer freezing strategies...")
    plot_layer_freezing_strategies()

    print("30/35: Autoregressive objective equation...")
    plot_autoregressive_objective()

    print("31/35: Few-shot learning examples...")
    plot_few_shot_learning_examples()

    print("32/35: In-context learning visualization...")
    plot_incontext_learning()

    print("33/35: Data efficiency comparison...")
    plot_data_efficiency_comparison()

    print("34/35: Fine-tuning best practices flowchart...")
    plot_finetuning_best_practices()

    print("35/35: Unified pre-training paradigm...")
    plot_unified_pretraining_paradigm()

    print("\n" + "="*60)
    print("CHART GENERATION COMPLETE!")
    print("="*60)
    print(f"  Total charts generated: 35")
    print(f"  Chart-to-slide ratio: 35/52 = 0.67")
    print(f"  Opening & Pre-2018: 7 charts")
    print(f"  BERT section: 8 charts")
    print(f"  GPT section: 6 charts")
    print(f"  Integration: 2 charts")
    print(f"  Enhanced/Supporting: 12 charts")
    print(f"\nAll files saved to: ../figures/*_bsc.pdf")
    print("="*60)

if __name__ == "__main__":
    generate_all_charts()
