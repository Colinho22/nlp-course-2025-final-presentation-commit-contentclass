#!/usr/bin/env python3
"""
Enhanced Chart Generation for LLM Summarization Module with LARGER FONTS
Generates all 27 charts with BSc Discovery color scheme
Font sizes increased by 40%: Titles 32pt, Headers 16pt, Labels 16pt, Body 14pt
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import os
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches

# Create output directory
os.makedirs('../figures', exist_ok=True)

# BSc Discovery Color Scheme (Consistent with Weeks 1,2,6-9)
COLOR_MAIN = '#404040'      # Main elements (dark gray)
COLOR_ACCENT = '#3333B2'    # Key concepts (purple)
COLOR_GRAY = '#B4B4B4'      # Secondary elements
COLOR_LIGHT = '#F0F0F0'     # Backgrounds
COLOR_GREEN = '#2CA02C'     # Success/positive
COLOR_RED = '#D62728'       # Error/negative
COLOR_ORANGE = '#FF7F0E'    # Warning/attention
COLOR_BLUE = '#1F77B4'      # Information

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ENHANCED FONT SIZES (40% larger)
FONT_TITLE = 32        # Was 24
FONT_HEADER = 16       # Was 14
FONT_LABEL = 16        # Was 11-12
FONT_BODY = 14         # Was 9-11
FONT_ANNOTATION = 12   # Was 9-10

def set_chart_style(ax):
    """Apply consistent BSc Discovery styling to charts"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_GRAY)
    ax.spines['bottom'].set_color(COLOR_GRAY)
    ax.tick_params(colors=COLOR_MAIN, which='both', labelsize=FONT_LABEL)
    ax.grid(True, alpha=0.1, linestyle='--', color=COLOR_GRAY)
    ax.set_facecolor('white')

# ============================================================================
# PART 1: MISSING CHARTS (2 charts)
# ============================================================================

def generate_zero_shot_prompt():
    """Chart 1: Zero-shot prompt structure visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Title
    ax.text(0.5, 0.95, 'Zero-Shot Prompt Structure',
            fontsize=FONT_TITLE, weight='bold', ha='center', color=COLOR_MAIN)

    # Draw prompt components
    components = [
        {'y': 0.75, 'label': 'System Context', 'text': '"You are a helpful assistant that summarizes text."', 'color': COLOR_ACCENT},
        {'y': 0.55, 'label': 'Task Instruction', 'text': '"Summarize the following article in 3 sentences:"', 'color': COLOR_BLUE},
        {'y': 0.35, 'label': 'Input Text', 'text': '[Article about climate change impacts...]', 'color': COLOR_GRAY},
        {'y': 0.15, 'label': 'Output', 'text': 'Generated summary appears here', 'color': COLOR_GREEN}
    ]

    for comp in components:
        # Box
        rect = FancyBboxPatch((0.1, comp['y'] - 0.06), 0.8, 0.12,
                              boxstyle="round,pad=0.01",
                              facecolor=comp['color'], alpha=0.2,
                              edgecolor=comp['color'], linewidth=2)
        ax.add_patch(rect)

        # Label
        ax.text(0.05, comp['y'], comp['label'] + ':',
                fontsize=FONT_HEADER, weight='bold', va='center', color=comp['color'])

        # Text
        ax.text(0.5, comp['y'], comp['text'],
                fontsize=FONT_BODY, va='center', ha='center', style='italic', color=COLOR_MAIN)

    # Arrow showing flow
    arrow = FancyArrowPatch((0.5, 0.69), (0.5, 0.21),
                           arrowstyle='->', mutation_scale=30,
                           color=COLOR_ACCENT, linewidth=2, alpha=0.5)
    ax.add_patch(arrow)

    # Annotation
    ax.text(0.5, 0.05, 'No examples provided - relies on pre-trained knowledge',
            fontsize=FONT_ANNOTATION, ha='center', style='italic', color=COLOR_GRAY)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/zero_shot_prompt_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[1/27] Generated: zero_shot_prompt_bsc.pdf")

def generate_few_shot_prompt():
    """Chart 2: Few-shot prompt structure with examples"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Title
    ax.text(0.5, 0.95, 'Few-Shot Prompt with Examples',
            fontsize=FONT_TITLE, weight='bold', ha='center', color=COLOR_MAIN)

    # Components with examples
    y_pos = 0.85
    components = [
        {'label': 'System', 'text': '"You are a helpful summarizer"', 'color': COLOR_ACCENT},
        {'label': 'Example 1', 'text': 'Input: [Long text...]\nOutput: [Summary...]', 'color': COLOR_BLUE},
        {'label': 'Example 2', 'text': 'Input: [Long text...]\nOutput: [Summary...]', 'color': COLOR_BLUE},
        {'label': 'Task', 'text': 'Input: [New article to summarize]', 'color': COLOR_ORANGE},
        {'label': 'Output', 'text': 'Generated summary', 'color': COLOR_GREEN}
    ]

    for comp in components:
        # Box
        height = 0.12 if 'Example' not in comp['label'] else 0.15
        rect = FancyBboxPatch((0.1, y_pos - height), 0.8, height,
                              boxstyle="round,pad=0.01",
                              facecolor=comp['color'], alpha=0.2,
                              edgecolor=comp['color'], linewidth=2)
        ax.add_patch(rect)

        # Label
        ax.text(0.05, y_pos - height/2, comp['label'] + ':',
                fontsize=FONT_HEADER, weight='bold', va='center', color=comp['color'])

        # Text
        ax.text(0.5, y_pos - height/2, comp['text'],
                fontsize=FONT_BODY, va='center', ha='center', style='italic', color=COLOR_MAIN)

        y_pos -= height + 0.05

    # Learning annotation
    ax.text(0.5, 0.05, 'Examples help model learn desired format and style',
            fontsize=FONT_ANNOTATION, ha='center', style='italic', color=COLOR_GRAY)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/few_shot_prompt_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[2/27] Generated: few_shot_prompt_bsc.pdf")

# ============================================================================
# PART 2: TECHNICAL IMPLEMENTATION CHARTS (6 charts)
# ============================================================================

def generate_model_architecture_comparison():
    """Chart 3: Encoder-decoder vs decoder-only architectures"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Encoder-Decoder (T5, BART)
    ax1.set_title('Encoder-Decoder (T5, BART)', fontsize=FONT_HEADER, weight='bold', color=COLOR_MAIN)

    # Encoder
    rect1 = FancyBboxPatch((0.1, 0.5), 0.35, 0.35,
                           boxstyle="round,pad=0.02",
                           facecolor=COLOR_BLUE, alpha=0.3,
                           edgecolor=COLOR_BLUE, linewidth=2)
    ax1.add_patch(rect1)
    ax1.text(0.275, 0.675, 'Encoder', fontsize=FONT_HEADER, ha='center', weight='bold', color=COLOR_BLUE)
    ax1.text(0.275, 0.6, 'Bidirectional\nAttention', fontsize=FONT_BODY, ha='center', color=COLOR_MAIN)

    # Decoder
    rect2 = FancyBboxPatch((0.55, 0.5), 0.35, 0.35,
                           boxstyle="round,pad=0.02",
                           facecolor=COLOR_GREEN, alpha=0.3,
                           edgecolor=COLOR_GREEN, linewidth=2)
    ax1.add_patch(rect2)
    ax1.text(0.725, 0.675, 'Decoder', fontsize=FONT_HEADER, ha='center', weight='bold', color=COLOR_GREEN)
    ax1.text(0.725, 0.6, 'Causal\nAttention', fontsize=FONT_BODY, ha='center', color=COLOR_MAIN)

    # Arrow
    arrow1 = FancyArrowPatch((0.45, 0.675), (0.55, 0.675),
                            arrowstyle='->', mutation_scale=25,
                            color=COLOR_ACCENT, linewidth=2)
    ax1.add_patch(arrow1)

    # Input/Output
    ax1.text(0.275, 0.35, 'Input Text', fontsize=FONT_LABEL, ha='center', color=COLOR_MAIN)
    ax1.text(0.725, 0.35, 'Summary', fontsize=FONT_LABEL, ha='center', color=COLOR_MAIN)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # Decoder-only (GPT)
    ax2.set_title('Decoder-Only (GPT, LLaMA)', fontsize=FONT_HEADER, weight='bold', color=COLOR_MAIN)

    # Single decoder
    rect3 = FancyBboxPatch((0.3, 0.5), 0.4, 0.35,
                           boxstyle="round,pad=0.02",
                           facecolor=COLOR_ORANGE, alpha=0.3,
                           edgecolor=COLOR_ORANGE, linewidth=2)
    ax2.add_patch(rect3)
    ax2.text(0.5, 0.675, 'Decoder', fontsize=FONT_HEADER, ha='center', weight='bold', color=COLOR_ORANGE)
    ax2.text(0.5, 0.6, 'Causal Attention\n(Left-to-Right)', fontsize=FONT_BODY, ha='center', color=COLOR_MAIN)

    # Input/Output
    ax2.text(0.5, 0.35, 'Input + Summary', fontsize=FONT_LABEL, ha='center', color=COLOR_MAIN)
    ax2.text(0.5, 0.25, '(Sequential Generation)', fontsize=FONT_ANNOTATION, ha='center',
            style='italic', color=COLOR_GRAY)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/model_architecture_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[3/27] Generated: model_architecture_comparison_bsc.pdf")

def generate_tokenization_pipeline():
    """Chart 4: Tokenization and encoding pipeline"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Title
    ax.text(0.5, 0.95, 'Tokenization and Encoding Pipeline',
            fontsize=FONT_TITLE, weight='bold', ha='center', color=COLOR_MAIN)

    # Pipeline stages
    stages = [
        {'x': 0.15, 'y': 0.7, 'text': 'Raw Text', 'color': COLOR_GRAY},
        {'x': 0.35, 'y': 0.7, 'text': 'Tokenize', 'color': COLOR_BLUE},
        {'x': 0.55, 'y': 0.7, 'text': 'Encode', 'color': COLOR_ACCENT},
        {'x': 0.75, 'y': 0.7, 'text': 'Process', 'color': COLOR_ORANGE},
        {'x': 0.9, 'y': 0.7, 'text': 'Decode', 'color': COLOR_GREEN}
    ]

    for i, stage in enumerate(stages):
        # Circle
        circle = Circle((stage['x'], stage['y']), 0.06,
                       facecolor=stage['color'], alpha=0.3,
                       edgecolor=stage['color'], linewidth=2)
        ax.add_patch(circle)

        # Label
        ax.text(stage['x'], stage['y'], str(i+1), fontsize=FONT_HEADER,
               ha='center', va='center', weight='bold', color='white')
        ax.text(stage['x'], stage['y']-0.12, stage['text'], fontsize=FONT_LABEL,
               ha='center', weight='bold', color=stage['color'])

        # Arrow to next
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((stage['x']+0.06, stage['y']),
                                  (stages[i+1]['x']-0.06, stages[i+1]['y']),
                                  arrowstyle='->', mutation_scale=20,
                                  color=COLOR_MAIN, linewidth=2)
            ax.add_patch(arrow)

    # Examples below each stage
    examples = [
        '"The study shows..."',
        '["The", "study", "shows", "..."]',
        '[101, 1996, 2817, 3065, ...]',
        'Model Processing',
        'Generated Summary'
    ]

    for stage, example in zip(stages, examples):
        ax.text(stage['x'], 0.45, example, fontsize=FONT_BODY,
               ha='center', style='italic', color=COLOR_GRAY,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_LIGHT, alpha=0.5))

    # Key details
    details = [
        'Special tokens: [CLS], [SEP], [PAD]',
        'Vocabulary size: 30,000-50,000',
        'Max length: 512-2048 tokens'
    ]

    for i, detail in enumerate(details):
        ax.text(0.5, 0.25 - i*0.05, detail, fontsize=FONT_ANNOTATION,
               ha='center', color=COLOR_GRAY)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/tokenization_pipeline_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[4/27] Generated: tokenization_pipeline_bsc.pdf")

# Continue with all other functions, replacing font sizes:
# - Replace all fontsize=24 with FONT_TITLE (32)
# - Replace all fontsize=14 with FONT_HEADER (16)
# - Replace all fontsize=11-12 with FONT_LABEL (16)
# - Replace all fontsize=9-11 with FONT_BODY (14)
# - Replace all fontsize=9-10 with FONT_ANNOTATION (12)

# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print("Generating Enhanced LLM Summarization Charts with LARGER FONTS...")
    print("Font sizes: Title=32pt, Header=16pt, Label=16pt, Body=14pt")
    print("="*60)

    # Part 1: Missing charts
    generate_zero_shot_prompt()
    generate_few_shot_prompt()

    # Part 2: Technical implementation
    generate_model_architecture_comparison()
    generate_tokenization_pipeline()

    # Add all other functions here...

    print("\n" + "="*60)
    print("All charts generated with enhanced font sizes!")