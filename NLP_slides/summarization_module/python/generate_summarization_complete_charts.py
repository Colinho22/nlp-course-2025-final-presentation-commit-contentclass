#!/usr/bin/env python3
"""
Complete Chart Generation for LLM Summarization Module
Generates all 27 charts with BSc Discovery color scheme
Includes: 2 missing + 6 technical + 6 decision trees + 7 comparisons + 6 mathematical
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

def set_chart_style(ax):
    """Apply consistent BSc Discovery styling to charts"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_GRAY)
    ax.spines['bottom'].set_color(COLOR_GRAY)
    ax.tick_params(colors=COLOR_MAIN, which='both')
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
            fontsize=24, weight='bold', ha='center', color=COLOR_MAIN)

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
                fontsize=14, weight='bold', va='center', color=comp['color'])

        # Text
        ax.text(0.5, comp['y'], comp['text'],
                fontsize=12, va='center', ha='center', style='italic', color=COLOR_MAIN)

    # Arrow showing flow
    arrow = FancyArrowPatch((0.5, 0.69), (0.5, 0.21),
                           arrowstyle='->', mutation_scale=30,
                           color=COLOR_ACCENT, linewidth=2, alpha=0.5)
    ax.add_patch(arrow)

    # Annotation
    ax.text(0.5, 0.05, 'No examples provided - relies on pre-trained knowledge',
            fontsize=11, ha='center', style='italic', color=COLOR_GRAY)

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
    ax.text(0.5, 0.95, 'Few-Shot Prompt Structure',
            fontsize=24, weight='bold', ha='center', color=COLOR_MAIN)

    # Components with examples
    y_pos = 0.85

    # System context
    rect = FancyBboxPatch((0.05, y_pos - 0.04), 0.9, 0.08,
                          boxstyle="round,pad=0.01",
                          facecolor=COLOR_ACCENT, alpha=0.2,
                          edgecolor=COLOR_ACCENT, linewidth=2)
    ax.add_patch(rect)
    ax.text(0.5, y_pos, 'System: "You are an expert summarizer"',
            fontsize=12, ha='center', color=COLOR_MAIN)
    y_pos -= 0.12

    # Example 1
    ax.text(0.05, y_pos, 'Example 1:', fontsize=14, weight='bold', color=COLOR_ORANGE)
    y_pos -= 0.05
    rect = FancyBboxPatch((0.1, y_pos - 0.03), 0.35, 0.06,
                          facecolor=COLOR_LIGHT, edgecolor=COLOR_GRAY)
    ax.add_patch(rect)
    ax.text(0.275, y_pos, 'Input: "Long tech article..."', fontsize=10, ha='center')

    rect = FancyBboxPatch((0.55, y_pos - 0.03), 0.35, 0.06,
                          facecolor=COLOR_GREEN, alpha=0.2, edgecolor=COLOR_GREEN)
    ax.add_patch(rect)
    ax.text(0.725, y_pos, 'Output: "Tech summary..."', fontsize=10, ha='center')

    # Arrow
    arrow = FancyArrowPatch((0.46, y_pos), (0.54, y_pos),
                           arrowstyle='->', color=COLOR_ACCENT, linewidth=2)
    ax.add_patch(arrow)
    y_pos -= 0.1

    # Example 2
    ax.text(0.05, y_pos, 'Example 2:', fontsize=14, weight='bold', color=COLOR_ORANGE)
    y_pos -= 0.05
    rect = FancyBboxPatch((0.1, y_pos - 0.03), 0.35, 0.06,
                          facecolor=COLOR_LIGHT, edgecolor=COLOR_GRAY)
    ax.add_patch(rect)
    ax.text(0.275, y_pos, 'Input: "Medical research..."', fontsize=10, ha='center')

    rect = FancyBboxPatch((0.55, y_pos - 0.03), 0.35, 0.06,
                          facecolor=COLOR_GREEN, alpha=0.2, edgecolor=COLOR_GREEN)
    ax.add_patch(rect)
    ax.text(0.725, y_pos, 'Output: "Medical summary..."', fontsize=10, ha='center')

    # Arrow
    arrow = FancyArrowPatch((0.46, y_pos), (0.54, y_pos),
                           arrowstyle='->', color=COLOR_ACCENT, linewidth=2)
    ax.add_patch(arrow)
    y_pos -= 0.1

    # Example 3
    ax.text(0.05, y_pos, 'Example 3:', fontsize=14, weight='bold', color=COLOR_ORANGE)
    y_pos -= 0.05
    rect = FancyBboxPatch((0.1, y_pos - 0.03), 0.35, 0.06,
                          facecolor=COLOR_LIGHT, edgecolor=COLOR_GRAY)
    ax.add_patch(rect)
    ax.text(0.275, y_pos, 'Input: "Financial report..."', fontsize=10, ha='center')

    rect = FancyBboxPatch((0.55, y_pos - 0.03), 0.35, 0.06,
                          facecolor=COLOR_GREEN, alpha=0.2, edgecolor=COLOR_GREEN)
    ax.add_patch(rect)
    ax.text(0.725, y_pos, 'Output: "Finance summary..."', fontsize=10, ha='center')

    # Arrow
    arrow = FancyArrowPatch((0.46, y_pos), (0.54, y_pos),
                           arrowstyle='->', color=COLOR_ACCENT, linewidth=2)
    ax.add_patch(arrow)
    y_pos -= 0.12

    # Actual task
    ax.text(0.05, y_pos, 'Your Task:', fontsize=14, weight='bold', color=COLOR_BLUE)
    y_pos -= 0.05
    rect = FancyBboxPatch((0.1, y_pos - 0.03), 0.35, 0.06,
                          facecolor=COLOR_BLUE, alpha=0.2, edgecolor=COLOR_BLUE, linewidth=2)
    ax.add_patch(rect)
    ax.text(0.275, y_pos, 'Input: "New article..."', fontsize=10, ha='center', weight='bold')

    rect = FancyBboxPatch((0.55, y_pos - 0.03), 0.35, 0.06,
                          facecolor='white', edgecolor=COLOR_BLUE, linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.text(0.725, y_pos, '? Model generates ?', fontsize=10, ha='center', style='italic', color=COLOR_GRAY)

    # Arrow
    arrow = FancyArrowPatch((0.46, y_pos), (0.54, y_pos),
                           arrowstyle='->', color=COLOR_BLUE, linewidth=2)
    ax.add_patch(arrow)

    # Bottom annotation
    ax.text(0.5, 0.05, 'Pattern learning: Model infers task from examples',
            fontsize=12, ha='center', weight='bold', color=COLOR_ACCENT)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/few_shot_prompt_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[2/27] Generated: few_shot_prompt_bsc.pdf")

# ============================================================================
# PART 2: TECHNICAL DIAGRAMS (6 charts)
# ============================================================================

def generate_system_prompt_anatomy():
    """Chart 3: System prompt anatomy breakdown"""
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.text(0.5, 0.95, 'System Prompt Anatomy',
            fontsize=24, weight='bold', ha='center', color=COLOR_MAIN)

    # Main prompt box
    main_box = FancyBboxPatch((0.15, 0.35), 0.7, 0.5,
                              boxstyle="round,pad=0.02",
                              facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT, linewidth=2)
    ax.add_patch(main_box)

    # Components with arrows
    components = [
        {'x': 0.25, 'y': 0.7, 'text': 'Role Definition\n"You are an expert\nsummarizer"', 'color': COLOR_BLUE},
        {'x': 0.5, 'y': 0.7, 'text': 'Task Constraints\n"Maximum 3 sentences\nFocus on key points"', 'color': COLOR_ORANGE},
        {'x': 0.75, 'y': 0.7, 'text': 'Output Format\n"Use bullet points\nBe concise"', 'color': COLOR_GREEN},
        {'x': 0.25, 'y': 0.5, 'text': 'Style Guide\n"Professional tone\nNo opinions"', 'color': COLOR_RED},
        {'x': 0.5, 'y': 0.5, 'text': 'Domain Context\n"Medical terminology\nTechnical accuracy"', 'color': COLOR_ACCENT},
        {'x': 0.75, 'y': 0.5, 'text': 'Quality Criteria\n"Factual only\nNo hallucinations"', 'color': COLOR_GRAY}
    ]

    for comp in components:
        # Small box
        box = FancyBboxPatch((comp['x'] - 0.08, comp['y'] - 0.05), 0.16, 0.1,
                             boxstyle="round,pad=0.01",
                             facecolor=comp['color'], alpha=0.3,
                             edgecolor=comp['color'], linewidth=1.5)
        ax.add_patch(box)

        # Text
        ax.text(comp['x'], comp['y'], comp['text'],
                fontsize=10, ha='center', va='center', color=COLOR_MAIN)

    # Central combination
    ax.text(0.5, 0.2, 'Combined Effect: Precise, Consistent, High-Quality Summaries',
            fontsize=14, ha='center', weight='bold', color=COLOR_ACCENT)

    # Examples on sides
    ax.text(0.05, 0.15, 'Bad Prompt:\n"Summarize this"',
            fontsize=10, ha='center', color=COLOR_RED, style='italic')

    ax.text(0.95, 0.15, 'Good Prompt:\n"As a medical expert,\nsummarize in 3 bullets,\nfocus on findings"',
            fontsize=10, ha='center', color=COLOR_GREEN, style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/system_prompt_anatomy_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[3/27] Generated: system_prompt_anatomy_bsc.pdf")

def generate_token_flow_pipeline():
    """Chart 4: Token flow from text to model output"""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.text(0.5, 0.9, 'Token Flow Pipeline',
            fontsize=24, weight='bold', ha='center', color=COLOR_MAIN)

    # Pipeline stages
    stages = [
        {'x': 0.1, 'label': 'Input Text', 'example': '"The cat sat"', 'color': COLOR_BLUE},
        {'x': 0.3, 'label': 'Tokenization', 'example': '["The", "cat", "sat"]', 'color': COLOR_ORANGE},
        {'x': 0.5, 'label': 'Token IDs', 'example': '[546, 1828, 2062]', 'color': COLOR_ACCENT},
        {'x': 0.7, 'label': 'Embeddings', 'example': '[[0.2, -0.1, ...],\n [0.5, 0.3, ...],\n [0.1, 0.7, ...]]', 'color': COLOR_GREEN},
        {'x': 0.9, 'label': 'Model Output', 'example': 'Summary tokens', 'color': COLOR_RED}
    ]

    for i, stage in enumerate(stages):
        # Box
        rect = FancyBboxPatch((stage['x'] - 0.07, 0.35), 0.14, 0.3,
                              boxstyle="round,pad=0.01",
                              facecolor=stage['color'], alpha=0.2,
                              edgecolor=stage['color'], linewidth=2)
        ax.add_patch(rect)

        # Label
        ax.text(stage['x'], 0.65, stage['label'],
                fontsize=12, ha='center', weight='bold', color=stage['color'])

        # Example
        ax.text(stage['x'], 0.45, stage['example'],
                fontsize=9, ha='center', style='italic', color=COLOR_MAIN,
                multialignment='center')

        # Arrow to next stage
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((stage['x'] + 0.07, 0.5),
                                   (stages[i+1]['x'] - 0.07, 0.5),
                                   arrowstyle='->', mutation_scale=20,
                                   color=COLOR_ACCENT, linewidth=2)
            ax.add_patch(arrow)

    # Annotations
    ax.text(0.2, 0.25, 'Subword\nsplitting', fontsize=9, ha='center', color=COLOR_GRAY)
    ax.text(0.4, 0.25, 'Vocabulary\nlookup', fontsize=9, ha='center', color=COLOR_GRAY)
    ax.text(0.6, 0.25, 'Vector\nrepresentation', fontsize=9, ha='center', color=COLOR_GRAY)
    ax.text(0.8, 0.25, 'Transformer\nprocessing', fontsize=9, ha='center', color=COLOR_GRAY)

    # Bottom note
    ax.text(0.5, 0.1, 'Each token limited to model vocabulary (typically 30K-50K tokens)',
            fontsize=11, ha='center', style='italic', color=COLOR_GRAY)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/token_flow_pipeline_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[4/27] Generated: token_flow_pipeline_bsc.pdf")

def generate_attention_sink_visual():
    """Chart 5: Attention sink visualization showing position bias"""
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.text(0.5, 0.95, 'Attention Sink Pattern',
            fontsize=24, weight='bold', ha='center', color=COLOR_MAIN)

    # Create attention heatmap
    positions = 20
    attention_matrix = np.zeros((positions, positions))

    # Create attention sink pattern (high attention to first few tokens)
    for i in range(positions):
        # Sink to first tokens
        attention_matrix[i, 0] = 0.3 + np.random.uniform(-0.05, 0.05)
        attention_matrix[i, 1] = 0.25 + np.random.uniform(-0.05, 0.05)
        attention_matrix[i, 2] = 0.2 + np.random.uniform(-0.05, 0.05)

        # Local context
        for j in range(max(0, i-2), min(positions, i+3)):
            if j not in [0, 1, 2]:
                attention_matrix[i, j] = 0.1 + np.random.uniform(-0.02, 0.02)

        # Normalize rows
        attention_matrix[i] = attention_matrix[i] / attention_matrix[i].sum()

    # Plot heatmap
    im = ax.imshow(attention_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.4)

    # Labels
    ax.set_xlabel('Token Position (Keys)', fontsize=12, weight='bold')
    ax.set_ylabel('Token Position (Queries)', fontsize=12, weight='bold')

    # Highlight sink positions
    for i in range(3):
        rect = patches.Rectangle((i-0.5, -0.5), 1, positions,
                                linewidth=2, edgecolor=COLOR_ACCENT,
                                facecolor='none', linestyle='--')
        ax.add_patch(rect)

    # Annotations
    ax.text(1, -1.5, 'Attention Sink\n(First tokens)',
            fontsize=11, ha='center', color=COLOR_ACCENT, weight='bold')

    ax.text(10, -1.5, 'Normal Attention\n(Local context)',
            fontsize=11, ha='center', color=COLOR_GRAY)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15, fontsize=11)

    # Side text
    ax.text(22, 10, 'Problem:\nFirst tokens accumulate\nattention regardless\nof relevance',
            fontsize=10, va='center', color=COLOR_RED)

    ax.text(22, 15, 'Solution:\nAccount for sink\nwhen extracting\nimportant content',
            fontsize=10, va='center', color=COLOR_GREEN)

    ax.set_xlim(-0.5, positions-0.5)
    ax.set_ylim(positions-0.5, -2)

    plt.tight_layout()
    plt.savefig('../figures/attention_sink_visual_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[5/27] Generated: attention_sink_visual_bsc.pdf")

def generate_flan_t5_architecture():
    """Chart 6: FLAN-T5 encoder-decoder architecture"""
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.text(0.5, 0.95, 'FLAN-T5 Architecture',
            fontsize=24, weight='bold', ha='center', color=COLOR_MAIN)

    # Encoder block
    encoder_box = FancyBboxPatch((0.1, 0.35), 0.35, 0.5,
                                boxstyle="round,pad=0.02",
                                facecolor=COLOR_BLUE, alpha=0.2,
                                edgecolor=COLOR_BLUE, linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(0.275, 0.8, 'ENCODER', fontsize=14, ha='center', weight='bold', color=COLOR_BLUE)

    # Encoder layers
    for i, layer in enumerate(['Self-Attention', 'Feed-Forward', 'Layer Norm']):
        y = 0.65 - i * 0.1
        rect = FancyBboxPatch((0.15, y - 0.03), 0.25, 0.06,
                              facecolor='white', edgecolor=COLOR_BLUE)
        ax.add_patch(rect)
        ax.text(0.275, y, layer, fontsize=10, ha='center')

    # Decoder block
    decoder_box = FancyBboxPatch((0.55, 0.35), 0.35, 0.5,
                                boxstyle="round,pad=0.02",
                                facecolor=COLOR_GREEN, alpha=0.2,
                                edgecolor=COLOR_GREEN, linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(0.725, 0.8, 'DECODER', fontsize=14, ha='center', weight='bold', color=COLOR_GREEN)

    # Decoder layers
    for i, layer in enumerate(['Masked Self-Attn', 'Cross-Attention', 'Feed-Forward', 'Layer Norm']):
        y = 0.7 - i * 0.08
        rect = FancyBboxPatch((0.6, y - 0.03), 0.25, 0.06,
                              facecolor='white', edgecolor=COLOR_GREEN)
        ax.add_patch(rect)
        ax.text(0.725, y, layer, fontsize=10, ha='center')

    # Cross-attention arrow
    arrow = FancyArrowPatch((0.45, 0.6), (0.55, 0.62),
                           arrowstyle='<->', mutation_scale=25,
                           color=COLOR_ACCENT, linewidth=3)
    ax.add_patch(arrow)
    ax.text(0.5, 0.65, 'Cross-\nAttention', fontsize=10, ha='center',
            color=COLOR_ACCENT, weight='bold')

    # Input/Output
    ax.text(0.275, 0.25, 'Input: Article Text\n(up to 512 tokens)',
            fontsize=11, ha='center', style='italic', color=COLOR_MAIN)

    ax.text(0.725, 0.25, 'Output: Summary\n(typically 50-150 tokens)',
            fontsize=11, ha='center', style='italic', color=COLOR_MAIN)

    # Model sizes
    ax.text(0.5, 0.12, 'Model Sizes:', fontsize=12, ha='center', weight='bold')
    sizes = ['Small (80M)', 'Base (250M)', 'Large (780M)', 'XL (3B)', 'XXL (11B)']
    for i, size in enumerate(sizes):
        x = 0.2 + i * 0.15
        color = COLOR_GREEN if i == 0 else (COLOR_ORANGE if i < 3 else COLOR_RED)
        ax.text(x, 0.05, size, fontsize=10, ha='center', color=color)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/flan_t5_architecture_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[6/27] Generated: flan_t5_architecture_bsc.pdf")

def generate_chunking_overlap_diagram():
    """Chart 7: Chunking with overlap visualization"""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.text(0.5, 0.9, 'Document Chunking with Overlap',
            fontsize=24, weight='bold', ha='center', color=COLOR_MAIN)

    # Document representation
    doc_length = 0.8
    doc_start = 0.1

    # Full document
    doc_rect = FancyBboxPatch((doc_start, 0.65), doc_length, 0.1,
                              facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN, linewidth=2)
    ax.add_patch(doc_rect)
    ax.text(0.5, 0.7, 'Original Document (2000 tokens)',
            fontsize=12, ha='center', weight='bold')

    # Chunks with overlap
    chunk_width = 0.25
    overlap_width = 0.05
    chunks_y = 0.4

    colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_RED]
    chunk_positions = [
        doc_start,
        doc_start + chunk_width - overlap_width,
        doc_start + 2 * chunk_width - 2 * overlap_width,
        doc_start + 3 * chunk_width - 3 * overlap_width
    ]

    for i, (pos, color) in enumerate(zip(chunk_positions, colors)):
        # Main chunk
        chunk_rect = FancyBboxPatch((pos, chunks_y), chunk_width, 0.1,
                                   facecolor=color, alpha=0.3,
                                   edgecolor=color, linewidth=2)
        ax.add_patch(chunk_rect)

        # Label
        ax.text(pos + chunk_width/2, chunks_y + 0.05, f'Chunk {i+1}\n(500 tokens)',
                fontsize=10, ha='center', va='center')

        # Overlap region
        if i > 0:
            overlap_rect = FancyBboxPatch((pos, chunks_y), overlap_width, 0.1,
                                         facecolor=COLOR_ACCENT, alpha=0.5,
                                         edgecolor=COLOR_ACCENT, linewidth=1,
                                         linestyle='--')
            ax.add_patch(overlap_rect)

    # Annotations
    ax.text(chunk_positions[1] + overlap_width/2, chunks_y - 0.05,
            '50 token\noverlap', fontsize=9, ha='center', color=COLOR_ACCENT)

    # Arrows showing flow
    for i in range(len(chunk_positions)):
        arrow = FancyArrowPatch((doc_start + i * chunk_width/3 + 0.1, 0.65),
                               (chunk_positions[i] + chunk_width/2, chunks_y + 0.1),
                               arrowstyle='->', color=COLOR_GRAY,
                               linewidth=1.5, alpha=0.5)
        ax.add_patch(arrow)

    # Benefits box
    ax.text(0.5, 0.2, 'Benefits of Overlap:', fontsize=12, ha='center', weight='bold')
    ax.text(0.25, 0.1, '• Preserves context', fontsize=10, color=COLOR_GREEN)
    ax.text(0.5, 0.1, '• No information loss', fontsize=10, color=COLOR_GREEN)
    ax.text(0.75, 0.1, '• Better coherence', fontsize=10, color=COLOR_GREEN)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/chunking_overlap_diagram_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[7/27] Generated: chunking_overlap_diagram_bsc.pdf")

def generate_multi_doc_deduplication():
    """Chart 8: Multi-document deduplication clustering"""
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.text(0.5, 0.95, 'Multi-Document Deduplication',
            fontsize=24, weight='bold', ha='center', color=COLOR_MAIN)

    # Document clusters
    clusters = [
        {'center': (0.25, 0.65), 'docs': 3, 'label': 'Financial Reports', 'color': COLOR_BLUE},
        {'center': (0.5, 0.7), 'docs': 2, 'label': 'Duplicates', 'color': COLOR_RED},
        {'center': (0.75, 0.6), 'docs': 4, 'label': 'Market Analysis', 'color': COLOR_GREEN},
        {'center': (0.35, 0.4), 'docs': 2, 'label': 'News Articles', 'color': COLOR_ORANGE}
    ]

    # Draw clusters
    for cluster in clusters:
        cx, cy = cluster['center']

        # Cluster circle
        circle = Circle(cluster['center'], 0.12,
                       facecolor=cluster['color'], alpha=0.2,
                       edgecolor=cluster['color'], linewidth=2)
        ax.add_patch(circle)

        # Documents in cluster
        angles = np.linspace(0, 2*np.pi, cluster['docs'] + 1)[:-1]
        for angle in angles:
            dx = cx + 0.08 * np.cos(angle)
            dy = cy + 0.08 * np.sin(angle)
            doc = Circle((dx, dy), 0.02, facecolor=cluster['color'],
                        edgecolor=COLOR_MAIN, linewidth=1)
            ax.add_patch(doc)

        # Label
        ax.text(cx, cy - 0.18, cluster['label'],
                fontsize=10, ha='center', color=cluster['color'], weight='bold')

    # Similarity threshold line
    ax.plot([0.1, 0.9], [0.25, 0.25], '--', color=COLOR_ACCENT, linewidth=2)
    ax.text(0.5, 0.22, 'Similarity Threshold (0.85)',
            fontsize=11, ha='center', color=COLOR_ACCENT)

    # Process flow
    ax.text(0.2, 0.08, '1. Compute\nEmbeddings',
            fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3",
                                                facecolor=COLOR_LIGHT))
    ax.text(0.4, 0.08, '2. Calculate\nSimilarity',
            fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3",
                                                facecolor=COLOR_LIGHT))
    ax.text(0.6, 0.08, '3. Cluster\nDocuments',
            fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3",
                                                facecolor=COLOR_LIGHT))
    ax.text(0.8, 0.08, '4. Select\nRepresentatives',
            fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3",
                                                facecolor=COLOR_LIGHT))

    # Arrows
    for i in range(3):
        x1 = 0.2 + i * 0.2 + 0.05
        x2 = 0.4 + i * 0.2 - 0.05
        arrow = FancyArrowPatch((x1, 0.08), (x2, 0.08),
                               arrowstyle='->', mutation_scale=15,
                               color=COLOR_ACCENT, linewidth=1.5)
        ax.add_patch(arrow)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/multi_doc_deduplication_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[8/27] Generated: multi_doc_deduplication_bsc.pdf")

# ============================================================================
# PART 3: DECISION TREES (6 charts)
# ============================================================================

def generate_prompt_engineering_flowchart():
    """Chart 9: Decision flowchart for prompt strategy selection"""
    fig, ax = plt.subplots(figsize=(14, 10))

    ax.text(0.5, 0.95, 'Prompt Engineering Decision Tree',
            fontsize=24, weight='bold', ha='center', color=COLOR_MAIN)

    # Define decision nodes and paths
    nodes = {
        'start': {'pos': (0.5, 0.85), 'text': 'Start:\nNeed Summary', 'color': COLOR_ACCENT},
        'examples': {'pos': (0.5, 0.7), 'text': 'Have good\nexamples?', 'color': COLOR_BLUE},
        'complex': {'pos': (0.3, 0.55), 'text': 'Complex\ndocument?', 'color': COLOR_ORANGE},
        'simple': {'pos': (0.7, 0.55), 'text': 'Need specific\nformat?', 'color': COLOR_ORANGE},
        'zero': {'pos': (0.15, 0.35), 'text': 'Zero-Shot\nPrompt', 'color': COLOR_GREEN, 'terminal': True},
        'cot': {'pos': (0.35, 0.35), 'text': 'Chain-of-\nThought', 'color': COLOR_GREEN, 'terminal': True},
        'few': {'pos': (0.55, 0.35), 'text': 'Few-Shot\nPrompt', 'color': COLOR_GREEN, 'terminal': True},
        'format': {'pos': (0.75, 0.35), 'text': 'Format-\nSpecific', 'color': COLOR_GREEN, 'terminal': True},
        'standard': {'pos': (0.9, 0.35), 'text': 'Standard\nZero-Shot', 'color': COLOR_GREEN, 'terminal': True}
    }

    # Draw nodes
    for node_id, node in nodes.items():
        if node.get('terminal'):
            # Terminal node (solution)
            rect = FancyBboxPatch((node['pos'][0] - 0.06, node['pos'][1] - 0.04),
                                  0.12, 0.08,
                                  boxstyle="round,pad=0.01",
                                  facecolor=node['color'], alpha=0.3,
                                  edgecolor=node['color'], linewidth=2)
        else:
            # Decision node
            rect = FancyBboxPatch((node['pos'][0] - 0.06, node['pos'][1] - 0.04),
                                  0.12, 0.08,
                                  boxstyle="round,pad=0.01",
                                  facecolor='white',
                                  edgecolor=node['color'], linewidth=2)
        ax.add_patch(rect)

        ax.text(node['pos'][0], node['pos'][1], node['text'],
                fontsize=11, ha='center', va='center',
                weight='bold' if node.get('terminal') else 'normal',
                color=COLOR_MAIN)

    # Draw connections with labels
    connections = [
        ('start', 'examples', ''),
        ('examples', 'complex', 'YES'),
        ('examples', 'simple', 'NO'),
        ('complex', 'zero', 'NO'),
        ('complex', 'cot', 'YES'),
        ('simple', 'few', 'YES'),
        ('simple', 'format', 'Format'),
        ('simple', 'standard', 'NO')
    ]

    for start, end, label in connections:
        start_pos = nodes[start]['pos']
        end_pos = nodes[end]['pos']

        arrow = FancyArrowPatch(start_pos, end_pos,
                               arrowstyle='->', mutation_scale=15,
                               color=COLOR_GRAY, linewidth=1.5)
        ax.add_patch(arrow)

        if label:
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = (start_pos[1] + end_pos[1]) / 2
            ax.text(mid_x, mid_y, label, fontsize=9,
                   ha='center', color=COLOR_ACCENT, weight='bold')

    # Add examples at bottom
    examples_y = 0.2
    ax.text(0.15, examples_y, 'Example:\n"Summarize this\nmedical article"',
            fontsize=9, ha='center', style='italic', color=COLOR_GRAY)
    ax.text(0.35, examples_y, 'Example:\n"Extract findings,\nmethods, conclusion"',
            fontsize=9, ha='center', style='italic', color=COLOR_GRAY)
    ax.text(0.55, examples_y, 'Example:\n"Like these 3\nprevious summaries"',
            fontsize=9, ha='center', style='italic', color=COLOR_GRAY)
    ax.text(0.75, examples_y, 'Example:\n"As bullet points\nmax 5 items"',
            fontsize=9, ha='center', style='italic', color=COLOR_GRAY)

    # Legend
    ax.text(0.5, 0.05, 'Decision Points → Solutions',
            fontsize=12, ha='center', weight='bold', color=COLOR_ACCENT)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/prompt_engineering_flowchart_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[9/27] Generated: prompt_engineering_flowchart_bsc.pdf")

def generate_method_selection_guide():
    """Chart 10: Method selection based on document characteristics"""
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.text(0.5, 0.95, 'Summarization Method Selection Guide',
            fontsize=24, weight='bold', ha='center', color=COLOR_MAIN)

    # Create decision matrix
    criteria = ['Document Count', 'Quality Need', 'Budget', 'Recommended Method']
    scenarios = [
        ['Single', 'High', 'High', 'GPT-4 + Few-Shot'],
        ['Single', 'High', 'Low', 'FLAN-T5-XL Local'],
        ['Single', 'Medium', 'Any', 'GPT-3.5 + Zero-Shot'],
        ['Multiple', 'High', 'High', 'GPT-4 + Map-Reduce'],
        ['Multiple', 'Medium', 'Low', 'FLAN-T5 + Chunking'],
        ['Many (>10)', 'Any', 'Low', 'Extractive + LLM Polish']
    ]

    # Table headers
    y_start = 0.75
    for i, criterion in enumerate(criteria):
        x = 0.15 + i * 0.2
        rect = FancyBboxPatch((x - 0.08, y_start - 0.03), 0.16, 0.06,
                              facecolor=COLOR_ACCENT, alpha=0.3,
                              edgecolor=COLOR_ACCENT, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y_start, criterion, fontsize=11, ha='center',
                va='center', weight='bold', color=COLOR_MAIN)

    # Table rows
    colors_cycle = [COLOR_LIGHT, 'white']
    for row_idx, scenario in enumerate(scenarios):
        y = y_start - 0.08 * (row_idx + 1)

        for col_idx, value in enumerate(scenario):
            x = 0.15 + col_idx * 0.2

            # Background
            rect = FancyBboxPatch((x - 0.08, y - 0.03), 0.16, 0.06,
                                  facecolor=colors_cycle[row_idx % 2],
                                  edgecolor=COLOR_GRAY, linewidth=0.5)
            ax.add_patch(rect)

            # Color code the recommendation
            if col_idx == 3:  # Recommendation column
                if 'GPT-4' in value:
                    color = COLOR_RED
                elif 'GPT-3.5' in value:
                    color = COLOR_ORANGE
                else:
                    color = COLOR_GREEN
                ax.text(x, y, value, fontsize=10, ha='center',
                       va='center', weight='bold', color=color)
            else:
                ax.text(x, y, value, fontsize=10, ha='center',
                       va='center', color=COLOR_MAIN)

    # Cost indicators
    ax.text(0.85, 0.6, 'Cost Level:', fontsize=11, weight='bold')
    ax.text(0.85, 0.55, 'High Cost', fontsize=10, color=COLOR_RED)
    ax.text(0.85, 0.50, 'Medium Cost', fontsize=10, color=COLOR_ORANGE)
    ax.text(0.85, 0.45, 'Low Cost', fontsize=10, color=COLOR_GREEN)

    # Bottom notes
    ax.text(0.5, 0.15, 'Key Factors:', fontsize=12, ha='center', weight='bold')
    ax.text(0.2, 0.08, '• Token limits constrain input size', fontsize=10, color=COLOR_GRAY)
    ax.text(0.5, 0.08, '• Quality vs Cost tradeoff', fontsize=10, color=COLOR_GRAY)
    ax.text(0.75, 0.08, '• Local models for sensitive data', fontsize=10, color=COLOR_GRAY)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/method_selection_guide_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[10/27] Generated: method_selection_guide_bsc.pdf")

def generate_parameter_tuning_tree():
    """Chart 11: Parameter tuning decision tree"""
    fig, ax = plt.subplots(figsize=(14, 10))

    ax.text(0.5, 0.95, 'Parameter Tuning Decision Tree',
            fontsize=24, weight='bold', ha='center', color=COLOR_MAIN)

    # Decision flow for parameter selection
    # Level 1: Task type
    ax.text(0.5, 0.85, 'Task Type?', fontsize=14, ha='center', weight='bold', color=COLOR_ACCENT)

    # Level 2: Task categories
    tasks = [
        {'x': 0.2, 'text': 'News/Facts', 'color': COLOR_BLUE},
        {'x': 0.5, 'text': 'Technical', 'color': COLOR_ORANGE},
        {'x': 0.8, 'text': 'Creative', 'color': COLOR_GREEN}
    ]

    for task in tasks:
        rect = FancyBboxPatch((task['x'] - 0.08, 0.7), 0.16, 0.08,
                              boxstyle="round,pad=0.01",
                              facecolor=task['color'], alpha=0.2,
                              edgecolor=task['color'], linewidth=2)
        ax.add_patch(rect)
        ax.text(task['x'], 0.74, task['text'], fontsize=12,
                ha='center', va='center', weight='bold')

        # Arrow from top
        arrow = FancyArrowPatch((0.5, 0.82), (task['x'], 0.78),
                               arrowstyle='->', color=COLOR_GRAY, linewidth=1.5)
        ax.add_patch(arrow)

    # Level 3: Accuracy needs
    ax.text(0.5, 0.6, 'Accuracy Priority?', fontsize=12, ha='center', color=COLOR_MAIN)

    # Level 4: Parameter recommendations
    params = [
        # News/Facts - High accuracy
        {'x': 0.1, 'y': 0.45, 'params': 'T=0.3\nTop-p=0.9\nPenalty=1.2', 'label': 'Factual\nPrecise'},
        {'x': 0.3, 'y': 0.45, 'params': 'T=0.5\nTop-p=0.95\nPenalty=1.0', 'label': 'Balanced'},

        # Technical - Moderate
        {'x': 0.4, 'y': 0.45, 'params': 'T=0.4\nTop-p=0.9\nPenalty=1.1', 'label': 'Technical\nClear'},
        {'x': 0.6, 'y': 0.45, 'params': 'T=0.6\nTop-p=0.95\nPenalty=1.0', 'label': 'Explanatory'},

        # Creative - Low accuracy, high diversity
        {'x': 0.7, 'y': 0.45, 'params': 'T=0.8\nTop-p=0.95\nPenalty=0.9', 'label': 'Creative\nVaried'},
        {'x': 0.9, 'y': 0.45, 'params': 'T=1.0\nTop-p=1.0\nPenalty=0.8', 'label': 'Maximum\nDiversity'}
    ]

    for param in params:
        # Parameter box
        rect = FancyBboxPatch((param['x'] - 0.05, param['y'] - 0.05), 0.1, 0.1,
                              boxstyle="round,pad=0.01",
                              facecolor=COLOR_GREEN if param['x'] < 0.4 else
                                       (COLOR_ORANGE if param['x'] < 0.7 else COLOR_RED),
                              alpha=0.3,
                              edgecolor=COLOR_MAIN, linewidth=1)
        ax.add_patch(rect)

        # Parameters
        ax.text(param['x'], param['y'], param['params'], fontsize=9,
                ha='center', va='center', color=COLOR_MAIN)

        # Label below
        ax.text(param['x'], param['y'] - 0.08, param['label'], fontsize=9,
                ha='center', style='italic', color=COLOR_GRAY)

    # Examples at bottom
    examples = [
        {'x': 0.2, 'text': 'Financial report:\nT=0.3, factual'},
        {'x': 0.5, 'text': 'Research paper:\nT=0.5, balanced'},
        {'x': 0.8, 'text': 'Blog summary:\nT=0.8, engaging'}
    ]

    for ex in examples:
        ax.text(ex['x'], 0.2, ex['text'], fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_LIGHT))

    # Legend
    ax.text(0.5, 0.1, 'Lower Temperature = More Factual | Higher Temperature = More Creative',
            fontsize=11, ha='center', weight='bold', color=COLOR_ACCENT)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/parameter_tuning_tree_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[11/27] Generated: parameter_tuning_tree_bsc.pdf")

def generate_context_strategy_selector():
    """Chart 12: Context handling strategy selection"""
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.text(0.5, 0.95, 'Context Strategy Selector',
            fontsize=24, weight='bold', ha='center', color=COLOR_MAIN)

    # Document length categories
    lengths = [
        {'range': '< 500 tokens', 'x': 0.15, 'strategy': 'Direct Input', 'color': COLOR_GREEN},
        {'range': '500-2K tokens', 'x': 0.35, 'strategy': 'Truncation', 'color': COLOR_BLUE},
        {'range': '2K-10K tokens', 'x': 0.55, 'strategy': 'Chunking', 'color': COLOR_ORANGE},
        {'range': '> 10K tokens', 'x': 0.75, 'strategy': 'Hierarchical', 'color': COLOR_RED}
    ]

    # Draw length categories
    for length in lengths:
        # Category box
        rect = FancyBboxPatch((length['x'] - 0.08, 0.65), 0.16, 0.15,
                              boxstyle="round,pad=0.01",
                              facecolor=length['color'], alpha=0.2,
                              edgecolor=length['color'], linewidth=2)
        ax.add_patch(rect)

        ax.text(length['x'], 0.75, length['range'], fontsize=12,
                ha='center', weight='bold', color=COLOR_MAIN)
        ax.text(length['x'], 0.68, length['strategy'], fontsize=11,
                ha='center', style='italic', color=length['color'])

    # Strategy details
    strategies = [
        {'x': 0.15, 'y': 0.45, 'title': 'Direct Input',
         'details': '• Full document\n• Single pass\n• No processing\n• Fastest',
         'time': '~2 sec', 'cost': 'USD 0.01'},

        {'x': 0.35, 'y': 0.45, 'title': 'Smart Truncation',
         'details': '• Keep intro/conclusion\n• Sample middle\n• Preserve structure\n• Good quality',
         'time': '~3 sec', 'cost': 'USD 0.02'},

        {'x': 0.55, 'y': 0.45, 'title': 'Chunking',
         'details': '• Split with overlap\n• Process each\n• Merge summaries\n• Reliable',
         'time': '~10 sec', 'cost': 'USD 0.05-0.10'},

        {'x': 0.75, 'y': 0.45, 'title': 'Hierarchical',
         'details': '• Chunk → Summarize\n• Merge → Summarize\n• Recursive\n• Best for books',
         'time': '~30+ sec', 'cost': 'USD 0.20+'}
    ]

    for strat in strategies:
        # Details box
        rect = FancyBboxPatch((strat['x'] - 0.08, strat['y'] - 0.12), 0.16, 0.24,
                              facecolor='white', edgecolor=COLOR_GRAY, linewidth=1)
        ax.add_patch(rect)

        # Title
        ax.text(strat['x'], strat['y'] + 0.08, strat['title'],
                fontsize=10, ha='center', weight='bold')

        # Details
        ax.text(strat['x'], strat['y'] - 0.02, strat['details'],
                fontsize=8, ha='center', va='center')

        # Time and cost
        ax.text(strat['x'] - 0.05, strat['y'] - 0.14, strat['time'],
                fontsize=9, color=COLOR_BLUE)
        ax.text(strat['x'] + 0.05, strat['y'] - 0.14, strat['cost'],
                fontsize=9, color=COLOR_GREEN if '0.01' in strat['cost'] or '0.02' in strat['cost'] else COLOR_ORANGE)

    # Memory constraints note
    ax.text(0.5, 0.15, 'Model Context Windows:', fontsize=12, ha='center', weight='bold')
    ax.text(0.5, 0.08, 'GPT-3.5: 4K tokens | GPT-4: 8K-32K tokens | Claude: 100K tokens',
            fontsize=10, ha='center', color=COLOR_GRAY)

    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/context_strategy_selector_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[12/27] Generated: context_strategy_selector_bsc.pdf")

def generate_model_selection_matrix():
    """Chart 13: Model selection based on requirements"""
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.text(0.5, 0.95, 'Model Selection Matrix',
            fontsize=24, weight='bold', ha='center', color=COLOR_MAIN)

    # Models and their characteristics
    models = [
        {'name': 'GPT-4', 'cost': 5, 'quality': 5, 'speed': 2, 'privacy': 1},
        {'name': 'GPT-3.5', 'cost': 3, 'quality': 4, 'speed': 4, 'privacy': 1},
        {'name': 'Claude-2', 'cost': 4, 'quality': 4.5, 'speed': 3, 'privacy': 1},
        {'name': 'FLAN-T5-XL', 'cost': 1, 'quality': 3, 'speed': 4, 'privacy': 5},
        {'name': 'LLaMA-2-13B', 'cost': 1.5, 'quality': 3.5, 'speed': 3, 'privacy': 5},
        {'name': 'Mistral-7B', 'cost': 1, 'quality': 3, 'speed': 5, 'privacy': 5}
    ]

    # Create scatter plot: Cost vs Quality
    for model in models:
        # Determine color based on privacy
        if model['privacy'] == 5:
            color = COLOR_GREEN  # Local/private
        else:
            color = COLOR_RED if model['cost'] > 4 else COLOR_ORANGE  # Cloud

        # Plot point
        ax.scatter(model['quality'], model['cost'], s=model['speed']*100,
                  c=color, alpha=0.6, edgecolors=COLOR_MAIN, linewidth=2)

        # Label
        ax.text(model['quality'] + 0.1, model['cost'], model['name'],
                fontsize=10, va='center', color=COLOR_MAIN, weight='bold')

    # Axes labels
    ax.set_xlabel('Summary Quality →', fontsize=12, weight='bold', color=COLOR_MAIN)
    ax.set_ylabel('Cost per 1K Summaries ($) →', fontsize=12, weight='bold', color=COLOR_MAIN)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', color=COLOR_GRAY)

    # Legend
    ax.text(2, 4.5, 'Bubble Size = Speed', fontsize=10, style='italic', color=COLOR_GRAY)
    ax.text(2, 4, 'Green = Local/Private', fontsize=10, color=COLOR_GREEN)
    ax.text(2, 3.5, 'Orange/Red = Cloud API', fontsize=10, color=COLOR_ORANGE)

    # Recommendation zones
    ax.add_patch(patches.Rectangle((4.5, 0.5), 0.8, 1,
                                  facecolor=COLOR_GREEN, alpha=0.1))
    ax.text(4.9, 0.8, 'Best\nValue', fontsize=9, ha='center', color=COLOR_GREEN, weight='bold')

    ax.add_patch(patches.Rectangle((4.5, 4), 0.8, 1.5,
                                  facecolor=COLOR_BLUE, alpha=0.1))
    ax.text(4.9, 4.5, 'Premium\nQuality', fontsize=9, ha='center', color=COLOR_BLUE, weight='bold')

    ax.add_patch(patches.Rectangle((2.5, 0.5), 1, 1.5,
                                  facecolor=COLOR_ORANGE, alpha=0.1))
    ax.text(3, 1, 'Budget\nOption', fontsize=9, ha='center', color=COLOR_ORANGE, weight='bold')

    # Set limits
    ax.set_xlim(2, 5.5)
    ax.set_ylim(0, 6)

    # Style
    set_chart_style(ax)

    plt.tight_layout()
    plt.savefig('../figures/model_selection_matrix_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[13/27] Generated: model_selection_matrix_bsc.pdf")

def generate_production_deployment_guide():
    """Chart 14: Production deployment decision guide"""
    fig, ax = plt.subplots(figsize=(14, 10))

    ax.text(0.5, 0.95, 'Production Deployment Guide',
            fontsize=24, weight='bold', ha='center', color=COLOR_MAIN)

    # Deployment scenarios
    scenarios = [
        {'scale': 'Small\n(<100/day)', 'latency': 'Flexible', 'infra': 'Cloud API',
         'solution': 'GPT-3.5 API', 'x': 0.2, 'y': 0.7},
        {'scale': 'Small\n(<100/day)', 'latency': 'Real-time', 'infra': 'Edge/Local',
         'solution': 'Mistral-7B', 'x': 0.4, 'y': 0.7},
        {'scale': 'Medium\n(100-1K/day)', 'latency': 'Batch OK', 'infra': 'Hybrid',
         'solution': 'GPT-3.5 + Cache', 'x': 0.6, 'y': 0.7},
        {'scale': 'Medium\n(100-1K/day)', 'latency': 'Real-time', 'infra': 'On-premise',
         'solution': 'FLAN-T5-XL', 'x': 0.8, 'y': 0.7},
        {'scale': 'Large\n(>1K/day)', 'latency': 'Batch OK', 'infra': 'Cloud Scale',
         'solution': 'Batch GPT-3.5', 'x': 0.3, 'y': 0.45},
        {'scale': 'Large\n(>1K/day)', 'latency': 'Real-time', 'infra': 'GPU Cluster',
         'solution': 'LLaMA-2 Fleet', 'x': 0.7, 'y': 0.45}
    ]

    # Draw scenario boxes
    for scenario in scenarios:
        # Main box
        rect = FancyBboxPatch((scenario['x'] - 0.08, scenario['y'] - 0.05), 0.16, 0.1,
                              boxstyle="round,pad=0.01",
                              facecolor=COLOR_LIGHT, edgecolor=COLOR_GRAY, linewidth=1)
        ax.add_patch(rect)

        # Scale
        ax.text(scenario['x'], scenario['y'] + 0.02, scenario['scale'],
                fontsize=9, ha='center', weight='bold', color=COLOR_MAIN)

        # Latency
        ax.text(scenario['x'], scenario['y'] - 0.01, scenario['latency'],
                fontsize=8, ha='center', color=COLOR_BLUE)

        # Solution
        rect = FancyBboxPatch((scenario['x'] - 0.06, scenario['y'] - 0.12), 0.12, 0.05,
                              facecolor=COLOR_GREEN if 'Local' in scenario['infra'] or 'premise' in scenario['infra']
                              else COLOR_ORANGE,
                              alpha=0.3, edgecolor=COLOR_MAIN, linewidth=1)
        ax.add_patch(rect)
        ax.text(scenario['x'], scenario['y'] - 0.095, scenario['solution'],
                fontsize=9, ha='center', weight='bold', color=COLOR_MAIN)

    # Infrastructure requirements
    ax.text(0.5, 0.3, 'Infrastructure Requirements', fontsize=14, ha='center', weight='bold')

    infra_reqs = [
        {'type': 'Cloud API', 'cpu': 'N/A', 'gpu': 'N/A', 'ram': 'N/A', 'cost': 'High/month'},
        {'type': 'Edge Device', 'cpu': '4+ cores', 'gpu': 'Optional', 'ram': '8GB+', 'cost': 'Low'},
        {'type': 'Server', 'cpu': '8+ cores', 'gpu': '1x RTX', 'ram': '32GB+', 'cost': 'Medium'},
        {'type': 'Cluster', 'cpu': '64+ cores', 'gpu': '4+ A100', 'ram': '256GB+', 'cost': 'Very High'}
    ]

    # Table headers
    headers = ['Type', 'CPU', 'GPU', 'RAM', 'Cost']
    for i, header in enumerate(headers):
        x = 0.2 + i * 0.12
        ax.text(x, 0.22, header, fontsize=10, ha='center', weight='bold', color=COLOR_ACCENT)

    # Table rows
    for j, req in enumerate(infra_reqs):
        y = 0.18 - j * 0.03
        values = [req['type'], req['cpu'], req['gpu'], req['ram'], req['cost']]
        for i, val in enumerate(values):
            x = 0.2 + i * 0.12
            color = COLOR_GREEN if val == 'Low' else (COLOR_RED if 'High' in val else (COLOR_ORANGE if val == 'Medium' else COLOR_MAIN))
            ax.text(x, y, val, fontsize=9, ha='center', color=color)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/production_deployment_guide_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[14/27] Generated: production_deployment_guide_bsc.pdf")

# ============================================================================
# PART 4: COMPARISONS (7 charts)
# ============================================================================

def generate_cost_quality_scatter():
    """Chart 15: Cost vs quality scatter plot for different models"""
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.text(0.5, 0.95, 'Cost vs Quality: Model Comparison',
            fontsize=20, weight='bold', ha='center', transform=ax.transAxes, color=COLOR_MAIN)

    # Model data (cost per 1K summaries, quality score 0-10)
    models = [
        {'name': 'GPT-4', 'cost': 30, 'quality': 9.5, 'color': COLOR_RED},
        {'name': 'GPT-3.5', 'cost': 2, 'quality': 8.0, 'color': COLOR_ORANGE},
        {'name': 'Claude-2', 'cost': 8, 'quality': 9.0, 'color': COLOR_BLUE},
        {'name': 'FLAN-T5-XL', 'cost': 0.5, 'quality': 7.0, 'color': COLOR_GREEN},
        {'name': 'FLAN-T5-Base', 'cost': 0.2, 'quality': 6.0, 'color': COLOR_GREEN},
        {'name': 'LLaMA-2-13B', 'cost': 0.8, 'quality': 7.5, 'color': COLOR_GREEN},
        {'name': 'Mistral-7B', 'cost': 0.3, 'quality': 6.5, 'color': COLOR_GREEN},
        {'name': 'BART-large', 'cost': 0.4, 'quality': 6.8, 'color': COLOR_GRAY}
    ]

    for model in models:
        ax.scatter(model['cost'], model['quality'], s=300, c=model['color'],
                  alpha=0.7, edgecolors=COLOR_MAIN, linewidth=2)
        ax.annotate(model['name'], (model['cost'], model['quality']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, color=COLOR_MAIN, weight='bold')

    # Add zones
    ax.axvspan(0, 1, alpha=0.1, color=COLOR_GREEN)
    ax.text(0.5, 5.5, 'Budget\nZone', fontsize=10, ha='center', color=COLOR_GREEN, weight='bold')

    ax.axvspan(1, 10, alpha=0.1, color=COLOR_ORANGE)
    ax.text(5, 5.5, 'Balanced\nZone', fontsize=10, ha='center', color=COLOR_ORANGE, weight='bold')

    ax.axvspan(10, 35, alpha=0.1, color=COLOR_RED)
    ax.text(20, 5.5, 'Premium\nZone', fontsize=10, ha='center', color=COLOR_RED, weight='bold')

    ax.set_xlabel('Cost per 1000 Summaries ($)', fontsize=12, weight='bold', color=COLOR_MAIN)
    ax.set_ylabel('Summary Quality (0-10)', fontsize=12, weight='bold', color=COLOR_MAIN)

    # Add best value line
    ax.plot([0, 30], [5.5, 9.5], '--', color=COLOR_ACCENT, alpha=0.5, linewidth=2)
    ax.text(15, 7.5, 'Optimal Value Line', rotation=30, fontsize=9, color=COLOR_ACCENT, style='italic')

    ax.set_xlim(-1, 35)
    ax.set_ylim(5, 10)

    set_chart_style(ax)

    plt.tight_layout()
    plt.savefig('../figures/cost_quality_scatter_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[15/27] Generated: cost_quality_scatter_bsc.pdf")

def generate_timeline_evolution():
    """Chart 16: Evolution of summarization methods 2018-2025"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Title
    ax.text(0.5, 0.95, 'Evolution of Summarization: 2018-2025',
            fontsize=24, weight='bold', ha='center', color=COLOR_MAIN)

    # Timeline events
    events = [
        {'year': 2018, 'name': 'Rule-Based\nExtraction', 'y': 0.3, 'color': COLOR_GRAY},
        {'year': 2019, 'name': 'BERT\nFine-tuning', 'y': 0.4, 'color': COLOR_BLUE},
        {'year': 2020, 'name': 'BART/T5\nSpecialized', 'y': 0.5, 'color': COLOR_ORANGE},
        {'year': 2021, 'name': 'Pegasus\nDomain-specific', 'y': 0.55, 'color': COLOR_GREEN},
        {'year': 2022, 'name': 'GPT-3\nFew-shot', 'y': 0.65, 'color': COLOR_RED},
        {'year': 2023, 'name': 'ChatGPT\nConversational', 'y': 0.75, 'color': COLOR_ACCENT},
        {'year': 2024, 'name': 'GPT-4\nMultimodal', 'y': 0.8, 'color': COLOR_ACCENT},
        {'year': 2025, 'name': 'Claude-3\n100K context', 'y': 0.85, 'color': COLOR_ACCENT}
    ]

    # Draw timeline
    ax.plot([2018, 2025], [0.25, 0.25], '-', color=COLOR_GRAY, linewidth=3)

    for event in events:
        # Vertical line
        ax.plot([event['year'], event['year']], [0.25, event['y']],
                '--', color=event['color'], alpha=0.5)

        # Event bubble
        circle = Circle((event['year'], event['y']), 0.3,
                       facecolor=event['color'], alpha=0.3,
                       edgecolor=event['color'], linewidth=2)
        ax.add_patch(circle)

        # Text
        ax.text(event['year'], event['y'], event['name'],
                fontsize=10, ha='center', va='center', weight='bold', color=COLOR_MAIN)

    # Era labels
    ax.text(2019, 0.15, 'Extractive Era', fontsize=11, ha='center', style='italic', color=COLOR_GRAY)
    ax.text(2021, 0.15, 'Fine-tuning Era', fontsize=11, ha='center', style='italic', color=COLOR_ORANGE)
    ax.text(2024, 0.15, 'Prompt Era', fontsize=11, ha='center', style='italic', color=COLOR_ACCENT)

    # Annotations
    ax.annotate('Revolution:\nNo training needed!', xy=(2022, 0.65), xytext=(2022.5, 0.45),
                arrowprops=dict(arrowstyle='->', color=COLOR_RED, lw=2),
                fontsize=10, color=COLOR_RED, weight='bold')

    ax.set_xlim(2017.5, 2025.5)
    ax.set_ylim(0.1, 0.95)
    ax.set_xticks(range(2018, 2026))
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/timeline_evolution_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[16/27] Generated: timeline_evolution_bsc.pdf")

def generate_model_capability_comparison():
    """Chart 17: Model capabilities comparison table"""
    fig, ax = plt.subplots(figsize=(14, 10))

    ax.text(0.5, 0.95, 'Model Capability Comparison',
            fontsize=24, weight='bold', ha='center', color=COLOR_MAIN)

    # Models
    models = ['GPT-4', 'GPT-3.5', 'Claude-2', 'FLAN-T5', 'LLaMA-2', 'BART']

    # Capabilities (0-5 scale)
    capabilities = {
        'Context Length': [4, 3, 5, 2, 3, 2],
        'Summary Quality': [5, 4, 4.5, 3, 3.5, 3],
        'Speed': [2, 4, 3, 5, 4, 5],
        'Cost Efficiency': [1, 3, 2, 5, 5, 5],
        'Customizability': [2, 2, 2, 5, 5, 4],
        'Privacy': [1, 1, 1, 5, 5, 5]
    }

    # Create radar chart style comparison
    categories = list(capabilities.keys())
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Colors for each model
    model_colors = [COLOR_RED, COLOR_ORANGE, COLOR_BLUE, COLOR_GREEN, COLOR_ACCENT, COLOR_GRAY]

    # Plot each model
    for idx, (model, color) in enumerate(zip(models, model_colors)):
        values = [capabilities[cat][idx] for cat in categories]
        values += values[:1]

        # Draw the line
        x_pos = 0.15 + (idx % 3) * 0.3
        y_pos = 0.6 if idx < 3 else 0.25

        # Create mini radar chart
        ax_radar = fig.add_axes([x_pos, y_pos, 0.2, 0.2], projection='polar')
        ax_radar.plot(angles, values, 'o-', linewidth=2, color=color, label=model)
        ax_radar.fill(angles, values, alpha=0.25, color=color)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, size=6)
        ax_radar.set_ylim(0, 5)
        ax_radar.set_title(model, size=11, weight='bold', color=color, pad=10)
        ax_radar.grid(True)

    # Legend for capabilities
    ax.text(0.85, 0.5, 'Scoring:', fontsize=12, weight='bold', transform=ax.transAxes)
    ax.text(0.85, 0.45, '5 = Excellent', fontsize=10, transform=ax.transAxes, color=COLOR_GREEN)
    ax.text(0.85, 0.42, '4 = Good', fontsize=10, transform=ax.transAxes, color=COLOR_BLUE)
    ax.text(0.85, 0.39, '3 = Average', fontsize=10, transform=ax.transAxes, color=COLOR_ORANGE)
    ax.text(0.85, 0.36, '2 = Fair', fontsize=10, transform=ax.transAxes, color=COLOR_GRAY)
    ax.text(0.85, 0.33, '1 = Poor', fontsize=10, transform=ax.transAxes, color=COLOR_RED)

    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/model_capability_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[17/27] Generated: model_capability_comparison_bsc.pdf")

def generate_prompt_effectiveness():
    """Chart 18: Prompt technique effectiveness comparison"""
    fig, ax = plt.subplots(figsize=(12, 8))

    techniques = ['Zero-Shot', 'Few-Shot\n(3 ex)', 'Few-Shot\n(5 ex)', 'Chain-of-\nThought', 'System +\nFew-Shot']
    scores = [0.65, 0.75, 0.78, 0.82, 0.88]
    colors = [COLOR_GRAY, COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_ACCENT]

    bars = ax.bar(techniques, scores, color=colors, alpha=0.7, edgecolor=COLOR_MAIN, linewidth=2)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.2f}', ha='center', va='bottom', fontsize=11, weight='bold')

    # Add improvement annotations
    improvements = [None, '+15%', '+3%', '+5%', '+7%']
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        if imp:
            ax.text(bar.get_x() + bar.get_width()/2., 0.5,
                   imp, ha='center', fontsize=10, color=COLOR_GREEN, weight='bold')

    ax.set_title('Prompt Engineering: Effectiveness Comparison',
                fontsize=16, weight='bold', pad=20, color=COLOR_MAIN)
    ax.set_ylabel('ROUGE-L Score', fontsize=12, weight='bold', color=COLOR_MAIN)
    ax.set_xlabel('Prompting Technique', fontsize=12, weight='bold', color=COLOR_MAIN)

    # Add baseline
    ax.axhline(y=0.65, color=COLOR_RED, linestyle='--', alpha=0.5)
    ax.text(4.5, 0.66, 'Baseline', fontsize=10, color=COLOR_RED, style='italic')

    ax.set_ylim(0, 1.0)

    set_chart_style(ax)

    plt.tight_layout()
    plt.savefig('../figures/prompt_effectiveness_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[18/27] Generated: prompt_effectiveness_bsc.pdf")

def generate_latency_comparison():
    """Chart 19: Response time comparison across models"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Data
    models = ['Mistral\n7B\n(Local)', 'FLAN-T5\nBase\n(Local)', 'FLAN-T5\nXL\n(Local)',
              'GPT-3.5\n(API)', 'GPT-4\n(API)', 'Claude-2\n(API)']
    latencies = [0.8, 1.2, 2.5, 3.0, 8.0, 5.0]  # seconds for 500-word doc
    colors = [COLOR_GREEN, COLOR_GREEN, COLOR_GREEN, COLOR_ORANGE, COLOR_RED, COLOR_BLUE]

    # Create horizontal bar chart
    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, latencies, color=colors, alpha=0.7, edgecolor=COLOR_MAIN, linewidth=2)

    # Add value labels
    for bar, latency in zip(bars, latencies):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                f'{latency:.1f}s', ha='left', va='center', fontsize=11, weight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Response Time (seconds for 500-word document)', fontsize=12, weight='bold', color=COLOR_MAIN)
    ax.set_title('Latency Comparison: Local vs API Models', fontsize=16, weight='bold', pad=20, color=COLOR_MAIN)

    # Add zones
    ax.axvspan(0, 3, alpha=0.1, color=COLOR_GREEN)
    ax.text(1.5, 5.5, 'Real-time\n(<3s)', fontsize=10, ha='center', color=COLOR_GREEN, weight='bold')

    ax.axvspan(3, 6, alpha=0.1, color=COLOR_ORANGE)
    ax.text(4.5, 5.5, 'Acceptable\n(3-6s)', fontsize=10, ha='center', color=COLOR_ORANGE, weight='bold')

    ax.axvspan(6, 10, alpha=0.1, color=COLOR_RED)
    ax.text(8, 5.5, 'Slow\n(>6s)', fontsize=10, ha='center', color=COLOR_RED, weight='bold')

    ax.set_xlim(0, 10)

    set_chart_style(ax)

    plt.tight_layout()
    plt.savefig('../figures/latency_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[19/27] Generated: latency_comparison_bsc.pdf")

def generate_context_window_limits():
    """Chart 20: Context window limits by model"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Model context windows (in thousands of tokens)
    models = ['BART', 'T5', 'FLAN-T5', 'GPT-3.5\n(4K)', 'GPT-3.5\n(16K)', 'GPT-4', 'GPT-4\n(32K)', 'Claude-2\n(100K)', 'Claude-3\n(200K)']
    windows = [1, 0.5, 0.5, 4, 16, 8, 32, 100, 200]
    colors = [COLOR_GRAY if w < 2 else COLOR_ORANGE if w < 20 else COLOR_GREEN if w < 50 else COLOR_ACCENT for w in windows]

    # Create bar chart
    bars = ax.bar(models, windows, color=colors, alpha=0.7, edgecolor=COLOR_MAIN, linewidth=2)

    # Add value labels
    for bar, window in zip(bars, windows):
        height = bar.get_height()
        label = f'{int(window)}K' if window >= 1 else f'{int(window*1000)}'
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                label, ha='center', va='bottom', fontsize=10, weight='bold')

    # Add document size reference lines
    doc_sizes = [
        {'size': 2, 'label': 'Short article', 'color': COLOR_BLUE},
        {'size': 10, 'label': 'Research paper', 'color': COLOR_ORANGE},
        {'size': 50, 'label': 'Book chapter', 'color': COLOR_RED},
        {'size': 200, 'label': 'Entire book', 'color': COLOR_ACCENT}
    ]

    for doc in doc_sizes:
        ax.axhline(y=doc['size'], color=doc['color'], linestyle='--', alpha=0.5, linewidth=1)
        ax.text(len(models) - 0.5, doc['size'], doc['label'], fontsize=9,
                va='center', color=doc['color'], style='italic')

    ax.set_ylabel('Context Window (thousands of tokens)', fontsize=12, weight='bold', color=COLOR_MAIN)
    ax.set_title('Context Window Limits: How Much Text Can Each Model Process?',
                fontsize=16, weight='bold', pad=20, color=COLOR_MAIN)

    ax.set_ylim(0, 220)

    # Rotate x labels
    plt.xticks(rotation=45, ha='right')

    set_chart_style(ax)

    plt.tight_layout()
    plt.savefig('../figures/context_window_limits_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[20/27] Generated: context_window_limits_bsc.pdf")

def generate_price_breakdown():
    """Chart 21: Detailed price breakdown per 1000 tokens"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # Left panel: Input costs
    models_input = ['GPT-3.5', 'GPT-4', 'Claude-2', 'Claude-3']
    costs_input = [0.0005, 0.01, 0.008, 0.003]
    colors1 = [COLOR_GREEN, COLOR_RED, COLOR_ORANGE, COLOR_BLUE]

    bars1 = ax1.bar(models_input, costs_input, color=colors1, alpha=0.7,
                    edgecolor=COLOR_MAIN, linewidth=2)

    for bar, cost in zip(bars1, costs_input):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0002,
                f'\\${cost:.4f}', ha='center', fontsize=10, weight='bold')

    ax1.set_title('Input Token Costs', fontsize=14, weight='bold', color=COLOR_MAIN)
    ax1.set_ylabel('Cost per 1K tokens ($)', fontsize=11, color=COLOR_MAIN)

    # Right panel: Output costs
    costs_output = [0.0015, 0.03, 0.024, 0.015]

    bars2 = ax2.bar(models_input, costs_output, color=colors1, alpha=0.7,
                    edgecolor=COLOR_MAIN, linewidth=2)

    for bar, cost in zip(bars2, costs_output):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0005,
                f'\\${cost:.4f}', ha='center', fontsize=10, weight='bold')

    ax2.set_title('Output Token Costs', fontsize=14, weight='bold', color=COLOR_MAIN)
    ax2.set_ylabel('Cost per 1K tokens ($)', fontsize=11, color=COLOR_MAIN)

    # Main title
    fig.suptitle('API Pricing Comparison: Input vs Output Tokens',
                fontsize=16, weight='bold', y=1.02, color=COLOR_MAIN)

    # Example calculation
    fig.text(0.5, 0.02, 'Example: 10-page document (3K tokens) → 300-word summary (400 tokens)\n'
                       r'GPT-3.5: $0.0015 + $0.0006 = $0.0021  |  '
                       r'GPT-4: $0.030 + $0.012 = $0.042',
            ha='center', fontsize=10, style='italic', color=COLOR_GRAY)

    set_chart_style(ax1)
    set_chart_style(ax2)

    plt.tight_layout()
    plt.savefig('../figures/price_breakdown_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[21/27] Generated: price_breakdown_bsc.pdf")

# ============================================================================
# PART 5: MATHEMATICAL VISUALIZATIONS (6 charts)
# ============================================================================

def generate_temperature_distributions():
    """Chart 22: Temperature effect on probability distributions"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    # Logits (before softmax)
    logits = np.array([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0])
    words = ['good', 'great', 'nice', 'fine', 'okay', 'bad', 'poor']

    temperatures = [0.3, 0.7, 1.5]
    temp_colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_RED]
    temp_labels = ['Low (T=0.3)\nVery Focused', 'Medium (T=0.7)\nBalanced', 'High (T=1.5)\nVery Creative']

    for ax, temp, color, label in zip(axes, temperatures, temp_colors, temp_labels):
        # Apply temperature and softmax
        scaled_logits = logits / temp
        probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))

        bars = ax.bar(words, probs, color=color, alpha=0.7, edgecolor=COLOR_MAIN, linewidth=2)

        # Add probability values
        for bar, prob in zip(bars, probs):
            if prob > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{prob:.2f}', ha='center', fontsize=9)

        ax.set_title(label, fontsize=12, weight='bold', color=color)
        ax.set_ylabel('Probability', fontsize=11, color=COLOR_MAIN)
        ax.set_ylim(0, 0.8)
        ax.set_xticklabels(words, rotation=45, ha='right')

        set_chart_style(ax)

    fig.suptitle('Temperature Controls Randomness in Word Selection',
                fontsize=16, weight='bold', color=COLOR_MAIN)

    plt.tight_layout()
    plt.savefig('../figures/temperature_distributions_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[22/27] Generated: temperature_distributions_bsc.pdf")

def generate_top_p_cumulative():
    """Chart 23: Top-p cumulative probability cutoff visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Words sorted by probability
    words = ['excellent', 'great', 'good', 'outstanding', 'superb', 'nice', 'fine', 'okay', 'decent', 'average']
    probs = [0.25, 0.20, 0.15, 0.12, 0.08, 0.07, 0.05, 0.04, 0.02, 0.02]
    cumsum = np.cumsum(probs)

    # Create bar chart
    bars = ax.bar(range(len(words)), probs, color=COLOR_GRAY, alpha=0.7, edgecolor=COLOR_MAIN, linewidth=2)

    # Color bars within top-p threshold
    p_threshold = 0.9
    for i, (bar, cs) in enumerate(zip(bars, cumsum)):
        if cs <= p_threshold:
            bar.set_color(COLOR_GREEN)
            bar.set_alpha(0.8)
        else:
            bar.set_color(COLOR_RED)
            bar.set_alpha(0.3)

    # Draw cumulative line
    ax2 = ax.twinx()
    ax2.plot(range(len(words)), cumsum, 'o-', color=COLOR_ACCENT, linewidth=3, markersize=8)

    # Mark threshold
    ax2.axhline(y=p_threshold, color=COLOR_ACCENT, linestyle='--', linewidth=2)
    ax2.text(len(words)-1, p_threshold, f'p={p_threshold}', fontsize=12,
             ha='right', va='bottom', color=COLOR_ACCENT, weight='bold')

    # Annotations
    cutoff_idx = np.where(cumsum > p_threshold)[0][0]
    ax.axvspan(cutoff_idx - 0.5, len(words) - 0.5, alpha=0.1, color=COLOR_RED)
    ax.text(cutoff_idx + 1, 0.2, 'EXCLUDED\n(cumulative > 0.9)',
            fontsize=11, ha='center', color=COLOR_RED, weight='bold')

    # Labels
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right')
    ax.set_ylabel('Individual Probability', fontsize=12, weight='bold', color=COLOR_MAIN)
    ax2.set_ylabel('Cumulative Probability', fontsize=12, weight='bold', color=COLOR_ACCENT)
    ax.set_title('Top-p (Nucleus) Sampling: Keep Most Likely Words Until p=0.9',
                fontsize=16, weight='bold', pad=20, color=COLOR_MAIN)

    # Add probability values
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(i, bar.get_height() + 0.005, f'{prob:.2f}', ha='center', fontsize=9)

    set_chart_style(ax)
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('../figures/top_p_cumulative_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[23/27] Generated: top_p_cumulative_bsc.pdf")

def generate_beam_search_tree_math():
    """Chart 24: Beam search tree with mathematical scores"""
    fig, ax = plt.subplots(figsize=(14, 10))

    ax.text(0.5, 0.95, 'Beam Search: Mathematical Score Tracking (beam_width=3)',
            fontsize=20, weight='bold', ha='center', color=COLOR_MAIN)

    # Tree structure with log probabilities
    # Start
    ax.scatter(0.5, 0.85, s=500, c=COLOR_ACCENT, edgecolors=COLOR_MAIN, linewidth=2, zorder=3)
    ax.text(0.5, 0.85, 'START\nlog P=0', ha='center', va='center', fontsize=10, color='white', weight='bold')

    # Level 1 candidates (keep top 3)
    level1 = [
        {'x': 0.2, 'word': 'The', 'log_p': -0.8, 'keep': True},
        {'x': 0.35, 'word': 'A', 'log_p': -1.2, 'keep': True},
        {'x': 0.5, 'word': 'This', 'log_p': -1.5, 'keep': True},
        {'x': 0.65, 'word': 'That', 'log_p': -2.0, 'keep': False},
        {'x': 0.8, 'word': 'Our', 'log_p': -2.5, 'keep': False}
    ]

    for node in level1:
        color = COLOR_GREEN if node['keep'] else COLOR_RED
        alpha = 0.8 if node['keep'] else 0.3
        size = 400 if node['keep'] else 300

        ax.scatter(node['x'], 0.65, s=size, c=color, alpha=alpha,
                  edgecolors=COLOR_MAIN, linewidth=2, zorder=2)
        ax.text(node['x'], 0.65, f"{node['word']}\nΣ={node['log_p']:.1f}",
               ha='center', va='center', fontsize=9, weight='bold' if node['keep'] else 'normal')

        # Connection from start
        ax.plot([0.5, node['x']], [0.82, 0.68], 'k-', alpha=0.3, linewidth=1)

    # Level 2 (expand only kept nodes)
    level2_text = 0.43
    level2_groups = [
        # From "The"
        [
            {'x': 0.1, 'phrase': 'The study', 'total': -1.5, 'keep': True},
            {'x': 0.2, 'phrase': 'The research', 'total': -1.8, 'keep': True},
            {'x': 0.3, 'phrase': 'The analysis', 'total': -2.5, 'keep': False}
        ],
        # From "A"
        [
            {'x': 0.35, 'phrase': 'A new', 'total': -2.0, 'keep': True},
            {'x': 0.45, 'phrase': 'A recent', 'total': -2.8, 'keep': False}
        ],
        # From "This"
        [
            {'x': 0.55, 'phrase': 'This paper', 'total': -3.0, 'keep': False},
            {'x': 0.65, 'phrase': 'This work', 'total': -3.2, 'keep': False}
        ]
    ]

    kept_count = 0
    for group_idx, group in enumerate(level2_groups):
        parent_x = level1[group_idx]['x']
        for node in group:
            color = COLOR_GREEN if node['keep'] else COLOR_RED
            alpha = 0.8 if node['keep'] else 0.3
            size = 350 if node['keep'] else 250

            ax.scatter(node['x'], level2_text, s=size, c=color, alpha=alpha,
                      edgecolors=COLOR_MAIN, linewidth=1.5, zorder=2)
            ax.text(node['x'], level2_text, f"{node['phrase']}\nΣ={node['total']:.1f}",
                   ha='center', va='center', fontsize=8, weight='bold' if node['keep'] else 'normal')

            # Connection from parent
            ax.plot([parent_x, node['x']], [0.62, level2_text+0.03], 'k-', alpha=0.3, linewidth=1)

            if node['keep']:
                kept_count += 1

    # Score formula
    ax.text(0.5, 0.28, 'Score Formula:', fontsize=12, weight='bold', color=COLOR_ACCENT)
    ax.text(0.5, 0.23, r'$\text{Score} = \sum_{i=1}^{t} \log P(w_i | w_{1:i-1})$',
            fontsize=14, ha='center', color=COLOR_MAIN)

    # Statistics box
    stats_text = f"Step 1: 5 candidates → 3 kept\n" \
                f"Step 2: 3×3 = 9 candidates → 3 kept\n" \
                f"Total paths explored: 14\n" \
                f"Paths kept: 3 (21%)"

    ax.text(0.85, 0.5, stats_text, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLOR_LIGHT, edgecolor=COLOR_GRAY))

    # Legend
    ax.scatter(0.85, 0.35, s=200, c=COLOR_GREEN, alpha=0.8, edgecolors=COLOR_MAIN, linewidth=2)
    ax.text(0.88, 0.35, 'Kept', fontsize=10, va='center')
    ax.scatter(0.85, 0.30, s=200, c=COLOR_RED, alpha=0.3, edgecolors=COLOR_MAIN, linewidth=2)
    ax.text(0.88, 0.30, 'Pruned', fontsize=10, va='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(0.15, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/beam_search_tree_math_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[24/27] Generated: beam_search_tree_math_bsc.pdf")

def generate_repetition_penalty_formula():
    """Chart 25: Repetition penalty mathematical breakdown"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top panel: Formula visualization
    ax1.text(0.5, 0.8, 'Repetition Penalty Formula', fontsize=18, weight='bold',
             ha='center', color=COLOR_MAIN)

    # Formula components (simplified for matplotlib - no \begin{cases})
    ax1.text(0.5, 0.6, r'$P_{adjusted}(w) = P(w) / \alpha$',
            fontsize=16, ha='center', color=COLOR_ACCENT)
    ax1.text(0.5, 0.48, 'if w ∈ previous tokens',
            fontsize=12, ha='center', color=COLOR_RED)

    ax1.text(0.5, 0.35, r'$P_{adjusted}(w) = P(w)$',
            fontsize=16, ha='center', color=COLOR_ACCENT)
    ax1.text(0.5, 0.23, 'otherwise',
            fontsize=12, ha='center', color=COLOR_GRAY)

    ax1.text(0.5, 0.1, r'where $\alpha$ = repetition penalty (typically 1.0 - 1.5)',
            fontsize=12, ha='center', style='italic', color=COLOR_GRAY)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # Bottom panel: Effect visualization
    words = ['excellent', 'great', 'study', 'shows', 'research', 'indicates', 'found', 'analysis']
    original_probs = [0.15, 0.14, 0.12, 0.11, 0.13, 0.12, 0.12, 0.11]
    previous_words = ['study', 'research', 'analysis']  # Already used

    # Apply penalty
    penalty = 1.3
    adjusted_probs = []
    for word, prob in zip(words, original_probs):
        if word in previous_words:
            adjusted_probs.append(prob / penalty)
        else:
            adjusted_probs.append(prob)

    # Normalize
    total = sum(adjusted_probs)
    adjusted_probs = [p/total for p in adjusted_probs]

    x = np.arange(len(words))
    width = 0.35

    # Plot bars
    bars1 = ax2.bar(x - width/2, original_probs, width, label='Original',
                    color=COLOR_GRAY, alpha=0.7, edgecolor=COLOR_MAIN, linewidth=2)
    bars2 = ax2.bar(x + width/2, adjusted_probs, width, label=f'With Penalty ({penalty})',
                    color=COLOR_ACCENT, alpha=0.7, edgecolor=COLOR_MAIN, linewidth=2)

    # Color repeated words differently
    for i, word in enumerate(words):
        if word in previous_words:
            bars1[i].set_color(COLOR_ORANGE)
            bars2[i].set_color(COLOR_RED)

    # Labels
    ax2.set_xlabel('Next Word Candidates', fontsize=12, weight='bold', color=COLOR_MAIN)
    ax2.set_ylabel('Probability', fontsize=12, weight='bold', color=COLOR_MAIN)
    ax2.set_title('Effect of Repetition Penalty on Word Probabilities',
                 fontsize=14, weight='bold', color=COLOR_MAIN)
    ax2.set_xticks(x)
    ax2.set_xticklabels(words, rotation=45, ha='right')

    # Add arrows showing reduction
    for i, word in enumerate(words):
        if word in previous_words:
            y_start = original_probs[i]
            y_end = adjusted_probs[i]
            ax2.annotate('', xy=(i + width/2, y_end), xytext=(i - width/2, y_start),
                        arrowprops=dict(arrowstyle='->', color=COLOR_RED, lw=2))

    ax2.legend()
    ax2.text(0.5, 0.22, '* Orange/Red bars = words that appeared previously',
            transform=ax2.transAxes, fontsize=10, style='italic', color=COLOR_ORANGE)

    set_chart_style(ax2)

    plt.tight_layout()
    plt.savefig('../figures/repetition_penalty_formula_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[25/27] Generated: repetition_penalty_formula_bsc.pdf")

def generate_attention_weight_heatmap():
    """Chart 26: Attention weight heatmap for summarization"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Sample text tokens
    source_tokens = ['The', 'study', 'found', 'significant', 'improvements', 'in', 'patient', 'outcomes',
                     'after', 'implementing', 'the', 'new', 'treatment', 'protocol', '.']
    summary_tokens = ['Study', 'shows', 'improved', 'patient', 'outcomes']

    # Create attention matrix (summary attending to source)
    np.random.seed(42)
    attention = np.random.random((len(summary_tokens), len(source_tokens)))

    # Make some connections stronger (semantic relevance)
    attention[0, 1] = 0.9  # Study → study
    attention[1, 2] = 0.8  # shows → found
    attention[2, 4] = 0.85  # improved → improvements
    attention[3, 6] = 0.95  # patient → patient
    attention[4, 7] = 0.9  # outcomes → outcomes

    # Normalize rows
    for i in range(len(summary_tokens)):
        attention[i] = attention[i] / attention[i].sum()

    # Plot heatmap
    im = ax.imshow(attention, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.3)

    # Set ticks
    ax.set_xticks(np.arange(len(source_tokens)))
    ax.set_yticks(np.arange(len(summary_tokens)))
    ax.set_xticklabels(source_tokens, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(summary_tokens, fontsize=11)

    # Labels
    ax.set_xlabel('Source Document Tokens', fontsize=12, weight='bold', color=COLOR_MAIN)
    ax.set_ylabel('Summary Tokens', fontsize=12, weight='bold', color=COLOR_MAIN)
    ax.set_title('Attention Weights: Which Source Words Influence Each Summary Word',
                fontsize=14, weight='bold', pad=20, color=COLOR_MAIN)

    # Add value annotations for strong connections
    for i in range(len(summary_tokens)):
        for j in range(len(source_tokens)):
            if attention[i, j] > 0.15:
                ax.text(j, i, f'{attention[i, j]:.2f}', ha='center', va='center',
                       color='white', fontsize=9, weight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20, fontsize=11)

    # Add interpretation
    ax.text(0.5, -0.15, 'Brighter = Stronger attention | Each row sums to 1.0',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic', color=COLOR_GRAY)

    plt.tight_layout()
    plt.savefig('../figures/attention_weight_heatmap_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[26/27] Generated: attention_weight_heatmap_bsc.pdf")

def generate_perplexity_length_curve():
    """Chart 27: Perplexity vs summary length analysis"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Data: summary length vs quality (perplexity)
    lengths = np.array([10, 25, 50, 75, 100, 150, 200, 300, 400, 500])
    perplexity = np.array([45, 32, 25, 22, 20, 19, 18.5, 19, 20, 22])  # Lower is better

    # Plot main curve
    ax.plot(lengths, perplexity, 'o-', color=COLOR_ACCENT, linewidth=3,
            markersize=8, label='Perplexity (lower is better)')

    # Fill zones
    ax.axhspan(18, 22, alpha=0.1, color=COLOR_GREEN)
    ax.text(450, 20, 'Optimal\nZone', fontsize=11, ha='center', color=COLOR_GREEN, weight='bold')

    ax.axhspan(22, 30, alpha=0.1, color=COLOR_ORANGE)
    ax.text(450, 26, 'Acceptable', fontsize=11, ha='center', color=COLOR_ORANGE)

    ax.axhspan(30, 50, alpha=0.1, color=COLOR_RED)
    ax.text(450, 40, 'Poor\nQuality', fontsize=11, ha='center', color=COLOR_RED)

    # Mark sweet spot
    min_idx = np.argmin(perplexity)
    ax.scatter(lengths[min_idx], perplexity[min_idx], s=200, c=COLOR_GREEN,
              edgecolors=COLOR_MAIN, linewidth=3, zorder=3)
    ax.annotate('Sweet Spot\n(150 words)', xy=(lengths[min_idx], perplexity[min_idx]),
               xytext=(200, 15), arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=2),
               fontsize=11, color=COLOR_GREEN, weight='bold')

    # Add annotations for extremes
    ax.annotate('Too short:\nMissing info', xy=(25, 32), xytext=(50, 38),
               arrowprops=dict(arrowstyle='->', color=COLOR_RED, lw=1.5),
               fontsize=10, color=COLOR_RED)

    ax.annotate('Too long:\nRedundant', xy=(400, 20), xytext=(350, 28),
               arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=1.5),
               fontsize=10, color=COLOR_ORANGE)

    # Labels
    ax.set_xlabel('Summary Length (words)', fontsize=12, weight='bold', color=COLOR_MAIN)
    ax.set_ylabel('Perplexity Score', fontsize=12, weight='bold', color=COLOR_MAIN)
    ax.set_title('Summary Quality vs Length: Finding the Sweet Spot',
                fontsize=16, weight='bold', pad=20, color=COLOR_MAIN)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', color=COLOR_GRAY)

    # Add secondary axis for compression ratio
    ax2 = ax.twiny()
    compression_ratios = [95, 90, 85, 80, 75, 65, 55, 40, 30, 25]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(lengths)
    ax2.set_xticklabels([f'{cr}%' for cr in compression_ratios], fontsize=9)
    ax2.set_xlabel('Compression Ratio (assuming 2000-word source)', fontsize=11, color=COLOR_GRAY)

    set_chart_style(ax)

    plt.tight_layout()
    plt.savefig('../figures/perplexity_length_curve_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[27/27] Generated: perplexity_length_curve_bsc.pdf")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("GENERATING COMPLETE LLM SUMMARIZATION CHART SUITE")
    print("BSc Discovery Color Scheme - 27 Charts Total")
    print("="*60 + "\n")

    # Part 1: Missing charts (2)
    print("Part 1: Generating missing charts...")
    generate_zero_shot_prompt()
    generate_few_shot_prompt()

    # Part 2: Technical diagrams (6)
    print("\nPart 2: Generating technical diagrams...")
    generate_system_prompt_anatomy()
    generate_token_flow_pipeline()
    generate_attention_sink_visual()
    generate_flan_t5_architecture()
    generate_chunking_overlap_diagram()
    generate_multi_doc_deduplication()

    # Part 3: Decision trees (6)
    print("\nPart 3: Generating decision trees...")
    generate_prompt_engineering_flowchart()
    generate_method_selection_guide()
    generate_parameter_tuning_tree()
    generate_context_strategy_selector()
    generate_model_selection_matrix()
    generate_production_deployment_guide()

    # Part 4: Comparisons (7)
    print("\nPart 4: Generating comparison charts...")
    generate_cost_quality_scatter()
    generate_timeline_evolution()
    generate_model_capability_comparison()
    generate_prompt_effectiveness()
    generate_latency_comparison()
    generate_context_window_limits()
    generate_price_breakdown()

    # Part 5: Mathematical visualizations (6)
    print("\nPart 5: Generating mathematical visualizations...")
    generate_temperature_distributions()
    generate_top_p_cumulative()
    generate_beam_search_tree_math()
    generate_repetition_penalty_formula()
    generate_attention_weight_heatmap()
    generate_perplexity_length_curve()

    print("\n" + "="*60)
    print("✓ GENERATION COMPLETE: All 27 charts generated successfully!")
    print("Chart ratio improved: 27 charts / 48 slides = 0.56 ratio")
    print("All charts use BSc Discovery color scheme")
    print("="*60 + "\n")