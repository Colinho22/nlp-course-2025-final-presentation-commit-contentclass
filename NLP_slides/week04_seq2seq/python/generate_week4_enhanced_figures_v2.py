#!/usr/bin/env python3
"""
Enhanced Figure Generation for Week 4: Sequence-to-Sequence Models
Generates high-quality educational visualizations with consistent color scheme.

Author: NLP Course 2025
Created: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set consistent style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Educational color scheme (from repository standards)
COLOR_CURRENT = '#FF6B6B'   # Red - current position/focus
COLOR_CONTEXT = '#4ECDC4'   # Teal - context/surrounding
COLOR_PREDICT = '#95E77E'   # Green - predictions/output
COLOR_NEUTRAL = '#E0E0E0'   # Gray - neutral elements
COLOR_ATTENTION = '#FAB563' # Orange - attention mechanism
COLOR_ENCODER = '#74B3F7'   # Blue - encoder components
COLOR_DECODER = '#F06292'   # Pink - decoder components
COLOR_BOTTLENECK = '#FF5722' # Red-orange - bottleneck problem

def save_figure(filename, dpi=300, bbox_inches='tight'):
    """Save figure with consistent settings."""
    plt.savefig(f'../figures/{filename}', dpi=dpi, bbox_inches=bbox_inches)
    print(f"[OK] Generated: {filename}")

def create_encoder_decoder_architecture():
    """Create comprehensive encoder-decoder architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Input sequence
    input_words = ['The', 'cat', 'sat', 'on', 'mat']
    for i, word in enumerate(input_words):
        rect = FancyBboxPatch((1 + i*1.5, 6.5), 1.2, 0.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor=COLOR_ENCODER, alpha=0.7,
                             edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(1.6 + i*1.5, 6.9, word, ha='center', va='center', 
                fontsize=11, fontweight='bold')
    
    # Encoder RNNs
    encoder_states = []
    for i in range(5):
        circle = Circle((1.6 + i*1.5, 5), 0.4, 
                       facecolor=COLOR_ENCODER, alpha=0.8,
                       edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(1.6 + i*1.5, 5, f'h{i+1}', ha='center', va='center', 
                fontsize=9, fontweight='bold', color='white')
        encoder_states.append((1.6 + i*1.5, 5))
        
        # Arrows from input to encoder
        ax.arrow(1.6 + i*1.5, 6.4, 0, -0.9, head_width=0.1, head_length=0.1,
                fc='black', ec='black')
    
    # Horizontal connections between encoder states
    for i in range(4):
        ax.arrow(1.6 + i*1.5 + 0.4, 5, 0.7, 0, head_width=0.1, head_length=0.1,
                fc=COLOR_ENCODER, ec=COLOR_ENCODER, linewidth=2)
    
    # Context vector (bottleneck)
    context_rect = FancyBboxPatch((8.5, 4.5), 2, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=COLOR_BOTTLENECK, alpha=0.8,
                                 edgecolor='black', linewidth=2)
    ax.add_patch(context_rect)
    ax.text(9.5, 5, 'Context\nVector', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white')
    
    # Arrow from last encoder state to context
    ax.arrow(8.1, 5, 0.3, 0, head_width=0.15, head_length=0.1,
            fc=COLOR_BOTTLENECK, ec=COLOR_BOTTLENECK, linewidth=3)
    
    # Decoder states
    output_words = ['Le', 'chat', 'est', 'assis', '<EOS>']
    for i in range(5):
        circle = Circle((1.6 + i*1.5, 2.5), 0.4, 
                       facecolor=COLOR_DECODER, alpha=0.8,
                       edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(1.6 + i*1.5, 2.5, f's{i+1}', ha='center', va='center', 
                fontsize=9, fontweight='bold', color='white')
        
        # Output words
        rect = FancyBboxPatch((1 + i*1.5, 1), 1.2, 0.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor=COLOR_DECODER, alpha=0.7,
                             edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(1.6 + i*1.5, 1.4, output_words[i], ha='center', va='center', 
                fontsize=11, fontweight='bold')
        
        # Arrows from decoder to output
        ax.arrow(1.6 + i*1.5, 2.1, 0, -0.7, head_width=0.1, head_length=0.1,
                fc='black', ec='black')
    
    # Horizontal connections between decoder states
    for i in range(4):
        ax.arrow(1.6 + i*1.5 + 0.4, 2.5, 0.7, 0, head_width=0.1, head_length=0.1,
                fc=COLOR_DECODER, ec=COLOR_DECODER, linewidth=2)
    
    # Arrows from context to all decoder states
    for i in range(5):
        start_x, start_y = 9.5, 4.4
        end_x, end_y = 1.6 + i*1.5, 2.9
        
        arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                               connectionstyle="arc3,rad=0.2", 
                               arrowstyle='->', 
                               mutation_scale=15,
                               color=COLOR_BOTTLENECK, linewidth=2, alpha=0.7)
        ax.add_patch(arrow)
    
    # Labels
    ax.text(4, 7.8, 'Input Sequence', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    ax.text(4, 3.8, 'Encoder', ha='center', va='center', 
            fontsize=14, fontweight='bold', color=COLOR_ENCODER)
    ax.text(4, 0.2, 'Output Sequence', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    ax.text(4, 1.8, 'Decoder', ha='center', va='center', 
            fontsize=14, fontweight='bold', color=COLOR_DECODER)
    
    # Information flow arrow
    ax.arrow(11.5, 5, 1.5, 0, head_width=0.2, head_length=0.2,
            fc=COLOR_CURRENT, ec=COLOR_CURRENT, linewidth=3)
    ax.text(12.25, 5.5, 'Information\nFlow', ha='center', va='center', 
            fontsize=12, fontweight='bold', color=COLOR_CURRENT)
    
    plt.title('Sequence-to-Sequence Architecture: Encoder-Decoder with Context Vector',
             fontsize=18, fontweight='bold', pad=20)
    
    save_figure('week4_encoder_decoder_architecture.pdf')
    plt.close()

def create_information_bottleneck_problem():
    """Visualize the information bottleneck problem."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Sentence length vs information content
    sentence_lengths = np.arange(1, 51)
    information_content = sentence_lengths * 10  # Assume 10 bits per word
    context_capacity_256 = np.full_like(sentence_lengths, 256)
    context_capacity_512 = np.full_like(sentence_lengths, 512)
    
    ax1.plot(sentence_lengths, information_content, linewidth=3, 
             color=COLOR_CURRENT, label='Information Content')
    ax1.axhline(y=256, color=COLOR_BOTTLENECK, linestyle='--', linewidth=2,
               label='256-dim Context Capacity')
    ax1.axhline(y=512, color=COLOR_CONTEXT, linestyle='--', linewidth=2,
               label='512-dim Context Capacity')
    
    # Fill bottleneck area
    ax1.fill_between(sentence_lengths, information_content, context_capacity_256, 
                    where=(information_content > context_capacity_256),
                    color=COLOR_BOTTLENECK, alpha=0.3, label='Information Loss')
    
    ax1.set_xlabel('Sentence Length (words)', fontsize=12)
    ax1.set_ylabel('Information (bits)', fontsize=12)
    ax1.set_title('The Information Bottleneck Problem', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Translation quality vs sentence length
    lengths = [5, 10, 15, 20, 25, 30, 40, 50]
    quality_vanilla = [0.9, 0.85, 0.75, 0.6, 0.45, 0.3, 0.15, 0.1]
    quality_attention = [0.92, 0.89, 0.87, 0.84, 0.81, 0.78, 0.72, 0.68]
    
    ax2.plot(lengths, quality_vanilla, 'o-', linewidth=3, markersize=8,
             color=COLOR_BOTTLENECK, label='Vanilla Seq2Seq')
    ax2.plot(lengths, quality_attention, 'o-', linewidth=3, markersize=8,
             color=COLOR_ATTENTION, label='With Attention')
    
    ax2.set_xlabel('Sentence Length (words)', fontsize=12)
    ax2.set_ylabel('Translation Quality (BLEU)', fontsize=12)
    ax2.set_title('Quality Degradation with Length', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Context vector compression visualization
    np.random.seed(42)
    sentence_representations = []
    for length in [3, 8, 15, 25]:
        # Simulate how context vector changes with sentence length
        context = np.random.randn(8)
        # Longer sentences get more "averaged out"
        context = context / np.sqrt(length)
        sentence_representations.append(context)
    
    context_matrix = np.array(sentence_representations)
    im = ax3.imshow(context_matrix.T, aspect='auto', cmap='RdBu_r', 
                   vmin=-1, vmax=1)
    ax3.set_xlabel('Sentence Length', fontsize=12)
    ax3.set_ylabel('Context Dimension', fontsize=12)
    ax3.set_title('Context Vector Compression Effect', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(['3 words', '8 words', '15 words', '25 words'])
    
    # 4. Memory vs computation trade-off
    context_dims = [64, 128, 256, 512, 1024]
    memory_usage = [d * 4 / 1024 for d in context_dims]  # KB
    computation_time = [d * 0.001 for d in context_dims]  # seconds
    quality_scores = [0.6, 0.75, 0.85, 0.9, 0.92]
    
    ax4_twin = ax4.twinx()
    
    bars1 = ax4.bar([i-0.2 for i in range(len(context_dims))], memory_usage, 
                   width=0.4, color=COLOR_CONTEXT, alpha=0.7, label='Memory (KB)')
    bars2 = ax4.bar([i+0.2 for i in range(len(context_dims))], computation_time, 
                   width=0.4, color=COLOR_PREDICT, alpha=0.7, label='Time (s)')
    
    line = ax4_twin.plot(range(len(context_dims)), quality_scores, 'o-', 
                        linewidth=3, markersize=8, color=COLOR_CURRENT, 
                        label='Quality')
    
    ax4.set_xlabel('Context Vector Dimensions', fontsize=12)
    ax4.set_ylabel('Resources', fontsize=12)
    ax4_twin.set_ylabel('Translation Quality', fontsize=12)
    ax4.set_title('Context Size Trade-offs', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(context_dims)))
    ax4.set_xticklabels(context_dims)
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    save_figure('week4_information_bottleneck_analysis.pdf')
    plt.close()

def create_attention_mechanism_visualization():
    """Create comprehensive attention mechanism visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Attention computation steps
    ax1.text(0.5, 0.9, 'Attention Mechanism Steps', ha='center', va='center',
             transform=ax1.transAxes, fontsize=16, fontweight='bold')
    
    steps = [
        '1. Compute similarity scores',
        '2. Apply softmax normalization', 
        '3. Weight encoder states',
        '4. Sum to get context vector'
    ]
    
    colors = [COLOR_ENCODER, COLOR_ATTENTION, COLOR_CONTEXT, COLOR_PREDICT]
    
    for i, (step, color) in enumerate(zip(steps, colors)):
        rect = FancyBboxPatch((0.1, 0.7 - i*0.15), 0.8, 0.1,
                             boxstyle="round,pad=0.02",
                             facecolor=color, alpha=0.7,
                             edgecolor='black', linewidth=1)
        ax1.add_patch(rect)
        ax1.text(0.5, 0.75 - i*0.15, step, ha='center', va='center',
                transform=ax1.transAxes, fontsize=12, fontweight='bold')
        
        if i < len(steps) - 1:
            ax1.arrow(0.5, 0.65 - i*0.15, 0, -0.05, transform=ax1.transAxes,
                     head_width=0.03, head_length=0.02, fc='black', ec='black')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # 2. Attention heatmap example
    source_words = ['The', 'black', 'cat', 'sat', 'on', 'mat']
    target_words = ['Le', 'chat', 'noir', 'est', 'assis']
    
    # Create realistic attention pattern
    np.random.seed(42)
    attention_matrix = np.random.random((len(target_words), len(source_words))) * 0.1
    
    # Add strong alignments
    alignments = [(0, 0), (1, 2), (2, 1), (3, 3), (4, 3), (4, 4)]
    for tgt_idx, src_idx in alignments:
        if tgt_idx < len(target_words):
            attention_matrix[tgt_idx, src_idx] += 0.7
    
    # Normalize rows
    for i in range(len(target_words)):
        attention_matrix[i] /= attention_matrix[i].sum()
    
    im = ax2.imshow(attention_matrix, cmap='Blues', aspect='auto')
    ax2.set_xticks(range(len(source_words)))
    ax2.set_xticklabels(source_words)
    ax2.set_yticks(range(len(target_words)))
    ax2.set_yticklabels(target_words)
    ax2.set_xlabel('Source (English)', fontsize=12)
    ax2.set_ylabel('Target (French)', fontsize=12)
    ax2.set_title('Attention Weights Heatmap', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Attention Weight', fontsize=10)
    
    # 3. Attention types comparison
    attention_types = ['Dot Product', 'Scaled Dot Product', 'Additive (Bahdanau)']
    performance = [0.75, 0.82, 0.85]
    complexity = [1, 1.1, 2.5]
    
    x = np.arange(len(attention_types))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, performance, width, label='Performance',
                   color=COLOR_PREDICT, alpha=0.8)
    bars2 = ax3.bar(x + width/2, [c/3 for c in complexity], width, 
                   label='Complexity (scaled)', color=COLOR_CURRENT, alpha=0.8)
    
    ax3.set_xlabel('Attention Type', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Attention Mechanisms Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(attention_types)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar1, bar2, perf, comp in zip(bars1, bars2, performance, complexity):
        ax3.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                f'{perf:.2f}', ha='center', va='bottom', fontsize=10)
        ax3.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                f'{comp:.1f}x', ha='center', va='bottom', fontsize=10)
    
    # 4. Attention evolution timeline
    years = [2014, 2015, 2016, 2017, 2018, 2019]
    models = ['Seq2Seq', 'Attention', 'Google NMT', 'Transformer', 'BERT', 'GPT-2']
    bleu_scores = [25, 35, 45, 52, 58, 60]
    
    ax4.plot(years, bleu_scores, 'o-', linewidth=3, markersize=10,
             color=COLOR_ATTENTION, markerfacecolor=COLOR_PREDICT,
             markeredgecolor=COLOR_ATTENTION, markeredgewidth=2)
    
    # Annotate key models
    for year, model, score in zip(years, models, bleu_scores):
        ax4.annotate(model, (year, score), 
                    textcoords="offset points", xytext=(0,15), ha='center',
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.3', 
                                         facecolor='wheat', alpha=0.7))
    
    ax4.set_xlabel('Year', fontsize=12)
    ax4.set_ylabel('BLEU Score', fontsize=12)
    ax4.set_title('Evolution of Translation Quality', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(20, 65)
    
    plt.tight_layout()
    save_figure('week4_attention_comprehensive_analysis.pdf')
    plt.close()

def create_beam_search_visualization():
    """Create beam search algorithm visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Beam search tree
    ax1.set_xlim(-0.5, 4.5)
    ax1.set_ylim(-0.5, 5.5)
    ax1.set_title('Beam Search Tree (beam_size=2)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Node positions
    positions = {
        'START': (0, 2.5),
        'Le': (1, 3.5),
        'Un': (1, 1.5),
        'La': (1, 0.5),
        'chat1': (2, 4),
        'chat2': (2, 3),
        'chien': (2, 2),
        'chat3': (2, 1),
        'noir1': (3, 4),
        'noir2': (3, 3),
        'dort': (3, 2),
        'assis1': (4, 4),
        'assis2': (4, 3),
    }
    
    # Edges with probabilities and beam status
    edges = [
        ('START', 'Le', 0.6, True),
        ('START', 'Un', 0.3, True),
        ('START', 'La', 0.1, False),
        ('Le', 'chat1', 0.48, True),  # 0.6 * 0.8
        ('Le', 'chien', 0.12, False),  # 0.6 * 0.2
        ('Un', 'chat2', 0.21, True),  # 0.3 * 0.7
        ('Un', 'chat3', 0.09, False),  # 0.3 * 0.3
        ('La', 'chat4', 0.08, False),  # 0.1 * 0.8
        ('chat1', 'noir1', 0.384, True),  # 0.48 * 0.8
        ('chat2', 'noir2', 0.147, True),  # 0.21 * 0.7
        ('chien', 'dort', 0.072, False),  # 0.12 * 0.6
        ('noir1', 'assis1', 0.307, True),  # 0.384 * 0.8
        ('noir2', 'assis2', 0.103, True),  # 0.147 * 0.7
    ]
    
    # Draw edges
    for start, end, prob, in_beam in edges:
        if start in positions and end in positions:
            start_pos = positions[start]
            end_pos = positions[end]
            
            color = COLOR_PREDICT if in_beam else COLOR_NEUTRAL
            alpha = 1.0 if in_beam else 0.3
            linewidth = 2 if in_beam else 1
            
            ax1.arrow(start_pos[0] + 0.15, start_pos[1], 
                     end_pos[0] - start_pos[0] - 0.3, 
                     end_pos[1] - start_pos[1],
                     head_width=0.08, head_length=0.08,
                     fc=color, ec=color, linewidth=linewidth, alpha=alpha)
            
            # Add probability labels
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = (start_pos[1] + end_pos[1]) / 2
            ax1.text(mid_x, mid_y, f'{prob:.3f}', fontsize=8, ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Draw nodes
    node_labels = {
        'START': 'START',
        'Le': 'Le',
        'Un': 'Un', 
        'La': 'La',
        'chat1': 'chat',
        'chat2': 'chat',
        'chien': 'chien',
        'chat3': 'chat',
        'noir1': 'noir',
        'noir2': 'noir',
        'dort': 'dort',
        'assis1': 'assis',
        'assis2': 'assis',
    }
    
    # Determine which nodes are in beam
    beam_nodes = {'START', 'Le', 'Un', 'chat1', 'chat2', 'noir1', 'noir2', 'assis1', 'assis2'}
    
    for node, pos in positions.items():
        if node in node_labels:
            in_beam = node in beam_nodes
            color = COLOR_CURRENT if in_beam else COLOR_NEUTRAL
            alpha = 1.0 if in_beam else 0.4
            
            circle = Circle(pos, 0.15, facecolor=color, alpha=alpha,
                           edgecolor='black', linewidth=1.5)
            ax1.add_patch(circle)
            
            text_color = 'white' if in_beam else 'gray'
            ax1.text(pos[0], pos[1], node_labels[node], ha='center', va='center',
                    fontsize=9, fontweight='bold', color=text_color)
    
    # Add beam size annotation
    ax1.text(4, 5, 'Beam Size = 2\nKeep top 2 paths', fontsize=12, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    # 2. Path scores comparison
    paths = [
        'Le chat noir assis',
        'Un chat noir assis', 
        'Le chien dort',
        'Un chat dort',
        'La chat noir'
    ]
    scores = [0.307, 0.103, 0.072, 0.063, 0.056]
    colors_paths = [COLOR_PREDICT if i < 2 else COLOR_NEUTRAL for i in range(len(paths))]
    alphas = [1.0 if i < 2 else 0.5 for i in range(len(paths))]
    
    bars = ax2.barh(range(len(paths)), scores, color=colors_paths)
    
    # Set individual alpha values
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
    
    ax2.set_yticks(range(len(paths)))
    ax2.set_yticklabels(paths)
    ax2.set_xlabel('Cumulative Probability', fontsize=12)
    ax2.set_title('Path Scores Ranking', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add score labels
    for bar, score in zip(bars, scores):
        ax2.text(score + 0.005, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=10)
    
    # Add beam status labels
    ax2.text(0.35, 4.3, 'Selected by Beam Search', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor=COLOR_PREDICT, alpha=0.3))
    ax2.text(0.35, 2.3, 'Pruned Paths', fontsize=11,
            bbox=dict(boxstyle='round', facecolor=COLOR_NEUTRAL, alpha=0.3))
    
    plt.tight_layout()
    save_figure('week4_beam_search_algorithm.pdf')
    plt.close()

def create_modern_applications_ecosystem():
    """Create modern applications ecosystem map."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Central seq2seq concept
    central_circle = Circle((7, 5), 1.5, facecolor=COLOR_CONTEXT, alpha=0.8,
                           edgecolor='black', linewidth=3)
    ax.add_patch(central_circle)
    ax.text(7, 5, 'Seq2Seq\nArchitecture', ha='center', va='center',
            fontsize=16, fontweight='bold', color='white')
    
    # Application categories
    applications = [
        # (x, y, label, sublabels, color)
        (3, 8.5, 'Translation', ['Google Translate', 'DeepL', 'Microsoft Translator'], COLOR_PREDICT),
        (11, 8.5, 'Summarization', ['News Headlines', 'Email Summaries', 'Document Abstracts'], COLOR_ATTENTION),
        (2, 2, 'Code Generation', ['GitHub Copilot', 'CodeT5', 'Natural → SQL'], COLOR_ENCODER),
        (12, 2, 'Conversational AI', ['ChatGPT Base', 'Customer Service', 'Virtual Assistants'], COLOR_DECODER),
        (1, 5, 'Question Answering', ['Reading Comprehension', 'FAQ Systems', 'Knowledge QA'], COLOR_CURRENT),
        (13, 5, 'Creative Writing', ['Story Generation', 'Poetry Creation', 'Content Writing'], COLOR_BOTTLENECK),
    ]
    
    for x, y, label, sublabels, color in applications:
        # Main application circle
        app_circle = Circle((x, y), 1, facecolor=color, alpha=0.7,
                           edgecolor='black', linewidth=2)
        ax.add_patch(app_circle)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
        
        # Connection to central concept
        arrow = FancyArrowPatch((x, y), (7, 5),
                               connectionstyle="arc3,rad=0.2",
                               arrowstyle='<->', mutation_scale=20,
                               color='gray', linewidth=2, alpha=0.6)
        ax.add_patch(arrow)
        
        # Sublabels
        for i, sublabel in enumerate(sublabels):
            angle = (i - 1) * 0.5  # Spread sublabels around main label
            sub_x = x + 2 * np.cos(angle)
            sub_y = y + 1.5 * np.sin(angle)
            
            # Ensure sublabels stay within bounds
            sub_x = max(0.5, min(13.5, sub_x))
            sub_y = max(0.5, min(9.5, sub_y))
            
            ax.text(sub_x, sub_y, sublabel, ha='center', va='center',
                   fontsize=9, fontweight='normal',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    # Industry statistics
    stats_text = """2024 Industry Impact:
• 1B+ daily translations (Google)
• 80% customer service automation
• 40% developer productivity increase
• $127B market size projection"""
    
    ax.text(7, 1, stats_text, ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Timeline arrow
    ax.arrow(1, 9.5, 11, 0, head_width=0.2, head_length=0.3,
            fc=COLOR_CURRENT, ec=COLOR_CURRENT, linewidth=3)
    ax.text(6.5, 9.8, '2014 → 2024: From Research to Production', 
           ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.title('Seq2Seq Models: Modern Applications Ecosystem (2024)',
             fontsize=18, fontweight='bold', pad=20)
    
    save_figure('week4_modern_applications_ecosystem.pdf')
    plt.close()

def create_performance_evolution_timeline():
    """Create performance evolution timeline with key milestones."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. BLEU scores evolution
    years = np.array([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    bleu_scores = np.array([22, 35, 45, 52, 58, 60, 62, 64, 66, 67, 68])
    
    # Plot the evolution
    ax1.plot(years, bleu_scores, linewidth=3, color=COLOR_PREDICT, alpha=0.8)
    ax1.scatter(years, bleu_scores, s=100, color=COLOR_CURRENT, zorder=5, edgecolor='white', linewidth=2)
    
    # Key milestones
    milestones = [
        (2014, 22, 'Vanilla Seq2Seq'),
        (2015, 35, 'Attention Mechanism'),
        (2016, 45, 'Google NMT'),
        (2017, 52, 'Transformer'),
        (2018, 58, 'BERT Era'),
        (2020, 62, 'GPT-3 Scale'),
        (2024, 68, 'Modern LLMs')
    ]
    
    for year, score, label in milestones:
        ax1.annotate(label, (year, score),
                    textcoords="offset points", xytext=(0, 20), ha='center',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
    
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('BLEU Score', fontsize=12)
    ax1.set_title('Translation Quality Evolution (English-German)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(15, 75)
    
    # Add performance regions
    ax1.axhspan(15, 30, alpha=0.2, color='red', label='Poor Quality')
    ax1.axhspan(30, 50, alpha=0.2, color='orange', label='Acceptable Quality')
    ax1.axhspan(50, 70, alpha=0.2, color='green', label='High Quality')
    ax1.legend(loc='upper left')
    
    # 2. Model complexity and parameters
    models = ['Seq2Seq\n(2014)', 'Attention\n(2015)', 'Transformer\n(2017)', 
              'BERT-Large\n(2018)', 'GPT-3\n(2020)', 'ChatGPT\n(2022)']
    parameters = [10, 15, 65, 340, 175000, 1700000]  # In millions
    training_time = [1, 2, 8, 16, 3500, 25000]  # In hours
    
    x = np.arange(len(models))
    width = 0.35
    
    # Use log scale for parameters
    bars1 = ax2.bar(x - width/2, np.log10(parameters), width, 
                   label='Parameters (log₁₀ millions)', color=COLOR_CONTEXT, alpha=0.8)
    bars2 = ax2.bar(x + width/2, np.log10(training_time), width,
                   label='Training Time (log₁₀ hours)', color=COLOR_ATTENTION, alpha=0.8)
    
    ax2.set_xlabel('Model Architecture', fontsize=12)
    ax2.set_ylabel('Scale (log₁₀)', fontsize=12)
    ax2.set_title('Model Complexity Evolution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add actual values as labels
    for i, (bar1, bar2, param, time) in enumerate(zip(bars1, bars2, parameters, training_time)):
        # Parameters label
        if param < 1000:
            param_text = f'{param}M'
        else:
            param_text = f'{param/1000:.1f}B'
        ax2.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.1,
                param_text, ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Training time label
        if time < 1000:
            time_text = f'{time}h'
        else:
            time_text = f'{time/24:.0f}d'
        ax2.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.1,
                time_text, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    save_figure('week4_performance_evolution_timeline.pdf')
    plt.close()

def main():
    """Generate all Week 4 enhanced figures."""
    print("Generating Week 4 Enhanced Figures with Educational Color Scheme")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs('../figures', exist_ok=True)
    
    # Generate all figures
    create_encoder_decoder_architecture()
    create_information_bottleneck_problem()
    create_attention_mechanism_visualization()
    create_beam_search_visualization()
    create_modern_applications_ecosystem()
    create_performance_evolution_timeline()
    
    print("=" * 60)
    print("[OK] All Week 4 enhanced figures generated successfully!")
    print("[OK] Figures saved in ../figures/ directory")
    print("[OK] Educational color scheme applied consistently")
    print("[OK] High-resolution PDFs ready for LaTeX inclusion")

if __name__ == "__main__":
    main()