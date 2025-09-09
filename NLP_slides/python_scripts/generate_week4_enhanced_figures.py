"""
Week 4: Sequence-to-Sequence Enhanced Visualizations
Generates 6 key figures for understanding seq2seq models and attention
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# Color scheme
COLOR_ENCODER = '#4ECDC4'  # Teal
COLOR_DECODER = '#95E77E'  # Green  
COLOR_ATTENTION = '#FF6B6B'  # Red
COLOR_CONTEXT = '#FFE66D'  # Yellow
COLOR_NEUTRAL = '#E0E0E0'  # Gray

def create_seq2seq_evolution_timeline():
    """Create timeline showing evolution from fixed-length to Transformers."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Timeline data
    years = [1980, 1990, 2000, 2014, 2015, 2017, 2020]
    models = ['N-grams', 'Statistical MT', 'RNNs', 'Seq2Seq', 'Attention', 'Transformers', 'GPT/BERT']
    descriptions = [
        'Fixed context\nCount-based',
        'Phrase-based\nAlignment models',
        'Sequential\nFixed output',
        'Variable I/O\nEncoder-Decoder',
        'Dynamic context\nBahdanau et al.',
        'Attention only\nVaswani et al.',
        'Pre-trained\nTransfer learning'
    ]
    colors = [COLOR_NEUTRAL, COLOR_NEUTRAL, COLOR_ENCODER, COLOR_CONTEXT, 
              COLOR_ATTENTION, COLOR_DECODER, '#9B59B6']
    
    # Draw timeline
    ax.plot(years, [0]*len(years), 'k-', linewidth=2)
    
    for i, (year, model, desc, color) in enumerate(zip(years, models, descriptions, colors)):
        # Timeline marker
        ax.scatter(year, 0, s=200, c=color, zorder=5, edgecolor='black', linewidth=2)
        
        # Model name above
        y_pos = 0.5 if i % 2 == 0 else -0.5
        ax.text(year, y_pos, model, ha='center', fontsize=12, fontweight='bold')
        
        # Description below
        ax.text(year, y_pos - 0.15 if y_pos > 0 else y_pos + 0.15, 
                desc, ha='center', fontsize=9, style='italic', alpha=0.8)
        
        # Connection line
        ax.plot([year, year], [0, y_pos*0.3], 'k--', alpha=0.3)
    
    # Highlight major breakthroughs
    breakthrough_years = [2014, 2015, 2017]
    for year in breakthrough_years:
        ax.add_patch(Circle((year, 0), 0.3, fill=False, edgecolor='red', 
                           linewidth=2, linestyle='--', alpha=0.5))
    
    ax.set_xlim(1975, 2025)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_title('Evolution of Sequence Modeling: From N-grams to Transformers', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Add legend for breakthroughs
    ax.text(2022, 0.8, '● Breakthrough', color='red', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../figures/seq2seq_evolution_timeline.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: seq2seq_evolution_timeline.pdf")

def create_encoder_decoder_flow():
    """Visualize step-by-step seq2seq processing with actual values."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Input sequence
    input_words = ['I', 'love', 'NLP']
    input_embeddings = [
        '[0.3, -0.2, 0.8, 0.1]',
        '[0.5, 0.1, -0.3, 0.7]',
        '[0.2, 0.6, 0.4, -0.1]'
    ]
    
    # Output sequence
    output_words = ["J'", 'aime', 'le', 'NLP', '<END>']
    
    # Draw encoder
    for i, (word, emb) in enumerate(zip(input_words, input_embeddings)):
        x = 1 + i * 2
        y = 6
        
        # Input word
        ax.text(x, y-1, word, ha='center', fontsize=14, fontweight='bold')
        ax.text(x, y-1.5, emb, ha='center', fontsize=8, style='italic')
        
        # Encoder LSTM
        encoder_box = FancyBboxPatch((x-0.6, y), 1.2, 1.5,
                                    boxstyle="round,pad=0.1",
                                    facecolor=COLOR_ENCODER,
                                    edgecolor='black',
                                    linewidth=2)
        ax.add_patch(encoder_box)
        ax.text(x, y+0.75, f'Enc{i+1}', ha='center', fontsize=12)
        
        # Hidden state values
        h_values = [f'h{i+1}: [0.{2+i}, -0.{1+i}, 0.{5-i}, 0.{3+i}]']
        ax.text(x, y+2, h_values[0], ha='center', fontsize=8)
        
        # Arrows between encoders
        if i < len(input_words) - 1:
            ax.arrow(x+0.7, y+0.75, 1.3, 0, head_width=0.2, 
                    head_length=0.2, fc=COLOR_ATTENTION, ec=COLOR_ATTENTION)
    
    # Context vector
    context_x = 8
    context_y = 6.75
    context_box = FancyBboxPatch((context_x-0.8, context_y-0.75), 1.6, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=COLOR_CONTEXT,
                                edgecolor='black',
                                linewidth=3)
    ax.add_patch(context_box)
    ax.text(context_x, context_y, 'Context', ha='center', fontsize=12, fontweight='bold')
    ax.text(context_x, context_y-0.3, '[0.5, 0.1, -0.3, 0.7]', ha='center', fontsize=9)
    
    # Arrow from encoder to context
    ax.arrow(5.7, 6.75, 1.5, 0, head_width=0.2, head_length=0.2, 
            fc='black', ec='black', linewidth=2)
    
    # Draw decoder
    for i, word in enumerate(output_words):
        x = 1 + i * 2
        y = 2
        
        # Decoder LSTM
        decoder_box = FancyBboxPatch((x-0.6, y), 1.2, 1.5,
                                    boxstyle="round,pad=0.1",
                                    facecolor=COLOR_DECODER,
                                    edgecolor='black',
                                    linewidth=2)
        ax.add_patch(decoder_box)
        ax.text(x, y+0.75, f'Dec{i+1}', ha='center', fontsize=12)
        
        # Output word
        ax.text(x, y-1, word, ha='center', fontsize=14, fontweight='bold')
        
        # Context connection (dashed)
        ax.plot([context_x, x], [context_y-0.75, y+1.5], 'b--', alpha=0.3)
        
        # Arrows between decoders
        if i < len(output_words) - 1:
            ax.arrow(x+0.7, y+0.75, 1.3, 0, head_width=0.2, 
                    head_length=0.2, fc=COLOR_ATTENTION, ec=COLOR_ATTENTION)
    
    # Title and labels
    ax.text(3, 9, 'ENCODER', fontsize=16, fontweight='bold', color=COLOR_ENCODER)
    ax.text(3, 0, 'DECODER', fontsize=16, fontweight='bold', color=COLOR_DECODER)
    
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-2, 10)
    ax.axis('off')
    ax.set_title('Encoder-Decoder Flow with Actual Values', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../figures/encoder_decoder_flow.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: encoder_decoder_flow.pdf")

def create_bottleneck_visualization():
    """Show information loss in long sequences."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Short sequence (no bottleneck)
    short_text = "I love you"
    short_words = short_text.split()
    
    for i, word in enumerate(short_words):
        y = 4 - i * 0.8
        ax1.text(0.2, y, word, fontsize=12)
        ax1.arrow(0.4, y, 0.3, 0, head_width=0.1, head_length=0.05, fc=COLOR_ENCODER)
    
    # Context vector (adequate size)
    context1 = FancyBboxPatch((0.8, 1.5), 0.6, 3,
                             boxstyle="round,pad=0.1",
                             facecolor=COLOR_CONTEXT,
                             edgecolor='green',
                             linewidth=3)
    ax1.add_patch(context1)
    ax1.text(1.1, 3, '256d\nvector', ha='center', fontsize=10)
    ax1.text(1.1, 0.5, '✓ No loss', ha='center', fontsize=12, color='green', fontweight='bold')
    
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 5)
    ax1.set_title('Short Sequence: Fits in Context', fontsize=14)
    ax1.axis('off')
    
    # Right: Long sequence (bottleneck)
    long_text = "The International Conference on Machine Learning accepted our paper about neural networks"
    long_words = long_text.split()
    
    for i, word in enumerate(long_words[:8]):  # Show first 8 words
        y = 4.5 - i * 0.5
        ax2.text(0.1, y, word, fontsize=10)
        ax2.arrow(0.35, y, 0.2, 0, head_width=0.08, head_length=0.03, fc=COLOR_ENCODER)
    
    ax2.text(0.2, 0.3, '...', fontsize=14, fontweight='bold')
    
    # Context vector (overflowing)
    context2 = FancyBboxPatch((0.7, 1.5), 0.6, 3,
                             boxstyle="round,pad=0.1",
                             facecolor=COLOR_CONTEXT,
                             edgecolor='red',
                             linewidth=3)
    ax2.add_patch(context2)
    ax2.text(1, 3, '256d\nvector', ha='center', fontsize=10)
    
    # Overflow indicators
    ax2.text(0.6, 4.8, '!', fontsize=20, color='red', fontweight='bold')
    ax2.text(0.6, 0.2, '!', fontsize=20, color='red', fontweight='bold')
    ax2.text(1, 0.5, '✗ Information lost!', ha='center', fontsize=12, color='red', fontweight='bold')
    
    ax2.set_xlim(0, 1.5)
    ax2.set_ylim(0, 5)
    ax2.set_title('Long Sequence: Bottleneck Problem', fontsize=14)
    ax2.axis('off')
    
    plt.suptitle('The Information Bottleneck Problem', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/bottleneck_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: bottleneck_visualization.pdf")

def create_attention_heatmap():
    """Create attention weights heatmap for translation."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Source and target sentences
    source = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    target = ['Le', 'chat', "s'est", 'assis', 'sur', 'le', 'tapis']
    
    # Attention weights (higher values = stronger attention)
    attention_weights = np.array([
        [0.8, 0.1, 0.05, 0.02, 0.03, 0.0],   # Le -> The
        [0.1, 0.85, 0.02, 0.01, 0.01, 0.01], # chat -> cat
        [0.05, 0.1, 0.7, 0.1, 0.03, 0.02],   # s'est -> sat
        [0.02, 0.05, 0.8, 0.08, 0.03, 0.02], # assis -> sat
        [0.01, 0.01, 0.02, 0.9, 0.04, 0.02], # sur -> on
        [0.02, 0.01, 0.01, 0.01, 0.9, 0.05], # le -> the
        [0.01, 0.02, 0.02, 0.02, 0.03, 0.9], # tapis -> mat
    ])
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(source)))
    ax.set_yticks(np.arange(len(target)))
    ax.set_xticklabels(source)
    ax.set_yticklabels(target)
    
    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(target)):
        for j in range(len(source)):
            text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                         ha="center", va="center", color="black" if attention_weights[i, j] < 0.5 else "white")
    
    ax.set_xlabel('Source (English)', fontsize=12)
    ax.set_ylabel('Target (French)', fontsize=12)
    ax.set_title('Attention Weights: English to French Translation', fontsize=14, fontweight='bold')
    
    # Add grid
    ax.set_xticks(np.arange(len(source)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(target)+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('../figures/attention_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: attention_heatmap.pdf")

def create_beam_search_tree():
    """Visualize beam search exploring multiple translation paths."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Tree structure for beam search (beam size = 3)
    # Level 0: Start
    ax.scatter(7, 9, s=300, c=COLOR_CONTEXT, edgecolor='black', linewidth=2, zorder=5)
    ax.text(7, 9, '<START>', ha='center', fontsize=10, fontweight='bold')
    
    # Level 1: First word choices
    level1_words = ['Le', 'La', 'Un']
    level1_scores = [0.6, 0.3, 0.1]
    level1_x = [4, 7, 10]
    
    for x, word, score in zip(level1_x, level1_words, level1_scores):
        # Draw node
        color = COLOR_DECODER if score > 0.3 else COLOR_NEUTRAL
        ax.scatter(x, 7, s=250, c=color, edgecolor='black', linewidth=2, zorder=5)
        ax.text(x, 7, word, ha='center', fontsize=10)
        ax.text(x, 6.5, f'{score:.1f}', ha='center', fontsize=8, style='italic')
        # Draw edge
        ax.plot([7, x], [9, 7], 'k-', alpha=0.3, linewidth=2)
    
    # Level 2: Second word choices (for top 3 from level 1)
    level2_data = [
        (4, ['chat', 'chien', 'homme'], [0.8, 0.15, 0.05]),
        (7, ['chat', 'table', 'maison'], [0.4, 0.4, 0.2]),
        (10, ['chat', 'grand', 'petit'], [0.3, 0.4, 0.3])
    ]
    
    for parent_x, words, scores in level2_data:
        for i, (word, score) in enumerate(zip(words, scores)):
            x = parent_x + (i-1) * 1.2
            y = 5
            
            # Calculate cumulative score
            parent_score = level1_scores[level1_x.index(parent_x)]
            cum_score = parent_score * score
            
            # Draw node
            color = COLOR_DECODER if cum_score > 0.2 else COLOR_NEUTRAL
            ax.scatter(x, y, s=200, c=color, edgecolor='black', linewidth=1.5, zorder=5)
            ax.text(x, y, word, ha='center', fontsize=9)
            ax.text(x, y-0.4, f'{cum_score:.2f}', ha='center', fontsize=7, style='italic')
            
            # Draw edge
            ax.plot([parent_x, x], [7, y], 'k-', alpha=0.2, linewidth=1.5)
    
    # Level 3: Continue for top beams
    best_paths = [
        (2.8, 5, "s'est", 0.48),
        (4, 5, 'est', 0.32),
        (7, 5, 'est', 0.16)
    ]
    
    for x, parent_y, word, score in best_paths:
        y = 3
        color = COLOR_DECODER if score > 0.3 else COLOR_NEUTRAL
        ax.scatter(x, y, s=180, c=color, edgecolor='black', linewidth=1.5, zorder=5)
        ax.text(x, y, word, ha='center', fontsize=9)
        ax.text(x, y-0.4, f'{score:.2f}', ha='center', fontsize=7, style='italic')
        
        # Find parent
        if x < 5:
            parent_x = 2.8
        elif x < 8:
            parent_x = 4
        else:
            parent_x = 7
        ax.plot([parent_x, x], [parent_y, y], 'k-', alpha=0.2, linewidth=1.5)
    
    # Highlight best path
    best_path_x = [7, 4, 2.8, 2.8]
    best_path_y = [9, 7, 5, 3]
    ax.plot(best_path_x, best_path_y, 'r-', linewidth=3, alpha=0.5, zorder=1)
    
    # Add legend
    ax.text(12, 8, 'Beam Search (k=3)', fontsize=14, fontweight='bold')
    ax.text(12, 7.5, '● High score', color=COLOR_DECODER, fontsize=10)
    ax.text(12, 7, '● Low score', color=COLOR_NEUTRAL, fontsize=10)
    ax.text(12, 6.5, '— Best path', color='red', fontsize=10)
    
    ax.set_xlim(0, 14)
    ax.set_ylim(2, 10)
    ax.axis('off')
    ax.set_title('Beam Search: Exploring Multiple Translation Paths', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../figures/beam_search_tree.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: beam_search_tree.pdf")

def create_applications_2024():
    """Create grid showing modern seq2seq applications."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    
    applications = [
        ('Translation', 'Google Translate\nDeepL', COLOR_ENCODER, 
         '100+ languages\nReal-time\nOffline mode'),
        ('Chatbots', 'Customer Service\nChatGPT', COLOR_DECODER,
         'Context aware\nMulti-turn\nPersonalized'),
        ('Code Gen', 'GitHub Copilot\nTabnine', COLOR_ATTENTION,
         'Comment → Code\nBug → Fix\nRefactoring'),
        ('Speech', 'Whisper\nSiri', COLOR_CONTEXT,
         'Audio → Text\nMultilingual\nOn-device'),
        ('Summarization', 'News → Headlines\nDocs → Abstract', '#9B59B6',
         'Extractive\nAbstractive\nMulti-document'),
        ('Vision', 'Image Captioning\nVideo Description', '#E74C3C',
         'CNN encoder\nLSTM decoder\nAttention over regions')
    ]
    
    for ax, (title, examples, color, features) in zip(axes.flat, applications):
        # Main box
        main_box = FancyBboxPatch((0.1, 0.3), 0.8, 0.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor=color,
                                 alpha=0.3,
                                 edgecolor=color,
                                 linewidth=3)
        ax.add_patch(main_box)
        
        # Title
        ax.text(0.5, 0.85, title, ha='center', fontsize=14, fontweight='bold')
        
        # Examples
        ax.text(0.5, 0.6, examples, ha='center', fontsize=10, style='italic')
        
        # Features
        ax.text(0.5, 0.35, features, ha='center', fontsize=8, alpha=0.8)
        
        # Icons (simplified)
        if 'Translation' in title:
            ax.text(0.5, 0.1, '🌍', ha='center', fontsize=20)
        elif 'Chatbots' in title:
            ax.text(0.5, 0.1, '💬', ha='center', fontsize=20)
        elif 'Code' in title:
            ax.text(0.5, 0.1, '💻', ha='center', fontsize=20)
        elif 'Speech' in title:
            ax.text(0.5, 0.1, '🎤', ha='center', fontsize=20)
        elif 'Summarization' in title:
            ax.text(0.5, 0.1, '📄', ha='center', fontsize=20)
        else:
            ax.text(0.5, 0.1, '📷', ha='center', fontsize=20)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.suptitle('Seq2Seq Applications in 2024', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/applications_2024.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: applications_2024.pdf")

def main():
    """Generate all Week 4 figures."""
    print("Generating Week 4 Seq2Seq visualizations...")
    
    create_seq2seq_evolution_timeline()
    create_encoder_decoder_flow()
    create_bottleneck_visualization()
    create_attention_heatmap()
    create_beam_search_tree()
    create_applications_2024()
    
    print("\nAll Week 4 figures generated successfully!")
    print("Files saved in ../figures/")

if __name__ == "__main__":
    main()