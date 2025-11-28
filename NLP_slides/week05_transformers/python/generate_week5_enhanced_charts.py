"""
Generate enhanced visualizations for Week 5: Transformers
Includes pedagogical improvements and rich visualizations
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrow, ConnectionPatch
import seaborn as sns
import os
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# Educational color scheme
COLOR_CURRENT = '#FF6B6B'  # Red - current position/focus
COLOR_CONTEXT = '#4ECDC4'  # Teal - context/surrounding  
COLOR_PREDICT = '#95E77E'  # Green - predictions/output
COLOR_NEUTRAL = '#E0E0E0'  # Gray - neutral elements
COLOR_ATTENTION = '#FFE66D'  # Yellow - attention weights
COLOR_QUERY = '#A8DADC'  # Light blue - queries
COLOR_KEY = '#F1FAEE'  # Off white - keys
COLOR_VALUE = '#457B9D'  # Dark blue - values

def plot_multihead_attention_patterns():
    """Show how different attention heads focus on different patterns"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    sentence = ['The', 'student', 'who', 'studied', 'hard', 'passed', 'the', 'exam']
    n_words = len(sentence)
    
    # Define different attention patterns for 8 heads
    attention_patterns = [
        {
            'name': 'Head 1: Subject-Verb',
            'pattern': 'grammatical',
            'highlights': [(1, 5), (1, 3)]  # student->passed, student->studied
        },
        {
            'name': 'Head 2: Determiner',
            'pattern': 'determiner',
            'highlights': [(0, 1), (6, 7)]  # The->student, the->exam
        },
        {
            'name': 'Head 3: Relative Clause',
            'pattern': 'relative',
            'highlights': [(2, 1), (2, 3)]  # who->student, who->studied
        },
        {
            'name': 'Head 4: Adverb Modifier',
            'pattern': 'adverb',
            'highlights': [(4, 3), (4, 5)]  # hard->studied, hard->passed
        },
        {
            'name': 'Head 5: Object Relations',
            'pattern': 'object',
            'highlights': [(5, 7), (3, 1)]  # passed->exam, studied->student
        },
        {
            'name': 'Head 6: Sequential',
            'pattern': 'sequential',
            'highlights': [(i, i+1) for i in range(n_words-1)]
        },
        {
            'name': 'Head 7: Long-range',
            'pattern': 'long',
            'highlights': [(0, 7), (1, 7)]  # The->exam, student->exam
        },
        {
            'name': 'Head 8: Self-attention',
            'pattern': 'self',
            'highlights': [(i, i) for i in range(n_words)]
        }
    ]
    
    for idx, head_info in enumerate(attention_patterns):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        
        # Create attention matrix
        attention = np.zeros((n_words, n_words))
        for src, tgt in head_info['highlights']:
            attention[src, tgt] = np.random.uniform(0.6, 1.0)
        
        # Add some background noise
        attention += np.random.uniform(0, 0.2, (n_words, n_words))
        
        # Normalize rows
        attention = attention / attention.sum(axis=1, keepdims=True)
        
        # Plot heatmap
        im = ax.imshow(attention, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # Labels
        ax.set_xticks(range(n_words))
        ax.set_yticks(range(n_words))
        ax.set_xticklabels(sentence, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(sentence, fontsize=8)
        ax.set_title(head_info['name'], fontsize=10, fontweight='bold')
        
        # Add grid
        ax.set_xticks(np.arange(n_words) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_words) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Add colorbar
    fig.colorbar(im, ax=fig.get_axes(), orientation='horizontal', 
                 fraction=0.02, pad=0.1, label='Attention Weight')
    
    plt.suptitle('Multi-Head Attention: 8 Different Perspectives on the Same Sentence', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('../figures/multihead_attention_patterns.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_positional_encoding_viz():
    """Visualize positional encoding patterns"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Parameters
    d_model = 512
    max_len = 100
    
    # Generate positional encodings
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    # Plot 1: Full encoding matrix
    im1 = ax1.imshow(pe[:50, :128].T, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xlabel('Position in Sequence')
    ax1.set_ylabel('Encoding Dimension')
    ax1.set_title('Positional Encoding Pattern\n(First 50 positions, 128 dimensions)')
    plt.colorbar(im1, ax=ax1)
    
    # Plot 2: Individual dimensions
    positions_to_show = [0, 1, 2, 5, 10, 20]
    colors = plt.cm.viridis(np.linspace(0, 1, len(positions_to_show)))
    
    for i, pos in enumerate(positions_to_show):
        ax2.plot(pe[pos, :64], label=f'Position {pos}', color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Encoding Value')
    ax2.set_title('Encoding Patterns for Different Positions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distance preservation
    # Show that relative positions have consistent patterns
    pos1, pos2 = 10, 15
    diff = pe[pos2] - pe[pos1]
    
    ax3.bar(range(64), diff[:64], color=COLOR_ATTENTION, alpha=0.7)
    ax3.set_xlabel('Dimension')
    ax3.set_ylabel('Difference in Encoding')
    ax3.set_title(f'Encoding Difference: Position {pos2} - Position {pos1}')
    ax3.grid(True, alpha=0.3)
    
    # Add text explanation
    ax3.text(0.5, -0.15, 
             'Consistent patterns for relative positions enable the model to understand distance',
             transform=ax3.transAxes, ha='center', fontsize=10, style='italic')
    
    plt.suptitle('Positional Encoding: Giving Words Their GPS Coordinates', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/positional_encoding_viz.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_transformer_architecture_detailed():
    """Create visual transformer architecture diagram with minimal text"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Visual representation using shapes and colors
    # Central column for main transformer blocks
    block_width = 0.15
    block_height = 0.08
    x_center = 0.5
    
    # Define visual layers with icons/patterns
    layers_visual = [
        (0.1, 'embeddings', COLOR_NEUTRAL),
        (0.2, 'position', COLOR_ATTENTION),
        (0.35, 'attention', COLOR_CONTEXT),
        (0.45, 'norm', COLOR_PREDICT),
        (0.55, 'ffn', COLOR_CURRENT),
        (0.65, 'norm', COLOR_PREDICT),
        (0.8, 'output', COLOR_VALUE)
    ]
    
    # Draw transformer blocks as visual elements
    for y, layer_type, color in layers_visual:
        if layer_type == 'embeddings':
            # Word embeddings - matrix visualization
            for i in range(5):
                for j in range(3):
                    rect = Rectangle((x_center - 0.075 + i*0.03, y - 0.03 + j*0.02), 
                                   0.025, 0.015, facecolor=color, 
                                   edgecolor='black', alpha=0.6)
                    ax.add_patch(rect)
            
        elif layer_type == 'position':
            # Sinusoidal waves for positional encoding
            x_wave = np.linspace(x_center - 0.075, x_center + 0.075, 100)
            y_wave1 = y + 0.02 * np.sin(20 * (x_wave - x_center + 0.075))
            y_wave2 = y + 0.02 * np.cos(15 * (x_wave - x_center + 0.075))
            ax.plot(x_wave, y_wave1, color=color, linewidth=2)
            ax.plot(x_wave, y_wave2, color=color, linewidth=2, linestyle='--')
            
        elif layer_type == 'attention':
            # Multi-head attention - 8 circles representing heads
            for i in range(8):
                angle = i * 2 * np.pi / 8
                cx = x_center + 0.05 * np.cos(angle)
                cy = y + 0.05 * np.sin(angle)
                circle = Circle((cx, cy), 0.015, facecolor=color, 
                              edgecolor='black', alpha=0.7)
                ax.add_patch(circle)
                # Connect to center
                ax.plot([x_center, cx], [y, cy], 'k-', alpha=0.3, linewidth=1)
            
        elif layer_type == 'norm':
            # Layer normalization - bell curve shape
            x_norm = np.linspace(x_center - 0.075, x_center + 0.075, 100)
            y_norm = y + 0.03 * np.exp(-((x_norm - x_center)**2) / (2 * 0.02**2))
            ax.fill_between(x_norm, y - 0.02, y_norm, color=color, alpha=0.6, 
                          edgecolor='black', linewidth=1)
            
        elif layer_type == 'ffn':
            # Feed-forward network - two-layer neural net visualization
            # First layer
            for i in range(4):
                cx1 = x_center - 0.05 + i * 0.033
                circle1 = Circle((cx1, y - 0.02), 0.01, facecolor=color, 
                               edgecolor='black', alpha=0.7)
                ax.add_patch(circle1)
            # Second layer (expanded)
            for i in range(6):
                cx2 = x_center - 0.075 + i * 0.03
                circle2 = Circle((cx2, y + 0.02), 0.01, facecolor=color, 
                               edgecolor='black', alpha=0.7)
                ax.add_patch(circle2)
            # Connections
            for i in range(4):
                for j in range(6):
                    ax.plot([x_center - 0.05 + i * 0.033, x_center - 0.075 + j * 0.03],
                           [y - 0.02, y + 0.02], 'k-', alpha=0.1, linewidth=0.5)
                    
        elif layer_type == 'output':
            # Output probabilities - bar chart style
            for i in range(8):
                height = np.random.uniform(0.01, 0.04)
                rect = Rectangle((x_center - 0.08 + i*0.02, y), 0.015, height,
                               facecolor=color, edgecolor='black', alpha=0.7)
                ax.add_patch(rect)
    
    # Draw main flow arrows
    arrow_positions = [0.15, 0.275, 0.4, 0.5, 0.6, 0.725]
    for i, y_pos in enumerate(arrow_positions):
        arrow = FancyArrow(x_center, y_pos, 0, 0.05,
                         width=0.015, head_width=0.03, head_length=0.02,
                         fc='gray', ec='gray', alpha=0.4)
        ax.add_patch(arrow)
    
    # Residual connections as curved lines
    # Around attention block
    style1 = "arc3,rad=.4"
    residual1 = mpatches.FancyArrowPatch((x_center - 0.12, 0.3), 
                                        (x_center - 0.12, 0.45),
                                        connectionstyle=style1,
                                        arrowstyle='->,head_width=0.3,head_length=0.2',
                                        color='red', alpha=0.5, linewidth=2,
                                        linestyle='--')
    ax.add_patch(residual1)
    
    # Around FFN block  
    residual2 = mpatches.FancyArrowPatch((x_center - 0.12, 0.5), 
                                        (x_center - 0.12, 0.65),
                                        connectionstyle=style1,
                                        arrowstyle='->,head_width=0.3,head_length=0.2',
                                        color='red', alpha=0.5, linewidth=2,
                                        linestyle='--')
    ax.add_patch(residual2)
    
    # Stack indicator - visual representation of 6 layers
    for i in range(6):
        alpha = 0.3 + i * 0.1
        rect = Rectangle((0.85 - i*0.01, 0.35 - i*0.01), 0.1, 0.35,
                       facecolor='none', edgecolor='purple', 
                       linewidth=2, alpha=alpha, linestyle=':')
        ax.add_patch(rect)
    ax.text(0.9, 0.52, '×6', fontsize=20, fontweight='bold', color='purple')
    
    # Minimal labels - only essential ones
    ax.text(x_center, 0.05, 'Words', ha='center', fontsize=10)
    ax.text(x_center, 0.85, 'Predictions', ha='center', fontsize=10)
    ax.text(x_center - 0.16, 0.375, '⟲', fontsize=16, color='red', fontweight='bold')
    ax.text(x_center - 0.16, 0.575, '⟲', fontsize=16, color='red', fontweight='bold')
    
    # Legend showing what shapes mean
    legend_elements = [
        mpatches.Patch(color=COLOR_CONTEXT, label='Attention'),
        mpatches.Patch(color=COLOR_CURRENT, label='Neural Net'),
        mpatches.Patch(color=COLOR_PREDICT, label='Normalize'),
        mpatches.Patch(color='red', alpha=0.5, label='Skip Connection')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    ax.set_xlim(0.2, 1)
    ax.set_ylim(0, 0.9)
    ax.axis('off')
    
    ax.set_title('Transformer Architecture: Visual Flow', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('../figures/transformer_architecture_detailed.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_transformer_vs_rnn_performance():
    """Comprehensive performance comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Training speed vs sequence length
    ax = axes[0, 0]
    seq_lengths = np.array([10, 50, 100, 200, 500, 1000, 2000])
    rnn_time = seq_lengths * 0.1
    transformer_time = np.ones_like(seq_lengths) * 0.5 + (seq_lengths/1000)**2 * 0.2
    
    ax.plot(seq_lengths, rnn_time, 'o-', label='RNN', color='red', linewidth=2, markersize=8)
    ax.plot(seq_lengths, transformer_time, 's-', label='Transformer', color='blue', linewidth=2, markersize=8)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Training Time (relative)')
    ax.set_title('Training Speed Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 2. GPU utilization
    ax = axes[0, 1]
    models = ['RNN', 'LSTM', 'GRU', 'Transformer']
    gpu_util = [15, 20, 18, 95]
    colors = ['red', 'orange', 'yellow', 'green']
    
    bars = ax.bar(models, gpu_util, color=colors, alpha=0.7)
    ax.set_ylabel('GPU Utilization (%)')
    ax.set_title('Hardware Efficiency')
    ax.set_ylim(0, 100)
    
    for bar, val in zip(bars, gpu_util):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val}%',
               ha='center', fontweight='bold')
    
    # 3. Model quality (BLEU scores)
    ax = axes[0, 2]
    years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    rnn_bleu = [20, 22, 24, 26, 27, 27.5, 28, 28, 28, 28, 28]
    transformer_bleu = [0, 0, 0, 28.4, 31, 33, 35, 37, 39, 41, 43]
    
    ax.plot(years, rnn_bleu, 'o-', label='Best RNN', color='red', linewidth=2)
    ax.plot(years, transformer_bleu, 's-', label='Best Transformer', color='blue', linewidth=2)
    ax.axvline(x=2017, color='gray', linestyle='--', alpha=0.5)
    ax.text(2017, 15, 'Transformer\nInvented', ha='center', fontsize=9)
    ax.set_xlabel('Year')
    ax.set_ylabel('BLEU Score')
    ax.set_title('Translation Quality Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Memory consumption
    ax = axes[1, 0]
    seq_lengths = np.array([100, 500, 1000, 2000, 5000, 10000])
    rnn_memory = seq_lengths * 0.001  # Linear
    transformer_memory = (seq_lengths ** 2) * 0.000001  # Quadratic
    
    ax.plot(seq_lengths, rnn_memory, 'o-', label='RNN', color='red', linewidth=2)
    ax.plot(seq_lengths, transformer_memory, 's-', label='Transformer', color='blue', linewidth=2)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Memory (GB)')
    ax.set_title('Memory Consumption')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 5. Long-range dependency
    ax = axes[1, 1]
    distances = np.arange(1, 101)
    rnn_strength = np.exp(-distances / 20)
    transformer_strength = np.ones_like(distances) * 0.8
    
    ax.plot(distances, rnn_strength, label='RNN', color='red', linewidth=2)
    ax.plot(distances, transformer_strength, label='Transformer', color='blue', linewidth=2)
    ax.set_xlabel('Distance Between Words')
    ax.set_ylabel('Dependency Strength')
    ax.set_title('Long-Range Dependencies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Parallelization capability
    ax = axes[1, 2]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    rnn_throughput = np.array(batch_sizes) * 1.1  # Slight improvement
    transformer_throughput = np.array(batch_sizes) * 10  # Linear scaling
    
    ax.plot(batch_sizes, rnn_throughput, 'o-', label='RNN', color='red', linewidth=2)
    ax.plot(batch_sizes, transformer_throughput, 's-', label='Transformer', color='blue', linewidth=2)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (sequences/sec)')
    ax.set_title('Parallelization Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.suptitle('Transformers vs RNNs: Comprehensive Performance Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/transformer_vs_rnn_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_transformer_applications_timeline():
    """Timeline of transformer applications and impact"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Timeline data
    events = [
        (2017.5, 'Transformer Paper', 'research', 'Attention Is All You Need'),
        (2018.0, 'GPT-1', 'model', '117M parameters'),
        (2018.5, 'BERT', 'model', '340M parameters'),
        (2019.0, 'GPT-2', 'model', '1.5B parameters'),
        (2019.5, 'RoBERTa', 'model', 'Optimized BERT'),
        (2020.0, 'T5', 'model', 'Text-to-Text'),
        (2020.5, 'GPT-3', 'model', '175B parameters'),
        (2021.0, 'DALL-E', 'application', 'Text to Image'),
        (2021.5, 'Codex', 'application', 'GitHub Copilot'),
        (2022.0, 'ChatGPT', 'application', 'Conversational AI'),
        (2022.5, 'Stable Diffusion', 'application', 'Open Image Gen'),
        (2023.0, 'GPT-4', 'model', '1.76T parameters'),
        (2023.5, 'Claude', 'application', 'Constitutional AI'),
        (2024.0, 'Gemini', 'model', 'Multimodal'),
        (2024.5, 'GPT-4o', 'model', 'Omnimodal')
    ]
    
    # Color mapping
    type_colors = {
        'research': '#FF6B6B',
        'model': '#4ECDC4',
        'application': '#95E77E'
    }
    
    # Plot timeline
    for i, (year, name, event_type, description) in enumerate(events):
        y_pos = 0.5 + (i % 3 - 1) * 0.2  # Stagger vertically
        
        # Event marker
        ax.scatter(year, y_pos, s=200, c=type_colors[event_type], 
                  alpha=0.7, edgecolors='black', linewidth=2, zorder=3)
        
        # Event label
        ax.annotate(f'{name}\n{description}', xy=(year, y_pos),
                   xytext=(0, 20), textcoords='offset points',
                   ha='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor=type_colors[event_type], alpha=0.3))
        
        # Connect to timeline
        ax.plot([year, year], [0.1, y_pos-0.05], 'k--', alpha=0.3, zorder=1)
    
    # Main timeline
    ax.plot([2017, 2025], [0.1, 0.1], 'k-', linewidth=3, zorder=2)
    
    # Year markers
    for year in range(2017, 2025):
        ax.plot([year, year], [0.08, 0.12], 'k-', linewidth=2)
        ax.text(year, 0.05, str(year), ha='center', fontsize=10)
    
    # Legend
    for event_type, color in type_colors.items():
        ax.scatter([], [], c=color, s=100, label=event_type.capitalize(), 
                  alpha=0.7, edgecolors='black', linewidth=2)
    ax.legend(loc='upper left', fontsize=10)
    
    # Impact annotations
    ax.text(2020, 0.9, '↑ Pre-training Era', fontsize=12, fontweight='bold', color='blue')
    ax.text(2022.5, 0.9, '↑ Consumer AI Era', fontsize=12, fontweight='bold', color='green')
    
    # Formatting
    ax.set_xlim(2016.5, 2025.5)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('The Transformer Revolution: From Research to Ubiquity', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('../figures/transformer_applications_timeline.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_attention_mechanism_breakdown():
    """Step-by-step breakdown of attention mechanism"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Example: "The cat sat"
    words = ['The', 'cat', 'sat']
    d_k = 4  # Small dimension for visualization
    
    # Step 1: Input embeddings
    ax1 = fig.add_subplot(gs[0, 0])
    embeddings = np.random.randn(3, d_k)
    im1 = ax1.imshow(embeddings, cmap='coolwarm', aspect='auto')
    ax1.set_yticks(range(3))
    ax1.set_yticklabels(words)
    ax1.set_xlabel('Embedding Dims')
    ax1.set_title('Step 1: Input Embeddings')
    plt.colorbar(im1, ax=ax1)
    
    # Step 2: Create Q, K, V
    ax2 = fig.add_subplot(gs[0, 1])
    Q = embeddings @ np.random.randn(d_k, d_k)
    K = embeddings @ np.random.randn(d_k, d_k)
    V = embeddings @ np.random.randn(d_k, d_k)
    
    # Show Q
    im2 = ax2.imshow(Q, cmap='Blues', aspect='auto')
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(words)
    ax2.set_xlabel('Query Dims')
    ax2.set_title('Step 2: Query Matrix (Q)')
    plt.colorbar(im2, ax=ax2)
    
    # Step 3: Compute QK^T
    ax3 = fig.add_subplot(gs[0, 2])
    attention_scores = Q @ K.T
    im3 = ax3.imshow(attention_scores, cmap='YlOrRd', aspect='auto')
    ax3.set_xticks(range(3))
    ax3.set_yticks(range(3))
    ax3.set_xticklabels(words)
    ax3.set_yticklabels(words)
    ax3.set_title('Step 3: QK^T (Raw Scores)')
    plt.colorbar(im3, ax=ax3)
    
    # Step 4: Scale
    ax4 = fig.add_subplot(gs[1, 0])
    scaled_scores = attention_scores / np.sqrt(d_k)
    im4 = ax4.imshow(scaled_scores, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(3))
    ax4.set_yticks(range(3))
    ax4.set_xticklabels(words)
    ax4.set_yticklabels(words)
    ax4.set_title(f'Step 4: Scale by √{d_k}')
    plt.colorbar(im4, ax=ax4)
    
    # Step 5: Softmax
    ax5 = fig.add_subplot(gs[1, 1])
    attention_weights = np.exp(scaled_scores) / np.exp(scaled_scores).sum(axis=-1, keepdims=True)
    im5 = ax5.imshow(attention_weights, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax5.set_xticks(range(3))
    ax5.set_yticks(range(3))
    ax5.set_xticklabels(words)
    ax5.set_yticklabels(words)
    ax5.set_title('Step 5: Softmax (Probabilities)')
    plt.colorbar(im5, ax=ax5)
    
    # Step 6: Multiply by V
    ax6 = fig.add_subplot(gs[1, 2])
    output = attention_weights @ V
    im6 = ax6.imshow(output, cmap='Greens', aspect='auto')
    ax6.set_yticks(range(3))
    ax6.set_yticklabels(words)
    ax6.set_xlabel('Output Dims')
    ax6.set_title('Step 6: Attention × Values')
    plt.colorbar(im6, ax=ax6)
    
    # Mathematical formula
    ax7 = fig.add_subplot(gs[2, :])
    ax7.text(0.5, 0.7, r'$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$',
            fontsize=20, ha='center', transform=ax7.transAxes)
    ax7.text(0.5, 0.3, 'Where each word decides what to attend to based on Query-Key similarity',
            fontsize=14, ha='center', style='italic', transform=ax7.transAxes)
    ax7.axis('off')
    
    plt.suptitle('Self-Attention Mechanism: Step-by-Step Computation', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/attention_mechanism_breakdown.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_scaling_comparison():
    """Show the scaling of transformer models over time"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Model data (name, year, parameters in millions)
    models = [
        ('Transformer', 2017, 65),
        ('GPT-1', 2018, 117),
        ('BERT-base', 2018, 110),
        ('BERT-large', 2018, 340),
        ('GPT-2', 2019, 1500),
        ('T5', 2020, 11000),
        ('GPT-3', 2020, 175000),
        ('PaLM', 2022, 540000),
        ('GPT-4', 2023, 1760000),
    ]
    
    names = [m[0] for m in models]
    years = [m[1] for m in models]
    params = [m[2] for m in models]
    
    # Plot 1: Parameters over time (log scale)
    ax1.scatter(years, params, s=100, c=years, cmap='viridis', 
               edgecolors='black', linewidth=2, alpha=0.7)
    
    for name, year, param in models:
        if param > 100000:  # Only label large models
            ax1.annotate(name, (year, param), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Parameters (log scale)', fontsize=12)
    ax1.set_title('Exponential Growth in Model Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(years, np.log10(params), 1)
    p = np.poly1d(z)
    years_extended = np.linspace(2017, 2025, 100)
    ax1.plot(years_extended, 10**p(years_extended), 'r--', alpha=0.5, 
            label='Exponential Trend')
    ax1.legend()
    
    # Plot 2: Relative sizes (bubble chart)
    # Normalize to GPT-1 = 1
    sizes_normalized = [p/117 for p in params]
    
    # Create bubble chart
    for i, (name, size) in enumerate(zip(names, sizes_normalized)):
        circle = plt.Circle((i % 3, i // 3), np.sqrt(size)/20, 
                           color=plt.cm.plasma(i/len(names)), 
                           alpha=0.6, edgecolor='black', linewidth=2)
        ax2.add_patch(circle)
        ax2.text(i % 3, i // 3, name, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white' if size > 100 else 'black')
    
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 3)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Relative Model Sizes (GPT-1 = 1)', fontsize=14, fontweight='bold')
    
    # Add size reference
    ax2.text(1, -0.3, 'Circle area proportional to parameter count', 
            ha='center', fontsize=10, style='italic')
    
    plt.suptitle('The Race to Scale: Transformer Model Evolution', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/model_scaling_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_layer_by_layer_transformation():
    """Show how representations evolve through transformer layers"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Example sentence
    words = ['The', 'cat', 'sat', 'on', 'mat']
    n_words = len(words)
    
    # Simulate representations getting more abstract through layers
    layer_names = ['Input', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 6']
    
    for idx, (ax, layer_name) in enumerate(zip(axes.flat, layer_names)):
        # Create increasingly abstract representations
        if idx == 0:
            # Input layer - word embeddings
            data = np.random.randn(n_words, 8) * 0.5
            for i in range(n_words):
                data[i, i % 8] = 2  # Word-specific features
        else:
            # Progressive transformation
            complexity = idx / 5
            data = np.random.randn(n_words, 8) * (1 + complexity)
            # Add layer-specific patterns
            if idx == 1:  # Early layer - local patterns
                data[0:2, 0:2] = 3  # Subject group
            elif idx == 2:  # Mid layer - syntactic
                data[1, :] = np.sin(np.linspace(0, 2*np.pi, 8)) * 3  # Noun pattern
                data[2, :] = np.cos(np.linspace(0, 2*np.pi, 8)) * 3  # Verb pattern
            elif idx >= 3:  # Late layers - semantic
                # Create semantic clusters
                data[0:2, 3:5] = 4  # Agent semantic
                data[2, 5:7] = 4  # Action semantic
                data[3:5, 1:3] = 3  # Location semantic
        
        # Visualize as heatmap
        im = ax.imshow(data, cmap='coolwarm', aspect='auto', vmin=-4, vmax=4)
        
        # Labels
        ax.set_yticks(range(n_words))
        ax.set_yticklabels(words)
        ax.set_xticks(range(8))
        ax.set_xticklabels([f'D{i}' for i in range(8)])
        ax.set_title(f'{layer_name}', fontsize=12, fontweight='bold')
        
        # Add attention connections for later layers
        if idx > 0:
            # Show increasing attention range
            attention_range = min(idx, 3)
            for i in range(n_words):
                for j in range(max(0, i-attention_range), min(n_words, i+attention_range+1)):
                    if i != j:
                        alpha = 0.1 * (1 + idx/5)
                        ax.plot([8.5, 8.5], [i, j], 'k-', alpha=alpha, linewidth=0.5)
        
        # Add interpretation
        if idx == 0:
            ax.text(9, 2, 'Raw\nwords', fontsize=9, va='center')
        elif idx == 1:
            ax.text(9, 2, 'Local\ncontext', fontsize=9, va='center')
        elif idx == 2:
            ax.text(9, 2, 'Syntax', fontsize=9, va='center')
        elif idx >= 3:
            ax.text(9, 2, 'Semantics', fontsize=9, va='center')
    
    # Add colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal',
                 fraction=0.02, pad=0.1, label='Feature Activation')
    
    plt.suptitle('Layer-by-Layer Transformation: From Words to Understanding', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/layer_by_layer_transformation.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_step_by_step_building():
    """Progressive building from single head to full transformer"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Step 1: Single attention head
    ax1 = fig.add_subplot(gs[0, 0])
    # Simple attention visualization
    words = ['The', 'cat', 'sat']
    n = len(words)
    attention = np.random.rand(n, n)
    attention = attention / attention.sum(axis=1, keepdims=True)
    im = ax1.imshow(attention, cmap='Blues', vmin=0, vmax=1)
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(words, fontsize=8)
    ax1.set_yticklabels(words, fontsize=8)
    ax1.set_title('Step 1: Single\nAttention Head', fontsize=10, fontweight='bold')
    
    # Step 2: Multi-head (4 heads)
    ax2 = fig.add_subplot(gs[0, 1])
    cmap_blues = plt.cm.Blues
    for i in range(2):
        for j in range(2):
            # Mini attention matrices
            x_start = j * 0.45
            y_start = i * 0.45
            rect = Rectangle((x_start, y_start), 0.4, 0.4, 
                           facecolor=cmap_blues((i*2+j+1)/5), alpha=0.6,
                           edgecolor='black', linewidth=1, transform=ax2.transAxes)
            ax2.add_patch(rect)
            ax2.text(x_start + 0.2, y_start + 0.2, f'H{i*2+j+1}',
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=9, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Step 2: Multi-Head\n(4 heads)', fontsize=10, fontweight='bold')
    
    # Step 3: Add FFN
    ax3 = fig.add_subplot(gs[0, 2])
    # Attention block
    rect1 = FancyBboxPatch((0.2, 0.6), 0.6, 0.2, boxstyle="round,pad=0.02",
                          facecolor='lightblue', edgecolor='black', linewidth=2)
    ax3.add_patch(rect1)
    ax3.text(0.5, 0.7, 'Multi-Head\nAttention', ha='center', va='center', fontsize=9)
    # Arrow
    arrow1 = FancyArrow(0.5, 0.55, 0, -0.1, width=0.05, head_width=0.1, 
                       head_length=0.05, fc='gray', ec='gray')
    ax3.add_patch(arrow1)
    # FFN block
    rect2 = FancyBboxPatch((0.2, 0.2), 0.6, 0.2, boxstyle="round,pad=0.02",
                          facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax3.add_patch(rect2)
    ax3.text(0.5, 0.3, 'Feed\nForward', ha='center', va='center', fontsize=9)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Step 3: Add\nFeed Forward', fontsize=10, fontweight='bold')
    
    # Step 4: Add residuals
    ax4 = fig.add_subplot(gs[0, 3])
    # Blocks
    rect1 = FancyBboxPatch((0.3, 0.6), 0.4, 0.15, boxstyle="round,pad=0.02",
                          facecolor='lightblue', edgecolor='black', linewidth=1)
    ax4.add_patch(rect1)
    rect2 = FancyBboxPatch((0.3, 0.3), 0.4, 0.15, boxstyle="round,pad=0.02",
                          facecolor='lightgreen', edgecolor='black', linewidth=1)
    ax4.add_patch(rect2)
    # Residual connections
    ax4.annotate('', xy=(0.2, 0.45), xytext=(0.2, 0.75),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, ls='--'))
    ax4.annotate('', xy=(0.2, 0.15), xytext=(0.2, 0.45),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, ls='--'))
    ax4.text(0.5, 0.675, 'Attention', ha='center', fontsize=8)
    ax4.text(0.5, 0.375, 'FFN', ha='center', fontsize=8)
    ax4.text(0.1, 0.45, '+', fontsize=14, color='red', fontweight='bold')
    ax4.text(0.1, 0.15, '+', fontsize=14, color='red', fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Step 4: Add\nResiduals', fontsize=10, fontweight='bold')
    
    # Step 5: Single layer complete
    ax5 = fig.add_subplot(gs[1, 0])
    components = ['Input', 'MHA', 'Add&Norm', 'FFN', 'Add&Norm', 'Output']
    y_positions = np.linspace(0.8, 0.2, len(components))
    colors = ['gray', 'lightblue', 'yellow', 'lightgreen', 'yellow', 'gray']
    
    for i, (comp, y, color) in enumerate(zip(components, y_positions, colors)):
        rect = FancyBboxPatch((0.2, y-0.05), 0.6, 0.08, boxstyle="round,pad=0.01",
                             facecolor=color, edgecolor='black', linewidth=1)
        ax5.add_patch(rect)
        ax5.text(0.5, y, comp, ha='center', va='center', fontsize=8)
        if i < len(components) - 1:
            arrow = FancyArrow(0.5, y-0.06, 0, -0.05, width=0.02, 
                             head_width=0.05, head_length=0.02, fc='gray', ec='gray')
            ax5.add_patch(arrow)
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    ax5.set_title('Step 5: One\nComplete Layer', fontsize=10, fontweight='bold')
    
    # Step 6: Stack layers
    ax6 = fig.add_subplot(gs[1, 1])
    for i in range(3):
        alpha = 0.3 + i * 0.2
        for j, (y, color) in enumerate([(0.7, 'lightblue'), (0.5, 'yellow'), (0.3, 'lightgreen')]):
            rect = FancyBboxPatch((0.2 - i*0.02, y - i*0.02), 0.6, 0.08,
                                 boxstyle="round,pad=0.01", facecolor=color,
                                 edgecolor='black', linewidth=1, alpha=alpha)
            ax6.add_patch(rect)
    ax6.text(0.5, 0.1, '× 6 layers', ha='center', fontsize=12, fontweight='bold')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_title('Step 6: Stack\n6 Layers', fontsize=10, fontweight='bold')
    
    # Step 7: Add encoder-decoder
    ax7 = fig.add_subplot(gs[1, 2])
    # Encoder stack
    rect1 = FancyBboxPatch((0.1, 0.3), 0.35, 0.5, boxstyle="round,pad=0.02",
                          facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax7.add_patch(rect1)
    ax7.text(0.275, 0.55, 'Encoder\n(6 layers)', ha='center', va='center', fontsize=8)
    # Decoder stack
    rect2 = FancyBboxPatch((0.55, 0.3), 0.35, 0.5, boxstyle="round,pad=0.02",
                          facecolor='lightskyblue', edgecolor='black', linewidth=2)
    ax7.add_patch(rect2)
    ax7.text(0.725, 0.55, 'Decoder\n(6 layers)', ha='center', va='center', fontsize=8)
    # Cross-attention arrow
    arrow = FancyArrow(0.45, 0.55, 0.1, 0, width=0.03, head_width=0.06,
                      head_length=0.03, fc='purple', ec='purple')
    ax7.add_patch(arrow)
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    ax7.axis('off')
    ax7.set_title('Step 7: Full\nEnc-Dec', fontsize=10, fontweight='bold')
    
    # Step 8: Scale up
    ax8 = fig.add_subplot(gs[1, 3])
    models = ['Base\n65M', 'Large\n355M', 'XL\n1.5B', 'XXL\n175B']
    sizes = [0.1, 0.2, 0.35, 0.5]
    colors_scale = ['lightblue', 'blue', 'darkblue', 'purple']
    
    for i, (model, size, color) in enumerate(zip(models, sizes, colors_scale)):
        circle = Circle((0.5, 0.5), size, facecolor=color, alpha=0.6,
                       edgecolor='black', linewidth=2)
        ax8.add_patch(circle)
        if i < 2:
            ax8.text(0.5, 0.5 - size - 0.1, model, ha='center', fontsize=8)
    ax8.text(0.5, 0.9, 'XXL\n175B', ha='center', fontsize=10, fontweight='bold', color='purple')
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    ax8.axis('off')
    ax8.set_title('Step 8: Scale\nto Billions', fontsize=10, fontweight='bold')
    
    plt.suptitle('Building a Transformer: From Single Head to GPT Scale', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/step_by_step_building.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all Week 5 enhanced figures"""
    print("Generating Week 5 Enhanced Transformer Visualizations...")
    
    print("1. Creating multi-head attention patterns...")
    plot_multihead_attention_patterns()
    
    print("2. Creating positional encoding visualization...")
    plot_positional_encoding_viz()
    
    print("3. Creating detailed architecture diagram...")
    plot_transformer_architecture_detailed()
    
    print("4. Creating comprehensive performance comparison...")
    plot_transformer_vs_rnn_performance()
    
    print("5. Creating applications timeline...")
    plot_transformer_applications_timeline()
    
    print("6. Creating attention mechanism breakdown...")
    plot_attention_mechanism_breakdown()
    
    print("7. Creating model scaling comparison...")
    plot_model_scaling_comparison()
    
    print("8. Creating layer-by-layer transformation...")
    plot_layer_by_layer_transformation()
    
    print("9. Creating step-by-step building...")
    plot_step_by_step_building()
    
    print("\nAll Week 5 enhanced figures generated successfully!")
    print("Files saved in ../figures/")

if __name__ == "__main__":
    main()