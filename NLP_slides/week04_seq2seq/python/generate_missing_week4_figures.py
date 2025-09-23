import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow, FancyArrowPatch
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

def plot_seq2seq_architecture():
    """Visualize seq2seq architecture with encoder-decoder"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Colors
    encoder_color = '#4A90E2'
    decoder_color = '#F5A623'
    context_color = '#D0021B'
    input_color = '#7ED321'
    output_color = '#BD10E0'
    
    # Encoder side
    encoder_words = ['The', 'cat', 'sat', '<END>']
    encoder_y = 2
    cell_width = 1.5
    cell_height = 0.8
    spacing = 2.2
    
    # Draw encoder
    ax.text(-1, encoder_y + 1.5, 'ENCODER', fontsize=16, fontweight='bold', 
           color=encoder_color, ha='center')
    
    encoder_cells = []
    for i, word in enumerate(encoder_words):
        x_pos = i * spacing
        
        # Input word
        input_box = Rectangle((x_pos - 0.4, encoder_y - 2), 0.8, 0.5,
                             facecolor=input_color, edgecolor='black')
        ax.add_patch(input_box)
        ax.text(x_pos, encoder_y - 1.75, word, ha='center', va='center',
               fontsize=10, fontweight='bold')
        
        # LSTM cell
        cell = FancyBboxPatch((x_pos - cell_width/2, encoder_y - cell_height/2),
                             cell_width, cell_height,
                             boxstyle="round,pad=0.1",
                             facecolor=encoder_color, edgecolor='black', linewidth=2)
        ax.add_patch(cell)
        ax.text(x_pos, encoder_y, f'LSTM\n{i+1}', ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')
        
        # Arrow from input to cell
        ax.arrow(x_pos, encoder_y - 1.5, 0, 0.8,
                head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Hidden state connections
        if i < len(encoder_words) - 1:
            ax.arrow(x_pos + cell_width/2, encoder_y, spacing - cell_width, 0,
                    head_width=0.08, head_length=0.1, fc='gray', ec='gray')
        
        encoder_cells.append((x_pos, encoder_y))
    
    # Context vector (bottleneck)
    context_x = len(encoder_words) * spacing + 1
    context_box = Circle((context_x, encoder_y), 0.6,
                        facecolor=context_color, edgecolor='black', linewidth=3)
    ax.add_patch(context_box)
    ax.text(context_x, encoder_y, 'Context\nVector\n(Fixed)', ha='center', va='center',
           fontsize=9, fontweight='bold', color='white')
    
    # Arrow from last encoder to context
    ax.arrow(encoder_cells[-1][0] + cell_width/2, encoder_y, 
            context_x - encoder_cells[-1][0] - cell_width/2 - 0.6, 0,
            head_width=0.1, head_length=0.1, fc=context_color, ec=context_color, linewidth=3)
    
    # Decoder side
    decoder_words = ['<START>', 'Le', 'chat', 'dort']
    decoder_outputs = ['Le', 'chat', 'dort', '<END>']
    decoder_y = -2
    decoder_start_x = context_x + 2
    
    ax.text(decoder_start_x + len(decoder_words) * spacing / 2, decoder_y - 1.5, 
           'DECODER', fontsize=16, fontweight='bold', color=decoder_color, ha='center')
    
    # Arrow from context to decoder
    ax.arrow(context_x, encoder_y - 0.6, 0, decoder_y - encoder_y + 1.2,
            head_width=0.15, head_length=0.15, fc=context_color, ec=context_color, linewidth=3)
    ax.text(context_x + 0.3, (encoder_y + decoder_y) / 2, 'Initialize\nDecoder', 
           fontsize=9, fontweight='bold', color=context_color)
    
    for i, (input_word, output_word) in enumerate(zip(decoder_words, decoder_outputs)):
        x_pos = decoder_start_x + i * spacing
        
        # Input word (previous output or <START>)
        input_box = Rectangle((x_pos - 0.4, decoder_y - 2), 0.8, 0.5,
                             facecolor=input_color, edgecolor='black')
        ax.add_patch(input_box)
        ax.text(x_pos, decoder_y - 1.75, input_word, ha='center', va='center',
               fontsize=10, fontweight='bold')
        
        # LSTM cell
        cell = FancyBboxPatch((x_pos - cell_width/2, decoder_y - cell_height/2),
                             cell_width, cell_height,
                             boxstyle="round,pad=0.1",
                             facecolor=decoder_color, edgecolor='black', linewidth=2)
        ax.add_patch(cell)
        ax.text(x_pos, decoder_y, f'LSTM\n{i+1}', ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')
        
        # Output word
        output_box = Rectangle((x_pos - 0.4, decoder_y + 1.2), 0.8, 0.5,
                              facecolor=output_color, edgecolor='black')
        ax.add_patch(output_box)
        ax.text(x_pos, decoder_y + 1.45, output_word, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')
        
        # Arrows
        ax.arrow(x_pos, decoder_y - 1.5, 0, 0.8,
                head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(x_pos, decoder_y + cell_height/2, 0, 0.5,
                head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Hidden state connections
        if i < len(decoder_words) - 1:
            ax.arrow(x_pos + cell_width/2, decoder_y, spacing - cell_width, 0,
                    head_width=0.08, head_length=0.1, fc='gray', ec='gray')
            
            # Feedback connection (previous output becomes next input)
            if i < len(decoder_words) - 1:
                ax.annotate('', xy=(x_pos + spacing, decoder_y - 1.75), 
                           xytext=(x_pos, decoder_y + 1.45),
                           arrowprops=dict(arrowstyle='->', color='purple', 
                                         lw=2, connectionstyle="arc3,rad=0.3"))
    
    # Labels and annotations
    ax.text(-1, encoder_y - 2, 'Input:', fontsize=12, fontweight='bold', ha='right')
    ax.text(decoder_start_x - 1, decoder_y - 2, 'Input:', fontsize=12, fontweight='bold', ha='right')
    ax.text(decoder_start_x - 1, decoder_y + 1.45, 'Output:', fontsize=12, fontweight='bold', ha='right')
    
    # Title
    ax.set_title('Sequence-to-Sequence Architecture with Encoder-Decoder', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Set limits and clean up
    ax.set_xlim(-2, decoder_start_x + len(decoder_words) * spacing + 1)
    ax.set_ylim(-4.5, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/seq2seq_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_bottleneck_problem():
    """Visualize information bottleneck in seq2seq"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top: Information flow visualization
    sentence_lengths = np.array([5, 10, 20, 30, 50, 100])
    context_size = 512  # Fixed context vector size
    
    # Information capacity
    input_info = sentence_lengths * 50  # Assume 50 bits per word
    context_capacity = np.full_like(sentence_lengths, context_size)
    
    ax1.bar(sentence_lengths - 1, input_info, width=2, alpha=0.7, 
           label='Input Information', color='blue')
    ax1.axhline(y=context_size, color='red', linestyle='--', linewidth=3,
               label=f'Context Vector Capacity ({context_size} units)')
    
    ax1.set_xlabel('Input Sentence Length (words)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Information Content', fontsize=12, fontweight='bold')
    ax1.set_title('Information Bottleneck Problem', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Annotations
    ax1.annotate('Information Loss!', xy=(50, input_info[4]), xytext=(70, 1500),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold', color='red')
    
    # Bottom: Performance degradation
    sentence_lengths_perf = np.arange(5, 101, 5)
    
    # Simulated BLEU scores (quality degrades with length)
    bleu_no_attention = 40 * np.exp(-0.02 * sentence_lengths_perf) + 10
    bleu_with_attention = 38 * np.exp(-0.005 * sentence_lengths_perf) + 20
    
    ax2.plot(sentence_lengths_perf, bleu_no_attention, 'r-o', linewidth=3, 
            markersize=6, label='Seq2Seq (No Attention)')
    ax2.plot(sentence_lengths_perf, bleu_with_attention, 'b-s', linewidth=3, 
            markersize=6, label='Seq2Seq + Attention')
    
    ax2.set_xlabel('Source Sentence Length', fontsize=12, fontweight='bold')
    ax2.set_ylabel('BLEU Score (Translation Quality)', fontsize=12, fontweight='bold')
    ax2.set_title('Translation Quality vs Sentence Length', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add threshold line
    ax2.axhline(y=30, color='orange', linestyle=':', linewidth=2, 
               label='Acceptable Quality Threshold')
    ax2.legend(fontsize=11)
    
    # Annotations
    ax2.annotate('Rapid degradation\nafter 30 words', 
                xy=(35, bleu_no_attention[6]), xytext=(60, 15),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red')
    
    ax2.annotate('Attention helps\nbut still limited', 
                xy=(70, bleu_with_attention[13]), xytext=(50, 35),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=11, fontweight='bold', color='blue')
    
    plt.suptitle('The Context Vector Bottleneck Problem', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/bottleneck_problem.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_attention_visualization():
    """Create attention heatmap visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Example 1: English to French
    english_words = ['The', 'black', 'cat', 'sleeps', 'quietly']
    french_words = ['Le', 'chat', 'noir', 'dort', 'silencieusement']
    
    # Create attention matrix (manually designed for good visualization)
    attention1 = np.array([
        [0.8, 0.1, 0.05, 0.03, 0.02],  # Le -> The
        [0.1, 0.05, 0.8, 0.03, 0.02],  # chat -> cat
        [0.05, 0.85, 0.05, 0.03, 0.02], # noir -> black
        [0.05, 0.05, 0.1, 0.75, 0.05],  # dort -> sleeps
        [0.05, 0.05, 0.1, 0.1, 0.7]     # silencieusement -> quietly
    ])
    
    im1 = ax1.imshow(attention1, cmap='Blues', aspect='auto')
    ax1.set_xticks(range(len(english_words)))
    ax1.set_yticks(range(len(french_words)))
    ax1.set_xticklabels(english_words, fontsize=12)
    ax1.set_yticklabels(french_words, fontsize=12)
    ax1.set_xlabel('Source (English)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Target (French)', fontsize=12, fontweight='bold')
    ax1.set_title('Attention Alignment: English → French', fontsize=14, fontweight='bold')
    
    # Add attention weights as text
    for i in range(len(french_words)):
        for j in range(len(english_words)):
            ax1.text(j, i, f'{attention1[i, j]:.2f}', ha='center', va='center',
                    color='white' if attention1[i, j] > 0.5 else 'black',
                    fontweight='bold')
    
    plt.colorbar(im1, ax=ax1, shrink=0.6)
    
    # Example 2: English to German (with reordering)
    english_words2 = ['I', 'have', 'a', 'red', 'car']
    german_words = ['Ich', 'habe', 'ein', 'rotes', 'Auto']
    
    # German word order is different - adjective comes after noun differently
    attention2 = np.array([
        [0.9, 0.05, 0.02, 0.02, 0.01],  # Ich -> I
        [0.05, 0.85, 0.05, 0.03, 0.02], # habe -> have
        [0.03, 0.05, 0.85, 0.05, 0.02], # ein -> a
        [0.02, 0.03, 0.1, 0.8, 0.05],   # rotes -> red (adjective)
        [0.02, 0.03, 0.05, 0.1, 0.8]    # Auto -> car
    ])
    
    im2 = ax2.imshow(attention2, cmap='Reds', aspect='auto')
    ax2.set_xticks(range(len(english_words2)))
    ax2.set_yticks(range(len(german_words)))
    ax2.set_xticklabels(english_words2, fontsize=12)
    ax2.set_yticklabels(german_words, fontsize=12)
    ax2.set_xlabel('Source (English)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Target (German)', fontsize=12, fontweight='bold')
    ax2.set_title('Attention Alignment: English → German', fontsize=14, fontweight='bold')
    
    # Add attention weights as text
    for i in range(len(german_words)):
        for j in range(len(english_words2)):
            ax2.text(j, i, f'{attention2[i, j]:.2f}', ha='center', va='center',
                    color='white' if attention2[i, j] > 0.5 else 'black',
                    fontweight='bold')
    
    plt.colorbar(im2, ax=ax2, shrink=0.6)
    
    plt.suptitle('Attention Mechanism: Learning What to Focus On', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/attention_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_seq2seq_performance():
    """Plot seq2seq performance evolution over time"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Timeline of major improvements
    years = np.array([2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])
    
    # BLEU scores on WMT English-German
    phrase_based_smt = np.array([20.1, 20.1, 20.1, 20.1, 20.1, 20.1, 20.1, 20.1])  # Baseline
    seq2seq_basic = np.array([0, 15.2, 16.5, 18.2, 19.1, 19.8, 20.2, 20.5])
    seq2seq_attention = np.array([0, 0, 22.8, 24.1, 25.3, 26.2, 27.1, 27.8])
    transformer = np.array([0, 0, 0, 0, 28.4, 29.8, 31.2, 32.5])
    
    ax1.plot(years, phrase_based_smt, 'k--', linewidth=3, label='Phrase-based SMT')
    ax1.plot(years[1:], seq2seq_basic[1:], 'r-o', linewidth=3, label='Basic Seq2Seq')
    ax1.plot(years[2:], seq2seq_attention[2:], 'b-s', linewidth=3, label='Seq2Seq + Attention')
    ax1.plot(years[4:], transformer[4:], 'g-^', linewidth=3, label='Transformer')
    
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
    ax1.set_title('Translation Quality Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Model sizes over time
    model_sizes = [50, 120, 150, 213, 340, 512, 850, 1200]  # Million parameters
    ax2.semilogy(years, model_sizes, 'purple', marker='o', linewidth=3, markersize=8)
    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Model Size (Million Parameters)', fontsize=12, fontweight='bold')
    ax2.set_title('Growing Model Complexity', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Training time comparison
    methods = ['SMT\n(Traditional)', 'Basic\nSeq2Seq', 'Seq2Seq +\nAttention', 'Transformer']
    training_hours = [0.5, 24, 48, 96]  # Hours on standard dataset
    colors = ['gray', 'red', 'blue', 'green']
    
    bars = ax3.bar(methods, training_hours, color=colors, alpha=0.7)
    ax3.set_ylabel('Training Time (Hours)', fontsize=12, fontweight='bold')
    ax3.set_title('Computational Requirements', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, hours in zip(bars, training_hours):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{hours}h', ha='center', va='bottom', fontweight='bold')
    
    # Applications growth
    applications = ['Translation', 'Summarization', 'Dialogue', 'Code Gen', 'Image Caption']
    year_introduced = [2014, 2015, 2016, 2018, 2017]
    adoption_2024 = [95, 80, 75, 60, 70]  # Percentage adoption
    
    scatter = ax4.scatter(year_introduced, adoption_2024, 
                         s=[200]*len(applications), 
                         c=['red', 'blue', 'green', 'orange', 'purple'], 
                         alpha=0.7)
    
    for i, app in enumerate(applications):
        ax4.annotate(app, (year_introduced[i], adoption_2024[i]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    ax4.set_xlabel('Year Introduced', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Industry Adoption % (2024)', fontsize=12, fontweight='bold')
    ax4.set_title('Seq2Seq Applications Spread', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(2013, 2019)
    ax4.set_ylim(50, 100)
    
    plt.suptitle('Seq2Seq: From Research to Production', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/seq2seq_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Generating missing Week 4 Seq2Seq figures...")
    
    print("1. Creating seq2seq architecture visualization...")
    plot_seq2seq_architecture()
    
    print("2. Creating bottleneck problem visualization...")
    plot_bottleneck_problem()
    
    print("3. Creating attention visualization...")
    plot_attention_visualization()
    
    print("4. Creating seq2seq performance timeline...")
    plot_seq2seq_performance()
    
    print("Week 4 figures generated successfully!")

if __name__ == "__main__":
    main()