import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrow
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# 1. Tokenization Spectrum
def plot_tokenization_spectrum():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Example text
    text = "unbelievably"
    
    # Different tokenization levels
    levels = [
        ('Character', list(text), 12, 'lightcoral'),
        ('Subword', ['un', 'believ', 'ably'], 3, 'lightgreen'),
        ('Word', ['unbelievably'], 1, 'lightblue')
    ]
    
    y_positions = [0.7, 0.5, 0.3]
    
    for (name, tokens, count, color), y in zip(levels, y_positions):
        # Label
        ax.text(0.05, y, name, fontsize=14, fontweight='bold', va='center')
        
        # Tokens
        x_start = 0.2
        token_width = 0.6 / max(len(tokens), 12)  # Normalize width
        
        for i, token in enumerate(tokens):
            x = x_start + i * token_width * len(token)
            width = token_width * len(token) * 0.9
            
            box = FancyBboxPatch((x, y-0.05), width, 0.1,
                                boxstyle="round,pad=0.02",
                                facecolor=color,
                                edgecolor='black',
                                linewidth=2)
            ax.add_patch(box)
            ax.text(x + width/2, y, token, ha='center', va='center', fontsize=10)
        
        # Token count
        ax.text(0.85, y, f'{count} tokens', fontsize=12, va='center')
    
    # Add spectrum arrow
    ax.arrow(0.9, 0.75, 0, -0.5, head_width=0.02, head_length=0.03, 
             fc='gray', ec='gray', linewidth=2)
    ax.text(0.92, 0.5, 'Granularity', rotation=-90, va='center', fontsize=12)
    
    # Annotations
    ax.text(0.5, 0.9, 'The Tokenization Spectrum', ha='center', 
            fontsize=16, fontweight='bold')
    ax.text(0.5, 0.85, 'Example: "unbelievably"', ha='center', 
            fontsize=12, style='italic')
    
    # Trade-offs
    trade_offs = [
        (0.7, 'Too long sequences\nModel learns spelling', 'red'),
        (0.5, 'Balanced\nOptimal for most tasks', 'green'),
        (0.3, 'Vocabulary explosion\nCan\'t handle new words', 'red')
    ]
    
    for y, text, color in trade_offs:
        ax.text(0.95, y, text, fontsize=9, va='center', color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=color))
    
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0.1, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/tokenization_spectrum.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 2. BPE Visualization
def plot_bpe_visualization():
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # BPE iterations
    iterations = [
        ('Initial', ['l', 'o', 'w', '_', 'l', 'o', 'w', 'e', 'r', '_', 'l', 'o', 'w', 'e', 's', 't']),
        ('Merge "lo"', ['lo', 'w', '_', 'lo', 'w', 'e', 'r', '_', 'lo', 'w', 'e', 's', 't']),
        ('Merge "low"', ['low', '_', 'low', 'e', 'r', '_', 'low', 'e', 's', 't']),
        ('Merge "er"', ['low', '_', 'low', 'er', '_', 'low', 'e', 's', 't']),
        ('Merge "est"', ['low', '_', 'low', 'er', '_', 'low', 'est']),
        ('Final', ['low', '_', 'lower', '_', 'lowest'])
    ]
    
    y_positions = np.linspace(0.85, 0.15, len(iterations))
    
    for (label, tokens), y in zip(iterations, y_positions):
        # Iteration label
        ax.text(0.05, y, label, fontsize=12, fontweight='bold', va='center')
        
        # Tokens
        x_start = 0.2
        for i, token in enumerate(tokens):
            # Determine color based on token type
            if len(token) > 1 and token != '_':
                color = 'lightgreen'  # Merged tokens
            elif token == '_':
                color = 'lightgray'  # Space
            else:
                color = 'lightcoral'  # Single characters
            
            width = 0.05 * len(token)
            x = x_start + i * 0.06
            
            box = FancyBboxPatch((x, y-0.03), width, 0.06,
                                boxstyle="round,pad=0.01",
                                facecolor=color,
                                edgecolor='black',
                                linewidth=1)
            ax.add_patch(box)
            ax.text(x + width/2, y, token, ha='center', va='center', fontsize=9)
        
        # Show merge operation
        if 'Merge' in label:
            merge_token = label.split('"')[1]
            ax.text(0.75, y, f'Freq({merge_token}) = high', 
                   fontsize=10, va='center', style='italic', color='blue')
    
    # Title and annotations
    ax.text(0.5, 0.95, 'Byte Pair Encoding: Learning Subwords', 
            ha='center', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.91, 'Corpus: "low lower lowest"', 
            ha='center', fontsize=12, style='italic')
    
    # Legend
    legend_items = [
        (0.2, 0.05, 'lightcoral', 'Character'),
        (0.4, 0.05, 'lightgreen', 'Subword'),
        (0.6, 0.05, 'lightgray', 'Space')
    ]
    
    for x, y, color, label in legend_items:
        box = FancyBboxPatch((x-0.02, y-0.02), 0.04, 0.04,
                            boxstyle="round,pad=0.01",
                            facecolor=color,
                            edgecolor='black')
        ax.add_patch(box)
        ax.text(x+0.03, y, label, ha='left', va='center', fontsize=10)
    
    # Key insight
    ax.text(0.95, 0.5, 'BPE discovers:\n• "low" = root\n• "er" = comparative\n• "est" = superlative',
            ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/bpe_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Multilingual Tokenization
def plot_multilingual_tokenization():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Examples in different languages
    examples = [
        ('English', 'Hello world', ['Hello', ' world'], 2),
        ('Chinese', '你好世界', ['你', '好', '世', '界'], 4),
        ('Arabic', 'مرحبا بالعالم', ['مر', 'حبا', ' بال', 'عالم'], 4),
        ('Japanese', 'こんにちは世界', ['こん', 'にちは', '世', '界'], 4),
        ('Korean', '안녕하세요', ['안녕', '하', '세요'], 3),
        ('Emoji', 'Hello 👋 🌍', ['Hello', ' ', '👋', ' ', '🌍'], 5)
    ]
    
    y_positions = np.linspace(0.85, 0.15, len(examples))
    
    # Bar chart data
    languages = [ex[0] for ex in examples]
    token_counts = [ex[3] for ex in examples]
    
    # Create bars
    bars = ax.barh(y_positions, token_counts, height=0.08, color=sns.color_palette("husl", len(examples)))
    
    # Add text and tokens
    for (lang, text, tokens, count), y, bar in zip(examples, y_positions, bars):
        # Language and text
        ax.text(0.02, y, f'{lang}:', fontsize=11, fontweight='bold', va='center')
        ax.text(0.15, y, text, fontsize=10, va='center')
        
        # Token visualization
        x_start = 0.4
        for i, token in enumerate(tokens):
            width = 0.08
            x = x_start + i * (width + 0.01)
            
            box = FancyBboxPatch((x, y-0.03), width, 0.06,
                                boxstyle="round,pad=0.01",
                                facecolor='lightblue',
                                edgecolor='black',
                                linewidth=1,
                                alpha=0.7)
            ax.add_patch(box)
            ax.text(x + width/2, y, token, ha='center', va='center', fontsize=8)
        
        # Token count on bar
        ax.text(count + 0.1, y, f'{count} tokens', va='center', fontsize=10)
    
    # English baseline
    ax.axvline(x=2, color='red', linestyle='--', alpha=0.5)
    ax.text(2, 0.92, 'English baseline', ha='center', fontsize=9, color='red')
    
    # Title and labels
    ax.text(0.5, 0.98, 'Tokenization Across Languages', 
            ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.94, 'Same meaning, different token counts', 
            ha='center', fontsize=12, style='italic', transform=ax.transAxes)
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add cost implication
    ax.text(0.95, 0.1, 'Cost Impact:\nChinese text costs\n2x more than English\non GPT-4!',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'),
            transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('../figures/multilingual_tokenization.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Tokenization Impact
def plot_tokenization_impact():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Performance vs Vocabulary Size
    vocab_sizes = [1000, 5000, 10000, 20000, 30000, 50000, 100000]
    performance = [72, 84, 89, 91, 92, 92.5, 91]  # Diminishing returns
    sequence_length = [450, 250, 180, 150, 140, 135, 140]  # Optimal around 30K
    
    ax1_twin = ax1.twinx()
    ax1.plot(vocab_sizes, performance, 'o-', color='blue', linewidth=2, 
             markersize=8, label='Performance')
    ax1_twin.plot(vocab_sizes, sequence_length, 's-', color='red', linewidth=2, 
                  markersize=8, label='Avg Length')
    
    ax1.set_xlabel('Vocabulary Size', fontsize=11)
    ax1.set_ylabel('Performance (%)', fontsize=11, color='blue')
    ax1_twin.set_ylabel('Avg Sequence Length', fontsize=11, color='red')
    ax1.set_title('Vocabulary Size Trade-offs', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Optimal range
    ax1.axvspan(20000, 50000, alpha=0.2, color='green')
    ax1.text(30000, 85, 'Optimal\nRange', ha='center', fontsize=9)
    
    # 2. Token Distribution
    token_types = ['Common\nWords', 'Subwords', 'Rare\nPieces', 'Special\nTokens']
    token_percentages = [45, 35, 15, 5]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    ax2.pie(token_percentages, labels=token_types, colors=colors, autopct='%1.1f%%',
            startangle=90)
    ax2.set_title('Typical Token Distribution (GPT-2)', fontsize=12, fontweight='bold')
    
    # 3. Compression Rates
    methods = ['Character', 'BPE-8K', 'BPE-32K', 'BPE-100K', 'Word']
    compression = [1.0, 3.2, 4.5, 4.8, 5.0]
    oov_rate = [0, 0.1, 0.5, 1.2, 15.0]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax3.bar(x - width/2, compression, width, label='Compression Rate', color='green')
    ax3.bar(x + width/2, oov_rate, width, label='OOV Rate (%)', color='red')
    
    ax3.set_xlabel('Tokenization Method', fontsize=11)
    ax3.set_ylabel('Rate', fontsize=11)
    ax3.set_title('Compression vs Out-of-Vocabulary Trade-off', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    
    # 4. Cost Impact
    tasks = ['English\nChat', 'Code\nGeneration', 'Chinese\nTranslation', 'Math\nProblems']
    tokens_per_1k_chars = [250, 400, 700, 350]
    cost_per_task = [0.025, 0.040, 0.070, 0.035]  # in dollars
    
    ax4.scatter(tokens_per_1k_chars, cost_per_task, s=200, alpha=0.7)
    for i, task in enumerate(tasks):
        ax4.annotate(task, (tokens_per_1k_chars[i], cost_per_task[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Trend line
    z = np.polyfit(tokens_per_1k_chars, cost_per_task, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(200, 750, 100)
    ax4.plot(x_trend, p(x_trend), '--', color='gray', alpha=0.5)
    
    ax4.set_xlabel('Tokens per 1K Characters', fontsize=11)
    ax4.set_ylabel('Cost per 1K Characters ($)', fontsize=11)
    ax4.set_title('Tokenization Affects API Costs', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('The Hidden Impact of Tokenization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/tokenization_impact.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Vocabulary Comparison
def plot_vocabulary_comparison():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Model vocabularies
    models = ['GPT-2/3/4', 'BERT', 'T5', 'LLaMA', 'XLM-R', 'mT5', 'CANINE']
    vocab_sizes = [50.257, 30.522, 32.0, 32.0, 250.0, 250.0, 0]  # in thousands
    tokenizer_types = ['BPE', 'WordPiece', 'SentencePiece', 'SentencePiece', 
                       'SentencePiece', 'SentencePiece', 'Character']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']
    
    # Create horizontal bars
    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, vocab_sizes, color=colors, alpha=0.7)
    
    # Add tokenizer type labels
    for i, (bar, tokenizer) in enumerate(zip(bars, tokenizer_types)):
        width = bar.get_width()
        if width > 0:
            ax.text(width + 2, bar.get_y() + bar.get_height()/2, 
                   f'{tokenizer}', va='center', fontsize=10)
            ax.text(width/2, bar.get_y() + bar.get_height()/2, 
                   f'{vocab_sizes[i]:.1f}K', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white')
        else:
            ax.text(2, bar.get_y() + bar.get_height()/2, 
                   'No vocabulary (character-level)', va='center', fontsize=10)
    
    # Labels and formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Vocabulary Size (thousands)', fontsize=12)
    ax.set_title('Vocabulary Sizes Across Popular Models', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, 280)
    
    # Add annotations
    ax.axvline(x=32, color='gray', linestyle='--', alpha=0.5)
    ax.text(32, -0.7, 'Common choice:\n~32K tokens', ha='center', fontsize=9)
    
    ax.axvline(x=250, color='gray', linestyle='--', alpha=0.5)
    ax.text(250, -0.7, 'Multilingual:\n~250K tokens', ha='center', fontsize=9)
    
    # Note about trade-offs
    ax.text(0.5, 0.02, 'Larger vocabulary = Better multilingual support but more memory usage',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig('../figures/vocabulary_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all Week 8 visualizations
if __name__ == "__main__":
    print("Generating Week 8 Tokenization visualizations...")
    plot_tokenization_spectrum()
    print("- Tokenization spectrum created")
    plot_bpe_visualization()
    print("- BPE visualization created")
    plot_multilingual_tokenization()
    print("- Multilingual tokenization comparison created")
    plot_tokenization_impact()
    print("- Tokenization impact visualization created")
    plot_vocabulary_comparison()
    print("- Vocabulary comparison created")
    print("\nWeek 8 visualizations completed!")