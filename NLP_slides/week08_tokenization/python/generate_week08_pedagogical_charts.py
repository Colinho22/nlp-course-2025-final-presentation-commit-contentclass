"""
Generate pedagogical charts for Week 8: Tokenization & Vocabulary
Focus: Visual representations, not text-based tables
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Minimalist monochromatic palette (following user preferences)
COLOR_MAIN = '#404040'         # RGB(64,64,64)
COLOR_ACCENT = '#B4B4B4'       # RGB(180,180,180)
COLOR_LIGHT = '#F0F0F0'        # RGB(240,240,240)
COLOR_HIGHLIGHT = '#FF6B6B'    # Red for emphasis
COLOR_SUCCESS = '#95E77E'      # Green for positive outcomes
COLOR_WARNING = '#FFD93D'      # Yellow for caution

plt.style.use('seaborn-v0_8-whitegrid')

def create_tokenization_comparison():
    """Visual comparison of character, word, and subword tokenization."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle('Tokenization Approaches: Visual Comparison',
                 fontsize=14, fontweight='bold', color=COLOR_MAIN)

    # Example text
    text = "unhappiness"

    # Character-level
    ax = axes[0]
    char_tokens = list(text)
    positions = np.arange(len(char_tokens))
    bars = ax.barh(positions, [1]*len(char_tokens), height=0.6,
                   color=COLOR_ACCENT, edgecolor=COLOR_MAIN, linewidth=2)
    ax.set_yticks(positions)
    ax.set_yticklabels([f"'{c}'" for c in char_tokens], fontsize=10)
    ax.set_xlabel('Token Length (arbitrary units)', fontsize=10, color=COLOR_MAIN)
    ax.set_title('Character-Level: 11 tokens (high granularity)',
                 fontsize=11, fontweight='bold', color=COLOR_MAIN)
    ax.set_xlim(0, 6)
    ax.invert_yaxis()

    # Word-level
    ax = axes[1]
    word_tokens = [text]
    positions = np.arange(len(word_tokens))
    bars = ax.barh(positions, [5]*len(word_tokens), height=0.6,
                   color=COLOR_ACCENT, edgecolor=COLOR_MAIN, linewidth=2)
    ax.set_yticks(positions)
    ax.set_yticklabels([f"'{w}'" for w in word_tokens], fontsize=10)
    ax.set_xlabel('Token Length (arbitrary units)', fontsize=10, color=COLOR_MAIN)
    ax.set_title('Word-Level: 1 token (low granularity, OOV risk)',
                 fontsize=11, fontweight='bold', color=COLOR_MAIN)
    ax.set_xlim(0, 6)
    ax.invert_yaxis()

    # Subword-level (BPE)
    ax = axes[2]
    subword_tokens = ['un', 'happi', 'ness']
    positions = np.arange(len(subword_tokens))
    bars = ax.barh(positions, [2, 3, 2], height=0.6,
                   color=COLOR_SUCCESS, edgecolor=COLOR_MAIN, linewidth=2)
    ax.set_yticks(positions)
    ax.set_yticklabels([f"'{t}'" for t in subword_tokens], fontsize=10)
    ax.set_xlabel('Token Length (arbitrary units)', fontsize=10, color=COLOR_MAIN)
    ax.set_title('Subword (BPE): 3 tokens (balanced granularity, meaning preserved)',
                 fontsize=11, fontweight='bold', color=COLOR_SUCCESS)
    ax.set_xlim(0, 6)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('../figures/tokenization_comparison_visual.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: tokenization_comparison_visual.pdf")


def create_bpe_algorithm_progression():
    """Visualize BPE merge operations as a progression."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Simulation of BPE merges
    iterations = ['Init', 'Merge 1', 'Merge 2', 'Merge 3', 'Merge 4', 'Merge 5']
    vocab_sizes = [50, 75, 95, 110, 122, 132]  # Growing vocabulary
    avg_token_length = [1.0, 1.3, 1.6, 1.9, 2.2, 2.4]  # Growing token complexity

    # Create dual-axis plot
    ax1 = ax
    color1 = COLOR_HIGHLIGHT
    ax1.plot(iterations, vocab_sizes, marker='o', linewidth=3,
             markersize=10, color=color1, label='Vocabulary Size')
    ax1.set_xlabel('BPE Iteration', fontsize=12, fontweight='bold', color=COLOR_MAIN)
    ax1.set_ylabel('Vocabulary Size', fontsize=12, fontweight='bold', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = COLOR_SUCCESS
    ax2.plot(iterations, avg_token_length, marker='s', linewidth=3,
             markersize=10, color=color2, linestyle='--', label='Avg Token Length')
    ax2.set_ylabel('Average Token Length (characters)', fontsize=12,
                   fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Add annotations
    ax1.annotate('Most frequent pairs merged', xy=(2, 95), xytext=(2, 70),
                arrowprops=dict(arrowstyle='->', color=COLOR_MAIN, lw=2),
                fontsize=10, color=COLOR_MAIN, ha='center')

    ax1.set_title('BPE Algorithm: Vocabulary Growth Through Merging',
                  fontsize=14, fontweight='bold', color=COLOR_MAIN, pad=20)

    # Create custom legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('../figures/bpe_progression_visual.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: bpe_progression_visual.pdf")


def create_vocab_size_vs_oov():
    """Vocabulary size vs Out-of-Vocabulary rate for different approaches."""
    fig, ax = plt.subplots(figsize=(10, 6))

    vocab_sizes = np.array([1000, 5000, 10000, 20000, 30000, 50000])

    # Simulated OOV rates (character-level has 0% OOV but impractical)
    word_level = np.array([35, 18, 12, 8, 6, 4])  # High OOV for rare words
    bpe = np.array([8, 3, 1.5, 0.8, 0.5, 0.3])    # Much better
    wordpiece = np.array([7, 2.8, 1.3, 0.7, 0.4, 0.2])  # Slightly better than BPE
    character = np.array([0, 0, 0, 0, 0, 0])      # Zero OOV but inefficient

    ax.plot(vocab_sizes, word_level, marker='o', linewidth=2.5, markersize=8,
            color=COLOR_HIGHLIGHT, label='Word-Level (OOV problem!)', linestyle='--')
    ax.plot(vocab_sizes, bpe, marker='s', linewidth=2.5, markersize=8,
            color=COLOR_SUCCESS, label='BPE (Balanced)')
    ax.plot(vocab_sizes, wordpiece, marker='^', linewidth=2.5, markersize=8,
            color='#4ECDC4', label='WordPiece (Optimal)')
    ax.plot(vocab_sizes, character, marker='D', linewidth=2.5, markersize=8,
            color=COLOR_ACCENT, label='Character-Level (No OOV but inefficient)')

    ax.set_xlabel('Vocabulary Size', fontsize=12, fontweight='bold', color=COLOR_MAIN)
    ax.set_ylabel('Out-of-Vocabulary Rate (%)', fontsize=12, fontweight='bold', color=COLOR_MAIN)
    ax.set_title('Vocabulary Size vs OOV Rate: Why Subword Tokenization Wins',
                 fontsize=14, fontweight='bold', color=COLOR_MAIN, pad=15)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    ax.set_xticks(vocab_sizes)
    ax.set_xticklabels([f'{v//1000}K' if v >= 1000 else str(v) for v in vocab_sizes])

    # Add shaded region for optimal range
    ax.axvspan(10000, 30000, alpha=0.15, color=COLOR_SUCCESS,
               label='Optimal Range (10K-30K)')

    plt.tight_layout()
    plt.savefig('../figures/vocab_size_oov_visual.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: vocab_size_oov_visual.pdf")


def create_multilingual_efficiency():
    """Multilingual tokenization efficiency comparison."""
    fig, ax = plt.subplots(figsize=(10, 7))

    languages = ['English', 'German', 'Chinese', 'Arabic', 'Finnish']

    # Tokens per 100 words (lower is more efficient)
    word_level = [100, 100, 100, 100, 100]  # Baseline
    bpe = [120, 145, 95, 130, 180]  # Varies by morphology
    wordpiece = [115, 140, 90, 125, 175]  # Slightly better

    x = np.arange(len(languages))
    width = 0.25

    bars1 = ax.bar(x - width, word_level, width, label='Word-Level',
                   color=COLOR_ACCENT, edgecolor=COLOR_MAIN, linewidth=1.5)
    bars2 = ax.bar(x, bpe, width, label='BPE',
                   color=COLOR_SUCCESS, edgecolor=COLOR_MAIN, linewidth=1.5)
    bars3 = ax.bar(x + width, wordpiece, width, label='WordPiece',
                   color='#4ECDC4', edgecolor=COLOR_MAIN, linewidth=1.5)

    # Add efficiency indicator line at 100
    ax.axhline(y=100, color=COLOR_HIGHLIGHT, linestyle='--', linewidth=2,
               alpha=0.7, label='Baseline (Word-Level)')

    ax.set_xlabel('Language', fontsize=12, fontweight='bold', color=COLOR_MAIN)
    ax.set_ylabel('Tokens per 100 Words', fontsize=12, fontweight='bold', color=COLOR_MAIN)
    ax.set_title('Multilingual Tokenization Efficiency\n(Lower is more efficient)',
                 fontsize=14, fontweight='bold', color=COLOR_MAIN, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(languages, fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotation for Finnish
    ax.annotate('Highly morphological\nlanguage needs more\nsubword units',
                xy=(4, 180), xytext=(3.2, 160),
                arrowprops=dict(arrowstyle='->', color=COLOR_MAIN, lw=2),
                fontsize=9, color=COLOR_MAIN, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT, alpha=0.8))

    plt.tight_layout()
    plt.savefig('../figures/multilingual_efficiency_visual.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: multilingual_efficiency_visual.pdf")


def create_rare_word_handling():
    """Visual comparison of rare word handling."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Rare Word Handling: Word-Level vs Subword',
                 fontsize=14, fontweight='bold', color=COLOR_MAIN)

    # Word-level (information loss)
    ax = axes[0]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Input
    ax.text(5, 9, 'Input: "pneumonoultramicroscopicsilicovolcanoconiosis"',
            ha='center', fontsize=10, fontweight='bold', color=COLOR_MAIN)
    ax.add_patch(FancyBboxPatch((1, 7), 8, 1, boxstyle='round,pad=0.1',
                                facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN, linewidth=2))
    ax.text(5, 7.5, 'Very long medical term (rare word)',
            ha='center', fontsize=9, color=COLOR_MAIN)

    # Arrow down
    ax.arrow(5, 6.8, 0, -1.5, head_width=0.3, head_length=0.2,
             fc=COLOR_HIGHLIGHT, ec=COLOR_HIGHLIGHT, linewidth=2)

    # Output (UNK token)
    ax.add_patch(FancyBboxPatch((3, 4), 4, 1, boxstyle='round,pad=0.1',
                                facecolor=COLOR_HIGHLIGHT, edgecolor=COLOR_MAIN, linewidth=2, alpha=0.5))
    ax.text(5, 4.5, '<UNK>', ha='center', fontsize=12, fontweight='bold', color='white')

    # Problem annotation
    ax.text(5, 3, 'PROBLEM: All information lost!', ha='center', fontsize=10,
            fontweight='bold', color=COLOR_HIGHLIGHT,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor=COLOR_HIGHLIGHT, linewidth=2))

    ax.text(5, 1.5, 'Model has no idea what\nthe original word meant',
            ha='center', fontsize=9, color=COLOR_MAIN)

    ax.set_title('Word-Level: Information Loss', fontsize=12,
                 fontweight='bold', color=COLOR_HIGHLIGHT, pad=10)

    # Subword (information preserved)
    ax = axes[1]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Input
    ax.text(5, 9, 'Input: "pneumonoultramicroscopicsilicovolcanoconiosis"',
            ha='center', fontsize=10, fontweight='bold', color=COLOR_MAIN)
    ax.add_patch(FancyBboxPatch((1, 7), 8, 1, boxstyle='round,pad=0.1',
                                facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN, linewidth=2))
    ax.text(5, 7.5, 'Very long medical term (rare word)',
            ha='center', fontsize=9, color=COLOR_MAIN)

    # Arrow down
    ax.arrow(5, 6.8, 0, -1.5, head_width=0.3, head_length=0.2,
             fc=COLOR_SUCCESS, ec=COLOR_SUCCESS, linewidth=2)

    # Output (subword units)
    subwords = ['pneum', 'ono', 'ultra', 'micro', 'scop', 'ic',
                'silic', 'o', 'volcano', 'coni', 'osis']
    y_start = 5.5
    for i, subword in enumerate(subwords[:6]):  # Show first 6
        ax.add_patch(FancyBboxPatch((0.5 + i*1.5, y_start - (i%2)*0.7), 1.3, 0.6,
                                    boxstyle='round,pad=0.05',
                                    facecolor=COLOR_SUCCESS, edgecolor=COLOR_MAIN,
                                    linewidth=1.5, alpha=0.7))
        ax.text(1.15 + i*1.5, y_start + 0.3 - (i%2)*0.7, subword,
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    ax.text(8, 4.5, '...', ha='center', fontsize=16, fontweight='bold', color=COLOR_MAIN)

    # Success annotation
    ax.text(5, 3, 'SUCCESS: Meaning preserved!', ha='center', fontsize=10,
            fontweight='bold', color=COLOR_SUCCESS,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor=COLOR_SUCCESS, linewidth=2))

    ax.text(5, 1.5, 'Model can understand:\n"pneum" (lung), "micro" (small),\n"scop" (viewing), etc.',
            ha='center', fontsize=9, color=COLOR_MAIN)

    ax.set_title('Subword (BPE): Meaning Preserved', fontsize=12,
                 fontweight='bold', color=COLOR_SUCCESS, pad=10)

    plt.tight_layout()
    plt.savefig('../figures/rare_word_handling_visual.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: rare_word_handling_visual.pdf")


def create_tokenization_impact_performance():
    """Impact of tokenization on model performance metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Tokenization Impact on Model Performance',
                 fontsize=14, fontweight='bold', color=COLOR_MAIN)

    approaches = ['Character', 'Word', 'BPE', 'WordPiece']

    # Perplexity (lower is better)
    ax = axes[0]
    perplexity = [85, 120, 45, 42]
    bars = ax.bar(approaches, perplexity, color=[COLOR_ACCENT, COLOR_HIGHLIGHT,
                                                   COLOR_SUCCESS, COLOR_SUCCESS],
                  edgecolor=COLOR_MAIN, linewidth=2)
    ax.set_ylabel('Perplexity (lower is better)', fontsize=10, fontweight='bold', color=COLOR_MAIN)
    ax.set_title('Language Model Quality', fontsize=11, fontweight='bold', color=COLOR_MAIN)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)

    # Training time (relative)
    ax = axes[1]
    training_time = [2.5, 1.0, 1.3, 1.2]  # Relative to word-level
    bars = ax.bar(approaches, training_time, color=[COLOR_HIGHLIGHT, COLOR_SUCCESS,
                                                      COLOR_SUCCESS, COLOR_SUCCESS],
                  edgecolor=COLOR_MAIN, linewidth=2)
    ax.set_ylabel('Training Time (relative)', fontsize=10, fontweight='bold', color=COLOR_MAIN)
    ax.set_title('Training Efficiency', fontsize=11, fontweight='bold', color=COLOR_MAIN)
    ax.axhline(y=1.0, color=COLOR_ACCENT, linestyle='--', linewidth=2, alpha=0.7)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)

    # Vocabulary size (thousands)
    ax = axes[2]
    vocab_size = [150, 50, 32, 30]
    bars = ax.bar(approaches, vocab_size, color=[COLOR_HIGHLIGHT, COLOR_WARNING,
                                                   COLOR_SUCCESS, COLOR_SUCCESS],
                  edgecolor=COLOR_MAIN, linewidth=2)
    ax.set_ylabel('Vocabulary Size (K)', fontsize=10, fontweight='bold', color=COLOR_MAIN)
    ax.set_title('Memory Efficiency', fontsize=11, fontweight='bold', color=COLOR_MAIN)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig('../figures/tokenization_performance_visual.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: tokenization_performance_visual.pdf")


def create_bpe_merge_step_detail():
    """Detailed visualization of a single BPE merge step."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Title
    ax.text(5, 9.5, 'BPE Merge Operation: Step-by-Step Example',
            ha='center', fontsize=14, fontweight='bold', color=COLOR_MAIN)

    # Step 1: Initial state
    ax.text(1, 8.5, 'Step 1: Count all adjacent pairs',
            fontsize=11, fontweight='bold', color=COLOR_MAIN)

    # Example corpus visualization
    corpus_text = ['l o w', 'l o w e r', 'n e w e s t']
    y_pos = 7.5
    for text in corpus_text:
        ax.add_patch(FancyBboxPatch((1, y_pos), 3, 0.4, boxstyle='round,pad=0.05',
                                    facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN, linewidth=1.5))
        ax.text(2.5, y_pos+0.2, text, ha='center', va='center', fontsize=9,
                fontfamily='monospace', color=COLOR_MAIN)
        y_pos -= 0.6

    # Step 2: Find most frequent pair
    ax.text(5.5, 8.5, 'Step 2: Most frequent pair',
            fontsize=11, fontweight='bold', color=COLOR_MAIN)

    pair_counts = [('e s', 2), ('l o', 2), ('o w', 2), ('n e', 1), ('w e', 1)]
    y_pos = 7.8
    for pair, count in pair_counts[:3]:
        color = COLOR_SUCCESS if count == 2 else COLOR_ACCENT
        ax.add_patch(FancyBboxPatch((5.5, y_pos), 2, 0.4, boxstyle='round,pad=0.05',
                                    facecolor=color, edgecolor=COLOR_MAIN,
                                    linewidth=1.5, alpha=0.7))
        ax.text(6.5, y_pos+0.2, f"'{pair}': {count}", ha='center', va='center',
                fontsize=9, fontfamily='monospace', color='white', fontweight='bold')
        y_pos -= 0.5

    # Step 3: Merge
    ax.text(1, 5.5, 'Step 3: Merge chosen pair (es)',
            fontsize=11, fontweight='bold', color=COLOR_SUCCESS)

    # Show merge operation
    ax.add_patch(FancyBboxPatch((1, 4.5), 1.5, 0.6, boxstyle='round,pad=0.05',
                                facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN, linewidth=1.5))
    ax.text(1.75, 4.8, 'e s', ha='center', va='center', fontsize=10,
            fontfamily='monospace', color=COLOR_MAIN)

    ax.arrow(2.8, 4.8, 0.8, 0, head_width=0.2, head_length=0.2,
             fc=COLOR_SUCCESS, ec=COLOR_SUCCESS, linewidth=2)

    ax.add_patch(FancyBboxPatch((4, 4.5), 1.2, 0.6, boxstyle='round,pad=0.05',
                                facecolor=COLOR_SUCCESS, edgecolor=COLOR_MAIN,
                                linewidth=2, alpha=0.8))
    ax.text(4.6, 4.8, 'es', ha='center', va='center', fontsize=10,
            fontfamily='monospace', color='white', fontweight='bold')

    # Step 4: Updated corpus
    ax.text(1, 3.5, 'Step 4: Update corpus with new token',
            fontsize=11, fontweight='bold', color=COLOR_MAIN)

    updated_corpus = ['l o w', 'l o w e r', 'n e w es t']
    y_pos = 2.5
    for i, text in enumerate(updated_corpus):
        ax.add_patch(FancyBboxPatch((1, y_pos), 3, 0.4, boxstyle='round,pad=0.05',
                                    facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN, linewidth=1.5))
        # Highlight the merged token
        if 'es' in text:
            parts = text.split('es')
            full_text = text.replace('es', '**es**')
            ax.text(2.5, y_pos+0.2, full_text, ha='center', va='center', fontsize=9,
                    fontfamily='monospace', color=COLOR_SUCCESS, fontweight='bold')
        else:
            ax.text(2.5, y_pos+0.2, text, ha='center', va='center', fontsize=9,
                    fontfamily='monospace', color=COLOR_MAIN)
        y_pos -= 0.6

    # Key insight box
    ax.add_patch(FancyBboxPatch((5.5, 1.5), 3.5, 1.5, boxstyle='round,pad=0.1',
                                facecolor=COLOR_WARNING, edgecolor=COLOR_MAIN,
                                linewidth=2, alpha=0.3))
    ax.text(7.25, 2.5, 'Key Insight:', ha='center', fontsize=10,
            fontweight='bold', color=COLOR_MAIN)
    ax.text(7.25, 2, 'BPE learns subword units\nfrom frequency patterns',
            ha='center', fontsize=9, color=COLOR_MAIN)
    ax.text(7.25, 1.3, 'No linguistic knowledge needed!',
            ha='center', fontsize=8, style='italic', color=COLOR_MAIN)

    plt.tight_layout()
    plt.savefig('../figures/bpe_merge_detail_visual.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: bpe_merge_detail_visual.pdf")


if __name__ == "__main__":
    print("Generating Week 8 pedagogical charts...")
    print("Focus: Visual representations (not text-based tables)")
    print("-" * 60)

    create_tokenization_comparison()
    create_bpe_algorithm_progression()
    create_vocab_size_vs_oov()
    create_multilingual_efficiency()
    create_rare_word_handling()
    create_tokenization_impact_performance()
    create_bpe_merge_step_detail()

    print("-" * 60)
    print("All charts generated successfully!")
    print("Output directory: NLP_slides/week08_tokenization/figures/")
