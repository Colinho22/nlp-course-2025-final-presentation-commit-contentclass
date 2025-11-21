#!/usr/bin/env python3
"""
Generate 4 missing charts for Week 9 Decoding module
Charts to generate:
1. topk_example_bsc.pdf - Top-k numerical example
2. contrastive_vs_nucleus_bsc.pdf - Direct comparison
3. computational_cost_comparison_bsc.pdf - Speed comparison
4. production_settings_bsc.pdf - What real systems use
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# BSc Discovery Color Scheme
COLOR_MAIN = '#404040'      # Main elements (dark gray)
COLOR_ACCENT = '#3333B2'    # Key concepts (purple)
COLOR_GRAY = '#B4B4B4'      # Secondary elements
COLOR_LIGHT = '#F0F0F0'     # Backgrounds
COLOR_GREEN = '#2CA02C'     # Success/positive
COLOR_RED = '#D62728'       # Error/negative
COLOR_ORANGE = '#FF7F0E'    # Warning/medium
COLOR_BLUE = '#0066CC'      # Information

# Font sizes
FONTSIZE_TITLE = 36
FONTSIZE_LABEL = 24
FONTSIZE_TICK = 20
FONTSIZE_LEGEND = 20
FONTSIZE_ANNOTATION = 18

def generate_topk_example():
    """Generate Top-k numerical example with actual probabilities"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Original vocabulary distribution
    words = ['the', 'cat', 'dog', 'bird', 'sat', 'ran', 'jumped', 'flew', 'walked', 'stood']
    probs = np.array([0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.04, 0.04, 0.02])

    # Left: Full distribution
    ax1.bar(words, probs, color=COLOR_GRAY, edgecolor=COLOR_MAIN, linewidth=2)
    ax1.set_title('Original Distribution', fontsize=FONTSIZE_TITLE, fontweight='bold', color=COLOR_MAIN)
    ax1.set_ylabel('Probability', fontsize=FONTSIZE_LABEL, color=COLOR_MAIN)
    ax1.set_ylim(0, 0.3)
    ax1.tick_params(axis='both', labelsize=FONTSIZE_TICK)
    ax1.grid(True, alpha=0.2, linestyle='--')

    # Add cumulative line
    cumsum = np.cumsum(probs)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(len(words)), cumsum, color=COLOR_ACCENT, linewidth=3,
                  marker='o', markersize=10, label='Cumulative')
    ax1_twin.set_ylabel('Cumulative Probability', fontsize=FONTSIZE_LABEL, color=COLOR_ACCENT)
    ax1_twin.set_ylim(0, 1.1)
    ax1_twin.tick_params(axis='y', labelsize=FONTSIZE_TICK, colors=COLOR_ACCENT)

    # Right: Top-k=3 truncation
    k = 3
    top_k_words = words[:k]
    top_k_probs_original = probs[:k]
    top_k_probs_renorm = top_k_probs_original / np.sum(top_k_probs_original)

    # Show both original and renormalized
    x = np.arange(k)
    width = 0.35

    bars1 = ax2.bar(x - width/2, top_k_probs_original, width,
                    label='Original', color=COLOR_GRAY, edgecolor=COLOR_MAIN, linewidth=2)
    bars2 = ax2.bar(x + width/2, top_k_probs_renorm, width,
                    label='Renormalized', color=COLOR_ACCENT, edgecolor=COLOR_MAIN, linewidth=2)

    ax2.set_title(f'Top-k=3 Truncation', fontsize=FONTSIZE_TITLE, fontweight='bold', color=COLOR_MAIN)
    ax2.set_ylabel('Probability', fontsize=FONTSIZE_LABEL, color=COLOR_MAIN)
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_k_words)
    ax2.tick_params(axis='both', labelsize=FONTSIZE_TICK)
    ax2.legend(fontsize=FONTSIZE_LEGEND, loc='upper right')
    ax2.set_ylim(0, 0.5)
    ax2.grid(True, alpha=0.2, linestyle='--')

    # Add value labels
    for bar, val in zip(bars1, top_k_probs_original):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=FONTSIZE_ANNOTATION)

    for bar, val in zip(bars2, top_k_probs_renorm):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=FONTSIZE_ANNOTATION,
                fontweight='bold', color=COLOR_ACCENT)

    # Add explanation box
    textstr = f'k=3: Keep top 3 words\nSum before: {np.sum(top_k_probs_original):.2f}\nRenormalized to sum to 1.0'
    props = dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT, linewidth=2)
    ax2.text(0.98, 0.98, textstr, transform=ax2.transAxes, fontsize=FONTSIZE_ANNOTATION,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.suptitle('Top-k Sampling: Numerical Example', fontsize=FONTSIZE_TITLE+4,
                 fontweight='bold', color=COLOR_MAIN, y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/topk_example_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: topk_example_bsc.pdf")

def generate_contrastive_vs_nucleus():
    """Direct comparison of contrastive search vs nucleus sampling"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Sample text generations
    context = "The weather today is"

    # Nucleus outputs (diverse but sometimes repetitive)
    nucleus_outputs = [
        "beautiful and sunny with clear skies",
        "perfect for a walk in the park",
        "absolutely gorgeous, gorgeous weather indeed",  # Some repetition
        "nice and warm, very warm actually"  # Some repetition
    ]

    # Contrastive outputs (diverse and non-repetitive)
    contrastive_outputs = [
        "beautiful with clear blue skies",
        "perfect for outdoor activities today",
        "absolutely gorgeous, ideal for hiking",
        "nice and warm with gentle breeze"
    ]

    # Top-left: Method comparison table
    ax1 = axes[0, 0]
    ax1.axis('tight')
    ax1.axis('off')

    comparison_data = [
        ['Aspect', 'Nucleus (Top-p)', 'Contrastive'],
        ['Diversity', 'High', 'High'],
        ['Repetition', 'Medium', 'Very Low'],
        ['Coherence', 'Good', 'Excellent'],
        ['Speed', 'Fast O(V log V)', 'Slower O(k×T²)'],
        ['Parameters', 'p, temperature', 'α, k, penalty'],
        ['Best for', 'General use', 'Long generation']
    ]

    table = ax1.table(cellText=comparison_data,
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.3, 0.35, 0.35])

    table.auto_set_font_size(False)
    table.set_fontsize(FONTSIZE_ANNOTATION)
    table.scale(1.2, 2.0)

    # Color header row
    for i in range(3):
        table[(0, i)].set_facecolor(COLOR_ACCENT)
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color alternating rows
    for i in range(1, 7):
        color = COLOR_LIGHT if i % 2 == 0 else 'white'
        for j in range(3):
            table[(i, j)].set_facecolor(color)

    ax1.set_title('Method Comparison', fontsize=FONTSIZE_TITLE, fontweight='bold', pad=20)

    # Top-right: Repetition comparison
    ax2 = axes[0, 1]
    methods = ['Greedy', 'Beam\nSearch', 'Top-k', 'Nucleus', 'Contrastive']
    repetition_rates = [0.35, 0.25, 0.18, 0.12, 0.05]
    colors = [COLOR_RED, COLOR_ORANGE, COLOR_GRAY, COLOR_BLUE, COLOR_GREEN]

    bars = ax2.bar(methods, repetition_rates, color=colors, edgecolor=COLOR_MAIN, linewidth=2)
    ax2.set_ylabel('Repetition Rate', fontsize=FONTSIZE_LABEL)
    ax2.set_title('Repetition Comparison', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax2.set_ylim(0, 0.4)
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.tick_params(labelsize=FONTSIZE_TICK)

    # Add value labels
    for bar, val in zip(bars, repetition_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.0%}', ha='center', va='bottom', fontsize=FONTSIZE_ANNOTATION, fontweight='bold')

    # Bottom-left: Diversity vs Quality scatter
    ax3 = axes[1, 0]

    # Data points for different methods
    diversity = [0.2, 0.4, 0.7, 0.8, 0.85]
    quality = [0.95, 0.85, 0.7, 0.75, 0.88]

    ax3.scatter(diversity[:4], quality[:4], s=300, c=colors[:4],
               edgecolor=COLOR_MAIN, linewidth=2, alpha=0.7)
    ax3.scatter(diversity[4], quality[4], s=500, c=COLOR_GREEN,
               edgecolor=COLOR_ACCENT, linewidth=4, marker='*', label='Contrastive')

    for i, method in enumerate(methods[:4]):
        ax3.annotate(method.replace('\n', ' '), (diversity[i], quality[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=FONTSIZE_ANNOTATION, fontweight='bold')

    ax3.annotate('Contrastive', (diversity[4], quality[4]),
                xytext=(10, -10), textcoords='offset points',
                fontsize=FONTSIZE_ANNOTATION+2, fontweight='bold', color=COLOR_GREEN,
                arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=2))

    ax3.set_xlabel('Diversity Score', fontsize=FONTSIZE_LABEL)
    ax3.set_ylabel('Quality Score', fontsize=FONTSIZE_LABEL)
    ax3.set_title('Quality-Diversity Tradeoff', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax3.grid(True, alpha=0.2)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0.6, 1)
    ax3.tick_params(labelsize=FONTSIZE_TICK)

    # Add ideal zone
    ideal_zone = plt.Circle((0.8, 0.85), 0.15, color=COLOR_GREEN, alpha=0.1)
    ax3.add_patch(ideal_zone)
    ax3.text(0.8, 0.85, 'Sweet\nSpot', ha='center', va='center',
            fontsize=FONTSIZE_ANNOTATION, fontweight='bold', color=COLOR_GREEN)

    # Bottom-right: Example outputs
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Title
    ax4.text(0.5, 0.95, f'Context: "{context}"', ha='center', va='top',
            fontsize=FONTSIZE_LABEL, fontweight='bold', transform=ax4.transAxes)

    # Nucleus outputs
    ax4.text(0.05, 0.80, 'Nucleus Sampling:', fontsize=FONTSIZE_ANNOTATION+2,
            fontweight='bold', color=COLOR_BLUE, transform=ax4.transAxes)

    y_pos = 0.70
    for output in nucleus_outputs[:2]:
        ax4.text(0.05, y_pos, f'• {output}', fontsize=FONTSIZE_ANNOTATION,
                transform=ax4.transAxes, wrap=True)
        y_pos -= 0.10

    # Contrastive outputs
    ax4.text(0.05, 0.40, 'Contrastive Search:', fontsize=FONTSIZE_ANNOTATION+2,
            fontweight='bold', color=COLOR_GREEN, transform=ax4.transAxes)

    y_pos = 0.30
    for output in contrastive_outputs[:2]:
        ax4.text(0.05, y_pos, f'• {output}', fontsize=FONTSIZE_ANNOTATION,
                transform=ax4.transAxes, wrap=True)
        y_pos -= 0.10

    plt.suptitle('Contrastive Search vs Nucleus Sampling',
                fontsize=FONTSIZE_TITLE+4, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/contrastive_vs_nucleus_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: contrastive_vs_nucleus_bsc.pdf")

def generate_computational_cost_comparison():
    """Generate speed comparison bar chart for all methods"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Method names
    methods = ['Greedy', 'Beam\nSearch', 'Temperature', 'Top-k', 'Nucleus\n(Top-p)', 'Contrastive']

    # Relative computation time (normalized to greedy = 1)
    relative_times = [1.0, 4.5, 1.1, 1.3, 2.2, 8.5]

    # Memory usage (relative)
    memory_usage = [1.0, 4.0, 1.0, 1.0, 1.2, 5.0]

    # Colors based on efficiency
    colors = []
    for t in relative_times:
        if t <= 1.5:
            colors.append(COLOR_GREEN)
        elif t <= 3:
            colors.append(COLOR_ORANGE)
        else:
            colors.append(COLOR_RED)

    # Left: Computation time comparison
    bars1 = ax1.bar(methods, relative_times, color=colors, edgecolor=COLOR_MAIN, linewidth=2)
    ax1.set_ylabel('Relative Computation Time', fontsize=FONTSIZE_LABEL)
    ax1.set_title('Speed Comparison (Relative to Greedy)', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax1.axhline(y=1, color=COLOR_MAIN, linestyle='--', alpha=0.5, label='Greedy baseline')
    ax1.set_ylim(0, 10)
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.tick_params(labelsize=FONTSIZE_TICK)

    # Add value labels
    for bar, val in zip(bars1, relative_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}×', ha='center', va='bottom', fontsize=FONTSIZE_ANNOTATION, fontweight='bold')

    # Add complexity annotations
    complexities = ['O(V·T)', 'O(k·V·T)', 'O(V·T)', 'O(V·T)', 'O(V log V·T)', 'O(k·T²)']
    for i, (bar, comp) in enumerate(zip(bars1, complexities)):
        ax1.text(bar.get_x() + bar.get_width()/2, -0.5,
                comp, ha='center', va='top', fontsize=14, style='italic')

    # Right: Memory usage comparison
    bars2 = ax2.bar(methods, memory_usage, color=[COLOR_GRAY]*6, edgecolor=COLOR_MAIN, linewidth=2)

    # Color based on memory usage
    for bar, mem in zip(bars2, memory_usage):
        if mem <= 1.5:
            bar.set_facecolor(COLOR_GREEN)
        elif mem <= 3:
            bar.set_facecolor(COLOR_ORANGE)
        else:
            bar.set_facecolor(COLOR_RED)

    ax2.set_ylabel('Relative Memory Usage', fontsize=FONTSIZE_LABEL)
    ax2.set_title('Memory Requirements (Relative to Greedy)', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax2.axhline(y=1, color=COLOR_MAIN, linestyle='--', alpha=0.5, label='Greedy baseline')
    ax2.set_ylim(0, 6)
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.tick_params(labelsize=FONTSIZE_TICK)

    # Add value labels
    for bar, val in zip(bars2, memory_usage):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}×', ha='center', va='bottom', fontsize=FONTSIZE_ANNOTATION, fontweight='bold')

    # Add legend for colors
    fast_patch = mpatches.Patch(color=COLOR_GREEN, label='Fast (≤1.5×)')
    medium_patch = mpatches.Patch(color=COLOR_ORANGE, label='Medium (1.5-3×)')
    slow_patch = mpatches.Patch(color=COLOR_RED, label='Slow (>3×)')
    ax1.legend(handles=[fast_patch, medium_patch, slow_patch],
              loc='upper left', fontsize=FONTSIZE_LEGEND)

    # Add note about batch processing
    fig.text(0.5, -0.02,
            'Note: Actual performance depends on implementation, hardware, and batch size. ' +
            'V = vocabulary size, T = sequence length, k = beam width',
            ha='center', fontsize=FONTSIZE_ANNOTATION-2, style='italic', wrap=True)

    plt.suptitle('Computational Cost Analysis', fontsize=FONTSIZE_TITLE+4,
                fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/computational_cost_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: computational_cost_comparison_bsc.pdf")

def generate_production_settings():
    """Generate table showing what real production systems use"""
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('tight')
    ax.axis('off')

    # Production settings data (as of 2025)
    data = [
        ['System/API', 'Default Method', 'Temperature', 'Top-p', 'Top-k', 'Other Parameters', 'Notes'],
        ['GPT-4 API', 'Nucleus', '0.7', '1.0', '—', 'frequency_penalty=0\npresence_penalty=0', 'Can adjust all params'],
        ['Claude API', 'Nucleus', '1.0', '0.999', '—', 'max_tokens required', 'Temperature ∈ [0,1]'],
        ['ChatGPT Web', 'Nucleus+Temp', '0.7', '0.95', '—', 'Not adjustable', 'Optimized for chat'],
        ['Gemini API', 'Top-k + Top-p', '1.0', '0.95', '40', 'candidate_count=1', 'Both k and p used'],
        ['Llama 2 (HF)', 'Configurable', '1.0', '0.9', '50', 'repetition_penalty=1.0', 'Full control'],
        ['Cohere API', 'Nucleus', '0.75', '0.999', '0', 'frequency_penalty=0', 'k=0 means disabled'],
        ['Mistral API', 'Nucleus', '0.7', '1.0', '—', 'safe_mode=false', 'Similar to OpenAI'],
        ['Together AI', 'Configurable', '0.7', '0.7', '50', 'repetition_penalty=1.0', 'Multiple options']
    ]

    # Create table
    table = ax.table(cellText=data,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.15, 0.10, 0.08, 0.08, 0.22, 0.22])

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.0, 2.5)

    # Style the table
    # Header row
    for i in range(7):
        table[(0, i)].set_facecolor(COLOR_ACCENT)
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=16)

    # Color rows by provider type
    provider_colors = {
        1: COLOR_BLUE,     # OpenAI
        2: COLOR_GREEN,    # Anthropic
        3: COLOR_BLUE,     # OpenAI
        4: COLOR_ORANGE,   # Google
        5: COLOR_GRAY,     # Meta
        6: COLOR_RED,      # Cohere
        7: COLOR_GRAY,     # Mistral
        8: COLOR_GRAY      # Together
    }

    for row_idx in range(1, 9):
        # Alternating row colors
        bg_color = COLOR_LIGHT if row_idx % 2 == 0 else 'white'
        for col_idx in range(7):
            table[(row_idx, col_idx)].set_facecolor(bg_color)

        # Provider name coloring
        table[(row_idx, 0)].set_text_props(weight='bold')

    # Highlight important cells
    # Temperature column - color code values
    for row_idx in range(1, 9):
        temp_cell = table[(row_idx, 2)]
        temp_val = data[row_idx][2]
        if temp_val == '0.7':
            temp_cell.set_facecolor('#E8F5E9')  # Light green
        elif temp_val == '1.0':
            temp_cell.set_facecolor('#FFF3E0')  # Light orange

    # Add title
    ax.text(0.5, 1.05, 'Production System Settings (2025)',
           transform=ax.transAxes, ha='center', fontsize=FONTSIZE_TITLE+4,
           fontweight='bold', color=COLOR_MAIN)

    # Add subtitle
    ax.text(0.5, 1.01, 'Default decoding parameters used by major LLM APIs and platforms',
           transform=ax.transAxes, ha='center', fontsize=FONTSIZE_LABEL,
           style='italic', color=COLOR_GRAY)

    # Add legend/notes
    notes = [
        "• Temperature: Lower = more deterministic, Higher = more creative",
        "• Top-p (Nucleus): Cumulative probability threshold (typically 0.9-1.0)",
        "• Top-k: Number of top tokens to consider (often 40-50 when used)",
        "• Most production systems use Nucleus sampling as default",
        "• ChatGPT and Claude optimize for conversational quality",
        "• APIs generally allow full parameter customization"
    ]

    y_pos = -0.05
    for note in notes:
        ax.text(0.05, y_pos, note, transform=ax.transAxes,
               fontsize=14, color=COLOR_MAIN)
        y_pos -= 0.03

    # Add color legend for provider types
    ax.text(0.95, -0.05, 'Provider Types:', transform=ax.transAxes,
           fontsize=14, fontweight='bold', ha='right')

    providers = [
        ('Proprietary LLM', COLOR_BLUE),
        ('Open Source', COLOR_GRAY),
        ('Optimized for Chat', COLOR_GREEN)
    ]

    y_pos = -0.08
    for label, color in providers:
        ax.add_patch(plt.Rectangle((0.82, y_pos-0.003), 0.02, 0.015,
                                  transform=ax.transAxes, facecolor=color))
        ax.text(0.85, y_pos, label, transform=ax.transAxes,
               fontsize=12, va='bottom')
        y_pos -= 0.03

    plt.tight_layout()
    plt.savefig('../figures/production_settings_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: production_settings_bsc.pdf")

def main():
    """Generate all missing charts"""
    print("Generating 4 missing charts for Week 9 Decoding...")
    print("-" * 50)

    # Generate each chart
    generate_topk_example()
    generate_contrastive_vs_nucleus()
    generate_computational_cost_comparison()
    generate_production_settings()

    print("-" * 50)
    print("✓ All 4 missing charts generated successfully!")
    print("Charts saved to ../figures/ directory")

if __name__ == '__main__':
    main()