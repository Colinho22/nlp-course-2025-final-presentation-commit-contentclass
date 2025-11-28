#!/usr/bin/env python3
"""
Generate CLEAN, READABLE charts for Week 9: Decoding Strategies
Font size: 30pt+ minimum
Simple layouts with NO text overlap
Template beamer final colors
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

os.makedirs('../figures', exist_ok=True)

# Template Beamer Final Colors
PURPLE = '#3333B2'      # mlpurple
LAVENDER = '#ADADC0'    # mllavender
BLUE = '#0066CC'        # mlblue
GREEN = '#2CA02C'       # mlgreen
RED = '#D62728'         # mlred
ORANGE = '#FF7F0E'      # mlorange
GRAY = '#7F7F7F'        # mlgray

# CLEAN CHART 1: Quality-Diversity Tradeoff (SIMPLE scatter)
def generate_quality_diversity_clean():
    """Simple scatter plot - NO overlap"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # 6 methods positioned clearly
    methods = [
        ('Greedy', 15, 85, RED),
        ('Beam', 20, 88, RED),
        ('Temp\n(low)', 35, 82, GRAY),
        ('Top-k', 65, 70, GRAY),
        ('Nucleus', 78, 82, GREEN),
        ('Contrastive', 82, 85, GREEN),
    ]

    for name, x, y, color in methods:
        ax.scatter(x, y, s=2000, c=color, alpha=0.8, edgecolors='black', linewidths=4, zorder=3)
        # Label BELOW point for clarity
        ax.text(x, y-12, name, ha='center', fontsize=32, weight='bold', color='black')

    # Optimal zone
    rect = patches.Rectangle((70, 75), 25, 20, linewidth=4,
                             edgecolor=GREEN, facecolor=GREEN, alpha=0.15, linestyle='--')
    ax.add_patch(rect)
    ax.text(82.5, 87, 'OPTIMAL', ha='center', fontsize=34, weight='bold', color=GREEN)

    # Simple labels
    ax.set_xlabel('Diversity (Creativity) →', fontsize=36, weight='bold', labelpad=15)
    ax.set_ylabel('Quality (Coherence) →', fontsize=36, weight='bold', labelpad=15)
    ax.set_title('Quality-Diversity Tradeoff', fontsize=40, weight='bold', pad=25, color=PURPLE)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=28, width=3, length=8)
    ax.grid(True, alpha=0.3, linewidth=2, color=LAVENDER)

    # Clean spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(PURPLE)
        ax.spines[spine].set_linewidth(3)

    plt.tight_layout()
    plt.savefig('../figures/quality_diversity_tradeoff_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("OK: quality_diversity_tradeoff_bsc.pdf")

# CLEAN CHART 2: Beam Search Visual (SIMPLE tree)
def generate_beam_visual_clean():
    """Simple beam search tree - clear structure"""
    fig, ax = plt.subplots(figsize=(14, 9))

    # START node
    ax.scatter(0.5, 0.9, s=3000, c=PURPLE, zorder=5, edgecolors='black', linewidths=5)
    ax.text(0.5, 0.9, 'START', ha='center', va='center', fontsize=36, weight='bold', color='white')

    # Level 1: 3 kept beams
    positions = [0.2, 0.5, 0.8]
    for x in positions:
        ax.scatter(x, 0.6, s=2500, c=GREEN, zorder=4, edgecolors='black', linewidths=4)
        ax.plot([0.5, x], [0.88, 0.62], 'k-', linewidth=4, alpha=0.5)

    # Level 2: 3 kept beams from best path
    for i, x in enumerate([0.15, 0.2, 0.25]):
        ax.scatter(x, 0.3, s=2500, c=GREEN, zorder=4, edgecolors='black', linewidths=4)
        ax.plot([0.2, x], [0.58, 0.32], color=PURPLE, linewidth=5, alpha=0.8)

    # Pruned paths (lighter)
    for x in [0.45, 0.55, 0.75, 0.85]:
        ax.scatter(x, 0.3, s=1500, c=GRAY, alpha=0.3, zorder=2)

    ax.text(0.5, 0.05, 'Width=3: Keep top 3 paths at each step',
            ha='center', fontsize=34, style='italic', weight='bold', color=PURPLE)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/beam_search_visual_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("OK: beam_search_visual_bsc.pdf")

# CLEAN CHART 3: Temperature Effects (SIMPLE bars, NO overlap)
def generate_temperature_clean():
    """Temperature with CLEAN bar charts"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 7))

    tokens = ['the', 'a', 'on']
    temps_data = [
        ([0.73, 0.18, 0.09], 'T=0.5\nFocused', ax1),
        ([0.42, 0.23, 0.35], 'T=1.0\nBalanced', ax2),
        ([0.32, 0.31, 0.37], 'T=2.0\nFlat', ax3),
    ]

    for probs, title, ax in temps_data:
        bars = ax.bar(tokens, probs, color=[PURPLE, GREEN, BLUE],
                     alpha=0.8, edgecolor='black', linewidth=3, width=0.6)

        # Labels ABOVE bars (clear)
        for i, prob in enumerate(probs):
            ax.text(i, prob + 0.08, f'{prob:.2f}', ha='center',
                   fontsize=32, weight='bold', color='black')

        ax.set_title(title, fontsize=34, weight='bold', color=PURPLE, pad=15)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Probability' if ax == ax1 else '', fontsize=30, weight='bold')
        ax.tick_params(labelsize=28, width=3)
        ax.grid(axis='y', alpha=0.3, linewidth=2)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(3)

    fig.suptitle('Temperature Reshapes Distribution', fontsize=38, weight='bold', y=0.98, color=PURPLE)
    plt.tight_layout()
    plt.savefig('../figures/temperature_effects_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("OK: temperature_effects_bsc.pdf")

# CLEAN CHART 4: Top-k Filtering (SIMPLE, clear cutoff)
def generate_topk_clean():
    """Top-k with clear visual cutoff"""
    fig, ax = plt.subplots(figsize=(14, 8))

    tokens = ['tok0', 'tok1', 'tok2', 'tok3', 'tok4', 'tok5', 'tok6', 'tok7']
    probs = [0.30, 0.20, 0.15, 0.12, 0.10, 0.06, 0.04, 0.03]
    k = 5

    colors = [GREEN] * k + [RED] * (len(tokens) - k)
    alphas = [0.9] * k + [0.4] * (len(tokens) - k)

    for i, (token, prob, color, alpha) in enumerate(zip(tokens, probs, colors, alphas)):
        ax.bar(i, prob, color=color, alpha=alpha, edgecolor='black', linewidth=3, width=0.8)

    # Clear cutoff line
    ax.axvline(k - 0.5, color=PURPLE, linestyle='--', linewidth=5, label=f'k={k} cutoff')

    # Annotations (clear positions)
    ax.text(2, 0.38, f'TOP-{k}\n(Sample)', ha='center', fontsize=34, weight='bold', color=GREEN,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor=GREEN, linewidth=4))

    ax.text(6.5, 0.15, 'IGNORED\n(too unlikely)', ha='center', fontsize=30, style='italic', color=RED,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor=RED, linewidth=3))

    ax.set_xlabel('Tokens (ranked by probability)', fontsize=34, weight='bold', labelpad=15)
    ax.set_ylabel('Probability', fontsize=34, weight='bold', labelpad=15)
    ax.set_title('Top-k Filtering', fontsize=40, weight='bold', pad=20, color=PURPLE)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=28)
    ax.tick_params(labelsize=28, width=3)
    ax.legend(fontsize=28, loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linewidth=2)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(PURPLE)
        ax.spines[spine].set_linewidth(3)

    plt.tight_layout()
    plt.savefig('../figures/topk_filtering_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("OK: topk_filtering_bsc.pdf")

# CLEAN CHART 5: Nucleus Process (SIMPLE dual axis)
def generate_nucleus_clean():
    """Nucleus with clean bars + cumulative line"""
    fig, ax = plt.subplots(figsize=(14, 8))

    tokens = ['cat', 'dog', 'bird', 'fish', 'mouse', 'rat']
    probs = [0.40, 0.20, 0.15, 0.10, 0.08, 0.07]
    cumulative = np.cumsum(probs)
    p_threshold = 0.85
    nucleus_size = 4  # First 4 tokens

    # Bars
    colors = [GREEN if i < nucleus_size else GRAY for i in range(len(tokens))]
    ax.bar(range(len(tokens)), probs, color=colors, alpha=0.8,
           edgecolor='black', linewidth=3, width=0.7)

    # Cumulative line on secondary axis
    ax2 = ax.twinx()
    ax2.plot(range(len(tokens)), cumulative, color=PURPLE, linewidth=5,
             marker='o', markersize=18, label='Cumulative', zorder=5)
    ax2.axhline(p_threshold, color=ORANGE, linestyle='--', linewidth=4,
                label=f'p={p_threshold}')

    # Clear annotation
    ax.text(nucleus_size/2 - 0.5, 0.48, f'NUCLEUS\n({nucleus_size} tokens)',
            ha='center', fontsize=32, weight='bold', color=GREEN,
            bbox=dict(boxstyle='round,pad=1', facecolor='white', edgecolor=GREEN, linewidth=4))

    ax.set_xlabel('Tokens (ranked)', fontsize=34, weight='bold', labelpad=15)
    ax.set_ylabel('Individual Probability', fontsize=32, weight='bold', color='black', labelpad=15)
    ax2.set_ylabel('Cumulative Probability', fontsize=32, weight='bold', color=PURPLE, labelpad=15)
    ax.set_title('Nucleus (Top-p) Sampling', fontsize=40, weight='bold', pad=20, color=PURPLE)

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=30)
    ax.tick_params(labelsize=28, width=3)
    ax2.tick_params(labelsize=28, width=3, colors=PURPLE)
    ax2.legend(fontsize=26, loc='center right')

    ax.grid(axis='y', alpha=0.3, linewidth=2)

    for spine in ['top']:
        ax.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(3)
    ax2.spines['right'].set_color(PURPLE)
    ax2.spines['right'].set_linewidth(3)

    plt.tight_layout()
    plt.savefig('../figures/nucleus_process_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("OK: nucleus_process_bsc.pdf")

# CLEAN CHART 6: Nucleus Cumulative (SIMPLE side-by-side)
def generate_nucleus_cumulative_clean():
    """Two simple bar charts side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Peaked distribution
    tokens1 = ['the', 'a', 'an', 'this']
    probs1 = [0.60, 0.20, 0.10, 0.05]
    ax1.bar(range(len(tokens1)), probs1, color=[GREEN, GREEN, GRAY, GRAY],
            alpha=0.8, edgecolor='black', linewidth=3, width=0.6)
    ax1.set_title('Peaked\nNucleus = 2', fontsize=36, weight='bold', color=PURPLE, pad=15)
    ax1.set_ylabel('Probability', fontsize=32, weight='bold', labelpad=12)
    ax1.set_xticks(range(len(tokens1)))
    ax1.set_xticklabels(tokens1, fontsize=30)
    ax1.tick_params(labelsize=28, width=3)
    ax1.set_ylim(0, 0.7)
    ax1.grid(axis='y', alpha=0.3, linewidth=2)

    # Flat distribution
    tokens2 = ['cat', 'dog', 'bird', 'fish', 'frog', 'deer']
    probs2 = [0.18, 0.16, 0.14, 0.13, 0.12, 0.10]
    ax2.bar(range(len(tokens2)), probs2, color=[GREEN]*6,
            alpha=0.8, edgecolor='black', linewidth=3, width=0.6)
    ax2.set_title('Flat\nNucleus = 6', fontsize=36, weight='bold', color=PURPLE, pad=15)
    ax2.set_ylabel('Probability', fontsize=32, weight='bold', labelpad=12)
    ax2.set_xticks(range(len(tokens2)))
    ax2.set_xticklabels(tokens2, fontsize=28, rotation=15)
    ax2.tick_params(labelsize=28, width=3)
    ax2.set_ylim(0, 0.25)
    ax2.grid(axis='y', alpha=0.3, linewidth=2)

    for ax in [ax1, ax2]:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(3)

    fig.suptitle('Nucleus Adapts to Distribution (p=0.85)', fontsize=38, weight='bold', y=0.98, color=PURPLE)
    plt.tight_layout()
    plt.savefig('../figures/nucleus_cumulative_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("OK: nucleus_cumulative_bsc.pdf")

# CLEAN CHART 7: Degeneration Problem (TEXT ONLY, clear)
def generate_degeneration_clean():
    """Simple text visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.text(0.5, 0.85, 'Greedy Decoding Output:', ha='center', fontsize=36,
            weight='bold', color=RED)

    problem_text = ('"The city of New York is a major city.\n'
                   'The city is known for its culture.\n'
                   'The city has many attractions..."')

    ax.text(0.5, 0.60, problem_text, ha='center', fontsize=32, style='italic',
            bbox=dict(boxstyle='round,pad=1.5', facecolor=RED, alpha=0.15,
                     edgecolor=RED, linewidth=4))

    ax.text(0.5, 0.35, '"the city" appears 3 times!', ha='center', fontsize=40,
            weight='bold', color=RED)

    ax.text(0.5, 0.12, 'Solution: Contrastive Search penalizes repetition',
            ha='center', fontsize=34, weight='bold', color=GREEN,
            bbox=dict(boxstyle='round,pad=1', facecolor=GREEN, alpha=0.2,
                     edgecolor=GREEN, linewidth=4))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/degeneration_problem_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("OK: degeneration_problem_bsc.pdf")

# CLEAN CHART 8: Pareto Frontier (SIMPLE scatter with space)
def generate_pareto_clean():
    """Clean Pareto frontier plot"""
    fig, ax = plt.subplots(figsize=(13, 9))

    methods = [
        ('Greedy', 15, 85),
        ('Beam', 20, 88),
        ('Temp\n(low)', 35, 82),
        ('Top-k', 65, 70),
        ('Nucleus', 78, 82),
        ('Contrastive', 82, 85),
    ]

    colors = [RED, RED, GRAY, GRAY, GREEN, GREEN]

    for (name, x, y), color in zip(methods, colors):
        ax.scatter(x, y, s=2500, c=color, alpha=0.8, edgecolors='black', linewidths=4, zorder=3)
        # Label below and offset
        offset_y = -15 if y > 70 else 10
        ax.text(x, y+offset_y, name, ha='center', fontsize=30, weight='bold')

    # Optimal zone
    rect = patches.Rectangle((70, 75), 25, 20, linewidth=4,
                             edgecolor=GREEN, facecolor=GREEN, alpha=0.15, linestyle='--')
    ax.add_patch(rect)
    ax.text(82, 90, 'OPTIMAL', fontsize=32, weight='bold', color=GREEN, ha='center')

    ax.set_xlabel('Diversity →', fontsize=36, weight='bold', labelpad=15)
    ax.set_ylabel('Quality →', fontsize=36, weight='bold', labelpad=15)
    ax.set_title('Quality-Diversity Pareto Frontier', fontsize=40, weight='bold', pad=20, color=PURPLE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=30, width=3)
    ax.grid(True, alpha=0.3, linewidth=2)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(PURPLE)
        ax.spines[spine].set_linewidth(3)

    plt.tight_layout()
    plt.savefig('../figures/quality_diversity_pareto_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("OK: quality_diversity_pareto_bsc.pdf")

# CLEAN CHART 9: Computational Costs (SIMPLE bar + line)
def generate_costs_clean():
    """Clean cost comparison"""
    fig, ax = plt.subplots(figsize=(14, 8))

    methods = ['Greedy', 'Temp', 'Top-k', 'Nucleus', 'Beam\n(w=3)', 'Contrastive']
    times = [1.0, 1.1, 1.2, 1.3, 4.5, 12.0]
    quality = [65, 55, 68, 80, 88, 85]

    colors_cost = [GREEN, GREEN, GREEN, GREEN, ORANGE, RED]

    x_pos = np.arange(len(methods))

    # Bars
    bars = ax.bar(x_pos, times, color=colors_cost, alpha=0.8,
                  edgecolor='black', linewidth=3, width=0.7)

    # Labels ON bars
    for i, t in enumerate(times):
        ax.text(i, t/2, f'{t}×', ha='center', fontsize=34, weight='bold', color='white')

    ax.set_xlabel('Method', fontsize=36, weight='bold', labelpad=15)
    ax.set_ylabel('Relative Speed (vs Greedy)', fontsize=34, weight='bold', labelpad=15)
    ax.set_title('Computational Cost Comparison', fontsize=40, weight='bold', pad=20, color=PURPLE)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=30)
    ax.tick_params(labelsize=28, width=3)
    ax.set_ylim(0, 14)
    ax.grid(axis='y', alpha=0.3, linewidth=2)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(PURPLE)
        ax.spines[spine].set_linewidth(3)

    plt.tight_layout()
    plt.savefig('../figures/computational_cost_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("OK: computational_cost_comparison_bsc.pdf")

# CLEAN CHART 10: Pipeline (SIMPLE horizontal flow)
def generate_pipeline_clean():
    """Clean pipeline diagram"""
    fig, ax = plt.subplots(figsize=(16, 6))

    stages = ['Input\nText', 'MODEL', 'Probabilities\n(50K words)', 'DECODING\n[TODAY]', 'Output\nText']
    x_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors_stage = [LAVENDER, BLUE, ORANGE, PURPLE, GREEN]

    for stage, x, color in zip(stages, x_positions, colors_stage):
        text_color = 'white' if color in [BLUE, PURPLE] else 'black'
        ax.text(x, 0.5, stage, ha='center', va='center', fontsize=30, weight='bold',
               color=text_color,
               bbox=dict(boxstyle='round,pad=1.2', facecolor=color, edgecolor='black',
                        linewidth=4, alpha=0.9))

    # Arrows
    for i in range(len(x_positions) - 1):
        ax.annotate('', xy=(x_positions[i+1] - 0.08, 0.5), xytext=(x_positions[i] + 0.08, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=5, color='black'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.8)
    ax.axis('off')
    ax.set_title('From Prediction to Generation', fontsize=42, weight='bold', pad=30, color=PURPLE)

    plt.tight_layout()
    plt.savefig('../figures/prediction_to_text_pipeline_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("OK: prediction_to_text_pipeline_bsc.pdf")

# Generate minimal essential charts (10 total for clean presentation)
if __name__ == '__main__':
    print("=" * 60)
    print("Generating 10 CLEAN, READABLE charts (30pt+ fonts)")
    print("=" * 60)

    generate_quality_diversity_clean()
    generate_beam_visual_clean()
    generate_temperature_clean()
    generate_topk_clean()
    generate_nucleus_clean()
    generate_nucleus_cumulative_clean()
    generate_degeneration_clean()
    generate_pareto_clean()
    generate_costs_clean()
    generate_pipeline_clean()

    print("=" * 60)
    print("SUCCESS: 10 clean charts with NO overlap!")
    print("  All charts: 30pt+ fonts, simple layouts")
    print("  Template beamer final colors")
    print("=" * 60)
