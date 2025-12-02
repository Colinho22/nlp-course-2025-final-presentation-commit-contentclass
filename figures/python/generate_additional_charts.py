"""
Generate additional charts for NLP Final Lecture (Revision 3)
12 new visualizations across RAG, Reasoning, and Alignment sections
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np
import os

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Color scheme (consistent with existing charts)
COLORS = {
    'mlblue': '#0066CC',
    'mlpurple': '#3333B2',
    'mlorange': '#FF7F0E',
    'mlgreen': '#2CA02C',
    'mlred': '#D62728',
    'mlgray': '#7F7F7F',
    'lightblue': '#C1C1E8',
    'lightgreen': '#E8F5E9',
    'lightred': '#FFEBEE',
    'lightorange': '#FFF3E0'
}

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def save_chart(fig, filename):
    """Save chart to PDF"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {filepath}")


# =============================================================================
# RAG/RETRIEVAL CHARTS (4)
# =============================================================================

def create_embedding_space_2d():
    """Chart 1: 2D embedding space visualization"""
    fig, ax = plt.subplots(figsize=(10, 8))

    np.random.seed(42)

    # Query point
    query = np.array([0.5, 0.6])

    # Document clusters
    # Cluster 1: Relevant documents (near query)
    relevant = np.random.randn(8, 2) * 0.15 + [0.55, 0.55]
    # Cluster 2: Somewhat relevant
    somewhat = np.random.randn(6, 2) * 0.12 + [0.3, 0.7]
    # Cluster 3: Irrelevant documents
    irrelevant1 = np.random.randn(10, 2) * 0.18 + [-0.3, -0.2]
    irrelevant2 = np.random.randn(8, 2) * 0.15 + [0.8, -0.3]

    # Plot irrelevant first (background)
    ax.scatter(irrelevant1[:, 0], irrelevant1[:, 1], c=COLORS['mlgray'],
               s=100, alpha=0.5, label='Irrelevant docs', marker='o')
    ax.scatter(irrelevant2[:, 0], irrelevant2[:, 1], c=COLORS['mlgray'],
               s=100, alpha=0.5, marker='o')

    # Plot somewhat relevant
    ax.scatter(somewhat[:, 0], somewhat[:, 1], c=COLORS['mlorange'],
               s=120, alpha=0.7, label='Partial match', marker='s')

    # Plot relevant
    ax.scatter(relevant[:, 0], relevant[:, 1], c=COLORS['mlgreen'],
               s=150, alpha=0.8, label='Relevant docs', marker='^')

    # Plot query (large, prominent)
    ax.scatter(query[0], query[1], c=COLORS['mlred'], s=400, marker='*',
               label='Query', zorder=5, edgecolors='black', linewidths=1)

    # Draw top-k boundary circle
    circle1 = plt.Circle(query, 0.25, fill=False, color=COLORS['mlpurple'],
                         linestyle='--', linewidth=2, label='Top-3 boundary')
    circle2 = plt.Circle(query, 0.45, fill=False, color=COLORS['mlblue'],
                         linestyle=':', linewidth=1.5, label='Top-10 boundary')
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Labels
    ax.set_xlabel('Embedding Dimension 1 (after PCA)', fontsize=12)
    ax.set_ylabel('Embedding Dimension 2 (after PCA)', fontsize=12)
    ax.set_title('Embedding Space: Query-Document Similarity', fontsize=14, fontweight='bold')

    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(-0.8, 1.2)
    ax.set_ylim(-0.8, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('Dense retrieval finds\nsemantically similar docs',
                xy=(0.6, 0.45), fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    save_chart(fig, 'embedding_space_2d.pdf')


def create_bm25_vs_dense():
    """Chart 2: BM25 vs Dense retrieval comparison"""
    fig, ax = plt.subplots(figsize=(12, 7))

    scenarios = ['Exact\nKeyword', 'Synonyms', 'Semantic\nSimilarity', 'Rare\nTerms', 'Long\nQueries']
    bm25_scores = [95, 45, 30, 85, 50]
    dense_scores = [70, 85, 95, 40, 80]

    x = np.arange(len(scenarios))
    width = 0.35

    bars1 = ax.bar(x - width/2, bm25_scores, width, label='BM25 (Sparse)',
                   color=COLORS['mlorange'], edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, dense_scores, width, label='Dense Retrieval',
                   color=COLORS['mlblue'], edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bar, score in zip(bars1, bm25_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{score}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, score in zip(bars2, dense_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{score}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Winner indicators
    for i, (b, d) in enumerate(zip(bm25_scores, dense_scores)):
        winner_x = x[i] - width/2 if b > d else x[i] + width/2
        ax.annotate('*', xy=(winner_x, max(b, d) + 8), fontsize=20,
                   ha='center', color=COLORS['mlgreen'], fontweight='bold')

    ax.set_ylabel('Retrieval Accuracy (%)', fontsize=12)
    ax.set_title('BM25 vs Dense Retrieval: When Each Method Wins', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    # Add recommendation box
    textstr = 'Best Practice: Use Hybrid Search\n(BM25 + Dense with fusion)'
    props = dict(boxstyle='round', facecolor=COLORS['lightgreen'], alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.grid(True, axis='y', alpha=0.3)

    save_chart(fig, 'bm25_vs_dense.pdf')


def create_chunk_size_tradeoff():
    """Chart 3: Chunk size vs precision/recall trade-off"""
    fig, ax = plt.subplots(figsize=(10, 7))

    chunk_sizes = [128, 256, 512, 1024, 2048]
    precision = [92, 85, 78, 65, 52]
    recall = [45, 62, 78, 88, 95]
    f1 = [2 * p * r / (p + r) for p, r in zip(precision, recall)]

    ax.plot(chunk_sizes, precision, 'o-', color=COLORS['mlblue'], linewidth=2.5,
            markersize=10, label='Precision', markeredgecolor='black')
    ax.plot(chunk_sizes, recall, 's-', color=COLORS['mlgreen'], linewidth=2.5,
            markersize=10, label='Recall', markeredgecolor='black')
    ax.plot(chunk_sizes, f1, '^--', color=COLORS['mlpurple'], linewidth=2,
            markersize=9, label='F1 Score', alpha=0.8)

    # Highlight optimal zone
    ax.axvspan(400, 650, alpha=0.2, color=COLORS['mlorange'], label='Optimal Zone')

    # Add crossover annotation
    ax.annotate('Crossover Point\n(512 tokens)', xy=(512, 78), xytext=(700, 85),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white'))

    ax.set_xlabel('Chunk Size (tokens)', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Chunking Strategy: Precision-Recall Trade-off', fontsize=14, fontweight='bold')
    ax.legend(loc='center right', fontsize=11)
    ax.set_xscale('log', base=2)
    ax.set_xticks(chunk_sizes)
    ax.set_xticklabels(chunk_sizes)
    ax.set_ylim(40, 100)
    ax.grid(True, alpha=0.3)

    # Add guidance text
    ax.text(150, 48, 'Small chunks:\nPrecise but\nmissing context', fontsize=9,
            ha='center', style='italic')
    ax.text(1800, 48, 'Large chunks:\nMore context but\nnoisy retrieval', fontsize=9,
            ha='center', style='italic')

    save_chart(fig, 'chunk_size_tradeoff.pdf')


# =============================================================================
# REASONING/COT CHARTS (4)
# =============================================================================

def create_cot_accuracy_gains():
    """Chart 5: CoT accuracy improvements bar chart"""
    fig, ax = plt.subplots(figsize=(11, 7))

    tasks = ['GSM8K\n(Math)', 'MATH\n(Olympiad)', 'HumanEval\n(Code)', 'LogiQA\n(Logic)']
    without_cot = [18, 8, 28, 35]
    with_cot = [58, 33, 48, 55]

    x = np.arange(len(tasks))
    width = 0.35

    bars1 = ax.bar(x - width/2, without_cot, width, label='Standard Prompting',
                   color=COLORS['mlgray'], edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, with_cot, width, label='Chain-of-Thought',
                   color=COLORS['mlgreen'], edgecolor='black', linewidth=1)

    # Add value labels
    for bar, val in zip(bars1, without_cot):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, with_cot):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add improvement arrows
    for i, (w, wo) in enumerate(zip(with_cot, without_cot)):
        improvement = w - wo
        ax.annotate(f'+{improvement}%', xy=(x[i] + width/2, w + 5),
                   fontsize=11, ha='center', color=COLORS['mlred'], fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Chain-of-Thought: Accuracy Gains Across Tasks (PaLM 540B)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_ylim(0, 75)
    ax.grid(True, axis='y', alpha=0.3)

    # Citation
    ax.text(0.98, 0.02, 'Based on Wei et al. (2022)', transform=ax.transAxes,
            fontsize=8, ha='right', style='italic', alpha=0.7)

    save_chart(fig, 'cot_accuracy_gains.pdf')


def create_self_consistency_voting():
    """Chart 6: Self-consistency voting diagram"""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Question box at top
    question_box = FancyBboxPatch((3, 8.5), 4, 1, boxstyle="round,pad=0.1",
                                   facecolor=COLORS['lightblue'], edgecolor=COLORS['mlpurple'], linewidth=2)
    ax.add_patch(question_box)
    ax.text(5, 9, 'Question: "What is 17 + 38?"', ha='center', va='center', fontsize=11, fontweight='bold')

    # Reasoning chains (5 paths)
    chain_positions = [(1, 5.5), (3, 5.5), (5, 5.5), (7, 5.5), (9, 5.5)]
    chain_texts = [
        '17+38\n=17+40-2\n=57-2\n=55',
        '17+38\n=20+35\n=55',
        '17+38\n=10+7+38\n=10+45\n=55',
        '17+38\n=15+40\n=55',
        '17+38\n=50+5\n=56'  # Wrong answer for demonstration
    ]
    answers = ['55', '55', '55', '55', '56']
    colors = [COLORS['mlgreen']] * 4 + [COLORS['mlred']]

    for i, ((x, y), text, ans, col) in enumerate(zip(chain_positions, chain_texts, answers, colors)):
        # Chain box
        box = FancyBboxPatch((x-0.8, y-1.5), 1.6, 3, boxstyle="round,pad=0.05",
                             facecolor='white', edgecolor=col, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, family='monospace')

        # Answer circle
        circle = plt.Circle((x, y-2.3), 0.4, facecolor=col, edgecolor='black', linewidth=1)
        ax.add_patch(circle)
        ax.text(x, y-2.3, ans, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Arrow from question
        ax.annotate('', xy=(x, y+1.5), xytext=(5, 8.5),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))

        # Label
        ax.text(x, y+1.7, f'Chain {i+1}', ha='center', fontsize=8, alpha=0.7)

    # Voting box
    vote_box = FancyBboxPatch((3.5, 0.5), 3, 1.2, boxstyle="round,pad=0.1",
                               facecolor=COLORS['lightgreen'], edgecolor=COLORS['mlgreen'], linewidth=2)
    ax.add_patch(vote_box)
    ax.text(5, 1.1, 'Majority Vote: 55 (4/5)', ha='center', va='center', fontsize=12, fontweight='bold')

    # Arrows to voting
    for x, _ in chain_positions:
        ax.annotate('', xy=(5, 1.7), xytext=(x, 2.8),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    ax.set_title('Self-Consistency: Multiple Reasoning Paths with Majority Voting',
                fontsize=14, fontweight='bold', pad=20)

    save_chart(fig, 'self_consistency_voting.pdf')


def create_deepseek_r1_pipeline():
    """Chart 7: DeepSeek-R1 4-stage training pipeline"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Stage boxes
    stages = [
        (1, 4, 'Stage 1:\nCold Start', 'Optional SFT on\nreasoning format\n<think>...</think>', COLORS['lightblue']),
        (4.5, 4, 'Stage 2:\nReasoning RL', 'GRPO with\nrule-based rewards\n(accuracy + format)', COLORS['lightorange']),
        (8, 4, 'Stage 3:\nRejection Sampling', 'Generate many\nFilter for quality\nCreate dataset', COLORS['lightgreen']),
        (11.5, 4, 'Stage 4:\nFinal SFT', 'Fine-tune on\ncurated data\nAdd general skills', COLORS['lightblue'])
    ]

    for x, y, title, desc, color in stages:
        box = FancyBboxPatch((x-1, y-1.5), 2.5, 3, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor=COLORS['mlpurple'], linewidth=2)
        ax.add_patch(box)
        ax.text(x+0.25, y+0.8, title, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x+0.25, y-0.3, desc, ha='center', va='center', fontsize=8)

    # Arrows between stages
    arrow_style = dict(arrowstyle='->', color=COLORS['mlpurple'], lw=2)
    for i in range(3):
        x_start = stages[i][0] + 1.5
        x_end = stages[i+1][0] - 1
        ax.annotate('', xy=(x_end, 4), xytext=(x_start, 4), arrowprops=arrow_style)

    # Input/Output labels
    ax.text(0, 4, 'Base\nModel', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    ax.annotate('', xy=(0.5, 4), xytext=(0.3, 4), arrowprops=dict(arrowstyle='->', color='gray'))

    ax.text(14, 4, 'DeepSeek\nR1', ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLORS['mlgreen'], edgecolor='black'))
    ax.annotate('', xy=(13.7, 4), xytext=(13.2, 4), arrowprops=dict(arrowstyle='->', color='gray'))

    # Key insight box
    insight_box = FancyBboxPatch((4, 0.5), 6, 1.5, boxstyle="round,pad=0.1",
                                  facecolor='white', edgecolor=COLORS['mlred'], linewidth=2)
    ax.add_patch(insight_box)
    ax.text(7, 1.25, 'Key Finding: Stage 2 alone (R1-Zero) discovers reasoning!\nNo supervised demonstrations needed.',
            ha='center', va='center', fontsize=10, style='italic')

    ax.set_title('DeepSeek-R1: Training Pipeline', fontsize=14, fontweight='bold', pad=20)

    save_chart(fig, 'deepseek_r1_pipeline.pdf')


def create_inference_scaling_curve():
    """Chart 8: Inference tokens vs accuracy (test-time scaling)"""
    fig, ax = plt.subplots(figsize=(10, 7))

    tokens = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000])

    # Different model behaviors
    gpt4 = 72 + 3 * np.log10(tokens/50)  # Nearly flat
    o1_mini = 50 + 20 * np.log10(tokens/50)  # Moderate scaling
    o1 = 55 + 25 * np.log10(tokens/50)  # Strong scaling

    ax.semilogx(tokens, gpt4, 'o-', color=COLORS['mlgray'], linewidth=2.5,
                markersize=8, label='GPT-4 (fixed reasoning)')
    ax.semilogx(tokens, o1_mini, 's-', color=COLORS['mlorange'], linewidth=2.5,
                markersize=8, label='o1-mini')
    ax.semilogx(tokens, o1, '^-', color=COLORS['mlgreen'], linewidth=2.5,
                markersize=8, label='o1')

    # Shade diminishing returns region
    ax.axvspan(5000, 10000, alpha=0.1, color='gray')
    ax.text(7000, 60, 'Diminishing\nreturns', fontsize=9, ha='center', style='italic', alpha=0.7)

    ax.set_xlabel('Inference Tokens (log scale)', fontsize=12)
    ax.set_ylabel('Accuracy on Hard Math Problems (%)', fontsize=12)
    ax.set_title('Test-Time Compute Scaling: More Thinking = Better Answers',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(45, 95)
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.annotate('Extended reasoning\nenables harder problems',
                xy=(2000, 85), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    save_chart(fig, 'inference_scaling_curve.pdf')


# =============================================================================
# ALIGNMENT/RLHF CHARTS (4)
# =============================================================================

def create_rlhf_detailed_pipeline():
    """Chart 9: Detailed RLHF 3-stage pipeline"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Stage 1: SFT
    sft_box = FancyBboxPatch((0.5, 6), 4, 2.5),
    ax.add_patch(FancyBboxPatch((0.5, 6), 4, 2.5, boxstyle="round,pad=0.1",
                                facecolor=COLORS['lightblue'], edgecolor=COLORS['mlblue'], linewidth=2))
    ax.text(2.5, 7.8, 'Stage 1: SFT', ha='center', fontsize=11, fontweight='bold')
    ax.text(2.5, 7.0, 'Human demonstrations\n(prompt, response) pairs\nSupervised fine-tuning',
            ha='center', fontsize=9)

    # Stage 2: Reward Model
    ax.add_patch(FancyBboxPatch((5, 6), 4, 2.5, boxstyle="round,pad=0.1",
                                facecolor=COLORS['lightorange'], edgecolor=COLORS['mlorange'], linewidth=2))
    ax.text(7, 7.8, 'Stage 2: Reward Model', ha='center', fontsize=11, fontweight='bold')
    ax.text(7, 7.0, 'Human preferences\n(y_w vs y_l) comparisons\nBradley-Terry model',
            ha='center', fontsize=9)

    # Stage 3: PPO
    ax.add_patch(FancyBboxPatch((9.5, 6), 4, 2.5, boxstyle="round,pad=0.1",
                                facecolor=COLORS['lightgreen'], edgecolor=COLORS['mlgreen'], linewidth=2))
    ax.text(11.5, 7.8, 'Stage 3: PPO', ha='center', fontsize=11, fontweight='bold')
    ax.text(11.5, 7.0, 'Policy optimization\nKL penalty to reference\nIterative updates',
            ha='center', fontsize=9)

    # Arrows between stages
    ax.annotate('', xy=(5, 7.25), xytext=(4.5, 7.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(9.5, 7.25), xytext=(9, 7.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # PPO Detail box
    ax.add_patch(FancyBboxPatch((3, 1), 8, 3.5, boxstyle="round,pad=0.1",
                                facecolor='white', edgecolor=COLORS['mlpurple'], linewidth=2))
    ax.text(7, 4, 'PPO Training Loop', ha='center', fontsize=11, fontweight='bold')

    # Components in PPO
    components = [
        (4, 2.8, 'Policy\n(training)', COLORS['mlgreen']),
        (6, 2.8, 'Reference\n(frozen)', COLORS['mlgray']),
        (8, 2.8, 'Reward\nModel', COLORS['mlorange']),
        (10, 2.8, 'Critic\n(optional)', COLORS['mlblue'])
    ]
    for x, y, label, color in components:
        circle = plt.Circle((x, y), 0.6, facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, y-1.2, label, ha='center', fontsize=8)

    # PPO equation
    ax.text(7, 1.5, r'$\mathcal{L} = \mathbb{E}[r(y)] - \beta \cdot KL(\pi_\theta \| \pi_{ref})$',
            ha='center', fontsize=11, family='serif')

    ax.set_title('RLHF: Three-Stage Training Pipeline', fontsize=14, fontweight='bold', pad=20)

    save_chart(fig, 'rlhf_detailed_pipeline.pdf')


def create_dpo_vs_rlhf_comparison():
    """Chart 10: DPO vs RLHF side-by-side comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 8)
        ax.axis('off')

    # RLHF side (left)
    ax1.set_title('RLHF (Traditional)', fontsize=13, fontweight='bold', color=COLORS['mlorange'])

    rlhf_boxes = [
        (3, 7, 'SFT Data', COLORS['lightblue']),
        (3, 5.5, 'Preference Data', COLORS['lightorange']),
        (1.5, 4, 'Reward\nModel', COLORS['mlorange']),
        (4.5, 4, 'Reference\nModel', COLORS['mlgray']),
        (3, 2.5, 'Policy Model', COLORS['mlgreen']),
        (3, 1, 'PPO Loop', COLORS['lightgreen'])
    ]

    for x, y, label, color in rlhf_boxes:
        box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=1)
        ax1.add_patch(box)
        ax1.text(x, y, label, ha='center', va='center', fontsize=9)

    # Arrows for RLHF
    ax1.annotate('', xy=(3, 6.6), xytext=(3, 5.9), arrowprops=dict(arrowstyle='->', color='gray'))
    ax1.annotate('', xy=(1.5, 5.1), xytext=(2.5, 5.5), arrowprops=dict(arrowstyle='->', color='gray'))
    ax1.annotate('', xy=(1.5, 3.6), xytext=(1.5, 2.9), arrowprops=dict(arrowstyle='->', color='gray'))
    ax1.annotate('', xy=(3, 2.1), xytext=(3, 1.4), arrowprops=dict(arrowstyle='->', color='gray'))
    ax1.annotate('', xy=(4.5, 3.6), xytext=(4.5, 2.9), arrowprops=dict(arrowstyle='->', color='gray'))

    ax1.text(3, 0.3, '3 models, iterative, complex', ha='center', fontsize=10,
             style='italic', color=COLORS['mlred'])

    # DPO side (right)
    ax2.set_title('DPO (Simplified)', fontsize=13, fontweight='bold', color=COLORS['mlgreen'])

    dpo_boxes = [
        (3, 6.5, 'Preference Data\n(y_w, y_l pairs)', COLORS['lightorange']),
        (3, 4, 'Reference Model\n(frozen)', COLORS['mlgray']),
        (3, 2, 'Policy Model\n(training)', COLORS['mlgreen'])
    ]

    for x, y, label, color in dpo_boxes:
        box = FancyBboxPatch((x-1.2, y-0.6), 2.4, 1.2, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax2.add_patch(box)
        ax2.text(x, y, label, ha='center', va='center', fontsize=9)

    # Arrows for DPO
    ax2.annotate('', xy=(3, 5.9), xytext=(3, 4.6), arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax2.annotate('', xy=(3, 3.4), xytext=(3, 2.6), arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    ax2.text(3, 0.8, 'Direct optimization\nNo reward model needed', ha='center', fontsize=10,
             style='italic', color=COLORS['mlgreen'])

    # Add "X" over reward model equivalent
    ax2.text(5, 5, 'No RM!', fontsize=12, color=COLORS['mlred'], fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['mlred']))

    fig.suptitle('Alignment Methods: Complexity Comparison', fontsize=14, fontweight='bold', y=0.98)

    save_chart(fig, 'dpo_vs_rlhf_comparison.pdf')


def create_reward_hacking_examples():
    """Chart 11: Reward hacking examples visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    examples = [
        ('Verbosity Hack',
         'Short: "Paris"\nReward: 0.6',
         'Long: "The capital city of\nFrance, which has been\nknown as Paris since..."\nReward: 0.95',
         'Model learns:\nlonger = higher reward\n(even if unnecessary)'),
        ('Sycophancy Hack',
         'User: "Is 2+2=5?"\nHonest: "No, 2+2=4"\nReward: 0.5',
         'User: "Is 2+2=5?"\nSycophant: "You raise\nan interesting point..."\nReward: 0.8',
         'Model learns:\nagree with user\n(even if wrong)'),
        ('Format Gaming',
         'Plain text answer\nReward: 0.6',
         'Bullet points:\n- Point 1\n- Point 2\n- Point 3\nReward: 0.9',
         'Model learns:\nformatting tricks\n(style over substance)')
    ]

    for ax, (title, before, after, lesson) in zip(axes, examples):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        ax.set_title(title, fontsize=12, fontweight='bold', color=COLORS['mlred'])

        # Before box
        ax.add_patch(FancyBboxPatch((0.5, 5.5), 4, 3.5, boxstyle="round,pad=0.1",
                                    facecolor='white', edgecolor=COLORS['mlgray'], linewidth=1))
        ax.text(2.5, 8.5, 'Before', ha='center', fontsize=10, fontweight='bold')
        ax.text(2.5, 7, before, ha='center', va='center', fontsize=8, family='monospace')

        # After box (hacked)
        ax.add_patch(FancyBboxPatch((5.5, 5.5), 4, 3.5, boxstyle="round,pad=0.1",
                                    facecolor=COLORS['lightred'], edgecolor=COLORS['mlred'], linewidth=1.5))
        ax.text(7.5, 8.5, 'Hacked', ha='center', fontsize=10, fontweight='bold', color=COLORS['mlred'])
        ax.text(7.5, 7, after, ha='center', va='center', fontsize=8, family='monospace')

        # Arrow
        ax.annotate('', xy=(5.5, 7.25), xytext=(4.5, 7.25),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

        # Lesson box
        ax.add_patch(FancyBboxPatch((1, 1), 8, 2.5, boxstyle="round,pad=0.1",
                                    facecolor=COLORS['lightorange'], edgecolor=COLORS['mlorange'], linewidth=1))
        ax.text(5, 2.25, lesson, ha='center', va='center', fontsize=9, style='italic')

    fig.suptitle('Reward Hacking: When Models Game the Reward Signal', fontsize=14, fontweight='bold', y=0.98)

    save_chart(fig, 'reward_hacking_examples.pdf')


def create_alignment_timeline():
    """Chart 12: Alignment methods evolution timeline"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(2021.5, 2025.5)
    ax.set_ylim(0, 10)

    # Timeline base
    ax.axhline(y=5, color='black', linewidth=2)

    events = [
        (2022, 'RLHF\n(InstructGPT)', 'above', COLORS['mlorange']),
        (2022.5, 'Constitutional AI\n(Anthropic)', 'below', COLORS['mlpurple']),
        (2023, 'RLAIF\n(AI Feedback)', 'above', COLORS['mlblue']),
        (2023.5, 'DPO\n(Stanford)', 'below', COLORS['mlgreen']),
        (2024, 'IPO, KTO\n(Variants)', 'above', COLORS['mlgray']),
        (2024.5, 'Process Supervision\n(Step-level)', 'below', COLORS['mlorange']),
        (2025, 'GRPO\n(DeepSeek)', 'above', COLORS['mlgreen'])
    ]

    for year, label, pos, color in events:
        y_offset = 6.5 if pos == 'above' else 3.5
        ax.plot(year, 5, 'o', markersize=15, color=color, markeredgecolor='black', zorder=5)
        ax.annotate(label, xy=(year, 5), xytext=(year, y_offset),
                   ha='center', fontsize=9, fontweight='bold',
                   arrowprops=dict(arrowstyle='-', color='gray', lw=1),
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=1.5))

    # Year markers
    for year in [2022, 2023, 2024, 2025]:
        ax.text(year, 4.3, str(year), ha='center', fontsize=11, fontweight='bold')

    # Trend arrow and label
    ax.annotate('', xy=(2025, 1.5), xytext=(2022, 1.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['mlgreen'], lw=3))
    ax.text(2023.5, 0.8, 'Trend: Simpler, More Scalable, Less Human Labeling',
            ha='center', fontsize=11, style='italic')

    ax.set_title('Evolution of Alignment Methods (2022-2025)', fontsize=14, fontweight='bold')
    ax.axis('off')

    save_chart(fig, 'alignment_timeline.pdf')


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("Generating additional charts for NLP Final Lecture...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # RAG Charts
    print("=== RAG/Retrieval Charts ===")
    create_embedding_space_2d()
    create_bm25_vs_dense()
    create_chunk_size_tradeoff()
    # Note: hybrid_search_flow.pdf generated separately with Graphviz

    # Reasoning Charts
    print("\n=== Reasoning/CoT Charts ===")
    create_cot_accuracy_gains()
    create_self_consistency_voting()
    create_deepseek_r1_pipeline()
    create_inference_scaling_curve()

    # Alignment Charts
    print("\n=== Alignment/RLHF Charts ===")
    create_rlhf_detailed_pipeline()
    create_dpo_vs_rlhf_comparison()
    create_reward_hacking_examples()
    create_alignment_timeline()

    print("\n" + "="*50)
    print("Generated 11 matplotlib charts.")
    print("Note: hybrid_search_flow.pdf needs separate Graphviz generation.")
    print("="*50)
