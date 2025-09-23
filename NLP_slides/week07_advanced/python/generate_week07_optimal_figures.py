"""
Generate optimal readability figures for Week 7: Advanced Transformers
Using monochromatic style with high contrast
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.patches import FancyArrowPatch
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define color palette (optimal readability)
COLOR_BLACK = '#000000'      # Pure black for text
COLOR_BLUE = '#003D7A'        # Deep blue for primary
COLOR_GRAY = '#4A4A4A'        # Dark gray for secondary
COLOR_LIGHT = '#E5E5E5'       # Light gray for backgrounds
COLOR_ACCENT = '#0066CC'      # Chart blue
COLOR_ORANGE = '#FF8800'      # Chart orange
COLOR_TEAL = '#00A0A0'        # Chart teal
COLOR_PURPLE = '#8B4789'      # Chart purple
COLOR_GREEN = '#228B22'       # Success green
COLOR_RED = '#CC0000'         # Warning red

def setup_figure(figsize=(10, 6)):
    """Setup figure with consistent style"""
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    return fig, ax

def save_figure(fig, filename):
    """Save figure with consistent settings"""
    fig.savefig(f'../figures/{filename}',
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

# 1. Emergent Abilities Chart
def create_emergent_abilities():
    fig, ax = setup_figure((10, 6))

    # Data
    params = np.logspace(8, 12, 100)  # 100M to 1T

    # Different abilities emerge at different scales
    arithmetic = 1 / (1 + np.exp(-(np.log10(params) - 10) * 4))
    reasoning = 1 / (1 + np.exp(-(np.log10(params) - 10.5) * 4))
    translation = 1 / (1 + np.exp(-(np.log10(params) - 9.5) * 4))
    coding = 1 / (1 + np.exp(-(np.log10(params) - 10.8) * 4))

    ax.semilogx(params, arithmetic * 100, COLOR_ACCENT, linewidth=2.5, label='Arithmetic')
    ax.semilogx(params, reasoning * 100, COLOR_ORANGE, linewidth=2.5, label='Reasoning')
    ax.semilogx(params, translation * 100, COLOR_TEAL, linewidth=2.5, label='Translation')
    ax.semilogx(params, coding * 100, COLOR_PURPLE, linewidth=2.5, label='Code Generation')

    # Add emergence points
    ax.axvline(x=1e10, color=COLOR_GRAY, linestyle='--', alpha=0.5)
    ax.text(1e10, 105, '10B: Phase Transition',
            ha='center', fontsize=10, color=COLOR_GRAY)

    ax.set_xlabel('Model Parameters', fontsize=12, fontweight='bold')
    ax.set_ylabel('Task Performance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Emergent Abilities at Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1e8, 1e12])
    ax.set_ylim([0, 110])

    save_figure(fig, 'emergent_abilities_chart.pdf')

# 2. T5 Span Corruption
def create_t5_span_corruption():
    fig, ax = setup_figure((12, 5))
    ax.axis('off')

    # Input text
    input_text = "The quick [MASK] jumps [MASK] the lazy dog"
    target_text = "[MASK1] brown fox [MASK2] over"

    # Draw boxes
    input_box = FancyBboxPatch((0.05, 0.6), 0.9, 0.25,
                                boxstyle="round,pad=0.02",
                                facecolor=COLOR_LIGHT,
                                edgecolor=COLOR_BLUE,
                                linewidth=2)
    ax.add_patch(input_box)

    target_box = FancyBboxPatch((0.05, 0.15), 0.9, 0.25,
                                 boxstyle="round,pad=0.02",
                                 facecolor='white',
                                 edgecolor=COLOR_ACCENT,
                                 linewidth=2)
    ax.add_patch(target_box)

    # Add text
    ax.text(0.5, 0.725, 'Input: Corrupted Text',
            ha='center', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.65, input_text,
            ha='center', fontsize=14, family='monospace')

    ax.text(0.5, 0.375, 'Target: Missing Spans',
            ha='center', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.25, target_text,
            ha='center', fontsize=14, family='monospace', color=COLOR_ACCENT)

    # Arrow
    arrow = FancyArrowPatch((0.5, 0.58), (0.5, 0.42),
                           arrowstyle='->', lw=2,
                           color=COLOR_BLUE,
                           mutation_scale=20)
    ax.add_patch(arrow)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    save_figure(fig, 't5_span_corruption.pdf')

# 3. GPT-3 Performance
def create_gpt3_performance():
    fig, ax = setup_figure((10, 6))

    # Data for different model sizes
    sizes = ['Small\n(125M)', 'Medium\n(350M)', 'Large\n(760M)',
             'XL\n(1.3B)', '2.7B', '6.7B', '13B', 'GPT-3\n(175B)']

    zero_shot = [15, 22, 28, 35, 42, 48, 55, 71]
    one_shot = [18, 26, 34, 42, 51, 58, 65, 79]
    few_shot = [22, 31, 40, 49, 58, 66, 73, 87]

    x = np.arange(len(sizes))
    width = 0.25

    bars1 = ax.bar(x - width, zero_shot, width, label='Zero-shot',
                   color=COLOR_ACCENT, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, one_shot, width, label='One-shot',
                   color=COLOR_ORANGE, edgecolor='black', linewidth=1)
    bars3 = ax.bar(x + width, few_shot, width, label='Few-shot',
                   color=COLOR_TEAL, edgecolor='black', linewidth=1)

    ax.set_xlabel('Model Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Performance (%)', fontsize=12, fontweight='bold')
    ax.set_title('In-Context Learning Performance by Model Size',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])

    # Add annotation for GPT-3
    ax.annotate('Breakthrough\nperformance', xy=(7, 87), xytext=(5.5, 95),
                arrowprops=dict(arrowstyle='->', color=COLOR_RED, lw=2),
                fontsize=10, color=COLOR_RED, fontweight='bold')

    save_figure(fig, 'gpt3_performance.pdf')

# 4. Architecture Comparison (Advanced)
def create_architecture_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), facecolor='white')

    titles = ['Encoder-Only\n(BERT)', 'Decoder-Only\n(GPT)', 'Encoder-Decoder\n(T5)']

    for idx, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])
        ax.axis('off')

        # Title
        ax.text(5, 9, title, ha='center', fontsize=12, fontweight='bold')

        if idx == 0:  # BERT
            # Bidirectional layers
            for i in range(3):
                rect = Rectangle((2, 6-i*1.5), 6, 1,
                               facecolor=COLOR_ACCENT, alpha=0.7,
                               edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                # Bidirectional arrows
                ax.annotate('', xy=(1.5, 6.5-i*1.5), xytext=(2, 6.5-i*1.5),
                           arrowprops=dict(arrowstyle='<->', color='black'))
                ax.annotate('', xy=(8, 6.5-i*1.5), xytext=(8.5, 6.5-i*1.5),
                           arrowprops=dict(arrowstyle='<->', color='black'))
            ax.text(5, 1, 'Bidirectional\nAttention', ha='center', fontsize=10)

        elif idx == 1:  # GPT
            # Autoregressive layers
            for i in range(3):
                rect = Rectangle((2, 6-i*1.5), 6, 1,
                               facecolor=COLOR_ORANGE, alpha=0.7,
                               edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                # Causal mask visualization
                if i == 0:
                    for j in range(3):
                        ax.plot([3+j*1.5, 3+j*1.5], [6, 7],
                               'k--', alpha=0.5, linewidth=0.5)
                        if j > 0:
                            ax.add_patch(Rectangle((3+(j-1)*1.5, 6),
                                                  1.5, 1,
                                                  facecolor='gray', alpha=0.3))
            ax.text(5, 1, 'Causal\nAttention', ha='center', fontsize=10)

        else:  # T5
            # Encoder stack
            for i in range(2):
                rect = Rectangle((1, 6-i*1.2), 3.5, 1,
                               facecolor=COLOR_TEAL, alpha=0.7,
                               edgecolor='black', linewidth=1)
                ax.add_patch(rect)
            ax.text(2.75, 3.2, 'Encoder', ha='center', fontsize=10)

            # Decoder stack
            for i in range(2):
                rect = Rectangle((5.5, 6-i*1.2), 3.5, 1,
                               facecolor=COLOR_PURPLE, alpha=0.7,
                               edgecolor='black', linewidth=1)
                ax.add_patch(rect)
            ax.text(7.25, 3.2, 'Decoder', ha='center', fontsize=10)

            # Cross-attention arrow
            arrow = FancyArrowPatch((4.5, 5.5), (5.5, 5.5),
                                  arrowstyle='->', lw=2,
                                  color=COLOR_RED)
            ax.add_patch(arrow)
            ax.text(5, 5.9, 'Cross-\nAttention', ha='center', fontsize=8)

    plt.suptitle('Transformer Architecture Paradigms',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_figure(fig, 'architecture_comparison_advanced.pdf')

# 5. MoE Architecture
def create_moe_architecture():
    fig, ax = setup_figure((12, 7))
    ax.set_xlim([0, 12])
    ax.set_ylim([0, 8])
    ax.axis('off')

    # Input token
    token_box = Rectangle((1, 1), 2, 0.8,
                          facecolor=COLOR_ACCENT,
                          edgecolor='black', linewidth=2)
    ax.add_patch(token_box)
    ax.text(2, 1.4, 'Token', ha='center', fontsize=11, color='white', fontweight='bold')

    # Router
    router_box = Rectangle((5, 0.5), 2, 1.8,
                          facecolor=COLOR_ORANGE,
                          edgecolor='black', linewidth=2)
    ax.add_patch(router_box)
    ax.text(6, 1.4, 'Router', ha='center', fontsize=11, fontweight='bold')

    # Experts
    expert_colors = [COLOR_TEAL, COLOR_PURPLE, COLOR_GREEN, COLOR_BLUE]
    expert_names = ['Expert 1', 'Expert 2', 'Expert 3', 'Expert N']

    for i, (color, name) in enumerate(zip(expert_colors, expert_names)):
        y_pos = 6 - i * 1.5
        expert = Rectangle((9, y_pos), 2, 1,
                          facecolor=color, alpha=0.7,
                          edgecolor='black', linewidth=2)
        ax.add_patch(expert)
        ax.text(10, y_pos + 0.5, name, ha='center', fontsize=10, fontweight='bold')

        # Router connections
        if i in [0, 2]:  # Only connect to selected experts
            arrow = FancyArrowPatch((7, 1.4), (9, y_pos + 0.5),
                                  arrowstyle='->', lw=2,
                                  color=COLOR_RED if i == 0 else COLOR_GRAY,
                                  alpha=1 if i == 0 else 0.3)
            ax.add_patch(arrow)

    # Input arrow
    arrow1 = FancyArrowPatch((3, 1.4), (5, 1.4),
                           arrowstyle='->', lw=2.5,
                           color='black')
    ax.add_patch(arrow1)

    # Labels
    ax.text(6, 7.5, 'Mixture of Experts (MoE)',
            ha='center', fontsize=14, fontweight='bold')
    ax.text(6, 7, 'Only 1-2 experts active per token',
            ha='center', fontsize=10, color=COLOR_GRAY)

    # Stats box
    stats_box = Rectangle((0.5, 4), 3.5, 3,
                         facecolor='white',
                         edgecolor=COLOR_GRAY, linewidth=1)
    ax.add_patch(stats_box)
    ax.text(2.25, 6.5, 'Switch Transformer', ha='center', fontsize=10, fontweight='bold')
    ax.text(2.25, 6, '1.6T total params', ha='center', fontsize=9)
    ax.text(2.25, 5.5, '100B active params', ha='center', fontsize=9, color=COLOR_GREEN)
    ax.text(2.25, 5, '2048 experts', ha='center', fontsize=9)
    ax.text(2.25, 4.5, '7x speedup', ha='center', fontsize=9, color=COLOR_RED)

    save_figure(fig, 'moe_architecture.pdf')

# 6. 3D Parallelism (2D representation)
def create_3d_parallelism():
    fig, ax = setup_figure((12, 7))
    ax.set_xlim([0, 12])
    ax.set_ylim([0, 7])
    ax.axis('off')

    # Title
    ax.text(6, 6.5, '3D Parallelism for Large Model Training',
           ha='center', fontsize=14, fontweight='bold')

    # Three types of parallelism
    parallelism_types = [
        ('Data Parallel', 2, COLOR_ACCENT,
         'Split batch\nacross GPUs'),
        ('Pipeline Parallel', 6, COLOR_ORANGE,
         'Split layers\nacross GPUs'),
        ('Tensor Parallel', 10, COLOR_TEAL,
         'Split matrices\nacross GPUs')
    ]

    for name, x, color, desc in parallelism_types:
        # Main box
        box = FancyBboxPatch((x-1.5, 3), 3, 2,
                            boxstyle="round,pad=0.1",
                            facecolor=color, alpha=0.3,
                            edgecolor=color, linewidth=2)
        ax.add_patch(box)

        # Title
        ax.text(x, 4.5, name, ha='center', fontsize=11, fontweight='bold')

        # Description
        ax.text(x, 3.5, desc, ha='center', fontsize=9)

        # GPU representation
        for i in range(2):
            for j in range(2):
                gpu = Rectangle((x-0.8+i*0.8, 3.8+j*0.4), 0.6, 0.3,
                              facecolor='white', edgecolor='black', linewidth=1)
                ax.add_patch(gpu)

    # Combination arrow
    arrow1 = FancyArrowPatch((3.5, 2.5), (5, 1.5),
                           arrowstyle='->', lw=2, color=COLOR_GRAY)
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((8.5, 2.5), (7, 1.5),
                           arrowstyle='->', lw=2, color=COLOR_GRAY)
    ax.add_patch(arrow2)

    # Combined result
    result_box = FancyBboxPatch((4.5, 0.5), 3, 1,
                               boxstyle="round,pad=0.1",
                               facecolor=COLOR_PURPLE, alpha=0.3,
                               edgecolor=COLOR_PURPLE, linewidth=2)
    ax.add_patch(result_box)
    ax.text(6, 1, '3D Combined\nOptimal Training', ha='center',
           fontsize=10, fontweight='bold')

    save_figure(fig, '3d_parallelism.pdf')

# 7. Compute Scaling
def create_compute_scaling():
    fig, ax = setup_figure((10, 6))

    # Historical compute requirements
    years = np.array([2018, 2019, 2020, 2021, 2022, 2023])
    models = ['BERT', 'GPT-2', 'GPT-3', 'Switch', 'PaLM', 'GPT-4']
    flops = [1e18, 1e19, 3.14e23, 1e24, 2.5e24, 1e25]  # Approximate

    ax.semilogy(years, flops, 'o-', color=COLOR_RED,
                linewidth=2.5, markersize=10)

    # Add model labels
    for year, model, flop in zip(years, models, flops):
        ax.annotate(model, (year, flop),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')

    # Trend line
    z = np.polyfit(years, np.log10(flops), 1)
    p = np.poly1d(z)
    years_extended = np.linspace(2018, 2025, 100)
    ax.semilogy(years_extended, 10**p(years_extended), '--',
                color=COLOR_GRAY, alpha=0.5, linewidth=1.5)

    # Add doubling time annotation
    ax.text(2021.5, 1e22, 'Doubling every\n3.4 months',
           ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLOR_RED))

    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Compute (FLOPs)', fontsize=12, fontweight='bold')
    ax.set_title('The Exponential Growth of Training Compute',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([2017.5, 2023.5])

    save_figure(fig, 'compute_scaling.pdf')

# 8. GPT-3 Applications
def create_gpt3_applications():
    fig, ax = setup_figure((12, 7))
    ax.set_xlim([0, 12])
    ax.set_ylim([0, 8])
    ax.axis('off')

    # Central GPT-3 node
    center = Circle((6, 4), 1.2, facecolor=COLOR_BLUE,
                   edgecolor='black', linewidth=2)
    ax.add_patch(center)
    ax.text(6, 4, 'GPT-3\nAPI', ha='center', va='center',
           fontsize=12, color='white', fontweight='bold')

    # Application categories
    applications = [
        ('Code\nGeneration', 2, 6.5, COLOR_ACCENT),
        ('Content\nCreation', 10, 6.5, COLOR_ORANGE),
        ('Customer\nService', 2, 1.5, COLOR_TEAL),
        ('Education', 10, 1.5, COLOR_PURPLE),
        ('Translation', 1, 4, COLOR_GREEN),
        ('Analysis', 11, 4, COLOR_RED)
    ]

    for name, x, y, color in applications:
        # Application circle
        circle = Circle((x, y), 0.8, facecolor=color, alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center',
               fontsize=9, fontweight='bold')

        # Connection to center
        arrow = FancyArrowPatch((x, y), (6, 4),
                              arrowstyle='-', lw=1.5,
                              color=COLOR_GRAY, alpha=0.5)
        ax.add_patch(arrow)

    # Title
    ax.text(6, 7.5, 'GPT-3 Application Ecosystem',
           ha='center', fontsize=14, fontweight='bold')

    # Stats
    stats_text = '100M+ API calls/day | 300K+ developers | 10K+ applications'
    ax.text(6, 0.5, stats_text, ha='center', fontsize=10,
           color=COLOR_GRAY, style='italic')

    save_figure(fig, 'gpt3_applications.pdf')

# 9. Alignment Challenge
def create_alignment_challenge():
    fig, ax = setup_figure((10, 6))
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 6])
    ax.axis('off')

    # Three circles representing different objectives
    circles = [
        ('What we\nwant', 2, 3, COLOR_GREEN),
        ('What we\ntrain', 5, 3, COLOR_ORANGE),
        ('What we\nget', 8, 3, COLOR_RED)
    ]

    for i, (label, x, y, color) in enumerate(circles):
        circle = Circle((x, y), 1.2, facecolor=color, alpha=0.3,
                       edgecolor=color, linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center',
               fontsize=11, fontweight='bold')

        # Arrows showing misalignment
        if i < len(circles) - 1:
            next_x = circles[i+1][1]
            arrow = FancyArrowPatch((x+1, y), (next_x-1, y),
                                  arrowstyle='->', lw=2,
                                  color='black')
            ax.add_patch(arrow)
            ax.text((x+next_x)/2, y+0.3, '≠',
                   ha='center', fontsize=16, color=COLOR_RED)

    # Title and subtitle
    ax.text(5, 5, 'The Alignment Problem',
           ha='center', fontsize=14, fontweight='bold')
    ax.text(5, 4.5, 'Intent ≠ Specification ≠ Behavior',
           ha='center', fontsize=10, color=COLOR_GRAY)

    # Examples
    examples = [
        'Helpful but not harmful',
        'Truthful but not overly verbose',
        'Creative but not hallucinating'
    ]

    for i, example in enumerate(examples):
        ax.text(5, 1.5-i*0.4, f'• {example}',
               ha='center', fontsize=9, color=COLOR_GRAY)

    save_figure(fig, 'alignment_challenge.pdf')

# 10. Model Explosion Timeline
def create_model_explosion():
    fig, ax = setup_figure((12, 7))

    # Timeline data
    timeline = [
        (2020, 'GPT-3', 175, COLOR_ACCENT),
        (2021, 'Switch', 1600, COLOR_ORANGE),
        (2021, 'MT-NLG', 530, COLOR_TEAL),
        (2022, 'PaLM', 540, COLOR_PURPLE),
        (2022, 'Chinchilla', 70, COLOR_GREEN),
        (2023, 'GPT-4', 1000, COLOR_RED),  # Estimated
        (2023, 'Llama 2', 70, COLOR_BLUE),
        (2023, 'Claude', 52, COLOR_GRAY)
    ]

    years = [t[0] for t in timeline]
    names = [t[1] for t in timeline]
    sizes = [t[2] for t in timeline]
    colors = [t[3] for t in timeline]

    # Create bubble chart
    for year, name, size, color in timeline:
        # Bubble size proportional to model size
        bubble_size = np.sqrt(size) * 50
        circle = Circle((year, size), bubble_size/1000,
                       facecolor=color, alpha=0.6,
                       edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(year, size, name, ha='center', va='center',
               fontsize=8, fontweight='bold')

    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Size (Billions of Parameters)', fontsize=12, fontweight='bold')
    ax.set_title('The Cambrian Explosion of Large Language Models',
                fontsize=14, fontweight='bold')

    ax.set_xlim([2019.5, 2023.5])
    ax.set_ylim([0, 1700])
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('Sparse models\nenable massive scale',
               xy=(2021, 1600), xytext=(2022, 1400),
               arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=1.5),
               fontsize=9, color=COLOR_ORANGE)

    save_figure(fig, 'model_explosion.pdf')

# Generate all figures
if __name__ == "__main__":
    print("Generating Week 7 Advanced Transformers figures...")

    create_emergent_abilities()
    print("- Emergent abilities chart")

    create_t5_span_corruption()
    print("- T5 span corruption")

    create_gpt3_performance()
    print("- GPT-3 performance")

    create_architecture_comparison()
    print("- Architecture comparison")

    create_moe_architecture()
    print("- MoE architecture")

    create_3d_parallelism()
    print("- 3D parallelism")

    create_compute_scaling()
    print("- Compute scaling")

    create_gpt3_applications()
    print("- GPT-3 applications")

    create_alignment_challenge()
    print("- Alignment challenge")

    create_model_explosion()
    print("- Model explosion timeline")

    print("\nAll figures generated successfully!")
    print("Saved to: NLP_slides/week07_advanced/figures/")