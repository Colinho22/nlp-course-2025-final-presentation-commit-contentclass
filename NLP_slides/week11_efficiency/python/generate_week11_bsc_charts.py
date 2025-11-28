"""
Generate BSc-level charts for Week 11: Efficiency & Optimization
Educational framework: Zero pre-knowledge, discovery-based pedagogy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrow, Ellipse, Wedge, Arc, Polygon
from matplotlib.path import Path
import matplotlib.patheffects as path_effects
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Color scheme (matching educational framework)
MLPURPLE = '#3333B2'    # Primary accent
DARKGRAY = '#404040'    # Main text
MIDGRAY = '#B4B4B4'     # Secondary elements
LIGHTGRAY = '#F0F0F0'   # Backgrounds

# Educational color scheme
COLOR_IMPOSSIBLE = '#FF6B6B'  # Red for impossibility/problem
COLOR_SOLUTION = '#95E77E'    # Green for solution
COLOR_WARNING = '#FFE66D'     # Yellow for caution
COLOR_INFO = '#4ECDC4'        # Teal for information

# Create figures directory
os.makedirs('../figures', exist_ok=True)

def set_minimalist_style(ax):
    """Apply minimalist style to axis"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MIDGRAY)
    ax.spines['bottom'].set_color(MIDGRAY)
    ax.tick_params(colors=DARKGRAY, which='both')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_facecolor('white')


# 1. Model Size Impossibility (Hook)
def plot_model_size_impossibility():
    fig, ax = plt.subplots(figsize=(14, 8))

    # Laptop RAM bar
    laptop_width = 16
    ax.barh(0, laptop_width, height=0.4, color='lightgreen',
            edgecolor='black', linewidth=2, label='Laptop RAM')
    ax.text(laptop_width/2, 0, '16 GB', ha='center', va='center',
            fontsize=14, fontweight='bold')

    # Model sizes
    models = [
        ('BERT-Base', 0.44, 1),
        ('GPT-2', 6, 2),
        ('BERT-Large', 1.3, 3),
        ('GPT-3 (INT8)', 175, 4),
        ('GPT-3 (FP32)', 700, 5)
    ]

    for name, size, y in models:
        if size <= 16:
            color = 'lightblue'
            alpha = 0.7
        else:
            color = 'lightcoral'
            alpha = 0.9

        # Clip width for visualization
        display_width = min(size, 750)
        ax.barh(y, display_width, height=0.4, color=color, alpha=alpha,
                edgecolor='black', linewidth=2)

        # Label
        ax.text(5, y, f'{name}: {size} GB', ha='left', va='center',
                fontsize=12, fontweight='bold')

        # Overflow indicator
        if size > 750:
            ax.text(750, y, '→', ha='right', va='center',
                    fontsize=20, color='red')

    # Impossibility zone
    ax.axvspan(16, 750, alpha=0.1, color='red')
    ax.text(383, 5.7, 'IMPOSSIBLE', ha='center', fontsize=16,
            fontweight='bold', color='darkred')
    ax.text(383, 5.4, 'Models won\'t fit in laptop memory',
            ha='center', fontsize=11, style='italic', color='darkred')

    # Feasible zone
    ax.axvspan(0, 16, alpha=0.1, color='green')
    ax.text(8, 5.7, 'FEASIBLE', ha='center', fontsize=12,
            fontweight='bold', color='darkgreen')

    # Vertical line at laptop limit
    ax.axvline(16, color='green', linewidth=3, linestyle='--', alpha=0.7)
    ax.text(16, -0.5, 'Laptop\nLimit', ha='center', fontsize=10,
            fontweight='bold', color='darkgreen')

    ax.set_xlabel('Model Size (GB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Models', fontsize=14, fontweight='bold')
    ax.set_title('The 350GB Problem: Most Models Won\'t Fit on Your Laptop\nCompression makes the impossible possible',
                fontsize=16, fontweight='bold', pad=15)

    ax.set_xlim(0, 750)
    ax.set_ylim(-1, 6)
    ax.set_yticks([])

    # Add insight box
    ax.text(0.98, 0.02, 'Key Insight: 4-bit quantization → 700GB becomes 87GB (fits on some GPUs!)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow'))

    plt.tight_layout()
    plt.savefig('../figures/model_size_impossibility.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("1/8: model_size_impossibility.pdf created")


# 2. Compression Spectrum Visual
def plot_compression_spectrum():
    fig, ax = plt.subplots(figsize=(14, 7))

    # Gradient background
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap='RdYlGn', extent=[0, 100, 0, 1],
              alpha=0.3)

    # Methods on spectrum
    methods = [
        (5, 'Weight\nSharing', '0% loss\n100% → 80%', 'green'),
        (20, 'FP32→FP16', '0.1% loss\n100% → 50%', 'lightgreen'),
        (40, 'INT8\nQuant', '1% loss\n100% → 25%', 'yellow'),
        (60, 'INT4\nQuant', '3% loss\n100% → 12.5%', 'orange'),
        (80, 'Pruning\n90%', '2-5% loss\n100% → 10%', 'darkorange'),
        (95, 'Distill\n10×', '5-10% loss\n100% → 10%', 'red')
    ]

    for x, name, desc, color in methods:
        # Vertical line
        ax.plot([x, x], [1, 1.4], 'k-', linewidth=2)

        # Marker
        ax.scatter(x, 1, s=400, c=color, edgecolors='black',
                  linewidth=2, zorder=3, marker='o')

        # Label
        ax.text(x, 1.6, name, ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))

        # Description
        ax.text(x, 0.5, desc, ha='center', va='center',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.2',
                facecolor='white', alpha=0.8))

    # Zones
    ax.text(12, -0.2, 'LOSSLESS\nZone', ha='center', fontsize=12,
            fontweight='bold', color='darkgreen')
    ax.text(50, -0.2, 'LOSSY BUT ACCURATE\nZone', ha='center', fontsize=12,
            fontweight='bold', color='darkorange')
    ax.text(88, -0.2, 'HIGH LOSS\nZone', ha='center', fontsize=12,
            fontweight='bold', color='darkred')

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.4, 2)
    ax.set_xlabel('Compression Aggressiveness →', fontsize=14, fontweight='bold')
    ax.set_title('The Compression Spectrum: From Lossless to Extreme Lossy\nAccuracy Loss vs Size Reduction Tradeoff',
                fontsize=16, fontweight='bold', pad=20)

    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig('../figures/compression_spectrum_visual.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("2/8: compression_spectrum_visual.pdf created")


# 3. Quantization Comparison
def plot_quantization_comparison():
    fig, ax = plt.subplots(figsize=(14, 9))

    # Data types
    dtypes = ['FP32', 'FP16', 'INT8', 'INT4', 'Binary']
    bits = [32, 16, 8, 4, 1]
    sizes = [100, 50, 25, 12.5, 3.125]  # Relative size
    accuracy = [100, 99.9, 99, 97, 85]  # Typical accuracy retention

    x_pos = np.arange(len(dtypes))
    width = 0.35

    # Size bars
    bars1 = ax.bar(x_pos - width/2, sizes, width, label='Relative Size (%)',
                   color=['darkred', 'red', 'orange', 'yellow', 'lightgreen'],
                   edgecolor='black', linewidth=2)

    # Accuracy bars
    bars2 = ax.bar(x_pos + width/2, accuracy, width, label='Accuracy Retention (%)',
                   color='lightblue', edgecolor='black', linewidth=2)

    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()

        ax.text(bar1.get_x() + bar1.get_width()/2, height1 + 2,
                f'{sizes[i]:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

        ax.text(bar2.get_x() + bar2.get_width()/2, height2 + 2,
                f'{accuracy[i]:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

        # Add bits info
        ax.text(i, -10, f'{bits[i]} bits', ha='center', fontsize=10,
                fontweight='bold')

    ax.set_xlabel('Data Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage', fontsize=14, fontweight='bold')
    ax.set_title('Quantization Tradeoffs: Size vs Accuracy\nFewer bits = Smaller model but slight accuracy loss',
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dtypes, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.set_ylim(0, 110)

    set_minimalist_style(ax)

    # Add sweet spot indicator
    ax.annotate('Sweet Spot!\n99% accuracy\n75% smaller',
                xy=(2, (sizes[2] + accuracy[2])/2),
                xytext=(3.5, 70),
                fontsize=11, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))

    plt.tight_layout()
    plt.savefig('../figures/quantization_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("3/8: quantization_comparison.pdf created")


# 4. Accuracy-Size Pareto Frontier
def plot_accuracy_size_pareto():
    fig, ax = plt.subplots(figsize=(12, 9))

    # Methods with (size %, accuracy %)
    methods = {
        'Baseline (FP32)': (100, 100, 'black', 200),
        'FP16': (50, 99.9, 'darkgreen', 180),
        'INT8 Quant': (25, 99, 'green', 250),
        'INT4 Quant': (12.5, 97, 'lightgreen', 250),
        'Pruning 50%': (50, 98.5, 'blue', 180),
        'Pruning 90%': (10, 95, 'lightblue', 180),
        'Distillation 2×': (50, 97, 'purple', 200),
        'Distillation 10×': (10, 92, 'violet', 200),
        'INT8 + Prune': (12.5, 97, 'orange', 250),
        'INT4 + Distill': (5, 90, 'red', 220),
    }

    for name, (size, acc, color, s) in methods.items():
        ax.scatter(size, acc, s=s, c=color, alpha=0.7,
                  edgecolors='black', linewidth=2, zorder=3)

        # Label
        if size > 40 or name == 'Baseline (FP32)':
            ax.annotate(name, (size, acc), xytext=(5, 5),
                       textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.6))
        else:
            ax.annotate(name, (size, acc), xytext=(-5, -15),
                       textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.6))

    # Pareto frontier (approximate)
    frontier_x = [100, 50, 25, 12.5, 10, 5]
    frontier_y = [100, 99.9, 99, 97, 95, 90]
    ax.plot(frontier_x, frontier_y, 'k--', linewidth=2, alpha=0.5,
            label='Pareto Frontier')

    # Good zone
    ax.axhspan(95, 100, alpha=0.1, color='green')
    ax.text(50, 97.5, 'Production Quality\n(>95% accuracy)', ha='center',
            fontsize=11, fontweight='bold', color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Deployment targets
    ax.axvline(16, color='blue', linewidth=2, linestyle=':', alpha=0.5)
    ax.text(16, 89, 'Edge Device\n(16GB)', ha='center', fontsize=9,
            color='blue', fontweight='bold')

    ax.axvline(4, color='purple', linewidth=2, linestyle=':', alpha=0.5)
    ax.text(4, 89, 'Mobile\n(4GB)', ha='center', fontsize=9,
            color='purple', fontweight='bold')

    ax.set_xlabel('Model Size (% of Original)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy Retention (%)', fontsize=14, fontweight='bold')
    ax.set_title('The Compression Pareto Frontier\nBest Accuracy-Size Tradeoffs for Different Methods',
                fontsize=16, fontweight='bold', pad=15)

    ax.set_xscale('log')
    ax.set_xlim(3, 150)
    ax.set_ylim(88, 101)
    ax.legend(loc='lower left', fontsize=11)

    set_minimalist_style(ax)

    plt.tight_layout()
    plt.savefig('../figures/accuracy_size_pareto.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("4/8: accuracy_size_pareto.pdf created")


# 5. Distillation Architecture
def plot_distillation_architecture():
    fig, ax = plt.subplots(figsize=(14, 9))

    # Teacher model (large)
    teacher_rect = FancyBboxPatch((0.05, 0.5), 0.35, 0.35,
                                  boxstyle="round,pad=0.02",
                                  facecolor='lightcoral',
                                  edgecolor='black',
                                  linewidth=3)
    ax.add_patch(teacher_rect)
    ax.text(0.225, 0.75, 'TEACHER MODEL\n(BERT-Large)',
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0.225, 0.68, '340M parameters\n94% accuracy\n1.3 GB',
            ha='center', va='center', fontsize=11)

    # Layers representation
    for i in range(5):
        y = 0.62 - i*0.03
        ax.add_patch(Rectangle((0.08, y), 0.29, 0.02,
                               facecolor='darkred', alpha=0.7))

    # Student model (small)
    student_rect = FancyBboxPatch((0.6, 0.5), 0.35, 0.35,
                                   boxstyle="round,pad=0.02",
                                   facecolor='lightgreen',
                                   edgecolor='black',
                                   linewidth=3)
    ax.add_patch(student_rect)
    ax.text(0.775, 0.75, 'STUDENT MODEL\n(BERT-Tiny)',
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0.775, 0.68, '14M parameters\n92% accuracy\n55 MB',
            ha='center', va='center', fontsize=11)

    # Fewer layers
    for i in range(2):
        y = 0.59 - i*0.03
        ax.add_patch(Rectangle((0.63, y), 0.29, 0.02,
                               facecolor='darkgreen', alpha=0.7))

    # Input
    input_box = FancyBboxPatch((0.225 - 0.1, 0.15), 0.2, 0.15,
                               boxstyle="round,pad=0.01",
                               facecolor='lightyellow',
                               edgecolor='black',
                               linewidth=2)
    ax.add_patch(input_box)
    ax.text(0.225, 0.225, 'Same\nInput', ha='center', va='center',
            fontsize=11, fontweight='bold')

    # Arrows: Input to both models
    ax.arrow(0.225, 0.31, 0, 0.17, head_width=0.03, head_length=0.02,
             fc='black', ec='black', linewidth=2)
    ax.arrow(0.225, 0.225, 0.55, 0.35, head_width=0.03, head_length=0.02,
             fc='black', ec='black', linewidth=2)

    # Knowledge transfer arrow (the key!)
    ax.annotate('', xy=(0.6, 0.675), xytext=(0.4, 0.675),
                arrowprops=dict(arrowstyle='->', lw=4, color='blue'))

    # Knowledge transfer box
    transfer_box = FancyBboxPatch((0.42, 0.2), 0.16, 0.25,
                                  boxstyle="round,pad=0.01",
                                  facecolor='lightblue',
                                  edgecolor='blue',
                                  linewidth=3)
    ax.add_patch(transfer_box)
    ax.text(0.5, 0.4, 'Knowledge\nTransfer', ha='center', va='center',
            fontsize=12, fontweight='bold', color='darkblue')
    ax.text(0.5, 0.3, 'Soft labels\n+\nTeacher logits',
            ha='center', va='center', fontsize=10)

    # Arrow from teacher to transfer
    ax.arrow(0.225, 0.49, 0.2, -0.12, head_width=0.02, head_length=0.015,
             fc='blue', ec='blue', linewidth=2, linestyle='--')

    # Arrow from transfer to student
    ax.arrow(0.58, 0.325, 0.08, 0.15, head_width=0.02, head_length=0.015,
             fc='blue', ec='blue', linewidth=2, linestyle='--')

    # Comparison table at bottom
    comparison_box = FancyBboxPatch((0.1, 0.02), 0.8, 0.1,
                                    boxstyle="round,pad=0.01",
                                    facecolor='lightyellow',
                                    edgecolor='black',
                                    linewidth=2)
    ax.add_patch(comparison_box)

    ax.text(0.3, 0.07, 'Teacher: 340M params, 94% acc',
            ha='center', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.07, '→',
            ha='center', fontsize=16, fontweight='bold', color='blue')
    ax.text(0.7, 0.07, 'Student: 14M params (24× smaller), 92% acc',
            ha='center', fontsize=10, fontweight='bold', color='green')

    # Title
    ax.text(0.5, 0.95, 'Knowledge Distillation: Teacher Trains Student\nTransfer knowledge from large model to small model',
            ha='center', va='center', fontsize=16, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/distillation_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("5/8: distillation_architecture.pdf created")


# 6. Pruning Strategies
def plot_pruning_strategies():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Unstructured pruning
    ax = axes[0]

    # Create weight matrix visualization
    np.random.seed(42)
    weights = np.random.randn(10, 10)

    # Prune 70% of weights
    threshold = np.percentile(np.abs(weights), 70)
    weights_pruned = weights.copy()
    weights_pruned[np.abs(weights) < threshold] = 0

    im1 = ax.imshow(weights_pruned != 0, cmap='RdYlGn', aspect='auto')
    ax.set_title('Unstructured Pruning\n(Remove Individual Weights)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('70% weights removed\nRandom pattern', fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add grid
    for i in range(11):
        ax.axhline(i-0.5, color='black', linewidth=0.5)
        ax.axvline(i-0.5, color='black', linewidth=0.5)

    # Structured pruning
    ax = axes[1]

    # Remove entire columns (neurons)
    weights_structured = weights.copy()
    cols_to_remove = [2, 3, 5, 6, 7, 8, 9]  # 70%
    weights_structured[:, cols_to_remove] = 0

    im2 = ax.imshow(weights_structured != 0, cmap='RdYlGn', aspect='auto')
    ax.set_title('Structured Pruning\n(Remove Entire Neurons)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('70% neurons removed\nStructured pattern', fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add grid
    for i in range(11):
        ax.axhline(i-0.5, color='black', linewidth=0.5)
        ax.axvline(i-0.5, color='black', linewidth=0.5)

    # Overall title
    fig.suptitle('Pruning Strategies: Random vs Structured\nBoth reduce parameters, but structured is hardware-friendly',
                 fontsize=15, fontweight='bold', y=0.98)

    # Comparison text at bottom
    fig.text(0.25, 0.02, 'Unstructured:\n✓ Better accuracy\n✗ Needs sparse ops',
             ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.5',
             facecolor='lightyellow'))
    fig.text(0.75, 0.02, 'Structured:\n✓ Real speedup\n✗ Slightly lower accuracy',
             ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.5',
             facecolor='lightgreen'))

    plt.tight_layout()
    plt.savefig('../figures/pruning_strategies.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("6/8: pruning_strategies.pdf created")


# 7. Deployment Platforms Comparison
def plot_deployment_platforms():
    fig, ax = plt.subplots(figsize=(14, 9))

    # Platforms with (memory, latency_req, power, typical_models)
    platforms = [
        ('Server\n(A100)', 80, 1000, 400, ['GPT-3\nFull', 'BERT-Large\nFull']),
        ('Edge\n(Jetson)', 16, 100, 30, ['GPT-2\nINT8', 'BERT-Base\nINT8']),
        ('Mobile\n(Phone)', 4, 50, 5, ['DistilBERT', 'MobileBERT']),
        ('MCU\n(Arduino)', 0.512, 10, 0.1, ['Tiny\nKeyword', 'Quantized\nTiny'])
    ]

    x_pos = np.arange(len(platforms))

    # Memory bars
    memories = [p[1] for p in platforms]
    bars1 = ax.bar(x_pos - 0.3, memories, 0.25, label='Memory (GB)',
                   color=['darkred', 'orange', 'yellow', 'lightgreen'],
                   edgecolor='black', linewidth=2)

    # Latency bars (scaled for visualization)
    latencies = [p[2]/10 for p in platforms]  # Scale down
    bars2 = ax.bar(x_pos, latencies, 0.25, label='Max Latency (×10 ms)',
                   color='lightblue', edgecolor='black', linewidth=2)

    # Power bars (scaled)
    powers = [p[3] for p in platforms]
    bars3 = ax.bar(x_pos + 0.3, powers, 0.25, label='Power (W)',
                   color='lightcoral', edgecolor='black', linewidth=2)

    # Add value labels
    for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 5,
                f'{memories[i]:.1f}GB', ha='center', fontsize=9, fontweight='bold')
        ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 5,
                f'{platforms[i][2]}ms', ha='center', fontsize=9, fontweight='bold')
        ax.text(bar3.get_x() + bar3.get_width()/2, bar3.get_height() + 5,
                f'{powers[i]}W', ha='center', fontsize=9, fontweight='bold')

    # Platform labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p[0] for p in platforms], fontsize=12, fontweight='bold')

    # Add typical models below
    for i, platform in enumerate(platforms):
        models_text = '\n'.join(platform[4])
        ax.text(i, -30, models_text, ha='center', va='top',
                fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='gray'))

    ax.set_ylabel('Resources (log scale)', fontsize=14, fontweight='bold')
    ax.set_title('Deployment Platforms: Resources & Constraints\nDifferent platforms need different compression strategies',
                fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_yscale('log')
    ax.set_ylim(0.05, 500)

    set_minimalist_style(ax)

    plt.tight_layout()
    plt.savefig('../figures/deployment_platforms_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("7/8: deployment_platforms_comparison.pdf created")


# 8. Compression Decision Tree
def plot_compression_decision_tree():
    fig, ax = plt.subplots(figsize=(14, 11))

    # Decision nodes
    nodes = {
        'root': (0.5, 0.95, 'What is your\ntarget platform?'),
        'server': (0.15, 0.8, 'Server\n(80GB)'),
        'edge': (0.4, 0.8, 'Edge\n(16GB)'),
        'mobile': (0.65, 0.8, 'Mobile\n(4GB)'),
        'mcu': (0.9, 0.8, 'MCU\n(512MB)'),

        'server_acc': (0.15, 0.65, 'Need max\naccuracy?'),
        'server_yes': (0.08, 0.5, 'FP32\nFull model'),
        'server_no': (0.22, 0.5, 'INT8\nQuantization'),

        'edge_time': (0.4, 0.65, 'Need fast\ninference?'),
        'edge_yes': (0.33, 0.5, 'INT8 +\nPruning'),
        'edge_no': (0.47, 0.5, 'INT4\nQuantization'),

        'mobile_size': (0.65, 0.65, 'Size critical?'),
        'mobile_yes': (0.58, 0.5, 'Distillation\n+ INT8'),
        'mobile_no': (0.72, 0.5, 'INT8\nQuantization'),

        'mcu_task': (0.9, 0.65, 'Complex\ntask?'),
        'mcu_yes': (0.83, 0.5, 'INT4 +\nDistillation'),
        'mcu_no': (0.97, 0.5, 'Binary\nNetworks'),

        # Final results
        'result1': (0.08, 0.35, '700GB\n100% acc'),
        'result2': (0.22, 0.35, '175GB\n99% acc'),
        'result3': (0.33, 0.35, '50GB\n97% acc\nFast!'),
        'result4': (0.47, 0.35, '87GB\n97% acc'),
        'result5': (0.58, 0.35, '10GB\n92% acc'),
        'result6': (0.72, 0.35, '175GB\n99% acc'),
        'result7': (0.83, 0.35, '5GB\n90% acc'),
        'result8': (0.97, 0.35, '500MB\n85% acc'),
    }

    # Draw nodes
    for key, (x, y, text) in nodes.items():
        if key == 'root':
            color = 'lightblue'
            size = 0.08
        elif key in ['server', 'edge', 'mobile', 'mcu']:
            color = 'lightyellow'
            size = 0.06
        elif 'result' in key:
            color = 'lightgreen'
            size = 0.05
        else:
            color = 'lightcoral'
            size = 0.06

        circle = Circle((x, y), size, facecolor=color,
                       edgecolor='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=8 if 'result' in key else 9,
                fontweight='bold')

    # Draw connections
    connections = [
        ('root', 'server'), ('root', 'edge'), ('root', 'mobile'), ('root', 'mcu'),
        ('server', 'server_acc'), ('server_acc', 'server_yes'), ('server_acc', 'server_no'),
        ('edge', 'edge_time'), ('edge_time', 'edge_yes'), ('edge_time', 'edge_no'),
        ('mobile', 'mobile_size'), ('mobile_size', 'mobile_yes'), ('mobile_size', 'mobile_no'),
        ('mcu', 'mcu_task'), ('mcu_task', 'mcu_yes'), ('mcu_task', 'mcu_no'),
        ('server_yes', 'result1'), ('server_no', 'result2'),
        ('edge_yes', 'result3'), ('edge_no', 'result4'),
        ('mobile_yes', 'result5'), ('mobile_no', 'result6'),
        ('mcu_yes', 'result7'), ('mcu_no', 'result8'),
    ]

    for start, end in connections:
        x1, y1, _ = nodes[start]
        x2, y2, _ = nodes[end]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, alpha=0.5, zorder=1)

    # Add yes/no labels
    yes_no_labels = [
        ((0.08, 0.58), 'Yes'), ((0.22, 0.58), 'No'),
        ((0.33, 0.58), 'Yes'), ((0.47, 0.58), 'No'),
        ((0.58, 0.58), 'Yes'), ((0.72, 0.58), 'No'),
        ((0.83, 0.58), 'Yes'), ((0.97, 0.58), 'No'),
    ]

    for (x, y), label in yes_no_labels:
        ax.text(x, y, label, ha='center', fontsize=8,
                style='italic', color='blue')

    # Recommendations at bottom
    recs = [
        (0.15, 0.2, 'Server:\nQuantize if possible', 'blue'),
        (0.4, 0.2, 'Edge:\nINT4 or prune', 'green'),
        (0.65, 0.2, 'Mobile:\nDistill + quantize', 'purple'),
        (0.9, 0.2, 'MCU:\nExtreme compression', 'red'),
    ]

    for x, y, text, color in recs:
        ax.text(x, y, text, ha='center', fontsize=9,
                style='italic', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=color, linewidth=2))

    # Title
    ax.text(0.5, 0.99, 'Compression Decision Tree: Choose Your Strategy\nBased on Platform Constraints and Requirements',
            ha='center', fontsize=15, fontweight='bold')

    # Legend
    ax.text(0.02, 0.05, 'Decision', ha='left', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow'))
    ax.text(0.15, 0.05, 'Method', ha='left', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral'))
    ax.text(0.25, 0.05, 'Result', ha='left', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/compression_decision_tree.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("8/8: compression_decision_tree.pdf created")


# Generate all Week 11 BSc charts
if __name__ == "__main__":
    print("\nGenerating Week 11 BSc-level charts (Educational Framework)...")
    print("="*60)

    plot_model_size_impossibility()
    plot_compression_spectrum()
    plot_quantization_comparison()
    plot_accuracy_size_pareto()
    plot_distillation_architecture()
    plot_pruning_strategies()
    plot_deployment_platforms()
    plot_compression_decision_tree()

    print("="*60)
    print("All 8 BSc charts generated successfully!")
    print("Location: NLP_slides/week11_efficiency/figures/")
