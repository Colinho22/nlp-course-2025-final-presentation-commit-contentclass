"""
Generate architecture comparison visualizations
Shows size and complexity differences between different neural network architectures
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# ML Theme Colors
ML_BLUE = '#0066CC'
ML_PURPLE = '#3333B2'
ML_LAVENDER = '#ADADE0'
ML_ORANGE = '#FF7F0E'
ML_GREEN = '#2CA02C'
ML_RED = '#D62728'

def create_architecture_size_comparison():
    """Create visual comparison of architecture sizes"""

    fig, ax = plt.subplots(figsize=(16, 10))

    # Architecture data: name, params, year, color, position
    architectures = [
        ("Perceptron\n1957", 20, 1957, ML_BLUE, 0),
        ("NetTalk\n1987", 18000, 1987, ML_PURPLE, 1),
        ("LeNet\n1998", 60000, 1998, ML_GREEN, 2),
        ("AlexNet\n2012", 60e6, 2012, ML_ORANGE, 3),
        ("VGG-16\n2014", 138e6, 2014, ML_RED, 4),
        ("ResNet-50\n2015", 25e6, 2015, '#8B4789', 5),
        ("BERT-Base\n2018", 110e6, 2018, '#2E8B57', 6),
        ("GPT-2\n2019", 1.5e9, 2019, '#FF1493', 7),
        ("GPT-3\n2020", 175e9, 2020, '#4B0082', 8),
        ("GPT-4\n2023", 1.8e12, 2023, '#DC143C', 9),
    ]

    # Use logarithmic scale for parameters
    max_params = max([arch[1] for arch in architectures])

    # Draw nested boxes showing relative sizes
    for name, params, year, color, pos in architectures:
        # Calculate box size based on sqrt of params (for area representation)
        box_size = 0.5 + 5 * np.sqrt(params / max_params)

        # Position
        x = pos * 1.5
        y = 5

        # Draw box
        box = FancyBboxPatch((x - box_size/2, y - box_size/2),
                            box_size, box_size,
                            boxstyle="round,pad=0.05",
                            edgecolor=color, facecolor=color,
                            alpha=0.3, linewidth=3)
        ax.add_patch(box)

        # Add label inside
        ax.text(x, y, name, ha='center', va='center',
               fontsize=9, fontweight='bold', color='black')

        # Add parameter count below
        if params < 1e6:
            param_str = f"{int(params):,}"
        elif params < 1e9:
            param_str = f"{params/1e6:.0f}M"
        elif params < 1e12:
            param_str = f"{params/1e9:.0f}B"
        else:
            param_str = f"{params/1e12:.1f}T"

        ax.text(x, y - box_size/2 - 0.3, f"{param_str}\nparams",
               ha='center', va='top', fontsize=8, color=color, fontweight='bold')

    ax.set_xlim(-1, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Add title and subtitle
    ax.text(7.5, 11.5, 'Neural Network Evolution: From Perceptron to GPT-4',
           ha='center', fontsize=18, fontweight='bold')
    ax.text(7.5, 10.8, 'Box size represents relative number of parameters',
           ha='center', fontsize=12, color='gray')

    # Add timeline
    timeline_y = 1
    ax.plot([0, 13.5], [timeline_y, timeline_y], 'k-', linewidth=2, alpha=0.3)
    for name, params, year, color, pos in architectures:
        x = pos * 1.5
        ax.plot([x, x], [timeline_y - 0.1, timeline_y + 0.1], color=color, linewidth=2)
        ax.text(x, timeline_y - 0.5, str(year), ha='center', fontsize=8, rotation=45)

    # Add growth annotations
    ax.annotate('', xy=(13.5, 8), xytext=(0, 3),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.5))
    ax.text(7, 6, '10¹² parameter growth\nin 66 years!',
           fontsize=11, ha='center', color='gray', style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('../figures/architecture_size_comparison.pdf', dpi=300, bbox_inches='tight')
    print("Saved: architecture_size_comparison.pdf")

def create_architecture_type_comparison():
    """Create comparison of different architecture types"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Architecture types
    arch_types = [
        ("Feedforward\n(MLP)", "Fully connected\nlayers", "Classification,\nRegression"),
        ("Convolutional\n(CNN)", "Spatial feature\ndetection", "Images,\nVideo"),
        ("Recurrent\n(RNN/LSTM)", "Sequential\nprocessing", "Text,\nTime-series"),
        ("Transformer", "Self-attention\nmechanism", "Language,\nEverything"),
        ("Graph Neural\n(GNN)", "Graph structure\nlearning", "Social networks,\nMolecules"),
        ("Generative\n(GAN/VAE)", "Generate new\nsamples", "Images,\nAudio"),
    ]

    for idx, (name, mechanism, use_case) in enumerate(arch_types):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Draw simplified architecture diagram
        if "Feedforward" in name:
            # Draw fully connected network
            for i in range(3):
                for j in range(4):
                    ax.add_patch(Circle((i*0.3, j*0.2), 0.05, color=ML_BLUE, alpha=0.6))
                    if i < 2:
                        for k in range(4):
                            ax.plot([i*0.3, (i+1)*0.3], [j*0.2, k*0.2],
                                   'gray', alpha=0.3, linewidth=1)

        elif "Convolutional" in name:
            # Draw convolution layers
            for i in range(3):
                width = 0.15 - i*0.03
                rect = Rectangle((i*0.25, 0.2), width, 0.4,
                                edgecolor=ML_GREEN, facecolor=ML_GREEN,
                                alpha=0.4, linewidth=2)
                ax.add_patch(rect)

        elif "Recurrent" in name:
            # Draw recurrent connections
            for i in range(4):
                circle = Circle((i*0.2, 0.4), 0.05, color=ML_ORANGE, alpha=0.6)
                ax.add_patch(circle)
                if i < 3:
                    ax.arrow(i*0.2 + 0.05, 0.4, 0.1, 0, head_width=0.03,
                            head_length=0.02, fc=ML_ORANGE, ec=ML_ORANGE)
                    # Self-loop
                    ax.add_patch(mpatches.FancyBboxPatch((i*0.2 - 0.03, 0.45),
                                                        0.06, 0.08,
                                                        boxstyle="round,pad=0.01",
                                                        edgecolor=ML_ORANGE,
                                                        facecolor='none',
                                                        linewidth=1.5))

        elif "Transformer" in name:
            # Draw attention mechanism
            positions = [(0.2, 0.3), (0.4, 0.5), (0.6, 0.3), (0.4, 0.1)]
            for i, pos in enumerate(positions):
                ax.add_patch(Circle(pos, 0.05, color=ML_PURPLE, alpha=0.6))
                for j, pos2 in enumerate(positions):
                    if i != j:
                        ax.plot([pos[0], pos2[0]], [pos[1], pos2[1]],
                               'gray', alpha=0.2, linewidth=1)

        elif "Graph" in name:
            # Draw graph structure
            positions = [(0.3, 0.5), (0.5, 0.6), (0.7, 0.5),
                        (0.4, 0.3), (0.6, 0.3)]
            edges = [(0,1), (1,2), (0,3), (1,3), (2,4), (3,4)]

            for i, j in edges:
                ax.plot([positions[i][0], positions[j][0]],
                       [positions[i][1], positions[j][1]],
                       'gray', alpha=0.4, linewidth=2)

            for pos in positions:
                ax.add_patch(Circle(pos, 0.04, color=ML_RED, alpha=0.6))

        else:  # Generative
            # Draw generator and discriminator
            ax.add_patch(Rectangle((0.1, 0.2), 0.2, 0.3,
                                  edgecolor='#8B4789', facecolor='#8B4789',
                                  alpha=0.4, linewidth=2))
            ax.text(0.2, 0.35, 'G', ha='center', fontsize=14, fontweight='bold')

            ax.add_patch(Rectangle((0.5, 0.2), 0.2, 0.3,
                                  edgecolor='#2E8B57', facecolor='#2E8B57',
                                  alpha=0.4, linewidth=2))
            ax.text(0.6, 0.35, 'D', ha='center', fontsize=14, fontweight='bold')

            ax.arrow(0.32, 0.35, 0.15, 0, head_width=0.04, head_length=0.03,
                    fc='gray', ec='gray')

        # Add text annotations
        ax.text(0.5, 0.85, name, ha='center', fontsize=12, fontweight='bold')
        ax.text(0.5, 0.05, f"Use: {use_case}", ha='center', fontsize=9, style='italic')
        ax.text(0.5, -0.05, mechanism, ha='center', fontsize=8, color='gray')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    fig.suptitle('Neural Network Architecture Types', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/architecture_type_comparison.pdf', dpi=300, bbox_inches='tight')
    print("Saved: architecture_type_comparison.pdf")

def create_complexity_growth_chart():
    """Create chart showing complexity growth over time"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Data
    years = np.array([1957, 1987, 1998, 2012, 2014, 2015, 2018, 2019, 2020, 2023])
    params = np.array([20, 1.8e4, 6e4, 6e7, 1.38e8, 2.5e7, 1.1e8, 1.5e9, 1.75e11, 1.8e12])
    names = ['Perceptron', 'NetTalk', 'LeNet', 'AlexNet', 'VGG-16',
            'ResNet-50', 'BERT', 'GPT-2', 'GPT-3', 'GPT-4']

    # Plot 1: Parameters over time (log scale)
    ax1.semilogy(years, params, 'o-', linewidth=2, markersize=8, color=ML_BLUE)

    # Annotate key points
    for i, (year, param, name) in enumerate(zip(years, params, names)):
        if i % 2 == 0 or i >= 7:  # Annotate every other and the last few
            ax1.annotate(name, (year, param), xytext=(10, 5),
                        textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Number of Parameters (log scale)', fontsize=12)
    ax1.set_title('Exponential Growth in Model Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add doubling time annotation
    ax1.text(2000, 1e10, 'Doubling time:\n~2.4 years',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    # Plot 2: Compute requirements (estimated)
    compute_flops = params * 6  # Rough estimate: 6 FLOPs per parameter per token

    ax2.loglog(params, compute_flops, 's-', linewidth=2, markersize=8, color=ML_ORANGE)

    # Annotate
    for i, (param, flop, name) in enumerate(zip(params, compute_flops, names)):
        if i % 3 == 0 or i >= 8:
            ax2.annotate(name, (param, flop), xytext=(10, 5),
                        textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))

    ax2.set_xlabel('Parameters', fontsize=12)
    ax2.set_ylabel('Training Compute (FLOPs estimate)', fontsize=12)
    ax2.set_title('Scaling: Parameters vs Compute', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add scaling law line
    x_line = np.logspace(1, 13, 100)
    y_line = x_line * 6
    ax2.plot(x_line, y_line, '--', color='gray', alpha=0.5, linewidth=2, label='Linear scaling')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('../figures/complexity_growth_chart.pdf', dpi=300, bbox_inches='tight')
    print("Saved: complexity_growth_chart.pdf")

if __name__ == "__main__":
    print("Generating architecture comparison visualizations...")

    create_architecture_size_comparison()
    create_architecture_type_comparison()
    create_complexity_growth_chart()

    print("\nAll architecture comparison visualizations generated successfully!")
    print("Files created:")
    print("  - architecture_size_comparison.pdf")
    print("  - architecture_type_comparison.pdf")
    print("  - complexity_growth_chart.pdf")