import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches

# Set the minimalist style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Minimalist color palette
COLOR_MAIN = '#404040'      # RGB(64,64,64)
COLOR_ACCENT = '#B4B4B4'    # RGB(180,180,180)
COLOR_LIGHT = '#F0F0F0'     # RGB(240,240,240)
COLOR_HIGHLIGHT = '#FF6B6B'  # Red for emphasis
COLOR_BLUE = '#4682B4'       # Steel blue for checkpoints

def generate_linear_to_nonlinear_bridge():
    """Visual bridge showing why XOR needs hidden layers"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Linear separable (OR)
    ax = axes[0]
    ax.set_title('Single Perceptron\n(Linear Boundary)', fontsize=10, color=COLOR_MAIN)

    # OR data points
    ax.scatter([0, 1], [1, 1], c=COLOR_HIGHLIGHT, s=100, label='Output=1', zorder=3)
    ax.scatter([0], [0], c=COLOR_BLUE, s=100, label='Output=0', zorder=3)
    ax.scatter([1], [0], c=COLOR_HIGHLIGHT, s=100, zorder=3)

    # Decision boundary
    x = np.linspace(-0.2, 1.2, 100)
    y = 0.5 - x
    ax.plot(x, y, '--', color=COLOR_MAIN, linewidth=2, alpha=0.7)
    ax.fill_between(x, y, 1.2, alpha=0.1, color=COLOR_HIGHLIGHT)

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel('Input 1', fontsize=9, color=COLOR_MAIN)
    ax.set_ylabel('Input 2', fontsize=9, color=COLOR_MAIN)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', fontsize=8)

    # XOR problem
    ax = axes[1]
    ax.set_title('XOR Problem\n(No Single Line Works)', fontsize=10, color=COLOR_MAIN)

    # XOR data points
    ax.scatter([0, 1], [1, 0], c=COLOR_HIGHLIGHT, s=100, zorder=3)
    ax.scatter([0, 1], [0, 1], c=COLOR_BLUE, s=100, zorder=3)

    # Show multiple failed attempts
    for angle in [0, 45, 90, 135]:
        x = np.array([-0.2, 1.2])
        y = 0.5 + np.tan(np.radians(angle)) * (x - 0.5)
        ax.plot(x, y, '--', color=COLOR_ACCENT, linewidth=1, alpha=0.3)

    # Add question mark
    ax.text(0.5, 0.5, '?', fontsize=40, color=COLOR_ACCENT, ha='center', va='center', alpha=0.5)

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel('Input 1', fontsize=9, color=COLOR_MAIN)
    ax.set_ylabel('Input 2', fontsize=9, color=COLOR_MAIN)
    ax.grid(True, alpha=0.2)

    # Solution with hidden layer
    ax = axes[2]
    ax.set_title('Hidden Layer Solution\n(Combine Two Lines)', fontsize=10, color=COLOR_MAIN)

    # XOR data points
    ax.scatter([0, 1], [1, 0], c=COLOR_HIGHLIGHT, s=100, zorder=3)
    ax.scatter([0, 1], [0, 1], c=COLOR_BLUE, s=100, zorder=3)

    # Two decision boundaries
    x1 = np.linspace(-0.2, 1.2, 100)
    y1 = 0.9 - 0.8*x1  # First boundary
    y2 = 0.1 + 0.8*x1  # Second boundary

    ax.plot(x1, y1, '-', color=COLOR_MAIN, linewidth=2, alpha=0.7, label='Hidden 1')
    ax.plot(x1, y2, '-', color=COLOR_MAIN, linewidth=2, alpha=0.7, label='Hidden 2')

    # Fill XOR regions
    x_fill = np.linspace(-0.2, 1.2, 100)
    y_fill_upper = np.minimum(0.9 - 0.8*x_fill, 1.2)
    y_fill_lower = np.maximum(0.1 + 0.8*x_fill, -0.2)

    mask1 = (y_fill_upper > y_fill_lower)
    ax.fill_between(x_fill[mask1], y_fill_lower[mask1], y_fill_upper[mask1],
                    alpha=0.1, color=COLOR_HIGHLIGHT)

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel('Input 1', fontsize=9, color=COLOR_MAIN)
    ax.set_ylabel('Input 2', fontsize=9, color=COLOR_MAIN)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', fontsize=8)

    # Add arrows showing progression
    fig.text(0.35, 0.5, '→', fontsize=30, ha='center', color=COLOR_ACCENT)
    fig.text(0.64, 0.5, '→', fontsize=30, ha='center', color=COLOR_ACCENT)

    plt.tight_layout()
    plt.savefig('../figures/linear_to_nonlinear_bridge.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_architecture_decision_tree():
    """Decision tree for choosing neural network architecture"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define node positions
    nodes = {
        'start': (5, 9),
        'sequential': (2.5, 7),
        'spatial': (7.5, 7),
        'text': (1, 5),
        'timeseries': (4, 5),
        'image': (6.5, 5),
        'video': (8.5, 5),
        'transformer': (0.5, 3),
        'rnn': (2, 3),
        'lstm': (3.5, 3),
        'cnn': (6, 3),
        'cnn_rnn': (8, 3),
        'feedforward': (5, 2)
    }

    # Draw nodes
    for name, (x, y) in nodes.items():
        if name == 'start':
            # Start node
            box = FancyBboxPatch((x-0.8, y-0.3), 1.6, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=COLOR_BLUE, edgecolor=COLOR_MAIN,
                                linewidth=2, alpha=0.3)
            ax.add_patch(box)
            ax.text(x, y, 'Your Data', ha='center', va='center',
                   fontsize=11, fontweight='bold', color=COLOR_MAIN)
        elif name in ['sequential', 'spatial']:
            # Question nodes
            box = FancyBboxPatch((x-1, y-0.25), 2, 0.5,
                                boxstyle="round,pad=0.05",
                                facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN,
                                linewidth=1.5)
            ax.add_patch(box)
            text = 'Sequential?\n(Order matters)' if name == 'sequential' else 'Spatial?\n(2D/3D structure)'
            ax.text(x, y, text, ha='center', va='center',
                   fontsize=9, color=COLOR_MAIN)
        elif name in ['text', 'timeseries', 'image', 'video']:
            # Data type nodes
            circle = Circle((x, y), 0.35, facecolor='white',
                          edgecolor=COLOR_ACCENT, linewidth=1.5)
            ax.add_patch(circle)
            labels = {'text': 'Text', 'timeseries': 'Time\nSeries',
                     'image': 'Images', 'video': 'Video'}
            ax.text(x, y, labels[name], ha='center', va='center',
                   fontsize=8, color=COLOR_MAIN)
        else:
            # Architecture nodes
            box = Rectangle((x-0.5, y-0.25), 1, 0.5,
                          facecolor=COLOR_HIGHLIGHT if name in ['transformer', 'cnn'] else COLOR_ACCENT,
                          edgecolor=COLOR_MAIN, linewidth=1, alpha=0.2)
            ax.add_patch(box)
            labels = {
                'transformer': 'Transformer',
                'rnn': 'RNN',
                'lstm': 'LSTM/GRU',
                'cnn': 'CNN',
                'cnn_rnn': 'CNN+RNN',
                'feedforward': 'Feedforward'
            }
            ax.text(x, y, labels[name], ha='center', va='center',
                   fontsize=9, fontweight='bold' if name in ['transformer', 'cnn'] else 'normal',
                   color=COLOR_MAIN)

    # Draw connections with labels
    connections = [
        ('start', 'sequential', 'Yes'),
        ('start', 'spatial', 'No'),
        ('sequential', 'text', 'Language'),
        ('sequential', 'timeseries', 'Numerical'),
        ('spatial', 'image', 'Static'),
        ('spatial', 'video', 'Dynamic'),
        ('text', 'transformer', 'Best'),
        ('text', 'rnn', 'Good'),
        ('timeseries', 'lstm', 'Best'),
        ('timeseries', 'rnn', 'Simple'),
        ('image', 'cnn', 'Best'),
        ('video', 'cnn_rnn', 'Best'),
        ('start', 'feedforward', 'Tabular')
    ]

    for start, end, label in connections:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]

        # Draw arrow
        ax.annotate('', xy=(x2, y2 + 0.3 if y2 < y1 else y2 - 0.3),
                   xytext=(x1, y1 - 0.3 if y1 > y2 else y1 + 0.3),
                   arrowprops=dict(arrowstyle='->', color=COLOR_MAIN,
                                 lw=1.5 if label == 'Best' else 1,
                                 alpha=0.7))

        # Add label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        if label:
            ax.text(mid_x, mid_y, label, fontsize=7, color=COLOR_ACCENT,
                   ha='center', va='bottom', style='italic')

    # Add legend
    ax.text(5, 0.5, '* Transformer is now often best for all sequential data',
           fontsize=8, color=COLOR_ACCENT, ha='center', style='italic')

    # Add title
    ax.text(5, 9.7, 'Architecture Selection Guide', fontsize=14,
           fontweight='bold', color=COLOR_MAIN, ha='center')

    plt.tight_layout()
    plt.savefig('../figures/architecture_decision_tree.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all figures
print("Generating enhanced checkpoint figures...")
generate_linear_to_nonlinear_bridge()
print("Generated: linear_to_nonlinear_bridge.pdf")

generate_architecture_decision_tree()
print("Generated: architecture_decision_tree.pdf")

print("\nAll enhanced checkpoint figures generated successfully!")