"""
Generate simple neuron visualizations for beginners
Clear, intuitive diagrams explaining what neurons do
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, FancyBboxPatch

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# ML Theme Colors
ML_BLUE = '#0066CC'
ML_ORANGE = '#FF7F0E'
ML_GREEN = '#2CA02C'
ML_RED = '#D62728'
ML_PURPLE = '#9467BD'

def create_single_neuron_anatomy():
    """Create diagram showing anatomy of a single neuron"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Basic neuron structure
    ax = axes[0, 0]

    # Draw inputs
    input_positions = [(-2, 1), (-2, 0), (-2, -1)]
    for i, pos in enumerate(input_positions):
        circle = Circle(pos, 0.2, facecolor=ML_BLUE, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos[0] - 0.5, pos[1], f'Input {i+1}', ha='right', fontsize=10)
        ax.text(pos[0], pos[1], f'x{i+1}', ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

        # Connection to neuron body
        arrow = FancyArrowPatch(pos, (0, 0),
                               connectionstyle="arc3,rad=.2",
                               arrowstyle='->', mutation_scale=15,
                               linewidth=2, color='gray')
        ax.add_patch(arrow)
        # Weight label
        ax.text((pos[0] + 0) / 2, pos[1] / 2, f'w{i+1}',
               ha='center', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6))

    # Neuron body
    circle = Circle((0, 0), 0.5, facecolor=ML_ORANGE, edgecolor='black', linewidth=3)
    ax.add_patch(circle)
    ax.text(0, 0, 'Σ', ha='center', va='center',
           fontsize=24, fontweight='bold', color='white')
    ax.text(0, -1, 'Neuron\n(Sum + Activate)', ha='center', fontsize=9)

    # Output
    arrow = FancyArrowPatch((0.5, 0), (2, 0),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=3, color=ML_GREEN)
    ax.add_patch(arrow)
    circle = Circle((2.5, 0), 0.3, facecolor=ML_GREEN, edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(2.5, 0, 'y', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    ax.text(2.5, -0.6, 'Output', ha='center', fontsize=9)

    ax.set_title('Anatomy of a Single Neuron', fontsize=14, fontweight='bold')
    ax.set_xlim(-3, 3.5)
    ax.set_ylim(-2, 2)
    ax.axis('off')

    # Panel 2: What happens inside
    ax = axes[0, 1]

    # Show computation steps
    steps = [
        ("1. Weighted Sum", ML_BLUE),
        ("z = Σ(wi × xi) + bias", 'black'),
        ("", 'black'),
        ("2. Apply Activation", ML_ORANGE),
        ("y = f(z)", 'black'),
        ("", 'black'),
        ("3. Output Result", ML_GREEN),
    ]

    y_pos = 1.5
    for text, color in steps:
        if text:
            fontweight = 'bold' if any(c in text for c in '123') else 'normal'
            fontsize = 11 if fontweight == 'bold' else 10
            ax.text(0, y_pos, text, fontsize=fontsize, fontweight=fontweight,
                   color=color, ha='left')
        y_pos -= 0.35

    # Draw visual flowchart
    boxes = [
        (0.1, -0.5, 'Inputs\n×\nWeights', ML_BLUE),
        (0.1, -1.2, 'Add\nBias', ML_PURPLE),
        (0.1, -1.9, 'Apply\nActivation', ML_ORANGE),
    ]

    for i, (x, y, text, color) in enumerate(boxes):
        box = FancyBboxPatch((x, y), 0.8, 0.5,
                            boxstyle="round,pad=0.05",
                            edgecolor=color, facecolor=color,
                            alpha=0.3, linewidth=2)
        ax.add_patch(box)
        ax.text(x + 0.4, y + 0.25, text, ha='center', va='center',
               fontsize=9, fontweight='bold')

        if i < len(boxes) - 1:
            arrow = FancyArrowPatch((x + 0.4, y), (x + 0.4, boxes[i+1][1] + 0.5),
                                   arrowstyle='->', mutation_scale=15,
                                   linewidth=2, color='gray')
            ax.add_patch(arrow)

    ax.set_title('What Happens Inside a Neuron', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-2.5, 2)
    ax.axis('off')

    # Panel 3: Concrete example
    ax = axes[1, 0]

    # Example values
    x = [2, 3, 1]
    w = [0.5, -0.3, 0.8]
    b = 0.1

    # Show calculation
    ax.text(0.5, 0.9, 'Example: Is this fruit ripe?', ha='center',
           fontsize=12, fontweight='bold', transform=ax.transAxes)

    calc_text = [
        "Inputs:",
        f"  x₁ = {x[0]} (color: red=high)",
        f"  x₂ = {x[1]} (softness: high)",
        f"  x₃ = {x[2]} (smell: low)",
        "",
        "Learned Weights:",
        f"  w₁ = {w[0]} (color matters)",
        f"  w₂ = {w[1]} (softness negative)",
        f"  w₃ = {w[2]} (smell important)",
        f"  bias = {b}",
        "",
        "Calculation:",
        f"z = ({x[0]}×{w[0]}) + ({x[1]}×{w[1]}) + ({x[2]}×{w[2]}) + {b}",
        f"z = {x[0]*w[0]:.1f} + {x[1]*w[1]:.1f} + {x[2]*w[2]:.1f} + {b}",
        f"z = {sum([xi*wi for xi, wi in zip(x, w)]) + b:.1f}",
        "",
        "Activation (sigmoid):",
        f"y = 1/(1 + e^(-z)) = 0.76",
        "",
        "Decision: 76% confident it's ripe!"
    ]

    y_pos = 0.75
    for line in calc_text:
        if line == "":
            y_pos -= 0.03
        elif "Inputs:" in line or "Weights:" in line or "Calculation:" in line or "Activation:" in line:
            ax.text(0.05, y_pos, line, fontsize=10, fontweight='bold',
                   transform=ax.transAxes)
            y_pos -= 0.04
        elif "Decision:" in line:
            ax.text(0.05, y_pos, line, fontsize=11, fontweight='bold',
                   color=ML_GREEN, transform=ax.transAxes,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))
            y_pos -= 0.05
        else:
            ax.text(0.08, y_pos, line, fontsize=9, family='monospace',
                   transform=ax.transAxes)
            y_pos -= 0.03

    ax.set_title('Concrete Example', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Panel 4: Different activation functions
    ax = axes[1, 1]

    x_range = np.linspace(-3, 3, 100)

    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x_range))
    ax.plot(x_range, sigmoid, linewidth=2, color=ML_BLUE, label='Sigmoid')

    # ReLU
    relu = np.maximum(0, x_range)
    ax.plot(x_range, relu, linewidth=2, color=ML_GREEN, label='ReLU')

    # Tanh
    tanh = np.tanh(x_range)
    ax.plot(x_range, tanh, linewidth=2, color=ML_ORANGE, label='Tanh')

    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.2)
    ax.set_xlabel('Input (z)', fontsize=11)
    ax.set_ylabel('Output (y)', fontsize=11)
    ax.set_title('Common Activation Functions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Understanding a Single Neuron', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/single_neuron_explained.pdf', dpi=300, bbox_inches='tight')
    print("Saved: single_neuron_explained.pdf")

def create_neuron_as_classifier():
    """Show how a single neuron acts as a classifier"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Decision boundary
    ax = axes[0]

    # Generate sample data
    np.random.seed(42)
    n_points = 30

    # Class 1 (below line)
    x1_class1 = np.random.randn(n_points) - 0.5
    x2_class1 = np.random.randn(n_points) - 0.5
    ax.scatter(x1_class1, x2_class1, c=ML_BLUE, s=50, alpha=0.6,
              edgecolors='black', linewidth=1, label='Class 0')

    # Class 2 (above line)
    x1_class2 = np.random.randn(n_points) + 0.5
    x2_class2 = np.random.randn(n_points) + 0.5
    ax.scatter(x1_class2, x2_class2, c=ML_RED, s=50, alpha=0.6,
              edgecolors='black', linewidth=1, label='Class 1')

    # Decision boundary: w1*x1 + w2*x2 + b = 0
    # Let's use w1=1, w2=1, b=0
    x_line = np.linspace(-3, 3, 100)
    y_line = -x_line  # From w1*x1 + w2*x2 = 0

    ax.plot(x_line, y_line, 'k--', linewidth=3, label='Decision boundary')

    # Add regions
    ax.fill_between(x_line, y_line, 3, alpha=0.1, color=ML_RED)
    ax.fill_between(x_line, y_line, -3, alpha=0.1, color=ML_BLUE)

    ax.set_xlabel('Feature x₁', fontsize=11)
    ax.set_ylabel('Feature x₂', fontsize=11)
    ax.set_title('Single Neuron = Linear Classifier', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    # Add equation
    ax.text(0, 2.5, 'w₁x₁ + w₂x₂ + b = 0', ha='center',
           fontsize=10, bbox=dict(boxstyle='round,pad=0.3',
                                 facecolor='yellow', alpha=0.7))

    # Panel 2: Weights determine orientation
    ax = axes[1]

    # Show different orientations
    angles = [0, 45, 90, 135]
    colors = [ML_BLUE, ML_GREEN, ML_ORANGE, ML_RED]

    for angle, color in zip(angles, colors):
        rad = np.radians(angle)
        w1, w2 = np.cos(rad), np.sin(rad)

        # Decision boundary
        if abs(w2) > 0.01:
            x_line = np.linspace(-2, 2, 100)
            y_line = -(w1 / w2) * x_line
            mask = (y_line >= -2) & (y_line <= 2)
            ax.plot(x_line[mask], y_line[mask], linewidth=2, color=color,
                   label=f'w=({w1:.1f}, {w2:.1f})')
        else:
            ax.axvline(x=0, linewidth=2, color=color,
                      label=f'w=({w1:.1f}, {w2:.1f})')

    ax.set_xlabel('Feature x₁', fontsize=11)
    ax.set_ylabel('Feature x₂', fontsize=11)
    ax.set_title('Weights Control Boundary Angle', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # Panel 3: Bias shifts the boundary
    ax = axes[2]

    # Fixed weights, different biases
    biases = [-1, 0, 1]
    colors = [ML_BLUE, ML_GREEN, ML_RED]

    for bias, color in zip(biases, colors):
        x_line = np.linspace(-2, 2, 100)
        y_line = -x_line - bias
        ax.plot(x_line, y_line, linewidth=2, color=color,
               label=f'bias = {bias}')

    # Add arrow showing direction
    ax.annotate('', xy=(0, 1.2), xytext=(0, -1.2),
               arrowprops=dict(arrowstyle='<->', lw=2, color='gray'))
    ax.text(0.3, 0, 'Bias shifts\nboundary', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax.set_xlabel('Feature x₁', fontsize=11)
    ax.set_ylabel('Feature x₂', fontsize=11)
    ax.set_title('Bias Shifts Boundary Position', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    fig.suptitle('Single Neuron as a Classifier: Weights and Bias', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/neuron_as_classifier.pdf', dpi=300, bbox_inches='tight')
    print("Saved: neuron_as_classifier.pdf")

def create_learning_process_simple():
    """Show how a neuron learns through examples"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Example: Learning AND logic
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 0, 0, 1])  # AND logic

    # Learning steps
    epochs = [0, 2, 5, 10]

    for idx, epoch in enumerate(epochs):
        ax = axes[idx // 2, idx % 2]

        # Simulate learning (weights approach correct values)
        progress = min(1.0, epoch / 10)
        w1 = progress * 1.5
        w2 = progress * 1.5
        b = -progress * 2.0

        # Plot data points
        for i, (x, y) in enumerate(zip(X_train, y_train)):
            color = ML_RED if y == 1 else ML_BLUE
            ax.scatter(x[0], x[1], c=color, s=200, alpha=0.6,
                      edgecolors='black', linewidth=2, zorder=5)
            ax.text(x[0], x[1] - 0.15, f'({x[0]},{x[1]})',
                   ha='center', fontsize=9)

        # Plot decision boundary
        if abs(w1) > 0.01 or abs(w2) > 0.01:
            x_line = np.linspace(-0.5, 1.5, 100)
            if abs(w2) > 0.01:
                y_line = -(w1 * x_line + b) / w2
                mask = (y_line >= -0.5) & (y_line <= 1.5)
                ax.plot(x_line[mask], y_line[mask], 'k--', linewidth=3,
                       label='Current boundary')

        # Calculate predictions and accuracy
        predictions = []
        for x in X_train:
            z = w1 * x[0] + w2 * x[1] + b
            pred = 1 if z > 0 else 0
            predictions.append(pred)

        accuracy = np.mean(np.array(predictions) == y_train)

        ax.set_xlabel('Input x₁', fontsize=11)
        ax.set_ylabel('Input x₂', fontsize=11)
        ax.set_title(f'Epoch {epoch} - Accuracy: {accuracy:.0%}', fontsize=12, fontweight='bold')
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.grid(True, alpha=0.3)

        # Add weights info
        info_text = f'w₁={w1:.2f}\nw₂={w2:.2f}\nb={b:.2f}'
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
               ha='right', va='top', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('How a Neuron Learns: Training Progress on AND Logic',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/neuron_learning_progress.pdf', dpi=300, bbox_inches='tight')
    print("Saved: neuron_learning_progress.pdf")

if __name__ == "__main__":
    print("Generating simple neuron visualizations...")

    create_single_neuron_anatomy()
    create_neuron_as_classifier()
    create_learning_process_simple()

    print("\nAll simple neuron visualizations generated successfully!")
    print("Files created:")
    print("  - single_neuron_explained.pdf")
    print("  - neuron_as_classifier.pdf")
    print("  - neuron_learning_progress.pdf")