"""
Generate foundational neural network figures for Week 3 RNN presentation.
These figures introduce basic concepts before diving into RNNs.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
from matplotlib.patches import ConnectionPatch
import seaborn as sns

# Set consistent style
plt.style.use('seaborn-v0_8-whitegrid')

# Monochromatic color scheme
COLOR_MAIN = '#404040'     # RGB(64,64,64)
COLOR_ACCENT = '#B4B4B4'   # RGB(180,180,180)
COLOR_LIGHT = '#F0F0F0'    # RGB(240,240,240)
COLOR_HIGHLIGHT = '#FF6B6B'  # For emphasis
COLOR_SUCCESS = '#95E77E'    # For positive
COLOR_WARNING = '#FFB366'    # For warning

def create_biological_neuron_simple():
    """Create a simple biological neuron diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Draw cell body
    body = Circle((0.5, 0.5), 0.15, color=COLOR_ACCENT, ec=COLOR_MAIN, linewidth=2)
    ax.add_patch(body)

    # Draw dendrites (inputs)
    dendrite_positions = [(0.1, 0.7), (0.1, 0.5), (0.1, 0.3)]
    for pos in dendrite_positions:
        arrow = FancyArrowPatch(pos, (0.35, 0.5),
                              arrowstyle='->', mutation_scale=20,
                              color=COLOR_MAIN, linewidth=1.5)
        ax.add_patch(arrow)
        ax.text(pos[0]-0.05, pos[1], 'Input', fontsize=9, ha='right')

    # Draw axon (output)
    arrow = FancyArrowPatch((0.65, 0.5), (0.9, 0.5),
                          arrowstyle='->', mutation_scale=20,
                          color=COLOR_MAIN, linewidth=2)
    ax.add_patch(arrow)
    ax.text(0.95, 0.5, 'Output', fontsize=9, ha='left')

    # Labels
    ax.text(0.5, 0.5, 'Cell\nBody', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.15, 'Biological Neuron', ha='center', fontsize=12, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/biological_neuron_simple.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_mathematical_neuron():
    """Create a mathematical neuron diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Input nodes
    input_y = [0.7, 0.5, 0.3]
    for i, y in enumerate(input_y, 1):
        circle = Circle((0.2, y), 0.05, color=COLOR_LIGHT, ec=COLOR_MAIN, linewidth=1.5)
        ax.add_patch(circle)
        ax.text(0.2, y, f'$x_{i}$', ha='center', va='center', fontsize=10)

        # Weight arrows
        arrow = FancyArrowPatch((0.25, y), (0.45, 0.5),
                              arrowstyle='->', mutation_scale=15,
                              color=COLOR_MAIN, linewidth=1.5)
        ax.add_patch(arrow)
        ax.text(0.35, y + (0.5-y)*0.5, f'$w_{i}$', fontsize=9, color=COLOR_ACCENT)

    # Neuron body
    neuron = Circle((0.5, 0.5), 0.08, color=COLOR_ACCENT, ec=COLOR_MAIN, linewidth=2)
    ax.add_patch(neuron)
    ax.text(0.5, 0.5, '$\\Sigma$', ha='center', va='center', fontsize=12, fontweight='bold')

    # Activation function
    rect = FancyBboxPatch((0.6, 0.45), 0.1, 0.1,
                          boxstyle="round,pad=0.01",
                          facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(0.65, 0.5, '$f$', ha='center', va='center', fontsize=11)

    # Output arrow
    arrow = FancyArrowPatch((0.7, 0.5), (0.85, 0.5),
                          arrowstyle='->', mutation_scale=15,
                          color=COLOR_MAIN, linewidth=2)
    ax.add_patch(arrow)
    ax.text(0.9, 0.5, '$y$', fontsize=11, ha='left')

    # Formula
    ax.text(0.5, 0.15, '$y = f(\\sum w_i x_i + b)$', ha='center', fontsize=12, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/mathematical_neuron.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_single_neuron_computation():
    """Create step-by-step single neuron computation."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Draw computation steps
    steps_y = [0.85, 0.7, 0.55, 0.4, 0.25]
    steps_text = [
        "1. Inputs: $x_1=0.5$, $x_2=0.3$",
        "2. Weights: $w_1=2.0$, $w_2=-1.0$",
        "3. Weighted sum: $(0.5 \\times 2.0) + (0.3 \\times -1.0) = 0.7$",
        "4. Add bias: $0.7 + 0.1 = 0.8$",
        "5. Apply activation: $\\tanh(0.8) = 0.664$"
    ]

    for y, text in zip(steps_y, steps_text):
        # Step box
        rect = FancyBboxPatch((0.1, y-0.06), 0.8, 0.08,
                              boxstyle="round,pad=0.02",
                              facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN, linewidth=1)
        ax.add_patch(rect)
        ax.text(0.5, y-0.02, text, ha='center', va='center', fontsize=11)

        if y > steps_y[-1]:
            # Arrow to next step
            arrow = FancyArrowPatch((0.5, y-0.08), (0.5, y-0.12),
                                  arrowstyle='->', mutation_scale=15,
                                  color=COLOR_ACCENT, linewidth=1.5)
            ax.add_patch(arrow)

    # Title
    ax.text(0.5, 0.95, 'Single Neuron Computation', ha='center', fontsize=14, fontweight='bold')

    # Result highlight
    result_rect = FancyBboxPatch((0.3, 0.05), 0.4, 0.1,
                                boxstyle="round,pad=0.02",
                                facecolor=COLOR_SUCCESS, alpha=0.3,
                                edgecolor=COLOR_MAIN, linewidth=2)
    ax.add_patch(result_rect)
    ax.text(0.5, 0.1, 'Output: 0.664', ha='center', fontsize=12, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/single_neuron_computation.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_activation_functions_comparison():
    """Create comparison of different activation functions."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    x = np.linspace(-3, 3, 100)

    # Sigmoid
    ax = axes[0, 0]
    y = 1 / (1 + np.exp(-x))
    ax.plot(x, y, color=COLOR_MAIN, linewidth=2)
    ax.fill_between(x, 0, y, alpha=0.1, color=COLOR_MAIN)
    ax.axhline(y=0, color=COLOR_ACCENT, linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color=COLOR_ACCENT, linestyle='--', linewidth=0.5)
    ax.set_title('Sigmoid: $\\sigma(x) = \\frac{1}{1+e^{-x}}$', fontsize=11)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.text(2, 0.5, 'Range: [0,1]', fontsize=9, color=COLOR_ACCENT)

    # Tanh
    ax = axes[0, 1]
    y = np.tanh(x)
    ax.plot(x, y, color=COLOR_HIGHLIGHT, linewidth=2)
    ax.fill_between(x, 0, y, alpha=0.1, color=COLOR_HIGHLIGHT)
    ax.axhline(y=0, color=COLOR_ACCENT, linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color=COLOR_ACCENT, linestyle='--', linewidth=0.5)
    ax.set_title('Tanh: $\\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$', fontsize=11)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.text(1.5, 0, 'Range: [-1,1]', fontsize=9, color=COLOR_ACCENT)

    # ReLU
    ax = axes[1, 0]
    y = np.maximum(0, x)
    ax.plot(x, y, color=COLOR_SUCCESS, linewidth=2)
    ax.fill_between(x, 0, y, alpha=0.1, color=COLOR_SUCCESS)
    ax.axhline(y=0, color=COLOR_ACCENT, linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color=COLOR_ACCENT, linestyle='--', linewidth=0.5)
    ax.set_title('ReLU: $\\max(0, x)$', fontsize=11)
    ax.set_ylim(-0.5, 3)
    ax.grid(True, alpha=0.3)
    ax.text(1.5, 1.5, 'Range: [0,∞)', fontsize=9, color=COLOR_ACCENT)

    # Linear
    ax = axes[1, 1]
    y = x
    ax.plot(x, y, color=COLOR_WARNING, linewidth=2, linestyle='--')
    ax.fill_between(x, 0, y, alpha=0.1, color=COLOR_WARNING)
    ax.axhline(y=0, color=COLOR_ACCENT, linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color=COLOR_ACCENT, linestyle='--', linewidth=0.5)
    ax.set_title('Linear: $f(x) = x$ (No activation)', fontsize=11)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.3)
    ax.text(1.5, -1.5, 'Range: (-∞,∞)', fontsize=9, color=COLOR_ACCENT)

    for ax in axes.flat:
        ax.set_xlabel('Input', fontsize=9)
        ax.set_ylabel('Output', fontsize=9)

    plt.suptitle('Activation Functions Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/activation_functions_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_linear_vs_nonlinear():
    """Show why we need non-linearity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Linear separation
    x = np.linspace(-2, 2, 100)
    y = 0.5 * x + 0.5

    ax1.plot(x, y, color=COLOR_MAIN, linewidth=2, label='Decision boundary')
    ax1.scatter([0.5, 1, 1.5], [0.5, 1, 1.5], color=COLOR_SUCCESS, s=50, label='Class A')
    ax1.scatter([-0.5, -1, -1.5], [0, -0.5, -1], color=COLOR_HIGHLIGHT, s=50, label='Class B')
    ax1.set_title('Linear: Can separate simple patterns', fontsize=11)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # Non-linear separation (XOR-like)
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.8
    x_circle = r * np.cos(theta)
    y_circle = r * np.sin(theta)

    ax2.plot(x_circle, y_circle, color=COLOR_MAIN, linewidth=2, label='Decision boundary')
    ax2.scatter([0.5, -0.5], [0.5, -0.5], color=COLOR_SUCCESS, s=50, label='Class A')
    ax2.scatter([0.5, -0.5], [-0.5, 0.5], color=COLOR_HIGHLIGHT, s=50, label='Class B')
    ax2.set_title('Non-linear: Can separate complex patterns', fontsize=11)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    for ax in [ax1, ax2]:
        ax.set_xlabel('Feature 1', fontsize=9)
        ax.set_ylabel('Feature 2', fontsize=9)
        ax.axhline(y=0, color=COLOR_ACCENT, linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color=COLOR_ACCENT, linestyle='--', linewidth=0.5)

    plt.suptitle('Linear vs Non-linear Decision Boundaries', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/linear_vs_nonlinear.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_tanh_detailed():
    """Create detailed tanh visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    x = np.linspace(-4, 4, 200)
    y = np.tanh(x)
    y_prime = 1 - np.tanh(x)**2

    # Tanh function
    ax1.plot(x, y, color=COLOR_MAIN, linewidth=2.5, label='tanh(x)')
    ax1.fill_between(x, 0, y, alpha=0.1, color=COLOR_MAIN)
    ax1.axhline(y=1, color=COLOR_ACCENT, linestyle='--', linewidth=1, label='y=1')
    ax1.axhline(y=-1, color=COLOR_ACCENT, linestyle='--', linewidth=1, label='y=-1')
    ax1.axhline(y=0, color=COLOR_ACCENT, linestyle='-', linewidth=0.5)
    ax1.axvline(x=0, color=COLOR_ACCENT, linestyle='-', linewidth=0.5)

    # Annotate regions
    ax1.text(2.5, 0.5, 'Saturates\n→ +1', fontsize=9, ha='center', color=COLOR_HIGHLIGHT)
    ax1.text(-2.5, -0.5, 'Saturates\n→ -1', fontsize=9, ha='center', color=COLOR_HIGHLIGHT)
    ax1.text(0, -0.3, 'Linear\nregion', fontsize=9, ha='center', color=COLOR_SUCCESS)

    ax1.set_title('Tanh Activation Function', fontsize=12)
    ax1.set_xlabel('Input (x)', fontsize=10)
    ax1.set_ylabel('Output tanh(x)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_ylim(-1.2, 1.2)

    # Derivative
    ax2.plot(x, y_prime, color=COLOR_HIGHLIGHT, linewidth=2.5, label="tanh'(x)")
    ax2.fill_between(x, 0, y_prime, alpha=0.1, color=COLOR_HIGHLIGHT)
    ax2.axhline(y=0, color=COLOR_ACCENT, linestyle='-', linewidth=0.5)
    ax2.axvline(x=0, color=COLOR_ACCENT, linestyle='-', linewidth=0.5)

    # Mark maximum
    ax2.plot(0, 1, 'o', color=COLOR_SUCCESS, markersize=8, label='Max gradient=1')
    ax2.text(0, 0.5, 'Strong\ngradient', fontsize=9, ha='center', color=COLOR_SUCCESS)
    ax2.text(2.5, 0.1, 'Vanishing\ngradient', fontsize=9, ha='center', color=COLOR_WARNING)
    ax2.text(-2.5, 0.1, 'Vanishing\ngradient', fontsize=9, ha='center', color=COLOR_WARNING)

    ax2.set_title('Tanh Derivative (Gradient)', fontsize=12)
    ax2.set_xlabel('Input (x)', fontsize=10)
    ax2.set_ylabel("Derivative (1 - tanh²(x))", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_ylim(-0.1, 1.1)

    plt.suptitle('Understanding Tanh for RNNs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/tanh_detailed.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_matrix_organization():
    """Show how matrices organize neural connections."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Draw progression: single → layer → matrix

    # Single neuron
    x_offset = 0.15
    ax.text(x_offset, 0.9, 'Single Neuron', fontsize=11, ha='center', fontweight='bold')
    circle = Circle((x_offset, 0.7), 0.02, color=COLOR_LIGHT, ec=COLOR_MAIN)
    ax.add_patch(circle)
    ax.text(x_offset, 0.55, '6 weights', fontsize=9, ha='center', color=COLOR_ACCENT)

    # Arrow
    arrow = FancyArrowPatch((x_offset + 0.05, 0.7), (x_offset + 0.15, 0.7),
                          arrowstyle='->', mutation_scale=20, color=COLOR_MAIN)
    ax.add_patch(arrow)

    # Layer of neurons
    x_offset = 0.4
    ax.text(x_offset, 0.9, 'Layer (4 neurons)', fontsize=11, ha='center', fontweight='bold')
    for i in range(4):
        y = 0.8 - i * 0.08
        circle = Circle((x_offset, y), 0.02, color=COLOR_LIGHT, ec=COLOR_MAIN)
        ax.add_patch(circle)
    ax.text(x_offset, 0.45, '24 weights!', fontsize=9, ha='center', color=COLOR_ACCENT)

    # Arrow
    arrow = FancyArrowPatch((x_offset + 0.05, 0.7), (x_offset + 0.15, 0.7),
                          arrowstyle='->', mutation_scale=20, color=COLOR_MAIN)
    ax.add_patch(arrow)

    # Matrix representation
    x_offset = 0.65
    ax.text(x_offset, 0.9, 'Matrix Notation', fontsize=11, ha='center', fontweight='bold')

    # Draw matrix
    matrix_rect = Rectangle((x_offset - 0.06, 0.55), 0.12, 0.2,
                           facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN, linewidth=2)
    ax.add_patch(matrix_rect)

    # Matrix elements
    for i in range(4):
        for j in range(6):
            x = x_offset - 0.05 + j * 0.02
            y = 0.72 - i * 0.04
            ax.text(x, y, '•', fontsize=6, ha='center', va='center', color=COLOR_MAIN)

    ax.text(x_offset, 0.5, '$W_{4×6}$', fontsize=11, ha='center', fontweight='bold')
    ax.text(x_offset, 0.4, 'All weights\nin one object!', fontsize=9, ha='center', color=COLOR_SUCCESS)

    # Matrix multiplication example
    ax.text(0.5, 0.25, 'Matrix Multiplication:', fontsize=11, fontweight='bold')
    ax.text(0.5, 0.15, '$y = Wx + b$', fontsize=12, ha='center')
    ax.text(0.5, 0.05, 'Computes all neurons at once!', fontsize=10, ha='center', color=COLOR_SUCCESS)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/matrix_organization.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_learning_process_analogy():
    """Create dart throwing learning analogy."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))

    # Dartboard parameters
    center = (0.5, 0.5)
    radius = 0.3

    titles = ['Initial Throw', 'Compute Error', 'Adjust Aim', 'Improved Throw']
    dart_positions = [(0.7, 0.3), (0.7, 0.3), (0.6, 0.4), (0.52, 0.48)]

    for ax, title, dart_pos in zip(axes, titles, dart_positions):
        # Draw dartboard
        for r, c in zip([0.3, 0.2, 0.1, 0.05], [COLOR_LIGHT, 'white', COLOR_LIGHT, COLOR_HIGHLIGHT]):
            circle = Circle(center, r, color=c, ec=COLOR_MAIN, linewidth=1)
            ax.add_patch(circle)

        # Draw dart
        ax.plot(dart_pos[0], dart_pos[1], 'o', color='red', markersize=8)

        # Special annotations for each step
        if title == 'Compute Error':
            # Draw error vector
            arrow = FancyArrowPatch(dart_pos, center,
                                  arrowstyle='<->', mutation_scale=15,
                                  color=COLOR_WARNING, linewidth=2, linestyle='--')
            ax.add_patch(arrow)
            ax.text(0.6, 0.35, 'Error', fontsize=9, color=COLOR_WARNING)

        elif title == 'Adjust Aim':
            # Show adjustment
            arrow = FancyArrowPatch(dart_pos, (0.6, 0.4),
                                  arrowstyle='->', mutation_scale=15,
                                  color=COLOR_SUCCESS, linewidth=2)
            ax.add_patch(arrow)
            ax.text(0.65, 0.35, 'Adjust', fontsize=9, color=COLOR_SUCCESS)

        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.suptitle('Learning Process: Like Learning to Throw Darts', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/learning_process_analogy.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_gradient_descent_2d():
    """Create 2D gradient descent visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Create error landscape (bowl shape)
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # Simple quadratic bowl

    # Contour plot
    levels = np.linspace(0, 8, 20)
    cs = ax.contour(X, Y, Z, levels=levels, colors=COLOR_ACCENT, linewidths=1, alpha=0.5)
    ax.contourf(X, Y, Z, levels=levels, cmap='RdYlBu_r', alpha=0.3)

    # Gradient descent path
    path_x = [1.5]
    path_y = [1.2]
    learning_rate = 0.2

    for _ in range(10):
        grad_x = 2 * path_x[-1]  # Gradient of x^2
        grad_y = 2 * path_y[-1]  # Gradient of y^2
        new_x = path_x[-1] - learning_rate * grad_x
        new_y = path_y[-1] - learning_rate * grad_y
        path_x.append(new_x)
        path_y.append(new_y)

    # Plot path
    ax.plot(path_x, path_y, 'o-', color=COLOR_HIGHLIGHT, linewidth=2, markersize=6,
            label='Gradient descent path')

    # Mark start and end
    ax.plot(path_x[0], path_y[0], 'o', color=COLOR_WARNING, markersize=10, label='Start')
    ax.plot(0, 0, '*', color=COLOR_SUCCESS, markersize=15, label='Minimum (goal)')

    # Gradient arrows at a few points
    for i in [1, 3, 5]:
        if i < len(path_x):
            grad_x = -2 * path_x[i] * 0.15  # Negative gradient direction
            grad_y = -2 * path_y[i] * 0.15
            ax.arrow(path_x[i], path_y[i], grad_x, grad_y,
                    head_width=0.1, head_length=0.05, fc=COLOR_MAIN, ec=COLOR_MAIN)

    ax.set_xlabel('Weight 1', fontsize=10)
    ax.set_ylabel('Weight 2', fontsize=10)
    ax.set_title('Gradient Descent: Finding the Minimum Error', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/gradient_descent_2d.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_backprop_flow_simple():
    """Create simple backpropagation flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Network layers
    layers_x = [0.2, 0.4, 0.6, 0.8]
    layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
    layer_sizes = [3, 4, 4, 2]

    # Draw forward pass
    for i, (x, name, size) in enumerate(zip(layers_x, layer_names, layer_sizes)):
        y_positions = np.linspace(0.3, 0.7, size)
        for y in y_positions:
            circle = Circle((x, y), 0.02, color=COLOR_LIGHT, ec=COLOR_MAIN)
            ax.add_patch(circle)

        ax.text(x, 0.2, name, ha='center', fontsize=10)

        # Forward connections
        if i < len(layers_x) - 1:
            arrow = FancyArrowPatch((x + 0.05, 0.5), (layers_x[i+1] - 0.05, 0.5),
                                  arrowstyle='->', mutation_scale=15,
                                  color=COLOR_MAIN, linewidth=2, alpha=0.5)
            ax.add_patch(arrow)
            ax.text((x + layers_x[i+1])/2, 0.52, 'Forward', fontsize=8, ha='center', color=COLOR_MAIN)

    # Draw backward pass (error propagation)
    for i in range(len(layers_x)-1, 0, -1):
        arrow = FancyArrowPatch((layers_x[i] - 0.05, 0.35), (layers_x[i-1] + 0.05, 0.35),
                              arrowstyle='->', mutation_scale=15,
                              color=COLOR_HIGHLIGHT, linewidth=2)
        ax.add_patch(arrow)

    ax.text(0.5, 0.32, 'Backward (gradients)', fontsize=8, ha='center', color=COLOR_HIGHLIGHT)

    # Error at output
    ax.text(0.85, 0.5, 'Error!', fontsize=10, color='red', fontweight='bold')

    # Annotations
    ax.text(0.5, 0.85, 'Backpropagation: Error Flows Backward', fontsize=12, fontweight='bold', ha='center')
    ax.text(0.5, 0.05, 'Each layer learns how much it contributed to the error', fontsize=10, ha='center', color=COLOR_ACCENT)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/backprop_flow_simple.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_feedforward_vs_recurrent():
    """Compare feedforward and recurrent architectures."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Feedforward network
    ax1.set_title('Feedforward: No Memory', fontsize=11, fontweight='bold')

    # Draw three separate mini-networks
    inputs = ['"The"', '"cat"', '"sat"']
    for i, inp in enumerate(inputs):
        x_base = 0.2 + i * 0.3

        # Input
        ax1.text(x_base, 0.2, inp, ha='center', fontsize=9)
        circle = Circle((x_base, 0.3), 0.03, color=COLOR_LIGHT, ec=COLOR_MAIN)
        ax1.add_patch(circle)

        # Hidden
        circle = Circle((x_base, 0.5), 0.03, color=COLOR_ACCENT, ec=COLOR_MAIN)
        ax1.add_patch(circle)

        # Output
        circle = Circle((x_base, 0.7), 0.03, color=COLOR_LIGHT, ec=COLOR_MAIN)
        ax1.add_patch(circle)
        ax1.text(x_base, 0.8, '?', ha='center', fontsize=9)

        # Connections
        arrow1 = FancyArrowPatch((x_base, 0.33), (x_base, 0.47),
                               arrowstyle='->', mutation_scale=10, color=COLOR_MAIN)
        arrow2 = FancyArrowPatch((x_base, 0.53), (x_base, 0.67),
                               arrowstyle='->', mutation_scale=10, color=COLOR_MAIN)
        ax1.add_patch(arrow1)
        ax1.add_patch(arrow2)

        # X mark to show no connection
        if i > 0:
            ax1.plot([x_base - 0.15, x_base - 0.05], [0.5, 0.5], 'x',
                    color='red', markersize=10, markeredgewidth=2)

    ax1.text(0.5, 0.05, 'Each word processed independently!', ha='center',
            fontsize=10, color=COLOR_WARNING)

    # Recurrent network
    ax2.set_title('Recurrent: With Memory', fontsize=11, fontweight='bold')

    # Draw connected network
    for i, inp in enumerate(inputs):
        x_base = 0.2 + i * 0.3

        # Input
        ax2.text(x_base, 0.2, inp, ha='center', fontsize=9)
        circle = Circle((x_base, 0.3), 0.03, color=COLOR_LIGHT, ec=COLOR_MAIN)
        ax2.add_patch(circle)

        # Hidden (with memory)
        circle = Circle((x_base, 0.5), 0.03, color=COLOR_SUCCESS, ec=COLOR_MAIN)
        ax2.add_patch(circle)

        # Output
        circle = Circle((x_base, 0.7), 0.03, color=COLOR_LIGHT, ec=COLOR_MAIN)
        ax2.add_patch(circle)

        # Connections
        arrow1 = FancyArrowPatch((x_base, 0.33), (x_base, 0.47),
                               arrowstyle='->', mutation_scale=10, color=COLOR_MAIN)
        arrow2 = FancyArrowPatch((x_base, 0.53), (x_base, 0.67),
                               arrowstyle='->', mutation_scale=10, color=COLOR_MAIN)
        ax2.add_patch(arrow1)
        ax2.add_patch(arrow2)

        # Memory connection
        if i < len(inputs) - 1:
            arrow = FancyArrowPatch((x_base + 0.03, 0.5), (x_base + 0.27, 0.5),
                                  arrowstyle='->', mutation_scale=12,
                                  color=COLOR_SUCCESS, linewidth=2)
            ax2.add_patch(arrow)
            ax2.text(x_base + 0.15, 0.54, 'memory', fontsize=8, ha='center', color=COLOR_SUCCESS)

    # Output predictions with context
    ax2.text(0.2, 0.8, '?', ha='center', fontsize=9)
    ax2.text(0.5, 0.8, '"sat"?', ha='center', fontsize=9)
    ax2.text(0.8, 0.8, '"on"!', ha='center', fontsize=9, color=COLOR_SUCCESS)

    ax2.text(0.5, 0.05, 'Context builds through sequence!', ha='center',
            fontsize=10, color=COLOR_SUCCESS)

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.9)
        ax.axis('off')

    plt.suptitle('Feedforward vs Recurrent Networks', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/feedforward_vs_recurrent.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_hidden_state_evolution():
    """Show how hidden state evolves through sequence."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Sequence to process
    words = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    x_positions = np.linspace(0.1, 0.9, len(words))

    # Draw the sequence processing
    for i, (word, x) in enumerate(zip(words, x_positions)):
        # Input word
        ax.text(x, 0.2, f'"{word}"', ha='center', fontsize=10)

        # Hidden state box
        rect = FancyBboxPatch((x-0.06, 0.4), 0.12, 0.15,
                             boxstyle="round,pad=0.01",
                             facecolor=COLOR_SUCCESS, alpha=0.3 + i*0.1,
                             edgecolor=COLOR_MAIN, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 0.475, f'$h_{i}$', ha='center', va='center', fontsize=11, fontweight='bold')

        # Memory content annotation
        memory_content = ' '.join(words[:i+1])
        if len(memory_content) > 15:
            memory_content = '...' + memory_content[-12:]
        ax.text(x, 0.35, memory_content, ha='center', fontsize=8,
                style='italic', color=COLOR_ACCENT)

        # Connection to next state
        if i < len(words) - 1:
            arrow = FancyArrowPatch((x + 0.06, 0.475), (x_positions[i+1] - 0.06, 0.475),
                                  arrowstyle='->', mutation_scale=15,
                                  color=COLOR_MAIN, linewidth=2)
            ax.add_patch(arrow)

        # Input arrow
        arrow = FancyArrowPatch((x, 0.25), (x, 0.39),
                              arrowstyle='->', mutation_scale=12,
                              color=COLOR_MAIN, linewidth=1.5)
        ax.add_patch(arrow)

        # Output
        ax.text(x, 0.65, 'output', ha='center', fontsize=8, color=COLOR_ACCENT)
        arrow = FancyArrowPatch((x, 0.55), (x, 0.62),
                              arrowstyle='->', mutation_scale=10,
                              color=COLOR_ACCENT, linewidth=1)
        ax.add_patch(arrow)

    # Title and annotations
    ax.text(0.5, 0.85, 'Hidden State Evolution: Building Context',
            ha='center', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.05, 'Each hidden state summarizes all previous words',
            ha='center', fontsize=10, color=COLOR_ACCENT)

    # Initial empty state
    ax.text(0.02, 0.475, '$h_0$=∅', fontsize=10, color=COLOR_ACCENT)
    arrow = FancyArrowPatch((0.05, 0.475), (0.04, 0.475),
                          arrowstyle='->', mutation_scale=12,
                          color=COLOR_ACCENT, linewidth=1.5)
    ax.add_patch(arrow)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/hidden_state_evolution.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_adding_recurrence_animation():
    """Show transformation from feedforward to recurrent."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    titles = ['1. Feedforward Network', '2. Add Recurrent Connection', '3. Recurrent Network']

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10, fontweight='bold')

        # Draw basic network structure
        # Input
        circle = Circle((0.3, 0.3), 0.05, color=COLOR_LIGHT, ec=COLOR_MAIN)
        ax.add_patch(circle)
        ax.text(0.3, 0.3, 'x', ha='center', va='center', fontsize=10)

        # Hidden
        circle = Circle((0.5, 0.5), 0.05, color=COLOR_ACCENT, ec=COLOR_MAIN)
        ax.add_patch(circle)
        ax.text(0.5, 0.5, 'h', ha='center', va='center', fontsize=10)

        # Output
        circle = Circle((0.7, 0.7), 0.05, color=COLOR_LIGHT, ec=COLOR_MAIN)
        ax.add_patch(circle)
        ax.text(0.7, 0.7, 'y', ha='center', va='center', fontsize=10)

        # Connections
        arrow1 = FancyArrowPatch((0.33, 0.33), (0.47, 0.47),
                               arrowstyle='->', mutation_scale=12, color=COLOR_MAIN)
        arrow2 = FancyArrowPatch((0.53, 0.53), (0.67, 0.67),
                               arrowstyle='->', mutation_scale=12, color=COLOR_MAIN)
        ax.add_patch(arrow1)
        ax.add_patch(arrow2)

        # Add recurrent connection progressively
        if title == '2. Add Recurrent Connection':
            # Draw the new connection being added
            arc = mpatches.FancyBboxPatch((0.45, 0.45), 0.15, 0.15,
                                         boxstyle="round,pad=0.05",
                                         fill=False, edgecolor=COLOR_HIGHLIGHT,
                                         linewidth=2, linestyle='--')
            ax.add_patch(arc)

            # Curved arrow (recurrent)
            arrow = mpatches.FancyArrowPatch((0.55, 0.5), (0.6, 0.35),
                                           connectionstyle="arc3,rad=-.5",
                                           arrowstyle='->', mutation_scale=15,
                                           color=COLOR_HIGHLIGHT, linewidth=2, linestyle='--')
            ax.add_patch(arrow)
            arrow = mpatches.FancyArrowPatch((0.6, 0.35), (0.45, 0.45),
                                           connectionstyle="arc3,rad=-.3",
                                           arrowstyle='->', mutation_scale=15,
                                           color=COLOR_HIGHLIGHT, linewidth=2, linestyle='--')
            ax.add_patch(arrow)
            ax.text(0.65, 0.35, 'Add loop!', fontsize=9, color=COLOR_HIGHLIGHT, fontweight='bold')

        elif title == '3. Recurrent Network':
            # Solid recurrent connection
            arrow = mpatches.FancyArrowPatch((0.55, 0.5), (0.6, 0.35),
                                           connectionstyle="arc3,rad=-.5",
                                           arrowstyle='->', mutation_scale=15,
                                           color=COLOR_SUCCESS, linewidth=2)
            ax.add_patch(arrow)
            arrow = mpatches.FancyArrowPatch((0.6, 0.35), (0.45, 0.45),
                                           connectionstyle="arc3,rad=-.3",
                                           arrowstyle='->', mutation_scale=15,
                                           color=COLOR_SUCCESS, linewidth=2)
            ax.add_patch(arrow)
            ax.text(0.65, 0.3, 'Memory\nloop', fontsize=8, color=COLOR_SUCCESS, ha='center')

            # Change hidden node color to indicate it has memory
            circle = Circle((0.5, 0.5), 0.05, color=COLOR_SUCCESS, ec=COLOR_MAIN, alpha=0.5)
            ax.add_patch(circle)

        ax.set_xlim(0.2, 0.8)
        ax.set_ylim(0.2, 0.8)
        ax.axis('off')

    plt.suptitle('Transforming Feedforward to Recurrent', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/adding_recurrence_animation.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_telephone_game_gradient():
    """Visualize gradient vanishing like telephone game."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Number of steps
    n_steps = 8
    x_positions = np.linspace(0.1, 0.9, n_steps)

    # Signal strength (decaying)
    signal_strength = [0.9 ** i for i in range(n_steps)]

    for i, (x, strength) in enumerate(zip(x_positions, signal_strength)):
        # Person icon (circle)
        circle = Circle((x, 0.5), 0.04,
                       color=COLOR_ACCENT, alpha=strength,
                       ec=COLOR_MAIN, linewidth=1.5)
        ax.add_patch(circle)

        # Signal strength bar
        bar_height = strength * 0.3
        rect = Rectangle((x - 0.02, 0.2), 0.04, bar_height,
                        facecolor=COLOR_HIGHLIGHT, alpha=0.7,
                        edgecolor=COLOR_MAIN, linewidth=1)
        ax.add_patch(rect)

        # Label
        ax.text(x, 0.15, f'Step {i+1}', ha='center', fontsize=8)
        ax.text(x, 0.1, f'{strength:.2f}', ha='center', fontsize=8, color=COLOR_ACCENT)

        # Arrow to next
        if i < n_steps - 1:
            arrow = FancyArrowPatch((x + 0.04, 0.5), (x_positions[i+1] - 0.04, 0.5),
                                  arrowstyle='->', mutation_scale=12,
                                  color=COLOR_MAIN, linewidth=1.5, alpha=strength)
            ax.add_patch(arrow)

            # Multiplication factor
            ax.text((x + x_positions[i+1])/2, 0.55, '×0.9',
                   ha='center', fontsize=8, color=COLOR_WARNING)

    # Annotations
    ax.text(0.5, 0.85, 'Gradient Vanishing: Like a Telephone Game',
            ha='center', fontsize=12, fontweight='bold')
    ax.text(0.1, 0.7, 'Strong\nSignal', ha='center', fontsize=9, color=COLOR_SUCCESS)
    ax.text(0.9, 0.7, 'Weak\nSignal', ha='center', fontsize=9, color=COLOR_WARNING)

    # Formula
    ax.text(0.5, 0.02, 'After n steps: gradient = $(0.9)^n$ → 0',
            ha='center', fontsize=10, style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/telephone_game_gradient.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all foundational figures."""
    print("Generating foundational neural network figures...")

    # Basic concepts
    print("- Creating biological neuron diagram")
    create_biological_neuron_simple()

    print("- Creating mathematical neuron diagram")
    create_mathematical_neuron()

    print("- Creating single neuron computation")
    create_single_neuron_computation()

    print("- Creating activation functions comparison")
    create_activation_functions_comparison()

    print("- Creating linear vs nonlinear visualization")
    create_linear_vs_nonlinear()

    print("- Creating detailed tanh visualization")
    create_tanh_detailed()

    print("- Creating matrix organization diagram")
    create_matrix_organization()

    # Learning process
    print("- Creating learning process analogy")
    create_learning_process_analogy()

    print("- Creating gradient descent visualization")
    create_gradient_descent_2d()

    print("- Creating backpropagation flow")
    create_backprop_flow_simple()

    # Recurrent concepts
    print("- Creating feedforward vs recurrent comparison")
    create_feedforward_vs_recurrent()

    print("- Creating hidden state evolution")
    create_hidden_state_evolution()

    print("- Creating recurrence transformation animation")
    create_adding_recurrence_animation()

    print("- Creating telephone game gradient visualization")
    create_telephone_game_gradient()

    print("\nAll foundational figures generated successfully!")

    # List all generated files
    import os
    figures_dir = '../figures/'
    foundation_figures = [
        'biological_neuron_simple.pdf',
        'mathematical_neuron.pdf',
        'single_neuron_computation.pdf',
        'activation_functions_comparison.pdf',
        'linear_vs_nonlinear.pdf',
        'tanh_detailed.pdf',
        'matrix_organization.pdf',
        'learning_process_analogy.pdf',
        'gradient_descent_2d.pdf',
        'backprop_flow_simple.pdf',
        'feedforward_vs_recurrent.pdf',
        'hidden_state_evolution.pdf',
        'adding_recurrence_animation.pdf',
        'telephone_game_gradient.pdf'
    ]

    print("\nGenerated files:")
    for fig in foundation_figures:
        path = os.path.join(figures_dir, fig)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # Size in KB
            print(f"  - {fig} ({size:.1f} KB)")
        else:
            print(f"  - {fig} (NOT FOUND)")

if __name__ == "__main__":
    main()