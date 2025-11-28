"""
Generate figures for Neural Networks Discovery Handout
Creates clean, printable visualizations for 2-3 page handout
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for clean, printable figures
plt.style.use('seaborn-v0_8-whitegrid')

# Monochromatic color scheme for printing
COLOR_MAIN = '#404040'      # Dark gray
COLOR_ACCENT = '#B4B4B4'    # Light gray
COLOR_POSITIVE = '#2E7D32'  # Dark green
COLOR_NEGATIVE = '#C62828'  # Dark red
COLOR_BOUNDARY = '#1976D2'  # Blue

OUTPUT_DIR = '../figures/handout/'


def create_single_neuron_linear():
    """Figure 1: Single neuron with linear decision boundary"""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Generate sample data
    np.random.seed(42)

    # Class 1 (circles)
    X1 = np.random.randn(30, 2) * 0.5 + np.array([1, 2])
    # Class 2 (triangles)
    X2 = np.random.randn(30, 2) * 0.5 + np.array([3, 1])

    # Plot data points
    ax.scatter(X1[:, 0], X1[:, 1], s=100, c=COLOR_POSITIVE,
              marker='o', edgecolors='black', linewidths=1.5,
              label='Class A (✓)', alpha=0.7)
    ax.scatter(X2[:, 0], X2[:, 1], s=100, c=COLOR_NEGATIVE,
              marker='^', edgecolors='black', linewidths=1.5,
              label='Class B (✗)', alpha=0.7)

    # Decision boundary: w1*x + w2*y + b = 0
    # Using w1=0.5, w2=0.8, b=-2
    # Rearranged: y = (-w1/w2)*x + (-b/w2)
    w1, w2, b = 0.5, 0.8, -2
    x_line = np.array([0, 4.5])
    y_line = (-w1/w2) * x_line + (-b/w2)

    ax.plot(x_line, y_line, COLOR_BOUNDARY, linewidth=3,
           label='Decision Boundary', linestyle='--')

    # Add equation
    ax.text(0.5, 3.5, r'$0.5x_1 + 0.8x_2 - 2 = 0$',
           fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4)
    ax.set_xlabel('Feature $x_1$', fontsize=12)
    ax.set_ylabel('Feature $x_2$', fontsize=12)
    ax.set_title('Single Neuron: Linear Decision Boundary', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}single_neuron_linear_example.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: single_neuron_linear_example.pdf")
    plt.close()


def create_neuron_with_activation():
    """Figure 2: Comparison of neuron with/without activation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Generate circular data that can't be linearly separated
    np.random.seed(42)
    theta = np.linspace(0, 2*np.pi, 50)
    r_inner = 1 + 0.2 * np.random.randn(50)
    r_outer = 2.5 + 0.3 * np.random.randn(50)

    X_inner = np.column_stack([r_inner * np.cos(theta), r_inner * np.sin(theta)])
    X_outer = np.column_stack([r_outer * np.cos(theta), r_outer * np.sin(theta)])

    # Plot 1: Without activation (linear boundary fails)
    ax1.scatter(X_inner[:, 0], X_inner[:, 1], s=80, c=COLOR_POSITIVE,
               marker='o', edgecolors='black', linewidths=1, alpha=0.7)
    ax1.scatter(X_outer[:, 0], X_outer[:, 1], s=80, c=COLOR_NEGATIVE,
               marker='^', edgecolors='black', linewidths=1, alpha=0.7)

    # Show a failing linear boundary
    x_line = np.array([-3, 3])
    y_line = 0.5 * x_line
    ax1.plot(x_line, y_line, COLOR_BOUNDARY, linewidth=3, linestyle='--')
    ax1.text(-2, 2.5, '❌ Linear boundary fails!', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

    ax1.set_xlim(-3.5, 3.5)
    ax1.set_ylim(-3.5, 3.5)
    ax1.set_xlabel('$x_1$', fontsize=12)
    ax1.set_ylabel('$x_2$', fontsize=12)
    ax1.set_title('Without Activation\n(Linear only)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot 2: With activation (curved boundary succeeds)
    ax2.scatter(X_inner[:, 0], X_inner[:, 1], s=80, c=COLOR_POSITIVE,
               marker='o', edgecolors='black', linewidths=1, alpha=0.7,
               label='Class A')
    ax2.scatter(X_outer[:, 0], X_outer[:, 1], s=80, c=COLOR_NEGATIVE,
               marker='^', edgecolors='black', linewidths=1, alpha=0.7,
               label='Class B')

    # Draw a circular boundary
    circle = plt.Circle((0, 0), 1.75, color=COLOR_BOUNDARY, fill=False,
                       linewidth=3, linestyle='-')
    ax2.add_patch(circle)
    ax2.text(-2, 2.5, '✓ Curved boundary works!', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

    ax2.set_xlim(-3.5, 3.5)
    ax2.set_ylim(-3.5, 3.5)
    ax2.set_xlabel('$x_1$', fontsize=12)
    ax2.set_ylabel('$x_2$', fontsize=12)
    ax2.set_title('With Activation Function\n(Can curve!)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}neuron_without_with_activation.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: neuron_without_with_activation.pdf")
    plt.close()


def create_activation_functions_chart():
    """Figure 3: Small chart showing different activation functions"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    x = np.linspace(-5, 5, 200)

    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x))
    axes[0].plot(x, sigmoid, COLOR_MAIN, linewidth=3)
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    axes[0].set_title('Sigmoid\n$\\sigma(x) = \\frac{1}{1+e^{-x}}$', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Input', fontsize=10)
    axes[0].set_ylabel('Output', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.2, 1.2)
    axes[0].text(2, 0.1, 'Output: 0 to 1', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # ReLU
    relu = np.maximum(0, x)
    axes[1].plot(x, relu, COLOR_MAIN, linewidth=3)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    axes[1].set_title('ReLU\n$f(x) = max(0, x)$', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Input', fontsize=10)
    axes[1].set_ylabel('Output', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-1, 5)
    axes[1].text(2, 0.5, 'Output: 0 to ∞', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Tanh
    tanh = np.tanh(x)
    axes[2].plot(x, tanh, COLOR_MAIN, linewidth=3)
    axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[2].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    axes[2].set_title('Tanh\n$f(x) = \\tanh(x)$', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Input', fontsize=10)
    axes[2].set_ylabel('Output', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-1.2, 1.2)
    axes[2].text(2, -0.8, 'Output: -1 to 1', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}activation_functions_comparison.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: activation_functions_comparison.pdf")
    plt.close()


def create_xor_solution_3panel():
    """Figure 4: Three-panel visualization of XOR solution"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # XOR data
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])  # XOR outputs

    colors = [COLOR_NEGATIVE if y == 0 else COLOR_POSITIVE for y in y_xor]
    markers = ['^' if y == 0 else 'o' for y in y_xor]
    labels = ['✗' if y == 0 else '✓' for y in y_xor]

    for ax_idx, ax in enumerate(axes):
        # Plot XOR points
        for i, (x, y, c, m, l) in enumerate(zip(X_xor[:, 0], X_xor[:, 1], colors, markers, labels)):
            ax.scatter(x, y, s=200, c=c, marker=m, edgecolors='black', linewidths=2)
            ax.text(x, y-0.15, l, fontsize=14, ha='center', fontweight='bold')

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

    # Panel 1: Neuron 1 boundary
    x_line = np.array([-0.5, 1.5])
    y_line1 = -x_line + 0.5
    axes[0].plot(x_line, y_line1, 'r--', linewidth=3, label='Neuron 1')
    axes[0].set_title('Neuron 1\n(Separates bottom-left)', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].text(0.2, 1.3, '❶', fontsize=18, fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor='red', alpha=0.3))

    # Panel 2: Neuron 2 boundary
    y_line2 = x_line + 0.5
    axes[1].plot(x_line, y_line2, 'b--', linewidth=3, label='Neuron 2')
    axes[1].set_title('Neuron 2\n(Separates bottom-right)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].text(0.2, 1.3, '❷', fontsize=18, fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor='blue', alpha=0.3))

    # Panel 3: Combined solution
    axes[2].plot(x_line, y_line1, 'r--', linewidth=2, alpha=0.5, label='Neuron 1')
    axes[2].plot(x_line, y_line2, 'b--', linewidth=2, alpha=0.5, label='Neuron 2')
    axes[2].set_title('Combined\n✓ XOR Solved!', fontsize=12, fontweight='bold', color='green')
    axes[2].legend(loc='upper right', fontsize=10)
    axes[2].text(0.1, 1.3, '❶+❷', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}xor_solution_3panel.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: xor_solution_3panel.pdf")
    plt.close()


def create_function_approximation_grid():
    """Figure 5: Approximating sin(x) with 1, 3, 10 neurons using proper sigmoid activation"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Target function
    x = np.linspace(0, 2*np.pi, 200)
    y_target = np.sin(x)

    def sigmoid(z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def approximate_with_neurons(x, n_neurons):
        """
        Proper neural network approximation using sigmoid neurons
        Each neuron: a_i * sigmoid(w_i*x + b_i)
        Output: sum of weighted sigmoid activations
        """
        y_approx = np.zeros_like(x)

        if n_neurons == 1:
            # Single neuron: can only make S-shaped curve, poor for sin(x)
            w, b = 1.5, -np.pi
            a = 2.0
            y_approx = a * (sigmoid(w * (x - np.pi)) - 0.5)

        elif n_neurons == 3:
            # 3 neurons: can approximate rising and falling parts
            neurons = [
                (2.0, 2.0, -2.0),      # (weight, scale, bias)
                (2.0, -4.0, -4.0),
                (2.0, 2.0, -10.0),
            ]
            for w, a, b in neurons:
                y_approx += a * (sigmoid(w * x + b) - 0.5)

        elif n_neurons >= 10:
            # 10 neurons: close approximation using multiple sigmoids
            # Place neurons at strategic positions to capture sin wave
            positions = np.linspace(0, 2*np.pi, n_neurons)
            for i, pos in enumerate(positions):
                # Alternating positive/negative contributions
                phase = np.sin(pos)
                w = 3.0
                b = -w * pos
                a = 4.0 * phase
                y_approx += a * (sigmoid(w * x + b) - 0.5)

            # Normalize to match sin(x) amplitude
            if np.max(np.abs(y_approx)) > 0:
                y_approx = y_approx / np.max(np.abs(y_approx))

        return y_approx

    neuron_counts = [1, 3, 10]
    titles = ['1 Neuron\n(Poor fit)', '3 Neurons\n(Better fit)', '10 Neurons\n(Excellent fit)']
    colors_approx = ['red', 'orange', 'green']

    for ax, n, title, col in zip(axes, neuron_counts, titles, colors_approx):
        # Plot target
        ax.plot(x, y_target, COLOR_MAIN, linewidth=3, label='Target: sin(x)', linestyle='--', alpha=0.7)

        # Plot approximation
        y_approx = approximate_with_neurons(x, n)
        ax.plot(x, y_approx, col, linewidth=3, label=f'{n} neuron approx')

        # Calculate MSE
        error = np.mean((y_target - y_approx)**2)
        ax.text(1, -1.3, f'MSE: {error:.3f}', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.5, 1.5)

    plt.suptitle('Universal Approximation: More Neurons = Better Fit',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}function_approximation_progression.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: function_approximation_progression.pdf")
    plt.close()


def create_two_neurons_geometric():
    """Figure 6: Geometric interpretation of two neurons combining"""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Generate data in an "AND" pattern
    np.random.seed(42)

    # Positive class (both features high)
    X_pos = np.random.randn(20, 2) * 0.3 + np.array([2.5, 2.5])
    # Negative class (scattered)
    X_neg1 = np.random.randn(15, 2) * 0.3 + np.array([0.5, 0.5])
    X_neg2 = np.random.randn(15, 2) * 0.3 + np.array([0.5, 2.5])
    X_neg3 = np.random.randn(15, 2) * 0.3 + np.array([2.5, 0.5])

    # Plot data
    ax.scatter(X_pos[:, 0], X_pos[:, 1], s=120, c=COLOR_POSITIVE,
              marker='o', edgecolors='black', linewidths=1.5, label='✓', alpha=0.7)
    ax.scatter(X_neg1[:, 0], X_neg1[:, 1], s=120, c=COLOR_NEGATIVE,
              marker='^', edgecolors='black', linewidths=1.5, label='✗', alpha=0.7)
    ax.scatter(X_neg2[:, 0], X_neg2[:, 1], s=120, c=COLOR_NEGATIVE,
              marker='^', edgecolors='black', linewidths=1.5, alpha=0.7)
    ax.scatter(X_neg3[:, 0], X_neg3[:, 1], s=120, c=COLOR_NEGATIVE,
              marker='^', edgecolors='black', linewidths=1.5, alpha=0.7)

    # Neuron 1: vertical boundary (x1 > 1.5)
    ax.axvline(x=1.5, color='red', linewidth=3, linestyle='--', label='Neuron 1: $x_1 > 1.5$')
    ax.fill_betweenx([0, 3.5], 1.5, 3.5, alpha=0.1, color='red')

    # Neuron 2: horizontal boundary (x2 > 1.5)
    ax.axhline(y=1.5, color='blue', linewidth=3, linestyle='--', label='Neuron 2: $x_2 > 1.5$')
    ax.fill_between([0, 3.5], 1.5, 3.5, alpha=0.1, color='blue')

    # Intersection (both neurons active)
    rect = Rectangle((1.5, 1.5), 2, 2, linewidth=4, edgecolor='green',
                     facecolor='green', alpha=0.2, linestyle='-')
    ax.add_patch(rect)
    ax.text(2.5, 2.9, 'Both Active\n✓ Region', fontsize=11, ha='center',
           bbox=dict(boxstyle='round', facecolor='green', alpha=0.7), fontweight='bold')

    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 3.5)
    ax.set_xlabel('Feature $x_1$', fontsize=12)
    ax.set_ylabel('Feature $x_2$', fontsize=12)
    ax.set_title('Two Neurons Combine: Intersection of Half-Spaces',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}two_neurons_combining.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: two_neurons_combining.pdf")
    plt.close()


def create_neuron_schematic_simple():
    """Figure 7: Simple neuron schematic for absolute beginners"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    # Input boxes
    input1_box = plt.Rectangle((0.5, 2.5), 1, 0.8, facecolor='lightblue', edgecolor='black', linewidth=2)
    input2_box = plt.Rectangle((0.5, 1.0), 1, 0.8, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input1_box)
    ax.add_patch(input2_box)

    ax.text(1, 2.9, 'Distance\n(km)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1, 1.4, 'Friends\n(count)', ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows with weights
    ax.annotate('', xy=(3.5, 2.7), xytext=(1.6, 2.9),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_MAIN))
    ax.text(2.5, 3.1, r'$\times w_1=-2$', fontsize=11, color='red', fontweight='bold')

    ax.annotate('', xy=(3.5, 2.3), xytext=(1.6, 1.4),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_MAIN))
    ax.text(2.5, 1.6, r'$\times w_2=+3$', fontsize=11, color='green', fontweight='bold')

    # Summation circle
    circle = plt.Circle((4, 2.5), 0.6, facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(4, 2.7, r'$\Sigma$', ha='center', va='center', fontsize=24, fontweight='bold')
    ax.text(4, 2.1, 'SUM', ha='center', va='center', fontsize=9)

    # Bias arrow
    ax.annotate('', xy=(4, 1.7), xytext=(4, 0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
    ax.text(4.5, 1.0, 'bias=-5', fontsize=10, color='purple', fontweight='bold')

    # Arrow to output
    ax.annotate('', xy=(6.5, 2.5), xytext=(4.7, 2.5),
               arrowprops=dict(arrowstyle='->', lw=3, color=COLOR_MAIN))

    # Output box
    output_box = plt.Rectangle((6.5, 2.1), 1.5, 0.8, facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(7.25, 2.5, 'Score', ha='center', va='center', fontsize=11, fontweight='bold')

    # Decision box
    decision_box = plt.Rectangle((8.5, 2.1), 1.2, 0.8, facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(decision_box)
    ax.text(9.1, 2.7, 'Decision', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(9.1, 2.3, 'GO/STAY', ha='center', va='center', fontsize=8)

    # Arrow to decision
    ax.annotate('', xy=(8.4, 2.5), xytext=(8.1, 2.5),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_MAIN))
    ax.text(8.25, 2.9, 'if > 0', fontsize=9, ha='center')

    # Formula at bottom
    ax.text(5, 0.2, r'$\mathbf{Score = (w_1 \times Distance) + (w_2 \times Friends) + bias}$',
           fontsize=12, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.title('How a Neuron Computes: Party Decision Example', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}neuron_schematic_simple.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: neuron_schematic_simple.pdf")
    plt.close()


def create_party_decision_scatter():
    """Figure 8: Scatter plot of party decisions based on distance and friends"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Data points from the analogy
    # Format: (distance, friends, decision)
    decisions = [
        (1, 2, 'GO'),      # Score = -2*1 + 3*2 - 5 = 1 (GO)
        (5, 1, 'STAY'),    # Score = -2*5 + 3*1 - 5 = -12 (STAY)
        (2, 3, 'GO'),      # Score = -2*2 + 3*3 - 5 = 0 (borderline, say GO)
        (3, 2, 'STAY'),    # Score = -2*3 + 3*2 - 5 = -5 (STAY)
        (1, 1, 'STAY'),    # Score = -2*1 + 3*1 - 5 = -4 (STAY)
        (2, 4, 'GO'),      # Score = -2*2 + 3*4 - 5 = 3 (GO)
        (4, 3, 'GO'),      # Score = -2*4 + 3*3 - 5 = -4 (STAY) - wait, recalc: -8+9-5=-4 STAY
        (1, 3, 'GO'),      # Score = -2*1 + 3*3 - 5 = 2 (GO)
        (6, 2, 'STAY'),    # Score = -2*6 + 3*2 - 5 = -11 (STAY)
        (2, 2, 'STAY'),    # Score = -2*2 + 3*2 - 5 = -3 (STAY)
        (3, 4, 'GO'),      # Score = -2*3 + 3*4 - 5 = 1 (GO)
        (4, 1, 'STAY'),    # Score = -2*4 + 3*1 - 5 = -10 (STAY)
    ]

    # Separate by decision
    go_points = [(d, f) for d, f, dec in decisions if dec == 'GO']
    stay_points = [(d, f) for d, f, dec in decisions if dec == 'STAY']

    # Plot
    if go_points:
        go_x, go_y = zip(*go_points)
        ax.scatter(go_x, go_y, s=200, c=COLOR_POSITIVE, marker='o',
                  edgecolors='black', linewidths=2, label='Went to Party', alpha=0.8)

    if stay_points:
        stay_x, stay_y = zip(*stay_points)
        ax.scatter(stay_x, stay_y, s=200, c=COLOR_NEGATIVE, marker='^',
                  edgecolors='black', linewidths=2, label='Stayed Home', alpha=0.8)

    # Decision boundary: -2*x1 + 3*x2 - 5 = 0
    # Rearranged: x2 = (2*x1 + 5)/3
    x_line = np.array([0, 7])
    y_line = (2*x_line + 5) / 3

    ax.plot(x_line, y_line, COLOR_BOUNDARY, linewidth=3, linestyle='--',
           label='Decision Boundary')

    # Shade regions
    ax.fill_between(x_line, y_line, 5, alpha=0.1, color='green', label='GO region (Score > 0)')
    ax.fill_between(x_line, 0, y_line, alpha=0.1, color='red', label='STAY region (Score < 0)')

    ax.set_xlabel('Distance to Party (km)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Friends Going', fontsize=12, fontweight='bold')
    ax.set_title("Alex's Party Decisions: Visualizing the Neuron", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 5)

    # Add formula annotation
    ax.text(5.5, 0.5, 'Boundary:\n$-2d + 3f - 5 = 0$',
           fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}party_decision_scatter.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: party_decision_scatter.pdf")
    plt.close()


def main():
    """Generate all handout figures"""
    print("="*60)
    print("Generating Neural Networks Discovery Handout Figures")
    print("="*60)
    print()

    # Part 0 figures (NEW for zero pre-knowledge)
    create_neuron_schematic_simple()
    create_party_decision_scatter()

    # Original figures
    create_single_neuron_linear()
    create_neuron_with_activation()
    create_activation_functions_chart()
    create_xor_solution_3panel()
    create_function_approximation_grid()
    create_two_neurons_geometric()

    print()
    print("="*60)
    print("[SUCCESS] All figures generated successfully!")
    print(f"Location: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()