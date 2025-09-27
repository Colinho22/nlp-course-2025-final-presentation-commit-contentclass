import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

# Set the minimalist style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Minimalist color palette
COLOR_MAIN = '#404040'      # RGB(64,64,64)
COLOR_ACCENT = '#B4B4B4'    # RGB(180,180,180)
COLOR_LIGHT = '#F0F0F0'     # RGB(240,240,240)
COLOR_HIGHLIGHT = '#FF6B6B'  # Red for emphasis
COLOR_BLUE = '#4682B4'       # Steel blue for checkpoints
COLOR_GREEN = '#90EE90'      # Light green for success
COLOR_ORANGE = '#FFA500'     # Orange for warnings

def generate_spam_detector_perceptron():
    """Visualize a spam detector perceptron with features"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 5)
    ax.axis('off')

    # Input features
    features = [
        ("Has 'FREE'", 1),
        ("Has 'money'", 1),
        ("Many '!'", 1),
        ("From friend", 0)
    ]

    # Draw input nodes
    for i, (label, value) in enumerate(features):
        y = 4 - i * 1
        # Input node
        circle = Circle((1, y), 0.3,
                       facecolor=COLOR_HIGHLIGHT if value == 1 else COLOR_LIGHT,
                       edgecolor=COLOR_MAIN, linewidth=2)
        ax.add_patch(circle)
        ax.text(1, y, str(value), ha='center', va='center',
               fontsize=12, fontweight='bold', color='white' if value == 1 else COLOR_MAIN)
        # Label
        ax.text(0, y, label, ha='right', va='center', fontsize=10, color=COLOR_MAIN)

    # Draw perceptron (output node)
    output_circle = Circle((6, 2), 0.5, facecolor=COLOR_BLUE,
                          edgecolor=COLOR_MAIN, linewidth=3, alpha=0.7)
    ax.add_patch(output_circle)
    ax.text(6, 2, 'Σ', ha='center', va='center', fontsize=20,
           fontweight='bold', color='white')

    # Draw weights as arrows
    weights = [3, 2, 2, -5]
    for i, w in enumerate(weights):
        y = 4 - i * 1
        # Arrow from input to output
        arrow = FancyArrowPatch((1.3, y), (5.5, 2),
                              connectionstyle="arc3,rad=0.1",
                              arrowstyle='->', mutation_scale=20,
                              linewidth=abs(w)/2,
                              color=COLOR_GREEN if w > 0 else COLOR_ORANGE,
                              alpha=0.7)
        ax.add_patch(arrow)
        # Weight label
        mid_x, mid_y = 3.5, (y + 2) / 2
        ax.text(mid_x, mid_y, f'w={w:+d}', fontsize=9,
               color=COLOR_GREEN if w > 0 else COLOR_ORANGE,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Bias
    ax.text(6, 1.2, 'bias = -2', ha='center', fontsize=9,
           color=COLOR_ACCENT, style='italic')

    # Calculation
    ax.text(4, 0.2, 'Sum = 3×1 + 2×1 + 2×1 + (-5)×0 + (-2) = 5',
           ha='center', fontsize=11, color=COLOR_MAIN,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT, alpha=0.8))

    # Output
    ax.text(7.2, 2, '→ SPAM!', ha='left', va='center', fontsize=14,
           fontweight='bold', color=COLOR_HIGHLIGHT)

    # Title
    ax.text(3.5, 4.7, 'Spam Detection Perceptron', ha='center', fontsize=14,
           fontweight='bold', color=COLOR_MAIN)

    plt.tight_layout()
    plt.savefig('../figures/spam_detector_perceptron.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_forward_pass_visualization():
    """Show forward pass calculation through a simple network"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 6)
    ax.axis('off')

    # Title
    ax.text(5, 5.5, 'Forward Pass: Step-by-Step Calculation',
           ha='center', fontsize=14, fontweight='bold', color=COLOR_MAIN)

    # Network structure
    # Input layer
    inputs = [1, 0]
    for i, val in enumerate(inputs):
        y = 3.5 - i * 1.5
        circle = Circle((1, y), 0.3, facecolor='white',
                       edgecolor=COLOR_MAIN, linewidth=2)
        ax.add_patch(circle)
        ax.text(1, y, f'x{i+1}={val}', ha='center', va='center',
               fontsize=10, color=COLOR_MAIN)

    # Hidden layer
    hidden_values = [0, 1.5]  # After ReLU
    for i, val in enumerate(hidden_values):
        y = 3.5 - i * 1.5
        circle = Circle((5, y), 0.3,
                       facecolor=COLOR_LIGHT if val == 0 else COLOR_BLUE,
                       edgecolor=COLOR_MAIN, linewidth=2, alpha=0.8)
        ax.add_patch(circle)
        ax.text(5, y, f'h{i+1}', ha='center', va='center',
               fontsize=10, color=COLOR_MAIN if val == 0 else 'white')
        # Show calculation
        if i == 0:
            calc_text = f"ReLU(1×0.5 + 0×2.0 - 0.5) = ReLU(0) = 0"
        else:
            calc_text = f"ReLU(1×1.0 + 0×(-1) + 0.5) = ReLU(1.5) = 1.5"
        ax.text(5, y - 0.6, calc_text, ha='center', fontsize=8,
               color=COLOR_ACCENT, style='italic')

    # Output layer
    output_val = 1.5
    circle = Circle((8.5, 2.5), 0.4, facecolor=COLOR_GREEN,
                   edgecolor=COLOR_MAIN, linewidth=3, alpha=0.7)
    ax.add_patch(circle)
    ax.text(8.5, 2.5, f'y={output_val}', ha='center', va='center',
           fontsize=11, fontweight='bold', color=COLOR_MAIN)
    ax.text(8.5, 1.8, '0×1.5 + 1.5×1.0 = 1.5', ha='center',
           fontsize=8, color=COLOR_ACCENT, style='italic')

    # Draw connections with weights
    # Input to hidden
    weights_ih = [[0.5, 2.0], [1.0, -1.0]]
    for i in range(2):
        for j in range(2):
            start_y = 3.5 - i * 1.5
            end_y = 3.5 - j * 1.5
            arrow = FancyArrowPatch((1.3, start_y), (4.7, end_y),
                                  connectionstyle="arc3,rad=0.1",
                                  arrowstyle='->', mutation_scale=15,
                                  linewidth=1, color=COLOR_ACCENT, alpha=0.5)
            ax.add_patch(arrow)
            # Weight label
            mid_x = 3
            mid_y = (start_y + end_y) / 2
            ax.text(mid_x, mid_y, f'{weights_ih[j][i]:.1f}',
                   fontsize=7, color=COLOR_ACCENT)

    # Hidden to output
    weights_ho = [1.5, 1.0]
    for i, w in enumerate(weights_ho):
        start_y = 3.5 - i * 1.5
        arrow = FancyArrowPatch((5.3, start_y), (8.1, 2.5),
                              connectionstyle="arc3,rad=0.1",
                              arrowstyle='->', mutation_scale=15,
                              linewidth=2 if i == 1 else 0.5,
                              color=COLOR_BLUE if i == 1 else COLOR_LIGHT,
                              alpha=0.8)
        ax.add_patch(arrow)
        mid_x = 6.7
        mid_y = (start_y + 2.5) / 2
        ax.text(mid_x, mid_y, f'{w:.1f}', fontsize=8,
               color=COLOR_BLUE if i == 1 else COLOR_ACCENT)

    # Layer labels
    ax.text(1, 4.5, 'Input', ha='center', fontsize=10, fontweight='bold', color=COLOR_MAIN)
    ax.text(5, 4.5, 'Hidden', ha='center', fontsize=10, fontweight='bold', color=COLOR_MAIN)
    ax.text(8.5, 4.5, 'Output', ha='center', fontsize=10, fontweight='bold', color=COLOR_MAIN)

    # Flow arrows
    ax.annotate('', xy=(2.5, 0.5), xytext=(1.5, 0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_HIGHLIGHT))
    ax.annotate('', xy=(6.5, 0.5), xytext=(5.5, 0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_HIGHLIGHT))
    ax.text(2, 0.2, 'Forward', ha='center', fontsize=9, color=COLOR_HIGHLIGHT)
    ax.text(6, 0.2, 'Forward', ha='center', fontsize=9, color=COLOR_HIGHLIGHT)

    plt.tight_layout()
    plt.savefig('../figures/forward_pass_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_multiple_detectors_intuition():
    """Show how multiple detectors work together"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Single detector (left)
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title('Single Detector\n(Linear)', fontsize=10, color=COLOR_MAIN)
    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)

    # Simple linear boundary
    x = np.linspace(0, 10, 100)
    y = 10 - x
    ax.plot(x, y, '--', color=COLOR_MAIN, linewidth=2)
    ax.fill_between(x, y, 10, alpha=0.1, color=COLOR_BLUE)
    ax.fill_between(x, 0, y, alpha=0.1, color=COLOR_ORANGE)

    # Add points
    np.random.seed(42)
    # Class 1 (above line)
    x1 = np.random.uniform(2, 8, 10)
    y1 = np.random.uniform(6, 9, 10)
    ax.scatter(x1, y1, c=COLOR_BLUE, s=30, alpha=0.7)
    # Class 2 (below line)
    x2 = np.random.uniform(2, 8, 10)
    y2 = np.random.uniform(1, 4, 10)
    ax.scatter(x2, y2, c=COLOR_ORANGE, s=30, alpha=0.7)

    ax.text(5, 8, '✓ Works!', ha='center', fontsize=9, color=COLOR_GREEN)

    # Two detectors (middle)
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title('Two Detectors\n(Can do XOR!)', fontsize=10, color=COLOR_MAIN)
    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)

    # Two lines
    x = np.linspace(0, 10, 100)
    y1 = 8 - 0.6*x
    y2 = 2 + 0.6*x
    ax.plot(x, y1, '--', color=COLOR_MAIN, linewidth=2, label='Detector 1')
    ax.plot(x, y2, '--', color=COLOR_MAIN, linewidth=2, label='Detector 2')

    # Fill XOR regions
    for i in range(len(x)):
        if y1[i] > y2[i]:
            ax.fill_between([x[i], x[i]+0.1], [y2[i], y2[i]], [y1[i], y1[i]],
                          alpha=0.2, color=COLOR_BLUE)

    # XOR points
    ax.scatter([2, 8], [8, 2], c=COLOR_BLUE, s=50, alpha=0.8, edgecolors=COLOR_MAIN)
    ax.scatter([2, 8], [2, 8], c=COLOR_ORANGE, s=50, alpha=0.8, edgecolors=COLOR_MAIN)

    ax.text(5, 9, '✓ XOR Solved!', ha='center', fontsize=9, color=COLOR_GREEN)

    # Many detectors (right)
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title('Many Detectors\n(Complex Patterns)', fontsize=10, color=COLOR_MAIN)
    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)

    # Create complex decision boundary
    theta = np.linspace(0, 2*np.pi, 100)
    r = 2 + 0.5*np.sin(5*theta)
    x_complex = 5 + r*np.cos(theta)
    y_complex = 5 + r*np.sin(theta)
    ax.plot(x_complex, y_complex, '-', color=COLOR_MAIN, linewidth=2)
    ax.fill(x_complex, y_complex, alpha=0.2, color=COLOR_BLUE)

    # Add sample points
    # Inside
    theta_in = np.random.uniform(0, 2*np.pi, 15)
    r_in = np.random.uniform(0, 1.5, 15)
    x_in = 5 + r_in*np.cos(theta_in)
    y_in = 5 + r_in*np.sin(theta_in)
    ax.scatter(x_in, y_in, c=COLOR_BLUE, s=30, alpha=0.7)

    # Outside
    theta_out = np.random.uniform(0, 2*np.pi, 15)
    r_out = np.random.uniform(3, 4, 15)
    x_out = 5 + r_out*np.cos(theta_out)
    y_out = 5 + r_out*np.sin(theta_out)
    ax.scatter(x_out, y_out, c=COLOR_ORANGE, s=30, alpha=0.7)

    ax.text(5, 9, '✓ Any Shape!', ha='center', fontsize=9, color=COLOR_GREEN)

    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')

    plt.suptitle('How Multiple Detectors Create Complex Boundaries',
                fontsize=12, fontweight='bold', color=COLOR_MAIN)
    plt.tight_layout()
    plt.savefig('../figures/multiple_detectors_intuition.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_network_debugging_flowchart():
    """Create a debugging flowchart for neural networks"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Neural Network Debugging Flowchart',
           ha='center', fontsize=14, fontweight='bold', color=COLOR_MAIN)

    # Define boxes with positions and text
    boxes = [
        (5, 8.5, "Network not learning?", COLOR_HIGHLIGHT),
        (3, 7, "Loss = NaN?", COLOR_ORANGE),
        (7, 7, "Loss decreasing?", COLOR_BLUE),
        (1.5, 5.5, "Check learning\nrate (too high)", COLOR_ORANGE),
        (3, 5.5, "Check for\ninf/nan in data", COLOR_ORANGE),
        (7, 5.5, "Overfitting?\n(train << val)", COLOR_BLUE),
        (9, 5.5, "Loss plateaued?", COLOR_BLUE),
        (7, 4, "Add dropout\nregularization", COLOR_GREEN),
        (9, 4, "Reduce\nlearning rate", COLOR_GREEN),
        (5, 3, "Can overfit\nsingle batch?", COLOR_MAIN),
        (3, 1.5, "Architecture\nproblem", COLOR_HIGHLIGHT),
        (7, 1.5, "Data/preprocessing\nproblem", COLOR_HIGHLIGHT)
    ]

    # Draw boxes
    for x, y, text, color in boxes:
        if color == COLOR_HIGHLIGHT:
            box = FancyBboxPatch((x-0.8, y-0.3), 1.6, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor=color, edgecolor=COLOR_MAIN,
                                linewidth=2, alpha=0.3)
        else:
            box = FancyBboxPatch((x-0.7, y-0.25), 1.4, 0.5,
                                boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor=color,
                                linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8,
               color=COLOR_MAIN, fontweight='bold' if color == COLOR_HIGHLIGHT else 'normal')

    # Draw arrows with labels
    arrows = [
        ((5, 8.2), (3, 7.3), "Check"),
        ((5, 8.2), (7, 7.3), "Check"),
        ((3, 6.7), (1.5, 5.8), "Yes"),
        ((3, 6.7), (3, 5.8), "No"),
        ((7, 6.7), (7, 5.8), "No"),
        ((7, 6.7), (9, 5.8), "Yes"),
        ((7, 5.2), (7, 4.3), "Yes"),
        ((9, 5.2), (9, 4.3), "Yes"),
        ((7, 5.2), (5, 3.3), "No"),
        ((9, 5.2), (5, 3.3), "No"),
        ((5, 2.7), (3, 1.8), "No"),
        ((5, 2.7), (7, 1.8), "Yes")
    ]

    for start, end, label in arrows:
        arrow = FancyArrowPatch(start, end,
                              connectionstyle="arc3,rad=0.1",
                              arrowstyle='->', mutation_scale=15,
                              linewidth=1, color=COLOR_ACCENT)
        ax.add_patch(arrow)
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y, label, fontsize=7, color=COLOR_ACCENT,
               style='italic', ha='center')

    # Add tips at bottom
    ax.text(5, 0.5, "💡 Always start by trying to overfit a single batch - if that fails, it's a code bug!",
           ha='center', fontsize=9, color=COLOR_MAIN,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT, alpha=0.8))

    plt.tight_layout()
    plt.savefig('../figures/network_debugging_flowchart.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_your_first_network_architecture():
    """Simple architecture diagram for first network"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(5, 5.5, 'Your First Neural Network Architecture',
           ha='center', fontsize=14, fontweight='bold', color=COLOR_MAIN)
    ax.text(5, 5, 'XOR Problem Solver', ha='center', fontsize=10,
           color=COLOR_ACCENT, style='italic')

    # Layer positions
    layers = [
        (2, [2, 3], "Input\n2 neurons", "x₁, x₂"),
        (5, [1.5, 2.5, 3.5], "Hidden\n4 neurons", "ReLU"),
        (8, [2.5], "Output\n1 neuron", "y")
    ]

    # Draw layers
    for x, y_positions, label, detail in layers:
        # Layer label
        ax.text(x, 4.5, label, ha='center', fontsize=9,
               fontweight='bold', color=COLOR_MAIN)

        # Draw neurons
        for y in y_positions:
            if x == 2:  # Input layer
                color = 'white'
            elif x == 5:  # Hidden layer
                color = COLOR_LIGHT
            else:  # Output layer
                color = COLOR_BLUE

            circle = Circle((x, y), 0.25, facecolor=color,
                          edgecolor=COLOR_MAIN, linewidth=2, alpha=0.8)
            ax.add_patch(circle)

        # Detail text
        ax.text(x, min(y_positions) - 0.6, detail, ha='center',
               fontsize=8, color=COLOR_ACCENT, style='italic')

    # Draw connections
    # Input to hidden
    for y1 in [2, 3]:
        for y2 in [1.5, 2.5, 3.5]:
            arrow = FancyArrowPatch((2.25, y1), (4.75, y2),
                                  connectionstyle="arc3,rad=0",
                                  arrowstyle='-', mutation_scale=10,
                                  linewidth=0.5, color=COLOR_ACCENT, alpha=0.3)
            ax.add_patch(arrow)

    # Hidden to output
    for y1 in [1.5, 2.5, 3.5]:
        arrow = FancyArrowPatch((5.25, y1), (7.75, 2.5),
                              connectionstyle="arc3,rad=0",
                              arrowstyle='-', mutation_scale=10,
                              linewidth=0.5, color=COLOR_ACCENT, alpha=0.3)
        ax.add_patch(arrow)

    # Add code snippet
    code_box = FancyBboxPatch((0.5, 0.2), 9, 0.8,
                             boxstyle="round,pad=0.05",
                             facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN,
                             linewidth=1, alpha=0.5)
    ax.add_patch(code_box)

    ax.text(5, 0.6, "keras.Sequential([Dense(4, activation='relu', input_shape=[2]), Dense(1)])",
           ha='center', fontsize=9, family='monospace', color=COLOR_MAIN)

    # Add training info
    info_text = "Training: 500 epochs | Optimizer: Adam | Loss: MSE"
    ax.text(5, 0.1, info_text, ha='center', fontsize=8,
           color=COLOR_ACCENT, style='italic')

    plt.tight_layout()
    plt.savefig('../figures/your_first_network_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all figures
print("Generating improved neural network primer figures...")

generate_spam_detector_perceptron()
print("Generated: spam_detector_perceptron.pdf")

generate_forward_pass_visualization()
print("Generated: forward_pass_visualization.pdf")

generate_multiple_detectors_intuition()
print("Generated: multiple_detectors_intuition.pdf")

generate_network_debugging_flowchart()
print("Generated: network_debugging_flowchart.pdf")

generate_your_first_network_architecture()
print("Generated: your_first_network_architecture.pdf")

print("\nAll improved figures generated successfully!")