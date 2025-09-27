"""
Generate forward pass animation frames showing signal propagation
Creates step-by-step visualization of how data flows through a neural network
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# ML Theme Colors
ML_BLUE = '#0066CC'
ML_ORANGE = '#FF7F0E'
ML_GREEN = '#2CA02C'
ML_RED = '#D62728'
ML_PURPLE = '#9467BD'

def create_forward_pass_animation():
    """Create animation frames showing forward pass step by step"""

    # Network architecture: [2, 3, 2, 1]
    layers = [2, 3, 2, 1]
    layer_names = ['Input\nLayer', 'Hidden\nLayer 1', 'Hidden\nLayer 2', 'Output\nLayer']

    # Create figure with 6 frames
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Example input and computation
    input_values = [0.8, 0.3]
    np.random.seed(42)
    weights_1 = np.random.randn(2, 3) * 0.5
    weights_2 = np.random.randn(3, 2) * 0.5
    weights_3 = np.random.randn(2, 1) * 0.5

    # Compute forward pass
    h1 = np.maximum(0, np.dot(input_values, weights_1))  # ReLU
    h2 = np.maximum(0, np.dot(h1, weights_2))  # ReLU
    output = 1 / (1 + np.exp(-np.dot(h2, weights_3)))  # Sigmoid

    activations = [input_values, h1.tolist(), h2.tolist(), output.tolist()]

    # Frames to show
    frames = [
        ("Frame 1: Input Data", 0, None, None),
        ("Frame 2: First Layer Computation", 0, 1, weights_1),
        ("Frame 3: Hidden Layer 1 Activated", 1, None, None),
        ("Frame 4: Second Layer Computation", 1, 2, weights_2),
        ("Frame 5: Hidden Layer 2 Activated", 2, None, None),
        ("Frame 6: Final Output", 2, 3, weights_3),
    ]

    for frame_idx, (title, active_layer, target_layer, weights) in enumerate(frames):
        ax = axes[frame_idx]

        # Draw network
        neuron_positions = {}
        layer_spacing = 3.0
        neuron_spacing = 1.5

        # Draw all layers
        for layer_idx, n_neurons in enumerate(layers):
            x = layer_idx * layer_spacing
            y_start = -(n_neurons - 1) * neuron_spacing / 2

            for neuron_idx in range(n_neurons):
                y = y_start + neuron_idx * neuron_spacing
                neuron_positions[(layer_idx, neuron_idx)] = (x, y)

                # Determine neuron color based on frame
                if layer_idx < active_layer:
                    color = 'lightgray'  # Already computed
                    alpha = 0.3
                elif layer_idx == active_layer:
                    color = ML_GREEN  # Currently active
                    alpha = 0.8
                    # Show activation value
                    if activations[layer_idx]:
                        value = activations[layer_idx][neuron_idx]
                        ax.text(x, y - 0.6, f'{value:.2f}',
                               ha='center', fontsize=9, fontweight='bold')
                elif layer_idx == target_layer and weights is not None:
                    color = ML_ORANGE  # Target of computation
                    alpha = 0.5
                else:
                    color = 'white'  # Not yet computed
                    alpha = 0.3

                # Draw neuron
                circle = Circle((x, y), 0.3, facecolor=color, edgecolor='black',
                               linewidth=2, alpha=alpha, zorder=3)
                ax.add_patch(circle)

        # Draw connections
        for layer_idx in range(len(layers) - 1):
            for neuron_from in range(layers[layer_idx]):
                for neuron_to in range(layers[layer_idx + 1]):
                    x1, y1 = neuron_positions[(layer_idx, neuron_from)]
                    x2, y2 = neuron_positions[(layer_idx + 1, neuron_to)]

                    # Highlight connections being used
                    if layer_idx == active_layer and layer_idx + 1 == target_layer and weights is not None:
                        weight_val = weights[neuron_from, neuron_to]
                        linewidth = 3
                        alpha = 0.7
                        color = ML_RED if weight_val > 0 else ML_BLUE
                        # Show weight value
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                        if abs(weight_val) > 0.3:  # Only show significant weights
                            ax.text(mid_x, mid_y, f'{weight_val:.1f}',
                                   fontsize=7, ha='center',
                                   bbox=dict(boxstyle='round,pad=0.2',
                                           facecolor='yellow', alpha=0.5))
                    elif layer_idx < active_layer:
                        linewidth = 1
                        alpha = 0.2
                        color = 'gray'
                    else:
                        linewidth = 1
                        alpha = 0.1
                        color = 'lightgray'

                    ax.plot([x1, x2], [y1, y2], color=color,
                           linewidth=linewidth, alpha=alpha, zorder=1)

        # Add layer labels
        for layer_idx, name in enumerate(layer_names):
            x = layer_idx * layer_spacing
            y_max = max([neuron_positions[(layer_idx, i)][1]
                        for i in range(layers[layer_idx])])
            ax.text(x, y_max + 1, name, ha='center', fontsize=10, fontweight='bold')

        # Add computation equation for computation frames
        if weights is not None:
            eq_text = f"Computing: h{target_layer} = f(W{active_layer+1} × h{active_layer})"
            ax.text(1.5 * layer_spacing, -3.5, eq_text,
                   ha='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow'))

        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlim(-0.5, (len(layers) - 1) * layer_spacing + 0.5)
        ax.set_ylim(-4, 4)
        ax.axis('off')

    fig.suptitle('Forward Pass: Step-by-Step Signal Propagation',
                fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/forward_pass_frames.pdf', dpi=300, bbox_inches='tight')
    print("Saved: forward_pass_frames.pdf")

def create_detailed_computation_example():
    """Create detailed example showing one neuron's computation"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Visual representation
    ax = axes[0]

    # Input neurons
    inputs = [0.5, 0.8, 0.3]
    weights = [0.7, -0.4, 0.9]
    bias = -0.2

    # Draw input neurons
    for i, (inp, weight) in enumerate(zip(inputs, weights)):
        y = 2 - i * 1.5
        # Input neuron
        circle = Circle((-2, y), 0.3, facecolor=ML_BLUE, edgecolor='black',
                       linewidth=2, alpha=0.7)
        ax.add_patch(circle)
        ax.text(-2, y, f'{inp}', ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')
        ax.text(-2.8, y, f'x{i+1}', ha='right', fontsize=9)

        # Weight on connection
        arrow = FancyArrowPatch((-1.7, y), (-0.3, 0),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=3, color=ML_RED if weight > 0 else ML_BLUE)
        ax.add_patch(arrow)
        ax.text(-1, y*0.5, f'w{i+1}={weight}', ha='center', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Target neuron
    circle = Circle((0, 0), 0.4, facecolor=ML_ORANGE, edgecolor='black',
                   linewidth=3, alpha=0.8)
    ax.add_patch(circle)
    ax.text(0, 0, '?', ha='center', va='center',
           fontsize=16, fontweight='bold')

    # Bias
    ax.text(0, -1, f'bias = {bias}', ha='center', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

    # Output
    z = sum([inp * weight for inp, weight in zip(inputs, weights)]) + bias
    activation = max(0, z)  # ReLU
    arrow = FancyArrowPatch((0.4, 0), (1.5, 0),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=3, color=ML_GREEN)
    ax.add_patch(arrow)
    circle = Circle((2, 0), 0.4, facecolor=ML_GREEN, edgecolor='black',
                   linewidth=3, alpha=0.8)
    ax.add_patch(circle)
    ax.text(2, 0, f'{activation:.2f}', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    ax.text(2, -0.8, 'Output', ha='center', fontsize=9)

    ax.set_title('Single Neuron Computation', fontsize=14, fontweight='bold')
    ax.set_xlim(-3.5, 2.8)
    ax.set_ylim(-2, 2.5)
    ax.axis('off')

    # Right panel: Step-by-step calculation
    ax = axes[1]

    steps = [
        "Step 1: Weighted Sum",
        f"z = w₁×x₁ + w₂×x₂ + w₃×x₃ + b",
        f"z = {weights[0]}×{inputs[0]} + {weights[1]}×{inputs[1]} + {weights[2]}×{inputs[2]} + {bias}",
        f"z = {weights[0]*inputs[0]:.2f} + {weights[1]*inputs[1]:.2f} + {weights[2]*inputs[2]:.2f} + {bias}",
        f"z = {z:.2f}",
        "",
        "Step 2: Apply Activation (ReLU)",
        "f(z) = max(0, z)",
        f"f({z:.2f}) = max(0, {z:.2f})",
        f"output = {activation:.2f}"
    ]

    y_pos = 0.95
    for i, step in enumerate(steps):
        if step == "":
            y_pos -= 0.05
        elif "Step" in step:
            ax.text(0.05, y_pos, step, fontsize=12, fontweight='bold',
                   transform=ax.transAxes)
            y_pos -= 0.08
        else:
            ax.text(0.1, y_pos, step, fontsize=10, family='monospace',
                   transform=ax.transAxes)
            y_pos -= 0.07

    # Highlight final answer
    ax.add_patch(mpatches.FancyBboxPatch((0.05, 0.02), 0.4, 0.08,
                                        boxstyle="round,pad=0.01",
                                        edgecolor=ML_GREEN, facecolor=ML_GREEN,
                                        alpha=0.2, linewidth=3,
                                        transform=ax.transAxes))

    ax.set_title('Detailed Calculation', fontsize=14, fontweight='bold')
    ax.axis('off')

    fig.suptitle('Understanding Neuron Computation', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/neuron_computation_detail.pdf', dpi=300, bbox_inches='tight')
    print("Saved: neuron_computation_detail.pdf")

if __name__ == "__main__":
    print("Generating forward pass visualizations...")

    create_forward_pass_animation()
    create_detailed_computation_example()

    print("\nAll forward pass visualizations generated successfully!")
    print("Files created:")
    print("  - forward_pass_frames.pdf")
    print("  - neuron_computation_detail.pdf")