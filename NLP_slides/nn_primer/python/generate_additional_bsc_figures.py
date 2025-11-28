#!/usr/bin/env python3
"""
Generate additional figures for the complete 55-slide BSc NN Primer
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
from matplotlib.patches import ConnectionPatch
import seaborn as sns
from pathlib import Path

# Set style for consistency
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define colors - minimalist palette
COLOR_MAIN = '#404040'
COLOR_ACCENT = '#B4B4B4'
COLOR_LIGHT = '#F0F0F0'
COLOR_ERROR = '#DC3232'
COLOR_SUCCESS = '#32B432'
COLOR_HIGHLIGHT = '#4A90E2'

# Output directory
output_dir = Path('../figures')
output_dir.mkdir(exist_ok=True)

def save_fig(name):
    """Save figure with consistent settings"""
    plt.savefig(output_dir / f"{name}.pdf", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

# 1. Network Depth Progression
def create_network_depth_progression():
    """Show evolution from 1-layer to deep networks"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Network Complexity Evolution', fontsize=14, fontweight='bold')

    # 1-Layer Network
    ax = axes[0]
    ax.set_title('1 Layer (Linear Only)', fontsize=11)
    # Draw network
    input_x = [0.2, 0.2, 0.2]
    input_y = [0.7, 0.5, 0.3]
    output_x = 0.8
    output_y = 0.5

    for x, y in zip(input_x, input_y):
        circle = Circle((x, y), 0.04, facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN)
        ax.add_patch(circle)
        ax.plot([x+0.04, output_x-0.04], [y, output_y], 'k-', alpha=0.5)

    circle = Circle((output_x, output_y), 0.04, facecolor=COLOR_SUCCESS, edgecolor=COLOR_MAIN)
    ax.add_patch(circle)

    # Show linear boundary
    ax.axvline(x=0.5, color=COLOR_ERROR, linestyle='--', alpha=0.5)
    ax.text(0.5, 0.15, 'Linear boundary only', ha='center', fontsize=9, color=COLOR_ERROR)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # 2-Layer Network
    ax = axes[1]
    ax.set_title('2 Layers (Non-linear)', fontsize=11)
    # Draw network
    hidden_x = [0.5, 0.5]
    hidden_y = [0.65, 0.35]

    for x, y in zip(input_x, input_y):
        circle = Circle((x, y), 0.04, facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN)
        ax.add_patch(circle)
        for hx, hy in zip(hidden_x, hidden_y):
            ax.plot([x+0.04, hx-0.04], [y, hy], 'k-', alpha=0.3)

    for x, y in zip(hidden_x, hidden_y):
        circle = Circle((x, y), 0.04, facecolor=COLOR_HIGHLIGHT, edgecolor=COLOR_MAIN)
        ax.add_patch(circle)
        ax.plot([x+0.04, output_x-0.04], [y, output_y], 'k-', alpha=0.3)

    circle = Circle((output_x, output_y), 0.04, facecolor=COLOR_SUCCESS, edgecolor=COLOR_MAIN)
    ax.add_patch(circle)

    # Show curved boundary
    x = np.linspace(0.3, 0.7, 100)
    y = 0.5 + 0.2*np.sin(10*x)
    ax.plot(x, y, color=COLOR_SUCCESS, linestyle='--', alpha=0.5)
    ax.text(0.5, 0.15, 'Curved boundaries', ha='center', fontsize=9, color=COLOR_SUCCESS)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Deep Network
    ax = axes[2]
    ax.set_title('Deep Network (Complex)', fontsize=11)
    # Draw network
    layer1_x = [0.35, 0.35, 0.35]
    layer1_y = [0.7, 0.5, 0.3]
    layer2_x = [0.55, 0.55]
    layer2_y = [0.6, 0.4]
    layer3_x = [0.7]
    layer3_y = [0.5]

    # Input layer
    for x, y in zip(input_x, input_y):
        circle = Circle((x, y), 0.03, facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN)
        ax.add_patch(circle)
        for l1x, l1y in zip(layer1_x, layer1_y):
            ax.plot([x+0.03, l1x-0.03], [y, l1y], 'k-', alpha=0.1)

    # First hidden
    for x, y in zip(layer1_x, layer1_y):
        circle = Circle((x, y), 0.03, facecolor='#8AC4F0', edgecolor=COLOR_MAIN)
        ax.add_patch(circle)
        for l2x, l2y in zip(layer2_x, layer2_y):
            ax.plot([x+0.03, l2x-0.03], [y, l2y], 'k-', alpha=0.1)

    # Second hidden
    for x, y in zip(layer2_x, layer2_y):
        circle = Circle((x, y), 0.03, facecolor=COLOR_HIGHLIGHT, edgecolor=COLOR_MAIN)
        ax.add_patch(circle)
        for l3x, l3y in zip(layer3_x, layer3_y):
            ax.plot([x+0.03, l3x-0.03], [y, l3y], 'k-', alpha=0.1)

    # Third hidden
    for x, y in zip(layer3_x, layer3_y):
        circle = Circle((x, y), 0.03, facecolor='#4A70E2', edgecolor=COLOR_MAIN)
        ax.add_patch(circle)
        ax.plot([x+0.03, output_x-0.03], [y, output_y], 'k-', alpha=0.2)

    # Output
    circle = Circle((output_x, output_y), 0.03, facecolor=COLOR_SUCCESS, edgecolor=COLOR_MAIN)
    ax.add_patch(circle)

    # Show complex boundary
    ax.text(0.5, 0.15, 'Any complex boundary', ha='center', fontsize=9, color=COLOR_SUCCESS)
    ax.text(0.5, 0.08, 'Hierarchical features', ha='center', fontsize=8, color=COLOR_ACCENT)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    save_fig('network_depth_progression')

# 2. Weight Evolution Heatmap
def create_weight_evolution_heatmap():
    """Show how weights change over training epochs"""
    fig, axes = plt.subplots(1, 5, figsize=(14, 4))
    fig.suptitle('Weight Evolution During Training', fontsize=14, fontweight='bold')

    epochs = [0, 10, 50, 100, 500]

    for idx, (ax, epoch) in enumerate(zip(axes, epochs)):
        ax.set_title(f'Epoch {epoch}', fontsize=10)

        # Generate weight matrix that evolves
        if epoch == 0:
            # Random initialization
            weights = np.random.randn(8, 5) * 0.1
        else:
            # Evolve towards pattern
            pattern = np.array([
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1]
            ])
            noise = np.random.randn(8, 5) * (1 / (1 + epoch/50))
            weights = pattern * (epoch / 500) + noise

        # Plot heatmap
        im = ax.imshow(weights, cmap='RdBu_r', vmin=-2, vmax=2, interpolation='nearest')
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_yticks([0, 2, 4, 6])
        ax.set_xticklabels(['P1', 'P2', 'P3', 'P4', 'P5'], fontsize=8)
        ax.set_yticklabels(['N1', 'N3', 'N5', 'N7'], fontsize=8)

        if idx == 0:
            ax.set_ylabel('Hidden Neurons', fontsize=9)
        if idx == 2:
            ax.set_xlabel('Input Pixels', fontsize=9)

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal',
                        pad=0.1, shrink=0.8)
    cbar.set_label('Weight Value', fontsize=9)

    plt.tight_layout()
    save_fig('weight_evolution_heatmap')

# 3. ResNet Highway (Skip Connections)
def create_resnet_highway():
    """Visualize skip connections as highways"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('ResNet Skip Connections: The Gradient Highway', fontsize=14, fontweight='bold')

    # Traditional deep network (left)
    ax1.set_title('Traditional Deep Network', fontsize=11)
    ax1.text(0.5, 0.95, 'Gradient Vanishing Problem', ha='center', fontsize=10, color=COLOR_ERROR)

    layers_y = np.linspace(0.1, 0.8, 6)
    for i, y in enumerate(layers_y):
        # Layer box
        rect = FancyBboxPatch((0.3, y-0.03), 0.4, 0.06,
                              boxstyle="round,pad=0.01",
                              facecolor=COLOR_LIGHT,
                              edgecolor=COLOR_MAIN,
                              linewidth=1)
        ax1.add_patch(rect)
        ax1.text(0.5, y, f'Layer {i+1}', ha='center', va='center', fontsize=9)

        # Gradient flow (getting weaker)
        if i < len(layers_y)-1:
            alpha = 0.8 - i*0.15
            color = COLOR_ERROR if i > 2 else COLOR_MAIN
            arrow = FancyArrowPatch((0.5, y+0.03), (0.5, layers_y[i+1]-0.03),
                                   arrowstyle='->', mutation_scale=15,
                                   color=color, alpha=alpha, linewidth=2-i*0.3)
            ax1.add_patch(arrow)
            if i > 2:
                ax1.text(0.75, (y + layers_y[i+1])/2, 'Weak', fontsize=8,
                        color=COLOR_ERROR, alpha=0.7)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # ResNet with skip connections (right)
    ax2.set_title('ResNet with Skip Connections', fontsize=11)
    ax2.text(0.5, 0.95, 'Gradient Highway System', ha='center', fontsize=10, color=COLOR_SUCCESS)

    for i, y in enumerate(layers_y):
        # Layer box
        rect = FancyBboxPatch((0.3, y-0.03), 0.4, 0.06,
                              boxstyle="round,pad=0.01",
                              facecolor=COLOR_LIGHT,
                              edgecolor=COLOR_MAIN,
                              linewidth=1)
        ax2.add_patch(rect)
        ax2.text(0.5, y, f'Layer {i+1}', ha='center', va='center', fontsize=9)

        # Regular connection
        if i < len(layers_y)-1:
            arrow = FancyArrowPatch((0.5, y+0.03), (0.5, layers_y[i+1]-0.03),
                                   arrowstyle='->', mutation_scale=10,
                                   color=COLOR_MAIN, alpha=0.5, linewidth=1)
            ax2.add_patch(arrow)

        # Skip connections (highways)
        if i < len(layers_y)-2:
            arrow = FancyArrowPatch((0.25, y), (0.25, layers_y[i+2]),
                                   arrowstyle='->', mutation_scale=12,
                                   color=COLOR_SUCCESS, linewidth=2,
                                   connectionstyle="arc3,rad=-.3")
            ax2.add_patch(arrow)
            ax2.text(0.15, (y + layers_y[i+2])/2, 'Highway', fontsize=8,
                    color=COLOR_SUCCESS, rotation=90)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    plt.tight_layout()
    save_fig('resnet_highway')

# 4. Attention Heatmap
def create_attention_heatmap():
    """Show what the network focuses on"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Attention Mechanism: Network Learns to Focus', fontsize=14, fontweight='bold')

    # Create sample sentence
    sentence = ['The', 'cat', 'sat', 'on', 'the', 'mat']

    # Different attention patterns
    attention_patterns = [
        # Query: "cat"
        np.array([[0.1, 0.8, 0.05, 0.03, 0.01, 0.01],
                  [0.2, 0.5, 0.15, 0.1, 0.03, 0.02],
                  [0.1, 0.3, 0.4, 0.1, 0.05, 0.05],
                  [0.05, 0.1, 0.2, 0.5, 0.1, 0.05],
                  [0.02, 0.05, 0.1, 0.2, 0.3, 0.33],
                  [0.01, 0.02, 0.05, 0.15, 0.27, 0.5]]),
        # Query: "sat"
        np.array([[0.05, 0.15, 0.6, 0.1, 0.05, 0.05],
                  [0.1, 0.2, 0.5, 0.1, 0.05, 0.05],
                  [0.05, 0.1, 0.7, 0.08, 0.04, 0.03],
                  [0.03, 0.07, 0.3, 0.4, 0.15, 0.05],
                  [0.02, 0.03, 0.15, 0.3, 0.3, 0.2],
                  [0.01, 0.02, 0.1, 0.2, 0.27, 0.4]]),
        # Self-attention
        np.array([[0.9, 0.05, 0.02, 0.01, 0.01, 0.01],
                  [0.05, 0.8, 0.1, 0.03, 0.01, 0.01],
                  [0.02, 0.1, 0.75, 0.08, 0.03, 0.02],
                  [0.01, 0.03, 0.08, 0.8, 0.05, 0.03],
                  [0.01, 0.01, 0.03, 0.05, 0.85, 0.05],
                  [0.01, 0.01, 0.02, 0.03, 0.05, 0.88]])
    ]

    titles = ['Attention on "cat"', 'Attention on "sat"', 'Self-Attention']

    for ax, attention, title in zip(axes, attention_patterns, titles):
        ax.set_title(title, fontsize=11)

        # Plot heatmap
        im = ax.imshow(attention, cmap='YlOrRd', vmin=0, vmax=1, interpolation='nearest')

        # Set ticks
        ax.set_xticks(range(len(sentence)))
        ax.set_yticks(range(len(sentence)))
        ax.set_xticklabels(sentence, fontsize=9)
        ax.set_yticklabels(sentence, fontsize=9)

        # Rotate x labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Add grid
        ax.set_xticks(np.arange(len(sentence))-0.5, minor=True)
        ax.set_yticks(np.arange(len(sentence))-0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=2)

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical',
                        pad=0.02, shrink=0.6)
    cbar.set_label('Attention Weight', fontsize=9)

    plt.tight_layout()
    save_fig('attention_heatmap')

# 5. Application Gallery
def create_application_gallery():
    """Grid showing modern AI applications"""
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.suptitle('Neural Networks in Action: Real-World Applications',
                fontsize=14, fontweight='bold')

    applications = [
        ('Image\nRecognition', 'Identify objects\nin photos', COLOR_HIGHLIGHT),
        ('Language\nTranslation', 'Google Translate\nDeepL', COLOR_SUCCESS),
        ('Voice\nAssistants', 'Siri, Alexa\nGoogle Assistant', '#FF9500'),
        ('Medical\nDiagnosis', 'Cancer detection\nX-ray analysis', COLOR_ERROR),

        ('Self-Driving\nCars', 'Tesla Autopilot\nWaymo', '#007AFF'),
        ('Game Playing', 'AlphaGo\nOpenAI Five', '#5856D6'),
        ('Art Generation', 'DALL-E\nMidjourney', '#FF2D55'),
        ('Code Writing', 'GitHub Copilot\nCodex', '#34C759'),

        ('Stock Trading', 'Prediction\nAlgorithmic trading', '#FFCC00'),
        ('Drug Discovery', 'Molecule design\nProtein folding', '#FF3B30'),
        ('Climate Models', 'Weather prediction\nClimate analysis', '#00C7BE'),
        ('Robotics', 'Movement control\nObject manipulation', '#5AC8FA')
    ]

    for ax, (title, desc, color) in zip(axes.flat, applications):
        # Create application box
        rect = FancyBboxPatch((0.1, 0.1), 0.8, 0.8,
                              boxstyle="round,pad=0.05",
                              facecolor=color, alpha=0.2,
                              edgecolor=color, linewidth=2)
        ax.add_patch(rect)

        # Add icon placeholder (circle)
        circle = Circle((0.5, 0.65), 0.15, facecolor=color, alpha=0.5)
        ax.add_patch(circle)

        # Add text
        ax.text(0.5, 0.35, title, ha='center', va='center',
                fontsize=10, fontweight='bold')
        ax.text(0.5, 0.15, desc, ha='center', va='center',
                fontsize=8, color=COLOR_ACCENT)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.tight_layout()
    save_fig('application_gallery')

# Generate all additional figures
def generate_additional_figures():
    """Generate all additional figures for the complete presentation"""
    print("Generating additional BSc Neural Network Primer figures...")

    figure_generators = [
        ('network_depth_progression', create_network_depth_progression),
        ('weight_evolution_heatmap', create_weight_evolution_heatmap),
        ('resnet_highway', create_resnet_highway),
        ('attention_heatmap', create_attention_heatmap),
        ('application_gallery', create_application_gallery)
    ]

    for name, generator in figure_generators:
        try:
            print(f"  Creating {name}...")
            generator()
        except Exception as e:
            print(f"  Error creating {name}: {e}")

    print(f"\nGenerated {len(figure_generators)} additional figures in {output_dir}")

if __name__ == "__main__":
    generate_additional_figures()