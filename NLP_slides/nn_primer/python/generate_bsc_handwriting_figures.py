#!/usr/bin/env python3
"""
Generate all figures for BSc-level Neural Network Primer with handwriting narrative
Focuses on visual clarity and intuitive understanding
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import seaborn as sns
from pathlib import Path

# Set style for consistency
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define colors - minimalist palette
COLOR_MAIN = '#404040'      # RGB(64,64,64)
COLOR_ACCENT = '#B4B4B4'    # RGB(180,180,180)
COLOR_LIGHT = '#F0F0F0'     # RGB(240,240,240)
COLOR_ERROR = '#DC3232'     # Red for errors
COLOR_SUCCESS = '#32B432'   # Green for success
COLOR_HIGHLIGHT = '#4A90E2' # Blue for emphasis

# Output directory
output_dir = Path('../figures')
output_dir.mkdir(exist_ok=True)

def save_fig(name):
    """Save figure with consistent settings"""
    plt.savefig(output_dir / f"{name}.pdf", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

# 1. Handwriting Variations
def create_handwriting_variations():
    """Show 20 different ways to write the letter A"""
    fig, axes = plt.subplots(4, 5, figsize=(12, 8))
    fig.suptitle('20 Different Ways People Write "A"', fontsize=16, fontweight='bold')

    # Different A styles (simplified representations)
    styles = [
        # Row 1: Print variations
        [[0,1,1,1,0], [1,0,0,0,1], [1,1,1,1,1], [1,0,0,0,1], [1,0,0,0,1]],
        [[0,0,1,0,0], [0,1,0,1,0], [1,1,1,1,1], [1,0,0,0,1], [1,0,0,0,1]],
        [[0,1,1,0,0], [1,0,0,1,0], [1,1,1,1,0], [1,0,0,1,0], [1,0,0,1,0]],
        [[0,0,1,1,0], [0,1,0,0,1], [0,1,1,1,1], [0,1,0,0,1], [0,1,0,0,1]],
        [[1,1,1,1,0], [1,0,0,0,1], [1,1,1,1,1], [1,0,0,0,1], [1,0,0,0,1]],
        # Row 2: Cursive variations
        [[0,0,1,0,0], [0,1,0,1,0], [1,0,0,0,1], [1,1,1,1,1], [1,0,0,0,1]],
        [[0,1,1,1,0], [1,0,0,0,0], [1,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1]],
        [[0,0,1,0,0], [0,1,0,1,0], [0,1,0,1,0], [1,1,1,1,1], [1,0,0,0,1]],
        [[0,1,1,0,0], [1,0,0,1,0], [1,0,0,1,0], [1,1,1,1,0], [1,0,0,0,1]],
        [[0,0,1,0,0], [0,1,0,1,0], [1,0,0,0,1], [1,1,1,1,1], [1,0,0,1,0]],
        # Row 3: Stylized variations
        [[1,0,0,0,1], [0,1,0,1,0], [0,0,1,0,0], [0,1,1,1,0], [1,0,0,0,1]],
        [[1,1,1,0,0], [0,0,0,1,0], [0,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1]],
        [[0,1,0,0,0], [1,0,1,0,0], [1,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1]],
        [[0,0,0,1,0], [0,0,1,0,1], [0,1,1,1,1], [1,0,0,0,1], [1,0,0,0,1]],
        [[0,1,1,1,1], [1,0,0,0,0], [1,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1]],
        # Row 4: Handwritten variations
        [[0,0,1,0,0], [0,1,0,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,1]],
        [[0,0,1,1,0], [0,1,0,0,1], [1,1,1,1,1], [1,0,0,0,1], [1,0,0,0,1]],
        [[1,1,1,0,0], [1,0,0,1,0], [1,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1]],
        [[0,0,1,0,0], [0,1,0,1,0], [0,1,0,1,0], [1,0,0,0,1], [1,1,1,1,1]],
        [[0,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,1], [1,0,0,0,1]]
    ]

    for idx, ax in enumerate(axes.flat):
        ax.imshow(styles[idx], cmap='gray_r', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Style {idx+1}', fontsize=10)

    plt.tight_layout()
    save_fig('handwriting_variations')

# 2. Pixel to Numbers Transformation
def create_pixel_to_numbers():
    """Visualize how a letter becomes numbers"""
    fig = plt.figure(figsize=(14, 6))

    # Create letter A pattern
    letter_a = np.array([
        [0,0,1,0,0],
        [0,1,0,1,0],
        [1,0,0,0,1],
        [1,1,1,1,1],
        [1,0,0,0,1]
    ])

    # Panel 1: Original letter
    ax1 = plt.subplot(1, 4, 1)
    ax1.text(0.5, 0.5, 'A', fontsize=80, ha='center', va='center',
             fontweight='bold', color=COLOR_MAIN)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('1. Handwritten Letter', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Panel 2: Grid overlay
    ax2 = plt.subplot(1, 4, 2)
    ax2.text(0.5, 0.5, 'A', fontsize=80, ha='center', va='center',
             fontweight='bold', color=COLOR_ACCENT, alpha=0.5)
    # Add grid
    for i in range(6):
        ax2.axhline(i/5, color=COLOR_MAIN, linewidth=1, alpha=0.5)
        ax2.axvline(i/5, color=COLOR_MAIN, linewidth=1, alpha=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('2. Divide into Grid', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Panel 3: Pixelated
    ax3 = plt.subplot(1, 4, 3)
    ax3.imshow(letter_a, cmap='gray_r', interpolation='nearest')
    ax3.set_title('3. Black/White Pixels', fontsize=12, fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Panel 4: Numbers
    ax4 = plt.subplot(1, 4, 4)
    ax4.imshow(letter_a, cmap='gray_r', interpolation='nearest', alpha=0.3)
    for i in range(5):
        for j in range(5):
            ax4.text(j, i, str(letter_a[i, j]), ha='center', va='center',
                    fontsize=14, fontweight='bold', color=COLOR_MAIN)
    ax4.set_title('4. Convert to Numbers', fontsize=12, fontweight='bold')
    ax4.set_xticks([])
    ax4.set_yticks([])

    plt.suptitle('From Handwriting to Numbers', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    save_fig('pixel_to_numbers')

# 3. Rules vs Learning
def create_rules_vs_learning():
    """Show why rules fail and learning succeeds"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Left: Rules approach
    ax1.set_title('Traditional Programming', fontsize=14, fontweight='bold')
    ax1.text(0.5, 0.9, 'IF-THEN Rules:', fontsize=12, ha='center', fontweight='bold')

    rules = [
        'IF has_triangle_top THEN maybe_A',
        'IF has_horizontal_bar THEN maybe_A',
        'IF two_diagonal_lines THEN maybe_A',
        'IF all_conditions THEN is_A',
        '...',
        'But what about cursive?',
        'What about rotated?',
        'What about stylized?',
        '❌ Impossible to cover all cases!'
    ]

    for i, rule in enumerate(rules):
        color = COLOR_ERROR if '❌' in rule or '?' in rule else COLOR_MAIN
        weight = 'bold' if '❌' in rule else 'normal'
        ax1.text(0.1, 0.75 - i*0.08, rule, fontsize=10,
                color=color, fontweight=weight)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # Right: Learning approach
    ax2.set_title('Machine Learning', fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.9, 'Learning from Examples:', fontsize=12, ha='center', fontweight='bold')

    # Show examples flowing to pattern
    examples_y = [0.7, 0.6, 0.5, 0.4]
    for i, y in enumerate(examples_y):
        ax2.text(0.1, y, f'Example {i+1}: A', fontsize=10, color=COLOR_MAIN)
        arrow = FancyArrowPatch((0.25, y), (0.4, 0.5),
                               arrowstyle='->', mutation_scale=15,
                               color=COLOR_ACCENT, alpha=0.5)
        ax2.add_patch(arrow)

    # Central learning box
    box = FancyBboxPatch((0.4, 0.45), 0.3, 0.1,
                         boxstyle="round,pad=0.02",
                         facecolor=COLOR_LIGHT,
                         edgecolor=COLOR_MAIN,
                         linewidth=2)
    ax2.add_patch(box)
    ax2.text(0.55, 0.5, 'Learn\nPattern', fontsize=11, ha='center', va='center',
            fontweight='bold')

    # Output arrow
    arrow = FancyArrowPatch((0.7, 0.5), (0.85, 0.5),
                           arrowstyle='->', mutation_scale=20,
                           color=COLOR_SUCCESS, linewidth=2)
    ax2.add_patch(arrow)

    ax2.text(0.9, 0.5, '✓', fontsize=24, ha='center', va='center',
            color=COLOR_SUCCESS, fontweight='bold')

    ax2.text(0.5, 0.2, '✅ Handles any variation!', fontsize=12,
            ha='center', color=COLOR_SUCCESS, fontweight='bold')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    plt.suptitle('Why Rules Fail and Learning Works', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('rules_vs_learning')

# 4. Neuron as Voter
def create_neuron_as_voter():
    """Visualize neuron as a voting system"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Title
    ax.text(0.5, 0.95, 'A Neuron is Like a Voting System',
            fontsize=16, ha='center', fontweight='bold')

    # Input votes
    inputs = ['Top pixel\n(black)', 'Middle pixel\n(black)',
              'Bottom pixel\n(white)', 'Corner pixel\n(black)']
    values = [1, 1, 0, 1]
    weights = [3, 2, -1, 1]

    y_positions = np.linspace(0.7, 0.3, 4)

    for i, (inp, val, w, y) in enumerate(zip(inputs, values, weights, y_positions)):
        # Input box
        ax.text(0.1, y, inp, fontsize=10, ha='center', va='center')
        ax.text(0.1, y-0.05, f'Value: {val}', fontsize=9, ha='center',
                color=COLOR_ACCENT)

        # Weight (importance)
        color = COLOR_SUCCESS if w > 0 else COLOR_ERROR
        ax.text(0.3, y, f'Weight: {w:+d}', fontsize=11, ha='center',
                color=color, fontweight='bold')
        ax.text(0.3, y-0.05, f'{"Important" if abs(w) > 1 else "Less important"}',
                fontsize=9, ha='center', color=COLOR_ACCENT)

        # Contribution
        contribution = val * w
        ax.text(0.5, y, f'{val} × {w:+d} = {contribution:+d}',
                fontsize=11, ha='center')

        # Arrow to sum
        arrow = FancyArrowPatch((0.6, y), (0.7, 0.5),
                               arrowstyle='->', mutation_scale=10,
                               color=COLOR_ACCENT, alpha=0.5)
        ax.add_patch(arrow)

    # Sum box (neuron body)
    neuron = Circle((0.75, 0.5), 0.08, facecolor=COLOR_LIGHT,
                   edgecolor=COLOR_MAIN, linewidth=2)
    ax.add_patch(neuron)

    total = sum(v*w for v, w in zip(values, weights))
    ax.text(0.75, 0.5, f'Σ = {total}', fontsize=12, ha='center', va='center',
            fontweight='bold')

    # Decision threshold
    ax.text(0.75, 0.35, f'Threshold: 5', fontsize=10, ha='center',
            color=COLOR_ACCENT)
    ax.text(0.75, 0.3, f'{total} > 5?', fontsize=10, ha='center')

    # Output arrow
    arrow = FancyArrowPatch((0.83, 0.5), (0.9, 0.5),
                           arrowstyle='->', mutation_scale=15,
                           color=COLOR_SUCCESS, linewidth=2)
    ax.add_patch(arrow)

    # Final decision
    ax.text(0.95, 0.5, 'YES!\nIt\'s an A', fontsize=12, ha='center', va='center',
            color=COLOR_SUCCESS, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    save_fig('neuron_as_voter')

# 5. Weight Adjustment Visualization
def create_weight_adjustment_visual():
    """Show how weights change when learning"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    titles = ['Before Training', 'Error Detected!', 'After Adjustment']
    weights_before = [0.5, 0.5, 0.5, 0.5]
    weights_after = [0.8, 0.7, 0.2, 0.6]

    for idx, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Draw neuron
        neuron = Circle((0.7, 0.5), 0.1, facecolor=COLOR_LIGHT,
                       edgecolor=COLOR_MAIN, linewidth=2)
        ax.add_patch(neuron)

        # Draw inputs and weights
        y_positions = np.linspace(0.8, 0.2, 4)
        for i, y in enumerate(y_positions):
            # Input
            ax.text(0.1, y, f'Input {i+1}', fontsize=10)

            if idx == 0:  # Before
                w = weights_before[i]
                color = COLOR_ACCENT
            elif idx == 1:  # Error
                w = weights_before[i]
                color = COLOR_ERROR
                ax.text(0.9, 0.5, '❌\nWrong!', fontsize=14, ha='center',
                       color=COLOR_ERROR, fontweight='bold')
            else:  # After
                w = weights_after[i]
                color = COLOR_SUCCESS if w > weights_before[i] else COLOR_ERROR

            # Weight on connection
            arrow = FancyArrowPatch((0.25, y), (0.6, 0.5),
                                   arrowstyle='->', mutation_scale=10,
                                   color=color, linewidth=2 if idx==2 else 1,
                                   alpha=0.7)
            ax.add_patch(arrow)
            ax.text(0.4, y, f'w={w:.1f}', fontsize=10, ha='center',
                   color=color, fontweight='bold' if idx==2 else 'normal')

            # Show change in third panel
            if idx == 2:
                change = weights_after[i] - weights_before[i]
                if change != 0:
                    ax.text(0.4, y-0.08, f'({change:+.1f})', fontsize=9,
                           ha='center', color=color)

        # Output
        if idx != 1:
            arrow = FancyArrowPatch((0.8, 0.5), (0.9, 0.5),
                                   arrowstyle='->', mutation_scale=12,
                                   color=COLOR_SUCCESS if idx==2 else COLOR_MAIN)
            ax.add_patch(arrow)
            if idx == 2:
                ax.text(0.95, 0.5, '✓', fontsize=20, ha='center',
                       color=COLOR_SUCCESS, fontweight='bold')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.suptitle('Learning: Adjusting Weights Based on Errors',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('weight_adjustment_visual')

# 6. Training Steps Numbered
def create_training_steps_numbered():
    """Show 5 steps of training with actual numbers"""
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))

    # Simulated training progress
    epochs = [1, 2, 3, 4, 5]
    accuracies = [0.2, 0.35, 0.6, 0.85, 0.95]
    errors = [0.8, 0.65, 0.4, 0.15, 0.05]

    for idx, (ax, epoch, acc, err) in enumerate(zip(axes, epochs, accuracies, errors)):
        ax.set_title(f'Step {epoch}', fontsize=11, fontweight='bold')

        # Show pattern being learned (progressively clearer)
        noise_level = 1 - acc
        pattern = np.random.random((5, 5))
        if idx > 2:  # Start showing A pattern
            a_pattern = np.array([[0,0,1,0,0],
                                 [0,1,0,1,0],
                                 [1,1,1,1,1],
                                 [1,0,0,0,1],
                                 [1,0,0,0,1]])
            pattern = a_pattern * acc + pattern * noise_level

        im = ax.imshow(pattern, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')

        # Stats below
        ax.text(0.5, -0.15, f'Accuracy: {acc*100:.0f}%',
                transform=ax.transAxes, ha='center',
                color=COLOR_SUCCESS if acc > 0.8 else COLOR_MAIN)
        ax.text(0.5, -0.3, f'Error: {err:.2f}',
                transform=ax.transAxes, ha='center',
                color=COLOR_ERROR if err > 0.5 else COLOR_MAIN, fontsize=9)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Training Progress: From Random to Recognition',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('training_steps_numbered')

# 7. XOR Impossible Line
def create_xor_impossible_line():
    """Show why XOR can't be solved with single line"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # XOR data points
    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]
    colors = ['blue', 'red', 'red', 'blue']  # XOR pattern
    labels = ['0', '1', '1', '0']

    for ax, title in zip([ax1, ax2], ['The XOR Problem', 'Why One Line Fails']):
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Plot points
        for xi, yi, c, label in zip(x, y, colors, labels):
            ax.scatter(xi, yi, s=200, c=c, edgecolor='black', linewidth=2, zorder=5)
            ax.text(xi, yi, label, fontsize=12, ha='center', va='center',
                   color='white', fontweight='bold', zorder=6)

        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel('Input 1', fontsize=11)
        ax.set_ylabel('Input 2', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add legend
        ax.text(0.5, 1.15, 'Red = Output 1, Blue = Output 0',
               transform=ax.transData, ha='center', fontsize=10)

    # Try to draw separating lines on second plot
    ax2.plot([-0.2, 1.2], [0.5, 0.5], 'g--', linewidth=2, alpha=0.5, label='Attempt 1')
    ax2.plot([0.5, 0.5], [-0.2, 1.2], 'y--', linewidth=2, alpha=0.5, label='Attempt 2')
    ax2.plot([-0.2, 1.2], [1.2, -0.2], 'm--', linewidth=2, alpha=0.5, label='Attempt 3')

    ax2.text(0.5, -0.35, '❌ No single line can separate red from blue!',
            ha='center', fontsize=11, color=COLOR_ERROR, fontweight='bold',
            transform=ax2.transData)

    plt.tight_layout()
    save_fig('xor_impossible_line')

# 8. XOR Two Questions Solution
def create_xor_two_questions():
    """Show how two neurons solve XOR"""
    fig = plt.figure(figsize=(14, 8))

    # Main title
    fig.suptitle('Solving XOR with Two Hidden Neurons', fontsize=14, fontweight='bold')

    # Left panel: Hidden neuron 1
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_title('Hidden Neuron 1:\n"Is it (0,1)?"', fontsize=11)
    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]
    colors1 = ['lightblue', 'red', 'lightblue', 'lightblue']
    for xi, yi, c in zip(x, y, colors1):
        ax1.scatter(xi, yi, s=150, c=c, edgecolor='black', linewidth=1.5)
    ax1.plot([-0.2, 0.5], [0.5, 0.5], 'r-', linewidth=2)
    ax1.plot([0.5, 0.5], [0.5, 1.2], 'r-', linewidth=2)
    ax1.set_xlim(-0.2, 1.2)
    ax1.set_ylim(-0.2, 1.2)
    ax1.grid(True, alpha=0.3)

    # Middle panel: Hidden neuron 2
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_title('Hidden Neuron 2:\n"Is it (1,0)?"', fontsize=11)
    colors2 = ['lightblue', 'lightblue', 'red', 'lightblue']
    for xi, yi, c in zip(x, y, colors2):
        ax2.scatter(xi, yi, s=150, c=c, edgecolor='black', linewidth=1.5)
    ax2.plot([0.5, 1.2], [0.5, 0.5], 'r-', linewidth=2)
    ax2.plot([0.5, 0.5], [-0.2, 0.5], 'r-', linewidth=2)
    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.grid(True, alpha=0.3)

    # Right panel: Combined output
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title('Output Neuron:\n"Is either true?"', fontsize=11)
    colors_final = ['blue', 'red', 'red', 'blue']
    labels = ['0', '1', '1', '0']
    for xi, yi, c, l in zip(x, y, colors_final, labels):
        ax3.scatter(xi, yi, s=150, c=c, edgecolor='black', linewidth=1.5)
        ax3.text(xi, yi, l, fontsize=10, ha='center', va='center',
                color='white', fontweight='bold')
    ax3.set_xlim(-0.2, 1.2)
    ax3.set_ylim(-0.2, 1.2)
    ax3.grid(True, alpha=0.3)
    ax3.text(0.5, -0.3, '✓ Perfect XOR!', ha='center', fontsize=11,
            color=COLOR_SUCCESS, fontweight='bold', transform=ax3.transData)

    # Bottom: Network diagram
    ax4 = plt.subplot(2, 1, 2)
    ax4.set_title('The Complete Network', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # Network positions
    input_x = [0.2, 0.2]
    input_y = [0.7, 0.3]
    hidden_x = [0.5, 0.5]
    hidden_y = [0.7, 0.3]
    output_x = 0.8
    output_y = 0.5

    # Draw neurons
    for x, y, label in zip(input_x, input_y, ['X₁', 'X₂']):
        circle = Circle((x, y), 0.05, facecolor='lightblue', edgecolor='black')
        ax4.add_patch(circle)
        ax4.text(x, y, label, ha='center', va='center', fontsize=10)

    for x, y, label in zip(hidden_x, hidden_y, ['H₁', 'H₂']):
        circle = Circle((x, y), 0.05, facecolor='lightgreen', edgecolor='black')
        ax4.add_patch(circle)
        ax4.text(x, y, label, ha='center', va='center', fontsize=10)

    circle = Circle((output_x, output_y), 0.05, facecolor='lightyellow', edgecolor='black')
    ax4.add_patch(circle)
    ax4.text(output_x, output_y, 'Y', ha='center', va='center', fontsize=10)

    # Draw connections
    for ix, iy in zip(input_x, input_y):
        for hx, hy in zip(hidden_x, hidden_y):
            ax4.plot([ix, hx], [iy, hy], 'k-', alpha=0.3)

    for hx, hy in zip(hidden_x, hidden_y):
        ax4.plot([hx, output_x], [hy, output_y], 'k-', alpha=0.3)

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    save_fig('xor_two_questions')

# 9. Hidden Layer Factory
def create_hidden_layer_factory():
    """Assembly line metaphor for layers"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title('Neural Network as Assembly Line', fontsize=14, fontweight='bold')

    # Assembly line stages
    stages = ['Raw Materials\n(Pixels)', 'Stage 1\n(Find Edges)',
              'Stage 2\n(Find Shapes)', 'Final Product\n(Letter)']
    x_positions = [0.1, 0.35, 0.6, 0.85]

    for x, stage in zip(x_positions, stages):
        # Conveyor belt section
        rect = FancyBboxPatch((x-0.08, 0.3), 0.16, 0.4,
                              boxstyle="round,pad=0.01",
                              facecolor=COLOR_LIGHT,
                              edgecolor=COLOR_MAIN,
                              linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 0.75, stage, ha='center', va='center',
                fontsize=11, fontweight='bold')

        # Show processing at each stage
        if x == 0.1:
            # Raw pixels
            ax.text(x, 0.5, '█ █ █\n█   █\n█ █ █', ha='center', va='center',
                   fontsize=10, family='monospace')
        elif x == 0.35:
            # Edges
            ax.text(x, 0.5, '| — \\', ha='center', va='center',
                   fontsize=14, fontweight='bold')
        elif x == 0.6:
            # Shapes
            ax.text(x, 0.5, '△ □', ha='center', va='center',
                   fontsize=14, fontweight='bold')
        else:
            # Final letter
            ax.text(x, 0.5, 'A', ha='center', va='center',
                   fontsize=20, fontweight='bold', color=COLOR_SUCCESS)

    # Arrows between stages
    for i in range(len(x_positions)-1):
        arrow = FancyArrowPatch((x_positions[i]+0.08, 0.5),
                               (x_positions[i+1]-0.08, 0.5),
                               arrowstyle='->', mutation_scale=20,
                               color=COLOR_HIGHLIGHT, linewidth=2)
        ax.add_patch(arrow)

    # Labels
    ax.text(0.5, 0.15, 'Each stage refines and combines features from the previous stage',
           ha='center', fontsize=11, color=COLOR_ACCENT, style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    save_fig('hidden_layer_factory')

# 10. Feature Hierarchy
def create_feature_hierarchy():
    """Show how features build from simple to complex"""
    fig, axes = plt.subplots(1, 4, figsize=(14, 6))
    fig.suptitle('Building Complex Features from Simple Ones', fontsize=14, fontweight='bold')

    titles = ['Layer 1: Edges', 'Layer 2: Parts', 'Layer 3: Components', 'Output: Letter']

    for idx, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        if idx == 0:  # Edges
            ax.plot([2, 8], [5, 5], 'k-', linewidth=3)  # horizontal
            ax.plot([5, 5], [2, 8], 'k-', linewidth=3)  # vertical
            ax.plot([2, 8], [2, 8], 'k-', linewidth=3)  # diagonal
            ax.plot([8, 2], [2, 8], 'k-', linewidth=3)  # other diagonal

        elif idx == 1:  # Parts
            # Corner
            ax.plot([2, 5], [5, 5], 'b-', linewidth=3)
            ax.plot([5, 5], [5, 8], 'b-', linewidth=3)
            # Intersection
            ax.plot([6, 9], [3, 3], 'r-', linewidth=3)
            ax.plot([7.5, 7.5], [1.5, 4.5], 'r-', linewidth=3)
            # Curve
            ax.add_patch(patches.Arc((5, 7), 3, 3, angle=0, theta1=0, theta2=180,
                                    color='g', linewidth=3))

        elif idx == 2:  # Components
            # Triangle top
            ax.plot([5, 3, 7, 5], [8, 4, 4, 8], 'purple', linewidth=3)
            # Cross bar
            ax.plot([3.5, 6.5], [5, 5], 'orange', linewidth=3)

        else:  # Complete letter
            # Draw 'A'
            ax.plot([5, 2, 5], [9, 1, 5], 'k-', linewidth=4)
            ax.plot([5, 8, 5], [9, 1, 5], 'k-', linewidth=4)
            ax.plot([3.5, 6.5], [5, 5], 'k-', linewidth=4)
            ax.text(5, -0.5, 'Complete A!', ha='center', fontsize=11,
                   color=COLOR_SUCCESS, fontweight='bold')

    plt.tight_layout()
    save_fig('feature_hierarchy')

# Continue with remaining figures...
# 11. Blame Distribution
def create_blame_distribution():
    """Pie chart showing error responsibility"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Left: Network with error
    ax1.set_title('Network Made an Error', fontsize=12, fontweight='bold')
    ax1.text(0.5, 0.8, 'Predicted: B', fontsize=14, ha='center', color=COLOR_ERROR)
    ax1.text(0.5, 0.65, 'Correct: A', fontsize=14, ha='center', color=COLOR_SUCCESS)
    ax1.text(0.5, 0.45, 'Who\'s responsible?', fontsize=12, ha='center', style='italic')

    # Draw simple network
    layers_x = [0.2, 0.5, 0.8]
    neurons_y = [[0.3, 0.2], [0.35, 0.25, 0.15], [0.2]]

    for x, ys in zip(layers_x, neurons_y):
        for y in ys:
            circle = Circle((x, y), 0.03, facecolor=COLOR_LIGHT,
                          edgecolor=COLOR_ERROR, linewidth=2)
            ax1.add_patch(circle)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # Right: Blame distribution
    ax2.set_title('Distributing the Blame', fontsize=12, fontweight='bold')

    labels = ['Output Layer\n(35%)', 'Hidden Layer 2\n(25%)',
              'Hidden Layer 1\n(20%)', 'Input Weights\n(20%)']
    sizes = [35, 25, 20, 20]
    colors = [COLOR_ERROR, '#FF9999', '#FFCCCC', COLOR_LIGHT]
    explode = (0.1, 0.05, 0, 0)

    ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.0f%%', shadow=True, startangle=90)

    ax2.text(0, -1.5, 'Each layer adjusts based on its share of blame',
            ha='center', fontsize=11, color=COLOR_ACCENT, style='italic')

    plt.tight_layout()
    save_fig('blame_distribution')

# 12. Error Flow Backward
def create_error_flow_backward():
    """Visualize backpropagation flow"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title('Backpropagation: Error Flows Backward', fontsize=14, fontweight='bold')

    # Network layers
    layers = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
    x_positions = [0.15, 0.35, 0.55, 0.75]

    # Forward pass (light arrows)
    for i in range(len(x_positions)-1):
        arrow = FancyArrowPatch((x_positions[i]+0.05, 0.6),
                               (x_positions[i+1]-0.05, 0.6),
                               arrowstyle='->', mutation_scale=15,
                               color=COLOR_ACCENT, alpha=0.3, linewidth=1)
        ax.add_patch(arrow)

    ax.text(0.45, 0.65, 'Forward Pass', ha='center', fontsize=10,
           color=COLOR_ACCENT, alpha=0.7)

    # Backward pass (bold arrows with gradient)
    gradient_colors = ['#FF0000', '#FF4444', '#FF8888', '#FFCCCC']
    for i in range(len(x_positions)-1, 0, -1):
        arrow = FancyArrowPatch((x_positions[i]-0.05, 0.4),
                               (x_positions[i-1]+0.05, 0.4),
                               arrowstyle='->', mutation_scale=20,
                               color=gradient_colors[3-i], linewidth=3)
        ax.add_patch(arrow)

    ax.text(0.45, 0.33, 'Backward Pass (Error)', ha='center', fontsize=11,
           color=COLOR_ERROR, fontweight='bold')

    # Draw layers
    for x, layer in zip(x_positions, layers):
        rect = FancyBboxPatch((x-0.05, 0.45), 0.1, 0.1,
                              boxstyle="round,pad=0.01",
                              facecolor=COLOR_LIGHT,
                              edgecolor=COLOR_MAIN,
                              linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 0.5, layer, ha='center', va='center',
                fontsize=10, fontweight='bold')
        ax.text(x, 0.25, f'Adjust\nWeights', ha='center', va='center',
                fontsize=9, color=COLOR_ACCENT)

    # Error indicator
    ax.text(0.75, 0.7, 'ERROR!', ha='center', fontsize=12,
           color=COLOR_ERROR, fontweight='bold')

    # Explanation
    ax.text(0.5, 0.1, 'Error signal tells each layer how to improve its weights',
           ha='center', fontsize=11, color=COLOR_ACCENT, style='italic')

    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 0.8)
    ax.axis('off')

    plt.tight_layout()
    save_fig('error_flow_backward')

# 13. Gradient Landscape 3D (simplified)
def create_gradient_landscape_3d():
    """Simple 3D valley for gradient descent"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create valley surface
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # Simple bowl shape

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.6,
                           linewidth=0, antialiased=True)

    # Show gradient descent path
    theta = np.linspace(0, 4*np.pi, 100)
    r = np.exp(-theta/4)
    path_x = r * np.cos(theta)
    path_y = r * np.sin(theta)
    path_z = path_x**2 + path_y**2

    ax.plot(path_x, path_y, path_z, 'r-', linewidth=3, label='Descent Path')

    # Mark start and end
    ax.scatter([path_x[0]], [path_y[0]], [path_z[0]],
              color='red', s=100, label='Start (Random)')
    ax.scatter([0], [0], [0], color='green', s=100, label='Goal (Minimum)')

    ax.set_xlabel('Weight 1', fontsize=11)
    ax.set_ylabel('Weight 2', fontsize=11)
    ax.set_zlabel('Error', fontsize=11)
    ax.set_title('Gradient Descent: Rolling Down to Find Best Weights',
                fontsize=14, fontweight='bold')

    ax.legend()
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    save_fig('gradient_landscape_3d')

# 14. Learning Curve Smooth
def create_learning_curve_smooth():
    """Show training progress over epochs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Generate smooth learning curves
    epochs = np.arange(1, 101)
    train_loss = 2 * np.exp(-epochs/20) + 0.1 + np.random.normal(0, 0.02, 100)
    val_loss = 2 * np.exp(-epochs/25) + 0.15 + np.random.normal(0, 0.03, 100)
    train_acc = 1 - np.exp(-epochs/15) + np.random.normal(0, 0.01, 100)
    val_acc = 1 - np.exp(-epochs/20) - 0.05 + np.random.normal(0, 0.015, 100)

    # Smooth the curves
    from scipy.ndimage import gaussian_filter1d
    train_loss = gaussian_filter1d(train_loss, sigma=2)
    val_loss = gaussian_filter1d(val_loss, sigma=2)
    train_acc = gaussian_filter1d(train_acc, sigma=2) * 100
    val_acc = gaussian_filter1d(val_acc, sigma=2) * 100

    # Left: Loss curve
    ax1.set_title('Training Progress: Loss', fontsize=12, fontweight='bold')
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, val_loss, 'r--', linewidth=2, label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 2.5)

    # Add annotations
    ax1.annotate('Rapid learning', xy=(10, train_loss[9]), xytext=(20, 1.8),
                arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, alpha=0.7))
    ax1.annotate('Convergence', xy=(80, train_loss[79]), xytext=(60, 0.5),
                arrowprops=dict(arrowstyle='->', color=COLOR_SUCCESS, alpha=0.7))

    # Right: Accuracy curve
    ax2.set_title('Training Progress: Accuracy', fontsize=12, fontweight='bold')
    ax2.plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, val_acc, 'r--', linewidth=2, label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    # Add milestone markers
    milestones = [25, 50, 75, 100]
    for m in milestones:
        ax2.axvline(m, color=COLOR_ACCENT, alpha=0.2, linestyle=':')
        ax2.text(m, 5, f'{int(val_acc[m-1])}%', ha='center', fontsize=9,
                color=COLOR_ACCENT)

    plt.suptitle('From Random Guessing to High Accuracy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('learning_curve_smooth')

# 15. Learned Features Visualization
def create_learned_features_visualization():
    """Show what hidden neurons learn to detect"""
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle('What Hidden Neurons Learn to Detect', fontsize=14, fontweight='bold')

    # Create different feature patterns
    features = [
        # Row 1: Edge detectors
        np.array([[0,0,1,1,1], [0,0,1,1,1], [0,0,1,1,1], [0,0,1,1,1], [0,0,1,1,1]]),  # Vertical
        np.array([[0,0,0,0,0], [0,0,0,0,0], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1]]),  # Horizontal
        np.eye(5),  # Diagonal
        np.fliplr(np.eye(5)),  # Other diagonal
        np.array([[0,0,0,0,0], [0,1,1,1,0], [0,1,0,1,0], [0,1,1,1,0], [0,0,0,0,0]]),  # Box
        # Row 2: Pattern detectors
        np.array([[0,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [0,1,1,1,0]]),  # Circle
        np.array([[0,0,1,0,0], [0,1,0,1,0], [1,0,0,0,1], [1,1,1,1,1], [1,0,0,0,1]]),  # A-like
        np.array([[1,1,1,1,0], [1,0,0,0,1], [1,1,1,1,0], [1,0,0,0,1], [1,1,1,1,0]]),  # B-like
        np.array([[0,0,1,0,0], [0,1,1,1,0], [0,0,1,0,0], [0,0,1,0,0], [0,1,1,1,0]]),  # T-like
        np.array([[1,0,0,0,1], [0,1,0,1,0], [0,0,1,0,0], [0,1,0,1,0], [1,0,0,0,1]]),  # X-like
    ]

    labels = ['Vertical', 'Horizontal', 'Diagonal', 'Anti-diag', 'Box',
              'Curves', 'A-pattern', 'B-pattern', 'T-pattern', 'X-pattern']

    for idx, (ax, feature, label) in enumerate(zip(axes.flat, features, labels)):
        im = ax.imshow(feature, cmap='hot', interpolation='nearest')
        ax.set_title(f'Neuron {idx+1}:\n{label}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    save_fig('learned_features_visualization')

# 16. Overfitting Memorization
def create_overfitting_memorization():
    """Student memorization analogy for overfitting"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Good learning
    ax1.set_title('Good Learning: Understanding', fontsize=12, fontweight='bold',
                 color=COLOR_SUCCESS)
    ax1.text(0.5, 0.85, 'Training Examples', fontsize=11, ha='center', fontweight='bold')

    # Show examples with patterns
    examples = ['2+2=4', '3+3=6', '4+4=8']
    for i, ex in enumerate(examples):
        ax1.text(0.2, 0.7-i*0.1, ex, fontsize=10)

    ax1.text(0.5, 0.35, '↓ Learns Pattern ↓', fontsize=11, ha='center',
            color=COLOR_SUCCESS, fontweight='bold')

    ax1.text(0.5, 0.2, 'Understands: "Add same number twice"',
            fontsize=11, ha='center', style='italic')

    ax1.text(0.5, 0.05, 'New Problem: 7+7 = ?', fontsize=11, ha='center')
    ax1.text(0.7, 0.05, '✓ 14', fontsize=11, color=COLOR_SUCCESS, fontweight='bold')

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # Right: Overfitting
    ax2.set_title('Overfitting: Memorization', fontsize=12, fontweight='bold',
                 color=COLOR_ERROR)
    ax2.text(0.5, 0.85, 'Training Examples', fontsize=11, ha='center', fontweight='bold')

    for i, ex in enumerate(examples):
        ax2.text(0.2, 0.7-i*0.1, ex, fontsize=10)

    ax2.text(0.5, 0.35, '↓ Memorizes Answers ↓', fontsize=11, ha='center',
            color=COLOR_ERROR, fontweight='bold')

    ax2.text(0.5, 0.2, 'Memorized: "2+2=4, 3+3=6, 4+4=8"',
            fontsize=11, ha='center', style='italic')

    ax2.text(0.5, 0.05, 'New Problem: 7+7 = ?', fontsize=11, ha='center')
    ax2.text(0.7, 0.05, '✗ ???', fontsize=11, color=COLOR_ERROR, fontweight='bold')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    plt.suptitle('The Danger of Overfitting', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('overfitting_memorization')

# 17. Scale Comparison
def create_scale_comparison():
    """Compare network sizes over time"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Data
    years = [1957, 1986, 1998, 2012, 2018, 2020, 2023]
    names = ['Perceptron', 'NETtalk', 'LeNet', 'AlexNet', 'BERT', 'GPT-3', 'GPT-4']
    params = [20, 18000, 60000, 60000000, 340000000, 175000000000, 1700000000000]
    colors = ['#FF9999', '#FF7777', '#FF5555', '#FF3333', '#FF0000', '#CC0000', '#990000']

    # Use log scale for parameters
    params_log = np.log10(params)

    # Create bars
    bars = ax.bar(range(len(years)), params_log, color=colors, alpha=0.8)

    # Customize
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([f'{year}\n{name}' for year, name in zip(years, names)],
                       fontsize=10)
    ax.set_ylabel('Parameters (log scale)', fontsize=12)
    ax.set_title('The Exponential Growth of Neural Networks', fontsize=14, fontweight='bold')

    # Add actual values on bars
    for i, (bar, param) in enumerate(zip(bars, params)):
        height = bar.get_height()
        if param < 1000000:
            label = f'{param/1000:.0f}K' if param >= 1000 else str(param)
        elif param < 1000000000:
            label = f'{param/1000000:.0f}M'
        elif param < 1000000000000:
            label = f'{param/1000000000:.0f}B'
        else:
            label = f'{param/1000000000000:.1f}T'

        ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
               label, ha='center', fontsize=10, fontweight='bold')

    # Add annotations
    ax.annotate('Could recognize\nsimple patterns', xy=(0, params_log[0]),
               xytext=(1, 2), arrowprops=dict(arrowstyle='->', alpha=0.5))
    ax.annotate('Deep learning\nrevolution begins', xy=(3, params_log[3]),
               xytext=(2, 9), arrowprops=dict(arrowstyle='->', alpha=0.5))
    ax.annotate('Human-level\ntext generation', xy=(5, params_log[5]),
               xytext=(4, 11), arrowprops=dict(arrowstyle='->', alpha=0.5))

    ax.set_ylim(0, 14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add scale reference
    ax.text(0.02, 0.98, 'Same core principles,\njust different scales!',
           transform=ax.transAxes, fontsize=11, style='italic',
           va='top', bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, alpha=0.8))

    plt.tight_layout()
    save_fig('scale_comparison')

# 18. Convolution Sliding Window
def create_convolution_sliding_window():
    """Visualize convolution as sliding window"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Convolution: Same Detector Everywhere', fontsize=14, fontweight='bold')

    # Create simple image with edge
    image = np.zeros((7, 7))
    image[:, 3:5] = 1  # Vertical edge

    # 3x3 edge detector kernel
    kernel = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

    # Positions to show
    positions = [(0,0), (0,2), (0,4), (2,0), (2,2), (2,4)]

    for idx, (ax, (row, col)) in enumerate(zip(axes.flat, positions)):
        # Show image
        ax.imshow(image, cmap='gray', interpolation='nearest', alpha=0.3)

        # Highlight current window
        rect = patches.Rectangle((col-0.5, row-0.5), 3, 3,
                                linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # Show kernel position
        ax.set_title(f'Position ({row},{col})', fontsize=10)

        # Calculate and show output
        if col == 2:  # Near edge
            output = 3
            color = COLOR_SUCCESS
        else:
            output = 0
            color = COLOR_MAIN

        ax.text(col+1, row+1, f'{output}', fontsize=14, ha='center', va='center',
               color=color, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(6.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    save_fig('convolution_sliding_window')

# 19. Network Evolution Timeline
def create_network_evolution_timeline():
    """Timeline showing network evolution"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Timeline data
    events = [
        (1943, 'McCulloch-Pitts', 'First neuron model', 'bottom'),
        (1957, 'Perceptron', 'First learning', 'top'),
        (1969, 'XOR Crisis', 'Limitations found', 'bottom'),
        (1986, 'Backprop', 'Multi-layer learning', 'top'),
        (1998, 'LeNet', 'Digit recognition', 'bottom'),
        (2012, 'AlexNet', 'Deep learning boom', 'top'),
        (2017, 'Transformer', 'Attention revolution', 'bottom'),
        (2020, 'GPT-3', '175B parameters', 'top'),
        (2023, 'ChatGPT', 'AI goes mainstream', 'bottom')
    ]

    # Draw timeline
    ax.axhline(y=0.5, color=COLOR_MAIN, linewidth=2, alpha=0.5)

    for year, name, desc, pos in events:
        x = (year - 1940) / 85  # Normalize to [0,1]
        y = 0.65 if pos == 'top' else 0.35

        # Event marker
        ax.scatter(x, 0.5, s=100, color=COLOR_HIGHLIGHT, zorder=5, edgecolor='white', linewidth=2)

        # Event box
        box_y = y + (0.05 if pos == 'top' else -0.05)
        rect = FancyBboxPatch((x-0.06, box_y-0.04), 0.12, 0.08,
                              boxstyle="round,pad=0.01",
                              facecolor=COLOR_LIGHT,
                              edgecolor=COLOR_MAIN,
                              linewidth=1)
        ax.add_patch(rect)

        # Text
        ax.text(x, box_y, f'{year}\n{name}', ha='center', va='center',
               fontsize=9, fontweight='bold')
        ax.text(x, box_y-0.08 if pos == 'top' else box_y+0.08,
               desc, ha='center', va='center' if pos == 'top' else 'center',
               fontsize=8, color=COLOR_ACCENT, style='italic')

        # Connection line
        ax.plot([x, x], [0.5, box_y], 'k--', alpha=0.3, linewidth=1)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.1, 0.9)
    ax.set_title('70 Years of Neural Network Evolution', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Era labels
    ax.text(0.15, 0.05, 'Early Theory', fontsize=10, ha='center', color=COLOR_ACCENT)
    ax.text(0.4, 0.05, 'AI Winter', fontsize=10, ha='center', color=COLOR_ACCENT)
    ax.text(0.65, 0.05, 'Deep Learning Era', fontsize=10, ha='center', color=COLOR_ACCENT)
    ax.text(0.9, 0.05, 'Scale Revolution', fontsize=10, ha='center', color=COLOR_ACCENT)

    plt.tight_layout()
    save_fig('network_evolution_timeline')

# Generate all figures
def generate_all_figures():
    """Generate all figures for the presentation"""
    print("Generating BSc Neural Network Primer figures...")

    figure_generators = [
        ('handwriting_variations', create_handwriting_variations),
        ('pixel_to_numbers', create_pixel_to_numbers),
        ('rules_vs_learning', create_rules_vs_learning),
        ('neuron_as_voter', create_neuron_as_voter),
        ('weight_adjustment_visual', create_weight_adjustment_visual),
        ('training_steps_numbered', create_training_steps_numbered),
        ('xor_impossible_line', create_xor_impossible_line),
        ('xor_two_questions', create_xor_two_questions),
        ('hidden_layer_factory', create_hidden_layer_factory),
        ('feature_hierarchy', create_feature_hierarchy),
        ('blame_distribution', create_blame_distribution),
        ('error_flow_backward', create_error_flow_backward),
        ('gradient_landscape_3d', create_gradient_landscape_3d),
        ('learning_curve_smooth', create_learning_curve_smooth),
        ('learned_features_visualization', create_learned_features_visualization),
        ('overfitting_memorization', create_overfitting_memorization),
        ('scale_comparison', create_scale_comparison),
        ('convolution_sliding_window', create_convolution_sliding_window),
        ('network_evolution_timeline', create_network_evolution_timeline)
    ]

    for name, generator in figure_generators:
        try:
            print(f"  Creating {name}...")
            generator()
        except Exception as e:
            print(f"  Error creating {name}: {e}")

    print(f"\nGenerated {len(figure_generators)} figures in {output_dir}")

if __name__ == "__main__":
    generate_all_figures()