"""
Generate 12 completely new conceptual charts for Week 5: Speed Revolution
Focus on visual metaphors and engaging visualizations
Created: 2025-09-28
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow, Wedge
import matplotlib.lines as mlines
import seaborn as sns
import os

os.makedirs('../figures', exist_ok=True)

# Dynamic color gradients for each act
# Act 1: Frustration (Red/Orange)
act1_primary = (214/255, 39/255, 40/255)  # Red
act1_secondary = (255/255, 127/255, 14/255)  # Orange
act1_accent = (255/255, 200/255, 87/255)  # Light orange

# Act 2: Disappointment (Blue/Gray)
act2_primary = (31/255, 119/255, 180/255)  # Blue
act2_secondary = (150/255, 150/255, 150/255)  # Gray
act2_accent = (200/255, 200/255, 200/255)  # Light gray

# Act 3: Breakthrough (Green/Teal)
act3_primary = (44/255, 160/255, 44/255)  # Green
act3_secondary = (23/255, 190/255, 207/255)  # Teal
act3_accent = (144/255, 238/255, 144/255)  # Light green

# Act 4: Triumph (Gold/Purple)
act4_primary = (255/255, 215/255, 0/255)  # Gold
act4_secondary = (148/255, 0/255, 211/255)  # Purple
act4_accent = (218/255, 165/255, 32/255)  # Goldenrod

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

def save_fig(filename):
    plt.savefig(f'../figures/{filename}', dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Generated: {filename}')
    plt.close()

print("Generating Week 5 Conceptual Charts...")
print("="*60)

# ============ ACT 1: THE WAITING GAME ============
print("\n=== Act 1: The Waiting Game ===")

# Chart 1: The Domino Effect
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# RNN: Sequential dominos
ax1.set_title('RNN: Sequential Processing', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 5)
ax1.axis('off')

# Draw dominos falling in sequence
for i in range(8):
    angle = min(80 - i*10, 80) if i < 4 else 0
    if i < 4:
        color = act1_primary
        alpha = 1 - i*0.15
    else:
        color = act1_secondary
        alpha = 0.3

    # Draw domino as rotated rectangle
    rect = Rectangle((i*1.2 + 1, 1), 0.3, 2,
                     angle=angle, color=color, alpha=alpha,
                     edgecolor='black', linewidth=2)
    ax1.add_patch(rect)

    if i < 7:
        # Draw arrow showing sequential dependency
        ax1.annotate('', xy=(i*1.2 + 1.5, 2), xytext=(i*1.2 + 1.3, 2),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7))

ax1.text(5, 4, 'Each token waits for the previous one',
         ha='center', fontsize=10, style='italic', color='darkred')

# Transformer: All blocks process simultaneously
ax2.set_title('Transformer: Parallel Processing', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 5)
ax2.axis('off')

# Draw all dominos standing and glowing
for i in range(8):
    # Draw standing domino
    rect = Rectangle((i*1.2 + 1, 1), 0.3, 2,
                     color=act3_primary, alpha=0.9,
                     edgecolor='black', linewidth=2)
    ax2.add_patch(rect)

    # Add glow effect
    for j in range(3):
        glow = Rectangle((i*1.2 + 1 - 0.05*j, 1 - 0.05*j),
                        0.3 + 0.1*j, 2 + 0.1*j,
                        color=act3_accent, alpha=0.2, zorder=-1)
        ax2.add_patch(glow)

ax2.text(5, 4, 'All tokens process simultaneously!',
         ha='center', fontsize=10, style='italic', color='darkgreen', fontweight='bold')

plt.suptitle('Chart 1: The Domino Effect', fontsize=14, fontweight='bold', y=1.02)
save_fig('new_01_domino_effect.pdf')

# Chart 2: Traffic Jam Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# RNN: Single lane traffic jam
ax1.set_title('RNN: Single Lane Highway (Sequential Bottleneck)', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 3)
ax1.axis('off')

# Draw road
road_color = (0.3, 0.3, 0.3)
ax1.add_patch(Rectangle((0, 1), 10, 1, color=road_color))

# Draw lane markings
for i in range(0, 10, 1):
    ax1.add_patch(Rectangle((i, 1.45), 0.5, 0.1, color='white'))

# Draw cars (tokens) in traffic jam
car_positions = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
car_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(car_positions)))

for i, (pos, color) in enumerate(zip(car_positions, car_colors)):
    # Car body
    ax1.add_patch(Rectangle((pos, 1.3), 0.8, 0.4, color=color, edgecolor='black'))
    # Windshield
    ax1.add_patch(Rectangle((pos + 0.2, 1.35), 0.2, 0.3, color='lightblue', alpha=0.5))
    # Token label
    ax1.text(pos + 0.4, 1.5, f'T{i+1}', ha='center', va='center', fontsize=8, fontweight='bold')

ax1.text(5, 2.5, 'Processing Speed: 1 token/cycle', ha='center', fontsize=10, color='darkred')
ax1.text(0.5, 0.5, 'BLOCKED', ha='left', fontsize=9, color='red', fontweight='bold')

# Transformer: Multi-lane highway
ax2.set_title('Transformer: 8-Lane Highway (Parallel Freedom)', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 5)
ax2.axis('off')

# Draw 8 lanes
for lane in range(4):
    ax2.add_patch(Rectangle((0, lane*0.8 + 0.5), 10, 0.7, color=road_color))
    # Lane markings
    for i in range(0, 10, 1):
        ax2.add_patch(Rectangle((i, lane*0.8 + 0.85), 0.5, 0.08, color='white'))

# Draw cars distributed across lanes (free flowing)
car_data = [
    (1, 0, 'T1'), (3, 1, 'T2'), (5, 2, 'T3'), (7, 3, 'T4'),
    (2, 2, 'T5'), (4, 0, 'T6'), (6, 1, 'T7'), (8, 3, 'T8')
]

for x, lane, label in car_data:
    y = lane * 0.8 + 0.65
    color = act3_primary
    # Car body
    ax2.add_patch(Rectangle((x, y), 0.8, 0.35, color=color, edgecolor='black'))
    # Windshield
    ax2.add_patch(Rectangle((x + 0.2, y + 0.05), 0.2, 0.25, color='lightblue', alpha=0.5))
    # Token label
    ax2.text(x + 0.4, y + 0.175, label, ha='center', va='center', fontsize=8, fontweight='bold')

    # Speed lines
    for j in range(3):
        ax2.plot([x - 0.2 - j*0.1, x - 0.1 - j*0.1],
                [y + 0.175, y + 0.175],
                color=act3_accent, linewidth=2, alpha=0.5 - j*0.15)

ax2.text(5, 4.2, 'Processing Speed: 8 tokens/cycle', ha='center', fontsize=10,
         color='darkgreen', fontweight='bold')

save_fig('new_02_traffic_visualization.pdf')

# Chart 3: The Assembly Line Problem
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# RNN: Traditional assembly line
ax1.set_title('RNN: Single Worker Assembly Line', fontsize=12, fontweight='bold')
ax1.set_xlim(-0.5, 10)
ax1.set_ylim(0, 4)
ax1.axis('off')

# Draw conveyor belt
ax1.add_patch(Rectangle((0, 1), 9, 0.3, color='gray', edgecolor='black'))
ax1.add_patch(Rectangle((0, 2.5), 9, 0.3, color='gray', edgecolor='black'))

# Draw single worker
worker_x = 4.5
ax1.add_patch(Circle((worker_x, 2), 0.3, color=act1_primary, edgecolor='black', linewidth=2))
ax1.text(worker_x, 2, '👷', ha='center', va='center', fontsize=20)

# Draw products on belt
products = ['The', 'cat', 'sat', 'on', 'the', 'mat']
for i, product in enumerate(products):
    x = i * 1.5
    if x < worker_x - 0.5:
        color = act1_secondary
        status = '✓'
    elif abs(x - worker_x) < 0.5:
        color = act1_primary
        status = '⚙'
    else:
        color = act1_accent
        status = '⏳'

    ax1.add_patch(Rectangle((x, 1.3), 0.8, 0.7, color=color, edgecolor='black', alpha=0.7))
    ax1.text(x + 0.4, 1.65, product, ha='center', va='center', fontsize=9, fontweight='bold')
    ax1.text(x + 0.4, 1.1, status, ha='center', va='center', fontsize=8)

ax1.text(4.5, 3.5, '1 token processed at a time', ha='center', fontsize=10,
         style='italic', color='darkred')

# Transformer: Parallel factory
ax2.set_title('Transformer: Parallel Processing Factory', fontsize=12, fontweight='bold')
ax2.set_xlim(-0.5, 10)
ax2.set_ylim(0, 4)
ax2.axis('off')

# Draw multiple workstations
for i in range(6):
    x = i * 1.5 + 0.5
    # Worker
    ax2.add_patch(Circle((x, 2), 0.3, color=act3_primary, edgecolor='black', linewidth=2))
    ax2.text(x, 2, '👷', ha='center', va='center', fontsize=16)

    # Product
    product = products[i]
    ax2.add_patch(Rectangle((x - 0.4, 1), 0.8, 0.7, color=act3_secondary,
                            edgecolor='black', alpha=0.9))
    ax2.text(x, 1.35, product, ha='center', va='center', fontsize=9, fontweight='bold')
    ax2.text(x, 0.8, '⚙', ha='center', va='center', fontsize=10)

    # Lightning bolt to show simultaneous processing
    ax2.plot([x, x], [1.7, 2.5], color='yellow', linewidth=3,
            linestyle='-', marker='v', markersize=8, markeredgecolor='orange')

ax2.text(4.5, 3.5, 'All tokens processed simultaneously!', ha='center', fontsize=10,
         style='italic', color='darkgreen', fontweight='bold')

save_fig('new_03_assembly_line.pdf')

# ============ ACT 2: THE DISAPPOINTMENT ============
print("\n=== Act 2: The Disappointment ===")

# Chart 4: Memory Maze
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title('Chart 4: The Memory Maze - Information Gets Lost', fontsize=14, fontweight='bold')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Create maze-like structure
np.random.seed(42)
maze_walls = []

# Horizontal walls
for i in range(1, 9):
    for j in range(0, 10, 2):
        if np.random.random() > 0.3:
            ax.plot([j, j+1], [i, i], color=act2_secondary, linewidth=2)

# Vertical walls
for i in range(1, 10):
    for j in range(1, 9):
        if np.random.random() > 0.3:
            ax.plot([i, i], [j, j+1], color=act2_secondary, linewidth=2)

# Draw path from start to end with fading
path_x = [0.5, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 9.5]
path_y = [0.5, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9.5]

for i in range(len(path_x)-1):
    alpha = 1.0 - (i / len(path_x)) * 0.8
    color = (act2_primary[0], act2_primary[1], act2_primary[2], alpha)
    ax.plot(path_x[i:i+2], path_y[i:i+2], color=color, linewidth=3)

# Start and end markers
ax.add_patch(Circle((0.5, 0.5), 0.3, color='green', edgecolor='black', linewidth=2))
ax.text(0.5, 0.5, 'Start', ha='center', va='center', fontsize=10, fontweight='bold')

ax.add_patch(Circle((9.5, 9.5), 0.3, color=act2_primary, alpha=0.3, edgecolor='black', linewidth=2))
ax.text(9.5, 9.5, 'End?', ha='center', va='center', fontsize=10, alpha=0.5)

# Add gradient overlay to show information decay
for i in range(10):
    alpha = i / 20
    ax.add_patch(Rectangle((i, 0), 1, 10, color='white', alpha=alpha, zorder=10))

ax.text(5, -0.5, 'Information degrades through sequential processing',
        ha='center', fontsize=11, style='italic', color=act2_primary)

save_fig('new_04_memory_maze.pdf')

# Chart 5: The Broken Telegraph
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title('Chart 5: The Broken Telegraph - Message Degradation', fontsize=14, fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis('off')

# Original message and degradation stages
messages = [
    "The quick brown fox jumps",
    "The quick brown fox jump",
    "The quik bron fox jump",
    "Te qik bro fx jmp",
    "T q b f j",
    "?????"
]

positions = [(1, 2), (3, 2), (5, 2), (7, 2), (9, 2), (11, 2)]
colors = plt.cm.Blues_r(np.linspace(0.2, 0.8, len(messages)))

for i, ((x, y), msg, color) in enumerate(zip(positions, messages, colors)):
    # Draw person icon
    ax.add_patch(Circle((x, y+0.5), 0.2, color=color, edgecolor='black'))
    ax.text(x, y+0.5, '🗣️', ha='center', va='center', fontsize=16)

    # Draw speech bubble
    bubble_width = len(msg) * 0.08 + 0.5
    bubble = FancyBboxPatch((x - bubble_width/2, y - 0.5), bubble_width, 0.4,
                            boxstyle="round,pad=0.1",
                            facecolor='white', edgecolor=color, linewidth=2)
    ax.add_patch(bubble)

    # Add text with increasing corruption
    if i < 4:
        ax.text(x, y - 0.3, msg, ha='center', va='center', fontsize=9,
                alpha=1 - i*0.15)
    else:
        ax.text(x, y - 0.3, msg, ha='center', va='center', fontsize=9,
                color='red', alpha=0.5)

    # Draw arrow to next
    if i < len(messages) - 1:
        ax.annotate('', xy=(x+0.8, y), xytext=(x+0.4, y),
                   arrowprops=dict(arrowstyle='->', color=act2_primary,
                                  lw=2, alpha=0.7 - i*0.1))
        # Add noise symbols
        ax.text(x+0.6, y+0.2, '⚡', fontsize=8, color='red', alpha=0.3 + i*0.1)

ax.text(6, 0.5, 'Sequential processing accumulates errors',
        ha='center', fontsize=11, style='italic', color=act2_primary)

save_fig('new_05_broken_telegraph.pdf')

# Chart 6: Computational Quicksand
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title('Chart 6: Computational Quicksand - Vanishing Gradients', fontsize=14, fontweight='bold')
ax.set_xlim(0, 10)
ax.set_ylim(-8, 2)
ax.axis('off')

# Draw quicksand layers
layers = 10
for i in range(layers):
    y = -i * 0.8
    color = (*act2_secondary, 0.2 + i*0.08)
    width = 8 - i*0.3
    ax.add_patch(Rectangle((5 - width/2, y - 0.4), width, 0.8,
                           color=color, edgecolor='black', linewidth=1))

    # Layer label
    ax.text(0.5, y, f'Layer {i+1}', fontsize=9, va='center')

    # Gradient magnitude
    gradient_mag = np.exp(-i*0.5)
    bar_width = gradient_mag * 3
    ax.add_patch(Rectangle((9 - bar_width, y - 0.2), bar_width, 0.4,
                           color=act2_primary, alpha=0.7))
    ax.text(9.5, y, f'{gradient_mag:.3f}', fontsize=8, va='center')

# Draw sinking figure
for i in range(5):
    y = -i * 1.5
    size = 0.3 - i*0.05
    alpha = 1 - i*0.15
    ax.add_patch(Circle((5, y), size, color='darkblue', alpha=alpha))
    ax.text(5, y, '📉', ha='center', va='center', fontsize=16 - i*2)

ax.text(5, 1.5, 'The deeper you go, the weaker the signal',
        ha='center', fontsize=11, fontweight='bold', color=act2_primary)
ax.text(9.5, 1, 'Gradient\nMagnitude', ha='center', fontsize=9, fontweight='bold')

save_fig('new_06_computational_quicksand.pdf')

# ============ ACT 3: THE BREAKTHROUGH ============
print("\n=== Act 3: The Breakthrough ===")

# Chart 7: The Attention Spotlight Theatre
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title('Chart 7: The Attention Spotlight Theatre', fontsize=14, fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.set_facecolor((0.1, 0.1, 0.1))  # Dark background

# Draw stage
stage = Rectangle((1, 1), 10, 2, color=(0.4, 0.3, 0.2), edgecolor='goldenrod', linewidth=3)
ax.add_patch(stage)

# Draw actors (tokens) on stage
tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
token_positions = [(2, 2), (4, 2), (5.5, 2), (7, 2), (8.5, 2), (10, 2)]

for (x, y), token in zip(token_positions, tokens):
    ax.add_patch(Circle((x, y), 0.3, color='white', edgecolor='black', linewidth=2))
    ax.text(x, y, token, ha='center', va='center', fontsize=10, fontweight='bold')

# Draw attention spotlights from different heads
spotlight_sources = [(3, 7), (6, 7), (9, 7)]
spotlight_targets = [
    [(2, 2), (4, 2)],  # Head 1 focuses on subject
    [(4, 2), (5.5, 2)],  # Head 2 focuses on verb
    [(8.5, 2), (10, 2)]  # Head 3 focuses on object
]
spotlight_colors = [act3_accent, 'yellow', 'cyan']

for (sx, sy), targets, color in zip(spotlight_sources, spotlight_targets, spotlight_colors):
    # Draw spotlight source
    ax.add_patch(Circle((sx, sy), 0.2, color=color, edgecolor='black'))
    ax.text(sx, sy + 0.5, f'Head', ha='center', fontsize=9, color='white')

    for (tx, ty) in targets:
        # Draw cone of light
        vertices = [(sx, sy), (tx - 0.5, ty - 0.5), (tx + 0.5, ty - 0.5)]
        triangle = patches.Polygon(vertices, closed=True,
                                  facecolor=color, alpha=0.3,
                                  edgecolor=color, linewidth=1)
        ax.add_patch(triangle)

# Add labels
ax.text(6, 0.3, 'Multi-Head Attention: Different heads focus on different relationships',
        ha='center', fontsize=11, color='white', style='italic')

ax.axis('off')
save_fig('new_07_attention_theatre.pdf')

# Chart 8: The Neural Network Circuit Board
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# RNN: Serial circuit
ax1.set_title('RNN: Serial Circuit', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
ax1.set_facecolor((0.05, 0.05, 0.15))
ax1.axis('off')

# Draw serial connection
nodes = [(1, 3), (3, 3), (5, 3), (7, 3), (9, 3)]
for i, (x, y) in enumerate(nodes):
    # Draw node
    ax1.add_patch(Circle((x, y), 0.3, color='gold', edgecolor='yellow', linewidth=2))
    ax1.text(x, y, f'T{i+1}', ha='center', va='center', fontsize=9, fontweight='bold')

    # Draw connection to next
    if i < len(nodes) - 1:
        # Single wire connection
        ax1.plot([x+0.3, nodes[i+1][0]-0.3], [y, y],
                color='red', linewidth=3, alpha=0.8)
        # Add resistance symbol
        ax1.text((x + nodes[i+1][0])/2, y+0.3, '⚡',
                fontsize=10, color='orange', ha='center')

ax1.text(5, 1, 'Information must pass through each node sequentially',
        ha='center', fontsize=10, color='white', style='italic')

# Transformer: Parallel circuit
ax2.set_title('Transformer: Parallel Circuit', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)
ax2.set_facecolor((0.05, 0.05, 0.15))
ax2.axis('off')

# Draw parallel connections
center = (5, 3)
radius = 2
num_nodes = 6
angles = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)

node_positions = [(center[0] + radius*np.cos(a),
                  center[1] + radius*np.sin(a)) for a in angles]

# Draw all connections first (behind nodes)
for i, (x1, y1) in enumerate(node_positions):
    for j, (x2, y2) in enumerate(node_positions):
        if i != j:
            alpha = 0.3 if abs(i-j) > 1 else 0.6
            ax2.plot([x1, x2], [y1, y2], color=act3_secondary,
                    linewidth=1, alpha=alpha)

# Draw nodes
for i, (x, y) in enumerate(node_positions):
    ax2.add_patch(Circle((x, y), 0.3, color=act3_primary,
                         edgecolor='lime', linewidth=2))
    ax2.text(x, y, f'T{i+1}', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')

ax2.text(5, 0.5, 'Direct connections enable parallel processing',
        ha='center', fontsize=10, color='white', style='italic')

save_fig('new_08_circuit_board.pdf')

# Chart 9: The Parallel Universe Portal
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Sequential Universe
ax1.set_title('Sequential Universe: Time = O(n)', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_facecolor((0.9, 0.9, 0.95))
ax1.axis('off')

# Draw time spiral
t = np.linspace(0, 4*np.pi, 100)
x_spiral = 5 + t * np.cos(t) / 3
y_spiral = 5 + t * np.sin(t) / 3

ax1.plot(x_spiral, y_spiral, color=act1_primary, linewidth=2, alpha=0.7)

# Add clock positions
clock_positions = np.linspace(0, len(x_spiral)-1, 8, dtype=int)
for i, idx in enumerate(clock_positions):
    ax1.add_patch(Circle((x_spiral[idx], y_spiral[idx]), 0.2,
                         color=act1_secondary, edgecolor='black'))
    ax1.text(x_spiral[idx], y_spiral[idx], f'{i}s',
            ha='center', va='center', fontsize=8, fontweight='bold')

ax1.text(5, 1, 'Processing time grows linearly',
        ha='center', fontsize=10, color=act1_primary, fontweight='bold')

# Parallel Universe
ax2.set_title('Parallel Universe: Time = O(1)', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_facecolor((0.9, 0.95, 0.9))
ax2.axis('off')

# Draw portal effect
for r in np.linspace(0.5, 3, 10):
    alpha = 0.5 - r/6
    ax2.add_patch(Circle((5, 5), r, color=act3_secondary,
                         alpha=alpha, edgecolor=act3_primary, linewidth=1))

# Draw instant connections
num_points = 8
angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
for i, angle in enumerate(angles):
    x = 5 + 2.5 * np.cos(angle)
    y = 5 + 2.5 * np.sin(angle)

    # Draw lightning to center (instant connection)
    ax2.plot([5, x], [5, y], color='yellow', linewidth=3, alpha=0.8)

    # Draw node
    ax2.add_patch(Circle((x, y), 0.3, color=act3_primary, edgecolor='black', linewidth=2))
    ax2.text(x, y, '0s', ha='center', va='center', fontsize=8, fontweight='bold', color='white')

ax2.text(5, 1, 'All processing happens instantly!',
        ha='center', fontsize=10, color=act3_primary, fontweight='bold')

save_fig('new_09_parallel_universe.pdf')

# ============ ACT 4: THE IMPACT ============
print("\n=== Act 4: The Impact ===")

# Chart 10: The Language Galaxy
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title('Chart 10: The Language Galaxy - Universal Understanding',
            fontsize=14, fontweight='bold')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_facecolor((0.02, 0.02, 0.08))
ax.axis('off')

# Create language star systems
languages = {
    'English': (5, 5),
    'Chinese': (3, 7),
    'Spanish': (7, 7),
    'Hindi': (2, 3),
    'Arabic': (8, 3),
    'French': (3, 4),
    'Russian': (7, 4),
    'Japanese': (6, 8),
    'German': (4, 2),
    'Korean': (6, 6)
}

# Draw attention bridges between languages
for lang1, pos1 in languages.items():
    for lang2, pos2 in languages.items():
        if lang1 < lang2:  # Avoid duplicate connections
            distance = np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
            if distance < 3:  # Only nearby connections
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                       color=act4_secondary, linewidth=0.5, alpha=0.3)

# Draw language stars
for lang, (x, y) in languages.items():
    # Star glow
    for r in [0.4, 0.3, 0.2]:
        ax.add_patch(Circle((x, y), r, color=act4_primary, alpha=0.3-r/2))

    # Star core
    ax.add_patch(Circle((x, y), 0.15, color='white', edgecolor=act4_primary, linewidth=2))
    ax.text(x, y-0.5, lang, ha='center', va='top', fontsize=8,
           color='white', fontweight='bold')

# Add constellation lines for major language families
ax.plot([5, 7, 7], [5, 7, 4], color=act4_accent, linewidth=1, alpha=0.5)  # Germanic
ax.plot([3, 3, 2], [7, 4, 3], color=act4_accent, linewidth=1, alpha=0.5)  # Indo-European

ax.text(5, 0.5, 'Transformers bridge all languages through universal attention',
        ha='center', fontsize=11, color='white', style='italic')

save_fig('new_10_language_galaxy.pdf')

# Chart 11: The AI Evolution Tree
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_title('Chart 11: The AI Evolution Tree', fontsize=14, fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Draw trunk
trunk_x = 6
ax.plot([trunk_x, trunk_x], [0, 3], color='brown', linewidth=20)
ax.text(trunk_x, 0.5, 'Transformer\n2017', ha='center', va='center',
        fontsize=10, fontweight='bold', color='white')

# Define branches (model families)
branches = [
    # (start_y, end_x, end_y, model, size, color)
    (3, 3, 5, 'BERT\n2018', 100, act4_primary),
    (3, 4, 6, 'RoBERTa\n2019', 125, act4_primary),
    (3, 2, 7, 'ELECTRA\n2020', 85, act4_primary),
    (3, 9, 5, 'GPT-2\n2019', 117, act4_secondary),
    (3, 10, 7, 'GPT-3\n2020', 175, act4_secondary),
    (3, 11, 9, 'GPT-4\n2023', 200, act4_secondary),
    (3, 6, 5, 'T5\n2019', 110, act3_primary),
    (3, 7, 6, 'FLAN-T5\n2022', 130, act3_primary),
    (3, 5, 8, 'PaLM\n2022', 180, act3_secondary),
    (3, 8, 8, 'LaMDA\n2022', 137, act3_secondary),
]

for start_y, end_x, end_y, model, size, color in branches:
    # Draw branch
    ax.plot([trunk_x, end_x], [start_y, end_y], color='brown',
           linewidth=10 * (size/200), alpha=0.7)

    # Draw leaf/flower (model)
    circle_size = size / 500
    ax.add_patch(Circle((end_x, end_y), circle_size, color=color,
                        edgecolor='black', linewidth=2, alpha=0.8))
    ax.text(end_x, end_y, model, ha='center', va='center',
           fontsize=8, fontweight='bold')

    # Add size label
    ax.text(end_x, end_y - circle_size - 0.2, f'{size}B',
           ha='center', fontsize=7, color='gray')

ax.text(6, 9.5, 'The Transformer spawned an entire ecosystem',
        ha='center', fontsize=12, fontweight='bold', color=act4_secondary)

save_fig('new_11_evolution_tree.pdf')

# Chart 12: The Scaling Rocket
fig, ax = plt.subplots(figsize=(10, 12))
ax.set_title('Chart 12: The Scaling Rocket - Exponential Growth', fontsize=14, fontweight='bold')
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.set_facecolor((0.9, 0.95, 1.0))
ax.axis('off')

# Draw rocket trajectory
t = np.linspace(0, 10, 100)
x_rocket = 5 + np.sin(t/3)
y_rocket = t

ax.plot(x_rocket, y_rocket, color='red', linewidth=3, linestyle='--', alpha=0.5)

# Rocket stages (model sizes over time)
stages = [
    (5, 1, '2017\nOriginal\n65M', 0.5, act1_primary),
    (5, 3, '2018\nBERT\n340M', 0.6, act2_primary),
    (5, 5, '2019\nGPT-2\n1.5B', 0.7, act3_primary),
    (5, 7, '2020\nGPT-3\n175B', 0.9, act4_secondary),
    (5, 9, '2023\nGPT-4\n1.7T', 1.1, act4_primary),
    (5, 11, '2024\n???\n10T+', 1.3, 'gold'),
]

for x, y, label, size, color in stages:
    # Draw rocket stage
    rocket_body = Rectangle((x - size/2, y - 0.3), size, 0.6,
                           color=color, edgecolor='black', linewidth=2)
    ax.add_patch(rocket_body)

    # Draw flames
    if y < 11:
        flame_sizes = np.random.uniform(0.1, 0.3, 5)
        for i, fs in enumerate(flame_sizes):
            flame_x = x + (i-2) * 0.1
            ax.add_patch(patches.Polygon([(flame_x-fs/2, y-0.3),
                                         (flame_x+fs/2, y-0.3),
                                         (flame_x, y-0.3-fs*2)],
                                        color='orange', alpha=0.7))

    # Add label
    ax.text(x + size/2 + 0.5, y, label, ha='left', va='center',
           fontsize=9, fontweight='bold')

# Add stars in background
np.random.seed(42)
stars_x = np.random.uniform(0, 10, 50)
stars_y = np.random.uniform(0, 12, 50)
ax.scatter(stars_x, stars_y, s=2, color='gray', alpha=0.5)

# Add scale on the side
ax.text(1, 6, 'Parameter\nCount', ha='center', va='center',
        fontsize=10, rotation=90, fontweight='bold')

for y, size_label in [(1, '10M'), (3, '100M'), (5, '1B'), (7, '100B'), (9, '1T'), (11, '10T')]:
    ax.text(0.5, y, size_label, ha='right', va='center', fontsize=8)
    ax.plot([0.6, 0.8], [y, y], color='gray', linewidth=1)

ax.text(5, -0.5, 'Model scale growing exponentially!',
        ha='center', fontsize=12, fontweight='bold', color=act4_secondary)

save_fig('new_12_scaling_rocket.pdf')

print("\n" + "="*60)
print("All 12 conceptual charts generated successfully!")
print("Charts use visual metaphors for better pedagogical impact")
print("="*60)