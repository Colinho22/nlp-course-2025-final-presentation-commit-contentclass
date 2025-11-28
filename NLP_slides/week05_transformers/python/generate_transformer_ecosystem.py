"""
Generate Transformer Application Ecosystem Diagram
Week 5 Transformers - For Slide 25 (2024 Landscape)
Shows transformer applications across 4 domains with model names
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import numpy as np

# Template colors
COLOR_MAIN = '#333366'
COLOR_BLUE = '#4472C4'
COLOR_PURPLE = '#8B5A9B'
COLOR_GREEN = '#44A044'
COLOR_ORANGE = '#FF7F0E'
COLOR_LIGHT_BG = '#F0F0F0'

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Transformer Architecture: Universal Foundation (2024)', ha='center', va='center',
        fontsize=14, fontweight='bold', color=COLOR_MAIN)

# Center: Transformer Architecture
center_x, center_y = 5, 5
center_circle = Circle((center_x, center_y), 0.8, facecolor=COLOR_PURPLE,
                       edgecolor=COLOR_MAIN, linewidth=3, alpha=0.6)
ax.add_patch(center_circle)
ax.text(center_x, center_y+0.2, 'Transformer', ha='center', va='center',
        fontsize=12, fontweight='bold', color='white')
ax.text(center_x, center_y-0.2, 'Architecture', ha='center', va='center',
        fontsize=10, fontweight='bold', color='white')

# Four quadrants
quadrants = [
    # (x, y, title, color, models)
    (2, 7.5, 'Language\n(NLP)', COLOR_BLUE,
     [('GPT-4', 1.7, 0.7), ('ChatGPT', 175, 0.5), ('Claude', 200, 0.45), ('LLaMA', 65, 0.3)]),

    (8, 7.5, 'Vision\n(Images)', COLOR_GREEN,
     [('ViT', 86, 0.4), ('DALL-E 3', 100, 0.5), ('Midjourney', 50, 0.35), ('SAM', 600, 0.45)]),

    (2, 2.5, 'Audio\n(Speech/Music)', COLOR_ORANGE,
     [('Whisper', 1.5, 0.5), ('MusicGen', 1.5, 0.35), ('AudioLM', 1, 0.3), ('Vall-E', 0.8, 0.25)]),

    (8, 2.5, 'Multimodal\n(Combined)', COLOR_PURPLE,
     [('GPT-4V', 1.7, 0.6), ('Gemini', 1.5, 0.5), ('CLIP', 0.4, 0.35), ('Flamingo', 80, 0.4)])
]

for quad_x, quad_y, title, color, models in quadrants:
    # Quadrant title box
    title_box = FancyBboxPatch((quad_x-1.2, quad_y+0.6), 2.4, 0.7,
                               boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor=COLOR_MAIN,
                               linewidth=2, alpha=0.3)
    ax.add_patch(title_box)
    ax.text(quad_x, quad_y+0.95, title, ha='center', va='center',
            fontsize=11, fontweight='bold', color=color)

    # Model bubbles
    for i, (model_name, params, radius) in enumerate(models):
        # Position bubbles in a small cluster
        angle = (i / len(models)) * 2 * np.pi
        bubble_x = quad_x + 0.6 * np.cos(angle)
        bubble_y = quad_y + 0.3 * np.sin(angle) - 0.4

        # Bubble (size proportional to parameters)
        bubble = Circle((bubble_x, bubble_y), radius * 0.5, facecolor=color,
                       edgecolor=COLOR_MAIN, linewidth=1.5, alpha=0.5)
        ax.add_patch(bubble)

        # Model name
        ax.text(bubble_x, bubble_y, model_name, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')

        # Parameter count (if significant)
        if params >= 1:
            if params >= 1000:
                param_str = f'{params/1000:.1f}T'
            else:
                param_str = f'{int(params)}B'
            ax.text(bubble_x, bubble_y-radius*0.5-0.15, param_str, ha='center', va='center',
                    fontsize=7, color=color)

    # Arrow from center to quadrant
    arrow = FancyArrowPatch((center_x, center_y), (quad_x, quad_y+0.3),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2.5, color=color, alpha=0.6)
    ax.add_patch(arrow)

# Key insight box at bottom
insight_box = FancyBboxPatch((1, 0.3), 8, 0.9,
                             boxstyle="round,pad=0.1",
                             facecolor=COLOR_LIGHT_BG, edgecolor=COLOR_MAIN, linewidth=2)
ax.add_patch(insight_box)
ax.text(5, 0.85, 'Same architecture, different data → Universal foundation for AI', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_MAIN)
ax.text(5, 0.5, '2017 insight: Attention is all you need → 2024 reality: Transformers power everything', ha='center', va='center',
        fontsize=9, style='italic', color=COLOR_MAIN)

plt.tight_layout()

# Save figure
output_path = '../figures/sr_24_transformer_ecosystem.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
