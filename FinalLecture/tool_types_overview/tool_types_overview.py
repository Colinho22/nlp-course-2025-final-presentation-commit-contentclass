"""
Tool Types Overview Chart
Shows the four main categories of tools available to AI agents
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# Define colors
colors = {
    'retrieval': '#3498db',
    'code': '#e74c3c',
    'api': '#2ecc71',
    'system': '#9b59b6',
    'center': '#34495e'
}

# Center circle - Agent
center = plt.Circle((5, 4), 0.8, color=colors['center'], zorder=3)
ax.add_patch(center)
ax.text(5, 4, 'AI\nAgent', ha='center', va='center', fontsize=14,
        fontweight='bold', color='white')

# Tool categories - positioned around center
categories = [
    {'name': 'Information\nRetrieval', 'pos': (2, 6), 'color': colors['retrieval'],
     'examples': ['Web Search', 'Vector DB', 'Document Lookup']},
    {'name': 'Code\nExecution', 'pos': (8, 6), 'color': colors['code'],
     'examples': ['Python REPL', 'Shell Commands', 'Sandboxed Env']},
    {'name': 'External\nAPIs', 'pos': (2, 2), 'color': colors['api'],
     'examples': ['Weather', 'Email', 'Calendar']},
    {'name': 'System\nOperations', 'pos': (8, 2), 'color': colors['system'],
     'examples': ['File I/O', 'Browser Control', 'App Automation']}
]

for cat in categories:
    # Draw category box
    box = mpatches.FancyBboxPatch((cat['pos'][0]-1.2, cat['pos'][1]-0.8), 2.4, 1.6,
                                   boxstyle="round,pad=0.1",
                                   facecolor=cat['color'], alpha=0.9)
    ax.add_patch(box)

    # Category name
    ax.text(cat['pos'][0], cat['pos'][1]+0.3, cat['name'],
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # Examples (smaller text)
    examples_text = '\n'.join([f"- {e}" for e in cat['examples']])
    ax.text(cat['pos'][0], cat['pos'][1]-0.5, examples_text,
            ha='center', va='top', fontsize=7, color='white', alpha=0.9)

    # Draw arrow from center to category
    dx = cat['pos'][0] - 5
    dy = cat['pos'][1] - 4
    dist = np.sqrt(dx**2 + dy**2)
    # Start arrow from edge of center circle
    start_x = 5 + 0.85 * dx/dist
    start_y = 4 + 0.85 * dy/dist
    # End arrow at edge of category box
    end_x = cat['pos'][0] - 0.9 * dx/dist
    end_y = cat['pos'][1] - 0.9 * dy/dist

    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

# Title
ax.text(5, 7.5, 'Tool Types Available to AI Agents',
        ha='center', va='center', fontsize=16, fontweight='bold')

# Subtitle
ax.text(5, 0.5, 'Tools extend agent capabilities beyond pure language understanding',
        ha='center', va='center', fontsize=10, style='italic', color='gray')

plt.tight_layout()
plt.savefig('tool_types_overview.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()
print("Chart saved: tool_types_overview.pdf")
