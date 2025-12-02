"""
Agent Memory Architectures Chart
Shows the three types of memory in AI agent systems
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Colors
colors = {
    'short': '#e74c3c',
    'long': '#3498db',
    'episodic': '#2ecc71',
    'agent': '#34495e'
}

# Title
ax.text(7, 7.5, 'Agent Memory Architectures',
        ha='center', va='center', fontsize=16, fontweight='bold')

# Agent in center
agent_circle = plt.Circle((7, 4), 1, color=colors['agent'], zorder=3)
ax.add_patch(agent_circle)
ax.text(7, 4, 'AI\nAgent', ha='center', va='center', fontsize=14,
        fontweight='bold', color='white')

# Memory types
memory_types = [
    {
        'name': 'Short-Term Memory',
        'subtitle': '(Working Memory)',
        'pos': (2.5, 5.5),
        'color': colors['short'],
        'items': [
            'Context window',
            'Current conversation',
            'Scratchpad for reasoning',
            'Limited by token count'
        ],
        'duration': 'Seconds to minutes'
    },
    {
        'name': 'Long-Term Memory',
        'subtitle': '(Persistent Storage)',
        'pos': (11.5, 5.5),
        'color': colors['long'],
        'items': [
            'Vector database',
            'Knowledge base',
            'User preferences',
            'Learned patterns'
        ],
        'duration': 'Days to permanent'
    },
    {
        'name': 'Episodic Memory',
        'subtitle': '(Experience Replay)',
        'pos': (7, 1),
        'color': colors['episodic'],
        'items': [
            'Past task executions',
            'Success/failure records',
            'Strategy refinement',
            'Few-shot examples'
        ],
        'duration': 'Session to permanent'
    }
]

for mem in memory_types:
    # Draw memory box
    width = 4.5
    height = 2.5
    box = mpatches.FancyBboxPatch((mem['pos'][0]-width/2, mem['pos'][1]-height/2),
                                   width, height,
                                   boxstyle="round,pad=0.1",
                                   facecolor=mem['color'], alpha=0.9)
    ax.add_patch(box)

    # Memory name
    ax.text(mem['pos'][0], mem['pos'][1]+0.8, mem['name'],
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(mem['pos'][0], mem['pos'][1]+0.4, mem['subtitle'],
            ha='center', va='center', fontsize=8, color='white', alpha=0.8)

    # Items
    for i, item in enumerate(mem['items']):
        ax.text(mem['pos'][0], mem['pos'][1]-0.1-i*0.35, f"- {item}",
                ha='center', va='center', fontsize=7, color='white')

    # Duration label
    ax.text(mem['pos'][0], mem['pos'][1]-height/2-0.2, f"Duration: {mem['duration']}",
            ha='center', va='top', fontsize=7, style='italic', color='gray')

    # Draw arrow from agent to memory
    dx = mem['pos'][0] - 7
    dy = mem['pos'][1] - 4
    dist = np.sqrt(dx**2 + dy**2)

    # Start from edge of agent circle
    start_x = 7 + 1.1 * dx/dist
    start_y = 4 + 1.1 * dy/dist

    # End at edge of memory box
    end_x = mem['pos'][0] - 1.5 * dx/dist
    end_y = mem['pos'][1] - 0.8 * dy/dist

    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=2))

# Read/Write labels
ax.text(4, 5.2, 'Read/\nWrite', ha='center', va='center', fontsize=8, color='gray')
ax.text(10, 5.2, 'Retrieve/\nStore', ha='center', va='center', fontsize=8, color='gray')
ax.text(7, 2.3, 'Recall/\nRecord', ha='center', va='center', fontsize=8, color='gray')

# Legend with capacity indicators
legend_y = 0.3
ax.text(2, legend_y, 'Capacity:', ha='left', fontsize=9, fontweight='bold')
ax.add_patch(mpatches.Rectangle((3.5, legend_y-0.15), 0.8, 0.3, color=colors['short']))
ax.text(4.5, legend_y, 'Small (4K-128K tokens)', fontsize=8)
ax.add_patch(mpatches.Rectangle((8, legend_y-0.15), 0.8, 0.3, color=colors['long']))
ax.text(9, legend_y, 'Large (millions of vectors)', fontsize=8)

plt.tight_layout()
plt.savefig('agent_memory_types.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()
print("Chart saved: agent_memory_types.pdf")
