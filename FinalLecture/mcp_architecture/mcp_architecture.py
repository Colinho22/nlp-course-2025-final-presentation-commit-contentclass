"""
Model Context Protocol (MCP) Architecture Chart
Shows how MCP connects LLMs to external tools through a standardized protocol
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set up the figure
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')

# Colors
colors = {
    'llm': '#3498db',
    'mcp': '#e74c3c',
    'server': '#2ecc71',
    'resource': '#95a5a6',
    'arrow': '#34495e'
}

# LLM Application (left side)
llm_box = mpatches.FancyBboxPatch((0.5, 2), 2.5, 3,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['llm'], alpha=0.9)
ax.add_patch(llm_box)
ax.text(1.75, 4.5, 'LLM\nApplication', ha='center', va='center',
        fontsize=12, fontweight='bold', color='white')
ax.text(1.75, 3, 'Claude Code\nChatbot\nIDE Plugin', ha='center', va='center',
        fontsize=9, color='white', alpha=0.9)

# MCP Client
client_box = mpatches.FancyBboxPatch((3.5, 2.8), 2, 1.4,
                                      boxstyle="round,pad=0.05",
                                      facecolor=colors['mcp'], alpha=0.8)
ax.add_patch(client_box)
ax.text(4.5, 3.5, 'MCP Client', ha='center', va='center',
        fontsize=10, fontweight='bold', color='white')

# Protocol layer (middle)
proto_box = mpatches.FancyBboxPatch((6, 2.3), 2, 2.4,
                                     boxstyle="round,pad=0.05",
                                     facecolor='white', edgecolor=colors['mcp'],
                                     linewidth=3)
ax.add_patch(proto_box)
ax.text(7, 4.2, 'JSON-RPC', ha='center', va='center',
        fontsize=10, fontweight='bold', color=colors['mcp'])
ax.text(7, 3.5, 'Protocol', ha='center', va='center',
        fontsize=9, color=colors['mcp'])
ax.text(7, 2.8, 'stdio/HTTP', ha='center', va='center',
        fontsize=8, color='gray')

# MCP Servers (right side)
servers = [
    {'name': 'File System\nServer', 'y': 5.5},
    {'name': 'Database\nServer', 'y': 3.5},
    {'name': 'API\nServer', 'y': 1.5}
]

for s in servers:
    server_box = mpatches.FancyBboxPatch((8.5, s['y']-0.6), 2, 1.2,
                                          boxstyle="round,pad=0.05",
                                          facecolor=colors['server'], alpha=0.9)
    ax.add_patch(server_box)
    ax.text(9.5, s['y'], s['name'], ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')

# External Resources (far right)
resources = [
    {'name': 'Local\nFiles', 'y': 5.5},
    {'name': 'PostgreSQL\nSQLite', 'y': 3.5},
    {'name': 'GitHub\nSlack', 'y': 1.5}
]

for r in resources:
    res_box = mpatches.FancyBboxPatch((11, r['y']-0.5), 1.8, 1,
                                       boxstyle="round,pad=0.05",
                                       facecolor=colors['resource'], alpha=0.7)
    ax.add_patch(res_box)
    ax.text(11.9, r['y'], r['name'], ha='center', va='center',
            fontsize=8, color='white')

# Arrows
# LLM to Client
ax.annotate('', xy=(3.4, 3.5), xytext=(3.1, 3.5),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

# Client to Protocol
ax.annotate('', xy=(5.9, 3.5), xytext=(5.6, 3.5),
            arrowprops=dict(arrowstyle='<->', color=colors['mcp'], lw=2))

# Protocol to Servers
for s in servers:
    ax.annotate('', xy=(8.4, s['y']), xytext=(8.1, 3.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5,
                               connectionstyle='arc3,rad=0.1'))

# Servers to Resources
for i, s in enumerate(servers):
    ax.annotate('', xy=(10.9, resources[i]['y']), xytext=(10.6, s['y']),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Title
ax.text(7, 6.7, 'Model Context Protocol (MCP) Architecture',
        ha='center', va='center', fontsize=14, fontweight='bold')

# Key benefits at bottom
benefits = ['Standardized Interface', 'Language Agnostic', 'Secure Sandboxing', 'Hot-swappable']
for i, b in enumerate(benefits):
    ax.text(2 + i*3, 0.5, f"[+] {b}", ha='center', va='center',
            fontsize=8, color=colors['server'])

plt.tight_layout()
plt.savefig('mcp_architecture.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()
print("Chart saved: mcp_architecture.pdf")
