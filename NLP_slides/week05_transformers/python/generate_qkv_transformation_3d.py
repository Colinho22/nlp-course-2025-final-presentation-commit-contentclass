"""
Generate 3D visualization of Query-Key-Value transformation
Shows how a word embedding is transformed into three different spaces
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Custom 3D arrow class
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

# Create figure
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# Word embedding (original vector)
word = "mat"
embedding_pos = np.array([0, 0, 0])
embedding_vec = np.array([0.7, 0.5, 0.6])

# Draw original word embedding as a large sphere
ax.scatter(*embedding_pos, s=800, c='gold', edgecolors='black',
          linewidth=2, alpha=0.9, marker='o')
ax.text(embedding_pos[0], embedding_pos[1], embedding_pos[2] - 0.3,
        f'Word: "{word}"\nEmbedding Vector\n[0.7, 0.5, 0.6]',
        fontsize=11, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Query space (top-left)
query_pos = np.array([-2, -2, 2])
query_vec = np.array([0.8, 0.3, 0.4])  # Transformed vector
ax.scatter(*query_pos, s=600, c='#FF6B6B', edgecolors='black',
          linewidth=2, alpha=0.9, marker='s')

# Draw transformation arrow to Query
arrow_q = Arrow3D([embedding_pos[0], query_pos[0]],
                  [embedding_pos[1], query_pos[1]],
                  [embedding_pos[2], query_pos[2]],
                  mutation_scale=20, lw=3, arrowstyle='-|>',
                  color='#FF6B6B', alpha=0.7)
ax.add_artist(arrow_q)

# Query vector visualization
query_arrow = Arrow3D([query_pos[0], query_pos[0] + query_vec[0]],
                      [query_pos[1], query_pos[1] + query_vec[1]],
                      [query_pos[2], query_pos[2] + query_vec[2]],
                      mutation_scale=15, lw=4, arrowstyle='-|>',
                      color='darkred')
ax.add_artist(query_arrow)

ax.text(query_pos[0], query_pos[1], query_pos[2] + 0.5,
        'QUERY (Q)\n"What am I looking for?"',
        fontsize=12, fontweight='bold', color='#FF6B6B', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add example annotations for Query
ax.text(query_pos[0] - 0.5, query_pos[1] - 0.5, query_pos[2] - 0.5,
        'Examples:\n• Need location info\n• Need subject info\n• Need action info',
        fontsize=9, color='darkred', alpha=0.8,
        bbox=dict(boxstyle='round', facecolor='#FFE0E0', alpha=0.7))

# Key space (top-right)
key_pos = np.array([2, -2, 2])
key_vec = np.array([0.6, 0.7, 0.3])
ax.scatter(*key_pos, s=600, c='#4ECDC4', edgecolors='black',
          linewidth=2, alpha=0.9, marker='^')

# Draw transformation arrow to Key
arrow_k = Arrow3D([embedding_pos[0], key_pos[0]],
                  [embedding_pos[1], key_pos[1]],
                  [embedding_pos[2], key_pos[2]],
                  mutation_scale=20, lw=3, arrowstyle='-|>',
                  color='#4ECDC4', alpha=0.7)
ax.add_artist(arrow_k)

# Key vector visualization
key_arrow = Arrow3D([key_pos[0], key_pos[0] + key_vec[0]],
                    [key_pos[1], key_pos[1] + key_vec[1]],
                    [key_pos[2], key_pos[2] + key_vec[2]],
                    mutation_scale=15, lw=4, arrowstyle='-|>',
                    color='darkcyan')
ax.add_artist(key_arrow)

ax.text(key_pos[0], key_pos[1], key_pos[2] + 0.5,
        'KEY (K)\n"What do I contain?"',
        fontsize=12, fontweight='bold', color='#4ECDC4', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add example annotations for Key
ax.text(key_pos[0] + 0.5, key_pos[1] - 0.5, key_pos[2] - 0.5,
        'Examples:\n• I have location info\n• I have object info\n• I have color info',
        fontsize=9, color='darkcyan', alpha=0.8,
        bbox=dict(boxstyle='round', facecolor='#E0FFF8', alpha=0.7))

# Value space (bottom)
value_pos = np.array([0, 2, 1])
value_vec = np.array([0.5, 0.4, 0.8])
ax.scatter(*value_pos, s=600, c='#95E77E', edgecolors='black',
          linewidth=2, alpha=0.9, marker='D')

# Draw transformation arrow to Value
arrow_v = Arrow3D([embedding_pos[0], value_pos[0]],
                  [embedding_pos[1], value_pos[1]],
                  [embedding_pos[2], value_pos[2]],
                  mutation_scale=20, lw=3, arrowstyle='-|>',
                  color='#95E77E', alpha=0.7)
ax.add_artist(arrow_v)

# Value vector visualization
value_arrow = Arrow3D([value_pos[0], value_pos[0] + value_vec[0]],
                      [value_pos[1], value_pos[1] + value_vec[1]],
                      [value_pos[2], value_pos[2] + value_vec[2]],
                      mutation_scale=15, lw=4, arrowstyle='-|>',
                      color='darkgreen')
ax.add_artist(value_arrow)

ax.text(value_pos[0], value_pos[1], value_pos[2] + 0.5,
        'VALUE (V)\n"What info do I provide?"',
        fontsize=12, fontweight='bold', color='#95E77E', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add example annotations for Value
ax.text(value_pos[0], value_pos[1] + 0.5, value_pos[2] - 0.8,
        'Examples:\n• Surface/floor context\n• Physical object info\n• Positional pattern',
        fontsize=9, color='darkgreen', alpha=0.8,
        bbox=dict(boxstyle='round', facecolor='#E8FFE0', alpha=0.7))

# Draw the interaction area (where Q and K meet)
interaction_center = np.array([0, -1.5, 2.5])
ax.scatter(*interaction_center, s=400, c='orange', alpha=0.3, marker='o')

# Draw dotted lines showing Q·K computation
q_k_line_x = [query_pos[0], interaction_center[0], key_pos[0]]
q_k_line_y = [query_pos[1], interaction_center[1], key_pos[1]]
q_k_line_z = [query_pos[2], interaction_center[2], key_pos[2]]
ax.plot(q_k_line_x, q_k_line_y, q_k_line_z, 'k--', alpha=0.5, linewidth=2)

ax.text(interaction_center[0], interaction_center[1], interaction_center[2],
        'Q · K = Attention Score\n(How relevant?)',
        fontsize=10, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

# Add transformation matrices indicators
ax.text(-3, 0, 0, 'WQ', fontsize=14, fontweight='bold', color='#FF6B6B',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax.text(3, 0, 0, 'WK', fontsize=14, fontweight='bold', color='#4ECDC4',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax.text(0, 3, 0, 'WV', fontsize=14, fontweight='bold', color='#95E77E',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Title and labels
ax.set_title('Query-Key-Value: Three Different Perspectives on Same Word\n' +
            'Each transformation extracts different aspects of meaning',
            fontsize=14, fontweight='bold', pad=20)

# Set viewing angle
ax.view_init(elev=15, azim=-35)

# Set limits
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-1, 3)

# Remove axis labels for cleaner look
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Add legend
legend_elements = [
    mpatches.Patch(color='#FF6B6B', label='Query: Seeking information'),
    mpatches.Patch(color='#4ECDC4', label='Key: Advertising content'),
    mpatches.Patch(color='#95E77E', label='Value: Actual information')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

# Add bottom explanation
ax.text2D(0.5, 0.02,
         'The same word "mat" is transformed into 3 spaces: Q asks what it needs, K advertises what it has, V provides the actual content',
         transform=ax.transAxes, fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Add formula
ax.text2D(0.85, 0.85,
         'Attention(Q,K,V) = softmax(QK^T)V',
         transform=ax.transAxes, fontsize=12, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('../figures/qkv_transformation_3d.pdf', dpi=300, bbox_inches='tight')
print("Generated: qkv_transformation_3d.pdf")
plt.close()