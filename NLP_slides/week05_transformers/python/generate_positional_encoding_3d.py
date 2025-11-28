import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create figure with two subplots
fig = plt.figure(figsize=(14, 6))

# Left panel: Without positional encoding
ax1 = fig.add_subplot(121, projection='3d')

# Example sentences
sentence1 = ["Dog", "bites", "man"]
sentence2 = ["Man", "bites", "dog"]

# Random word embeddings (same for same words)
embeddings = {
    "Dog": np.array([0.7, 0.2, 0.5]),
    "dog": np.array([0.7, 0.2, 0.5]),
    "Man": np.array([0.3, 0.8, 0.4]),
    "man": np.array([0.3, 0.8, 0.4]),
    "bites": np.array([0.5, 0.5, 0.9])
}

# Plot sentence 1
for i, word in enumerate(sentence1):
    vec = embeddings[word]
    ax1.scatter(vec[0], vec[1], vec[2], s=300, c='red', alpha=0.7, edgecolors='darkred', linewidth=2)
    ax1.text(vec[0]+0.05, vec[1]+0.05, vec[2]+0.05, f'{word}', fontsize=10)

# Plot sentence 2
for i, word in enumerate(sentence2):
    vec = embeddings[word]
    ax1.scatter(vec[0], vec[1], vec[2], s=300, c='blue', alpha=0.7, edgecolors='darkblue', linewidth=2)
    ax1.text(vec[0]-0.05, vec[1]-0.05, vec[2]-0.05, f'{word}', fontsize=10)

ax1.set_title('Without Position: Same Words = Same Location\n"Dog bites man" = "Man bites dog" ❌',
             fontsize=11, weight='bold')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_zlim([0, 1])

# Right panel: With positional encoding
ax2 = fig.add_subplot(122, projection='3d')

# Generate positional encodings using sine/cosine
def get_positional_encoding(pos, d_model=3):
    angles = np.array([pos / (10000 ** (2 * i / d_model)) for i in range(d_model)])
    encoding = np.zeros(d_model)
    encoding[0::2] = np.sin(angles[0::2])  # Even indices
    encoding[1::2] = np.cos(angles[1::2])  # Odd indices
    return encoding * 0.2  # Scale down

# Plot sentence 1 with positions
for i, word in enumerate(sentence1):
    vec = embeddings[word] + get_positional_encoding(i)
    ax2.scatter(vec[0], vec[1], vec[2], s=300, c='red', alpha=0.7, edgecolors='darkred', linewidth=2)
    ax2.text(vec[0]+0.05, vec[1]+0.05, vec[2]+0.05, f'{word}[{i}]', fontsize=10)

    # Draw position vector
    orig = embeddings[word]
    ax2.plot([orig[0], vec[0]], [orig[1], vec[1]], [orig[2], vec[2]],
            'r--', alpha=0.3, linewidth=1)

# Plot sentence 2 with positions
for i, word in enumerate(sentence2):
    vec = embeddings[word] + get_positional_encoding(i)
    ax2.scatter(vec[0], vec[1], vec[2], s=300, c='blue', alpha=0.7, edgecolors='darkblue', linewidth=2)
    ax2.text(vec[0]-0.05, vec[1]-0.05, vec[2]-0.05, f'{word}[{i}]', fontsize=10)

    # Draw position vector
    orig = embeddings[word]
    ax2.plot([orig[0], vec[0]], [orig[1], vec[1]], [orig[2], vec[2]],
            'b--', alpha=0.3, linewidth=1)

ax2.set_title('With Position: Word + Position = Unique\n"Dog[0] bites[1] man[2]" ≠ "Man[0] bites[1] dog[2]" ✓',
             fontsize=11, weight='bold')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.set_zlim([0, 1.2])

# Add sine wave visualization at bottom
fig.add_subplot(2, 1, 2)
positions = np.arange(0, 10)
dim = 3

# Generate sine/cosine patterns
for d in range(dim):
    if d % 2 == 0:
        values = np.sin(positions / (10000 ** (d / dim)))
        plt.plot(positions, values, label=f'sin(pos/10000^{d/dim:.1f})', linewidth=2)
    else:
        values = np.cos(positions / (10000 ** ((d-1) / dim)))
        plt.plot(positions, values, '--', label=f'cos(pos/10000^{(d-1)/dim:.1f})', linewidth=2)

plt.xlabel('Position in Sentence', fontsize=11)
plt.ylabel('Encoding Value', fontsize=11)
plt.title('Positional Encoding: Unique Sine/Cosine Waves for Each Dimension', fontsize=12, weight='bold')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.xlim([0, 9])
plt.ylim([-1.2, 1.2])

plt.tight_layout()
plt.savefig('../figures/positional_encoding_3d.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Generated positional_encoding_3d.pdf")