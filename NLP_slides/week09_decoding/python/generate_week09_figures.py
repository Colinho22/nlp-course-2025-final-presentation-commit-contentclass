"""
Generate figures for Week 9: Decoding Strategies
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

def create_main_figure():
    """Create main visualization for Decoding Strategies."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Placeholder visualization
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y, linewidth=2)
    ax.set_title('Week 9: Decoding Strategies')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.grid(True, alpha=0.3)

    plt.savefig('../figures/week09_main.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Week 9 figures...")
    create_main_figure()
    print("Figures generated successfully!")
