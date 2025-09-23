"""
Generate course roadmap visualization showing the 12-week NLP journey
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define colors
COLOR_FOUNDATION = '#FF6B6B'  # Red
COLOR_REVOLUTION = '#4ECDC4'  # Teal
COLOR_APPLICATION = '#95E77E'  # Green
COLOR_CONNECTION = '#E0E0E0'  # Gray

def create_course_roadmap():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Week information
    weeks = [
        # Foundation Phase
        {'num': 1, 'title': 'Foundations\n& N-grams', 'phase': 'foundation', 'complexity': 1},
        {'num': 2, 'title': 'Neural LM\n& Embeddings', 'phase': 'foundation', 'complexity': 2},
        {'num': 3, 'title': 'RNN/LSTM\n/GRU', 'phase': 'foundation', 'complexity': 3},
        {'num': 4, 'title': 'Seq2Seq\n& Attention', 'phase': 'foundation', 'complexity': 4},
        # Revolution Phase
        {'num': 5, 'title': 'Transformers\n& Self-Attention', 'phase': 'revolution', 'complexity': 5},
        {'num': 6, 'title': 'BERT\n& GPT', 'phase': 'revolution', 'complexity': 6},
        {'num': 7, 'title': 'Advanced\nTransformers', 'phase': 'revolution', 'complexity': 7},
        {'num': 8, 'title': 'Tokenization\nStrategies', 'phase': 'revolution', 'complexity': 4},
        # Application Phase
        {'num': 9, 'title': 'Decoding\nStrategies', 'phase': 'application', 'complexity': 5},
        {'num': 10, 'title': 'Fine-tuning\n& Prompts', 'phase': 'application', 'complexity': 6},
        {'num': 11, 'title': 'Efficiency\n& Optimization', 'phase': 'application', 'complexity': 7},
        {'num': 12, 'title': 'Ethics\n& Fairness', 'phase': 'application', 'complexity': 8}
    ]
    
    # Position calculation
    x_spacing = 1.0
    y_base = 0
    
    # Draw connections first (behind nodes)
    for i in range(len(weeks) - 1):
        x1 = i * x_spacing
        x2 = (i + 1) * x_spacing
        y1 = weeks[i]['complexity'] * 0.8
        y2 = weeks[i+1]['complexity'] * 0.8
        
        # Draw curved connection
        mid_x = (x1 + x2) / 2
        mid_y = max(y1, y2) + 0.3
        
        # Create curved path
        t = np.linspace(0, 1, 100)
        bezier_x = (1-t)**2 * x1 + 2*(1-t)*t * mid_x + t**2 * x2
        bezier_y = (1-t)**2 * y1 + 2*(1-t)*t * mid_y + t**2 * y2
        
        ax.plot(bezier_x, bezier_y, color=COLOR_CONNECTION, linewidth=2, alpha=0.5, zorder=1)
    
    # Draw phase backgrounds
    phase_colors = {
        'foundation': COLOR_FOUNDATION,
        'revolution': COLOR_REVOLUTION,
        'application': COLOR_APPLICATION
    }
    
    # Foundation phase background
    foundation_patch = FancyBboxPatch((-0.5, -0.5), 3.5, 9,
                                     boxstyle="round,pad=0.1",
                                     facecolor=COLOR_FOUNDATION, alpha=0.1,
                                     edgecolor=COLOR_FOUNDATION, linewidth=2,
                                     linestyle='--', zorder=0)
    ax.add_patch(foundation_patch)
    
    # Revolution phase background
    revolution_patch = FancyBboxPatch((3.5, -0.5), 4, 9,
                                     boxstyle="round,pad=0.1",
                                     facecolor=COLOR_REVOLUTION, alpha=0.1,
                                     edgecolor=COLOR_REVOLUTION, linewidth=2,
                                     linestyle='--', zorder=0)
    ax.add_patch(revolution_patch)
    
    # Application phase background
    application_patch = FancyBboxPatch((7.5, -0.5), 3.5, 9,
                                      boxstyle="round,pad=0.1",
                                      facecolor=COLOR_APPLICATION, alpha=0.1,
                                      edgecolor=COLOR_APPLICATION, linewidth=2,
                                      linestyle='--', zorder=0)
    ax.add_patch(application_patch)
    
    # Draw week nodes
    for i, week in enumerate(weeks):
        x = i * x_spacing
        y = week['complexity'] * 0.8
        
        # Determine color based on phase
        color = phase_colors[week['phase']]
        
        # Draw main circle
        circle = Circle((x, y), 0.35, facecolor=color, edgecolor='white',
                       linewidth=3, zorder=3, alpha=0.9)
        ax.add_patch(circle)
        
        # Add week number
        ax.text(x, y + 0.05, f"W{week['num']}", fontsize=14, fontweight='bold',
                ha='center', va='center', color='white', zorder=4)
        
        # Add title below
        ax.text(x, y - 0.6, week['title'], fontsize=9, ha='center', va='top',
                color='black', fontweight='bold', zorder=4)
        
        # Add complexity indicator
        for j in range(int(week['complexity'])):
            star_y = -0.2 - j * 0.1
            ax.plot(x, star_y, marker='*', markersize=8, color='gold',
                   markeredgecolor='orange', markeredgewidth=0.5, zorder=2)
    
    # Add phase labels
    ax.text(1.5, 7.5, 'FOUNDATION PHASE', fontsize=14, fontweight='bold',
            color=COLOR_FOUNDATION, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax.text(5.5, 7.5, 'REVOLUTION PHASE', fontsize=14, fontweight='bold',
            color=COLOR_REVOLUTION, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax.text(9.5, 7.5, 'APPLICATION PHASE', fontsize=14, fontweight='bold',
            color=COLOR_APPLICATION, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Add milestone markers
    milestones = [
        (1, 8, 'Start: Basic Probability'),
        (4, 8, 'First Neural Models'),
        (6, 8, 'Transformer Revolution'),
        (10, 8, 'Industry Ready!')
    ]
    
    for x, y, text in milestones:
        ax.annotate(text, xy=(x * x_spacing, weeks[x-1]['complexity'] * 0.8),
                   xytext=(x * x_spacing, y - 0.5),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                   fontsize=10, ha='center', color='darkblue',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Add complexity legend
    ax.text(11.5, 1, 'Complexity:', fontsize=10, fontweight='bold')
    for i in range(1, 9):
        y_pos = 0.5 - i * 0.1
        ax.plot(11.5, y_pos, marker='*', markersize=8, color='gold',
               markeredgecolor='orange', markeredgewidth=0.5)
        if i == 1:
            ax.text(11.7, y_pos, 'Low', fontsize=8, va='center')
        elif i == 8:
            ax.text(11.7, y_pos, 'High', fontsize=8, va='center')
    
    # Set limits and remove axes
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1.5, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    ax.text(5.5, -1.2, 'Your 12-Week Journey to NLP Mastery', 
            fontsize=16, fontweight='bold', ha='center', color='darkblue')
    
    plt.tight_layout()
    plt.savefig('../figures/course_roadmap.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

if __name__ == '__main__':
    create_course_roadmap()
    print("Course roadmap visualization generated successfully!")