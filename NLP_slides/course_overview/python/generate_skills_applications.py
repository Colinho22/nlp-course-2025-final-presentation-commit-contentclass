"""
Generate skills and applications Sankey diagram visualization
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_skills_applications():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Left plot: Skills to Technologies to Applications flow
    create_sankey_diagram(ax1)
    
    # Right plot: Industry demand and growth
    create_industry_metrics(ax2)
    
    plt.tight_layout()
    plt.savefig('../figures/skills_applications.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def create_sankey_diagram(ax):
    """Create a Sankey-style flow diagram"""
    
    # Define categories
    skills = [
        'Mathematical\nFoundations',
        'Programming\n(Python)',
        'Deep Learning\nBasics',
        'NLP Theory',
        'Model\nEngineering',
        'Deployment\nSkills'
    ]
    
    technologies = [
        'PyTorch/\nTensorFlow',
        'Hugging Face\nTransformers',
        'LangChain/\nLlamaIndex',
        'ONNX/\nTensorRT',
        'Docker/\nKubernetes'
    ]
    
    applications = [
        'Chatbots &\nAssistants',
        'Translation\nSystems',
        'Code\nGeneration',
        'Content\nCreation',
        'Search\nEngines',
        'Analytics\nTools'
    ]
    
    # Positions
    y_skills = np.linspace(0.9, 0.1, len(skills))
    y_tech = np.linspace(0.85, 0.15, len(technologies))
    y_apps = np.linspace(0.9, 0.1, len(applications))
    
    # Draw boxes for each category
    box_width = 0.15
    colors = ['#FF6B6B', '#4ECDC4', '#95E77E']
    
    # Skills boxes
    for i, (skill, y) in enumerate(zip(skills, y_skills)):
        rect = FancyBboxPatch((0.05, y - 0.04), box_width, 0.08,
                              boxstyle="round,pad=0.01",
                              facecolor=colors[0], alpha=0.7,
                              edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(0.05 + box_width/2, y, skill, fontsize=8, 
               ha='center', va='center', fontweight='bold', color='white')
    
    # Technologies boxes
    for i, (tech, y) in enumerate(zip(technologies, y_tech)):
        rect = FancyBboxPatch((0.425, y - 0.04), box_width, 0.08,
                              boxstyle="round,pad=0.01",
                              facecolor=colors[1], alpha=0.7,
                              edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(0.425 + box_width/2, y, tech, fontsize=8,
               ha='center', va='center', fontweight='bold', color='white')
    
    # Applications boxes
    for i, (app, y) in enumerate(zip(applications, y_apps)):
        rect = FancyBboxPatch((0.8, y - 0.04), box_width, 0.08,
                              boxstyle="round,pad=0.01",
                              facecolor=colors[2], alpha=0.7,
                              edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(0.8 + box_width/2, y, app, fontsize=8,
               ha='center', va='center', fontweight='bold', color='white')
    
    # Draw flowing connections
    # Skills to Technologies connections
    connections_st = [
        (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1),
        (3, 1), (3, 2), (4, 1), (4, 2), (4, 3), (5, 3), (5, 4)
    ]
    
    for skill_idx, tech_idx in connections_st:
        draw_flow(ax, 0.05 + box_width, y_skills[skill_idx],
                 0.425, y_tech[tech_idx], 
                 alpha=0.3, color=colors[0])
    
    # Technologies to Applications connections
    connections_ta = [
        (0, 2), (1, 0), (1, 1), (1, 3), (2, 0), (2, 3),
        (2, 4), (3, 2), (3, 5), (4, 0), (4, 5)
    ]
    
    for tech_idx, app_idx in connections_ta:
        draw_flow(ax, 0.425 + box_width, y_tech[tech_idx],
                 0.8, y_apps[app_idx],
                 alpha=0.3, color=colors[1])
    
    # Add category headers
    ax.text(0.125, 1.05, 'SKILLS', fontsize=12, fontweight='bold',
           ha='center', color=colors[0])
    ax.text(0.5, 1.05, 'TECHNOLOGIES', fontsize=12, fontweight='bold',
           ha='center', color=colors[1])
    ax.text(0.875, 1.05, 'APPLICATIONS', fontsize=12, fontweight='bold',
           ha='center', color=colors[2])
    
    # Add flow direction indicators
    arrow_y = 0.02
    ax.annotate('', xy=(0.35, arrow_y), xytext=(0.25, arrow_y),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.annotate('', xy=(0.72, arrow_y), xytext=(0.62, arrow_y),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_title('From Learning to Real-World Impact', fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')

def draw_flow(ax, x1, y1, x2, y2, alpha=0.3, color='blue'):
    """Draw a flowing connection between two points"""
    # Create bezier curve
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    
    t = np.linspace(0, 1, 100)
    bezier_x = (1-t)**2 * x1 + 2*(1-t)*t * mid_x + t**2 * x2
    bezier_y = (1-t)**2 * y1 + 2*(1-t)*t * mid_y + t**2 * y2
    
    # Variable width along the curve
    widths = 0.02 * np.sin(np.pi * t)
    
    for i in range(len(t)-1):
        ax.plot(bezier_x[i:i+2], bezier_y[i:i+2], 
               color=color, alpha=alpha, 
               linewidth=widths[i]*500)

def create_industry_metrics(ax):
    """Create industry demand and market metrics visualization"""
    
    # Job demand data
    years = np.array([2020, 2021, 2022, 2023, 2024, 2025])
    jobs = np.array([2000, 3500, 5500, 8000, 10000, 13000])
    salaries = np.array([120, 130, 145, 160, 175, 190])  # in thousands
    
    # Market size data
    market_size = np.array([15, 20, 26, 33, 40, 48])  # in billions
    
    # Create twin axes
    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    
    # Plot job openings
    line1 = ax.plot(years, jobs, 'o-', color='#FF6B6B', linewidth=3, 
                   markersize=10, label='Job Openings')
    ax.fill_between(years, 0, jobs, alpha=0.3, color='#FF6B6B')
    
    # Plot average salaries
    line2 = ax2.plot(years, salaries, 's-', color='#4ECDC4', linewidth=3,
                    markersize=10, label='Avg Salary ($k)')
    ax2.fill_between(years, 0, salaries, alpha=0.3, color='#4ECDC4')
    
    # Plot market size
    line3 = ax3.plot(years, market_size, '^-', color='#95E77E', linewidth=3,
                    markersize=10, label='Market Size ($B)')
    
    # Formatting
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Job Openings', fontsize=11, fontweight='bold', color='#FF6B6B')
    ax2.set_ylabel('Average Salary ($k)', fontsize=11, fontweight='bold', color='#4ECDC4')
    ax3.set_ylabel('Market Size ($B)', fontsize=11, fontweight='bold', color='#95E77E')
    
    ax.tick_params(axis='y', labelcolor='#FF6B6B')
    ax2.tick_params(axis='y', labelcolor='#4ECDC4')
    ax3.tick_params(axis='y', labelcolor='#95E77E')
    
    ax.set_title('NLP Industry Growth Metrics', fontsize=14, fontweight='bold', pad=20)
    
    # Add annotations for key points
    ax.annotate('10,000+\nOpenings', xy=(2024, 10000), xytext=(2023, 11500),
               fontsize=10, fontweight='bold', color='#FF6B6B',
               arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=1.5))
    
    ax2.annotate('$175k\nAverage', xy=(2024, 175), xytext=(2022.5, 185),
                fontsize=10, fontweight='bold', color='#4ECDC4',
                arrowprops=dict(arrowstyle='->', color='#4ECDC4', lw=1.5))
    
    # Add growth rate box
    growth_box = FancyBboxPatch((2020.5, 12000), 2, 1500,
                               boxstyle="round,pad=0.1",
                               facecolor='yellow', alpha=0.7,
                               edgecolor='orange', linewidth=2)
    ax.add_patch(growth_box)
    ax.text(2021.5, 12750, '45% Annual\nGrowth Rate', fontsize=10,
           fontweight='bold', ha='center', va='center')
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=10)
    
    ax.set_xlim(2019.5, 2025.5)
    ax.set_ylim(0, 15000)
    ax2.set_ylim(0, 220)
    ax3.set_ylim(0, 55)
    ax.grid(True, alpha=0.3)

if __name__ == '__main__':
    create_skills_applications()
    print("Skills and applications visualization generated successfully!")