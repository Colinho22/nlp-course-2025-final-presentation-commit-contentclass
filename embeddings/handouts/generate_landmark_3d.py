"""
Generate 3D visualizations with landmark and geographic examples
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import pandas as pd
import os

# Create figures directory
os.makedirs('../figures', exist_ok=True)

# Color scheme
COLORS = {
    'cities': '#2ECC71',      # Green
    'landmarks': '#F39C12',   # Orange
    'animals': '#3498DB',     # Blue
    'music': '#9B59B6',       # Purple
}

def generate_landmark_analogies_3d():
    """Generate 3D visualization of landmark analogies"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define positions (Location, Type, Fame)
    cities_landmarks = {
        # Cities (Type=1)
        'Paris': [1, 1, 2],
        'London': [3, 1, 2],
        'NYC': [4.5, 1, 2.2],
        'Rome': [2, 1, 1.8],

        # Landmarks (Type=3, higher on y-axis)
        'Eiffel Tower': [1, 3, 2.5],
        'Big Ben': [3, 3, 2.5],
        'Statue of Liberty': [4.5, 3, 2.7],
        'Colosseum': [2, 3, 2.3],
    }

    # Plot cities
    cities = ['Paris', 'London', 'NYC', 'Rome']
    for city in cities:
        pos = cities_landmarks[city]
        ax.scatter(*pos, s=200, c=COLORS['cities'], marker='o',
                  edgecolors='black', linewidths=2, alpha=0.8)
        ax.text(pos[0], pos[1], pos[2]+0.1, city, fontsize=9)

    # Plot landmarks
    landmarks = ['Eiffel Tower', 'Big Ben', 'Statue of Liberty', 'Colosseum']
    for landmark in landmarks:
        pos = cities_landmarks[landmark]
        ax.scatter(*pos, s=250, c=COLORS['landmarks'], marker='^',
                  edgecolors='black', linewidths=2, alpha=0.8)
        ax.text(pos[0], pos[1], pos[2]+0.1, landmark, fontsize=8)

    # Draw parallel relationships (city → landmark)
    ax.plot([1, 1], [1, 3], [2, 2.5], 'r-', linewidth=2, alpha=0.7)
    ax.plot([3, 3], [1, 3], [2, 2.5], 'r-', linewidth=2, alpha=0.7)
    ax.plot([4.5, 4.5], [1, 3], [2.2, 2.7], 'r-', linewidth=2, alpha=0.7)
    ax.plot([2, 2], [1, 3], [1.8, 2.3], 'r-', linewidth=2, alpha=0.7)

    # Show analogy: Paris - France + UK = London
    ax.plot([1, 3], [1, 1], [2, 2], 'b--', linewidth=1.5, alpha=0.5)
    ax.plot([1, 3], [3, 3], [2.5, 2.5], 'b--', linewidth=1.5, alpha=0.5)

    ax.set_xlabel('Geographic Location', fontsize=11)
    ax.set_ylabel('Type (City/Landmark)', fontsize=11)
    ax.set_zlabel('Global Fame', fontsize=11)
    ax.set_title('Geographic Analogies: Cities and Their Landmarks', fontsize=14, fontweight='bold')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['cities'],
               markersize=10, label='Cities'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['landmarks'],
               markersize=10, label='Landmarks'),
        Line2D([0], [0], color='r', linewidth=2, label='City→Landmark'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig('../figures/landmark_analogies_3d.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_cultural_analogies_3d():
    """Generate 3D visualization of cultural analogies (Beatles, Mozart, etc.)"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define positions (Geography, Time Period, Genre)
    cultural = {
        # Musicians
        'Beatles': [1, 3, 2],      # UK, 1960s, Rock
        'Mozart': [3, 1, 3],       # Austria, 1700s, Classical
        'Elvis': [5, 2.5, 2],      # USA, 1950s, Rock
        'Beethoven': [3, 1.2, 3],  # Germany, 1800s, Classical

        # Associated places
        'Liverpool': [1, 3, 1],    # UK, Modern
        'Vienna': [3, 1, 1],       # Austria, Historical
        'Memphis': [5, 2.5, 1],    # USA, Modern
        'Bonn': [3, 1.2, 1],       # Germany, Historical
    }

    # Plot musicians
    musicians = ['Beatles', 'Mozart', 'Elvis', 'Beethoven']
    for musician in musicians:
        pos = cultural[musician]
        ax.scatter(*pos, s=250, c=COLORS['music'], marker='*',
                  edgecolors='gold', linewidths=2, alpha=0.9)
        ax.text(pos[0], pos[1], pos[2]+0.15, musician, fontsize=10, fontweight='bold')

    # Plot places
    places = ['Liverpool', 'Vienna', 'Memphis', 'Bonn']
    for place in places:
        pos = cultural[place]
        ax.scatter(*pos, s=200, c=COLORS['cities'], marker='s',
                  edgecolors='black', linewidths=2, alpha=0.7)
        ax.text(pos[0], pos[1], pos[2]+0.15, place, fontsize=9)

    # Draw relationships
    ax.plot([1, 1], [3, 3], [1, 2], 'orange', linewidth=2)
    ax.plot([3, 3], [1, 1], [1, 3], 'orange', linewidth=2)
    ax.plot([5, 5], [2.5, 2.5], [1, 2], 'orange', linewidth=2)
    ax.plot([3, 3], [1.2, 1.2], [1, 3], 'orange', linewidth=2)

    ax.set_xlabel('Geography (UK←→Austria←→USA)', fontsize=11)
    ax.set_ylabel('Time Period (Classical←→Modern)', fontsize=11)
    ax.set_zlabel('Type (Place/Artist)', fontsize=11)
    ax.set_title('Cultural Analogies: Beatles:Liverpool :: Mozart:Vienna',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/cultural_analogies_3d.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_semantic_clusters_3d():
    """Generate clean 3D visualization of semantic clusters"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define word clusters
    clusters = {
        'Animals': {
            'words': ['cat', 'dog', 'pet', 'kitten', 'puppy'],
            'center': [1, 3, 2.5],
            'color': COLORS['animals'],
            'spread': 0.3
        },
        'Landmarks': {
            'words': ['Eiffel', 'Big Ben', 'Liberty', 'Tower Bridge', 'Louvre'],
            'center': [3, 3, 3],
            'color': COLORS['landmarks'],
            'spread': 0.4
        },
        'Cities': {
            'words': ['Paris', 'London', 'NYC', 'Tokyo', 'Berlin'],
            'center': [4, 1, 1],
            'color': COLORS['cities'],
            'spread': 0.35
        }
    }

    # Plot each cluster
    for cluster_name, cluster_info in clusters.items():
        center = np.array(cluster_info['center'])

        # Generate positions around center
        n_words = len(cluster_info['words'])
        angles = np.linspace(0, 2*np.pi, n_words, endpoint=False)

        for i, word in enumerate(cluster_info['words']):
            # Add some randomness for natural clustering
            pos = center + cluster_info['spread'] * np.array([
                np.cos(angles[i]) + np.random.normal(0, 0.1),
                np.sin(angles[i]) + np.random.normal(0, 0.1),
                np.random.normal(0, 0.2)
            ])

            ax.scatter(*pos, s=150, c=cluster_info['color'],
                      alpha=0.7, edgecolors='black', linewidths=1.5)
            ax.text(pos[0], pos[1], pos[2]+0.08, word, fontsize=8)

        # Label cluster
        ax.text(center[0], center[1], center[2]+0.7, cluster_name,
               fontsize=12, fontweight='bold', color=cluster_info['color'])

    # Show example distances
    # Cat to Dog (small)
    cat_pos = clusters['Animals']['center'] + [0.2, 0.1, 0]
    dog_pos = clusters['Animals']['center'] + [-0.1, 0.2, 0.1]
    ax.plot([cat_pos[0], dog_pos[0]], [cat_pos[1], dog_pos[1]],
           [cat_pos[2], dog_pos[2]], 'orange', linewidth=2)

    # Cat to Paris (large)
    paris_pos = clusters['Cities']['center'] + [0.1, 0, 0]
    ax.plot([cat_pos[0], paris_pos[0]], [cat_pos[1], paris_pos[1]],
           [cat_pos[2], paris_pos[2]], 'red', linewidth=2, linestyle='--')

    ax.set_xlabel('Dimension 1', fontsize=11)
    ax.set_ylabel('Dimension 2', fontsize=11)
    ax.set_zlabel('Dimension 3', fontsize=11)
    ax.set_title('Distance = Semantic Difference in 3D Space',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/semantic_clusters_3d.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_interactive_landmark_plot():
    """Create interactive 3D plot with landmarks using plotly"""

    # Prepare data
    data = []

    # Cities and landmarks
    city_landmark_pairs = [
        ('Paris', [2, 1, 2], 'Eiffel Tower', [2, 3, 2.5]),
        ('London', [4, 1, 2], 'Big Ben', [4, 3, 2.5]),
        ('NYC', [6, 1, 2.2], 'Statue of Liberty', [6, 3, 2.7]),
        ('Rome', [3, 1, 1.8], 'Colosseum', [3, 3, 2.3]),
        ('Sydney', [8, 1, 2.1], 'Opera House', [8, 3, 2.6]),
    ]

    cities_x, cities_y, cities_z, cities_names = [], [], [], []
    landmarks_x, landmarks_y, landmarks_z, landmarks_names = [], [], [], []

    for city, city_pos, landmark, landmark_pos in city_landmark_pairs:
        cities_x.append(city_pos[0])
        cities_y.append(city_pos[1])
        cities_z.append(city_pos[2])
        cities_names.append(city)

        landmarks_x.append(landmark_pos[0])
        landmarks_y.append(landmark_pos[1])
        landmarks_z.append(landmark_pos[2])
        landmarks_names.append(landmark)

    # Create figure
    fig = go.Figure()

    # Add cities
    fig.add_trace(go.Scatter3d(
        x=cities_x, y=cities_y, z=cities_z,
        mode='markers+text',
        name='Cities',
        text=cities_names,
        textposition='top center',
        marker=dict(
            size=12,
            color='green',
            opacity=0.8,
            symbol='circle',
            line=dict(color='black', width=2)
        )
    ))

    # Add landmarks
    fig.add_trace(go.Scatter3d(
        x=landmarks_x, y=landmarks_y, z=landmarks_z,
        mode='markers+text',
        name='Landmarks',
        text=landmarks_names,
        textposition='top center',
        marker=dict(
            size=15,
            color='orange',
            opacity=0.9,
            symbol='diamond',
            line=dict(color='black', width=2)
        )
    ))

    # Add connecting lines
    for i in range(len(city_landmark_pairs)):
        fig.add_trace(go.Scatter3d(
            x=[cities_x[i], landmarks_x[i]],
            y=[cities_y[i], landmarks_y[i]],
            z=[cities_z[i], landmarks_z[i]],
            mode='lines',
            line=dict(color='red', width=4),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Update layout
    fig.update_layout(
        title='Interactive 3D: Geographic Analogies (Rotate to Explore!)',
        scene=dict(
            xaxis_title='Geographic Location',
            yaxis_title='Type (City/Landmark)',
            zaxis_title='Global Fame',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=700,
        showlegend=True
    )

    # Save
    fig.write_html('../figures/interactive_landmarks_3d.html')
    return fig

def generate_context_movement_3d():
    """Show how context moves word positions"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Paris in different contexts
    contexts = {
        'Geographic': {
            'Paris': [1, 1, 1],
            'related': ['France', 'city', 'Europe'],
            'positions': [[1.3, 0.8, 1], [0.8, 1.2, 0.9], [1.1, 1, 1.2]],
            'color': 'green'
        },
        'Mythology': {
            'Paris': [4, 4, 3],
            'related': ['Troy', 'Helen', 'Greece'],
            'positions': [[4.2, 3.8, 3.1], [3.8, 4.2, 2.9], [4.1, 4.1, 3.2]],
            'color': 'purple'
        }
    }

    for context_name, context_info in contexts.items():
        # Plot main word
        pos = context_info['Paris']
        ax.scatter(*pos, s=400, c=context_info['color'], marker='*',
                  edgecolors='black', linewidths=2, alpha=0.9)
        ax.text(pos[0], pos[1], pos[2]+0.15, f'Paris\n({context_name})',
               fontsize=10, fontweight='bold', ha='center')

        # Plot related words
        for word, word_pos in zip(context_info['related'], context_info['positions']):
            ax.scatter(*word_pos, s=150, c=context_info['color'],
                      alpha=0.5, edgecolors='black', linewidths=1)
            ax.text(word_pos[0], word_pos[1], word_pos[2]+0.1, word, fontsize=8)

    # Show movement
    ax.plot([1.5, 3.5], [1.5, 3.5], [1.2, 2.8],
           'r--', linewidth=3, alpha=0.7)
    ax.text(2.5, 2.5, 2, 'Context\nShift', fontsize=12,
           fontweight='bold', color='red', ha='center')

    ax.set_xlabel('Semantic Axis 1', fontsize=11)
    ax.set_ylabel('Semantic Axis 2', fontsize=11)
    ax.set_zlabel('Semantic Axis 3', fontsize=11)
    ax.set_title('Context Changes Word Position in 3D Space',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/context_movement_3d.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all 3D visualizations with landmark examples"""
    print("Generating landmark-focused 3D visualizations...")

    print("  - Landmark analogies...")
    generate_landmark_analogies_3d()

    print("  - Cultural analogies...")
    generate_cultural_analogies_3d()

    print("  - Semantic clusters...")
    generate_semantic_clusters_3d()

    print("  - Interactive landmark plot...")
    create_interactive_landmark_plot()

    print("  - Context movement...")
    generate_context_movement_3d()

    print("\nAll visualizations generated successfully!")
    print("Files saved in ../figures/")
    print("Interactive plot: ../figures/interactive_landmarks_3d.html")

if __name__ == "__main__":
    main()