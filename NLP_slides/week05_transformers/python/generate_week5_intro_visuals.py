"""
Generate stunning K-means inspired 3D visualizations for Week 5 Transformers BSc
Features ultra-vibrant colors, crystal effects, and galaxy-style clustering
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, LightSource
from matplotlib.patches import Circle, Ellipse
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.spatial import Voronoi, SphericalVoronoi, voronoi_plot_2d
from scipy.stats import multivariate_normal
import matplotlib.patches as mpatches

# Set style for professional visuals
plt.style.use('seaborn-v0_8-whitegrid')

# Define vibrant color palettes
NEON_COLORS = ['#FF00FF', '#00FFFF', '#FFFF00', '#FF1493', '#00CED1', 
               '#32CD32', '#FFD700', '#FF69B4', '#9370DB', '#FF4500',
               '#00FA9A', '#FF6347', '#4169E1', '#FFA500', '#8A2BE2']

GALAXY_COLORS = ['#FF1493', '#00CED1', '#FFD700', '#FF69B4', '#32CD32',
                 '#FF4500', '#9370DB', '#00FA9A', '#4169E1', '#FFA500',
                 '#DC143C', '#00BFFF', '#ADFF2F', '#FF00FF', '#F0E68C']

def create_attention_landscape():
    """Create an ultra-vibrant 3D crystalline attention visualization"""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create high-resolution meshgrid for smoother surface
    resolution = 200
    x = np.linspace(0, 10, resolution)
    y = np.linspace(0, 10, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create complex attention landscape with crystalline structure
    Z = np.zeros_like(X)
    
    # Define attention peaks with more dramatic variations
    peaks = [
        (2, 3, 3.5, 0.6, 0.6, '#FF00FF', 'Subject-Verb'),   
        (3, 7, 3.2, 0.5, 0.7, '#00FFFF', 'Verb-Object'),    
        (5, 5, 2.8, 0.8, 0.8, '#FFFF00', 'Self-Attention'),  
        (7, 2, 3.0, 0.6, 0.6, '#FF1493', 'Reference'),       
        (8, 8, 3.3, 0.4, 0.4, '#00CED1', 'Adjective-Noun'),  
        (1, 8, 2.5, 0.7, 0.5, '#32CD32', 'Long-Range'),      
        (4, 1, 2.7, 0.5, 0.6, '#FFD700', 'Dependency'),      
        (6, 9, 2.9, 0.6, 0.5, '#FF69B4', 'Context'),         
        (9, 4, 2.6, 0.5, 0.7, '#9370DB', 'Modifier'),        
    ]
    
    # Create main attention peaks with crystalline facets
    for px, py, height, sx, sy, color, label in peaks:
        # Main gaussian peak
        gaussian = height * np.exp(-((X - px)**2 / (2 * sx**2) + (Y - py)**2 / (2 * sy**2)))
        
        # Add crystalline facets using angular modulation
        angle = np.arctan2(Y - py, X - px)
        facet_modulation = 1 + 0.3 * np.sin(6 * angle)  # 6-sided crystal
        gaussian *= facet_modulation
        
        Z += gaussian
    
    # Add interference patterns for holographic effect
    interference = 0.2 * np.sin(2 * np.pi * X / 2) * np.cos(2 * np.pi * Y / 2)
    Z += interference
    
    # Apply smoothing with preserved peaks
    Z_smooth = gaussian_filter(Z, sigma=0.8)
    Z = 0.7 * Z + 0.3 * Z_smooth
    
    # Normalize and enhance contrast
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    Z = np.power(Z, 0.8)  # Enhance peaks
    
    # Create stunning rainbow metallic colormap
    colors_rainbow = ['#000033', '#1a0066', '#4B0082', '#9370DB', '#FF00FF', 
                     '#FF1493', '#FF69B4', '#FFB6C1', '#FF4500', '#FF8C00',
                     '#FFD700', '#FFFF00', '#ADFF2F', '#00FF00', '#00CED1',
                     '#00BFFF', '#0000FF', '#FF00FF']
    n_bins = 256
    cmap_crystal = LinearSegmentedColormap.from_list('crystal', colors_rainbow, N=n_bins)
    
    # Create multiple light sources for dramatic effect
    ls1 = LightSource(azdeg=315, altdeg=45)
    ls2 = LightSource(azdeg=135, altdeg=30)
    
    # Compute shading from both light sources
    rgb1 = ls1.shade(Z, cmap=cmap_crystal, vert_exag=0.15, blend_mode='overlay')
    rgb2 = ls2.shade(Z, cmap=cmap_crystal, vert_exag=0.1, blend_mode='soft')
    rgb = 0.6 * rgb1 + 0.4 * rgb2  # Blend both light sources
    
    # Plot the crystalline surface
    surf = ax.plot_surface(X, Y, Z, facecolors=rgb, 
                           linewidth=0, antialiased=True, 
                           alpha=0.9, rcount=200, ccount=200)
    
    # Add glowing energy orbs at peaks
    for px, py, height, sx, sy, color, label in peaks:
        z_peak = height / 3.5  # Normalized peak height
        
        # Multiple transparent spheres for glow effect
        for radius_factor, alpha in [(1.5, 0.1), (1.0, 0.2), (0.5, 0.4)]:
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 20)
            sphere_x = 0.3 * radius_factor * np.outer(np.cos(u), np.sin(v)) + px
            sphere_y = 0.3 * radius_factor * np.outer(np.ones(np.size(u)), np.cos(v)) + py
            sphere_z = 0.15 * radius_factor * np.outer(np.ones(np.size(u)), np.sin(v)) + z_peak
            
            ax.plot_surface(sphere_x, sphere_y, sphere_z, 
                           color=color, alpha=alpha,
                           shade=True, linewidth=0, antialiased=True)
        
        # Add floating label with glow effect
        ax.text(px, py, z_peak + 0.3, label,
               fontsize=10, fontweight='bold', ha='center',
               color='white',
               bbox=dict(boxstyle='round,pad=0.4', 
                        facecolor=color, alpha=0.8,
                        edgecolor='white', linewidth=2))
    
    # Add particle system - glowing dots flowing between peaks
    n_particles = 500
    particles_x = np.random.uniform(0, 10, n_particles)
    particles_y = np.random.uniform(0, 10, n_particles)
    particles_z = np.random.uniform(0, 1, n_particles)
    particle_colors = np.random.choice(NEON_COLORS, n_particles)
    particle_sizes = np.random.uniform(1, 20, n_particles)
    
    ax.scatter(particles_x, particles_y, particles_z,
              c=particle_colors, s=particle_sizes, 
              alpha=0.6, edgecolors='none', marker='.')
    
    # Add holographic grid planes at multiple levels
    grid_levels = [0.2, 0.4, 0.6, 0.8]
    for level in grid_levels:
        xx, yy = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 10, 20))
        zz = np.ones_like(xx) * level
        ax.plot_wireframe(xx, yy, zz, color='cyan', alpha=0.1, linewidth=0.5)
    
    # Add energy connections between strong attention pairs
    connections = [(2, 3, 3, 7), (5, 5, 8, 8), (1, 8, 7, 2)]
    for x1, y1, x2, y2 in connections:
        # Create bezier curve for smooth connection
        t = np.linspace(0, 1, 50)
        # Control point for curve
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cz = 0.8  # Height of arc
        
        # Bezier curve points
        bx = (1-t)**2 * x1 + 2*(1-t)*t * cx + t**2 * x2
        by = (1-t)**2 * y1 + 2*(1-t)*t * cy + t**2 * y2
        bz = (1-t)**2 * 0.5 + 2*(1-t)*t * cz + t**2 * 0.5
        
        # Plot with gradient effect
        for i in range(len(t)-1):
            alpha = 0.8 * (1 - abs(i - 25)/25)  # Fade at ends
            ax.plot(bx[i:i+2], by[i:i+2], bz[i:i+2],
                   color='yellow', alpha=alpha, linewidth=3)
    
    # Add dramatic contour lines
    contour_levels = np.linspace(0, Z.max(), 15)
    contours = ax.contour(X, Y, Z, levels=contour_levels,
                         colors='white', alpha=0.2, linewidths=0.5,
                         offset=0, zdir='z')
    
    # Contour projections on walls with neon colors
    ax.contour(X, Y, Z, levels=8, colors='cyan', alpha=0.3,
              linewidths=1, offset=10, zdir='y')
    ax.contour(X, Y, Z, levels=8, colors='magenta', alpha=0.3,
              linewidths=1, offset=10, zdir='x')
    
    # Labels and styling
    ax.set_xlabel('Query Token Position', fontsize=14, fontweight='bold', 
                 labelpad=15, color='#00FFFF')
    ax.set_ylabel('Key Token Position', fontsize=14, fontweight='bold', 
                 labelpad=15, color='#FF00FF')
    ax.set_zlabel('Attention Intensity', fontsize=14, fontweight='bold', 
                 labelpad=15, color='#FFFF00')
    
    ax.set_title('CRYSTALLINE ATTENTION LANDSCAPE\nHow Transformers Perceive Language Relationships',
                fontsize=18, fontweight='bold', pad=25, color='white')
    
    # Set optimal viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Set axis limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 1.2)
    
    # Style the plot background
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor('#000033')
    ax.yaxis.pane.set_facecolor('#000033')
    ax.zaxis.pane.set_facecolor('#000033')
    ax.xaxis.pane.set_alpha(0.9)
    ax.yaxis.pane.set_alpha(0.9)
    ax.zaxis.pane.set_alpha(0.9)
    
    # Grid styling
    ax.grid(True, alpha=0.2, linestyle='--', color='cyan')
    
    # Set dark figure background
    fig.patch.set_facecolor('#000022')
    ax.set_facecolor('#000022')
    
    plt.tight_layout()
    plt.savefig('../figures/attention_revolution_3d.pdf', dpi=300, bbox_inches='tight',
                facecolor='#000022', edgecolor='none')
    plt.close()

def create_embedding_space_clustering():
    """Create a stunning galaxy-style K-means clustering visualization"""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate complex embedding data with many clusters
    np.random.seed(42)
    n_points_per_cluster = 80
    
    # Define 12 semantic clusters with galactic theme
    clusters = {
        'Animals': {'center': [3, 2, 2], 'cov': 0.4, 'color': '#FF00FF', 'particles': 100},
        'Technology': {'center': [-3, 2, -1], 'cov': 0.5, 'color': '#00FFFF', 'particles': 120},
        'Emotions': {'center': [2, -3, 0], 'cov': 0.35, 'color': '#FFFF00', 'particles': 90},
        'Actions': {'center': [-2, -2, 2], 'cov': 0.4, 'color': '#FF1493', 'particles': 110},
        'Objects': {'center': [0, 0, -2.5], 'cov': 0.45, 'color': '#00CED1', 'particles': 95},
        'Places': {'center': [0, 3.5, 0], 'cov': 0.4, 'color': '#32CD32', 'particles': 105},
        'Concepts': {'center': [-3, -3, -2], 'cov': 0.5, 'color': '#FFD700', 'particles': 115},
        'People': {'center': [3, -1, -1], 'cov': 0.35, 'color': '#FF69B4', 'particles': 85},
        'Time': {'center': [-1, 1, 3], 'cov': 0.3, 'color': '#9370DB', 'particles': 75},
        'Nature': {'center': [2, 3, -1], 'cov': 0.4, 'color': '#FF4500', 'particles': 100},
        'Science': {'center': [-3, 0, 1], 'cov': 0.45, 'color': '#00FA9A', 'particles': 95},
        'Art': {'center': [1, -2, 2.5], 'cov': 0.35, 'color': '#4169E1', 'particles': 88}
    }
    
    # Create nebula-like background density field
    x_range = np.linspace(-5, 5, 60)
    y_range = np.linspace(-5, 5, 60)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    # Calculate combined density field for nebula effect
    Z_density = np.zeros_like(X_grid)
    for cluster_name, params in clusters.items():
        rv = multivariate_normal(params['center'][:2], 
                                 [[params['cov']*2, 0], [0, params['cov']*2]])
        pos = np.dstack((X_grid, Y_grid))
        Z_density += rv.pdf(pos) * 0.5
    
    # Apply smoothing for nebula effect
    Z_density = gaussian_filter(Z_density, sigma=2)
    
    # Create nebula background with gradient colors
    nebula_colors = plt.cm.twilight(Z_density / Z_density.max())
    nebula_colors[..., 3] = Z_density / Z_density.max() * 0.3  # Transparency
    
    # Plot nebula cloud at bottom
    ax.plot_surface(X_grid, Y_grid, np.ones_like(X_grid) * -3.5,
                   facecolors=nebula_colors,
                   alpha=0.6, linewidth=0, antialiased=True, shade=False)
    
    # Generate main cluster points
    all_points = []
    all_colors = []
    all_sizes = []
    
    for cluster_name, params in clusters.items():
        # Generate main cluster points
        cov_matrix = np.eye(3) * params['cov']
        # Add some correlation for more interesting shapes
        cov_matrix[0, 1] = params['cov'] * 0.2
        cov_matrix[1, 0] = params['cov'] * 0.2
        cov_matrix[1, 2] = params['cov'] * 0.15
        cov_matrix[2, 1] = params['cov'] * 0.15
        
        points = np.random.multivariate_normal(params['center'], cov_matrix, n_points_per_cluster)
        
        # Calculate distance from center for size variation
        center = np.array(params['center'])
        distances = np.linalg.norm(points - center, axis=1)
        sizes = 150 * np.exp(-distances / 1.5)  # Larger points near center
        
        # Plot main cluster points with glow effect
        for i, point in enumerate(points):
            # Multiple layers for glow effect
            for glow_size, glow_alpha in [(sizes[i]*3, 0.05), (sizes[i]*2, 0.1), (sizes[i], 0.3)]:
                ax.scatter(point[0], point[1], point[2],
                          c=params['color'], s=glow_size,
                          alpha=glow_alpha, edgecolors='none')
        
        all_points.append(points)
        
        # Add stardust particle cloud around cluster
        n_particles = params['particles']
        particle_cov = np.eye(3) * params['cov'] * 2  # Wider spread for particles
        particles = np.random.multivariate_normal(params['center'], particle_cov, n_particles)
        
        ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2],
                  c=params['color'], s=np.random.uniform(0.5, 5, n_particles),
                  alpha=0.4, edgecolors='none', marker='.')
        
        # Add glowing plasma core at cluster center
        center = params['center']
        
        # Multiple concentric spheres for plasma glow
        for radius, alpha, emissive in [(0.6, 0.2, 1.5), (0.4, 0.3, 1.2), (0.2, 0.5, 1.0)]:
            u = np.linspace(0, 2 * np.pi, 40)
            v = np.linspace(0, np.pi, 30)
            x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[1]
            z = radius * np.outer(np.ones(np.size(u)), np.sin(v)) + center[2]
            
            # Create color array with emissive glow
            color_rgba = plt.cm.colors.hex2color(params['color'])
            color_rgba = tuple([min(1.0, c * emissive) for c in color_rgba])
            
            ax.plot_surface(x, y, z, color=color_rgba,
                           alpha=alpha, shade=True,
                           linewidth=0, antialiased=True)
        
        # Add holographic floating label
        ax.text(center[0], center[1], center[2] + 1.0, cluster_name.upper(),
               fontsize=12, fontweight='bold', ha='center', color='white',
               bbox=dict(boxstyle='round,pad=0.5',
                        facecolor=params['color'], alpha=0.9,
                        edgecolor='white', linewidth=2))
    
    # Add spiral galaxy arms connecting related clusters
    connections = [
        ('Animals', 'Nature', 0.4, 20),
        ('Technology', 'Science', 0.5, 25),
        ('Emotions', 'People', 0.4, 22),
        ('Places', 'Nature', 0.3, 18),
        ('Art', 'Emotions', 0.4, 20),
        ('Concepts', 'Science', 0.3, 15),
        ('Time', 'Places', 0.3, 17),
        ('Objects', 'Technology', 0.4, 23)
    ]
    
    for cluster1, cluster2, alpha, n_points in connections:
        p1 = np.array(clusters[cluster1]['center'])
        p2 = np.array(clusters[cluster2]['center'])
        
        # Create spiral path
        t = np.linspace(0, 1, n_points)
        spiral_angle = t * 2 * np.pi
        
        # Bezier curve with spiral
        control = (p1 + p2) / 2 + np.array([0, 0, 1])
        base_curve = np.array([(1-ti)**2 * p1 + 2*(1-ti)*ti * control + ti**2 * p2 for ti in t])
        
        # Add spiral motion
        spiral_offset = np.column_stack([
            0.2 * np.sin(spiral_angle),
            0.2 * np.cos(spiral_angle),
            np.zeros_like(spiral_angle)
        ])
        spiral_curve = base_curve + spiral_offset
        
        # Plot with comet trail effect
        for i in range(len(t)-1):
            segment_alpha = alpha * (1 - i/len(t)) * 0.8  # Fade trail
            segment_width = 3 * (1 - i/len(t)) + 1  # Taper width
            
            ax.plot(spiral_curve[i:i+2, 0], 
                   spiral_curve[i:i+2, 1],
                   spiral_curve[i:i+2, 2],
                   color='white', alpha=segment_alpha, 
                   linewidth=segment_width)
            
            # Add glow particles along the path
            if i % 3 == 0:
                ax.scatter(spiral_curve[i, 0], spiral_curve[i, 1], spiral_curve[i, 2],
                          c='white', s=10, alpha=0.6, edgecolors='none')
    
    # Add 3D Voronoi cells as crystalline boundaries (simplified projection)
    # Project cluster centers to 2D for Voronoi
    centers_2d = np.array([c['center'][:2] for c in clusters.values()])
    
    # Add boundary points to create finite Voronoi regions
    boundary_points = []
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        boundary_points.append([10*np.cos(angle), 10*np.sin(angle)])
    
    all_points_2d = np.vstack([centers_2d, boundary_points])
    vor = Voronoi(all_points_2d)
    
    # Plot Voronoi edges as holographic boundaries
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            points = vor.vertices[simplex]
            if np.all(np.abs(points) < 6):  # Only plot edges within view
                # Plot at multiple Z levels for 3D effect
                for z_level in [-3, -2, -1, 0, 1, 2]:
                    ax.plot(points[:, 0], points[:, 1], [z_level, z_level],
                           'cyan', alpha=0.1, linewidth=0.5)
    
    # Add aurora borealis effect in background
    n_aurora = 200
    aurora_x = np.random.uniform(-5, 5, n_aurora)
    aurora_y = np.random.uniform(-5, 5, n_aurora)
    aurora_z = np.random.uniform(2, 3, n_aurora)
    aurora_colors = np.random.choice(['#00FF00', '#00FFFF', '#FF00FF', '#FFFF00'], n_aurora)
    
    ax.scatter(aurora_x, aurora_y, aurora_z,
              c=aurora_colors, s=np.random.uniform(1, 30, n_aurora),
              alpha=0.2, edgecolors='none', marker='o')
    
    # Styling
    ax.set_xlabel('Semantic Dimension', fontsize=14, fontweight='bold',
                 labelpad=15, color='#00FFFF')
    ax.set_ylabel('Syntactic Dimension', fontsize=14, fontweight='bold',
                 labelpad=15, color='#FF00FF')
    ax.set_zlabel('Abstract Dimension', fontsize=14, fontweight='bold',
                 labelpad=15, color='#FFFF00')
    
    ax.set_title('TRANSFORMER EMBEDDING GALAXY\nSemantic Clusters in Hyperspace',
                fontsize=18, fontweight='bold', pad=25, color='white')
    
    # Set viewing angle
    ax.view_init(elev=25, azim=55)
    
    # Set limits
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-4, 4)
    
    # Dark space background
    ax.xaxis.pane.set_facecolor('#000011')
    ax.yaxis.pane.set_facecolor('#000011')
    ax.zaxis.pane.set_facecolor('#000011')
    ax.xaxis.pane.set_alpha(0.9)
    ax.yaxis.pane.set_alpha(0.9)
    ax.zaxis.pane.set_alpha(0.9)
    
    # Grid styling
    ax.grid(True, alpha=0.1, linestyle='--', color='cyan')
    
    # Set dark figure background
    fig.patch.set_facecolor('#000011')
    ax.set_facecolor('#000011')
    
    # Add legend with cluster information
    legend_elements = [mpatches.Patch(facecolor=params['color'],
                                     edgecolor='white', linewidth=2,
                                     label=name, alpha=0.9)
                      for name, params in list(clusters.items())[:8]]
    ax.legend(handles=legend_elements, loc='upper left',
             bbox_to_anchor=(0.02, 0.98), fontsize=9,
             framealpha=0.8, edgecolor='white', facecolor='#000033')
    
    # Add information panel
    info_text = (
        "FEATURES\n"
        "* Plasma cores: Cluster centroids\n"
        "* Stardust: Token embeddings\n"
        "* Spiral arms: Semantic connections\n"
        "* Aurora: Attention flow\n"
        "* Voronoi cells: Semantic boundaries"
    )
    ax.text2D(0.98, 0.02, info_text,
             transform=ax.transAxes, fontsize=10,
             ha='right', va='bottom', color='white',
             bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='#000066', alpha=0.8,
                      edgecolor='cyan', linewidth=2))
    
    plt.tight_layout()
    plt.savefig('../figures/paradigm_shift_3d.pdf', dpi=300, bbox_inches='tight',
                facecolor='#000011', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    print("Generating ULTRA-VIBRANT K-means Style 3D Visualizations for Week 5...")
    
    # Generate the visualizations
    print("Creating crystalline attention landscape...")
    create_attention_landscape()
    print("Created: attention_revolution_3d.pdf")
    
    print("Creating embedding galaxy visualization...")
    create_embedding_space_clustering()
    print("Created: paradigm_shift_3d.pdf")
    
    print("\nAll stunning visualizations generated successfully!")