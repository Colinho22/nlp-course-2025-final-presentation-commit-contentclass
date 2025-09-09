"""
Generate enhanced visualizations for Week 3: RNNs, LSTMs, and GRUs
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set consistent style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Educational color scheme
COLOR_CURRENT = '#FF6B6B'  # Red - current position/focus
COLOR_CONTEXT = '#4ECDC4'  # Teal - context/surrounding  
COLOR_PREDICT = '#95E77E'  # Green - predictions/output
COLOR_NEUTRAL = '#E0E0E0'  # Gray - neutral elements
COLOR_MEMORY = '#FFE66D'    # Yellow - memory/hidden state
COLOR_GRADIENT = '#A8E6CF' # Light green - gradient flow

def create_rnn_evolution_timeline():
    """Create timeline showing evolution from n-grams to Transformers"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Timeline data
    years = [1980, 1986, 1997, 2014, 2017, 2024]
    models = ['N-grams', 'RNN', 'LSTM', 'GRU', 'Transformers', 'Current']
    descriptions = [
        'Count-based\nNo memory',
        'Hidden states\nShort memory', 
        'Gated memory\nLong sequences',
        'Simplified gates\nFaster training',
        'Attention only\nParallel processing',
        'Hybrid models\nEfficient & powerful'
    ]
    
    # Main timeline
    ax.plot(years, [0]*len(years), 'k-', linewidth=3)
    
    # Add points and labels
    for i, (year, model, desc) in enumerate(zip(years, models, descriptions)):
        # Milestone point
        ax.scatter(year, 0, s=200, c=COLOR_CURRENT if model == 'RNN' else COLOR_CONTEXT, 
                  zorder=5, edgecolors='black', linewidth=2)
        
        # Model name above
        ax.text(year, 0.3, model, ha='center', fontsize=12, fontweight='bold')
        
        # Description below
        ax.text(year, -0.3, desc, ha='center', fontsize=9, style='italic', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Year label
        ax.text(year, -0.6, str(year), ha='center', fontsize=10)
    
    # Add arrows showing progression
    for i in range(len(years)-1):
        ax.annotate('', xy=(years[i+1]-2, 0), xytext=(years[i]+2, 0),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.5))
    
    # Highlight RNN family
    rect = Rectangle((1985, -0.8), 2018-1985, 1.6, 
                    facecolor=COLOR_MEMORY, alpha=0.2, 
                    edgecolor=COLOR_MEMORY, linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.text(2000, 0.65, 'RNN Family Era', fontsize=11, fontweight='bold', 
           color=COLOR_MEMORY, ha='center')
    
    # Key innovations annotations
    ax.annotate('Memory introduced', xy=(1986, 0), xytext=(1986, -1.2),
               arrowprops=dict(arrowstyle='->', color='red'),
               fontsize=9, ha='center')
    
    ax.annotate('Gates solve\nvanishing gradient', xy=(1997, 0), xytext=(1997, 1),
               arrowprops=dict(arrowstyle='->', color='red'),
               fontsize=9, ha='center')
    
    ax.set_xlim(1975, 2025)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_title('Evolution of Sequential Models in NLP', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('../figures/rnn_evolution_timeline.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_hidden_state_flow():
    """Visualize how hidden states update step by step"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sentence to process
    words = ['The', 'cat', 'sat', 'on', 'mat']
    
    # Positions
    x_positions = np.linspace(1, 9, len(words))
    y_input = 3
    y_hidden = 1.5
    y_output = 0
    
    # Draw RNN cells
    for i, (x, word) in enumerate(zip(x_positions, words)):
        # RNN cell
        rect = FancyBboxPatch((x-0.3, y_hidden-0.3), 0.6, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor=COLOR_CONTEXT, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y_hidden, 'RNN', ha='center', va='center', fontweight='bold')
        
        # Input word
        ax.text(x, y_input, word, ha='center', va='center', 
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=COLOR_CURRENT, alpha=0.7))
        
        # Arrow from input to RNN
        ax.arrow(x, y_input-0.2, 0, -0.8, head_width=0.1, head_length=0.1, 
                fc='black', ec='black')
        
        # Output
        ax.text(x, y_output, f'y_{i+1}', ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor=COLOR_PREDICT, alpha=0.7))
        
        # Arrow from RNN to output
        ax.arrow(x, y_hidden-0.35, 0, -0.8, head_width=0.1, head_length=0.1,
                fc='gray', ec='gray')
        
        # Hidden state flow
        if i < len(words)-1:
            ax.arrow(x+0.35, y_hidden, x_positions[i+1]-x-0.7, 0,
                    head_width=0.15, head_length=0.15, 
                    fc=COLOR_MEMORY, ec='black', linewidth=2)
            
            # Hidden state values (example)
            h_values = [
                '[0.2, -0.1, 0.3]',
                '[0.3, 0.5, -0.2]', 
                '[0.1, 0.8, 0.4]',
                '[-0.2, 0.6, 0.7]'
            ]
            if i < len(h_values):
                ax.text((x + x_positions[i+1])/2, y_hidden+0.4, f'h_{i+1}',
                       ha='center', fontsize=10, fontweight='bold')
                ax.text((x + x_positions[i+1])/2, y_hidden-0.5, h_values[i],
                       ha='center', fontsize=8, style='italic')
    
    # Initial hidden state
    ax.text(0.3, y_hidden, 'h_0 = [0,0,0]', ha='center', va='center',
           fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor=COLOR_MEMORY, alpha=0.7))
    ax.arrow(0.6, y_hidden, 0.3, 0, head_width=0.15, head_length=0.15,
            fc=COLOR_MEMORY, ec='black', linewidth=2)
    
    # Title and labels
    ax.text(5, 4, 'Processing: "The cat sat on mat"', ha='center', 
           fontsize=14, fontweight='bold')
    ax.text(-0.5, y_input, 'Input:', ha='right', fontsize=11, fontweight='bold')
    ax.text(-0.5, y_hidden, 'Hidden:', ha='right', fontsize=11, fontweight='bold')
    ax.text(-0.5, y_output, 'Output:', ha='right', fontsize=11, fontweight='bold')
    
    # Legend
    ax.text(10, y_hidden, 'Hidden state\naccumulates\ninformation', 
           ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')
    ax.set_title('RNN Hidden State Flow: Step-by-Step Memory Updates', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../figures/hidden_state_flow.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_gradient_vanishing_visualization():
    """Show how gradients decay over time steps"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top plot: Gradient magnitude over time
    time_steps = np.arange(1, 31)
    gradient_09 = 0.9 ** time_steps
    gradient_08 = 0.8 ** time_steps
    gradient_10 = 1.0 ** time_steps  # LSTM case
    
    ax1.semilogy(time_steps, gradient_09, 'b-', linewidth=3, label='RNN (factor=0.9)')
    ax1.semilogy(time_steps, gradient_08, 'r-', linewidth=3, label='RNN (factor=0.8)')
    ax1.semilogy(time_steps, gradient_10, 'g-', linewidth=3, label='LSTM (factor≈1.0)')
    ax1.axhline(y=0.01, color='black', linestyle='--', alpha=0.5, label='Effective zero')
    
    ax1.fill_between(time_steps, 0, 0.01, alpha=0.2, color='red', 
                     label='Vanishing zone')
    
    ax1.set_xlabel('Time Steps', fontsize=12)
    ax1.set_ylabel('Gradient Magnitude (log scale)', fontsize=12)
    ax1.set_title('Vanishing Gradient Problem in RNNs', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.001, 2])
    
    # Bottom plot: Visual representation of gradient flow
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 3)
    
    # RNN path with vanishing gradient
    y_rnn = 2
    for i in range(10):
        alpha = gradient_09[i] if i < len(gradient_09) else 0
        color = (1, 0, 0, max(alpha, 0.1))  # Red with fading alpha
        
        circle = Circle((i+0.5, y_rnn), 0.3, facecolor=color, edgecolor='black')
        ax2.add_patch(circle)
        ax2.text(i+0.5, y_rnn, f't_{i}', ha='center', va='center', fontsize=9)
        
        if i < 9:
            arrow_alpha = max(alpha, 0.1)
            ax2.arrow(i+0.8, y_rnn, 0.4, 0, head_width=0.1, head_length=0.1,
                     fc=(1,0,0,arrow_alpha), ec='black', linewidth=1)
    
    ax2.text(5, y_rnn+0.6, 'RNN: Gradient vanishes', ha='center', fontsize=11, 
            fontweight='bold', color='red')
    
    # LSTM path with preserved gradient
    y_lstm = 0.8
    for i in range(10):
        circle = Circle((i+0.5, y_lstm), 0.3, facecolor=COLOR_PREDICT, 
                       edgecolor='black', alpha=0.9)
        ax2.add_patch(circle)
        ax2.text(i+0.5, y_lstm, f't_{i}', ha='center', va='center', fontsize=9)
        
        if i < 9:
            ax2.arrow(i+0.8, y_lstm, 0.4, 0, head_width=0.1, head_length=0.1,
                     fc='green', ec='black', linewidth=2)
    
    # Highway for LSTM
    ax2.plot([0.5, 9.5], [y_lstm-0.4, y_lstm-0.4], 'g-', linewidth=3, alpha=0.5)
    ax2.text(5, y_lstm-0.6, 'Gradient Highway', ha='center', fontsize=9, 
            style='italic', color='green')
    
    ax2.text(5, y_lstm+0.6, 'LSTM: Gradient preserved', ha='center', fontsize=11,
            fontweight='bold', color='green')
    
    ax2.axis('off')
    ax2.set_title('Gradient Flow Comparison: RNN vs LSTM', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../figures/gradient_vanishing_animated.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_lstm_gates_intuitive():
    """Create intuitive visualization of LSTM gates"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Central memory cell
    cell_x, cell_y = 5, 5
    cell_width, cell_height = 3, 2
    
    # Draw cell state highway
    ax.plot([1, 9], [cell_y, cell_y], 'g-', linewidth=5, alpha=0.3)
    ax.text(5, cell_y+2.5, 'Cell State Highway (Long-term Memory)', 
           ha='center', fontsize=12, fontweight='bold', color='green')
    
    # Draw main cell
    cell = FancyBboxPatch((cell_x-cell_width/2, cell_y-cell_height/2), 
                          cell_width, cell_height,
                          boxstyle="round,pad=0.1",
                          facecolor=COLOR_MEMORY, edgecolor='black', linewidth=3)
    ax.add_patch(cell)
    ax.text(cell_x, cell_y, 'Memory\nCell', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Gates with intuitive icons
    gates = [
        {'name': 'Forget Gate', 'x': 2, 'y': cell_y, 'color': '#FF6B6B', 
         'icon': '🗑', 'desc': 'Delete\nirrelevant\nmemory', 'value': '0.1 = forget 90%'},
        {'name': 'Input Gate', 'x': cell_x, 'y': 2, 'color': '#4ECDC4',
         'icon': '💾', 'desc': 'Save new\nimportant\ninfo', 'value': '0.8 = save 80%'},
        {'name': 'Output Gate', 'x': 8, 'y': cell_y, 'color': '#95E77E',
         'icon': '📤', 'desc': 'Output\nrelevant\ninfo now', 'value': '0.7 = use 70%'}
    ]
    
    for gate in gates:
        # Gate box
        gate_box = FancyBboxPatch((gate['x']-0.7, gate['y']-0.5), 1.4, 1,
                                  boxstyle="round,pad=0.05",
                                  facecolor=gate['color'], alpha=0.7, 
                                  edgecolor='black', linewidth=2)
        ax.add_patch(gate_box)
        
        # Gate name
        ax.text(gate['x'], gate['y']+0.8, gate['name'], ha='center', 
               fontsize=11, fontweight='bold')
        
        # Gate function
        ax.text(gate['x'], gate['y'], gate['desc'], ha='center', va='center',
               fontsize=9)
        
        # Example value
        ax.text(gate['x'], gate['y']-0.8, gate['value'], ha='center',
               fontsize=8, style='italic')
        
        # Arrows to/from cell
        if gate['name'] == 'Forget Gate':
            ax.arrow(gate['x']+0.7, gate['y'], 0.8, 0, head_width=0.2, 
                    head_length=0.2, fc=gate['color'], ec='black')
        elif gate['name'] == 'Input Gate':
            ax.arrow(gate['x'], gate['y']+0.5, 0, 1.3, head_width=0.2,
                    head_length=0.2, fc=gate['color'], ec='black')
        else:
            ax.arrow(cell_x+1.5, cell_y, 0.8, 0, head_width=0.2,
                    head_length=0.2, fc=gate['color'], ec='black')
    
    # Input and output
    ax.text(1, 2, 'Input\n(x_t)', ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=COLOR_CURRENT, alpha=0.7))
    ax.arrow(1.5, 2, 2.5, 0, head_width=0.15, head_length=0.15, 
            fc='black', ec='black', linestyle='--')
    
    ax.text(9, 2, 'Output\n(h_t)', ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=COLOR_PREDICT, alpha=0.7))
    
    # Previous hidden state
    ax.text(1, cell_y-1.5, 'Previous\nHidden\n(h_{t-1})', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor=COLOR_CONTEXT, alpha=0.5))
    ax.arrow(1.5, cell_y-1.5, 2.5, 0, head_width=0.15, head_length=0.15,
            fc='gray', ec='gray', linestyle='--')
    
    # Email analogy
    ax.text(5, 0.5, 'Email Inbox Analogy:', fontsize=11, fontweight='bold')
    ax.text(5, 0, 'Forget = Delete spam | Input = Save important | Output = Reply now',
           ha='center', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 8)
    ax.axis('off')
    ax.set_title('LSTM Gates: Controlling Information Flow', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../figures/lstm_gates_intuitive.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_rnn_applications_2024():
    """Current real-world applications of RNNs"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    applications = {
        'Speech Recognition': {
            'examples': ['Google RNN-T', 'Siri on-device', 'Alexa'],
            'advantage': 'Streaming processing',
            'color': COLOR_CURRENT
        },
        'Time Series': {
            'examples': ['Stock prediction', 'Weather forecast', 'Energy demand'],
            'advantage': 'Sequential patterns',
            'color': COLOR_CONTEXT
        },
        'Music Generation': {
            'examples': ['MuseNet', 'Magenta', 'MIDI composition'],
            'advantage': 'Temporal dependencies',
            'color': COLOR_PREDICT
        },
        'Healthcare': {
            'examples': ['ECG analysis', 'Patient monitoring', 'Drug discovery'],
            'advantage': 'Real-time processing',
            'color': COLOR_MEMORY
        }
    }
    
    # Create circular layout
    angles = np.linspace(0, 2*np.pi, len(applications)+1)[:-1]
    radius = 3
    
    for i, (app_name, app_info) in enumerate(applications.items()):
        x = radius * np.cos(angles[i])
        y = radius * np.sin(angles[i])
        
        # Application circle
        circle = Circle((x, y), 1.2, facecolor=app_info['color'], 
                       alpha=0.6, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        # Application name
        ax.text(x, y+0.3, app_name, ha='center', fontsize=11, 
               fontweight='bold')
        
        # Examples
        for j, example in enumerate(app_info['examples']):
            ax.text(x, y-0.1-j*0.3, f'• {example}', ha='center', 
                   fontsize=8)
        
        # Advantage
        ax.text(x, y-1.5, app_info['advantage'], ha='center',
               fontsize=9, style='italic', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Center text
    ax.text(0, 0, 'RNNs\nin 2024', ha='center', va='center',
           fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # Title
    ax.text(0, 5, 'RNNs Still Excel in These Domains', ha='center',
           fontsize=16, fontweight='bold')
    
    # Note about transformers
    ax.text(0, -5, 'Note: Despite Transformers, RNNs dominate when:\n' +
                   '• Memory constraints exist (mobile/edge)\n' +
                   '• Streaming/online processing needed\n' +
                   '• Clear sequential patterns present',
           ha='center', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/rnn_applications_2024.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_memory_comparison():
    """Compare how RNN/LSTM/GRU handle long sequences"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    sequence = "The student who studied very hard for the difficult exam"
    words = sequence.split()
    
    models = ['RNN', 'LSTM', 'GRU']
    memory_patterns = [
        [1.0, 0.9, 0.81, 0.73, 0.66, 0.59, 0.53, 0.48, 0.43, 0.39],  # RNN
        [1.0, 0.95, 0.92, 0.90, 0.88, 0.85, 0.83, 0.80, 0.78, 0.76],  # LSTM
        [1.0, 0.93, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73, 0.70, 0.68]   # GRU
    ]
    
    colors = [COLOR_CURRENT, COLOR_PREDICT, COLOR_CONTEXT]
    
    for ax, model, pattern, color in zip(axes, models, memory_patterns, colors):
        # Plot memory retention
        x = np.arange(len(pattern))
        ax.bar(x, pattern, color=color, alpha=0.7, edgecolor='black')
        
        # Add word labels
        for i in range(len(pattern)):
            if i < len(words):
                ax.text(i, -0.1, words[i], rotation=45, ha='right', fontsize=8)
        
        # Highlight key word
        ax.bar(0, pattern[0], color='gold', alpha=0.9, edgecolor='black', linewidth=2)
        ax.text(0, pattern[0]+0.05, 'student', ha='center', fontsize=9, fontweight='bold')
        
        # Threshold line
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.text(9.5, 0.5, 'Effective\nthreshold', ha='right', fontsize=8, color='red')
        
        ax.set_title(f'{model} Memory Retention', fontsize=12, fontweight='bold')
        ax.set_ylabel('Memory Strength', fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.set_xlim([-0.5, len(pattern)-0.5])
        
        # Add model characteristics
        if model == 'RNN':
            ax.text(5, 0.85, 'Rapid decay\nForgets "student"', ha='center',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
        elif model == 'LSTM':
            ax.text(5, 0.85, 'Gradual decay\nRemembers well', ha='center',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))
        else:
            ax.text(5, 0.85, 'Moderate decay\nGood balance', ha='center',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='blue', alpha=0.2))
    
    plt.suptitle('Memory Retention Across Different Architectures', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/memory_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Week 3 Enhanced Visualizations...")
    
    print("1. Creating RNN evolution timeline...")
    create_rnn_evolution_timeline()
    
    print("2. Creating hidden state flow visualization...")
    create_hidden_state_flow()
    
    print("3. Creating gradient vanishing visualization...")
    create_gradient_vanishing_visualization()
    
    print("4. Creating LSTM gates intuitive diagram...")
    create_lstm_gates_intuitive()
    
    print("5. Creating RNN applications 2024...")
    create_rnn_applications_2024()
    
    print("6. Creating memory comparison chart...")
    create_memory_comparison()
    
    print("All Week 3 visualizations generated successfully!")