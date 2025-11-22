"""
Generate BSc-level figures for Week 3: RNNs
Simple, clear visualizations for introductory students
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define BSc-friendly color palette
COLOR_BLUE = '#2E86C1'        # Primary blue
COLOR_ORANGE = '#FF8800'      # Accent orange
COLOR_GREEN = '#27AE60'       # Success green
COLOR_RED = '#E74C3C'         # Warning red
COLOR_GRAY = '#95A5A6'        # Neutral gray
COLOR_LIGHT = '#ECF0F1'       # Light background
COLOR_PURPLE = '#9B59B6'      # Insight purple

def setup_figure(figsize=(10, 6)):
    """Setup figure with consistent style"""
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax

def save_figure(fig, filename):
    """Save figure with consistent settings"""
    fig.savefig(f'../figures/{filename}',
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()
    print(f"Generated: {filename}")


# 1. Simple RNN Unrolled Diagram
def create_rnn_simple_unrolled():
    """Simple unrolled RNN showing parameter sharing"""
    fig, ax = setup_figure((14, 5))

    # Title
    ax.text(7, 4.5, 'RNN Unrolled Through Time', ha='center', fontsize=16, fontweight='bold')

    # Draw 4 time steps
    time_steps = 4
    cell_width = 2.5
    start_x = 1.5

    for t in range(time_steps):
        x_pos = start_x + t * 3

        # Input arrow
        ax.annotate('', xy=(x_pos, 1.8), xytext=(x_pos, 0.8),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_BLUE))
        ax.text(x_pos, 0.5, f'$x_{{t+{t}}}$', ha='center', fontsize=12, color=COLOR_BLUE, fontweight='bold')

        # RNN cell
        rect = FancyBboxPatch((x_pos - 0.6, 1.8), 1.2, 1.2,
                              boxstyle="round,pad=0.05",
                              facecolor=COLOR_LIGHT,
                              edgecolor=COLOR_BLUE, linewidth=3)
        ax.add_patch(rect)
        ax.text(x_pos, 2.4, 'RNN', ha='center', va='center', fontsize=11, fontweight='bold')

        # Output arrow
        ax.annotate('', xy=(x_pos, 3.5), xytext=(x_pos, 3.0),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_GREEN))
        ax.text(x_pos, 3.7, f'$h_{{t+{t}}}$', ha='center', fontsize=12, color=COLOR_GREEN, fontweight='bold')

        # Hidden state arrow to next cell
        if t < time_steps - 1:
            ax.annotate('', xy=(x_pos + 2.4, 2.4), xytext=(x_pos + 0.6, 2.4),
                       arrowprops=dict(arrowstyle='->', lw=3, color=COLOR_ORANGE))
            ax.text(x_pos + 1.5, 2.7, 'memory', ha='center', fontsize=9,
                   color=COLOR_ORANGE, style='italic')

    # Key insight box
    insight_text = 'Key Idea: Same RNN cell used at each time step\n(Shared parameters: $W_x$, $W_h$, $b$)'
    ax.text(7, 0.1, insight_text, ha='center', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PURPLE, alpha=0.2))

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)

    save_figure(fig, 'rnn_simple_unrolled_bsc.pdf')


# 2. Name Prediction Visual
def create_name_prediction_visual():
    """Character-by-character name prediction example"""
    fig, ax = setup_figure((12, 6))

    # Title
    ax.text(6, 5.5, 'Example: Predicting Names Character by Character', ha='center',
           fontsize=14, fontweight='bold')

    # Input sequence "Joh"
    chars = ['J', 'o', 'h', '?']
    predictions = ['o', 'h', 'n', '']

    for i, (char, pred) in enumerate(zip(chars, predictions)):
        x_pos = 1.5 + i * 2.5

        # Input character box
        if char != '?':
            rect = Rectangle((x_pos - 0.3, 3.5), 0.6, 0.8,
                            facecolor=COLOR_BLUE, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_pos, 3.9, char, ha='center', va='center',
                   fontsize=20, fontweight='bold', color='white')
            ax.text(x_pos, 2.9, 'Input', ha='center', fontsize=9, style='italic')
        else:
            # Question mark for prediction
            ax.text(x_pos, 3.9, char, ha='center', va='center',
                   fontsize=30, fontweight='bold', color=COLOR_RED)
            ax.text(x_pos, 2.9, 'Predict!', ha='center', fontsize=9,
                   style='italic', color=COLOR_RED, fontweight='bold')

        # RNN processing
        if i < 3:
            ax.annotate('', xy=(x_pos, 2.6), xytext=(x_pos, 3.4),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))

            circle = Circle((x_pos, 2.0), 0.35, facecolor=COLOR_LIGHT,
                          edgecolor=COLOR_BLUE, linewidth=2)
            ax.add_patch(circle)
            ax.text(x_pos, 2.0, 'RNN', ha='center', va='center', fontsize=9, fontweight='bold')

            # Output prediction
            ax.annotate('', xy=(x_pos, 1.4), xytext=(x_pos, 1.65),
                       arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_GREEN))

            rect_pred = Rectangle((x_pos - 0.25, 0.7), 0.5, 0.6,
                                 facecolor=COLOR_GREEN, edgecolor='black', linewidth=2)
            ax.add_patch(rect_pred)
            ax.text(x_pos, 1.0, pred, ha='center', va='center',
                   fontsize=16, fontweight='bold', color='white')
            ax.text(x_pos, 0.3, 'Predict', ha='center', fontsize=8, style='italic')

            # Memory arrow
            if i < 2:
                ax.annotate('', xy=(x_pos + 2.0, 2.0), xytext=(x_pos + 0.4, 2.0),
                           arrowprops=dict(arrowstyle='->', lw=2.5, color=COLOR_ORANGE))
                ax.text(x_pos + 1.2, 2.3, 'memory', ha='center', fontsize=8,
                       color=COLOR_ORANGE, style='italic', fontweight='bold')

    # Show final answer
    ax.text(10, 1.0, 'Answer: n', ha='center', fontsize=18,
           fontweight='bold', color=COLOR_GREEN,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_GREEN, alpha=0.2))
    ax.text(10, 0.3, '(completes "John")', ha='center', fontsize=10, style='italic')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    save_figure(fig, 'name_prediction_visual_bsc.pdf')


# 3. Vanishing Gradient Telephone Game
def create_vanishing_gradient_telephone():
    """Telephone game analogy for vanishing gradient"""
    fig, ax = setup_figure((14, 6))

    # Title
    ax.text(7, 5.5, 'The Memory Problem: Telephone Game Analogy', ha='center',
           fontsize=14, fontweight='bold')

    # Draw 6 people in telephone game
    people = 6
    for i in range(people):
        x_pos = 1.5 + i * 2.2

        # Person (circle with face)
        circle = Circle((x_pos, 3.0), 0.4, facecolor=COLOR_BLUE if i == 0 else COLOR_GRAY,
                       edgecolor='black', linewidth=2)
        ax.add_patch(circle)

        # Message clarity
        if i == 0:
            ax.text(x_pos, 3.0, 'A', ha='center', va='center',
                   fontsize=16, fontweight='bold', color='white')
            ax.text(x_pos, 2.0, 'START\n"The cat\nis fluffy"', ha='center', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_GREEN, alpha=0.3))
        elif i == people - 1:
            ax.text(x_pos, 3.0, 'F', ha='center', va='center',
                   fontsize=16, fontweight='bold', color='white')
            ax.text(x_pos, 2.0, 'END\n"The...uh\n...fluffy?"', ha='center', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_RED, alpha=0.3))
        else:
            ax.text(x_pos, 3.0, chr(65 + i), ha='center', va='center',
                   fontsize=14, color='white')

        # Whisper arrow
        if i < people - 1:
            ax.annotate('', xy=(x_pos + 1.6, 3.0), xytext=(x_pos + 0.5, 3.0),
                       arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_ORANGE,
                                     alpha=1.0 - i * 0.15))
            ax.text(x_pos + 1.05, 3.4, 'whisper', ha='center', fontsize=7,
                   style='italic', color=COLOR_ORANGE, alpha=1.0 - i * 0.15)

    # Gradient strength visualization
    ax.text(7, 1.2, 'Information Loss Over Time Steps', ha='center',
           fontsize=11, fontweight='bold')

    # Gradient bars
    x_start = 2
    bar_width = 1.5
    for i in range(people):
        height = max(0.15, 0.8 * (0.65 ** i))  # Exponential decay
        x_pos = x_start + i * 2.2

        rect = Rectangle((x_pos - bar_width/2, 0.1), bar_width, height,
                        facecolor=COLOR_GREEN if i == 0 else
                                 (COLOR_ORANGE if height > 0.4 else COLOR_RED),
                        edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x_pos, 0.05, f'{int(height*100)}%', ha='center', fontsize=8)

    # Problem statement
    problem_text = ('Problem: Earlier information gets WEAKER as it travels through time\n'
                   'Solution: LSTM uses gates to preserve important information')
    ax.text(7, 4.8, problem_text, ha='center', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PURPLE, alpha=0.2))

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)

    save_figure(fig, 'vanishing_gradient_telephone_bsc.pdf')


# 4. LSTM Gates Simple Diagram
def create_lstm_gates_simple():
    """Simplified LSTM gate explanation"""
    fig, ax = setup_figure((12, 7))

    # Title
    ax.text(6, 6.5, 'LSTM Solution: Three Smart Gates', ha='center',
           fontsize=14, fontweight='bold')

    # Three gates side by side
    gates = [
        {'name': 'Forget Gate', 'color': COLOR_RED, 'desc': 'What to\nforget?',
         'example': 'Forget old topic\nwhen new sentence\nstarts'},
        {'name': 'Input Gate', 'color': COLOR_BLUE, 'desc': 'What to\nremember?',
         'example': 'Remember current\nsubject of\nsentence'},
        {'name': 'Output Gate', 'color': COLOR_GREEN, 'desc': 'What to\noutput?',
         'example': 'Output info needed\nfor next word\nprediction'}
    ]

    for i, gate in enumerate(gates):
        x_pos = 2 + i * 4

        # Gate box
        rect = FancyBboxPatch((x_pos - 1.2, 3.8), 2.4, 2.0,
                              boxstyle="round,pad=0.1",
                              facecolor=gate['color'], alpha=0.2,
                              edgecolor=gate['color'], linewidth=3)
        ax.add_patch(rect)

        # Gate name
        ax.text(x_pos, 5.5, gate['name'], ha='center', fontsize=12,
               fontweight='bold', color=gate['color'])

        # Description
        ax.text(x_pos, 4.7, gate['desc'], ha='center', fontsize=10,
               fontweight='bold')

        # Example box
        example_box = FancyBboxPatch((x_pos - 1.1, 2.5), 2.2, 1.0,
                                     boxstyle="round,pad=0.08",
                                     facecolor='white',
                                     edgecolor=gate['color'], linewidth=2)
        ax.add_patch(example_box)
        ax.text(x_pos, 3.0, gate['example'], ha='center', fontsize=8,
               va='center', style='italic')

    # Memory cell visualization
    ax.text(6, 2.0, 'Memory Cell (Cell State)', ha='center', fontsize=11,
           fontweight='bold')

    # Cell state line
    ax.plot([1, 11], [1.5, 1.5], linewidth=8, color=COLOR_ORANGE, alpha=0.5)
    ax.text(1, 1.2, '$C_{t-1}$', ha='center', fontsize=10, color=COLOR_ORANGE, fontweight='bold')
    ax.text(11, 1.2, '$C_t$', ha='center', fontsize=10, color=COLOR_ORANGE, fontweight='bold')

    # Gate interactions with cell state
    for i, gate in enumerate(gates):
        x_pos = 2 + i * 4
        ax.annotate('', xy=(x_pos, 1.5), xytext=(x_pos, 2.4),
                   arrowprops=dict(arrowstyle='->', lw=2, color=gate['color']))

    # Key insight
    insight = 'Key: Gates control information flow to solve vanishing gradient problem'
    ax.text(6, 0.7, insight, ha='center', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PURPLE, alpha=0.2))

    ax.set_xlim(0, 12)
    ax.set_ylim(0.3, 7)

    save_figure(fig, 'lstm_gates_simple_bsc.pdf')


# 5. RNN Applications Chart
def create_rnn_applications():
    """Real-world RNN applications"""
    fig, ax = setup_figure((12, 8))

    # Title
    ax.text(6, 7.5, 'Where Are RNNs Used? Real-World Applications', ha='center',
           fontsize=14, fontweight='bold')

    # Applications with icons and descriptions
    applications = [
        {'name': 'Text Generation', 'color': COLOR_BLUE, 'y': 6.0,
         'desc': 'Write stories, poetry, code', 'example': 'ChatGPT, GitHub Copilot'},
        {'name': 'Machine Translation', 'color': COLOR_GREEN, 'y': 4.8,
         'desc': 'Translate between languages', 'example': 'Google Translate'},
        {'name': 'Speech Recognition', 'color': COLOR_ORANGE, 'y': 3.6,
         'desc': 'Convert speech to text', 'example': 'Siri, Alexa'},
        {'name': 'Sentiment Analysis', 'color': COLOR_PURPLE, 'y': 2.4,
         'desc': 'Analyze emotions in text', 'example': 'Product reviews, social media'},
        {'name': 'Time Series Prediction', 'color': COLOR_RED, 'y': 1.2,
         'desc': 'Predict stock prices, weather', 'example': 'Financial forecasting'}
    ]

    for app in applications:
        y_pos = app['y']

        # Application box
        rect = FancyBboxPatch((0.5, y_pos - 0.35), 3.5, 0.7,
                              boxstyle="round,pad=0.08",
                              facecolor=app['color'], alpha=0.3,
                              edgecolor=app['color'], linewidth=2)
        ax.add_patch(rect)

        # Name
        ax.text(2.25, y_pos + 0.15, app['name'], ha='center', fontsize=11,
               fontweight='bold', color=app['color'])

        # Description
        ax.text(6.5, y_pos + 0.15, app['desc'], ha='left', fontsize=10, va='center')

        # Example
        ax.text(6.5, y_pos - 0.15, f'Example: {app["example"]}', ha='left',
               fontsize=8, style='italic', color=COLOR_GRAY, va='center')

    # Statistics box
    stats_text = ('RNNs power billions of daily interactions:\n'
                 '- 100+ billion Google Translate requests/day\n'
                 '- Millions of voice assistant queries\n'
                 '- Content generation in social media')
    ax.text(6, 0.3, stats_text, ha='center', fontsize=9,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_BLUE, alpha=0.1))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    save_figure(fig, 'rnn_applications_bsc.pdf')


# Generate all figures
if __name__ == '__main__':
    print("Generating BSc-level Week 3 RNN figures...")
    print("-" * 50)

    create_rnn_simple_unrolled()
    create_name_prediction_visual()
    create_vanishing_gradient_telephone()
    create_lstm_gates_simple()
    create_rnn_applications()

    print("-" * 50)
    print("All BSc-level figures generated successfully!")
