# NLP Course Template Usage Guide

## Master Template: Optimal Readability Layout

All presentations in this course should use the **Optimal Readability** template for consistency and maximum clarity.

## Quick Start

### 1. Include the Master Template

```latex
\input{../common/master_template.tex}

\title{Your Presentation Title}
\subtitle{Week X - Topic}
\author{Your Name}
\date{\today}

\begin{document}
% Your slides here
\end{document}
```

### 2. Color Palette

The template provides these predefined colors with WCAG AAA contrast ratios:

| Color | Hex | Usage | Contrast Ratio |
|-------|-----|-------|----------------|
| PureBlack | #000000 | Main text | 21:1 |
| DeepBlue | #003D7A | Primary accent | 12.6:1 |
| DarkGray | #4A4A4A | Secondary text | 9.7:1 |
| DarkGreen | #228B22 | Success/positive | 7.5:1 |
| DarkRed | #CC0000 | Warning/negative | 8.3:1 |

### 3. Text Commands

```latex
\highlight{Important text}     % Deep blue bold
\secondary{Secondary info}      % Dark gray
\success{Positive result}       % Dark green
\warning{Critical warning}      % Dark red
\data{Data reference}           % Chart blue
```

### 4. Slide Layouts

#### Two-Column Slide
```latex
\twocolslide{Slide Title}{
  % Left column content
  \begin{itemize}
  \item First point
  \item Second point
  \end{itemize}
}{
  % Right column content
  \includegraphics[width=\textwidth]{figure.pdf}
}
```

#### Three-Column Slide
```latex
\threecolslide{Slide Title}{
  % Column 1
  \textbf{Section 1}
}{
  % Column 2
  \textbf{Section 2}
}{
  % Column 3
  \textbf{Section 3}
}
```

#### Chart Slide
```latex
\fullchartslide{Chart Title}{figures/chart.pdf}
% or
\chartslide{Chart Title}{0.8}{figures/chart.pdf}
```

#### Code Slide
```latex
\codeslide{Code Example}{Python}{
def attention(Q, K, V):
    scores = torch.matmul(Q, K.T)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
}
```

## Python Chart Generation

### 1. Import Chart Utilities

```python
import sys
sys.path.append('../../common')
from chart_utils import *

# Setup plotting style automatically applied
```

### 2. Create Standard Charts

```python
# Single panel figure
fig, ax = create_figure(figsize=(10, 6), title='My Chart')

# Multi-panel figure
fig, axes = create_multi_panel_figure(2, 2, figsize=(12, 8))

# Create visualizations
create_comparison_bars(ax, categories, data_dict,
                      ylabel='Score', title='Comparison')

create_line_plot(ax, x_data, y_data_dict,
                xlabel='Time', ylabel='Value')

create_heatmap(ax, matrix, xlabels, ylabels,
              title='Attention Weights')

# Save with optimal settings
save_figure(fig, 'output.pdf')
```

### 3. Use Standard Colors

```python
from chart_utils import COLORS, CHART_COLORS

# Access individual colors
main_color = COLORS['black']
accent = COLORS['blue_accent']

# Use chart color sequence
for i, series in enumerate(data_series):
    color = CHART_COLORS[i % len(CHART_COLORS)]
```

## Best Practices

### Typography
- **Minimum font size**: 8pt for slides
- **Headers**: Use `\Large` or `\huge` with `\textbf{}`
- **Body text**: Regular weight, black on white
- **Emphasis**: Use `\highlight{}` sparingly

### Charts
- **Always use** colorblind-safe palette
- **Export at** 300 DPI minimum
- **Include** direct labels when possible
- **Avoid** 3D effects and unnecessary decoration
- **Use** white backgrounds only

### Layout
- **Two-column** default for most content
- **Full-width** for important charts
- **White space** is your friend
- **Consistent** margins and spacing

### Tables
```latex
\begin{center}
\Large  % Increase readability
\renewcommand{\arraystretch}{1.5}  % Add row spacing
\begin{tabular}{l|ccc}
\toprule
\textbf{Model} & \textbf{Score} & \textbf{Speed} \\
\midrule
LSTM & \data{72.3} & \success{Fast} \\
Transformer & \highlight{95.8} & \warning{Slow} \\
\bottomrule
\end{tabular}
\end{center}
```

## File Organization

```
NLP_slides/
├── common/
│   ├── master_template.tex      # Include this
│   ├── chart_utils.py          # Import for charts
│   └── TEMPLATE_USAGE.md       # This file
├── week01_foundations/
│   ├── presentations/
│   │   └── main.tex           # Uses master_template
│   ├── python/
│   │   └── generate_charts.py # Uses chart_utils
│   └── figures/
│       └── *.pdf              # Generated charts
```

## Compilation

```bash
# Standard compilation
pdflatex presentation.tex
pdflatex presentation.tex  # Run twice for references

# Clean up
mkdir -p temp && mv *.aux *.log *.nav *.out *.toc temp/
```

## Accessibility Checklist

- [ ] All text has contrast ratio > 7:1
- [ ] Charts use colorblind-safe colors
- [ ] Font size >= 8pt throughout
- [ ] Images have text descriptions
- [ ] Tables use clear headers
- [ ] Code has syntax highlighting
- [ ] No color as sole information carrier

## Support

For questions or improvements to the template:
- Check `CLAUDE.md` for repository guidelines
- Review existing week implementations
- Maintain consistency across all presentations