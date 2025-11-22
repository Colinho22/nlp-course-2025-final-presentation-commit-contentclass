# Week 9: Decoding Strategies - Interactive React App

Interactive learning platform for understanding text generation decoding methods in NLP.

## Features

- **Interactive Slide Viewer**: Navigate through 16+ educational slides with keyboard controls
- **Decoding Playground**: Hands-on experimentation with 6 decoding methods
- **Real-time Parameter Tuning**: Adjust temperature, top-k, top-p, beam width, and alpha
- **Method Comparison**: See all 6 methods side-by-side on the same prompt
- **Quality Metrics**: Automatic calculation of repetition rate, distinct-n scores
- **Preset Configurations**: Quick switches between Factual, Creative, and Balanced modes

## Decoding Methods Implemented

1. **Greedy Decoding**: Always select highest probability token (deterministic)
2. **Beam Search**: Maintain top-k sequences (deterministic)
3. **Temperature Sampling**: Reshape distribution for creativity control (stochastic)
4. **Top-k Sampling**: Filter to top-k then sample (stochastic)
5. **Nucleus (Top-p)**: Adaptive cumulative probability cutoff (stochastic)
6. **Contrastive Search**: Penalize repetition and similar tokens (deterministic)

## Getting Started

### Installation

```bash
cd NLP_slides/week09_decoding/react-app
npm install
```

### Development

```bash
npm run dev
```

The app will start at `http://localhost:5173`

### Build for Production

```bash
npm run build
npm run preview
```

## Project Structure

```
react-app/
├── src/
│   ├── components/
│   │   ├── SlideViewer/       # Main presentation viewer
│   │   └── DecodingDemo/      # Interactive playground
│   ├── utils/
│   │   ├── decodingAlgorithms.js  # 6 method implementations
│   │   └── mockLM.js              # Simulated language model
│   ├── data/
│   │   └── slideContent.js    # Slide content and metadata
│   ├── App.jsx                # Main app component
│   └── index.css              # Global styles + Tailwind
├── public/
│   ├── figures/               # Converted charts (to be added)
│   └── data/                  # Additional data files
├── package.json
└── README.md
```

## Technology Stack

- **React 18**: UI framework
- **Vite**: Build tool (fast HMR)
- **React Router**: Navigation
- **Tailwind CSS**: Styling with BSc Discovery color palette
- **Recharts**: Charts and visualizations (planned)
- **D3.js**: Interactive visualizations (planned)

## Color Palette (BSc Discovery)

- Purple (`#3333B2`): Primary brand color
- Dark Gray (`#404040`): Main text
- Light Gray (`#B4B4B4`): Secondary elements
- Lavender shades: Backgrounds and accents
- Green (`#2CA02C`): Success/positive
- Red (`#D62728`): Error/negative
- Orange (`#FF7F0E`): Warning/medium
- Blue (`#0066CC`): Information

## Usage Guide

### Slide Viewer

- Use **arrow keys** or **navigation buttons** to move between slides
- Click **section buttons** to jump to specific topics
- **Progress bar** shows your location in the presentation

### Playground

1. Enter a **prompt** (starting text)
2. Adjust **parameters** using sliders
3. Select **method(s)** to compare (or show all)
4. Click **Generate Text** to see results
5. Compare **outputs** and **metrics**

### Preset Modes

- **Factual**: Low temperature (0.2), greedy-like, for Q&A
- **Creative**: High temperature (0.9), nucleus sampling, for stories
- **Balanced**: Medium settings (T=0.7, p=0.9), general purpose

## Key Metrics Explained

- **Length**: Total tokens generated
- **Unique Tokens**: Vocabulary diversity
- **Repetition Rate**: Percentage of repeated bigrams (lower is better)
- **Distinct-1**: Ratio of unique unigrams to total (higher is better)
- **Distinct-2**: Ratio of unique bigrams to total (higher is better)

## Educational Content

Based on Week 9 of the NLP Course 2025:
- Slides extracted from `20251119_0943_week09_decoding_final.pdf`
- Pedagogical structure: Extremes → Toolbox → Problems → Integration
- Discovery-based learning approach

## Future Enhancements

- [ ] Interactive beam search tree visualization (D3.js)
- [ ] Animated probability distribution charts
- [ ] Checkpoint quizzes with immediate feedback
- [ ] Quality-diversity scatter plot
- [ ] Real-time token-by-token generation display
- [ ] Export/share results functionality
- [ ] Integration with actual language models (GPT-2)

## Development Notes

### Mock Language Model

The app uses a **simulated language model** (`mockLM.js`) that:
- Generates realistic Zipf-law probability distributions
- Provides context-aware predictions for common phrases
- Avoids requiring actual LLM API calls
- Runs entirely in the browser

### Algorithm Implementations

All 6 decoding algorithms are implemented in pure JavaScript with:
- No external LLM dependencies
- Realistic probability calculations
- Step-by-step history tracking
- Configurable parameters

## Keyboard Shortcuts

- **←/→**: Navigate slides (in Slide Viewer)
- **Tab**: Focus navigation buttons
- **Enter**: Click focused button

## Browser Support

- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support
- Mobile browsers: Responsive design

## License

Part of NLP Course 2025 materials (MIT License)

## Acknowledgments

- Course content: Week 9 Decoding Strategies
- Framework: React + Vite
- Design: BSc Discovery pedagogical approach
- Algorithms: Based on research papers and production implementations

---

**Last Updated**: November 20, 2025
**Version**: 1.0.0 (MVP)
**Status**: Functional with 2 main components (Slides + Playground)
