# Week 9 Decoding - Learning App with Sidebar & Progress Tracking

**Modern learning-focused React app** with Material-UI, progress tracking, and organized learning goals.

## 🎯 Features

### 280px Sidebar
- **3 Learning Goals** with icons (⚖️ 🛠️ 🎯)
- **Progress bars** for each goal
- **Checkmarks** when complete
- **Click to jump** to any goal
- **Overall progress tracker**

### Flexible Main Area
- **Top progress bar** (Slide X of 62)
- **Slide content** using Material-UI Cards
- **PDF figures** displayed inline
- **Math formulas** rendered with KaTeX
- **Bottom navigation** with prev/next buttons

### Progress Tracking
- **localStorage** persistence (survives page refresh)
- **Auto-marks slides** as viewed after 3 seconds
- **Per-goal progress** calculation
- **Overall percentage** tracking

### Navigation
- **Keyboard**: Arrow keys (← →), Home, End
- **Mouse**: Prev/Next buttons, sidebar clicks
- **Icons**: Home (⏮) and Last (⏭) buttons

## 🚀 Quick Start

```bash
cd NLP_slides/week09_decoding/learning-app
npm run dev
```

Open: **http://localhost:5180/**

## 📚 3 Learning Goals

### Goal 1: Understanding the Extremes ⚖️
**Slides 1-10** (10 slides)
- The decoding challenge (50,000 words)
- Why greedy is too narrow (0.01% coverage)
- Why full search is too broad (exponential explosion)
- Finding the sweet spot (1-5%)

### Goal 2: The Toolbox 🛠️
**Slides 11-43** (33 slides)
- Method 1: Greedy (argmax)
- Method 2: Beam Search (top-k paths)
- Method 3: Temperature (reshape distribution)
- Method 4: Top-k (filter then sample)
- Method 5: Nucleus/Top-p (adaptive cutoff)
- Method 6: Contrastive (penalize repetition)
- Quiz 1: Match methods to mechanisms

### Goal 3: Choosing the Right Method 🎯
**Slides 44-62** (19 slides)
- The 6 problems each method solves
- Quiz 2: Match methods to problems
- Integration & decision trees
- Quiz 3: Real-world task selection
- Conclusion & complete journey
- Technical appendix

## 🎨 Design

### Colors (Material-UI Theme):
- **Primary Purple**: #3333B2
- **Secondary Lavender**: #ADADE0
- **Success Green**: #2CA02C
- **Warning Orange**: #FF7F0E
- **Error Red**: #D62728

### Components Used:
- Material-UI Drawer, Cards, Lists, Buttons
- Recharts (for future chart slides)
- Framer Motion (slide transitions)
- react-pdf (PDF figures)
- react-katex (math formulas)

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `→` | Next slide |
| `←` | Previous slide |
| `Home` | First slide |
| `End` | Last slide |

## 🎓 For Students

### Self-Paced Learning:
1. **Start with Goal 1** - Click ⚖️ Understanding the Extremes
2. **Navigate through slides** - Use arrow keys
3. **Watch your progress** - See percentage increase
4. **Complete all 3 goals** - Get checkmarks

## 🔄 Reset Progress

Open browser console (F12):
```javascript
localStorage.removeItem('week9-decoding-progress');
location.reload();
```

---

**Status**: ✅ READY
**Live**: http://localhost:5180/
