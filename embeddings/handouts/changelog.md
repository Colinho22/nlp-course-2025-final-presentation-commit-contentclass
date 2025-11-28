# Changelog - Word Embeddings Discovery Handout

## 2025-09-18 - Neutral Academic Version (Latest)
### Files Created:
- `discovery_handout_neutral.tex` - Academic-style worksheet (5 pages)
- `discovery_handout_neutral.pdf` - Compiled PDF (160KB)

### Key Features:
- **Neutral styling**: Minimal colors, simple framed boxes
- **No Python code**: Pure mathematical approach
- **NEW Section**: "Understanding Dot Product as Similarity"
  - Mathematical foundation of dot product
  - Geometric interpretation with angles
  - Practice calculations
- **Guided calculations**: Students fill in blanks for:
  - Character overlap percentages
  - One-hot dot products
  - Dense vector similarity scores
- **Worksheet format**: Fill-in exercises throughout

## 2025-09-18 - Progressive Discovery Version
### Files Created:
- `discovery_handout_progressive.tex` - Main progressive handout (5 pages)
- `discovery_handout_progressive.pdf` - Compiled PDF (155KB)

### Structure:
1. **Discovery 1**: Character Similarity Doesn't Work
   - Shows cat/car vs cat/kitten character overlap problem
   - Includes comparison table and code example

2. **Discovery 2**: One-Hot Encoding Doesn't Work
   - Demonstrates orthogonality problem
   - 3D visualization showing all vectors at 90 degrees

3. **Discovery 3**: Dense Vectors Work (Distance = Meaning)
   - 3D clustering visualization
   - Shows semantic distance relationships

4. **Discovery 4**: Relationships Work (Vector Arithmetic)
   - Paris/Eiffel Tower, London/Big Ben examples
   - king - man + woman = queen analogy
   - 3D parallel relationship visualization

5. **Discovery 5**: Context Works (Dynamic Embeddings)
   - Bank (financial) vs Bank (river) example
   - Shows word position movement based on context

### Supporting Files:
- `generate_landmark_3d.py` - Creates 3D landmark analogies
- `generate_3d_charts.py` - Creates general 3D embeddings visualizations
- Generated figures in `../figures/`:
  - `landmark_analogies_3d.pdf`
  - `cultural_analogies_3d.pdf`
  - `semantic_clusters_3d.pdf`
  - `context_movement_3d.pdf`
  - `interactive_landmarks_3d.html`

## Previous Versions:
- `discovery_handout_focused.tex` - 3-page landmark-focused version
- `discovery_handout_compact.tex` - 4-page condensed version
- `discovery_handout.tex` - Original 13-page comprehensive version

## Key Features:
- Progressive structure: failures → successes
- 3D TikZ visualizations throughout
- Landmark examples (Eiffel Tower, Big Ben)
- Minimal calculations, maximum visual learning
- Interactive elements with "Try This" code snippets
- Color-coded discovery boxes (red for failures, green for successes)