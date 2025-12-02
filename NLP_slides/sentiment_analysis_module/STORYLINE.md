# SENTIMENT ANALYSIS WITH TRANSFORMERS
## PEDAGOGICAL STORYLINE & NARRATIVE DESIGN

**Module Type**: Standalone BSc-level lecture
**Duration**: 10 main slides + 6 appendix = 16-17 total
**Narrative Structure**: Mystery/Investigation
**Pedagogical Approach**: Discovery-based with explicit scaffolding

---

## MYSTERY/INVESTIGATION ARC

**The Setup**: 10,000 reviews/day problem (Slide 1-2)
**The Puzzle**: 85% accuracy fails users (Slide 2)
**Clue Collection**: Context failures revealed (Slide 3)
**Dead Ends**: Traditional methods structurally limited (Slide 4)
**Breakthrough**: BERT bidirectionality solves puzzle (Slide 5)
**The Method**: Fine-tuning pipeline (Slide 6)
**Reality Check**: Practical training details (Slide 7)
**Proof**: Performance validation (Slide 8)
**Understanding**: Attention visualization (Slide 9)
**Wisdom**: When to use each approach (Slide 10)

---

## SLIDE-BY-SLIDE PEDAGOGICAL RATIONALE

### SLIDE 1: Title Slide
- **Narrative**: The Setup
- **Emotional Hook**: "Understanding" not just "classification"
- **Prevents**: "Sentiment = keyword counting" misconception
- **Forward Reference**: Mystery about to unfold

### SLIDE 2: Learning Goals + The Puzzle
- **Narrative**: Mystery Introduction
- **Learning Objective**: Goal 1 (context matters)
- **Cognitive Dissonance**: 85% sounds good BUT users complain
- **Emotional Hook**: Relatable "angry users" scenario
- **Prevents**: Surface-level metric satisfaction
- **Scaffolds**: Classification metrics → real-world problems
- **Forward Reference**: "What makes this review hard?"

### SLIDE 3: Clue #1 - Context Problem (CHART)
- **Narrative**: Collecting Clues
- **Learning Objective**: Goal 1 (bidirectional context)
- **Concrete Examples**: Sarcasm, negation, intensity
- **Emotional Hook**: "Aha! Word order matters!"
- **Prevents**: "Add bigrams" premature solution
- **Scaffolds**: BOW independence → positional semantics
- **Backward Callback**: "Remember 'Great, another boring movie'?"
- **Forward Reference**: "How do traditional methods handle this?"

### SLIDE 4: Dead End - Traditional Architectures (CHART)
- **Narrative**: Dead Ends
- **Learning Objective**: Goal 2 (BERT fine-tuning by contrast)
- **Frustration → Readiness**: Traditional can't solve puzzle
- **Prevents**: "Why not TF-IDF?" resistance
- **Scaffolds**: Independent → contextual processing
- **Backward Callback**: "Can't handle Slide 3 clues"
- **Forward Reference**: "What if bidirectional processing?"

### SLIDE 5: Breakthrough - BERT Architecture
- **Narrative**: The Breakthrough
- **Learning Objective**: Goal 2 (BERT architecture)
- **Emotional Hook**: "Aha!" moment
- **Prevents**: "BERT = better word vectors" misconception
- **Scaffolds**: Problem (context) → solution (bidirectionality)
- **Backward Callback**: "'Great' understood with 'boring'"
- **Forward Reference**: "How do we teach BERT sentiment?"

### SLIDE 6: Fine-Tuning Pipeline (CHART)
- **Narrative**: Solution Implementation
- **Learning Objective**: Goal 2 (fine-tuning process)
- **Emotional Hook**: "We don't start from zero!"
- **Prevents**: Confusion about what's learned when
- **Scaffolds**: Architecture → training process
- **Backward Callback**: "Stage 1 gives contextual understanding"
- **Forward Reference**: "What happens during training?"

**CHECKPOINT QUIZ** (after Slide 6): Reinforce stages understanding

### SLIDE 7: Training Details
- **Narrative**: Reality Check
- **Learning Objective**: Goal 4 (tradeoffs - resources)
- **Emotional Hook**: Relief - "Actually doable!"
- **Prevents**: "Only works at Google scale" defeatism
- **Scaffolds**: Conceptual → practical feasibility
- **Backward Callback**: "Why fine-tuning is efficient"
- **Forward Reference**: "Does this actually work?"

### SLIDE 8: Performance Comparison (CHART)
- **Narrative**: Evidence
- **Learning Objective**: Goal 3 (interpret metrics)
- **Emotional Hook**: Satisfaction - "Puzzle solved!"
- **Prevents**: Skepticism about worth of complexity
- **Scaffolds**: Theoretical → quantitative evidence
- **Backward Callback**: "Remember 85%? BERT gets 93-95%"
- **Forward Reference**: "HOW does it solve hard examples?"

### SLIDE 9: Attention Heatmap (CHART)
- **Narrative**: Understanding the Magic
- **Learning Objective**: Goal 3 (interpret attention)
- **Emotional Hook**: Curiosity - "See inside black box!"
- **Prevents**: "Uninterpretable model" fear
- **Scaffolds**: Performance numbers → mechanisms
- **Backward Callback**: "See how it handles negation clue?"
- **Forward Reference**: "When should we use BERT?"

### SLIDE 10: When to Use BERT
- **Narrative**: Epilogue - The Lesson
- **Learning Objective**: Goals 4-5 (tradeoffs, when appropriate)
- **Emotional Hook**: Empowerment - "Make informed choices"
- **Prevents**: Cargo-cult "always use best" thinking
- **Scaffolds**: Technical knowledge → engineering judgment
- **Backward Callback**: "Our puzzle needed BERT for context"
- **Forward Reference**: "See appendix for details"

---

## APPENDIX SLIDES (Advanced Material)

**A1**: Cross-Entropy Loss - Mathematical formulation
**A2**: Optimization - AdamW, learning rates, warmup
**A3**: Aspect-Based Sentiment - Per-feature extraction
**A4**: Multi-Label Sentiment - Multiple emotions
**A5**: Zero-Shot Classification - Prompting approach
**A6**: Resources & Further Reading

---

## PEDAGOGICAL FEATURES

### Progressive Complexity
- Each slide adds ONE new concept
- Builds on ALL prior understanding
- No forward references to unknown concepts

### Explicit Connections
- **Forward References**: "This raises question..." (9 instances)
- **Backward Callbacks**: "Remember when..." (9 instances)
- Creates coherent narrative thread

### Misconception Prevention
Every slide addresses common student misconceptions:
- Sentiment ≠ keyword counting (Slide 1)
- 85% ≠ good enough (Slide 2)
- Can't just add features (Slide 3)
- TF-IDF won't work (Slide 4)
- BERT ≠ better word vectors (Slide 5)
- Don't train from scratch (Slide 6)
- Not just for Google (Slide 7)
- Worth the complexity (Slide 8)
- Not a black box (Slide 9)
- Not always the answer (Slide 10)

### Emotional Engagement
Each slide designed to evoke specific emotion:
- Curiosity (Slides 1-2)
- Recognition (Slide 3)
- Frustration → Readiness (Slide 4)
- "Aha!" moment (Slide 5)
- Efficiency appreciation (Slide 6)
- Relief (Slide 7)
- Satisfaction (Slide 8)
- Wonder (Slide 9)
- Empowerment (Slide 10)

---

## LEARNING OBJECTIVES MAPPING

1. **Explain bidirectional context** → Slides 3, 4, 5
2. **Describe BERT fine-tuning** → Slides 5, 6, 7
3. **Interpret performance & attention** → Slides 8, 9
4. **Evaluate tradeoffs** → Slides 7, 10
5. **Identify when appropriate** → Slide 10

---

## CHART USAGE (Minimal - Text-First Approach)

**ONLY 1 CHART RETAINED** (Nov 24, 2025 revision):

1. **attention_heatmap_bsc.pdf** (Slide 9): Demystify black box
   - ONLY chart that genuinely requires visualization
   - Shows 2D attention pattern that cannot be expressed in text
   - Provides "see inside the model" pedagogical moment

**CONVERTED TO TEXT-BASED SLIDES**:
- Slide 3 (context_matters): Table with examples (sarcasm, negation, intensity)
- Slide 4 (architecture_comparison): Two-column bullet comparison
- Slide 6 (finetuning_pipeline): Numbered list of 4 stages
- Slide 8 (performance_comparison): Table with F1-scores

**RATIONALE**: Charts should only exist when visualization genuinely adds clarity.
Pipelines, comparisons, and lists work better as structured text.

---

## BSc DISCOVERY STANDARDS APPLIED

- **Font sizes**: 16-24pt (FONTSIZE constants)
- **Color scheme**: BSc Discovery palette
- **Pedagogical comments**: Embedded in LaTeX
- **Bottom notes**: Educational context on chart slides
- **Progressive reveal**: One concept per slide
- **Checkpoint quiz**: After critical concept (Slide 6)

---

**Generated**: November 23, 2025
**Module**: Sentiment Analysis with Transformers
**Pedagogical Framework**: Discovery-based BSc
