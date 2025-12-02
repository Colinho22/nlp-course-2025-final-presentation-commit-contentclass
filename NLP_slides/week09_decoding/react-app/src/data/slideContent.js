/**
 * Slide Content for Week 9: Decoding Strategies
 * Extracted from 20251119_0943_week09_decoding_final.tex
 */

export const slides = [
  {
    id: 1,
    section: "intro",
    title: "Decoding Strategies",
    subtitle: "From Probabilities to Text",
    content: [
      "How do we convert model predictions into human-readable text?",
      "The decoding challenge: 50,000+ possible next words",
      "6 methods to balance quality and diversity"
    ]
  },
  {
    id: 2,
    section: "intro",
    title: "The Decoding Challenge",
    content: [
      "Language models output probability distributions",
      "Vocabulary size: 50,000 tokens (GPT-2/3)",
      "At each step: which token should we choose?",
      "Trade-off: Quality vs Diversity vs Speed"
    ],
    keyPoint: "Models predict - decoding decides"
  },
  {
    id: 3,
    section: "intro",
    title: "How We Got Here",
    content: [
      "RNN/LSTM: Sequential, slow, limited context",
      "Transformers: Parallel, fast, long context",
      "BERT: Bidirectional (masked language modeling)",
      "GPT: Autoregressive (next token prediction)",
      "Modern LLMs: GPT-3, GPT-4, Claude, etc."
    ]
  },
  {
    id: 4,
    section: "extremes",
    title: "Extreme 1: Greedy Decoding",
    content: [
      "Strategy: Always pick highest probability token",
      "Coverage: 0.01% of search space",
      "Problem: Too narrow, misses better paths",
      "Speed: Fastest (O(n))",
      "Deterministic: Same input → same output"
    ],
    demo: "greedy"
  },
  {
    id: 5,
    section: "extremes",
    title: "Greedy's Fatal Flaw",
    content: [
      "High probability ≠ Better text quality",
      "Example: 'The cat saw...'",
      "Greedy path 1: 'it' (P=0.50) → Total: 0.076",
      "Better path: 'a mouse' (P=0.30 × 0.25) → Total: 0.062",
      "Greedy picks 'it' but misses the better story"
    ],
    demo: "greedy_flaw"
  },
  {
    id: 6,
    section: "extremes",
    title: "Extreme 2: Full Search",
    content: [
      "Strategy: Explore all possible paths",
      "Coverage: 100% of search space",
      "Problem: Exponential explosion (50,000^n)",
      "For 10 tokens: 9.77 × 10^46 paths",
      "Computationally impossible"
    ],
    demo: "full_search"
  },
  {
    id: 7,
    section: "extremes",
    title: "The Sweet Spot",
    content: [
      "Neither extreme works in practice",
      "Need: 1-5% coverage",
      "Goal: High quality + acceptable diversity",
      "Solution: 6 practical methods"
    ]
  },
  {
    id: 8,
    section: "toolbox",
    title: "The Toolbox: 6 Methods",
    content: [
      "1. Greedy: argmax (deterministic)",
      "2. Beam Search: top-k paths (deterministic)",
      "3. Temperature: reshape distribution (stochastic)",
      "4. Top-k: filter then sample (stochastic)",
      "5. Nucleus (Top-p): adaptive cutoff (stochastic)",
      "6. Contrastive: penalize repetition (deterministic)"
    ]
  },
  {
    id: 9,
    section: "toolbox",
    title: "Method 1: Greedy Decoding",
    formula: "w_t = argmax P(w | context)",
    content: [
      "Mechanism: Select highest probability token",
      "Type: Deterministic",
      "Coverage: 0.01%",
      "Speed: O(n) - fastest",
      "Use case: Factual Q&A (temperature 0.1-0.3)"
    ],
    demo: "greedy"
  },
  {
    id: 10,
    section: "toolbox",
    title: "Method 2: Beam Search",
    formula: "Keep top-k sequences at each step",
    content: [
      "Mechanism: Maintain k best partial sequences",
      "Type: Deterministic",
      "Coverage: 0.1-1%",
      "Speed: O(kn) - k times slower than greedy",
      "Use case: Machine translation (k=4-5)"
    ],
    demo: "beam_search"
  },
  {
    id: 11,
    section: "toolbox",
    title: "Method 3: Temperature Sampling",
    formula: "P(w_i) = exp(z_i/T) / Σ exp(z_j/T)",
    content: [
      "Mechanism: Reshape probability distribution",
      "T < 1: More confident (sharper)",
      "T > 1: More random (flatter)",
      "Type: Stochastic",
      "Use case: Control creativity level"
    ],
    demo: "temperature"
  },
  {
    id: 12,
    section: "toolbox",
    title: "Method 4: Top-k Sampling",
    formula: "Sample from k highest probability tokens",
    content: [
      "Mechanism: Filter to top-k, then sample",
      "Type: Stochastic",
      "Coverage: 1-2%",
      "Fixed set size (k=40-50 common)",
      "Use case: Moderate creativity"
    ],
    demo: "topk"
  },
  {
    id: 13,
    section: "toolbox",
    title: "Method 5: Nucleus (Top-p)",
    formula: "Sample from tokens where Σ P(w_i) ≥ p",
    content: [
      "Mechanism: Adaptive cumulative probability cutoff",
      "Type: Stochastic",
      "Coverage: 2-5%",
      "Adaptive set size (p=0.9 common)",
      "Use case: General purpose (most popular)"
    ],
    demo: "nucleus"
  },
  {
    id: 14,
    section: "toolbox",
    title: "Method 6: Contrastive Search",
    formula: "Score = (1-α)·log P(w) - α·max_similarity",
    content: [
      "Mechanism: Penalize similar/repeated tokens",
      "Type: Deterministic",
      "Coverage: 1-3%",
      "Prevents degeneration (repetition)",
      "Use case: Long text generation (α=0.6)"
    ],
    demo: "contrastive"
  },
  {
    id: 15,
    section: "quiz",
    title: "Checkpoint Quiz 1",
    type: "quiz",
    question: "Match each method to its mechanism",
    options: [
      { method: "Greedy", mechanism: "argmax", correct: true },
      { method: "Beam", mechanism: "top-k paths", correct: true },
      { method: "Temperature", mechanism: "reshape distribution", correct: true },
      { method: "Top-k", mechanism: "filter then sample", correct: true },
      { method: "Nucleus", mechanism: "cumulative cutoff", correct: true },
      { method: "Contrastive", mechanism: "similarity penalty", correct: true }
    ]
  },
  {
    id: 16,
    section: "playground",
    title: "Interactive Playground",
    type: "demo",
    content: [
      "Try all 6 methods on the same prompt",
      "Adjust parameters in real-time",
      "Compare outputs and metrics",
      "See quality-diversity tradeoffs"
    ]
  }
];

export const sections = [
  { id: "intro", name: "Introduction", slides: [1, 2, 3] },
  { id: "extremes", name: "Extreme Cases", slides: [4, 5, 6, 7] },
  { id: "toolbox", name: "The Toolbox", slides: [8, 9, 10, 11, 12, 13, 14] },
  { id: "quiz", name: "Quiz", slides: [15] },
  { id: "playground", name: "Interactive Playground", slides: [16] }
];

export const methodInfo = {
  greedy: {
    name: "Greedy Decoding",
    type: "Deterministic",
    coverage: "0.01%",
    speed: "Fastest",
    formula: "w_t = argmax P(w | context)",
    useCase: "Factual Q&A",
    params: { temperature: 0.1 }
  },
  beam: {
    name: "Beam Search",
    type: "Deterministic",
    coverage: "0.1-1%",
    speed: "Medium",
    formula: "Keep top-k sequences",
    useCase: "Machine Translation",
    params: { beamWidth: 4 }
  },
  temperature: {
    name: "Temperature Sampling",
    type: "Stochastic",
    coverage: "Variable",
    speed: "Fast",
    formula: "P(w_i) = exp(z_i/T) / Σ exp(z_j/T)",
    useCase: "Creativity Control",
    params: { temperature: 0.7 }
  },
  topk: {
    name: "Top-k Sampling",
    type: "Stochastic",
    coverage: "1-2%",
    speed: "Fast",
    formula: "Sample from top-k tokens",
    useCase: "Moderate Creativity",
    params: { k: 40, temperature: 1.0 }
  },
  nucleus: {
    name: "Nucleus (Top-p)",
    type: "Stochastic",
    coverage: "2-5%",
    speed: "Fast",
    formula: "Σ P(w_i) ≥ p",
    useCase: "General Purpose",
    params: { p: 0.9, temperature: 0.7 }
  },
  contrastive: {
    name: "Contrastive Search",
    type: "Deterministic",
    coverage: "1-3%",
    speed: "Medium",
    formula: "(1-α)·log P(w) - α·similarity",
    useCase: "Long Text",
    params: { k: 4, alpha: 0.6 }
  }
};

export default { slides, sections, methodInfo };
