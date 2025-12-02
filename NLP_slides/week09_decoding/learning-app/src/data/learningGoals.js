/**
 * Learning Goals for Week 9: Decoding Strategies
 * Organizes all 62 slides into 3 pedagogical goals
 */

export const learningGoals = [
  {
    id: 'goal1',
    title: 'Understanding the Extremes',
    icon: '⚖️',
    description: 'Discover why neither greedy nor full search works',
    color: '#FF7F0E',  // Orange
    slideRange: [1, 10],  // Slides 1-10
    slides: 10,
    objectives: [
      'Understand the 50,000-word vocabulary challenge',
      'See why greedy decoding is too narrow (0.01% coverage)',
      'See why full search is too broad (exponential explosion)',
      'Identify the sweet spot (1-5% coverage)'
    ]
  },
  {
    id: 'goal2',
    title: 'The Toolbox: 6 Decoding Methods',
    icon: '🛠️',
    description: 'Master all 6 methods with worked examples',
    color: '#3333B2',  // Purple
    slideRange: [11, 43],  // Slides 11-43 (33 slides)
    slides: 33,
    objectives: [
      'Greedy: argmax selection (deterministic)',
      'Beam Search: top-k paths (deterministic)',
      'Temperature: reshape distribution (stochastic)',
      'Top-k: filter then sample (stochastic)',
      'Nucleus (Top-p): adaptive cutoff (stochastic)',
      'Contrastive: penalize repetition (deterministic)'
    ],
    subsections: [
      { name: 'Greedy', slides: [11] },
      { name: 'Beam Search', slides: [12, 13, 14] },
      { name: 'Temperature', slides: [15, 16, 17] },
      { name: 'Top-k', slides: [18, 19, 20] },
      { name: 'Nucleus', slides: [21, 22, 23] },
      { name: 'Contrastive', slides: [24, 25, 26, 27] },
      { name: 'Quiz 1', slides: [28] }
    ]
  },
  {
    id: 'goal3',
    title: 'Choosing the Right Method',
    icon: '🎯',
    description: 'Match methods to tasks and integrate knowledge',
    color: '#2CA02C',  // Green
    slideRange: [44, 62],  // Slides 44-62 (19 slides)
    slides: 19,
    objectives: [
      'Match each method to its problem',
      'Use decision trees for task selection',
      'Apply task-specific recommendations',
      'Understand quality-diversity tradeoffs',
      'Complete the learning journey'
    ],
    subsections: [
      { name: 'The 6 Problems', slides: [29, 30, 31, 32, 33, 34, 35] },
      { name: 'Quiz 2', slides: [36] },
      { name: 'Integration', slides: [37, 38, 39] },
      { name: 'Quiz 3', slides: [40] },
      { name: 'Conclusion', slides: [41, 42] },
      { name: 'Appendix', slides: [43, 44, 45] }  // Note: Slide IDs may be off by ~15
    ]
  }
];

// Total slides across all goals
export const totalSlides = 62;

// Helper to get goal for a slide number
export const getGoalForSlide = (slideNumber) => {
  for (const goal of learningGoals) {
    if (slideNumber >= goal.slideRange[0] && slideNumber <= goal.slideRange[1]) {
      return goal;
    }
  }
  return learningGoals[0]; // Default to first goal
};

// Helper to calculate goal progress
export const calculateGoalProgress = (goal, completedSlides) => {
  const goalSlides = [];
  for (let i = goal.slideRange[0]; i <= goal.slideRange[1]; i++) {
    goalSlides.push(i);
  }

  const completed = goalSlides.filter(s => completedSlides[s]).length;
  return Math.round((completed / goalSlides.length) * 100);
};

export default learningGoals;
