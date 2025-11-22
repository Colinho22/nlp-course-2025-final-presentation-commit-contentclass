/**
 * Mock Language Model for demonstration purposes
 * Simulates realistic probability distributions without requiring actual LLM
 */

// Sample vocabulary (simplified)
const VOCAB = [
  'the', 'cat', 'sat', 'on', 'mat', 'dog', 'ran', 'quickly', 'jumped', 'over',
  'fence', 'is', 'was', 'and', 'but', 'a', 'an', 'to', 'in', 'at',
  'mouse', 'hunt', 'began', 'quietly', 'through', 'grass', 'spotted', 'chased', 'caught', 'escaped',
  'forest', 'dark', 'night', 'bright', 'day', 'moon', 'sun', 'stars', 'sky', 'clouds',
  'suddenly', 'slowly', 'carefully', 'loudly', 'silently', 'happily', 'sadly', 'angrily', 'peacefully', 'frantically'
];

// Extended vocabulary for more realistic distribution
const createExtendedVocab = (size = 5000) => {
  const extended = [...VOCAB];
  for (let i = VOCAB.length; i < size; i++) {
    extended.push(`token_${i}`);
  }
  return extended;
};

const EXTENDED_VOCAB = createExtendedVocab();

/**
 * Generate realistic probability distribution using Zipf's law
 * (common in natural language)
 */
const generateZipfDistribution = (size, temperature = 1.0) => {
  const probs = [];
  let sum = 0;

  // Generate Zipf distribution
  for (let i = 0; i < size; i++) {
    const prob = 1.0 / Math.pow(i + 1, 1.2); // Zipf exponent
    probs.push(prob);
    sum += prob;
  }

  // Normalize
  const normalized = probs.map(p => p / sum);

  // Apply temperature
  if (temperature !== 1.0) {
    const adjusted = normalized.map(p => Math.pow(p, 1.0 / temperature));
    const adjustedSum = adjusted.reduce((a, b) => a + b, 0);
    return adjusted.map(p => p / adjustedSum);
  }

  return normalized;
};

/**
 * Context-aware prediction patterns
 */
const CONTEXT_PATTERNS = {
  'The cat': ['sat', 'jumped', 'ran', 'spotted', 'chased'],
  'cat sat': ['on', 'quietly', 'down', 'still', 'waiting'],
  'sat on': ['the', 'a', 'its', 'my', 'her'],
  'on the': ['mat', 'fence', 'grass', 'roof', 'floor'],
  'It was': ['a', 'the', 'an', 'quite', 'very'],
  'was a': ['mouse', 'dog', 'cat', 'bird', 'rabbit'],
  'a mouse': ['and', 'that', 'which', 'who', 'quietly'],
  'mouse and': ['the', 'it', 'she', 'he', 'they'],
  'and the': ['hunt', 'chase', 'search', 'game', 'race'],
  'the hunt': ['began', 'started', 'commenced', 'continued', 'ended'],
};

/**
 * Mock language model class
 */
class MockLanguageModel {
  constructor(vocabSize = 50) {
    this.vocabSize = vocabSize;
    this.vocab = EXTENDED_VOCAB.slice(0, vocabSize);
  }

  /**
   * Predict next token probabilities given context
   * @param {string} context - The current text context
   * @param {number} temperature - Sampling temperature (default 1.0)
   * @returns {Object} {tokens: string[], probs: number[], logits: number[]}
   */
  predict(context, temperature = 1.0) {
    const words = context.trim().split(/\s+/);
    const lastTwo = words.slice(-2).join(' ');
    const lastOne = words.slice(-1)[0] || '';

    // Check for context patterns
    const contextPattern = CONTEXT_PATTERNS[lastTwo] || CONTEXT_PATTERNS[lastOne];

    if (contextPattern) {
      return this._generateContextAwareDistribution(contextPattern, temperature);
    }

    // Default: Zipf distribution
    return this._generateDefaultDistribution(temperature);
  }

  /**
   * Generate context-aware distribution
   */
  _generateContextAwareDistribution(likelyWords, temperature) {
    const tokens = [];
    const probs = [];

    // High probability for context-relevant words
    const baseProb = 0.6 / likelyWords.length;
    for (const word of likelyWords) {
      if (this.vocab.includes(word)) {
        tokens.push(word);
        probs.push(baseProb);
      }
    }

    // Fill remaining with random tokens
    const remaining = this.vocabSize - tokens.length;
    const remainingProb = 0.4 / remaining;

    for (const word of this.vocab) {
      if (!tokens.includes(word)) {
        tokens.push(word);
        probs.push(remainingProb);
      }
    }

    // Apply temperature
    const adjusted = this._applyTemperature(probs, temperature);

    // Calculate logits (inverse of softmax)
    const logits = adjusted.map(p => Math.log(p + 1e-10));

    return { tokens, probs: adjusted, logits };
  }

  /**
   * Generate default Zipf distribution
   */
  _generateDefaultDistribution(temperature) {
    const tokens = [...this.vocab];
    const probs = generateZipfDistribution(this.vocabSize, temperature);
    const logits = probs.map(p => Math.log(p + 1e-10));

    return { tokens, probs, logits };
  }

  /**
   * Apply temperature to probability distribution
   */
  _applyTemperature(probs, temperature) {
    if (temperature === 1.0) return probs;

    const adjusted = probs.map(p => Math.pow(p, 1.0 / temperature));
    const sum = adjusted.reduce((a, b) => a + b, 0);
    return adjusted.map(p => p / sum);
  }

  /**
   * Sample from probability distribution
   */
  sample(probs) {
    const r = Math.random();
    let cumsum = 0;

    for (let i = 0; i < probs.length; i++) {
      cumsum += probs[i];
      if (r < cumsum) {
        return i;
      }
    }

    return probs.length - 1;
  }
}

// Export singleton instance
export const mockLM = new MockLanguageModel(50);

// Export class for custom instances
export default MockLanguageModel;

// Export vocabulary for testing
export { VOCAB, EXTENDED_VOCAB };
