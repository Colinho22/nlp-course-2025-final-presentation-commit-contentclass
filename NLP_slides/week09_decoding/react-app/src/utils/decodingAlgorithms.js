/**
 * Decoding Algorithms for Text Generation
 * Implements 6 core methods: Greedy, Beam Search, Temperature, Top-k, Nucleus, Contrastive
 */

import { mockLM } from './mockLM';

/**
 * 1. GREEDY DECODING
 * Always selects the highest probability token
 * Deterministic - same input always produces same output
 */
export const greedyDecode = (prompt, maxLength = 20, model = mockLM) => {
  let text = prompt;
  const history = [{ step: 0, text: prompt, token: null, prob: null }];

  for (let step = 1; step <= maxLength; step++) {
    const { tokens, probs } = model.predict(text);

    // Find argmax
    const maxIdx = probs.indexOf(Math.max(...probs));
    const selectedToken = tokens[maxIdx];
    const selectedProb = probs[maxIdx];

    text += ' ' + selectedToken;

    history.push({
      step,
      text,
      token: selectedToken,
      prob: selectedProb,
      allTokens: tokens.slice(0, 5),
      allProbs: probs.slice(0, 5)
    });
  }

  return { text, history };
};

/**
 * 2. BEAM SEARCH
 * Maintains top-k most likely sequences
 * Deterministic - explores multiple paths but returns best
 */
export const beamSearch = (prompt, maxLength = 20, beamWidth = 3, model = mockLM) => {
  // Initialize with starting sequence
  let beams = [{
    text: prompt,
    score: 0.0, // log probability
    history: []
  }];

  for (let step = 1; step <= maxLength; step++) {
    const candidates = [];

    // Expand each beam
    for (const beam of beams) {
      const { tokens, probs } = model.predict(beam.text);

      // Get top-k tokens for this beam
      const topK = tokens.map((token, idx) => ({
        token,
        prob: probs[idx],
        logProb: Math.log(probs[idx] + 1e-10)
      }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, beamWidth);

      // Create new candidates
      for (const { token, prob, logProb } of topK) {
        candidates.push({
          text: beam.text + ' ' + token,
          score: beam.score + logProb,
          history: [...beam.history, { step, token, prob, parentScore: beam.score }]
        });
      }
    }

    // Keep top beamWidth candidates
    beams = candidates
      .sort((a, b) => b.score - a.score)
      .slice(0, beamWidth);
  }

  // Return best beam
  const best = beams[0];
  return {
    text: best.text,
    score: best.score,
    history: best.history,
    allBeams: beams
  };
};

/**
 * 3. TEMPERATURE SAMPLING
 * Reshapes probability distribution to control randomness
 * Stochastic - different outputs each time
 */
export const temperatureSample = (prompt, maxLength = 20, temperature = 0.7, model = mockLM) => {
  let text = prompt;
  const history = [{ step: 0, text: prompt, token: null, prob: null }];

  for (let step = 1; step <= maxLength; step++) {
    const { tokens, probs } = model.predict(text, temperature);

    // Sample from distribution
    const sampledIdx = model.sample(probs);
    const selectedToken = tokens[sampledIdx];
    const selectedProb = probs[sampledIdx];

    text += ' ' + selectedToken;

    history.push({
      step,
      text,
      token: selectedToken,
      prob: selectedProb,
      temperature,
      allTokens: tokens.slice(0, 5),
      allProbs: probs.slice(0, 5)
    });
  }

  return { text, history };
};

/**
 * 4. TOP-K SAMPLING
 * Only sample from top-k most likely tokens
 * Stochastic with controlled diversity
 */
export const topkSample = (prompt, maxLength = 20, k = 40, temperature = 1.0, model = mockLM) => {
  let text = prompt;
  const history = [{ step: 0, text: prompt, token: null, prob: null }];

  for (let step = 1; step <= maxLength; step++) {
    const { tokens, probs } = model.predict(text, temperature);

    // Get top-k indices
    const indexed = probs.map((p, i) => ({ p, i }))
      .sort((a, b) => b.p - a.p)
      .slice(0, k);

    // Renormalize top-k probabilities
    const topKSum = indexed.reduce((sum, { p }) => sum + p, 0);
    const topKProbs = indexed.map(({ p }) => p / topKSum);

    // Sample from top-k
    const sampledIdx = model.sample(topKProbs);
    const actualIdx = indexed[sampledIdx].i;
    const selectedToken = tokens[actualIdx];
    const selectedProb = probs[actualIdx];

    text += ' ' + selectedToken;

    history.push({
      step,
      text,
      token: selectedToken,
      prob: selectedProb,
      k,
      topKTokens: indexed.slice(0, 5).map(({ i }) => tokens[i]),
      topKProbs: topKProbs.slice(0, 5)
    });
  }

  return { text, history };
};

/**
 * 5. NUCLEUS (TOP-P) SAMPLING
 * Sample from smallest set of tokens whose cumulative probability >= p
 * Adaptive - set size changes based on distribution shape
 */
export const nucleusSample = (prompt, maxLength = 20, p = 0.9, temperature = 1.0, model = mockLM) => {
  let text = prompt;
  const history = [{ step: 0, text: prompt, token: null, prob: null }];

  for (let step = 1; step <= maxLength; step++) {
    const { tokens, probs } = model.predict(text, temperature);

    // Sort by probability (descending)
    const sorted = probs.map((prob, i) => ({ prob, i }))
      .sort((a, b) => b.prob - a.prob);

    // Find nucleus cutoff
    let cumsum = 0;
    let cutoff = 0;
    for (let i = 0; i < sorted.length; i++) {
      cumsum += sorted[i].prob;
      if (cumsum >= p) {
        cutoff = i + 1;
        break;
      }
    }

    const nucleus = sorted.slice(0, cutoff);

    // Renormalize nucleus probabilities
    const nucleusSum = nucleus.reduce((sum, { prob }) => sum + prob, 0);
    const nucleusProbs = nucleus.map(({ prob }) => prob / nucleusSum);

    // Sample from nucleus
    const sampledIdx = model.sample(nucleusProbs);
    const actualIdx = nucleus[sampledIdx].i;
    const selectedToken = tokens[actualIdx];
    const selectedProb = probs[actualIdx];

    text += ' ' + selectedToken;

    history.push({
      step,
      text,
      token: selectedToken,
      prob: selectedProb,
      p,
      nucleusSize: nucleus.length,
      nucleusTokens: nucleus.slice(0, 5).map(({ i }) => tokens[i]),
      nucleusProbs: nucleusProbs.slice(0, 5)
    });
  }

  return { text, history };
};

/**
 * 6. CONTRASTIVE SEARCH
 * Balances probability and diversity (penalizes repetition)
 * Deterministic - degeneration penalty
 */
export const contrastiveSearch = (prompt, maxLength = 20, k = 4, alpha = 0.6, model = mockLM) => {
  let text = prompt;
  const tokens_generated = [];
  const history = [{ step: 0, text: prompt, token: null, prob: null }];

  for (let step = 1; step <= maxLength; step++) {
    const { tokens, probs } = model.predict(text);

    // Get top-k candidates
    const topK = probs.map((p, i) => ({ p, i, token: tokens[i] }))
      .sort((a, b) => b.p - a.p)
      .slice(0, k);

    // Calculate contrastive score for each candidate
    const scores = topK.map(({ p, token }) => {
      const modelScore = Math.log(p + 1e-10);

      // Degeneration penalty (similarity to previous tokens)
      let maxSimilarity = 0;
      if (tokens_generated.length > 0) {
        // Simple similarity: token matching (in real impl, would use embeddings)
        const matches = tokens_generated.filter(t => t === token).length;
        maxSimilarity = matches / tokens_generated.length;
      }

      // Contrastive score: balance probability and diversity
      const contrastiveScore = (1 - alpha) * modelScore - alpha * maxSimilarity;

      return {
        token,
        prob: p,
        modelScore,
        similarity: maxSimilarity,
        contrastiveScore
      };
    });

    // Select token with highest contrastive score
    const best = scores.reduce((a, b) =>
      a.contrastiveScore > b.contrastiveScore ? a : b
    );

    text += ' ' + best.token;
    tokens_generated.push(best.token);

    history.push({
      step,
      text,
      token: best.token,
      prob: best.prob,
      alpha,
      modelScore: best.modelScore,
      similarity: best.similarity,
      contrastiveScore: best.contrastiveScore,
      allScores: scores.slice(0, 3)
    });
  }

  return { text, history, tokensGenerated: tokens_generated };
};

/**
 * Helper: Calculate text statistics
 */
export const calculateStats = (text) => {
  const tokens = text.trim().split(/\s+/);

  // Repetition rate (repeated bigrams)
  const bigrams = [];
  for (let i = 0; i < tokens.length - 1; i++) {
    bigrams.push(tokens[i] + ' ' + tokens[i + 1]);
  }
  const uniqueBigrams = new Set(bigrams);
  const repetitionRate = bigrams.length > 0
    ? (1 - uniqueBigrams.size / bigrams.length) * 100
    : 0;

  // Distinct-1 and Distinct-2
  const distinct1 = new Set(tokens).size / tokens.length;
  const distinct2 = uniqueBigrams.size / bigrams.length;

  return {
    length: tokens.length,
    uniqueTokens: new Set(tokens).size,
    repetitionRate: repetitionRate.toFixed(1),
    distinct1: (distinct1 * 100).toFixed(1),
    distinct2: (distinct2 * 100).toFixed(1)
  };
};

/**
 * Helper: Compare multiple methods on same prompt
 */
export const compareAllMethods = (prompt, config = {}) => {
  const {
    maxLength = 15,
    temperature = 0.7,
    beamWidth = 3,
    topK = 40,
    topP = 0.9,
    alpha = 0.6
  } = config;

  const results = {
    greedy: greedyDecode(prompt, maxLength),
    beam: beamSearch(prompt, maxLength, beamWidth),
    temperature: temperatureSample(prompt, maxLength, temperature),
    topk: topkSample(prompt, maxLength, topK, temperature),
    nucleus: nucleusSample(prompt, maxLength, topP, temperature),
    contrastive: contrastiveSearch(prompt, maxLength, 4, alpha)
  };

  // Add statistics to each
  for (const [method, result] of Object.entries(results)) {
    result.stats = calculateStats(result.text);
    result.method = method;
  }

  return results;
};

export default {
  greedyDecode,
  beamSearch,
  temperatureSample,
  topkSample,
  nucleusSample,
  contrastiveSearch,
  calculateStats,
  compareAllMethods
};
