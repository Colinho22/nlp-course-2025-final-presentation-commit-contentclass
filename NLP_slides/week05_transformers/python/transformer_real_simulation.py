"""
Real Transformer Simulation with Actual Computations
Educational framework: Show REAL numbers, not dummy values
Input: "The cat sat on the ___"
Output: Predict next word with actual softmax probabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# Set random seed for reproducibility
np.random.seed(42)

# Template colors
COLOR_BLUE = '#4472C4'
COLOR_PURPLE = '#8B5A9B'
COLOR_GREEN = '#44A044'
COLOR_ORANGE = '#FF7F0E'
COLOR_RED = '#D62728'
COLOR_MAIN = '#333366'

# ============================================
# STEP 1: Define vocabulary and input sentence
# ============================================

VOCAB = [
    'The', 'cat', 'sat', 'on', 'the', 'mat', 'floor', 'table', 'rug',
    'dog', 'bird', 'ran', 'flew', 'in', 'under', 'over', 'chair',
    'bed', 'sofa', 'window', '<PAD>', '<START>', '<END>'
]

# Input sentence
input_sentence = ['The', 'cat', 'sat', 'on', 'the']
input_ids = [VOCAB.index(w) for w in input_sentence]

# Model parameters
d_model = 8  # Embedding dimension (small for visualization)
n_heads = 4  # Number of attention heads
d_head = d_model // n_heads  # Dimension per head (2)
vocab_size = len(VOCAB)

# ============================================
# STEP 2: Create real word embeddings
# ============================================

# Fixed random embeddings for each word in vocabulary
embedding_matrix = np.random.randn(vocab_size, d_model) * 0.5

# Get embeddings for input sentence
input_embeddings = embedding_matrix[input_ids]  # (5, 8)

print("=" * 60)
print("STEP 1: Word Embeddings")
print("=" * 60)
print(f"Input sentence: {input_sentence}")
print(f"Input IDs: {input_ids}")
print(f"\nEmbedding matrix shape: {embedding_matrix.shape}")
print(f"Input embeddings shape: {input_embeddings.shape}")
print(f"\nExample: 'cat' embedding (first 4 dims): {input_embeddings[1, :4]}")

# ============================================
# STEP 3: Add positional encodings
# ============================================

def get_positional_encoding(seq_len, d_model):
    """Real sinusoidal positional encoding"""
    pos_enc = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (2 * i / d_model)))

    return pos_enc

pos_encoding = get_positional_encoding(len(input_sentence), d_model)
input_with_pos = input_embeddings + pos_encoding  # (5, 8)

print("\n" + "=" * 60)
print("STEP 2: Positional Encoding")
print("=" * 60)
print(f"Positional encoding shape: {pos_encoding.shape}")
print(f"\nExample: Position 0 encoding (first 4 dims): {pos_encoding[0, :4]}")
print(f"Example: Position 1 encoding (first 4 dims): {pos_encoding[1, :4]}")
print(f"\nAfter adding position: {input_with_pos.shape}")

# ============================================
# STEP 4: Single-head attention (for explanation)
# ============================================

def softmax(x):
    """Numerical stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Single attention head (for educational clarity)
W_Q = np.random.randn(d_model, d_head) * 0.1
W_K = np.random.randn(d_model, d_head) * 0.1
W_V = np.random.randn(d_model, d_head) * 0.1

# Compute Q, K, V
Q = input_with_pos @ W_Q  # (5, 2)
K = input_with_pos @ W_K  # (5, 2)
V = input_with_pos @ W_V  # (5, 2)

# Attention scores
scores = Q @ K.T / np.sqrt(d_head)  # (5, 5)
attention_weights = softmax(scores)  # (5, 5)

# Weighted combination
attention_output = attention_weights @ V  # (5, 2)

print("\n" + "=" * 60)
print("STEP 3: Single-Head Attention")
print("=" * 60)
print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
print(f"\nAttention scores (raw) shape: {scores.shape}")
print(f"Attention weights (after softmax) shape: {attention_weights.shape}")
print(f"\nAttention weights for 'cat' (word 1):")
for i, word in enumerate(input_sentence):
    print(f"  {word:8s}: {attention_weights[1, i]:.3f}")

# ============================================
# STEP 5: Multi-head attention
# ============================================

def multi_head_attention(x, n_heads, d_model):
    """Real multi-head attention with different Q, K, V for each head"""
    seq_len = x.shape[0]
    d_head = d_model // n_heads

    heads_output = []
    heads_weights = []

    for head_idx in range(n_heads):
        # Different W_Q, W_K, W_V for each head
        W_Q_h = np.random.randn(d_model, d_head) * 0.1
        W_K_h = np.random.randn(d_model, d_head) * 0.1
        W_V_h = np.random.randn(d_model, d_head) * 0.1

        Q_h = x @ W_Q_h
        K_h = x @ W_K_h
        V_h = x @ W_V_h

        scores_h = Q_h @ K_h.T / np.sqrt(d_head)
        weights_h = softmax(scores_h)
        output_h = weights_h @ V_h

        heads_output.append(output_h)
        heads_weights.append(weights_h)

    # Concatenate all heads
    multi_head_output = np.concatenate(heads_output, axis=-1)  # (5, 8)

    return multi_head_output, heads_weights

multi_head_output, all_heads_weights = multi_head_attention(input_with_pos, n_heads, d_model)

print("\n" + "=" * 60)
print("STEP 4: Multi-Head Attention (4 heads)")
print("=" * 60)
print(f"Multi-head output shape: {multi_head_output.shape}")
print(f"\nAttention patterns for 'cat' (word 1):")
for head_idx in range(n_heads):
    print(f"\nHead {head_idx + 1}:")
    for i, word in enumerate(input_sentence):
        print(f"  {word:8s}: {all_heads_weights[head_idx][1, i]:.3f}")

# ============================================
# STEP 6: Final prediction layer
# ============================================

# Output projection (simplified: just last token predicts next word)
W_out = np.random.randn(d_model, vocab_size) * 0.1
b_out = np.random.randn(vocab_size) * 0.01

# Use last token's representation to predict next word
last_token_repr = multi_head_output[-1, :]  # (8,)
logits = last_token_repr @ W_out + b_out  # (vocab_size,)
probs = softmax(logits.reshape(1, -1)).flatten()

# Get top predictions
top_k = 5
top_indices = np.argsort(probs)[::-1][:top_k]
top_words = [VOCAB[idx] for idx in top_indices]
top_probs = [probs[idx] for idx in top_indices]

print("\n" + "=" * 60)
print("STEP 5: Final Prediction")
print("=" * 60)
print(f"Input: {' '.join(input_sentence)} ___")
print(f"\nTop {top_k} predictions:")
for word, prob in zip(top_words, top_probs):
    print(f"  {word:10s}: {prob:.1%}")

# Store all data for chart generation
SIMULATION_DATA = {
    'vocab': VOCAB,
    'input_sentence': input_sentence,
    'input_ids': input_ids,
    'embeddings': input_embeddings,
    'pos_encoding': pos_encoding,
    'input_with_pos': input_with_pos,
    'attention_weights': attention_weights,
    'all_heads_weights': all_heads_weights,
    'multi_head_output': multi_head_output,
    'top_words': top_words,
    'top_probs': top_probs,
    'd_model': d_model,
    'n_heads': n_heads
}

print("\n" + "=" * 60)
print("Simulation complete! Data saved for chart generation.")
print("=" * 60)
