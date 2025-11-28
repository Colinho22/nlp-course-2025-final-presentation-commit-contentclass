"""
Educational Transformer Simulation with Semantically Meaningful Patterns
Designed to show WHY multi-head attention works
Input: "The cat sat on the ___"
Output: Predict "mat" with interpretable attention patterns
"""

import numpy as np
import pickle

# Set random seed
np.random.seed(42)

# ============================================
# STEP 1: Create SEMANTIC embeddings
# ============================================

VOCAB = [
    'The', 'cat', 'sat', 'on', 'the', 'mat', 'floor', 'table', 'rug',
    'dog', 'bird', 'ran', 'flew', 'in', 'under', 'over', 'chair',
    'bed', 'sofa', 'window'
]

vocab_size = len(VOCAB)
d_model = 8

# Create embeddings with semantic structure
embedding_matrix = np.zeros((vocab_size, d_model))

# Dimension roles:
# [0-1]: Animal/Object type
# [2-3]: Action/State
# [4-5]: Furniture/Location
# [6-7]: Grammar role

# Articles
embedding_matrix[0] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # The
embedding_matrix[4] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # the (same)

# Animals (high on dim 0)
embedding_matrix[1] = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1]  # cat
embedding_matrix[9] = [0.9, 0.2, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1]  # dog
embedding_matrix[10] = [0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1]  # bird

# Actions (high on dim 2)
embedding_matrix[2] = [0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.3, 0.2]  # sat
embedding_matrix[11] = [0.0, 0.0, 0.8, 0.3, 0.0, 0.0, 0.3, 0.2]  # ran
embedding_matrix[12] = [0.0, 0.0, 0.7, 0.4, 0.0, 0.0, 0.3, 0.2]  # flew

# Prepositions (high on dim 6-7)
embedding_matrix[3] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.8]  # on
embedding_matrix[13] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.7]  # in
embedding_matrix[14] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.6]  # under
embedding_matrix[15] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.5]  # over

# Furniture (high on dim 4-5)
embedding_matrix[5] = [0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.2, 0.3]  # mat
embedding_matrix[6] = [0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.2, 0.3]  # floor
embedding_matrix[7] = [0.0, 0.0, 0.0, 0.0, 0.7, 0.4, 0.2, 0.3]  # table
embedding_matrix[8] = [0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.2, 0.3]  # rug
embedding_matrix[16] = [0.0, 0.0, 0.0, 0.0, 0.7, 0.3, 0.2, 0.3]  # chair
embedding_matrix[17] = [0.0, 0.0, 0.0, 0.0, 0.8, 0.3, 0.2, 0.3]  # bed
embedding_matrix[18] = [0.0, 0.0, 0.0, 0.0, 0.7, 0.5, 0.2, 0.3]  # sofa
embedding_matrix[19] = [0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 0.2, 0.3]  # window

# Input sentence
input_sentence = ['The', 'cat', 'sat', 'on', 'the']
input_ids = [VOCAB.index(w) for w in input_sentence]
input_embeddings = embedding_matrix[input_ids]

print("=" * 70)
print("EDUCATIONAL TRANSFORMER SIMULATION")
print("=" * 70)
print(f"Input: {' '.join(input_sentence)} ___")
print(f"Goal: Predict 'mat' (or similar furniture)")
print("\n" + "=" * 70)
print("STEP 1: Word Embeddings (Semantic Structure)")
print("=" * 70)
print("Embedding dimensions represent:")
print("  [0-1]: Animal/Object type")
print("  [2-3]: Action/State")
print("  [4-5]: Furniture/Location")
print("  [6-7]: Grammar role")
print()
for i, word in enumerate(input_sentence):
    print(f"{word:5s}: {input_embeddings[i]}")

# ============================================
# STEP 2: Positional encoding
# ============================================

def get_positional_encoding(seq_len, d_model):
    pos_enc = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (2 * i / d_model)))
    return pos_enc

pos_encoding = get_positional_encoding(len(input_sentence), d_model)
input_with_pos = input_embeddings + pos_encoding * 0.3  # Scale down position for clarity

print("\n" + "=" * 70)
print("STEP 2: Add Positional Encoding")
print("=" * 70)
print("Position encoding tells model WHERE each word is in the sentence")
print()
for i, word in enumerate(input_sentence):
    print(f"Pos {i} ({word:5s}): {pos_encoding[i, :4]} ... (showing first 4 dims)")

# ============================================
# STEP 3: Interpretable multi-head attention
# ============================================

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

n_heads = 4
d_head = d_model // n_heads  # 2

# Design interpretable attention heads
heads_output = []
all_heads_weights = []

print("\n" + "=" * 70)
print("STEP 3: Multi-Head Attention (4 Specialized Heads)")
print("=" * 70)

# HEAD 1: Grammar patterns (verbs attend to subjects)
print("\nHEAD 1: Grammar Patterns")
print("Purpose: Verbs look at subjects, prepositions look at verbs")
W_Q1 = np.eye(d_model, d_head) * [0.0, 0.0]  # Focus on grammar dims
W_K1 = np.eye(d_model, d_head) * [1.0, 0.0]  # Focus on objects
W_V1 = np.eye(d_model, d_head) * [1.0, 0.5]
Q1 = input_with_pos @ W_Q1
K1 = input_with_pos @ W_K1
V1 = input_with_pos @ W_V1
scores1 = Q1 @ K1.T / np.sqrt(d_head)
weights1 = softmax(scores1)
output1 = weights1 @ V1
heads_output.append(output1)
all_heads_weights.append(weights1)

print("Attention for 'sat' (word 2):")
for i, word in enumerate(input_sentence):
    print(f"  {word:5s}: {weights1[2, i]:.3f}")

# HEAD 2: Meaning/Semantics (furniture words cluster)
print("\nHEAD 2: Semantic Relationships")
print("Purpose: Find semantically related words (cat -> animals, on -> locations)")
W_Q2 = np.eye(d_model, d_head) * [0.5, 0.5]  # Semantic dims
W_K2 = np.eye(d_model, d_head) * [0.5, 0.5]
W_V2 = np.eye(d_model, d_head) * [1.0, 0.8]
Q2 = input_with_pos @ W_Q2
K2 = input_with_pos @ W_K2
V2 = input_with_pos @ W_V2
scores2 = Q2 @ K2.T / np.sqrt(d_head)
weights2 = softmax(scores2)
output2 = weights2 @ V2
heads_output.append(output2)
all_heads_weights.append(weights2)

print("Attention for 'the' (last word, word 4):")
for i, word in enumerate(input_sentence):
    print(f"  {word:5s}: {weights2[4, i]:.3f}")

# HEAD 3: Local position (attend to nearby words)
print("\nHEAD 3: Positional Patterns")
print("Purpose: Words attend to nearby neighbors")
# Create position-based W matrices
W_Q3 = np.random.randn(d_model, d_head) * 0.2
W_K3 = W_Q3.copy()  # Same Q and K creates local attention
W_V3 = np.random.randn(d_model, d_head) * 0.5
Q3 = input_with_pos @ W_Q3
K3 = input_with_pos @ W_K3
V3 = input_with_pos @ W_V3
# Add position bias to favor nearby words
position_bias = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :])
scores3 = (Q3 @ K3.T - position_bias * 0.5) / np.sqrt(d_head)
weights3 = softmax(scores3)
output3 = weights3 @ V3
heads_output.append(output3)
all_heads_weights.append(weights3)

print("Attention for 'on' (word 3):")
for i, word in enumerate(input_sentence):
    print(f"  {word:5s}: {weights3[3, i]:.3f}")

# HEAD 4: Global context (attend to first/last words)
print("\nHEAD 4: Global Context")
print("Purpose: Look at sentence boundaries and key words")
W_Q4 = np.random.randn(d_model, d_head) * 0.2
W_K4 = np.random.randn(d_model, d_head) * 0.2
W_V4 = np.random.randn(d_model, d_head) * 0.5
Q4 = input_with_pos @ W_Q4
K4 = input_with_pos @ W_K4
V4 = input_with_pos @ W_V4
# Bias toward first and last positions
global_bias = np.zeros((5, 5))
global_bias[:, 0] = 1.0  # Attend to first word
global_bias[:, -1] = 1.0  # Attend to last word
scores4 = (Q4 @ K4.T + global_bias) / np.sqrt(d_head)
weights4 = softmax(scores4)
output4 = weights4 @ V4
heads_output.append(output4)
all_heads_weights.append(weights4)

print("Attention for 'cat' (word 1):")
for i, word in enumerate(input_sentence):
    print(f"  {word:5s}: {weights4[1, i]:.3f}")

# Concatenate all heads
multi_head_output = np.concatenate(heads_output, axis=-1)

# ============================================
# STEP 4: Final prediction
# ============================================

# Create output layer that favors furniture after "on the"
W_out = np.random.randn(d_model, vocab_size) * 0.1

# Boost furniture words when dims 4-5 are high
furniture_indices = [5, 6, 7, 8, 16, 17, 18]  # mat, floor, table, rug, chair, bed, sofa
W_out[4:6, furniture_indices] += 0.8  # Boost furniture if furniture dims are active

b_out = np.random.randn(vocab_size) * 0.01
b_out[furniture_indices] += 0.5  # General boost for furniture

last_token = multi_head_output[-1, :]
logits = last_token @ W_out + b_out
probs = softmax(logits.reshape(1, -1)).flatten()

# Get top predictions
top_k = 8
top_indices = np.argsort(probs)[::-1][:top_k]
top_words = [VOCAB[idx] for idx in top_indices]
top_probs = [probs[idx] for idx in top_indices]

print("\n" + "=" * 70)
print("STEP 4: Final Prediction")
print("=" * 70)
print(f"Input: {' '.join(input_sentence)} ___")
print(f"\nTop {top_k} predictions:")
for word, prob in zip(top_words, top_probs):
    is_furniture = word in ['mat', 'floor', 'table', 'rug', 'chair', 'bed', 'sofa']
    marker = " <-- FURNITURE!" if is_furniture else ""
    print(f"  {word:10s}: {prob:6.1%}{marker}")

# Save all data for charts
SIMULATION_DATA = {
    'vocab': VOCAB,
    'input_sentence': input_sentence,
    'input_ids': input_ids,
    'embeddings': input_embeddings,
    'pos_encoding': pos_encoding,
    'input_with_pos': input_with_pos,
    'all_heads_weights': all_heads_weights,
    'multi_head_output': multi_head_output,
    'top_words': top_words,
    'top_probs': top_probs,
    'd_model': d_model,
    'n_heads': n_heads,
    'embedding_matrix': embedding_matrix
}

# Save to pickle for chart generation
with open('simulation_data.pkl', 'wb') as f:
    pickle.dump(SIMULATION_DATA, f)

print("\n" + "=" * 70)
print("[OK] Simulation complete! Data saved to simulation_data.pkl")
print("=" * 70)
