import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def self_attention(query, key, value, mask=None):

    d_m = query.shape[-1]
    scores = np.matmul(query, key.transpose(0, 2, 1)) / d_m

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = softmax(scores)
    outputs = np.matmul(scores, value)

    return outputs, attention_weights


embed_size = 4  
batch_size = 1
seq_len = 3     

np.random.seed(42)
query = np.random.randn(batch_size, seq_len, embed_size)
key = np.random.randn(batch_size, seq_len, embed_size)
value = np.random.randn(batch_size, seq_len, embed_size)

output, attention_weights = self_attention(query, key, value)

print("Output:")
print(output)
print("\nAttention Weights:")
print(attention_weights)
