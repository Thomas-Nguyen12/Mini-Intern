# Notes for Mini-Intern project

## What is a small language model? 

A small language model is one that has fewer than one billion parameters


## How are small language models trained? 

### Underlying architecture
Small language models are trained using the **transformer architecture** 

The transformer architecture uses self-attention mechanisms to process entire sequences of data simultaneously, allowing tokens to communicate and determine contextual relevance without processing words sequentially. 

- Tokenization and Embeddings: Raw text is split into smaller units called tokens, which are then converted into high-dimensional numeric vectors that capture semantic meaning.
- Positional Encoding: Because transformers process all tokens at once and have no inherent sense of order, positional information is added to the embeddings to preserve word sequence.
- Query, Key, and Value Vectors (Q, K, V): For every token, the model creates three distinct vectors via matrix multiplications:• Query: What information the token is looking for.
- Key: What information the token contains to match against queries.
- Value: The actual semantic content the token holds.
- Scaled Dot-Product and Multi-Head Attention: The model computes the dot product of queries and keys to score relevance, applies a softmax function to get attention weights, and takes a weighted sum of values. Multi-head attention runs multiple of these attention operations in parallel to capture different types of relationships.
- Feed-Forward Networks (MLPs): After attention mixing, each token passes through an identical multilayer perceptron independently to refine its internal representation.
- Residual Connections and Layer Normalization: Added around sub-layers to stabilize training, prevent overfitting, and mitigate vanishing gradients.


When trained from the ground up, SLMs use the same core transformer architecture as large language models (LLMs).
- Next-Token Prediction: The model reads a sequence of text and learns by guessing the next piece of text (token).
- Backpropagation: It compares its guess to the actual text, measures the error using a loss function, and updates its internal weights to become more accurate.
- Curated Data: Unlike massive LLMs trained on raw internet scrapes, SLMs rely heavily on smaller, high-quality, and synthetic datasets (such as clean code repositories or textbook-quality data) where quality matters more than sheer volume.

### Evaluation 

Evaluation occurs at various stages of model training: 

**Pre-training**

During pre-training, SLMs use **Cross-entropy Loss** as their primary loss function for **next token prediction**. This measures the difference in the model's predicted probability distribution and the actual true next token in the text sequence.

The formula for **Cross entropy loss** is: 

## Per-token loss

For a single prediction at position $t$, given context $x_{<t}$ and the true next token $x_t$:

$$
\mathcal{L}_t = -\log p_\theta(x_t \mid x_{<t})
$$

## Sequence-level loss (average over tokens)

For a sequence $x = (x_1, x_2, \dots, x_T)$:

$$
\mathcal{L}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

## Dataset-level loss

Averaged over $N$ sequences, where sequence $i$ has length $T_i$:

$$
\mathcal{L}(\theta) = -\frac{1}{\sum_{i=1}^{N} T_i} \sum_{i=1}^{N} \sum_{t=1}^{T_i} \log p_\theta\left(x_t^{(i)} \mid x_{<t}^{(i)}\right)
$$

## General form (one-hot target)

Written as cross-entropy between the true distribution $q$ and the model distribution $p_\theta$ over vocabulary $V$:

$$
H(q, p_\theta) = -\sum_{v \in V} q(v) \log p_\theta(v \mid x_{<t})
$$

Since $q$ is one-hot (1 for the true token $x_t$, 0 otherwise), this reduces to $-\log p_\theta(x_t \mid x_{<t})$.

## Softmax over logits

With logits $z \in \mathbb{R}^{|V|}$ produced by the model:

$$
p_\theta(v \mid x_{<t}) = \frac{\exp(z_v)}{\sum_{v' \in V} \exp(z_{v'})}
$$

$$
\mathcal{L}_t = -z_{x_t} + \log \sum_{v' \in V} \exp(z_{v'})
$$

## Related: Perplexity

$$
\text{PPL} = \exp\big(\mathcal{L}(\theta)\big)
$$

**Knowledge distillation** 

> Where a large, complex model (the teacher) transfers its learning and reasoning patterns to a smaller, faster model

The most common metric used here is the **Kullback-Leibler (KL) Divergence Loss**

This forces the smaller student model to mimic the exact output probability distribution of a larger, more powerful teacher model.



**Fine Tuning**

SLMs may use specialised ranking or binary losses to align the model's outputs with human preferences and safety guidelines








### RAG



### Datasets
Typically, language models use a **base dataset** for general language learning and grammar, and then a **tuning dataset** to focus the model on specific tasks. For example, 





## Data Source for this project

For small language models, there are several different potential datasets, with different purposes. 

**Popular choices include:** 

- Tiny Shakespeare
- Alpaca dataset - A stanford dataset meant for training language models to receive instructions 
- Cosmopedia: The largest open soruce synthetic dataset for pretraining language models 

The toy version of Cosmopedia is a manageable 100,000-sample subset of the larger 30-million-file synthetic pre-training dataset generated by Hugging Face. 

This dataset contains educational texts, including synthetic textbooks, blog stories, forum posts, WikiHow articles. 
