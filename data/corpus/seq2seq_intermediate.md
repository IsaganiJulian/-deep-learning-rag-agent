# Sequence-to-Sequence Models

## Encoder-Decoder Architecture

The Seq2Seq (sequence-to-sequence) architecture consists of two components: an encoder and a decoder, each typically implemented as an LSTM or GRU. The encoder reads the entire input sequence and compresses it into a fixed-size context vector, also called the thought vector. This vector is intended to capture the meaning of the entire input. The decoder then uses this context vector as its initial hidden state and generates the output sequence one token at a time, feeding each generated token back as input to produce the next. This architecture enabled end-to-end trainable machine translation and other sequence transformation tasks, replacing earlier pipeline-based approaches that required manual feature engineering.

## The Encoder

The encoder processes the input sequence token by token, updating its hidden state at each step. After processing the final token, the encoder's hidden state is used as the context vector passed to the decoder. In LSTM-based encoders, both the cell state and hidden state are typically transferred. The encoder learns to compress variable-length input sequences into a fixed-dimensional representation that captures the semantic content needed by the decoder. Bidirectional encoders process the sequence in both forward and backward directions and concatenate the resulting hidden states, giving the encoder context from both past and future tokens at each position. This significantly improves the quality of the context vector.

## The Decoder

The decoder generates the output sequence autoregressively: it produces one token at a time, using the previously generated token as input for the next step. At the first step, the decoder receives the context vector from the encoder as its initial hidden state and a special start-of-sequence token as input. At each subsequent step, it samples or takes the argmax of the output probability distribution to select the next token, then feeds that token back as input. This process continues until the decoder produces an end-of-sequence token. At training time, teacher forcing is used: the ground-truth previous token is fed as input rather than the model's own prediction, which accelerates convergence but can cause exposure bias at inference time.

## Relation to Autoencoders

When the encoder and decoder of a Seq2Seq model share the same architecture and the input and output sequences are identical, the model is functionally equivalent to a sequence autoencoder. The context vector plays the same role as the latent representation in a standard autoencoder. This connection is exploited in pre-training: Seq2Seq models can be trained as denoising sequence autoencoders by corrupting the input and training the model to reconstruct the original. Both architectures suffer from the same fixed-size bottleneck: compressing an arbitrarily long sequence into a single vector loses information. Attention was introduced to solve this problem in Seq2Seq models in the same way that variational autoencoders extend standard autoencoders.

## Attention Mechanism

The attention mechanism, introduced by Bahdanau et al. in 2015, addresses the fixed-size bottleneck of the basic Seq2Seq architecture. Instead of forcing the entire input sequence into a single context vector, attention allows the decoder at each generation step to look back at all encoder hidden states and compute a weighted sum. The weights are determined by a learned alignment model that scores how relevant each encoder state is to the current decoding step. This allows the decoder to focus on different parts of the input at each step — for example, attending to the subject when generating a subject pronoun. Attention dramatically improved translation quality, especially for long sequences, and was the precursor to the transformer's self-attention mechanism.
