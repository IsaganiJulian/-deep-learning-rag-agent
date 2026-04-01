# Recurrent Neural Networks

## Hidden State and Sequential Processing

A recurrent neural network processes sequences by maintaining a hidden state that is updated at each time step. At time t, the network receives the current input and the previous hidden state, combines them through a learned weight matrix and activation function, and produces an updated hidden state. This hidden state acts as a compressed memory of all inputs seen so far. Unlike feedforward networks, the same weights are reused at every time step, making RNNs parameter-efficient for sequential data. The hidden state is passed forward to the next time step, allowing information to persist across the sequence. This recurrent connection is what gives RNNs their ability to model temporal dependencies.

## Backpropagation Through Time

Backpropagation Through Time (BPTT) is the algorithm for training RNNs. The network is unrolled across all time steps, transforming the recurrent structure into a deep feedforward network where each layer corresponds to one time step. Gradients are then computed using standard backpropagation through this unrolled graph. The gradient at each time step depends on the gradient at the next step, so the total gradient is a product of many Jacobians across all time steps. This is computationally expensive for long sequences. Truncated BPTT addresses this by limiting the number of time steps over which gradients are propagated, trading accuracy for efficiency.

## Vanishing Gradients in RNNs

RNNs suffer severely from the vanishing gradient problem during BPTT. Because the same weight matrix is multiplied repeatedly across all time steps, its gradient either shrinks exponentially toward zero (vanishing) or grows exponentially (exploding) as the sequence length increases. Vanishing gradients prevent the network from learning long-range dependencies — information from many steps ago cannot influence the current weight update. Gradient clipping addresses exploding gradients by capping the gradient norm. Vanishing gradients are more fundamental and motivated the development of gated architectures: LSTMs and GRUs introduce gating mechanisms that allow gradients to flow over long sequences without exponential decay.

## Applications of RNNs

RNNs are suited to any task where the order of inputs matters. Language modeling trains an RNN to predict the next word in a sequence, learning statistical regularities of text. Sentiment analysis classifies the overall tone of a document by processing it word by word. Machine translation using early RNNs encoded a source sentence into a fixed-size vector and decoded it into the target language, though this bottleneck limited performance on long sequences. Speech recognition maps audio frames to phonemes or characters over time. Time series forecasting uses RNNs to predict future values from historical data. Despite being largely supplanted by transformers in NLP, RNNs remain valuable for low-latency or resource-constrained sequential tasks.
