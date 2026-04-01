# Long Short-Term Memory Networks

## Solving the Vanishing Gradient Problem

LSTMs were introduced by Hochreiter and Schmidhuber in 1997 specifically to solve the vanishing gradient problem that prevents standard RNNs from learning long-range dependencies. The key innovation is the cell state: a separate memory channel that runs alongside the hidden state and can carry information across many time steps with minimal transformation. Because the cell state is updated through addition rather than repeated matrix multiplication, gradients can flow backward through time without exponential decay. The gating mechanisms selectively control what information is written to, read from, and erased from the cell state, giving the network fine-grained control over its long-term memory.

## The Forget Gate

The forget gate is the first gate an input encounters inside an LSTM cell. It decides what information to discard from the cell state. The gate takes the previous hidden state and the current input, concatenates them, passes them through a learned weight matrix, and applies a sigmoid activation. The sigmoid output is a vector of values between zero and one. Values close to zero tell the network to forget the corresponding dimension of the cell state; values close to one tell it to keep that information. This output is multiplied element-wise with the cell state, selectively erasing information that is no longer relevant to the current context.

## The Input Gate

The input gate controls what new information is written to the cell state at the current time step. It has two components. First, a sigmoid layer called the input gate layer decides which dimensions of the cell state to update, producing values between zero and one. Second, a tanh layer creates a vector of candidate values that could be added to the cell state, with values between -1 and 1. The element-wise product of these two vectors determines the actual update: dimensions where the input gate is close to one receive a large update; dimensions where it is close to zero are left unchanged. This update is then added to the cell state scaled by the forget gate output.

## The Output Gate

The output gate controls what information is read from the cell state to produce the current hidden state, which is passed to the next time step and to downstream layers. First, a sigmoid layer decides which parts of the cell state to output, producing a mask between zero and one. Then the cell state is passed through tanh to normalize its values between -1 and 1. The element-wise product of the sigmoid mask and the tanh-transformed cell state produces the hidden state output. This two-stage process allows the LSTM to maintain a rich internal memory in the cell state while exposing only the relevant subset as its visible output at each time step.

## LSTM vs RNN

The fundamental difference between LSTMs and standard RNNs is the presence of the cell state and gating mechanisms. A vanilla RNN has a single hidden state updated by a simple tanh nonlinearity at every step, which causes gradients to vanish over long sequences. An LSTM maintains both a hidden state and a cell state, using three learnable gates to regulate information flow. This architecture allows LSTMs to selectively remember information for hundreds of time steps, something RNNs cannot reliably do. In practice, LSTMs outperform RNNs on tasks requiring long-range dependencies such as machine translation, language modeling, and speech recognition. The GRU is a simplified variant with two gates that achieves similar performance with fewer parameters.
