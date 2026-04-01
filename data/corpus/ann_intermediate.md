# Artificial Neural Networks

## Forward Propagation

Forward propagation is the process by which an artificial neural network computes its output from a given input. Data flows from the input layer through one or more hidden layers to the output layer. At each layer, every neuron computes a weighted sum of its inputs, adds a bias term, and applies a non-linear activation function. The result is passed as input to the next layer. This sequential computation produces a prediction that can be compared to the true label using a loss function.

## Backpropagation

Backpropagation is the algorithm used to train neural networks by computing gradients of the loss function with respect to every weight in the network. It applies the chain rule of calculus, propagating the error signal from the output layer backward through each hidden layer. At each layer, the gradient of the loss with respect to the weights is computed and used to update those weights via gradient descent. Backpropagation made training deep networks feasible and remains the foundation of modern deep learning optimization.

## Activation Functions

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns beyond simple linear transformations. Common choices include sigmoid, which outputs values between 0 and 1 and was historically popular; tanh, which outputs between -1 and 1 and is zero-centered; and ReLU (Rectified Linear Unit), which outputs max(0, x) and is the default in most modern architectures. ReLU is preferred because it is computationally efficient, does not saturate for positive inputs, and empirically accelerates training. Leaky ReLU and ELU variants address the dying ReLU problem where neurons output zero permanently.

## Loss Functions

A loss function measures the discrepancy between the network's predictions and the true labels, providing the signal that backpropagation uses to update weights. For binary classification, binary cross-entropy is standard. For multi-class classification, categorical cross-entropy compares the predicted probability distribution to the one-hot encoded true label. For regression tasks, mean squared error (MSE) penalizes large errors more heavily than small ones. The choice of loss function must match the output activation: softmax output pairs with categorical cross-entropy; sigmoid output pairs with binary cross-entropy.

## Vanishing Gradients

The vanishing gradient problem occurs during backpropagation when gradients become exponentially small as they are propagated through many layers, causing weights in early layers to receive negligible updates and learn very slowly or not at all. It is especially severe with sigmoid and tanh activations, whose derivatives are always less than one, so repeated multiplication across layers drives gradients toward zero. Solutions include using ReLU activations, careful weight initialization (Xavier or He initialization), batch normalization, and residual connections. The vanishing gradient problem was a primary motivation for developing LSTMs and other gated architectures.
