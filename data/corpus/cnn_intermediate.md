# Convolutional Neural Networks

## Convolution Operation

The convolution operation is the core building block of CNNs. A small matrix called a filter or kernel slides across the input image with a defined stride, computing an element-wise dot product at each position. This produces a scalar value at each location, and the collection of these values forms a feature map. The filter learns to detect a specific local pattern such as an edge or texture. Multiple filters are applied in parallel to detect different features simultaneously. Because the same filter is applied across the entire input, CNNs exploit spatial locality and achieve translation invariance — the same feature is detected regardless of where it appears in the image.

## Pooling Layers

Pooling layers reduce the spatial dimensions of feature maps, decreasing computational load and making the representation progressively more abstract. Max pooling takes the maximum value within each pooling window, retaining the most prominent activation. Average pooling computes the mean. A typical 2x2 max pooling with stride 2 halves both height and width. Pooling provides a degree of translation invariance: a feature detected slightly off-center still produces a strong pooled response. Global average pooling, used in modern architectures, collapses each feature map to a single value and serves as an alternative to fully connected layers.

## Feature Maps

Feature maps are the outputs of convolutional layers. Each filter in a layer produces one feature map, so a layer with 64 filters produces 64 feature maps. Early layers learn low-level features such as edges and color gradients. Middle layers combine these into textures and shapes. Deep layers represent high-level semantic concepts such as faces or objects. This hierarchical feature learning is why CNNs are effective for image tasks: they automatically discover the feature hierarchy that humans would otherwise engineer by hand. Visualizing feature maps is a standard debugging and interpretability technique.

## LeNet Architecture

LeNet, introduced by LeCun et al. in 1998, was one of the first successful CNN architectures, designed for handwritten digit recognition on the MNIST dataset. It consists of two convolutional layers each followed by average pooling, then three fully connected layers ending in a softmax output. LeNet demonstrated that convolutional layers could automatically learn spatial feature hierarchies without manual feature engineering. It established the pattern of alternating convolution and pooling layers that remains standard today. Although small by modern standards, LeNet proved that gradient-based learning of convolution filters was practical and effective.

## AlexNet Architecture

AlexNet, winner of the 2012 ImageNet competition, demonstrated that deep CNNs trained on GPUs could dramatically outperform traditional computer vision methods. It stacked five convolutional layers followed by three fully connected layers, used ReLU activations instead of sigmoid, applied dropout for regularization, and used local response normalization. Its top-5 error rate of 15.3% was more than 10 percentage points better than the second-place entry, triggering widespread adoption of deep learning in computer vision. AlexNet showed that depth, GPU training, large datasets, and regularization techniques could be combined to achieve human-competitive performance on large-scale image classification.
