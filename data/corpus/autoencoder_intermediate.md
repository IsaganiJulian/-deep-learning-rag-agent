# Autoencoders

## Autoencoder Architecture

An autoencoder is a neural network trained to reconstruct its input at the output layer. It consists of two parts: an encoder that maps the input to a lower-dimensional latent representation, and a decoder that maps the latent representation back to the original input space. The encoder compresses the data through one or more hidden layers of decreasing size. The decoder mirrors this structure with layers of increasing size. The network is trained by minimizing reconstruction loss — typically mean squared error for continuous inputs or binary cross-entropy for binary inputs. The bottleneck layer between encoder and decoder forces the network to learn a compressed, information-rich representation of the data.

## Latent Space

The latent space is the low-dimensional representation produced by the encoder. It is the compressed encoding of the input that the decoder must use to reconstruct the original. Well-structured latent spaces capture the most salient features of the data: similar inputs cluster together, and the dimensions of the latent space correspond to meaningful factors of variation. The size of the latent space is a key hyperparameter: too small and the autoencoder cannot capture enough information to reconstruct accurately; too large and the network may learn a trivial identity mapping. Visualizing the latent space with dimensionality reduction techniques like t-SNE reveals the structure the autoencoder has learned about the data distribution.

## Denoising Autoencoders

A denoising autoencoder (DAE) is trained to reconstruct clean inputs from corrupted versions. During training, noise is added to the input — for example, randomly zeroing pixels in an image or adding Gaussian noise — and the autoencoder learns to recover the original uncorrupted input. This forces the network to learn robust representations that capture the underlying data structure rather than memorizing noise. Denoising autoencoders are more powerful feature learners than standard autoencoders because they must learn what aspects of the input are signal versus noise. They are used for image denoising, data imputation, and as a pre-training technique where the learned representations are transferred to downstream supervised tasks.

## Variational Autoencoders

A Variational Autoencoder (VAE) is a generative model that extends the standard autoencoder with a probabilistic latent space. Instead of encoding an input as a single point, the encoder outputs the parameters of a probability distribution — typically a mean vector and a log-variance vector — from which the latent code is sampled. The decoder then reconstructs the input from this sample. Training minimizes the sum of reconstruction loss and a KL divergence term that regularizes the latent distribution to be close to a standard Gaussian. This regularization ensures the latent space is smooth and continuous, enabling generation of new samples by sampling from the prior and decoding. VAEs are widely used for image generation, anomaly detection, and data augmentation.

## Applications of Autoencoders

Autoencoders have a broad range of practical applications. Dimensionality reduction: autoencoders can learn non-linear projections to low-dimensional spaces, outperforming linear methods like PCA on complex data. Anomaly detection: an autoencoder trained on normal data will reconstruct normal inputs well but fail to reconstruct anomalies, making high reconstruction error a reliable anomaly signal. Image denoising: denoising autoencoders are used in medical imaging and photography to recover clean images from noisy inputs. Data compression: autoencoders learn compact representations tailored to a specific data distribution. Pre-training: autoencoders can initialize the weights of a deep network before fine-tuning on a supervised task, which was important before batch normalization made end-to-end training from random initialization reliable.
