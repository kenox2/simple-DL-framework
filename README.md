# Custom Neural Network Framework

This repository implements a simple neural network framework in Python using **NumPy**. It supports multiple loss functions, activation functions, regularization techniques, dropout, weight noise, and mini-batch training. The goal of this project is to create an easily extensible and educational neural network framework.

## Features

- **Fully connected layers** (Dense layers).
- **Activation functions**: ReLU, Softmax.
- **Loss functions**: Cross-Entropy, Mean Squared Error (MSE).
- **Regularization**: L1 and L2 regularization.
- **Weight Noise**: Add random noise to weights during training to improve generalization.
- **Dropout**: Dropout regularization during training to prevent overfitting.
- **Mini-batch Training**: Efficient training using mini-batches to improve performance on large datasets.
- **Training/Inference Modes**: Switch between training mode (with dropout and noise) and inference mode (for evaluation).

## Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/neural-network-framework.git
cd neural-network-framework
pip install numpy
```

## Usage

### Example Code

Here’s an example of how to define a simple neural network, train it using mini-batches, and switch between training and inference modes.

```python
import numpy as np
from network import Network, Layer, Loss, Activation

# Define the neural network layers
layer1 = Layer(num_in=784, num_out=128, activation=Activation.relu, activation_deriv=Activation.relu_derivative, 
               learning_rate=0.01, dropout=0.8, l1_lambda=0.01, l2_lambda=0.01, weight_noise_std=0.001)
layer2 = Layer(num_in=128, num_out=10, activation=Activation.softmax, activation_deriv=None)

# Create the network
network = Network(layers=[layer1, layer2])

# Switch to training mode (default)
network.training_mode()

# Example mini-batch gradient descent training loop
batch_size = 32
epochs = 10
for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        # Forward pass
        output = network.forward(X_batch)
        
        # Compute the loss
        loss = Loss.cross_entropy(y_batch, output)
        
        # Compute the gradients
        grad = Loss.cross_entropy_gradient(y_batch, output)
        
        # Backward pass
        network.backward(grad)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

# Switch to inference mode when evaluating the model
network.inference_mode()
```

### Training and Inference Modes

You can easily switch between **training mode** and **inference mode**:

- **Training Mode**: Used when training the model with dropout, weight noise, and other regularization methods.
  - Activate by calling `network.training_mode()`.
  
- **Inference Mode**: Used when evaluating or deploying the model (no dropout or noise).
  - Activate by calling `network.inference_mode()`.

### Mini-batch Training

The framework supports **mini-batch training**. Mini-batch training improves training efficiency by splitting the dataset into small batches and performing updates for each batch.

- In the training loop, the dataset is divided into mini-batches, and for each batch:
  - The forward pass is computed.
  - The loss is calculated.
  - The backward pass updates the weights.

Example mini-batch usage:

```python
batch_size = 32
for i in range(0, len(X_train), batch_size):
    X_batch = X_train[i:i+batch_size]
    y_batch = y_train[i:i+batch_size]
    
    # Perform forward and backward passes for each mini-batch
    output = network.forward(X_batch)
    loss = Loss.cross_entropy(y_batch, output)
    grad = Loss.cross_entropy_gradient(y_batch, output)
    network.backward(grad)
```

### Regularization

L1 and L2 regularization are supported:

- **L1 regularization**: Adds a penalty based on the absolute values of the weights, helping to sparsify the model.
- **L2 regularization**: Adds a penalty based on the squared values of the weights, helping to prevent large weights and overfitting.

Both regularization terms can be added during training:

```python
loss += Loss.regularization_loss(layers=[layer1, layer2], l1=True, l2=True)
```

### Weight Noise

To prevent overfitting, you can add **random noise** to the weights during training. This is especially useful in situations where you have small datasets or noisy data. To use weight noise, simply specify a `weight_noise_std` value when defining a layer:

```python
layer1 = Layer(num_in=784, num_out=128, activation=Activation.relu, weight_noise_std=0.001)
```

### Loss Functions

- **Cross-Entropy Loss**: Commonly used for classification problems, particularly with a softmax output layer.
- **Mean Squared Error (MSE)**: Used for regression tasks.

Each loss function also has an associated gradient for backpropagation:

```python
grad = Loss.cross_entropy_gradient(y_true, y_pred)
```

### Activation Functions

- **ReLU**: A popular activation function for hidden layers, which helps mitigate the vanishing gradient problem.
- **Softmax**: Used in the output layer for classification tasks, converting the output into probabilities.

## Requirements

- **NumPy**: A numerical computing library for Python. Install it with:

```bash
pip install numpy
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
