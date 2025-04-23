
# Loss functions and gradients
class Loss:
    @staticmethod
    def cross_entropy(y_true, y_pred):
        # Clip values to avoid log(0) issues
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        loss = -np.mean( np.sum(y_true * np.log(y_pred), axis=1) )
        return loss

    @staticmethod
    def cross_entropy_gradient(y_true, y_pred):
        return y_pred - y_true

    @staticmethod
    def MSE(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)/2

    @staticmethod
    def MSE_gradient(y_true, y_pred):
        return y_pred - y_true
    

    @staticmethod
    def regularization_loss(layers, l1 = False, l2=True):
      l1_loss = 0
      l2_loss = 0
      for layer in layers:
          if hasattr(layer, 'l1_lambda') and hasattr(layer, 'l2_lambda'):
              if l1:
                l1_loss += layer.l1_lambda * np.sum(np.abs(layer.weights))
              if l2:
                l2_loss += layer.l2_lambda * np.sum(layer.weights ** 2)
      return l1_loss + 0.5 * l2_loss 

# Activation functions and derivatives
class Activation:
    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return np.where(z > 0, 1, 0)

    @staticmethod
    def softmax(z):
        # Subtract max for numerical stability
        z -= np.max(z, axis=1, keepdims=True)
        exp_vals = np.exp(z)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    

    # For softmax used with cross-entropy, the derivative is taken care of in the loss gradient

# Fully connected layer definition
class Layer:
    def __init__(self, num_in, num_out, activation, activation_deriv, learning_rate=0.01, dropout = 1.0, isTraining = True, l1_lambda = 0.0, l2_lambda = 0.0, weight_noise_std = 0.0):
        self.rng = np.random.default_rng(42)
        # He initialization for layers using ReLU
        self.weights = self.rng.standard_normal((num_in, num_out)) * np.sqrt(2 / num_in)
        self.biases = np.zeros((1, num_out))
        self.activation = activation
        self.activation_deriv = activation_deriv
        self.learning_rate = learning_rate
        self.inputs = None
        self.z = None
        if dropout == 0: dropout = 0.2
        self.dropout = dropout
        self.isTraining = isTraining
        self.mask = None
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.noise_std = weight_noise_std

    def generate_binary_matrix(self, num_features):
      # One dropout mask for the whole batch
      return (np.random.rand(1, num_features) < self.dropout).astype(np.float32)

    def forward(self, inputs):
        if self.isTraining and self.noise_std > 0:
          noise = self.rng.normal(0, self.noise_std, self.weights.shape)
          weights_noisy = self.weights + noise
        else:
          weights_noisy = self.weights
        if self.dropout != 1.0 and self.isTraining:
          self.mask = self.generate_binary_matrix(inputs.shape[1])
          inputs = inputs * self.mask
        self.inputs = inputs
        self.z = np.dot(inputs, weights_noisy) + self.biases
        output = self.activation(self.z)

        if self.dropout < 1.0 and self.isTraining:
            output = output / self.dropout

        return output


    def backward(self, dl_dy):
        # For layers with ReLU or similar activations
        if self.activation_deriv is not None:
            dl_dz = dl_dy * self.activation_deriv(self.z)
        else:
            dl_dz = dl_dy
        dl_dw = self.inputs.T @ dl_dz
        dl_db = np.sum(dl_dz, axis=0, keepdims=True)
        dl_dx = np.dot(dl_dz, self.weights.T)

        if self.dropout < 1.0 and self.isTraining and self.mask is not None:
          dl_dx *= self.mask

        batch_size = self.inputs.shape[0]
        self.weights -= self.learning_rate * ((dl_dw / batch_size) + self.l1_lambda * np.sign(self.weights) + self.l2_lambda * self.weights)
        self.biases  -= self.learning_rate * (dl_db / batch_size)
        return dl_dx

    def inference_mode(self):
      self.isTraining = False
      return self

    def training_mode(self):
      self.isTraining = True
      return self

class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, dl_dy):
        for layer in reversed(self.layers):
            dl_dy = layer.backward(dl_dy)
        return dl_dy

    def inference_mode(self):
      for layer in self.layers:
        layer.inference_mode()
      return self

    def training_mode(self):
      for layer in self.layers:
        layer.training_mode()
      return self
