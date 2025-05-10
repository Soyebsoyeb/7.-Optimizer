// OPTIMIZERS IN NEURAL NETWORK

ABOUT THE STRUCTURE OF THE DATASET:->
🔹 Dense layers
🔹 Activation functions (ReLU & Softmax)
🔹 Loss function (Categorical Cross-Entropy)
🔹 Optimization.

Core Components:->
Layer_Dense: Fully connected neural network layer 🔗
Activation_ReLU: ReLU activation (with backpropagation) ⚡
Activation_Softmax: Stable softmax function 📈
Loss_CategoricalCrossentropy: Categorical loss with gradients 🎯
Activation_Softmax_Loss_CategoricalCrossentropy: Efficient combo 🔥


Training
✅ Complete forward & backward propagation
✅ SGD optimizer implementation
📊 Accuracy & loss tracking


🖼️ Visualization
Spiral data visualization 🌪️
Optional training progress plots 📉



Key Implementation Details
🔁 Forward Propagation
Layer math:
output = inputs @ weights + biases

ReLU:
max(0, x)

Softmax (w/ numerical stability):
exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)



🔄 Backward Propagation
GRADIENTS:->
Weights: inputs.T @ dvalues
Biases: np.sum(dvalues, axis=0)
Inputs: dvalues @ weights.T
ReLU derivative: 0 where input < 0
Combined Softmax + CrossEntropy: Optimized for performance ⚡


⚙️ Optimization

(1) GRADIENT DESCENT
Stochastic Gradient Descent (SGD):
layer.weights += -learning_rate * layer.dweights
layer.biases += -learning_rate * layer.dbiases





(2) DECAY_RATE
# Optimizer updates model weights and biases to minimize the loss function.

class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay=0.):
        # 🔢 Initial learning rate
        self.learning_rate = learning_rate
        # 📉 Learning rate decay (optional)
        self.decay = decay
        # ↪️ Tracks how many updates have been done
        self.iterations = 0
        # 🎯 Learning rate used in current iteration
        self.current_learning_rate = learning_rate


  LEARNING RATE:   
  current_lr= (1+decay⋅iterations/initial_lr)^(-1)
​
    def pre_update_params(self):
        # 🔁 Apply decay if specified
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

  def update_params(self, layer):
        # 🧮 Gradient Descent Step:
        # W = W - learning_rate × dW
        layer.weights += -self.current_learning_rate * layer.dweights
        layer.biases += -self.current_learning_rate * layer.dbiases

  def post_update_params(self):
        # ➕ Increment iteration count
        self.iterations += 1






(3) MOMENTUM
Gradient Descent is the process of iteratively adjusting model weights to minimize the loss. This implementation improves it with:
📉 Learning Rate Decay: Slows down learning over time for finer convergence.
🏃 Momentum: Helps speed up training and smooth out updates by remembering previous gradients.


🔧 Key Features
learning_rate: Controls the size of each weight update.
decay: Reduces the learning rate over time:
lr = initial_lr/(1+decay×iterations)

momentum: Combines current and previous gradients for faster convergence:
update = momentum×previous_update−lr×current_gradient






​
(4) 🚀 Adagrad Optimizer
Adagrad (Adaptive Gradient Algorithm) adapts the learning rate for each parameter individually based on how frequently it's updated.

📌 Key Concepts:
Adaptive learning rate: Parameters that receive frequent updates get smaller learning rates, while infrequent ones get larger rates.
No need to manually adjust the learning rate often.
Good for dealing with sparse data (like text or embeddings).

ε (epsilon) is a small constant to prevent division by zero.
cache grows over time, so learning slows down (which can be a downside).


🧠 Formula:
For weight w:
cache += gradient²
w -= (learning_rate / sqrt(cache + ε)) * gradient






(5) ⚡ RMSprop Optimizer
RMSprop (Root Mean Square Propagation) is an adaptive learning rate method designed to handle non-stationary objectives and improve training stability and speed, especially for RNNs or noisy gradients.

🔍 How It Works
Maintains an exponentially decaying average of squared gradients for each parameter.
Divides the learning rate by the square root of this moving average.
This stabilizes learning by dampening oscillations in steep or noisy directions.


📌 Formula
For a parameter θ:
cache = ρ * cache + (1 - ρ) * gradient²  
θ -= learning_rate * gradient / (sqrt(cache) + epsilon)





(6) 🔥 Adam Optimizer
Adam (Adaptive Moment Estimation) combines the benefits of Momentum and RMSprop to adapt the learning rate of each parameter and ensure stable, efficient training. It's one of the most widely used optimizers due to its efficiency and ability to work well in most scenarios.

📌 Key Concepts:
Momentum: Uses an exponentially decaying average of past gradients (m) to speed up convergence.
RMSprop: Uses an exponentially decaying average of past squared gradients (v) to adjust the learning rate.
Bias correction: Corrects initialization bias (especially in the initial iterations) for m and v.

🔢 How It Works:
First Moment Estimate (m): Exponentially weighted average of the gradients.
Second Moment Estimate (v): Exponentially weighted average of the squared gradients.
Bias Correction: At the beginning of training, the moment estimates are biased toward zero. We apply a correction to avoid this bias.
Parameter Update: Parameters are updated using the corrected m and v.


📌 Formula:
For a parameter θ:

m = β1 * m + (1 - β1) * gradient
v = β2 * v + (1 - β2) * gradient²

m_corrected = m / (1 - β1^t)
v_corrected = v / (1 - β2^t)
θ -= learning_rate * m_corrected / (sqrt(v_corrected) + epsilon)


β1, β2 are the decay rates for the first and second moments, respectively (default values: 0.9 and 0.999).
ε is a small constant (default: 1e-7) to prevent division by zero.
t is the timestep (iteration number).








